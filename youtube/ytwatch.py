#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "faster-whisper>=1.2",
#   "pillow>=11",
#   # Windows has no pacman, so yt-dlp, deno (yt-dlp's YouTube JS runtime)
#   # and ffmpeg come from PyPI there.
#   "yt-dlp[default,deno]; sys_platform == 'win32'",
#   "imageio-ffmpeg; sys_platform == 'win32'",
# ]
# ///
"""ytwatch: turn a video into something Claude can read and look at.

  ytwatch URL                    info + transcript + on-screen text + frames
  ytwatch FILE.mp4               same for a local video (no network)
  ytwatch URL --from 5:00 --to 9:30   only that stretch (transcript and frames)
  ytwatch URL --at 6:12,7:40     sharp 720p stills at those moments
  ytwatch URL --whisper          ignore YouTube captions, transcribe locally
  ytwatch URL --tempo 2          whisper hears the audio at 2x (GPU default 1.5)
  ytwatch URL --no-frames        transcript only (no video download)
  ytwatch URL --no-ocr           frames from 360p video, no screen.txt
  ytwatch FILE --screen          screen-recording mode (see below)
  ytwatch FILE --video           normal mode for a file named like a screen recording
  screenwatch FILE               = ytwatch --screen FILE
  ytwatch clean                  delete the cache

URL is a YouTube link or anything yt-dlp reads (TikTok, Instagram, X,
Vimeo, Reddit, ...). Everything lands in ~/.cache/ytwatch/<dir>/, where
<dir> is the bare video id for YouTube, <site>-<id> for other sites and
file-<hash> for local files (same file, same dir):
  info.txt         title, channel, length, chapters, description
  transcript.txt   "[mm:ss] text" paragraphs, ~30 s each
  screen.txt       text OCR'd off the frames (code, slides), when it changed
                   (every frame, in screen mode)
  timeline.txt     screen mode only: what appeared/disappeared, and what was said
  frames/mm-ss.jpg every kept frame at 720p, readable without --at
  sheets/NN.jpg    3x3 contact sheets of those frames, labeled with times
  at/mm-ss.jpg     the --at stills

Frames are taken at scene cuts plus one every span/--frames seconds, then
near-duplicates are dropped, so a talking head shrinks to a few frames and
every slide or code change gets one.

Screen mode is for screen recordings, where one changed line matters. It
keeps a frame at every visible change (a toggle, one edited text line; a
static screen gives one frame, pointer and cursor blinks are ignored), at
native size up to 1568 px on the long edge, named frames/mm-ss.s.jpg. It
OCRs every frame into screen.txt and writes timeline.txt: per frame, the
lines that appeared (+) and went away (-), with speech merged in as
"said:", so the sequence reads as text. Portrait recordings get 4x2
sheets. Local files whose name says screen recording (screen record,
Screen Recording, ScreenRecording_, Screen_recording_, screenrecording-)
get it automatically; --video turns it off. Links default to normal mode.
No audio (common for screen recordings) prints "no audio, no transcript".

Captions come from YouTube (manual first, then auto, as json3; for auto
captions the en-orig track, since YouTube rate limits the translated "en"
one). With none, when YouTube refuses them (429), or with --whisper, the
audio is transcribed by faster-whisper instead: large-v3-turbo in float16
on an NVIDIA GPU (~65x real time at 1.5x, half a minute for a half-hour
video), small.en on the CPU otherwise (~7x real time). Models are shared
with scribe's cache. The first GPU run on a Linux machine fetches CUDA 12
cuBLAS + cuDNN 9 (2.2 GB) into ~/.cache/ytwatch-cuda. On the GPU the audio
is sped up 1.5x (--tempo); on the CPU that saves nothing, so it stays 1x.
Silence is skipped by whisper's VAD and times are mapped back to the
video. The prompt is the title, chapters and description so names come
out right. The model dies with the process. The video used for frames is
deleted after they are cut.

Windows: install uv, then `uv run --script ytwatch.py URL`. yt-dlp, deno and
ffmpeg come from PyPI; tesseract is optional (winget install
UB-Mannheim.TesseractOCR). YTWATCH_BUNDLED=1 uses the PyPI copies on Linux
too, to test that path.
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

CACHE = Path("~/.cache/ytwatch").expanduser()
WHISPER_DIR = Path("~/.cache/scribe/whisper").expanduser()
# CUDA 12 cuBLAS + cuDNN 9 wheels for ctranslate2 (Arch ships CUDA 13).
# Outside CACHE so `ytwatch clean` doesn't throw away 1.3 GB.
CUDA_LIBS = Path("~/.cache/ytwatch-cuda").expanduser()
SCENE = 0.25  # ffmpeg scene score that counts as a cut
SAME = 20     # dHash bits (of 256) under which two frames are the same picture
ONE_SCENE = 0.9  # colour-histogram overlap above which frames share a scene
# Screen mode: samples per second, compare size, the per-pixel step that
# counts as changed (above codec noise), 8 px cells, and the longest edge
# kept frames are saved at (what Claude reads at full detail).
SCREEN_FPS = 10
SCREEN_CMP = 960
SCREEN_DIFF = 32
CELL = 8
SCREEN_CAP = 1568
SCREEN_NAME = re.compile(r"screen[ _-]?record", re.I)

BUNDLED = os.name == "nt" or os.environ.get("YTWATCH_BUNDLED") == "1"
if BUNDLED:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    YTDLP = [sys.executable, "-m", "yt_dlp", "--ffmpeg-location", FFMPEG]
    os.environ["PYTHONUTF8"] = "1"  # yt-dlp's -J output, not the code page
else:
    FFMPEG = "ffmpeg"
    YTDLP = ["yt-dlp"]


def find_tesseract():
    """PATH first; the Windows installer doesn't add itself to PATH."""
    if found := shutil.which("tesseract"):
        return found
    for root, sub in (("ProgramFiles", "Tesseract-OCR"),
                      ("LOCALAPPDATA", "Programs/Tesseract-OCR")):
        exe = Path(os.environ.get(root, "/nonexistent"), sub, "tesseract.exe")
        if exe.exists():
            return str(exe)
    return None


TESSERACT = find_tesseract()


def say(msg):
    print(f"ytwatch: {msg}", file=sys.stderr, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def clock(sec):
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def clock_ds(sec):
    """mm:ss.s, for screen recordings (short, where tenths matter)."""
    ds = round(sec * 10)
    m, s = ds // 600, ds % 600 / 10
    return f"{m:02d}:{s:04.1f}"


def parse_time(t):
    """'90', '1:30', '1:02:03' -> seconds."""
    sec = 0.0
    for part in t.strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


# ---------- metadata ----------

def player_url(url):
    """vimeo.com/<id> pages need a login in yt-dlp now (2026-11); the
    embed player at player.vimeo.com serves the same video without one."""
    if m := re.match(r"https?://(?:www\.)?vimeo\.com/(\d+)(?:/([0-9a-f]{6,}))?/?(?:[?#].*)?$", url):
        return f"https://player.vimeo.com/video/{m[1]}" + (f"?h={m[2]}" if m[2] else "")
    return url


def fetch_info(url):
    try:
        out = run([*YTDLP, "-J", "--no-playlist", "--no-warnings", url],
                  capture_output=True, encoding="utf-8", errors="replace").stdout
    except subprocess.CalledProcessError as e:
        # yt-dlp's own reason (login needed, no video in the post, ...)
        err = [l for l in (e.stderr or "").splitlines() if l.startswith("ERROR")]
        sys.exit(f"ytwatch: yt-dlp: {err[-1][7:] if err else f'exit {e.returncode}'}")
    return json.loads(out)


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".3gp", ".ts", ".flv", ".wmv", ".mpg", ".mpeg"}


def is_url(arg):
    return re.match(r"https?://", arg) is not None


def file_hash(path):
    """Same recording, same cache dir: sha256 of size + first and last MB."""
    import hashlib
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read())
    return h.hexdigest()[:10]


def file_info(path):
    """yt-dlp-shaped info for a local video, read from ffmpeg's banner (no
    network). ffmpeg rather than ffprobe: the Windows build only ships ffmpeg."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True,
                         encoding="utf-8", errors="replace").stderr
    if not (m := re.search(r"Duration: (\d+):(\d\d):(\d\d(?:\.\d+)?)", err)):
        sys.exit(f"ytwatch: {path} is not a video ffmpeg can read")
    h, mi, sec = m.groups()
    video = re.search(r"Stream #.*?: Video: .*?(\d{2,})x(\d{2,})", err)
    if not video:
        sys.exit(f"ytwatch: {path} has no video stream")
    created = re.search(r"creation_time\s*:\s*(\S+)", err)
    # Phone clips are stored sideways with a rotation tag; ffmpeg turns the
    # frames upright, so report the upright size.
    vw, vh = int(video[1]), int(video[2])
    if (rot := re.search(r"rotation of (-?[\d.]+)", err)) and abs(float(rot[1])) % 180 == 90:
        vw, vh = vh, vw
    return {
        "id": f"file-{file_hash(path)}",
        "title": path.name,
        "duration": int(h) * 3600 + int(mi) * 60 + float(sec),
        "webpage_url": str(path),
        "extractor_key": "file",
        "local": path,
        "size": path.stat().st_size,
        "width": vw, "height": vh,
        "has_audio": re.search(r"Stream #.*?: Audio:", err) is not None,
        "created": created[1] if created else None,
    }


def cache_dir(info):
    """YouTube: the bare id (the skill and the hub rely on it). Local files:
    file-<hash>. Other sites: <extractor>-<id>."""
    key = (info.get("extractor_key") or info.get("extractor") or "").lower()
    if key in ("youtube", "file"):
        name = info["id"]
    else:
        name = f"{key or 'video'}-{info['id']}".lower()
    return CACHE / re.sub(r"[^\w.-]", "_", name)


def write_info(info, path):
    if info.get("local"):
        lines = [
            f"title:    {info['title']}",
            f"file:     {info['webpage_url']}",
            f"length:   {clock(info['duration'])} ({info['duration']:.1f} s)",
            f"size:     {info['width']}x{info['height']}, {info['size'] / 1e6:.1f} MB",
            f"audio:    {'yes' if info['has_audio'] else 'none'}",
            f"created:  {info['created']}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines = [
        f"title:    {info.get('title')}",
        f"channel:  {info.get('channel') or info.get('uploader')}",
        f"uploaded: {info.get('upload_date')}",
        f"length:   {clock(info.get('duration') or 0)}",
        f"views:    {info.get('view_count')}",
        f"url:      {info.get('webpage_url')}",
    ]
    if info.get("chapters"):
        lines.append("\nchapters:")
        lines += [f"  [{clock(c['start_time'])}] {c['title']}" for c in info["chapters"]]
    lines.append("\ndescription:\n" + (info.get("description") or "").strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- captions ----------

def pick_caption_lang(info):
    """Prefer human captions, then YouTube's auto ones. English first.
    Auto "en" is YouTube's translation of the track (and gets 429'd);
    "en-orig" is the speech-recognized original, so it goes first."""
    for key in ("subtitles", "automatic_captions"):
        tracks = info.get(key) or {}
        wants = ("en", "en-US", "en-GB") if key == "subtitles" else ("en-orig", "en", "en-US", "en-GB")
        for want in wants:
            if want in tracks:
                return key, want
        for lang in tracks:
            if lang.startswith("en") and "-" not in lang[3:]:
                return key, lang
    return None, None


def fetch_captions(url, info, work):
    key, lang = pick_caption_lang(info)
    if not key:
        say("no captions, falling back to whisper")
        return None, None
    flag = "--write-subs" if key == "subtitles" else "--write-auto-subs"
    try:
        run([*YTDLP, "--skip-download", "--no-playlist", "--no-warnings", "-q",
             flag, "--sub-langs", lang, "--sub-format", "json3/vtt",
             "-o", str(work / "subs"), url],
            capture_output=True, encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        why = "429" if "429" in err else (err.splitlines() or [f"exit {e.returncode}"])[-1]
        say(f"captions blocked ({why}), falling back to whisper")
        return None, None
    kind = "manual captions" if key == "subtitles" else "auto captions"
    if sub := next(work.glob("subs*.json3"), None):
        return parse_json3(sub.read_text(encoding="utf-8", errors="replace")), f"{kind} ({lang})"
    if sub := next(work.glob("subs*.vtt"), None):
        return parse_vtt(sub.read_text(encoding="utf-8", errors="replace")), f"{kind} ({lang}, vtt)"
    say("no captions, falling back to whisper")
    return None, None


def parse_json3(text):
    """json3 -> [(start_sec, line)]. One event per caption, no rolling repeats."""
    out = []
    for ev in json.loads(text).get("events", []):
        line = "".join(s.get("utf8", "") for s in ev.get("segs") or [])
        line = " ".join(line.split())
        if line:
            out.append((ev.get("tStartMs", 0) / 1000, line))
    return out


TS = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d+)\s+-->")


def parse_vtt(text):
    """VTT -> [(start_sec, line)]. Auto captions repeat each line across a
    few rolling cues; only the first sighting of a line is kept."""
    out, recent = [], []
    for block in re.split(r"\n\s*\n", text):
        m = TS.search(block)
        if not m:
            continue
        h, mi, s, ms = m.groups()
        start = int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000
        text_lines = block.split("\n")
        cue = next(i for i, l in enumerate(text_lines) if "-->" in l)
        for raw in text_lines[cue + 1:]:
            line = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
            if not line or line in recent:
                continue
            out.append((start, line))
            recent = (recent + [line])[-4:]
    return out


# ---------- whisper ----------

def whisper_prompt(info):
    """Description, chapters and title as whisper's prior text, so names and
    jargon are spelled right. Whisper keeps the end of a long prompt, so the
    title goes last."""
    desc = re.sub(r"https?://\S+", "", info.get("description") or "")
    parts = [" ".join(desc.split())[:400]]
    parts += [c["title"] for c in info.get("chapters") or []]
    parts.append(info.get("title") or "")
    return ". ".join(p for p in parts if p)[-700:]


def cuda_ready():
    """True when whisper can run on an NVIDIA GPU. ctranslate2 is built for
    CUDA 12 cuBLAS and cuDNN 9; those come as pip wheels, fetched once into
    CUDA_LIBS and preloaded so its dlopen finds them. Linux only."""
    if sys.platform != "linux" or not shutil.which("nvidia-smi"):
        return False
    import ctranslate2
    if ctranslate2.get_cuda_device_count() == 0:
        return False

    def find():
        return sorted([*CUDA_LIBS.glob("nvidia/cublas/lib/libcublas*.so.12"),
                       *CUDA_LIBS.glob("nvidia/cudnn/lib/libcudnn*.so.9")])
    if not find():
        say(f"fetching CUDA 12 cuBLAS + cuDNN 9 for the GPU (2.2 GB, once) into {CUDA_LIBS}")
        try:
            run(["uv", "pip", "install", "--quiet", "--target", str(CUDA_LIBS),
                 "--python", sys.executable, "nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10"])
        except (OSError, subprocess.CalledProcessError):
            return False
    # Load order matters (cublas needs cublasLt, cudnn parts need each
    # other): retry the failures until a pass loads nothing new.
    import ctypes
    pending = find()
    while pending:
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(pending):
            break
        pending = failed
    loaded = {p.name for p in find()} - {p.name for p in pending}
    return {"libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"} <= loaded


def transcribe(url, info, work, model_name, start, end, tempo):
    """Returns ([(start_sec, text)], label). The audio is played to whisper
    at `tempo` x; whisper's VAD skips the silences and gives times in the
    sped-up audio, so original time = start + t * tempo. tempo None = 1.5
    on the GPU (about 20% faster, same words) and 1 on the CPU, where
    speeding up saved no time and 2x cost accuracy."""
    from faster_whisper import WhisperModel
    devices = [("cuda", "float16"), ("cpu", "int8")] if cuda_ready() else [("cpu", "int8")]
    if tempo is None:
        tempo = 1.5 if devices[0][0] == "cuda" else 1
    if isinstance(url, Path):
        audio = url
    else:
        say("downloading audio")
        # The whole track, cut by ffmpeg below: --download-sections is slower
        # for audio and starts up to ~10 s early, which shifts every timestamp.
        run([*YTDLP, "-f", "bestaudio/best", "--no-playlist", "--no-warnings", "-q",
             "-o", str(work / "audio.%(ext)s"), url])
        audio = next(work.glob("audio.*"))
        # Some posts (old Instagram videos, muted clips) have no sound at all.
        banner = subprocess.run([FFMPEG, "-hide_banner", "-i", str(audio)], capture_output=True,
                                encoding="utf-8", errors="replace").stderr
        if not re.search(r"Stream #.*?: Audio:", banner):
            say("no audio, no transcript")
            return [], "no audio"
    # Decode with ffmpeg, not PyAV: faster-whisper's av call breaks on new av.
    import numpy as np
    cut = (["-ss", str(start)] if start is not None else []) + (["-to", str(end)] if end is not None else [])
    speed = ["-af", f"atempo={tempo}"] if tempo != 1 else []
    pcm = run([FFMPEG, "-v", "error", *cut, "-i", str(audio), *speed, "-f", "f32le",
               "-ac", "1", "-ar", "16000", "-"], capture_output=True).stdout
    pcm = np.frombuffer(pcm, np.float32)

    for device, compute in devices:
        name = model_name or ("large-v3-turbo" if device == "cuda" else "small.en")
        say(f"transcribing with whisper {name} ({device.upper()}, {tempo}x)")
        try:
            model = WhisperModel(name, device=device, compute_type=compute,
                                 cpu_threads=os.cpu_count() or 4,
                                 download_root=str(WHISPER_DIR))
            segs, _ = model.transcribe(pcm, vad_filter=True, word_timestamps=True,
                                       initial_prompt=whisper_prompt(info))
            words = [w for seg in segs for w in seg.words]
            break
        except RuntimeError as e:  # CUDA out of memory, missing kernel, ...
            if device == "cpu":
                raise
            say(f"GPU failed ({str(e).splitlines()[0]}), using the CPU")
    # turbo makes one segment per 30 s window, so lines are rebuilt from
    # word times: a new line after a sentence end or a 1 s pause.
    offset = start or 0
    lines, cur, last = [], [], 0
    for w in words:
        if cur and (w.start - last > 1 or cur[-1].endswith((".", "?", "!"))):
            lines.append((t0, " ".join(cur)))
            cur = []
        if not cur:
            t0 = offset + w.start * tempo
        if w.word.strip():
            cur.append(w.word.strip())
        last = w.end
    if cur:
        lines.append((t0, " ".join(cur)))
    return lines, f"whisper {name}, {device.upper()}, {tempo}x"


def write_transcript(lines, path, start, end, para=30):
    """Merge lines into ~30 s paragraphs, each led by its start time."""
    paras, cur, cur_t = [], [], None
    for t, line in lines:
        if start is not None and t < start - 1:
            continue
        if end is not None and t > end:
            break
        if cur_t is None:
            cur_t = t
        elif t - cur_t >= para:
            paras.append(f"[{clock(cur_t)}] " + " ".join(cur))
            cur, cur_t = [], t
        cur.append(line)
    if cur:
        paras.append(f"[{clock(cur_t)}] " + " ".join(cur))
    path.write_text("\n\n".join(paras) + "\n", encoding="utf-8")
    return sum(len(p.split()) - 1 for p in paras)


# ---------- frames ----------

def grab(video, t, dest, width):
    run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(dest)])


def download_video(url, work, height, section=None, screen=False):
    # Screen mode wants the shorter edge near 1568 whatever the orientation,
    # which a height filter can't say for portrait video; -S res can.
    fmt = (["-f", "bv*/b", "-S", f"res:{SCREEN_CAP}"] if screen
           else ["-f", f"bv*[height<={height}]/b[height<={height}]/worst"])
    cmd = [*YTDLP, "--no-playlist", "--no-warnings", "-q", *fmt,
           "-o", str(work / "video.%(ext)s"), url]
    if section:
        cmd[len(YTDLP):len(YTDLP)] = ["--download-sections", section, "--force-keyframes-at-cuts"]
    run(cmd)
    return next(work.glob("video.*"))


def cut_frames(video, work, every, shift, cut=(), cap=None):
    """One decode pass: a frame at each scene cut, plus one whenever `every`
    seconds pass without a cut. `cut` = input -ss/-to, `cap` = longest edge
    for local files (downloads come sized already). -> [(time, path)]"""
    raw = work / "raw"
    raw.mkdir()
    pick = f"isnan(prev_selected_t)+gt(scene\\,{SCENE})+gte(t-prev_selected_t\\,{every:.2f})"
    size = (f",scale='min({cap},iw)':'min({cap},ih)':force_original_aspect_ratio=decrease"
            if cap else "")
    err = run([FFMPEG, "-v", "info", "-nostats", *cut, "-i", str(video),
               "-vf", f"select='{pick}'{size},showinfo", "-fps_mode", "vfr", "-q:v", "2",
               str(raw / "%05d.jpg")], capture_output=True, encoding="utf-8",
              errors="replace").stderr
    times = [float(t) + shift for t in re.findall(r"pts_time:(-?[\d.]+)", err)]
    return list(zip(times, sorted(raw.glob("*.jpg"))))


def dhash(path, n=16):
    """256-bit difference hash: is each pixel brighter than its right neighbour."""
    from PIL import Image
    with Image.open(path) as im:
        px = im.convert("L").resize((n + 1, n)).tobytes()
    bits = 0
    for y in range(n):
        row = px[y * (n + 1):(y + 1) * (n + 1)]
        for x in range(n):
            bits = bits << 1 | (row[x] > row[x + 1])
    return bits


def drop_repeats(frames, limit):
    """Keep frames that differ from every kept one; thin evenly to `limit`."""
    kept, hashes = [], []
    for t, f in frames:
        h = dhash(f)
        if all((h ^ k).bit_count() > SAME for k in hashes):
            kept.append((t, f))
            hashes.append(h)
    if len(kept) > limit:
        step = (len(kept) - 1) / max(limit - 1, 1)
        kept = [kept[round(i * step)] for i in range(limit)]
    return kept


def blobs(a, b, pointer, top=0):
    """Changed regions between two compare-size gray images, as bounding
    boxes in cells, minus the ones that aren't a visible change: thin
    strips (a blinking text cursor, a scrollbar edge), anything inside the
    `top` cells (a phone's status bar: clock, battery, the blinking
    recording dot) and, when a mouse pointer may be on screen,
    pointer-sized specks."""
    from PIL import Image, ImageChops
    mask = ImageChops.difference(a, b).point(lambda v: 255 if v > SCREEN_DIFF else 0)
    if not mask.getbbox():
        return []
    gw, gh = a.width // CELL, a.height // CELL
    # A cell is hot when at least 4 of its 64 pixels changed.
    px = mask.resize((gw, gh), Image.BOX).tobytes()
    hot = {(i % gw, i // gw) for i, v in enumerate(px) if v * CELL * CELL >= 4 * 255}
    found = []
    while hot:
        todo = [hot.pop()]
        xs, ys, n = [], [], 0
        while todo:
            x, y = todo.pop()
            xs.append(x)
            ys.append(y)
            n += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (x + dx, y + dy) in hot:
                        hot.remove((x + dx, y + dy))
                        todo.append((x + dx, y + dy))
        bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if min(bw, bh) < 2 or max(ys) < top:
            continue
        if pointer and max(bw, bh) < 6 and n < 16:
            continue
        found.append((min(xs), min(ys), bw, bh, n))
    return found


def screen_frames(video, work, size, cut=(), shift=0, limit=120):
    """Screen mode: sample SCREEN_FPS frames a second at SCREEN_CMP px, keep
    one each time the screen changes and settles (a toggle, one new text
    line, a menu), or one a second while it keeps moving (a scroll, a video
    playing). Kept frames are then cut at full size, longest edge capped at
    SCREEN_CAP. -> ([(time, path)], samples looked at)"""
    from PIL import Image
    w, h = size
    k = SCREEN_CMP / max(w, h)
    cw, ch = max(2, round(w * k / 2) * 2), max(2, round(h * k / 2) * 2)
    pointer = w > h  # landscape = a desktop, where a mouse pointer wanders
    # Portrait = a phone: its status bar is the top ~5.5% of the screen.
    top = 0 if pointer else round(ch * 0.055 / CELL)
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", *cut, "-i", str(video), "-vf",
         f"fps={SCREEN_FPS},scale={cw}:{ch}:flags=area,format=gray",
         "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    frame = cw * ch

    def samples():
        while len(buf := proc.stdout.read(frame)) == frame:
            yield Image.frombytes("L", (cw, ch), buf)

    keep, base, base_n, moving = [], None, 0, None
    cur, n = None, -1
    for nxt in samples():
        if cur is not None:
            n += 1
            if base is None:
                keep.append(n)
                base, base_n = cur, n
            elif blobs(base, cur, pointer, top):
                settled = not blobs(cur, nxt, pointer, top)
                if moving is None:
                    moving = n
                if settled or (n - moving >= SCREEN_FPS and n - base_n >= SCREEN_FPS):
                    keep.append(n)
                    base, base_n = cur, n
                    moving = None if settled else n
            else:
                moving = None
        cur = nxt
    if cur is not None:  # the last sample has nothing after it: settled
        n += 1
        if base is None or blobs(base, cur, pointer, top):
            keep.append(n)
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, [FFMPEG])
    if len(keep) > limit:
        step = (len(keep) - 1) / max(limit - 1, 1)
        keep = [keep[round(i * step)] for i in range(limit)]

    raw = work / "raw"
    raw.mkdir()
    pick = "+".join(f"eq(n\\,{i})" for i in keep)
    size_vf = (f"scale='min({SCREEN_CAP},iw)':'min({SCREEN_CAP},ih)'"
               ":force_original_aspect_ratio=decrease:flags=lanczos")
    err = run([FFMPEG, "-v", "info", "-nostats", *cut, "-i", str(video),
               "-vf", f"fps={SCREEN_FPS},select='{pick}',{size_vf},showinfo",
               "-fps_mode", "vfr", "-q:v", "2", str(raw / "%05d.jpg")],
              capture_output=True, encoding="utf-8", errors="replace").stderr
    times = [float(t) + shift for t in re.findall(r"pts_time:(-?[\d.]+)", err)]
    return list(zip(times, sorted(raw.glob("*.jpg")))), n + 1


def colours(path):
    """8 bins per RGB channel, as fractions of the image."""
    from PIL import Image
    with Image.open(path) as im:
        h = im.convert("RGB").resize((160, 90)).histogram()
    return [sum(h[i:i + 32]) / (160 * 90 * 3) for i in range(0, 768, 32)]


def one_scene(frames):
    """True when every frame looks like the same place: a talking head,
    a podium, a podcast set. Compared by colour, so pose and zoom don't count."""
    hs = [colours(f) for _, f in frames]
    overlaps = sorted(sum(map(min, a, b)) for a, b in zip(hs, hs[1:]))
    return bool(overlaps) and overlaps[len(overlaps) // 2] >= ONE_SCENE


def ocr(path, scratch, numbers=False, boxes=False):
    """Tesseract on a cleaned copy (gray, dark themes flipped, 2x). Keeps
    confident lines and rebuilds indentation from where each line starts.
    numbers=True (screen mode) also keeps lines that are only a number,
    price or time ("$39.48", "12:30"), which apps are full of.
    boxes=True returns (text, [(left, top, right, bottom) per line]) in the
    frame's own pixels."""
    from PIL import Image, ImageOps, ImageStat
    with Image.open(path) as im:
        im = im.convert("L")
    if ImageStat.Stat(im).mean[0] < 128:
        im = ImageOps.invert(im)
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    # A file, not stdin: Windows tesseract reads stdin in text mode.
    clean = scratch / f"{path.stem}.png"
    im.save(clean)
    tsv = run([TESSERACT, str(clean), "stdout", "tsv"],
              capture_output=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"}).stdout
    lines = {}
    for row in tsv.decode(errors="replace").splitlines()[1:]:
        c = row.split("\t")
        if len(c) == 12 and c[0] == "5" and c[11].strip():
            lines.setdefault((int(c[2]), int(c[3]), int(c[4])), []).append(
                (int(c[6]), int(c[8]), float(c[10]), c[11].strip(), int(c[7]), int(c[9])))
    good = {}
    for key, words in lines.items():
        conf = sum(w[2] for w in words) / len(words)
        texts = [w[3] for w in words]
        grid = len(texts) > 8 and max(map(texts.count, texts)) > len(texts) / 2
        line = " ".join(texts)
        wordy = re.search(r"\w{3}", line) or (numbers and re.search(r"\d[.,:/]\d", line))
        if conf >= 70 and not grid and wordy:
            good[key] = words
    if not good:
        return ("", []) if boxes else ""
    widths = sorted(w[1] / len(w[3]) for ws in good.values() for w in ws)
    char = widths[len(widths) // 2]
    margin = {}
    for (block, _, _), ws in good.items():
        margin[block] = min(margin.get(block, ws[0][0]), ws[0][0])
    out, where = [], []
    for key in sorted(good):
        ws = good[key]
        indent = round((ws[0][0] - margin[key[0]]) / char)
        out.append(" " * min(indent, 40) + " ".join(w[3] for w in ws))
        where.append((min(w[0] for w in ws) // 2, min(w[4] for w in ws) // 2,
                      max(w[0] + w[1] for w in ws) // 2 + 1, max(w[4] + w[5] for w in ws) // 2 + 1))
    text = "\n".join(out)
    return (text, where) if boxes else text


def write_screen(frames, path, scratch):
    """OCR the kept frames; write a frame's text only when it adds to what
    was last written. Returns the number of lines written."""
    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        texts = list(pool.map(partial(ocr, scratch=scratch), [f for _, f in frames]))
    blocks, shown, count = [], set(), 0
    for (t, _), text in zip(frames, texts):
        cur = {l.strip() for l in text.splitlines()}
        if not cur or len(cur & shown) >= 0.8 * len(cur):
            continue
        blocks.append(f"[{clock(t)}]\n{text}")
        shown, count = cur, count + len(cur)
    if blocks:
        path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    else:
        path.unlink(missing_ok=True)
    return count


FIRST_LINES = 40   # text lines the timeline shows for the first frame
STEP_LINES = 25    # ... and per later frame, each way (+ and -)
SAME_LINE = 0.85   # difflib ratio above which two OCR reads are one line


def screen_texts(frames, scratch):
    """OCR every kept frame (in parallel). Returns (text, line boxes) per frame."""
    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        return list(pool.map(partial(ocr, scratch=scratch, numbers=True, boxes=True),
                             [f for _, f in frames]))


def changed(a, b, box):
    """How much of the text under `box` differs between frames a and b:
    changed pixels over ink pixels (anything off the background colour in
    either frame), so a sparse line ("Total:   $16") appearing scores
    high and a pointer crossing a long line scores low."""
    from PIL import Image, ImageChops

    def count(mask):
        return mask.histogram()[255]

    def over(im, ref):
        return ImageChops.difference(im, ref).point(lambda v: 255 if v > SCREEN_DIFF else 0)

    a, b = a.crop(box), b.crop(box)
    hist = a.histogram()
    bg = Image.new("L", a.size, hist.index(max(hist)))
    ink = count(ImageChops.lighter(over(a, bg), over(b, bg)))
    return count(over(a, b)) / max(ink, 1)


def edited(line, others):
    """True when `others` holds a changed version of `line` ("Qty: 1" vs
    "Qty: 2"): a real edit can be a few pixels, so it gets no slack."""
    from difflib import SequenceMatcher
    return any(SequenceMatcher(None, line.lower(), o.lower(), autojunk=False).ratio() >= 0.5
               for o in others)


def line_diff(before, after):
    """Lines of `after` not in `before` and lines of `before` gone from
    `after`. OCR misreads a letter here and there between two frames of
    the same screen, so near-identical lines count as the same line."""
    from difflib import SequenceMatcher

    def norm(l):
        return " ".join(l.split()).lower()

    def same(a, b):
        return a == b or SequenceMatcher(None, a, b, autojunk=False).ratio() >= SAME_LINE

    old = [norm(l) for l in before]
    new = [norm(l) for l in after]
    old_set, new_set = set(old), set(new)
    added = [l for l, n in zip(after, new) if n not in old_set and not any(same(n, o) for o in old)]
    gone = [l for l, o in zip(before, old) if o not in new_set and not any(same(o, n) for n in new)]
    return added, gone


def write_timeline(frames, texts, speech, path):
    """What happened, in order: per kept frame the text lines that appeared
    (+) and went away (-) since the frame before, with what was said merged
    in at its times. The first frame lists its text (capped).

    Tesseract reads the same screen a little differently each time (drops a
    line, glues two together), so a line only counts as gone or new when
    the pixels under it changed too."""
    from PIL import Image
    events = [(t, 1, ("frame", k)) for k, (t, _) in enumerate(frames)]
    events += [(t, 0, ("said", line)) for t, line in speech]
    out, prev, prev_im = [], [], None
    for t, _, (kind, what) in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == "said":
            out.append(f"[{clock_ds(t)}] said: {what}")
            continue
        name = frames[what][1].name
        if texts is None:
            out.append(f"[{clock_ds(t)}] frame {name}")
            continue
        text, boxes = texts[what]
        cur = [(l.strip(), b) for l, b in zip(text.splitlines(), boxes)]
        with Image.open(frames[what][1]) as im:
            im = im.convert("L")
        if what == 0:
            shown = cur[:FIRST_LINES]
            head = f"{len(cur)} lines of text" if cur else "no readable text"
            out.append(f"[{clock_ds(t)}] frame {name}: {head}")
            out += [f"  {l}" for l, _ in shown]
            if len(cur) > len(shown):
                out.append(f"  ... {len(cur) - len(shown)} more in screen.txt")
            prev, prev_im = cur, im
            continue
        added, gone = line_diff([l for l, _ in prev], [l for l, _ in cur])
        box_of = {l: b for l, b in prev} | {l: b for l, b in cur}
        # Unchanged pixels = the same text. A line with no edited twin also
        # gets slack for a pointer passing over it (~10% of a long line's ink).
        def kept(l, others):
            return changed(prev_im, im, box_of[l]) <= (0.02 if edited(l, others) else 0.25)
        missed = [l for l in gone if kept(l, added)]
        added = [l for l in added if not kept(l, gone)]
        gone = [l for l in gone if l not in missed]
        if not added and not gone:
            out.append(f"[{clock_ds(t)}] frame {name}: same text, the picture changed")
        else:
            out.append(f"[{clock_ds(t)}] frame {name}: +{len(added)} -{len(gone)}")
            for sign, ls in (("-", gone), ("+", added)):
                out += [f"  {sign} {l}" for l in ls[:STEP_LINES]]
                if len(ls) > STEP_LINES:
                    out.append(f"  {sign} ... {len(ls) - STEP_LINES} more")
        # A line tesseract missed this time is still on screen.
        prev = cur + [(l, box_of[l]) for l in missed]
        prev_im = im
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_screen_all(frames, texts, path):
    """Screen mode: every frame's text under its time, even when it repeats,
    so screen.txt answers "what did the screen say at 00:04.2". Returns the
    number of lines written."""
    blocks, count = [], 0
    for (t, p), (text, _) in zip(frames, texts):
        body = text if text.strip() else "(no readable text)"
        blocks.append(f"[{clock_ds(t)}] {p.name}\n{body}")
        count += sum(1 for l in text.splitlines() if l.strip())
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return count


def make_frames(url, work, out, duration, start, end, max_frames, read_text, screen=False, size=None, speech=()):
    a = start or 0
    b = end if end is not None else duration
    every = max((b - a) / max_frames, 2)
    height = 720 if read_text else 360
    if screen:
        if isinstance(url, Path):
            video, shift = url, a if start is not None else 0
            cut = (["-ss", str(a)] if start is not None else []) + (["-t", str(b - a)] if end is not None else [])
        else:
            say("downloading the video at full size for screen mode")
            section = f"*{a}-{b}" if (start is not None or end is not None) else None
            video = download_video(url, work, height, section, screen=True)
            shift, cut = (a if section else 0), []
            info = file_info(video)
            size = (info["width"], info["height"])
        say(f"screen mode: a frame at every change ({SCREEN_FPS} looks a second)")
        kept, found = screen_frames(video, work, size, cut, shift, max_frames)
        if video is not url:
            video.unlink()
    elif isinstance(url, Path):
        say(f"reading frames at cuts and every {every:.0f} s")
        # -ss before -i restarts the clock at 0, so the shift adds it back.
        cut = (["-ss", str(a)] if start is not None else []) + (["-t", str(b - a)] if end is not None else [])
        found = cut_frames(url, work, every, a if start is not None else 0, cut, height * 16 // 9)
    else:
        say(f"downloading {height}p video; frames at cuts and every {every:.0f} s")
        section = f"*{a}-{b}" if (start is not None or end is not None) else None
        video = download_video(url, work, height, section)
        shift = a if section else 0  # a cut download starts at 0
        found = cut_frames(video, work, every, shift)
        video.unlink()
    if not screen:
        kept = drop_repeats(found, max_frames)
        found = len(found)
    stamp = clock_ds if screen else clock

    frames = out / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir()
    named = []
    for t, f in kept:
        p = frames / f"{stamp(t).replace(':', '-')}.jpg"
        shutil.copy(f, p)
        named.append((t, p))

    if screen:
        texts = screen_texts(named, work) if read_text else None
        lines = write_screen_all(named, texts, out / "screen.txt") if read_text else 0
        write_timeline(named, texts, speech, out / "timeline.txt")
    elif read_text:
        lines = write_screen(named, out / "screen.txt", work)
    else:
        lines = 0
    for stale, keep in (("screen.txt", read_text), ("timeline.txt", screen)):
        if not keep:
            (out / stale).unlink(missing_ok=True)  # a stale one would mislead
    # No text anywhere and one unchanging scene: three frames show it all.
    # The rest stay in frames/ for a closer look.
    shown = named
    if not screen and read_text and not lines and len(named) > 3 and one_scene(named):
        shown = [named[round(i * (len(named) - 1) / 2)] for i in range(3)]

    sheets = out / "sheets"
    shutil.rmtree(sheets, ignore_errors=True)
    sheets.mkdir()
    # Portrait screen recordings: 4 tall frames x 2 rows, big enough to
    # tell screens apart. Everything else: 3x3.
    w, h = size or (16, 9)
    grid = (4, 2, 300, 652) if screen and h > w else (3, 3, 426, 240)
    cols, rows, cw, ch = grid
    per = cols * rows
    paths = []
    for n, i in enumerate(range(0, len(shown), per), 1):
        s = sheets / f"{n:02d}.jpg"
        contact_sheet(shown[i:i + per], s, cw, ch, cols=cols, stamp=stamp)
        paths.append(s)
    return paths, found, len(named), len(shown), every, lines, f"{cols}x{rows}"


def label_font(size):
    from PIL import ImageFont
    for name in ("LiberationSans-Regular.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def contact_sheet(frames, dest, w=426, h=240, pad=4, label=28, cols=3, stamp=clock):
    """`cols` across, each frame fit into w x h with its time centred under it."""
    from PIL import Image, ImageDraw
    rows = -(-len(frames) // cols)
    cell_w, cell_h = w + 2 * pad, h + label + 2 * pad
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#111")
    draw = ImageDraw.Draw(sheet)
    font = label_font(20)
    for k, (t, p) in enumerate(frames):
        x, y = k % cols * cell_w + pad, k // cols * cell_h + pad
        with Image.open(p) as im:
            im.thumbnail((w, h))
            sheet.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))
        draw.text((x + w // 2, y + h + label // 2), stamp(t), fill="white",
                  font=font, anchor="mm")
    sheet.save(dest, quality=82)


def make_stills(url, work, out, times):
    stills = out / "at"
    stills.mkdir(exist_ok=True)
    paths = []
    for t in times:
        p = stills / f"{clock(t).replace(':', '-')}.jpg"
        if isinstance(url, Path):
            grab(url, t, p, 1280)
            paths.append(p)
            continue
        sub = Path(tempfile.mkdtemp(dir=work))
        say(f"grabbing {clock(t)} at 720p")
        video = download_video(url, sub, 720, f"*{max(t - 2, 0)}-{t + 2}")
        grab(video, min(2, t), p, 1280)
        paths.append(p)
    return paths


# ---------- main ----------

def main():
    for stream in (sys.stdout, sys.stderr):  # titles vs a Windows code page
        stream.reconfigure(errors="replace")
    if sys.argv[1:2] == ["clean"]:
        size = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file()) if CACHE.exists() else 0
        shutil.rmtree(CACHE, ignore_errors=True)
        print(f"removed {CACHE} ({size / 1e6:.1f} MB)")
        return

    # `screenwatch` is a symlink to this script: screen mode by default.
    as_screen = Path(sys.argv[0]).stem == "screenwatch"
    ap = argparse.ArgumentParser(prog="screenwatch" if as_screen else "ytwatch",
                                 description=__doc__.split("\n\n")[0],
                                 epilog="screenwatch FILE = ytwatch --screen FILE. Full docs: the top of this script.")
    ap.add_argument("url", help="a YouTube or other yt-dlp link, or a local video file")
    ap.add_argument("--from", dest="start", help="start time, e.g. 5:00")
    ap.add_argument("--to", dest="end", help="end time, e.g. 9:30")
    ap.add_argument("--at", help="comma-separated times for sharp stills, e.g. 6:12,7:40")
    ap.add_argument("--frames", type=int, help="most frames to keep (default 36 = 4 sheets; 120 in screen mode)")
    ap.add_argument("--no-frames", action="store_true", help="skip the video download")
    ap.add_argument("--no-ocr", action="store_true", help="360p frames, no screen.txt")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--screen", action="store_true",
                      help="screen recording: a full-size frame at every visible change (auto for files named like screen recordings)")
    mode.add_argument("--video", action="store_true", help="normal mode even when the file name says screen recording")
    ap.add_argument("--whisper", action="store_true", help="transcribe locally even if captions exist")
    ap.add_argument("--model", help="whisper model (default large-v3-turbo on an NVIDIA GPU, small.en on the CPU; 'small' for other languages)")
    ap.add_argument("--tempo", type=float, help="speed the audio up this much for whisper (default 1.5 on an NVIDIA GPU, 1 = off on the CPU)")
    if as_screen:
        ap.set_defaults(screen=True)
    args = ap.parse_args()

    start = parse_time(args.start) if args.start else None
    end = parse_time(args.end) if args.end else None
    read_text = not args.no_ocr and TESSERACT is not None
    if not args.no_ocr and not read_text:
        say("tesseract not installed, skipping screen.txt")

    if is_url(args.url):
        src = player_url(args.url)
        say("reading video info")
        info = fetch_info(src)
    else:
        src = Path(args.url).expanduser().resolve()
        if not src.is_file():
            sys.exit(f"ytwatch: no such file or link: {args.url}")
        info = file_info(src)
    screen = not args.video and (args.screen or (info.get("local") is not None
                                                 and SCREEN_NAME.search(info["title"]) is not None))
    max_frames = args.frames or (120 if screen else 36)
    out = cache_dir(info)
    out.mkdir(parents=True, exist_ok=True)
    write_info(info, out / "info.txt")
    duration = info.get("duration") or 0
    if info.get("is_live"):
        sys.exit("ytwatch: live streams aren't supported")

    with tempfile.TemporaryDirectory(prefix="ytwatch-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)

        if args.at:
            times = [parse_time(t) for t in args.at.split(",")]
            for p in make_stills(src, work, out, times):
                print(p)
            return

        if info.get("local") and not info["has_audio"]:
            say("no audio, no transcript")
            lines, source = [], "no audio"
        elif info.get("local") or args.whisper:
            lines, source = None, None
        else:
            lines, source = fetch_captions(src, info, work)
        if lines is None or (not lines and source != "no audio"):
            lines, source = transcribe(src, info, work, args.model, start, end, args.tempo)
        words = write_transcript(lines, out / "transcript.txt", start, end)

        sheets = []
        if not args.no_frames:
            size = (info["width"], info["height"]) if info.get("local") else None
            sheets, found, kept, shown, every, screen_lines, grid = make_frames(
                src, work, out, duration, start, end, max_frames, read_text, screen, size,
                [(t, l) for t, l in lines if (start is None or t >= start - 1) and (end is None or t <= end)])

    rng = f"{clock(start or 0)}-{clock(end if end is not None else duration)}"
    who = "local file" if info.get("local") else info.get("channel") or info.get("uploader") or info.get("extractor_key")
    print(f"{info.get('title')}  ({who}, {clock(duration)})")
    print(f"range:      {rng}")
    print(f"info:       {out / 'info.txt'}")
    if source == "no audio":
        print("transcript: no audio, no transcript")
    elif not words:
        print(f"transcript: no speech heard ({source})")
    else:
        print(f"transcript: {out / 'transcript.txt'}  ({words} words, {source})")
    if sheets:
        if read_text:
            print(f"screen:     {out / 'screen.txt'}  ({screen_lines} lines)" if screen_lines
                  else "screen:     no readable text on screen")
        if screen:
            print(f"timeline:   {out / 'timeline.txt'}  (read this first: what changed when, and what was said)")
        how = (f"screen mode, one per visible change, {SCREEN_FPS} looks a second" if screen
               else f"cuts + one per {every:.0f} s")
        print(f"frames:     {out / 'frames'}/  ({kept} kept of {found}: {how})")
        note = "" if shown == kept else f", {shown} of the frames: one unchanging scene"
        print(f"sheets:     {len(sheets)} x {grid}{note}")
        for p in sheets:
            print(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        name = "yt-dlp" if "yt_dlp" in e.cmd else Path(e.cmd[0]).name
        sys.exit(f"ytwatch: {name} failed (exit {e.returncode})")
