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
"""ytwatch: turn a YouTube video into something Claude can read and look at.

  ytwatch URL                    info + transcript + on-screen text + frames
  ytwatch URL --from 5:00 --to 9:30   only that stretch (transcript and frames)
  ytwatch URL --at 6:12,7:40     sharp 720p stills at those moments
  ytwatch URL --whisper          ignore YouTube captions, transcribe locally
  ytwatch URL --tempo 2          whisper hears the audio at 2x (GPU default 1.5)
  ytwatch URL --no-frames        transcript only (no video download)
  ytwatch URL --no-ocr           frames from 360p video, no screen.txt
  ytwatch clean                  delete the cache

Everything lands in ~/.cache/ytwatch/<video id>/:
  info.txt         title, channel, length, chapters, description
  transcript.txt   "[mm:ss] text" paragraphs, ~30 s each
  screen.txt       text OCR'd off the frames (code, slides), when it changed
  frames/mm-ss.jpg every kept frame at 720p, readable without --at
  sheets/NN.jpg    3x3 contact sheets of those frames, labeled with times
  at/mm-ss.jpg     the --at stills

Frames are taken at scene cuts plus one every span/--frames seconds, then
near-duplicates are dropped, so a talking head shrinks to a few frames and
every slide or code change gets one.

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


def parse_time(t):
    """'90', '1:30', '1:02:03' -> seconds."""
    sec = 0.0
    for part in t.strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


# ---------- metadata ----------

def fetch_info(url):
    out = run([*YTDLP, "-J", "--no-playlist", "--no-warnings", url],
              capture_output=True, encoding="utf-8").stdout
    return json.loads(out)


def write_info(info, path):
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
    say("downloading audio")
    # The whole track, cut by ffmpeg below: --download-sections is slower
    # for audio and starts up to ~10 s early, which shifts every timestamp.
    run([*YTDLP, "-f", "bestaudio/best", "--no-playlist", "--no-warnings", "-q",
         "-o", str(work / "audio.%(ext)s"), url])
    audio = next(work.glob("audio.*"))
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


def download_video(url, work, height, section=None):
    cmd = [*YTDLP, "--no-playlist", "--no-warnings", "-q",
           "-f", f"bv*[height<={height}]/b[height<={height}]/worst",
           "-o", str(work / "video.%(ext)s"), url]
    if section:
        cmd[len(YTDLP):len(YTDLP)] = ["--download-sections", section, "--force-keyframes-at-cuts"]
    run(cmd)
    return next(work.glob("video.*"))


def cut_frames(video, work, every, shift):
    """One decode pass: a frame at each scene cut, plus one whenever `every`
    seconds pass without a cut. -> [(time, path)]"""
    raw = work / "raw"
    raw.mkdir()
    pick = f"isnan(prev_selected_t)+gt(scene\\,{SCENE})+gte(t-prev_selected_t\\,{every:.2f})"
    err = run([FFMPEG, "-v", "info", "-nostats", "-i", str(video),
               "-vf", f"select='{pick}',showinfo", "-fps_mode", "vfr", "-q:v", "2",
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


def ocr(path, scratch):
    """Tesseract on a cleaned copy (gray, dark themes flipped, 2x). Keeps
    confident lines and rebuilds indentation from where each line starts."""
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
                (int(c[6]), int(c[8]), float(c[10]), c[11].strip()))
    good = {}
    for key, words in lines.items():
        conf = sum(w[2] for w in words) / len(words)
        texts = [w[3] for w in words]
        grid = len(texts) > 8 and max(map(texts.count, texts)) > len(texts) / 2
        if conf >= 70 and not grid and re.search(r"\w{3}", " ".join(texts)):
            good[key] = words
    if not good:
        return ""
    widths = sorted(w[1] / len(w[3]) for ws in good.values() for w in ws)
    char = widths[len(widths) // 2]
    margin = {}
    for (block, _, _), ws in good.items():
        margin[block] = min(margin.get(block, ws[0][0]), ws[0][0])
    out = []
    for key in sorted(good):
        ws = good[key]
        indent = round((ws[0][0] - margin[key[0]]) / char)
        out.append(" " * min(indent, 40) + " ".join(w[3] for w in ws))
    return "\n".join(out)


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


def make_frames(url, work, out, duration, start, end, max_frames, read_text):
    a = start or 0
    b = end if end is not None else duration
    every = max((b - a) / max_frames, 2)
    height = 720 if read_text else 360
    say(f"downloading {height}p video; frames at cuts and every {every:.0f} s")
    section = f"*{a}-{b}" if (start is not None or end is not None) else None
    video = download_video(url, work, height, section)
    shift = a if section else 0  # a cut download starts at 0
    found = cut_frames(video, work, every, shift)
    video.unlink()
    kept = drop_repeats(found, max_frames)

    frames = out / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir()
    named = []
    for t, f in kept:
        p = frames / f"{clock(t).replace(':', '-')}.jpg"
        shutil.copy(f, p)
        named.append((t, p))

    if read_text:
        lines = write_screen(named, out / "screen.txt", work)
    else:
        lines = 0
        (out / "screen.txt").unlink(missing_ok=True)  # a stale one would mislead
    # No text anywhere and one unchanging scene: three frames show it all.
    # The rest stay in frames/ for a closer look.
    shown = named
    if read_text and not lines and len(named) > 3 and one_scene(named):
        shown = [named[round(i * (len(named) - 1) / 2)] for i in range(3)]

    sheets = out / "sheets"
    shutil.rmtree(sheets, ignore_errors=True)
    sheets.mkdir()
    paths = []
    for n, i in enumerate(range(0, len(shown), 9), 1):
        s = sheets / f"{n:02d}.jpg"
        contact_sheet(shown[i:i + 9], s)
        paths.append(s)
    return paths, len(found), len(named), len(shown), every, lines


def label_font(size):
    from PIL import ImageFont
    for name in ("LiberationSans-Regular.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def contact_sheet(frames, dest, w=426, h=240, pad=4, label=28):
    """3 across, each frame fit into w x h with its time centred under it."""
    from PIL import Image, ImageDraw
    rows = -(-len(frames) // 3)
    cell_w, cell_h = w + 2 * pad, h + label + 2 * pad
    sheet = Image.new("RGB", (3 * cell_w, rows * cell_h), "#111")
    draw = ImageDraw.Draw(sheet)
    font = label_font(20)
    for k, (t, p) in enumerate(frames):
        x, y = k % 3 * cell_w + pad, k // 3 * cell_h + pad
        with Image.open(p) as im:
            im.thumbnail((w, h))
            sheet.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))
        draw.text((x + w // 2, y + h + label // 2), clock(t), fill="white",
                  font=font, anchor="mm")
    sheet.save(dest, quality=82)


def make_stills(url, work, out, times):
    stills = out / "at"
    stills.mkdir(exist_ok=True)
    paths = []
    for t in times:
        sub = Path(tempfile.mkdtemp(dir=work))
        say(f"grabbing {clock(t)} at 720p")
        video = download_video(url, sub, 720, f"*{max(t - 2, 0)}-{t + 2}")
        p = stills / f"{clock(t).replace(':', '-')}.jpg"
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

    ap = argparse.ArgumentParser(prog="ytwatch", description=__doc__.split("\n\n")[0])
    ap.add_argument("url")
    ap.add_argument("--from", dest="start", help="start time, e.g. 5:00")
    ap.add_argument("--to", dest="end", help="end time, e.g. 9:30")
    ap.add_argument("--at", help="comma-separated times for sharp stills, e.g. 6:12,7:40")
    ap.add_argument("--frames", type=int, default=36, help="most frames to keep (default 36 = 4 sheets)")
    ap.add_argument("--no-frames", action="store_true", help="skip the video download")
    ap.add_argument("--no-ocr", action="store_true", help="360p frames, no screen.txt")
    ap.add_argument("--whisper", action="store_true", help="transcribe locally even if captions exist")
    ap.add_argument("--model", help="whisper model (default large-v3-turbo on an NVIDIA GPU, small.en on the CPU; 'small' for other languages)")
    ap.add_argument("--tempo", type=float, help="speed the audio up this much for whisper (default 1.5 on an NVIDIA GPU, 1 = off on the CPU)")
    args = ap.parse_args()

    start = parse_time(args.start) if args.start else None
    end = parse_time(args.end) if args.end else None
    read_text = not args.no_ocr and TESSERACT is not None
    if not args.no_ocr and not read_text:
        say("tesseract not installed, skipping screen.txt")

    say("reading video info")
    info = fetch_info(args.url)
    out = CACHE / info["id"]
    out.mkdir(parents=True, exist_ok=True)
    write_info(info, out / "info.txt")
    duration = info.get("duration") or 0
    if info.get("is_live"):
        sys.exit("ytwatch: live streams aren't supported")

    with tempfile.TemporaryDirectory(prefix="ytwatch-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)

        if args.at:
            times = [parse_time(t) for t in args.at.split(",")]
            for p in make_stills(args.url, work, out, times):
                print(p)
            return

        lines, source = (None, None) if args.whisper else fetch_captions(args.url, info, work)
        if not lines:
            lines, source = transcribe(args.url, info, work, args.model, start, end, args.tempo)
        words = write_transcript(lines, out / "transcript.txt", start, end)

        sheets = []
        if not args.no_frames:
            sheets, found, kept, shown, every, screen = make_frames(
                args.url, work, out, duration, start, end, args.frames, read_text)

    rng = f"{clock(start or 0)}-{clock(end if end is not None else duration)}"
    print(f"{info.get('title')}  ({info.get('channel') or info.get('uploader')}, {clock(duration)})")
    print(f"range:      {rng}")
    print(f"info:       {out / 'info.txt'}")
    print(f"transcript: {out / 'transcript.txt'}  ({words} words, {source})")
    if sheets:
        if read_text:
            print(f"screen:     {out / 'screen.txt'}  ({screen} lines)" if screen
                  else "screen:     no readable text on screen")
        print(f"frames:     {out / 'frames'}/  ({kept} kept of {found}: cuts + one per {every:.0f} s)")
        note = "" if shown == kept else f", {shown} of the frames: one unchanging scene"
        print(f"sheets:     {len(sheets)} x 3x3{note}")
        for p in sheets:
            print(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        name = "yt-dlp" if "yt_dlp" in e.cmd else Path(e.cmd[0]).name
        sys.exit(f"ytwatch: {name} failed (exit {e.returncode})")
