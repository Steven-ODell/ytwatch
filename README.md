# ytwatch

Lets Claude Code "watch" a video: a YouTube link, a TikTok, Instagram, X or
Vimeo link (anything yt-dlp reads), or a video file on your computer. It saves:

- `info.txt`: title, channel, length, chapters, description
- `transcript.txt`: timestamped transcript from the site's captions, or a
  local whisper transcription when there are none (local files always use
  whisper)
- `screen.txt`: code and slide text read off the screen with OCR
- `frames/` and `sheets/`: frames picked at scene cuts, plus labeled
  contact sheets Claude can look at

Screen recordings get a more detailed mode: a frame at every visible change
(one edited line or a toggle counts), frames at full size so small text
reads, and `timeline.txt`, which lists what appeared and went away on screen
at each moment along with what was said. Claude reads that first and only
opens the frames that matter. Files named like a screen recording
("Screen Recording ...", Windows' Snipping Tool names included) get it on
their own; add `--screen` for anything else.

It ships as a Claude Code skill. Once it's installed, paste a video link or
a file path into Claude Code and it runs on its own.

## Install on Windows (PowerShell)

1. Install uv. It fetches Python, yt-dlp, deno and ffmpeg for you.

   ```powershell
   winget install astral-sh.uv
   ```

   Close and reopen the terminal afterwards.

2. Download this repo (green "Code" button, "Download ZIP") and unzip it.
   From the unzipped folder, copy `youtube` into your Claude skills folder:

   ```powershell
   mkdir -Force $HOME\.claude\skills
   Copy-Item -Recurse .\youtube $HOME\.claude\skills\
   ```

   You should end up with `%USERPROFILE%\.claude\skills\youtube\SKILL.md`
   and `ytwatch.py` next to it.

3. Optional, for reading code and slides off the screen:

   ```powershell
   winget install UB-Mannheim.TesseractOCR
   ```

4. Try it once by hand. The first run downloads everything, so give it a
   minute or two:

   ```powershell
   uv run --script $HOME\.claude\skills\youtube\ytwatch.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"
   ```

   It prints where it saved things (under `%USERPROFILE%\.cache\ytwatch`).

Then restart Claude Code and paste it a video link, or the path of a video
file.

## Install on Linux or macOS

Install `uv`, `yt-dlp`, `ffmpeg`, `deno` (yt-dlp needs it for YouTube) and,
optionally, `tesseract` with your package manager. Then:

```bash
mkdir -p ~/.claude/skills
cp -r youtube ~/.claude/skills/
uv run --script ~/.claude/skills/youtube/ytwatch.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

## Usage

```
ytwatch.py URL                         info + transcript + screen text + frames
ytwatch.py C:\path\to\clip.mp4         a local video, no network
ytwatch.py FILE --screen               screen-recording mode + timeline.txt
ytwatch.py FILE --video                normal mode for a file named like a screen recording
ytwatch.py URL --from 5:00 --to 9:30   only that stretch, denser frames
ytwatch.py URL --at 6:12,7:40          sharp 720p stills of exact moments
ytwatch.py URL --no-frames             transcript only, fastest
ytwatch.py URL --no-ocr                360p frames, no screen.txt
ytwatch.py URL --whisper               ignore captions, transcribe locally
ytwatch.py clean                       delete the cache
```

Local files and screen recordings work the same on Windows as on Linux. A
recording with no sound is fine; it says "no audio, no transcript" and
carries on.

Whisper runs `small.en` on the CPU, several times faster than real time
(about 7x on a desktop CPU), and downloads its model (~500 MB) the first
time. On Linux with an NVIDIA GPU it runs `large-v3-turbo` on the GPU
instead, about 65x real time, after a one-time 2.2 GB CUDA library
download. Videos with captions skip it; local files always use it when they
have sound.

## When downloads start failing

Sites change things and yt-dlp catches up within days. On Windows, update
it with one run like this:

```powershell
uv run --upgrade-package yt-dlp --script $HOME\.claude\skills\youtube\ytwatch.py URL
```

On Linux or macOS, update yt-dlp with your package manager.
