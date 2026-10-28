# ytwatch

Lets Claude Code "watch" a YouTube video. Give it a link and it saves:

- `info.txt`: title, channel, length, chapters, description
- `transcript.txt`: timestamped transcript from YouTube's captions, or a local
  whisper transcription when there are none
- `screen.txt`: code and slide text read off the screen with OCR
- `frames/` and `sheets/`: 720p frames picked at scene cuts, plus labeled 3x3
  contact sheets Claude can look at

It ships as a Claude Code skill. Once it's installed, paste a YouTube link
into Claude Code and it runs on its own.

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

Then restart Claude Code and paste it a YouTube link.

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
ytwatch.py URL --from 5:00 --to 9:30   only that stretch, denser frames
ytwatch.py URL --at 6:12,7:40          sharp 720p stills of exact moments
ytwatch.py URL --no-frames             transcript only, fastest
ytwatch.py URL --no-ocr                360p frames, no screen.txt
ytwatch.py URL --whisper               ignore captions, transcribe locally
ytwatch.py clean                       delete the cache
```

Whisper runs on the CPU, about real time, and downloads its model (~500 MB)
the first time. Videos with captions skip it.

## When downloads start failing

YouTube changes things and yt-dlp catches up within days. On Windows, update
it with one run like this:

```powershell
uv run --upgrade-package yt-dlp --script $HOME\.claude\skills\youtube\ytwatch.py URL
```

On Linux or macOS, update yt-dlp with your package manager.
