---
name: youtube
description: >-
  Watch a YouTube video with ytwatch.py: it saves the video's info, a
  timestamped transcript (YouTube captions, or local whisper when they are
  missing or blocked), the text OCR'd off the screen (code, slides), 720p frames and
  labeled 3x3 contact sheets you can Read as images. Use whenever the user
  shares a YouTube link or asks what a video says, shows, or how to do what
  a video does. Triggers: "youtube.com", "youtu.be", "watch this video",
  "what does this video say", "summarize this video", "ytwatch".
---

# Watching a YouTube video

The script sits next to this file. Run it with uv (works in Git Bash and
PowerShell):

```
uv run --script "$HOME/.claude/skills/youtube/ytwatch.py" URL
```

Below, `ytwatch` means that whole command.

```
ytwatch URL                         # info + transcript + screen text + frames
ytwatch URL --from 5:00 --to 9:30   # only that stretch, denser frames
ytwatch URL --at 6:12,7:40          # sharp 720p stills of exact moments
ytwatch URL --no-frames             # transcript only, fastest
ytwatch URL --no-ocr                # 360p frames, no screen.txt (faster download)
ytwatch URL --whisper               # captions are bad: transcribe locally
ytwatch URL --frames 18             # keep at most 18 frames (2 sheets)
ytwatch clean                       # wipe ~/.cache/ytwatch
```

It prints the paths. Files live in `~/.cache/ytwatch/<video id>/`. The
first run takes a minute or two while uv fetches Python, yt-dlp, deno and
ffmpeg; later runs start right away.

## How to watch

1. Run `ytwatch URL`. Read `info.txt` (chapters help) and `transcript.txt`.
2. Read `screen.txt` if it exists. It has the code, slide text and labels
   seen on screen, with times and rebuilt indentation. Text is far cheaper
   than images, so start here for anything on screen. OCR can garble a
   character or two (`1l`, `@` for `0`); check a frame when it matters.
   No screen.txt and a "tesseract not installed" note means OCR is off;
   suggest `winget install UB-Mannheim.TesseractOCR` once.
3. Read the sheets in `sheets/` when what things look like matters. Each
   frame has its time under it. Frames are picked at scene cuts, and
   repeated shots are dropped. For a single unchanging scene with no
   text (talking head, podium), the sheet shows only 3 frames.
4. Need a closer look at a moment? `frames/mm-ss.jpg` holds every kept
   frame at 720p. Read the one nearest the time before reaching for `--at`.
   Use `--at` only for a moment between kept frames.
5. Long video and only one part matters: use `--from/--to` instead of
   reading the whole thing.

## Cost and limits

- A sheet costs about 1.4k image tokens, a 720p frame about 1.2k. screen.txt
  for a code-heavy video is usually under 1k tokens. Don't read sheets
  or frames you don't need.
- Captions: auto captions come from the `en-orig` track. If YouTube still
  refuses them (HTTP 429), ytwatch says "captions blocked (429), falling
  back to whisper" and transcribes the audio itself. If the audio download
  is blocked too, that needs cookies or a VPN, not a whisper fix.
- On Windows whisper runs `small.en` on the CPU: about 7x real time on a
  desktop CPU (a 30 min video in ~5 min), slower on a laptop. Say so before
  a long one and prefer `--from/--to`. Non-English audio: `--model small`.
  The first whisper run downloads the model (~500 MB). On Linux with an
  NVIDIA GPU it uses `large-v3-turbo` on the GPU instead (~65x real time,
  plus a one-time 2.2 GB CUDA library download).
- Video and audio downloads go to a temp dir and are deleted. Only the text
  and JPGs stay in the cache (a few MB per video).
- Age-restricted, members-only and live videos fail at the info step.
- If downloads that used to work start failing, YouTube changed something:
  rerun once with `uv run --upgrade-package yt-dlp --script ...` to update
  yt-dlp.
- Summarize or quote short bits for the user. Don't paste the whole
  transcript back.
