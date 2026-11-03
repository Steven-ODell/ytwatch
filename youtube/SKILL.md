---
name: youtube
description: >-
  Watch a video with ytwatch.py (YouTube, any yt-dlp site like TikTok,
  Instagram, X, Vimeo, Reddit, or a local video file): it saves the video's
  info, a timestamped transcript (captions, or local whisper), the text OCR'd
  off the screen (code, slides), frames and labeled contact sheets you can
  Read as images. Screen recordings get screen mode: a frame at every visible
  change plus timeline.txt, what appeared on screen and what was said, step
  by step. Use whenever the user shares a video link or file, or asks what a
  video says, shows, or how to do what a video does. Triggers:
  "youtube.com", "youtu.be", "tiktok.com", "instagram.com/reel", "x.com",
  "twitter.com", "vimeo.com", ".mp4", ".mov", ".webm", "screen recording",
  "watch this video", "what does this video say", "summarize this video",
  "ytwatch".
---

# Watching a video

The script sits next to this file. Run it with uv (works in Git Bash and
PowerShell):

```
uv run --script "$HOME/.claude/skills/youtube/ytwatch.py" URL
```

Below, `ytwatch` means that whole command.

```
ytwatch URL                         # info + transcript + screen text + frames
ytwatch "C:\Users\me\Videos\clip.mp4"  # a local file, same output, no network
ytwatch FILE --screen               # screen recording: every change + timeline.txt
ytwatch URL --from 5:00 --to 9:30   # only that stretch, denser frames
ytwatch URL --at 6:12,7:40          # sharp 720p stills of exact moments
ytwatch URL --no-frames             # transcript only, fastest
ytwatch URL --no-ocr                # 360p frames, no screen.txt (faster download)
ytwatch URL --whisper               # captions are bad: transcribe locally
ytwatch URL --frames 18             # keep at most 18 frames (2 sheets)
ytwatch FILE --video                # normal mode for a file named like a screen recording
ytwatch clean                       # wipe ~/.cache/ytwatch
```

URL is a YouTube link or anything yt-dlp reads. TikTok, Instagram, X (a post
with a video), Vimeo and Dailymotion worked without cookies in November 2026.
It prints the paths. Files live in `~/.cache/ytwatch/<dir>/`: the bare video
id for YouTube, `<site>-<id>` for other sites (`tiktok-...`, `twitter-...`),
`file-<hash>` for local files (the same file always lands in the same dir).
The first run takes a minute or two while uv fetches Python, yt-dlp, deno
and ffmpeg; later runs start right away.

## Screen recordings

Local files named like a screen recording (`Screen Recording ...`,
`ScreenRecording_...`, Android `Screen_recording_...`, anything with "screen
record") get screen mode on their own. Windows' Snipping Tool names match;
Xbox Game Bar clips (named after the app) and OBS files don't, so pass
`--screen` for those. `--video` turns it off. Links default to normal mode.

What changes in screen mode:
- A frame at every visible change: a toggle flipping or one text line
  changing counts, a static screen gives one frame, cursor blinks, the
  pointer and a phone's status bar are ignored. While the screen keeps moving
  (scrolling, a video playing) it keeps one a second.
- Frames at native size up to 1568 px on the long edge, so phone text reads.
  Named `frames/mm-ss.s.jpg` (tenths of a second). Portrait recordings get
  4x2 sheets, landscape 3x3.
- `screen.txt` has the OCR of every frame. `timeline.txt` lists, per frame,
  `[mm:ss.s]` then the lines that appeared (`+`) and went away (`-`), with
  speech merged in as `said: ...`. **Read timeline.txt first**: it tells
  what the user did step by step as text. Open only the frames for the
  moments that matter (use the time to pick the file).
- No audio is normal for screen recordings: it prints "no audio, no
  transcript" and carries on.
- Without tesseract there is no OCR, so timeline.txt lists frames and speech
  only. Suggest `winget install UB-Mannheim.TesseractOCR` once.

## How to watch

1. Run `ytwatch URL` (or `ytwatch FILE --screen`). Read `info.txt` (chapters
   help) and `transcript.txt`. Screen recordings: `timeline.txt` first.
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
- Screen-mode frames are bigger (up to 1568 px, ~1.6k tokens each) and
  there can be dozens: pick them from timeline.txt, don't read them all.
- Captions: auto captions come from the `en-orig` track. If YouTube still
  refuses them (HTTP 429), ytwatch says "captions blocked (429), falling
  back to whisper" and transcribes the audio itself. If the audio download
  is blocked too, that needs cookies or a VPN, not a whisper fix.
- On Windows whisper runs `small.en` on the CPU: about 7x real time on a
  desktop CPU (a 30 min video in ~5 min), slower on a laptop. Say so before
  a long one and prefer `--from/--to`. Local files and screen recordings
  with speech go through whisper too (they have no captions). Non-English
  audio: `--model small`. The first whisper run downloads the model
  (~500 MB). On Linux with an NVIDIA GPU it uses `large-v3-turbo` on the GPU
  instead (~65x real time, plus a one-time 2.2 GB CUDA library download).
- Video and audio downloads go to a temp dir and are deleted. Only the text
  and JPGs stay in the cache (a few MB per video). Local files are read in
  place, never copied.
- Age-restricted, members-only and live videos fail at the info step. A
  site that wants a login (private reels, some X posts) fails there too;
  that needs cookies, say so instead of retrying.
- Vimeo page links need a login in yt-dlp, so ytwatch rewrites them to the
  player.vimeo.com link itself. Instagram often reports duration 0; that's
  fine.
- If downloads that used to work start failing, the site changed something:
  rerun once with `uv run --upgrade-package yt-dlp --script ...` to update
  yt-dlp.
- Summarize or quote short bits for the user. Don't paste the whole
  transcript back.
