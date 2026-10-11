# Salience

Salience records your Fortnite sessions and keeps the clips *you* would have kept.
You play; when you stop recording, it watches the whole session, describes every
8 seconds with a vision model, and ranks the moments by a taste model trained on
your own keep/skip ratings. No OBS, and nothing touches the game process.

```
game window ──WGC──▶ NVENC segments (+ game & mic audio)          during play
                     │
                     ▼                                              after stop
  8s coverage windows ─▶ 7 frames each ─▶ VLM description ─▶ embedding
                                                               │
            your keep/skip ratings ─▶ taste ranker ◀───────────┘
                                          │
                                          ▼
                              highlights/ (top 10, with lead-in)
```

## Use it

Windows, Python 3.11+, FFmpeg on PATH, an NVIDIA GPU, and a Fireworks API key in
`.env` (`FIREWORKS_API_KEY`, `FIREWORKS_MODEL`).

```powershell
python -m pip install -e ".[dev]"
python -m pipeline_v2.capture.record      # or double-click "Start Recording.cmd"
```

Start Fortnite, play, press **Ctrl+C** when done. The highlights folder opens
when ranking finishes. To teach it your taste, rate clips at
http://localhost:8777/ (**K** keep, **J** skip, **W** description is wrong):

```powershell
python -m pipeline_v2 review --variant v3.6-ownkills-window@minimax-m3
python -m pipeline_v2.capture.highlights <session folder>   # re-rank with new ratings
```

## How it works, and why

- **Capture is passive.** Windows Graphics Capture of the game window, WASAPI
  loopback plus microphone as separate tracks, keyboard state by polling. An
  anti-cheat sees an ordinary screen recorder, which was a hard requirement.
- **Nothing heavy runs while you play.** The encoder writes 30s segments with a
  keyframe every 2s. All cutting and describing happens after you stop, and
  windows are stream-copied rather than re-encoded (0.2s each instead of 2.4s).
- **Every moment is looked at, not just the ones a detector flagged.** The
  first design described only where a live detector fired (button-mashing
  bursts, voice spikes). A seeded random-control experiment showed why that
  fails: the trigger overlapped **5 of 15** moments the player kept. Describing
  the whole session costs about **$0.001 per window, ~$0.25 per half hour**.
- **Taste is learned from descriptions.** Each description is embedded and a
  logistic ranker is trained on the player's ratings.

## Measured (session 1: 31 minutes of real play, 146 rated clips)

| Ordering | Agrees with the player, pairwise |
|---|---|
| Taste ranker, held-out minutes (time-blocked CV) | **~91%** |
| Rule: eliminations, then knocks | 74% |
| Which detector fired the clip | 57% |
| Chance | 50% |

The player skipped some eliminations and kept moments with no kill, which is
what a fixed rule cannot learn. Caveat: one session and 17 keeps. Whether taste
learned on one session ranks the next is the open question.

## Layout

- [`pipeline_v2/capture/`](pipeline_v2/capture/): recorder (`record`, `wgc`,
  `audio_pump`, `input_activity`), cutting (`clips`), the post-session pipeline
  (`highlights`), and the detector experiments (`triggers`, `detector_control`).
- [`pipeline_v2/`](pipeline_v2/README.md): describe (`vlm`, `prompt`), store,
  embed, `taste`, `ranking`, and the review page.
- [`salience_media/`](salience_media/): shared FFmpeg helpers.
- `backend/`, `frontend/`: the retired v1 (local student model), kept for reference.

```powershell
python -m pytest pipeline_v2 salience_media
python -m ruff check pipeline_v2 salience_media
```
