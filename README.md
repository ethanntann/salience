# Salience

Salience records your Fortnite sessions and keeps the clips *you* would have kept.
You play; while you play, it describes every 8 seconds of your matches with a
vision model, and when you stop it ranks the moments by a taste model trained
on your own keep/skip ratings, so the highlights are ready seconds later. No
OBS, and nothing touches the game process.

```
game window ──WGC──▶ NVENC segments (+ game & mic audio)          during play
                     │  each finished 30s segment, inside a match:
                     ▼
  8s coverage windows ─▶ 7 frames each ─▶ VLM description
          mic track ─▶ Whisper (local, between matches) ─▶ what was said
                                                               │   after stop
                                        embeddings + speech vector
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
python -m pip install -e ".[dev,audio]"   # audio: local mic transcription
python -m pipeline_v2.capture.record      # or double-click "Start Recording.cmd"
```

Start Fortnite, play, press **Ctrl+C** when done. The highlights folder opens
when ranking finishes. To teach it your taste, rate clips at
http://localhost:8777/ (**K** keep, **J** skip, **W** description is wrong):

```powershell
python -m pipeline_v2 review --variant v3.7-wordbank-window@deepseek-v4p1-flash --only <session folder>
python -m pipeline_v2.capture.highlights <session folder>   # re-rank with new ratings
```

## How it works, and why

- **Capture is passive.** Windows Graphics Capture of the game window, WASAPI
  loopback plus microphone as separate tracks, keyboard state by polling. An
  anti-cheat sees an ordinary screen recorder, which was a hard requirement.
- **Described while you play, so there is no wait.** The encoder writes 30s
  segments with a keyframe every 2s. As each one closes, its windows inside a
  match (per Fortnite's log) are stream-copied out (0.2s each, no re-encode)
  and sent to the describer. Frames are decoded on the GPU's video decoder,
  which the game does not use (72% less CPU, byte-identical frames), one at a
  time. That mattered: as a burst of CPU decoders it dipped a 165fps game to
  100–120 every 30s; as a trickle it holds ~160, the same as capture alone.
  Replaying session 2 this way: 8s from stop to highlights. Whisper runs only between matches. Describing
  after the fact, one call at a time, had been 4.6s per window (about 20
  minutes a session); 16 calls in flight brought that to 0.8s
  (`--no-live` still works that way).
- **Every moment is looked at, not just the ones a detector flagged.** The
  first design described only where a live detector fired (button-mashing
  bursts, voice spikes). A seeded random-control experiment showed why that
  fails: the trigger overlapped **5 of 15** moments the player kept. Describing
  the whole session costs about **$0.002 per window, ~$0.35 per half hour of
  matches**. Most of that is the 7 frames; 4 frames cut tokens by a third but
  missed kills (agreement 97.3% → 94.5%), so 7 stays.
- **Game vocabulary, not captions.** The describer gets a Fortnite word bank
  (`pipeline_v2/games/fortnite.py`): each term with the visible cue that earns
  it (a crack, a box, piece control, a third party) and a ban on scenery.
- **Ranking is what happened, then taste.** A win, then eliminations, then
  knocks; within each, a logistic ranker trained on the player's own ratings
  of the descriptions. Taste alone lost to that order out of sample.
- **What was said counts only if it matters to you.** The mic is transcribed
  locally and kept as a separate signal. Its weight is chosen by
  cross-validation on your own ratings, so funny-moment players get it and a
  player whose talk is callouts does not. Mixing speech into the descriptions
  measured 92.6% → 83.3% on session 1, which is why it is kept separate.

## Measured (out of sample)

Taste trained on session 1 (31 min, 146 ratings) and scored against the
player's ratings of session 2 (43 min, a different evening; the 271 windows
of actual gameplay). Three describers, same footage, same ratings:

| | minimax-m3 | deepseek-v4p1-flash | **deepseek + word bank** (shipped) |
|---|---|---|---|
| What happened, then taste (pairwise) | 96.5% | 96.4% | **97.4%** |
| Taste alone | 86.9% | 90.5% | 88.6% |
| Ordering kills among themselves | 42.0% | 56.2% | 53.7% |
| Kept moments in the top 10 / top 15 | 6 / 9 of 16 | 8 / 11 | 7 / 11 |
| Summaries mentioning scenery | 112 / 320 | 103 / 320 | **73 / 320** |
| Game terms per summary | 0.31 | 0.35 | **0.51** |

Ten random picks cover about 2 of the 16 kept moments. Skipped before any of
this: lobby, loading, the pre-game island and spectating (49 of 320 windows),
identified from Fortnite's own log, none of them a keep.

The describer was chosen by a bake-off of eight Fireworks vision models on the
nine hardest windows (the player's "description wrong" flags and the cases two
models disagreed on). minimax-m3 named the wrong weapon on five of nine and
called a 106-damage shield crack a kill; deepseek matched kimi-k3, the most
accurate, on eight of nine at a twentieth of its price (~$0.002 a window).

What is not solved: **which kills** a player keeps. Most keeps are kills, but
so are many skipped fights, and ordering kill windows among themselves is
barely above chance (~54%, on ~30 windows). Also: two sessions, one player.

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
