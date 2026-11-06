# Capture

```powershell
python -m pipeline_v2.capture.record            # record, then make highlights
python -m pipeline_v2.capture.highlights <run>  # re-run the post-session pipeline
python -m pipeline_v2 review --variant v3.9-said-window@deepseek-v4p1-flash --only <session>
```

## During play: `record` → `gfx`, `audio_pump`, `input_activity`, `live`

- **Video:** Windows Graphics Capture of the Fortnite window (`--monitor` for
  the whole screen), captured at up to 60 fps and paced to a constant 30 fps
  (capturing at exactly 30 repeated ~10% of frames, see `wgc`), NVENC at 8 Mbps, 30s MKV
  segments listed in `segments.csv`. Keyframe every 2s (OBS's default), so
  windows can be cut later without re-encoding.
- **Audio:** game loopback and microphone as two separate AAC tracks.
- **Input:** key-down presses, held keys and `GetLastInputInfo` activity, polled
  every 0.5s into `input.tsv`. There are no hotkeys: nothing for the player
  to press (removed 2026-10-23; old sessions' `marks.json` stays readable).
- **Detectors** journal moments to `triggers.jsonl` and cost nothing. They no
  longer decide what is described (see below).

No hooks, no injection, nothing read from the game's memory: anti-cheat
compatibility is a hard constraint.

- **Live describing (`live`):** every 30s, the windows that are fully on disk
  and that the log places inside a match are cut and described, one window
  and one frame at a time, GPU-decoded, below normal priority (a burst of CPU
  decoders cost ~50fps; the trickle costs none measurable). Between matches
  the mic is transcribed too. It does only what step 2–4 below would do, the
  same way, so the final run finds them done; anything it missed (the last
  segment, a failed call, no log) is left to that run.

## After play: `highlights` → `clips`

1. `coverage.jsonl`: one request per 8s of footage, so the analysis spans tile
   the session with no gap and no overlap.
   Windows outside the player's own gameplay are skipped (`gameplay`): match
   phases (pre-game island, Battle Bus, match), death and match end come from
   Fortnite's own log, which the game always writes. Session 2: 49 of 320
   windows (lobby, item shop, loading, pre-game island, spectating) skipped,
   none of them a keep. No log, no gating.
2. `windows/`: each 8s window cut (stream copy on 2s-keyframe footage), with a
   `.capture.json` sidecar placing it on the session timeline.
3. `mic_transcript.json`: the mic track transcribed locally (Parakeet TDT 0.6B
   int8 on the CPU, every second of it, no voice detector), about 2s per 30s
   of footage on all cores, cached per segment
   (`voice`). Each window gets what was said during it.
4. Described (7 frames over the 8s, `index`, 16 calls in flight), stored,
   embedded. Every paid response is cached, and only windows with no
   description are looked at, so after a live session this is the last
   segment or two.
5. Ranked by taste when there are ≥10 ratings of both kinds in the variant,
   otherwise by an explicit fallback (`basis` in `highlights.json` says which).
   Speech is weighed in only if it improves this player's held-out accuracy
   (see below).
6. `highlights/`: the top 10, never two from one fight, each re-cut from the
   raw footage with 8s of lead-in.

`--prune-raw` deletes the raw segments once highlights exist. It refuses if any
window failed to cut.

## Why coverage instead of triggers (session 1, 2026-09-28, 31 min)

A seeded uniform control arm was cut by the same code as the triggered clips
(`detector_control`). Kill/knock rates are from the VLM's descriptions. Ratings
are the player's; the review page showed filenames, so they were not blind to arm.

| Arm | clips | own kill/knock (VLM) | player kept |
|---|---|---|---|
| input bursts | 29 | 24% | 5 |
| voice spikes | 44 | 4.5% | 5 |
| random control | 73 | 8.2% | 7 |

Input bursts beat random on events (z = +2.2, fragile). But the gate is the
problem: input bursts overlapped only 5 of the 15 moments the player kept, and
the random arm found more keeps than either detector. With full coverage, the
taste-ranked top 15 exports cover 9 of those 15, and taste orders windows the
way the player rated the overlapping clips 86% of the time. That is in-sample
(the ratings came from the same session). Session 2 was the out-of-sample
check, and taste alone did not hold up: 86.9% against 92.3% for "what
happened" alone, 96.5% for the two combined, which is now the ranking (see
the top-level README).

## Speech: weighed per player, not assumed

The vision model sees frames and hears nothing, so a moment that is funny for
what was said looks like nothing happened. Session 1 (104 of 146 rated clips
had speech), held-out taste, 60s / 120s blocks:

| speech | pairwise |
|---|---|
| none (descriptions only) | 92.6% / 92.1% |
| appended to the description text | 83.3% / 83.8% |
| own vector, weight 1.0 | 86.2% / 91.3% |
| own vector, weight 0.3 | 92.8% / 91.7% |

This player's talk was callouts, and mixing it into the description drowned
out the picture. So speech is kept as its own vector, and `ranking` picks its
weight from {0, 0.3, 1} by cross-validation on the player's own ratings,
using it only when it beats descriptions alone by a point. For session 1 it
chose 0, so the ranking is unchanged. Whether it switches on for a player
whose keeps are the funny moments is tested only on synthetic data
(`test_speech_weight`); no such player has rated yet.

## Legacy

`launcher`, `collection` (Desktop Duplication, video only), `daemon`, `recorder`,
`trigger`, `sweep`, `obs_client` and `visual_trigger` are the earlier capture
experiments. Nothing in the current path imports them. The DDA collector caused
FPS drops and a crash on 2026-09-07, which is why WGC replaced it.
