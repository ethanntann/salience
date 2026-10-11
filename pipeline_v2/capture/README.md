# Capture

```powershell
python -m pipeline_v2.capture.record            # record, then make highlights
python -m pipeline_v2.capture.highlights <run>  # re-run the post-session pipeline
python -m pipeline_v2 review --variant v3.6-ownkills-window@minimax-m3
```

## During play: `record` → `wgc`, `audio_pump`, `input_activity`, `marks`

- **Video:** Windows Graphics Capture of the Fortnite window (`--monitor` for
  the whole screen), paced to a constant 30 fps, NVENC at 8 Mbps, 30s MKV
  segments listed in `segments.csv`. Keyframe every 2s (OBS's default), so
  windows can be cut later without re-encoding.
- **Audio:** game loopback and microphone as two separate AAC tracks.
- **Input:** key-down presses, held keys and `GetLastInputInfo` activity, polled
  every 0.5s into `input.tsv`. F9/F10 marks go to `marks.json`.
- **Detectors** journal moments to `triggers.jsonl` and cost nothing. They no
  longer decide what is described (see below).

No hooks, no injection, nothing read from the game's memory: anti-cheat
compatibility is a hard constraint.

## After play: `highlights` → `clips`

1. `coverage.jsonl`: one request per 8s of footage, so the analysis spans tile
   the session with no gap and no overlap.
2. `windows/`: each 8s window cut (stream copy on 2s-keyframe footage), with a
   `.capture.json` sidecar placing it on the session timeline.
3. Described (7 frames over the 8s, `index`), stored, embedded. Every paid
   response is cached, so re-runs are free.
4. Ranked by taste when there are ≥10 ratings of both kinds in the variant,
   otherwise by an explicit fallback (`basis` in `highlights.json` says which).
5. `highlights/`: the top 10, never two from one fight, each re-cut from the
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
(the ratings came from the same session); session 2 is the out-of-sample check.

## Legacy

`launcher`, `collection` (Desktop Duplication, video only), `daemon`, `recorder`,
`trigger`, `sweep`, `obs_client` and `visual_trigger` are the earlier capture
experiments. Nothing in the current path imports them. The DDA collector caused
FPS drops and a crash on 2026-09-07, which is why WGC replaced it.
