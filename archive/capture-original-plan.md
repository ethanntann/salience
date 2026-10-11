> Historical snapshot, superseded by the root README and pipeline_v2/SCOPE.md.
> Relative links below refer to the original document location.

# Capture — scope

Goal: one standalone app that records gameplay itself, so the product does not
assume the user runs ShadowPlay or Medal, nor that they enabled its auto-clip
feature, nor that they press a hotkey mid-fight.

The last point is the product reason this exists: asking a competitive player to
find a key during a duel means the best clips are the least likely to be saved.

## What "standalone" requires, and what it does not

Bundling **ffmpeg** is not the same as bundling OBS. ffmpeg is one ~5MB binary
with no UI, no tray icon, no config files and no user-visible process; LGPL builds
also avoid the GPL question OBS raises. The app still installs and runs as one
thing. A native encoder is therefore an optimisation, not a requirement for
shipping standalone.

## Capture API choice

| API | what it gives | status here |
| --- | --- | --- |
| **NVFBC** | driver-level, lowest overhead | **unavailable** -- NVIDIA gates it; ShadowPlay only |
| **Swapchain hooking** | game backbuffer before compositing, lowest practical overhead | **off the table** -- injecting into an Easy Anti-Cheat process risks user bans and a blacklisted binary. Not a code problem; incumbents have anti-cheat trust a new app does not |
| **DDA** (Desktop Duplication) | composited monitor, stays in VRAM | **in use**, via ffmpeg `ddagrab` |
| **WGC** (Windows.Graphics.Capture) | per-*window* capture, better HDR/DPI, actively supported | **deferred** -- see M5 |

DDA and WGC cost about the same. WGC's advantage is capturing the game window
rather than the whole monitor, and being the path Microsoft supports going
forward. Neither closes the gap to NVFBC or hooking, and nothing available to a
new entrant does.

## Measured, already

| configuration | CPU | RAM |
| --- | ---: | ---: |
| **`ddagrab` + `h264_nvenc`** (this design) | **4.3% of one core** | 296 MB |
| `gdigrab` + `h264_nvenc` | 54% of one core | 606 MB |
| `gdigrab` + `libx264` | 360% (3.6 cores) | 1653 MB |

1080p60 at 12 Mbps. The 80x spread is entirely about whether pixels leave the
GPU: `ddagrab` captures into VRAM and NVENC encodes there, so ffmpeg's process
only shepherds compressed bytes. The naive configuration -- the one most tutorials
show -- would cost a player frames. This one is noise.

For scale, the NVIDIA stack on the same machine measured 1.55% of one core and
693 MB across nine processes, in whatever state it was idling in. Same order of
magnitude, and a smaller footprint here.

**Not yet measured, and it is the number that actually matters:** in-game
frametimes with and without the recorder armed. CPU percent says nothing about
GPU contention, VRAM pressure, or 1% lows, and 1% lows are what a competitive
player notices. This needs a real game session.

## Architecture

```text
ddagrab ──► NVENC ──► wrapping segment ring        (video, never leaves VRAM)

WASAPI loopback ──┬──► LevelTrigger                (when to save)
                  └──► desktop audio track
microphone ──────────► mic audio track             (separate track, deliberately)

on trigger: concat ring ──► clip ──► existing pipeline
```

Two properties worth stating because they solve problems the watched-folder
version could not:

- **The event timestamp is known.** Saving at moment T puts the event a fixed
  distance from the end of the file. The localization problem that consumed a day
  of measurement -- `0.75 x duration`, the OCR banner detour -- does not exist
  here.
- **Audio tracks are separate.** The NVIDIA clips mix game and mic into one
  stereo track, which made voice-versus-effects analysis impossible. Recording
  our own removes that limit, and it is what any future work on reaction-driven
  clips needs.

### Why a segment ring rather than an in-memory ring

Raw frames are impossible: 30s of 1080p60 is ~11GB. Encoded frames in memory
means writing a muxer. ffmpeg's `-f segment -segment_wrap N` overwrites
`seg000..segNNN` cyclically -- a disk ring with no cleanup code and the muxing
already solved. Measured cost: 40 MB for a 30s buffer, ~1.3 MB/s sustained.

`-force_key_frames` at every segment boundary makes each segment independently
decodable, which is what allows concatenation to land exactly on a frame.

## Milestones

### M1 — Segment ring — **done**

`recorder.py`. Verified: the ring wraps correctly (`seg000` becomes newest),
concatenation produces a valid clip, 4.3% CPU.

Two details that only real runs exposed: segments must be ordered by **mtime not
filename** (the ring wraps, so names lie about age), and the newest segment must
be **dropped** on save because ffmpeg is still writing it.

### M2 — Trigger — **logic done, untuned**

`trigger.py`. Rolling-baseline peak detector: 90s baseline, +9.5 dB threshold,
25s cooldown, 20s warmup. Verified on simulated levels -- fires on real bursts,
suppresses one inside cooldown.

Audio is the trigger because loudness relative to a clip's own baseline
correlates **+0.237** with the user keeping a clip, measured over 334 labelled
clips. That is the strongest cheap signal found, and far cheaper than running a
CNN on frames continuously.

**Named honestly: it detects action intensity, not reactions.** Splitting peaks
by cause showed effect-driven peaks are *more* predictive (34.2% keep rate vs
25.2% for voice-driven), and voice fraction correlates **negatively** (-0.125).
Loud means a firefight, not friends laughing.

Remaining: tune the threshold against real gameplay. +9.5 dB comes from
clip-level envelope analysis, not from live meters.

### M3 — Audio tracks

Desktop audio via `soundcard` WASAPI loopback (verified working; ffmpeg has no
loopback input, and this machine has no Stereo Mix), mic as a second track. One
loopback reader feeds both the trigger and the recording.

**The known risk in this whole plan.** Piping two PCM streams from Python into
ffmpeg while it captures video invites A/V drift, and Python's GIL scheduling a
real-time audio callback is not ideal. Fallback if drift appears: give audio its
own segment ring and mux at save time -- more code, but the timing decouples
completely.

### M4 — Daemon and lifecycle

Detect a game running, arm the buffer, disarm when it exits. Restart ffmpeg if it
dies. Handle the DDA session breaking on resolution change, fullscreen
transitions, and UAC prompts -- these are *expected*, not exceptional, and the
recorder must recover rather than silently stop.

Then: on trigger, save, name the clip, hand it to the existing pipeline with the
event offset attached.

### M5 — WGC, deferred

Move from DDA to Windows.Graphics.Capture for per-window capture and better
HDR/DPI handling. Needs a native component (C# is the least painful for WinRT)
plus IPC to Python. Weeks, not days.

**Revisit only when:** users report broken capture on multi-monitor or HDR
setups, or capturing overlays/taskbar becomes a real complaint. Not before --
DDA is measured at 4.3% and the marginal gain is small.

### M6 — Packaging

Code signing (a bundled binary plus a background process is exactly what
SmartScreen flags unsigned), installer, ffmpeg LGPL notices, auto-update.

## Out of scope

Swapchain hooking (anti-cheat), NVFBC (gated), a native encoder (M5 first, and
probably never), macOS, capturing anything other than the primary monitor until
WGC lands.
