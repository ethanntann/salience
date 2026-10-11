# Capture contract: experimental

**Update:** [the play-only collector](README.md) saved evaluation footage but
caused FPS drops and game crashes on September 7. It is paused for normal use.
The ring-buffer experiment below remains separate and is not a validated fallback.

[The main scope](../SCOPE.md) owns priorities. Current path: WASAPI output levels
→ peak/sustained detector → delayed save → DDA/NVENC ring → **video-only** MP4.
`obs_client.py` is an unused alternative, not a daemon dependency.

Install `python -m pip install -e ".[capture]"`. From the root:

```powershell
python -m pipeline_v2.capture.daemon --list-devices
python -m pipeline_v2.capture.daemon --dry-run --monitor --seconds 600 --log tmp/session.csv
python -m pipeline_v2.capture.sweep tmp/session.csv
```

Dry run reads output audio for levels but saves no video/audio. Mark `]` for
wanted clips and `[` for kills. Failed preflight cleans up recording. Removing
`--dry-run` records the monitor and saves silent clips. Capture makes no VLM calls.

Each run owns an isolated temporary ring. Six five-second slots hold at most
**25 seconds of closed segments** because the active segment is excluded. A
trigger waits three seconds of post-roll plus a segment interval. Copy/concat
runs on one worker so metering continues. Changed sources reject the snapshot;
existing clips are not replaced. Shutdown cleans up process/ring; a trigger still
waiting for post-roll is not saved.

The signal time is not a game-event timestamp; saved event offsets are **not
known exactly**. Detector delay, scheduling, and segment boundaries matter.
The 75%-duration localizer is not validated for this recorder's output.

Release work: game/mic audio and sync, display/sleep/device recovery, hardware
checks and supported encoders, source selection, event metadata and handoff,
live recall at a bounded save rate, then packaging/updates. Default encoding is
NVIDIA-specific. Keep DDA until a tested requirement calls for WGC. OS capture
without injection is not an anti-cheat approval guarantee.

Earlier claims of a universal 5MB FFmpeg binary, equal API overhead, exact
arbitrary frame cuts, and zero-cost capture were not established. Use the main
assessment's sources and measurement plan.
