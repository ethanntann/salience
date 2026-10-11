# September 7 unseen-session pilot

The recorder is **not ready for normal gameplay**. The saved footage is usable
for offline investigation; user-reported FPS drops and crashes fail the product's
capture requirement. v2 remains the authoritative pipeline.

## Capture

Session `20260907-142124-24be3d` contains 194 finalized chunks, 5,746.902 seconds
of video and 8,584,498,498 bytes. All media/sidecar checks passed. Three preselected
moving chunks decoded cleanly; downscaled grayscale adjacent-frame comparisons
found 0–0.33% near-identical frames and no repeated span over 100ms. Static scenes
can also repeat. This is **not** a game-frametime benchmark or evidence against
the user's experience.

Fortnite's crash context and log confirm an `EXCEPTION_ACCESS_VIOLATION` at
18:24:31 UTC while capture was active. The game used the RTX 4060 Laptop GPU;
capture used DDA + Intel Quick Sync on the internal display. The crash report did
not flag out-of-memory. Root cause is unresolved: an access violation does not
identify capture, a driver, or the game itself as the cause.

The shortcut no longer arms this configuration. `--test-capture` is an explicit
developer test escape hatch, not a repaired recording mode. Before re-enabling
normal recording, measure comparable no-capture and capture gameplay with game
frame times/1% lows, GPU engines, memory and crashes. Start with reduced capture
cadence as a hypothesis, then compare the encoding/capture path independently.
Do not change the user's game or driver settings to hide recording overhead.

## Frozen description pilot

Before inference, selected 12 equally spaced chunk indices across the session,
using each chunk's existing uniform four-second window. Separately selected the
six middle full chunks of the longest uninterrupted run: run-006 parts 00037–00042.
These cover 180 seconds and were swept with 48 consecutive windows of at most
four seconds. The uniform and continuous cohorts overlap in source footage and
are not independent test sets.

All 60 calls used the configured `qwen3p7-plus`, v3.3 capture prompt, and seven
1440px frames. Responses used 563,100 input and 6,057 output tokens. Estimated
cost is $0.299721 using the higher of the conflicting price pairs on the
[provider model card](https://fireworks.ai/blog/qwen-3p7-plus), not an invoice.
One explicit HTTP 403 rejection interrupted the batch after 58 completions; the
remaining two succeeded after a pause. Completed requests were not repeated.

**14 of 18 spectator windows explicitly misdescribe spectator gameplay as a
pre-match lobby. All 60 responses claim high confidence.** For example, after the
recording player dies, “Finding Next Match” overlays remain while the opponent
continues an actual match. The model mistakes that UI for pregame. At sweep-22 it
describes an elimination but does not identify the spectated actor. Coarse
`no_event`/`death`/`elimination` agreement therefore hides consequential errors.
No overall factual-accuracy percentage is claimed; weapon names, motion details,
health and attribution still need independent user review.

Assistant annotations were made from contact sheets before reading predictions.
Full-resolution adjudication afterwards corrected a missed NPC elimination:
`ELIMINATION / NITE Unit Sentry` is legible at sweep-03, with its banner persisting
into sweep-04. Both original and corrected references are preserved. These are
provisional machine-assisted annotations, not human ground truth or preference
labels. This inspected session is now validation data; do not call it untouched
after tuning prompts against these results.

## Highlight-selection diagnostic

Historical labels point to the old Windows user folder. Ranking now retains
those labels when their descriptions exist, without requiring the old video
files. Actual video indexing still excludes missing media.

The baseline fits TF-IDF/LSA only on 334 historical labelled descriptions (105
keeps) and trains the v2 logistic ranker. No current-session labels or summaries
are used in fitting. This does **not** reproduce the historical MiniLM semantic
encoder, which is not installed in this environment.

Its three highest-ranked continuous windows are 8–12s, 50–54s and 46–50s:
approach/mobility followed by spectator aiming. Merging adjacent windows yields
two proposals. None overlaps the three provisionally annotated completed events:
NPC elimination at 14–16s, own death at 20–24s, and spectator elimination at 84–88s.
The first-three and seeded-random-three window baselines also miss them. This
small event-overlap check is not highlight precision/recall: whether those events
are wanted is unknown, and an appealing clip could occur without an elimination.

The existing one-uniform-window-per-chunk sampler overlaps only the spectator
elimination in this stretch. It misses the recording player's NPC kill and death.
All footage remains saved; these are **analysis coverage** misses, not lost video.
The offline dense sweep is an evaluation extension, not a live detector wired
to the recorder.

## Review and reproduce

The user completed all **12 uniform session samples**: **12 skips**, with
**11 descriptions rated accurate and one rated minor error**. Their notes identify
lobby waiting, traversal, looting and building/editing without a wanted kill.
This supports description usefulness on these background windows. It does not
measure kill-description accuracy or missed-highlight recall: there are no
user-labelled positive examples in this subset. The continuous timeline is not
yet marked complete in the saved review, so no user highlight recall is reported.

Retain these as evaluation negatives. The next evaluation needs verified action
and wanted moments from the full saved footage alongside this background sample.
Selection must be able to return zero highlights from an uneventful stretch;
always presenting a fixed top count guarantees unwanted output on all-skip input.
Do not assume every kill is wanted, or exclude quiet moments by definition.

Pilot files are in ignored `.local-data/evaluation-20260907/`: frozen plan, source
hashes, exact frames, predictions, scores, original/adjudicated references, media,
capture audit and metrics. Nothing was added to the training/preference database.

```powershell
python -m pipeline_v2.evaluation_review .local-data/evaluation-20260907
python -m pipeline_v2.evaluation_metrics .local-data/evaluation-20260907
```

Open `http://127.0.0.1:8778`. First watch the continuous three minutes, mark every
moment you would keep, and confirm the review is complete. Review the 12 spread
samples next: save Keep/Skip/Unsure before revealing the description, then judge
accuracy. Decisions persist to `user-labels.json`, outside training. User highlight
recall remains unset until the complete timeline is labelled. The metrics command
then compares the frozen candidate lists against those wanted intervals.

Priority order: restore reliable low-impact capture; fix spectator/own-player
attribution and measure it on a fresh session; validate temporal candidate
selection; then measure whether audio adds enough benefit to justify its overhead.
Do not expand the live recorder or prioritize transcription based on this pilot.
