# Automatic event proposals and shot evidence

The offline prototype now locates candidate events in complete saved chunks and
derives focused shot windows without manual event timestamps. It is opt-in,
cloud-assisted, and not integrated into live capture or automatic highlight saves.

## Method

`localization.py` sends 16 timestamped full-frame samples spanning a finalized
chunk, asks for visible completed events (including none), and refines each
proposal with 19 frames around its interval, padded by one second. It reports
event kind, actor, evidence and timestamp bounds. A coarse miss cannot be recovered
by refinement. Persistent banners and unrelated kill-feed events are explicitly
excluded in the instructions, but instruction compliance still needs validation.

Only recording-player elimination intervals at most 0.75 seconds wide produce a
shot-evidence window, padded by one second before and 0.15 seconds after. Broad,
victory, unknown-actor and spectator events do not automatically enter that step.
These are provisional development settings, not tuned deployment thresholds.
The supplied shot window is evidence for description, not a proposed final clip
duration and not a keep decision.

## Seven-chunk development test

The cohort was selected and frozen before localization. Four source chunks contain
previously reviewed own-player events, one contains the known spectator kill, and
two are combat/pregame controls. All inputs span their complete approximately
30-second chunk; no known event time was given to the localizer. The controls were
inspected by the assistant rather than exhaustively annotated by the user.

| Source case | Refined model proposal, source seconds | Result |
| --- | --- | --- |
| action-00 | Own elimination, 13.348–13.679 | Known event found |
| action-02 | Own elimination, 14.336–14.557 | Known event found |
| action-04 | Own elimination, 6.060–6.392 | Known event found |
| action-06 | Victory, 25.203–28.795 | Known event found; interval remains broad |
| action-05 | None | No event proposed on combat control |
| sample-00 | None | No event proposed on pregame control |
| sweep-22 | None | Known spectator kill missed by coarse scan |

Times are model-selected brackets between supplied samples, not frame-perfect
ground truth. The victory banner is visible beginning in the sampled frame at
28.795 seconds, but the entire proposed interval does not isolate its decisive hit.
The results cover roughly 3.5 minutes selected from a known recording, not the
whole 96-minute session. They do not establish full-session recall or precision.

## Automatically selected weapon inputs

The three eligible narrow proposals generated 19-frame shot windows through a
fixed rule; no manual timestamp correction was made. The unchanged experimental
attribution prompt returned:

| Case | Generated window | Returned kill weapon | Interpretation |
| --- | --- | --- | --- |
| action-00 | 12.348–13.829 | SMG | Family agrees with user Drum Gun reference and observed SMG kill feed |
| action-02 | 13.336–14.707 | Burst Auto Shotgun | Shotgun family agrees; exact name and narrated earlier shots are unverified |
| action-04 | 5.060–6.542 | Pump Shotgun | Shotgun family agrees; exact name and narrated switch remain unverified |

All three match at family level on these known cases. This is not evidence of
three fully accurate descriptions. In particular, action-02 changes its exact
shotgun name from the manually centered trial and adds a revolver shot sequence.
Model confidence is high despite those unresolved details. Do not use confidence
or numbered citations as independent factual verification.

## Engineering and cost

The localization and description stages share one cached request transport in
`vlm.py`, preserving existing request keys and raw-response-before-validation
behavior. Invalid localization responses remain cached rather than being paid
for again to repair parsing. Tests cover empty results, invalid intervals/actors,
unchanged description caching, changed input rejection, and shot-window eligibility.

Localization used 11 requests, 243,033 input and 879 output tokens, estimated
$0.124154. Three automatic shot descriptions added 75,417 input and 894 output
tokens, estimated $0.040391. Combined estimated cost: **$0.164544** under the
previous conservative price assumptions. This excludes earlier experiments and
is not an invoice or a measured full-session cost. Local processing performance
and gameplay impact were not benchmarked; recording remains paused.

The prototype accepts only finalized capture chunks, caps this pilot at 60 seconds
per input, freezes source/model/prompt/resolution, validates intervals, verifies
persisted frame hashes on resume, and never exports or deletes gameplay clips.
No existing user judgments, baseline predictions or preference records changed.

## Use and next validation

This is separate from `python -m pipeline_v2 localize`, which only inspects the
old arithmetic/sidecar estimate without a model call. The new opt-in command is:

```powershell
python -m pipeline_v2.localization path/to/finalized-chunk.mkv .local-data/proposal-test --max-calls 4
```

The default allowance is zero new calls. Each chunk can require one coarse call
plus up to three refinements. No candidate means no refinement. Resuming completed
stages uses persisted results. Outputs include plans, exact model inputs, raw and
parsed responses, and `proposals.json` with eligible `shot_windows`.

Current artifacts are in `.local-data/evaluation-localization-20260908/` and
`.local-data/evaluation-auto-shot-20260908/`; the latter includes a full comparison
report against preserved user notes. The source cohort and automatic window rules
are retained for audit.

Next validation should use untouched, fully reviewed gameplay stretches, include
quiet events and transitions across chunk boundaries, and measure missed wanted
moments plus unwanted proposals per hour. Address coarse-scan misses before
assuming all action reaches the better shot describer. Piece control and spectator
match-state narration remain unresolved in their separate evaluations. Default
v3.3 descriptions and the paused recorder remain unchanged.
