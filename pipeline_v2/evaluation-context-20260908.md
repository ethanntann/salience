# Focused shot windows, longer build context, and spectator checks

The focused shot-window experiment improved weapon-family attribution on the
three known failures. Longer lead-in still did not produce piece-control
recognition. Explicit spectator ownership generalized to four more windows in
the same sequence, but the accompanying match-state narration stayed wrong.
These remain development results; the default pipeline is unchanged.

## Design

The model and attribution prompt from the [previous comparison](evaluation-attribution-20260908.md)
were held fixed. Twelve calls were frozen before inference:

- Three manually centered shot windows, 19 full frames each: action-00 at
  12–13.7 seconds, action-02 at 13.3–14.7, action-04 at 4.5–6.7.
- Two longer action-04 inputs, 19 frames each: current chunk 0.1–7.5 seconds;
  and previous chunk 22–29.9 followed by current chunk 0.1–7.5. The latter covers
  approximately 15.5 seconds with a 0.2-second sampling gap across the boundary.
  Sidecars confirm the two chunks are consecutive in the same run (301 seconds
  is their common boundary). Longer coverage also changes sample spacing.
- Four additional spectator windows from the original pilot (sweep-08, 12, 16,
  20), plus loading (24), pregame (26), and own-gameplay (34) controls, each using
  its exact original seven images. These context references are prior assistant
  annotations, not newly supplied user labels or independent sessions.

The shot times are manually selected from known events. This tests whether
better-targeted evidence helps; it does not implement automatic event localization.
No labels or model-predicted explanations were supplied as answers in the prompt.

## Results

| Case | User reference | Earlier four-second attribution, 19 frames | Focused shot window, 19 frames |
| --- | --- | --- | --- |
| action-00 | Drum Gun | Assault Rifle | SMG |
| action-02 | Shotgun one-pump | Wrecker Revolver | Pump Shotgun |
| action-04 | Shotgun kill | Pump Shotgun | Pump Shotgun |

All three focused results agree at the family level with the user reference
(the action-00 visible kill feed also says SMG). This does not establish exact
weapon names or correct narrated swaps and damage. For example, the new action-02
description adds an initial pistol hit and weapon switch; those details have not
been adjudicated. Do not treat the whole sentence as verified because its final
weapon family is plausible. One response per known case cannot establish a
reliable success rate.

The current-chunk lead-in reports editing and edit-reset but still misattributes
the kill to a Nemesis AR. The cross-chunk input reports a floor edit, close-range
shotgun kill, and damage taken. Neither reports piece control. Earlier placements
are now represented, but ownership and how the opponent's routes were restricted
remain unestablished by the model. This does not prove that all decisive ownership
evidence was captured; the longer input has wider spacing and is still sampled
stills. Specific edit/reset sequences need annotation before scoring correctness.

All four additional spectator windows return `viewpoint=spectating`. None of the
three controls is marked spectating: loading is non_gameplay, pregame and the
drop sequence are own_gameplay. However, **all four spectator summaries still
incorrectly describe a pregame lobby**. They appear to associate the visible
Finding Next Match overlay with the state of the watched match. This is an
interpretation of the errors, not an established internal model mechanism.
Viewpoint ownership is improving; scene-state narration is not yet reliable.

Twelve new responses used 195,254 input and 2,663 output tokens, estimated
$0.105616 with the prior conservative assumptions. No extra calls followed
inspection. No prompt change or default promotion occurred. Original user labels,
baseline predictions and training data remain intact. Recording stays paused.

## Engineering outcome and next step

The comparison runner now verifies additional source-chunk hashes as well as the
primary video and exact input images. Its report retains every prediction and
separates contract checks from visual accuracy. All twelve responses pass internal
contract checks, illustrating that this alone does not detect incorrect narration.

The promising direction is separate evidence windows for separate questions:
short shot-centered evidence for kill attribution, longer context for build
mechanics. Before shipping that approach, automatic localization must be evaluated
on untouched fights and references must distinguish observable editing from
ownership/control. Spectator labeling and active-match state should be validated
separately. Do not retrain the preference ranker on unverified mechanics or promote
this experimental prompt as a general fix.

Local artifacts: `.local-data/evaluation-context-20260908/` contains `plan.json`,
frames, results, separate reference snapshots, `comparison.json`, and `comparison.md`.
Rebuild the report without inference:

```powershell
python -m pipeline_v2.evaluation_attribution report .local-data/evaluation-context-20260908
```
