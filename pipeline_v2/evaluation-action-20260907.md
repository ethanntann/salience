# Targeted action follow-up: September 7 recording

The first pilot's 12 user-reviewed windows were all skips: 11 descriptions rated
accurate, one minor error. This follow-up adds actual action to diagnose description
and ranking separately. It does not change the recorder's failed acceptance status.

## Review

Open **http://127.0.0.1:8779/** while the local server is running. Ten targeted
moments have been reviewed; the 12 completed background reviews have been copied
with source-study provenance. The original pilot and its labels remain intact.
Watch, save Keep/Skip/Unsure and notes, then reveal v2 and rate its description.
Playback includes context; Replay model window isolates the four seconds supplied
to v2. The seven actual model frames remain inspectable. Evaluation labels never
enter the preference database or ranker training.

Restart from the repository root if necessary:

```powershell
python -m pipeline_v2.evaluation_review .local-data/evaluation-action-20260907 --port 8779
```

## Selection and provenance

Manual discovery sampled the midpoint of all 194 saved chunks, then inspected
one frame per second in 11 promising chunks. Seven new windows were frozen with
provisional assistant references before their predictions were requested. This
is not exhaustive viewing or an automatic event detector.

| ID | Source within session | Model seconds | Provisional visual reference | V2 outcome |
| --- | --- | --- | --- | --- |
| action-00 | run-004/part-00048 | 11.5–15.5 | Own opponent elimination, dissolution and banner | elimination |
| action-01 | run-004/part-00011 | 12–16 | Build fight, no confirmed elimination in window | no_event |
| action-02 | run-004/part-00065 | 12.5–16.5 | Glide into close shot, 227 damage and elimination | elimination |
| action-03 | run-005/part-00018 | 0–4 | Shots through builds, no confirmed elimination | no_event |
| action-04 | run-006/part-00010 | 3.5–7.5 | Close fight, opponent elimination, own low-health survival | elimination |
| action-05 | run-006/part-00017 | 14–18 | Rooftop shots at distant target, no confirmed elimination | no_event |
| action-06 | run-006/part-00080 | 25.9–29.9 | Final opponent engagement and Victory Royale | elimination |

Three controls reuse the first pilot's exact inputs/results: sweep-03 (NPC kill),
sweep-05 (own death), and sweep-22 (spectated player's kill). Their predictions
were already known, so they are not new blind evidence. The original 12 uniform
windows and user decisions are also reused, without new inference charges.

Session: `20260907-142124-24be3d`. Local manifests, frames, predictions, ranking,
assistant references, and user judgments live separately in
`.local-data/evaluation-action-20260907/`. Discovery contact sheets and their
source mapping remain in `.local-data/fight-discovery/`.

## What the results establish

The unchanged v3.3 prompt and configured Qwen3.7-Plus VLM, using seven 1440-pixel
frames per window, recognized the coarse outcome of the four selected opponent
eliminations and did not assert an elimination in the three combat controls.
It also mentioned Victory Royale in the final example. These are provisional
assistant comparisons made before user review. Outcome
agreement does not establish correct weapons, actors, health, or every sentence.

## Completed user review and error analysis

The user completed all ten targeted moments: three keeps and seven skips;
seven descriptions rated accurate, two minor errors, one major error. The
keeps are action-02 (landing into a shotgun kill), action-04 (piece control and
shotgun kill), and action-06 (final elimination and victory). A kill alone is
not the preference target: action-00 and the NPC elimination were skipped.

The user's follow-up identifies detail errors that the coarse rating totals
understate. Preserve the submitted ratings exactly: action-04 was rated accurate,
but its note says shotgun and piece control while v2 says Drum Gun and generic
build fighting. Record that discrepancy separately; do not silently relabel it.

| Failure | Evidence | Next experiment |
| --- | --- | --- |
| Kill weapon confused with later held weapon | action-00: user says Drum Gun; v2 says Tactical Pistol. Model frame 3 shows firing with the Drum Gun selected; frame 6 shows Tactical Pistol selected after the elimination. | Require shot-time weapon evidence separately from the weapon held afterward; allow unknown when the decisive transition is missing. |
| Shotgun kill misidentified | action-02: user says one-pump; v2 says Wrecker Revolver. action-04: user says shotgun; v2 says Drum Gun. | Inspect frames immediately before/at damage and first elimination indication; do not assume the weapon in a banner frame caused the kill. |
| Build mechanics flattened | User identifies piece control in action-04 and reports missed editing mechanics; v2 gives generic build-fight descriptions. | Describe observed placement/edit/reset sequences with evidence; distinguish an edit from proof of ownership/control over the opponent's space. |
| Spectator action attributed without ownership | sweep-22: user marks major error and spectator footage; v2 reports an elimination without saying it belongs to a spectated player. | Explicit viewpoint and event-actor fields, with unknown available; do not count a spectator kill as the recording player's highlight. |

The input is seven stills over four seconds, approximately 667 ms apart. Fast
shots, swaps, edits and resets can occur between those observations. The
action-00 frames already contain evidence of a weapon change, so temporal
reasoning is also implicated: more frames are not automatically the whole fix.
The current prompt requests a single weapon name without defining its temporal
role, does not request viewpoint/actor ownership, and its fixed mechanics
vocabulary excludes editing and piece control. The optional old tracking prompt
is not active on capture windows. These are implementation observations, not
proof that a prompt change alone will resolve the errors.

Use these reviewed clips as development/regression cases from now on. Compare
the unchanged baseline against explicit attribution rules on the same frames,
then test denser temporal evidence separately to distinguish prompt effects
from sampling effects. Keep model, frame width and other settings fixed. Score
kill-weapon correctness/abstention, actor ownership, and supported mechanics;
do not count vague descriptions as fully correct merely because the outcome is
right. Validate any chosen change on separately annotated unseen fights.

No prompt, model, training data, or saved user judgments were changed after this
feedback. No additional inference was run. Recording remains paused.

There is a direct sampling miss in footage the user reviewed: sample-03 covers
18.396–22.396 seconds of run-004/part-00048, after the elimination around 14 seconds.
Its travel description can be accurate while the session's action is missed.

The same historical-only TF-IDF/LSA logistic baseline trained on 334 examples
(105 keeps) ranks user-skipped sample-07 (gliding onto builds, score 0.762) above
all four newly selected elimination windows. For the paired chunk, travel
sample-03 scores 0.191 while its elimination action-00 scores -0.813. These are
raw ranking scores, not probabilities. No new labels or evaluation descriptions
were used to fit the encoder or ranker. This is not a reproduction of the
historical semantic-embedding result and does not prove every kill is wanted.

Seven new requests used 65,695 input and 830 output tokens, estimated $0.03534
with the first pilot's conservative price assumptions; 15 responses were reused.
No prompt tuning or additional model calls followed inspection of these outputs.

## Decision

Continue testing the VLM description approach. Fix temporal coverage and validate
ranking against the user's action preferences before promoting this to automatic
highlight selection. A ranker must be allowed to select nothing from downtime;
always taking the top few will manufacture unwanted clips. Choose thresholds on
development sessions, then measure missed wanted moments and unwanted saves per
hour on fully reviewed, separate sessions.

This manually enriched set cannot estimate natural action prevalence, session
recall, or deployment precision. The original uniform sample remains evidence
about background description, and the original continuous pilot remains separate.
The metrics code explicitly leaves session highlight metrics unset here, even
if a timeline-complete flag is supplied.

Recording stays paused. Audio inference remains experimental; usable recorded
audio and acceptable game performance still need separate capture validation.
No new capture or gameplay benchmark was run for this study.

Validation: 64 relevant tests pass; Ruff passes. Browser review displays 10
targeted moments, 12 retained background judgments, hidden predictions, and
working video playback. Tests cover label preservation and prevent a curated
study from reporting invented session recall or charging reused responses again.
