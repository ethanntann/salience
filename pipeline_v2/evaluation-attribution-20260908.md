# Attribution and frame-density comparison

The expanded attribution prompt is **experimental, not the new default**. It
corrected spectator ownership in the known spectator example, but did not
reliably fix the reviewed weapon errors. Increasing frame density alone was
insufficient. Piece-control recognition remains unvalidated.

## Frozen design

`evaluation_attribution.py` compares the unchanged v3.3 baseline with opt-in
`v3.4-attribution-dev`. The variant asks for viewpoint, event actor, decisive-hit
weapon, cited frame evidence, and observable build/edit sequences. It adds editing,
edit-reset and piece-control vocabulary without assuming every build fight contains
those mechanics. Legacy descriptions default to unknown ownership, not self.

The same seven input frames were used for all ten reviewed action/control cases
and two background cases (sample-00, sample-07). Four known failures—action-00,
action-02, action-04, sweep-22—also received 19-frame baseline and attribution runs
over exactly the same four-second bounds. Spacing changes from approximately
667 ms to 222 ms. Model, resolution, temperature, and output limit stay fixed.
The new fields necessarily increase response length; this is a prompt-package
comparison, not an isolation of one sentence of instruction.

32 responses: 12 reused originals and 20 new calls. New usage was 318,148 input
and 4,694 output tokens; estimated cost $0.173156 using the earlier pilot's
conservative pricing assumptions. No repeated calls were made after inspecting
these results. Inputs and prompt fingerprints were frozen; user references were
snapshotted separately and never sent as model instructions.

## Observed changes

| Known case | User reference | Baseline, 7 frames | Attribution, 7 frames | Baseline, 19 frames | Attribution, 19 frames |
| --- | --- | --- | --- | --- | --- |
| action-00 | Drum Gun kill | Tactical Pistol | Pistol | Tactical Pistol | Assault Rifle |
| action-02 | Shotgun one-pump | Wrecker Revolver | Wrecker Revolver | Wrecker Revolver | Wrecker Revolver |
| action-04 | Shotgun kill and piece control | Drum Gun | Drum Gun, editing | Drum Gun | Pump Shotgun, editing |
| sweep-22 | Spectated player | Ownership omitted | Explicitly spectated | Ownership omitted; pickaxe kill claimed | Explicitly spectated; pistol kill claimed |

Weapon entries describe the attributed kill weapon, drawn from the baseline's
summary/weapon field or the variant's explicit decisive-hit field. Agreement on
"shotgun" in action-04 with the combined change is a family-level improvement;
it does not validate the exact Pump Shotgun name or the narrated weapon sequence.
None of the variants reports piece control in that case. New editing claims
require checking their specific sequence, not merely counting the added tag.

Spectator ownership improved on the same frames, so extra density was unnecessary
for that correction. This is one previously identified spectator case, not a
measured general spectator classifier. More frames did not reliably improve
weapon recognition and encouraged extra unsupported detail: the baseline's
19-frame spectator summary claims a pickaxe kill, whereas its original answer
left the weapon unknown. The seven-frame attribution variant leaves that weapon
unknown and lowers confidence to medium; the dense variant claims a pistol at
high confidence without a verified reference for that weapon.

The seven-frame variant retains the original coarse outcome on all twelve cases,
but this is not full description accuracy. It continues to describe transparent
builds as glass despite the instruction. Its death example also fills the
decisive-hit weapon field with the opponent's weapon, contrary to this experiment's
non-elimination rule. The report flags that contract violation separately, without
rewriting the raw prediction. Mechanically valid citations are not proof that the
cited images support a claim; action-00 still gives confident, incorrect attribution
with numbered evidence.

## Decision and next boundary

Keep v3.3 as the default while retaining this opt-in experiment and its results.
Do not increase frame count globally or retrain ranking on these descriptions.
The next useful step is a focused check of decisive-hit weapon evidence, with
explicit unknown outcomes when the visual evidence cannot establish the answer.
Isolate and validate spectator handling independently before promoting it. Build
mechanics need references that identify the edit/placement sequence and ownership,
rather than using generic outcome agreement as a substitute.

These clips are development cases from one reviewed session. No held-out accuracy,
end-to-end recall, or capture performance claim follows. Recording remains paused.
No user ratings, original predictions, or preference-training records were changed.

## Reproduce and inspect

The local comparison lives in `.local-data/evaluation-attribution-20260907/`:
`plan.json`, exact frame paths/hashes, `user-reference-snapshot.json`, all results,
`comparison.json`, and a readable `comparison.md`. The directory date identifies
the recording; this report uses the current September 8 date.

```powershell
python -m pipeline_v2.evaluation_attribution report .local-data/evaluation-attribution-20260907
```

For a new comparison directory, `prepare SOURCE OUT` freezes the design and
extracts images without inference. `run OUT --max-calls N` explicitly bounds new
calls and resumes completed results. It refuses changed source media, frames,
model, prompt or resolution. Reports separate missing results and contract issues
from visual accuracy, which requires annotation.
