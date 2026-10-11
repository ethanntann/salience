> Historical snapshot, superseded by the root README and pipeline_v2/SCOPE.md.
> Relative links below refer to the original document location.

# pipeline_v2 — M1

VLM-first clip understanding. See [SCOPE.md](SCOPE.md) for the plan; this file
covers running it and what M1 measured.

## Run it

Free — no API calls, verifies event localization and OCR facts:

```bash
python pipeline_v2/cli.py localize sample-clips --limit 6
```

One paid call, with token counts printed:

```bash
python pipeline_v2/cli.py process "sample-clips/<clip>.mp4" --max-calls 1
python pipeline_v2/cli.py spend
```

`--max-calls` defaults to **1**. Cached clips are free and do not count against
it, so re-running a processed clip costs nothing.

Configuration comes from the repo-root `.env` (`FIREWORKS_API_KEY`,
`FIREWORKS_MODEL`). Set `V2_INPUT_PRICE_PER_MTOK` and
`V2_OUTPUT_PRICE_PER_MTOK` from the provider's current pricing page to turn the
token counts into dollar figures — they are configuration, not knowledge, so
nothing here guesses them.

## What M1 measured

**Cost per clip: ~4,175 input + ~120 output tokens.** Seven 960px frames plus
the prompt. That is the real number to multiply by current pricing; against
typical VLM rates the $12 budget buys thousands of clips, so cost is not the
binding constraint the plan assumed.

**Event localization works, after two corrections that only real clips revealed:**

1. *The kill feed is global.* The upper-left feed reports **every** player's
   kills — the first clip tested localized onto a stranger's shotgun knock.
   Only the centre-screen personal hit feed (`banner` region) is the recording
   player's own event. Localizing on the kill feed points the window at somebody
   else's fight.
2. *Clips contain several events.* A 22s clip held two knocks and two
   eliminations. "Find the finish" is underspecified; the question is which
   finish caused the save.

The rule that follows, and holds across every clip tested: among marker groups
matching the event type in the filename, **the last group is the trigger**.
Measured placement is consistently ~76% through the clip, because the
auto-clipper keeps mostly preceding footage. Marker onset is the tightest
available upper bound on the event, so the frame window is biased backwards
(2.5s before, 1.5s after).

The count of *other* events in the clip is passed to the VLM as a fact. It is
nearly free for OCR to produce and stops the model describing a multi-kill clip
as a single kill.

**The reasoning-model trap.** The configured model is a reasoning model. The
first call spent its entire 600-token output budget thinking and returned no
JSON — full price, zero result. `reasoning_effort: "none"` fixed it and cut
output to ~116 tokens. Worth checking on any model swap.

**`hud_conflict` earned its place on the first clip.** OCR read the weapon as
`HUNTING RIFLE`; the model looked at the frames, saw a pistol, and set
`hud_conflict: "weapon"`. Under the rejected `"do not contradict the HUD"`
design that would have been a silent wrong label with no signal that anything
had gone wrong.

## How this differs from v1

| | v1 | v2 |
| --- | --- | --- |
| Finding the event | Locator CNN over ~20-60 timeline frames | OCR on the personal hit feed, plus the filename |
| Understanding it | ~20 classification heads over a 7-frame window | One VLM call returning JSON + a sentence |
| Vocabulary | 28 fixed label keys | Open — the sentence can say anything |
| Multi-event handling | `highlights.py` reducer: dedupe, contradiction rules, primary hierarchy | Localize the triggering event; count the rest as context |
| OCR's role | Post-hoc corroboration, reconciled against model output | Prompt input, decided before the model runs |
| Taste input | 33 numbers: 5 features + 28 label flags | Sentence embedding + structured fields |
| Where it runs | Fully local, CPU | Frames/OCR local, one cloud call |
| Cost | Free per clip | ~4,300 tokens per clip |
| Lines to reason about | Locator + heads + calibration + reducer | Localizer + prompt + schema |

The reduction is real but it is a *trade*, not a free win: v2 buys open
vocabulary and far less code by taking on a per-clip cost and a network
dependency. v1 remains the only version that runs with no API key.

What v2 keeps from v1 is the part that was hardest to get right: frame
extraction, region-scoped OCR, and the weapon ontology, all imported rather
than rewritten.

## Weapon attribution: the same trap, re-derived

The first version of `weapon_from_reads` took the highest-confidence weapon read
anywhere in the clip. Measured on one 22s clip, the player's HUD went:

```text
t= 0.9  HUNTING RIFLE          <- what "most confident anywhere" returned
t= 3.5  FOCUS SHOTGUN
t= 5.3  MYTHIC HUNTING RIFLE
t= 8.8  FOCUS SHOTGUN
t=15.0  FOCUS SHOTGUN          <- the weapon actually held at the kill
t=17.7  (kill)
t=19.4  OVERDRIVE GRENADE      <- what "most recent read" would return
```

Sixteen seconds before the shot, and a post-kill swap two seconds after it. Both
naive selectors are wrong, in opposite directions.

The rule now: only frames at or before the event count, and among those the
**latest** wins. This is the same conclusion v1 reached across two commits of
weapon-attribution work; it is re-derived here rather than inherited, because v2
reads different frames.

The related fix: the weapon is read from the **dense window**, not the coarse
scan. Scan frames are ~0.9s apart, which is coarser than a weapon swap, so any
scan-based weapon answer is a guess dressed as a fact. Window frames sit ~0.65s
apart around the shot and are the only honest place to ask.

Visible effect: one clip that previously reported a confident `HUNTING RIFLE`
from 16s earlier now reports `FOCUS SHOTGUN`, and another that previously
reported the post-kill grenade now reports `unknown`. The second is the more
important improvement — abstaining beats a confident wrong fact, especially when
that fact goes into the prompt.

## Layout

| File | Role |
| --- | --- |
| `config.py` | Paths, credentials, prompt version, budget ceiling |
| `frames.py` | Coarse localization scan, then the dense window (imports v1 extraction) |
| `facts.py` | Filename event, banner localization, weapon/victory OCR (imports v1 `ocr_regions`) |
| `prompt.py` | Prompt construction; OCR enters as prior, never as law |
| `schema.py` | Pydantic response contract |
| `vlm.py` | Fireworks client, retries, usage accounting |
| `cache.py` | Content-hash cache and spend ledger |
| `cli.py` | `process`, `localize`, `spend` |

Nothing under `backend/` is modified; v1 modules are imported read-only.

## M2 findings (first 10 clips)

**The summaries work.** They are specific, spatial, and name weapons the label
set never had: *"uses a Peely Surfer to glide from a cliff to a building, then
eliminates an opponent on the ground below with a headshot from an Uncommon
Hunting Rifle."* No arrangement of 28 binary flags produces that. This is the
premise of v2 and it holds.

The model also narrated the weapon-swap case correctly and unprompted:
*"aims a Hunting Rifle at a distant opponent, then switches to a Lancehead
Pistol to secure a headshot elimination."*

**Two fields are dead weight.** `player_health_state` returned `healthy` on
10/10 clips and `opponent` returned `human` on 10/10. A field with no variance
carries no ranking signal. Either the model cannot read the health bar from
these frames, or the value is genuinely constant — and health matters for taste
("clutch at low HP" is exactly the kind of thing a user cares about), so the
next step is to read it deterministically from the HUD via OCR rather than
asking the model.

**`outcome` is nearly redundant.** It matched the filename event on 10/10.
The filename is free and comes from the game engine; the model adds nothing
here. Keep it only as a cross-check.

**Weapon categories are unstable, weapon names are not.** The same gun came back
as `other` in one clip and `automatic` in the next ("Chaos Exploder Rifle" /
"Chaos Explorer Rifle"). But the *name* in the summary was right both times. The
fix is to have the model report the weapon name as free text and let the existing
`weapon_ontology` map it to a category — deterministic, versioned, and already
written.

**`mechanics` vocabulary drifts.** `high_ground` and `high_ground_advantage`
both appeared for the same concept. Harmless for the embedding, which sees the
sentence anyway, but useless for filtering. Either constrain to a fixed list or
drop the field and let the embedding carry it.

**A product finding, not a model finding: the auto-clipper saves near-duplicate
clips.** `04.44.58 Down` and `04.45.03 Elimination` are five seconds apart and
describe the same engagement — the knock and then the finish. Same for
`04.49.49` / `04.49.59`. Roughly 40% of this batch is duplicate coverage of the
same fight. The inbox would show both. Filenames carry timestamps, so
same-engagement grouping is cheap and deterministic, and it wants deciding before
the taste model trains on what are effectively duplicate rows.

**Cost, measured across 12 calls:** ~4,150 input and ~115 output tokens per clip,
tightly clustered. Multiply by current pricing; the $12 budget is not close to
being the constraint.

## M2 round two: revised schema (prompt v2.2)

Changes made in response to the findings above, then re-run on six fresh
engagements:

- `weapon` enum -> free-text `weapon_name`, category derived via
  `weapon_ontology`
- `opponent` dropped (zero variance)
- `mechanics` constrained to a fixed vocabulary
- the model told **where** the health bars are, rather than just asked

Results:

- **Free-text weapon works.** The ontology mapped 5 of 6 names correctly and the
  names themselves were right 6 of 6. Deriving the category beats asking for it.
- **Health now varies** (`damaged` appeared), where before it was `healthy` on
  10/10. Telling the model where to look was the whole fix.
- **Mechanics are clean** and queryable now the vocabulary is fixed.
- **Dedupe removed 4 of 10 clips** on this batch -- 40% of the calls not made.

Two new findings, both about weapon attribution:

**The ontology is a maintenance surface.** `Mythic 9mm Baba Yaga` mapped to
`unknown` because the term table predates that weapon. Every Fortnite season
adds guns. The table is versioned data, which is the right design, but it needs
an owner and an "unknown rate" to watch. (Not fixed here: `weapon_ontology.py`
belongs to v1 and v2 imports it read-only.)

**OCR is the *weaker* weapon source, and the kill feed is the stronger one.**
`hud_conflict` fired on 2 of 6 clips, and on inspection the model looks right
both times: OCR read `EXTENDING FOCUS SHOTGUN` where the kill feed said
`9mm Baba Yaga`, and read `PICKAXE` where the finish was clearly a pistol.

That inverts an assumption in SCOPE.md. The weapon HUD shows the *selected slot*,
which drifts around a kill; the kill feed states the weapon that **actually got
the kill**. Earlier the kill feed was ruled out as a *localizer* because it is
global -- correct, and unchanged. But for weapon attribution the fix is to match
kill-feed rows against the player's own name (already legible in these frames)
and prefer that row over the HUD read.

Worth noting this is the same lesson as v1's whole reconciliation workstream,
arriving from the other direction: the weapon question is not "what is held" but
"what did the damage".

## Kill-feed attribution: a hypothesis that half failed

The previous section concluded the kill feed should replace the weapon HUD,
because it names what did the damage rather than what was held. Implemented and
measured, that turned out to be **right about the source and wrong about the
selection rule**.

Selecting the feed row by timing alone -- the row absent before the event and
present after it -- fired on 2 of 10 clips and got one of them wrong. On
`11.39.13.114` the HUD said `STRIKER PUMP SHOTGUN`, the model independently
described a pump shotgun kill, and the feed handed back `SMG`: a stranger's kill
landing in the same instant. The global feed is exactly as ambiguous for
attribution as it was for localization; onset timing does not disambiguate it.

So `weapon_for_event` now **requires a player name** and abstains without one.
With `V2_PLAYER_NAME` unset the HUD read wins and behaviour returns to the
previous, better state; set it, and feed rows are filtered to the killer
position by squashed-name match before any of them can win.

Two things worth taking from this:

- The abstention is the fix, not a workaround. A wrong weapon entering the
  prompt as a stated fact is worse than no fact, because the model treats it as
  context to prefer.
- It cost nothing to find out. Verification ran through `localize`, which makes
  no API calls, so a wrong hypothesis was measured against ten real clips for
  free. Keeping a no-spend path over the same code is worth more than it looks.

Open: infer the player name automatically. It is almost certainly recoverable --
the same name recurs in the personal banner region across every clip in a
library, and it never changes between clips of one player. That would turn the
feed from opt-in to default.

## M3 — storage, embeddings, and the sanity check

```bash
python pipeline_v2/cli.py index sample-clips --max-calls 10
python pipeline_v2/cli.py index demo-video   --max-calls 10
python pipeline_v2/cli.py similar --top 2          # free, no API calls
```

`index` describes clips cache-first, stores them in `.data/v2.sqlite`, and
rebuilds every embedding. Embeddings are rebuilt wholesale rather than appended
because the TF-IDF baseline is corpus-dependent — adding one clip changes every
other vector, and mixing generations would silently corrupt similarity.

**The encoder is deliberately the weakest reasonable one.** TF-IDF matches shared
words, so "pistol" and "handgun" are unrelated to it. That is the point: it is
free, needs no download, and any neural encoder has to beat something this naive
before its cost is justified. `sentence-transformers` is used automatically if
installed.

### The check that decides whether v2 can work

If nearest neighbours by summary are not intuitively similar, the summaries do
not separate clips, and no taste model built on them will either. Run on 14
indexed clips, they are:

| Cluster | Score | What grouped |
| --- | --- | --- |
| Finishing a downed opponent up close | 0.41, 0.24, 0.21 | Three clips, each other's top neighbours |
| Long-range Lancehead Pistol eliminations | 0.31 | Mutual top-1 |
| Hunting rifle at range | 0.24 | Mutual top-1 |

Every clip's top neighbour is a defensible match, and the groupings are
*semantic* rather than filename- or weapon-only. This is the M3 gate, and it
passes.

Two honest caveats before treating this as settled:

- Absolute scores are low (0.1–0.4) because TF-IDF over 14 short documents is a
  sparse, high-variance estimate. The *ordering* is what matters here, not the
  magnitude.
- Part of what clusters is weapon name — which the structured fields already
  provide. The interesting question for M4 is whether the embedding adds signal
  *beyond* the fields, and the answer only arrives with pairwise accuracy
  measured both ways. Phrases like "drops from a height into a building" and
  "uses a Peely Surfer to launch vertically from a cliff" are the ones no label
  set contains, and they are the reason to expect it does.

Also worth watching: summaries are formulaic ("securing an elimination and
wiping the enemy team" recurs constantly). IDF weighting discounts boilerplate,
but a future prompt revision should push for more variation in what is
described, not just in how it is phrased.

## M4 preference data — confirmed available

v1's `.local-data/salience.db` holds **573 feedback events across 563 clips, and
every referenced file still exists** on disk under `C:\Users\Ethan\Videos\
NVIDIAFortnite`. M4 does not need a fresh rating session.

```text
supervised_skip  451
supervised_keep  102
keep               9
favorite           6
skip               5
```

Two things to weigh before training on it:

- **It is dominated by supervised imports** (553 of 573), not organic review.
  v1 already discounts those with a 0.45 source multiplier for good reason: an
  imported judgement is weaker evidence of taste than a deliberate keep. The
  20 organic events are the highest-quality signal and by far the scarcest.
- **Pairs must respect the engagement grouping.** Two clips of one fight are not
  an independent preference pair, and roughly a third of a library is duplicate
  coverage. Build pairs across engagements, never within one.

Describing 563 clips costs ~2.4M input tokens. That is affordable against the
budget but is the single largest planned spend, so it should follow a smaller
run — a few dozen clips spanning keeps and skips — that confirms pairwise
accuracy is measurable at all before committing the rest.

## Zero Build: the library has no build fights

Frames sampled from a 2026 session and a 2025 session both show **Zero Build** --
there is no wood/brick/metal counter anywhere in the HUD, because building is
not in the mode. The numbers in the bottom-right are gold and per-slot ammo.

Consequences:

- `build_fight` is **removed** from the mechanics vocabulary. Offering a tag for
  an event that cannot occur only invites false positives.
- The planned material-counter detector is dropped. There is nothing to count.
- v1's `build_fight` context head reports 95.1% accuracy on 81 examples, and
  that number now looks like what it probably is: a near-constant label scored
  by answering "no". `clutch` and `competitive_context` report 100% on the same
  split. `artifact_meta.json` records no precision, recall, or positive support
  for context heads, so none of them can demonstrate positive recognition.

### Player name: configure it, do not infer it

An earlier note here guessed the recording player's name from on-screen text and
got it wrong -- `Nerual102` is the *duo*, not the recording player, and the
library also spans more than one account. Names appear on nameplates, in the
squad list, and in the kill feed, and none of those positions reliably marks
"you".

So `V2_PLAYER_NAME` stays configuration. A desktop app knows the signed-in
account; a CLI can ask once. Inferring an identity from pixels, when the answer
is something the user can simply state, is effort spent to become less reliable.

Kill-feed weapon attribution already abstains without this value, so nothing
breaks while it is unset -- it just falls back to the HUD read.

### Mode-specific vocabulary

Zero Build is this library's mode, not every user's. The correct handling is not
to hardcode either way but to detect per clip: **the material counter exists in
Build mode and does not exist in Zero Build**, so its presence in the HUD is
itself the mode signal, readable by the same OCR pass.

That makes `build_fight` a *conditional* tag -- offered when the counter is
present, withheld when it is not -- rather than a permanent removal. Left
unimplemented deliberately: there is no Build-mode footage here to validate it
against, and a detector written against zero examples is a guess with tests.

The general shape, already true of `FORTNITE_REGIONS` and the weapon ontology:
game- and mode-specific knowledge belongs in versioned data tables, not in
pipeline code.

## Deterministic health (prompt v2.3)

Health is now read by OCR and stated to the model as fact, rather than asked for.
It took two attempts, and the failure is instructive.

**First attempt, wrong.** One generous box (`x 0-0.30, y 0.85-0.99`) plus "take
the lowest legible number" returned 2, 3, and 25 on clips the model had
independently described as full health. The box swept in the teammate row, and
the `3` was coming out of the *name* `laurenlover3`. Taking the minimum then
guaranteed the worst misread won -- the same failure shape as picking the most
confident weapon read anywhere in a clip.

**Second attempt, measured.** Cropping and enlarging the corner showed exactly
where the numerals sit at 960x540: two tight boxes at `x 0.195-0.28`, shield at
`y 0.872`, health at `y 0.910`, both clear of the teammate row above. Parsing now
requires the *entire* string to be digits rather than fishing for digits inside
it, and range-checks against Fortnite's cap of 100.

Results across two sessions: `100, 100, 100, 100, 90` and `94, 100, 80, 100` --
plausible, varying, and matching the model's independent descriptions.

The general point, which cost one wrong implementation to relearn: **a region
guessed from intuition is not a measurement.** Cropping one frame and looking at
it took a minute and produced coordinates that worked first time.

Because this value is validated and range-checked it is now passed into the
prompt as a stated fact, replacing the v2.2 approach of telling the model where
to look and hoping.

## M4 — first pairwise accuracy numbers

```bash
python pipeline_v2/cli.py index --from-feedback .local-data/salience.db --limit 30 --max-calls 30
python pipeline_v2/cli.py taste --folds 3        # free, no API calls
```

24 labelled clips indexed (14 liked, 10 not), drawn organic-first from v1's
feedback. Cross-validated by clip, pairs formed only across engagements:

| folds | pairs | fields | embedding | both |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 68 | 69.1% | 36.8% | 67.6% |
| 3 | 42 | 78.6% | 64.3% | 76.2% |
| 4 | 34 | 70.6% | 47.1% | 70.6% |
| 5 | 17 | 52.9% | 29.4% | 47.1% |

Chance is 50%.

**Read this as provisional.** 24 clips is far too few, the pairs are not
independent (the same clips recur across many pairs), and `fields` swings from
53% to 79% depending only on how the folds are cut. No confidence interval
around these numbers excludes chance.

With that said, one pattern holds in **every** setting and is worth acting on:

- `fields` beats `embedding` every time, often by 20+ points.
- `both` tracks `fields` almost exactly — the embedding adds nothing and
  slightly dilutes.
- `embedding` alone lands below chance twice.

### What this does and does not mean

It does **not** mean the VLM is not earning its cost. The `fields` vector is
built from the VLM's own structured output -- outcome, weapon, health,
mechanics -- plus OCR facts. So this is not "VLM vs no VLM". It is **"the
VLM's structured output vs its free text, as encoded by TF-IDF"**, and the
structured half is winning.

Two things must change before concluding anything about embeddings:

1. **TF-IDF is a bad encoder here.** 527 dimensions over 24 documents is wildly
   overparameterised, and word-overlap matching cannot see that "pistol" and
   "handgun" are related. It was chosen as a floor to beat, and the floor may
   simply be too low to be informative. A real sentence encoder is the honest
   test.
2. **There is no v1 baseline yet.** The comparison that decides whether to keep
   building v2 is against v1's ranking on the same pairs, and that has not been
   run.

### Rerun at 111 clips with a real encoder — the result reverses

Same code path, three things changed: 24 → 111 labelled clips,
`all-MiniLM-L6-v2` (384 dims) instead of TF-IDF, and per-block unit
normalisation with cosine scoring.

| folds | pairs | fields | embedding | both |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 985 | 67.8% | **77.2%** | 74.5% |
| 4 | 737 | 69.7% | **80.5%** | 75.7% |
| 5 | 577 | 71.1% | **78.3%** | 74.4% |

**The embedding now beats the structured fields by 8-10 points, in every
setting.** It is also stable: `fields` sits at ~70% and `embedding` at ~78%
across fold counts, where the 24-clip run swung between 53% and 79% on fold
choice alone. 577-985 pairs instead of 17-68.

So the earlier conclusion — "the VLM's structured output carries the signal, its
prose does not" — **was an artifact of a bad encoder and too little data**, and
is withdrawn. The prose carries more.

Three things worth recording:

- **~78% pairwise accuracy** means the ranker puts a kept clip above a skipped
  one about four times in five. Against 50% chance, and against ~70% for the
  v1-shaped field representation.
- **`both` is worse than `embedding` alone** (~75%). Concatenating equal-weighted
  blocks dilutes the stronger signal with the weaker one. If both are kept, they
  need learned weights rather than an even split.
- **Switching encoders also removed a subtle leak.** TF-IDF is fitted on the
  whole corpus, so held-out documents influenced their own vectors -- mild
  transductive leakage that flattered the old embedding numbers. MiniLM encodes
  each sentence independently, so the new numbers have none.

### Head-to-head against v1, and the organic-label check

Both run on the same clips and the same pairs. `v1 score` is v1's shipped
`final_score` -- a finished ranking, so no folds and no training.

**All 111 labelled clips:**

| ranker | pairwise acc | pairs |
| --- | ---: | ---: |
| v2 embedding | **80.5%** | 737 |
| both | 75.7% | 737 |
| v1 shipped score | 74.2% | 2989 |
| v2 fields | 69.7% | 737 |
| chance | 50.0% | |

v2 beats v1 by ~6 points -- and the gap is if anything *understated*, because
v1's `final_score` is 25% base plus 75% personal, and that personal model was
fitted on these exact feedback events. v1 is being scored on its own training
data, uncross-validated, and still loses to a cross-validated v2.

**Organic labels only (14 clips, 11 liked / 3 not, 16 pairs):**

| ranker | pairwise acc |
| --- | ---: |
| v2 fields | 81.2% |
| both | 81.2% |
| v1 shipped score | 39.4% |
| **v2 embedding** | **6.2%** |

The embedding does not merely weaken on organic labels -- it **inverts**, and v1
also drops below chance. At 16 pairs from 14 clips with 3 negatives, this is
statistically nearly meaningless: one badly-oriented fold produces exactly this.
It cannot be dismissed either.

### Are the deterministic checks earning their place?

Every fact in this pipeline was added on reasoning, never measured. Coverage
across the 125 indexed clips:

| check | how often it fires | verdict |
| --- | ---: | --- |
| banner localizer | 119/125 (95%) | **largely redundant** -- see below |
| event census (>1 event) | 68% | high coverage, value unmeasured |
| weapon known | 73% (all from HUD) | value unmeasured |
| kill-feed weapon | **0%** | inert -- needs `V2_PLAYER_NAME`, which is unset |
| victory banner | **1%** | too rare to validate here |

**The localizer is mostly reproducing a one-line heuristic.** Compared against
naive `0.75 x duration`:

```text
median difference  +0.20s
77% of clips within +/-1.0s
94% of clips within +/-2.0s
```

The frame window spans 2.5s before to 1.5s after, so a +/-1s error still
contains the event. On roughly three quarters of clips, OCR banner detection,
marker grouping, and last-run selection produce the same window a single
multiplication would have. It earns its keep only on the tail -- and it produces
the event census as a byproduct, which the heuristic cannot.

That is a real candidate for deletion, and it is also the most intricate code in
`facts.py`.

**A caution against deleting on frequency alone.** The victory banner fires on
1% of clips, which looks like dead weight -- but winning a match is plausibly a
near-certain keep, so rarity is not the same as uselessness. Judging a rare
signal by how often it appears is the same mistake as judging a rare label by
accuracy. It stays, unvalidated and marked as such.

**The kill-feed weapon path is genuinely inert**, contributing to zero clips
because `V2_PLAYER_NAME` is unset. It is not wrong, it is simply switched off.

### How to answer this properly

Ablation: remove one fact from the prompt, re-describe the corpus, re-measure
pairwise accuracy. That is ~111 calls per variant, roughly 480k tokens -- easily
affordable.

**But not yet.** Ablating against labels that may not encode the user's taste
would produce confident numbers about the wrong objective, and every variant
would need re-running once real labels exist. Organic labels first, ablations
second.

### Correction: the "bulk imports" are hand labels

This document previously discounted the 553 `supervised_*` events as a bulk
import that might encode some mechanical rule rather than the user's taste.
Checking `backend/salience_api/training/labels.py` shows that is wrong.

`import_labeled_jsonl` reads a JSONL file of `{"path": ..., "label": 1 | -1}`
and records `supervised_keep` / `supervised_skip`. Nothing derives those labels
from teacher output, scores, or any heuristic -- the file is a list of
human decisions. The v1 database records the source as
`supervised:clips.jsonl`, 102 keeps and 451 skips.

So they are hand labels that arrived through a file rather than through clicks
in a UI. That is a difference in *input method*, not in kind, and it makes the
80.5% far better supported than the caveats here implied.

Two things follow:

- **The organic-vs-imported split is much weaker evidence than it looked.** The
  14 "organic" clips are simply the ones rated in the UI afterwards; both groups
  are the same person's judgements. A 6.2% on 16 pairs is far more likely to be
  small-sample noise than a sign the larger set encodes the wrong objective.
- **`_SUPERVISED_MULTIPLIER = 0.45` may be miscalibrated.** v1 discounts these
  to 45%, and v2 copied that. If they are ordinary hand labels the discount is
  arbitrary, and it is worth measuring with it set to 1.0.

Still worth confirming with the person who made the file how carefully those
553 decisions were made -- rating a long list quickly is different from watching
each clip -- but "these are not real preferences" is no longer the default
assumption.

**Note:** `clips.jsonl` is no longer on disk. The 553 labels now exist only
inside `.local-data/salience.db`. They should be re-exported before any
destructive operation on that database.

### The hard test: can it pick the right clip *within* one fight?

The across-fight metric can be won cheaply. A model that learns nothing but
"eliminations beat knocks" scores well on pairs drawn from different fights,
because that correlation is real and pervasive.

Within one fight it cannot. Both clips share the map, the weapon, the opponent,
the health state, and are seconds apart -- the user kept one and skipped the
other. Almost every confound is held constant, so the ordering isolates taste.

93 clips from 33 split fights were indexed for this (they were largely missing
before, because dedupe had been discarding exactly these siblings -- the bug was
hiding its own evidence). Corpus now 175 labelled clips.

| | fields | embedding | both | v1 |
| --- | ---: | ---: | ---: | ---: |
| within-fight (folds 2, 37 pairs) | 62.2% | **73.0%** | 64.9% | -- |
| within-fight (folds 3, 26 pairs) | 61.5% | **84.6%** | 76.9% | -- |
| across-fight (2494 pairs) | 66.5% | **71.5%** | 69.1% | 63.3% |

**The embedding holds up on the hard test.** Within-fight accuracy (73-85%) is
at least as high as across-fight, which means the model is not merely riding
event-type correlation -- it can tell a better version of a moment from a worse
one. That is the strongest evidence so far that the VLM summary is doing real
work.

Caveat: 26-37 pairs is small, and the fold-to-fold spread (73% vs 85%) reflects
that. The direction is consistent; the magnitude is not yet pinned down.

**Why the across-fight numbers fell** (embedding 80.5% -> 71.5%, v1 74.2% ->
63.3%): the corpus changed, not the models. The 93 new clips are all from fights
the user split, which are by construction the ambiguous cases. Every ranker
dropped by a similar amount on a harder, more representative corpus. The old and
new across-fight numbers are not comparable; the v2-vs-v1 gap measured *on the
same corpus* is what matters, and it widened slightly to ~8 points.

### Ablation: do the deterministic facts help?

`V2_ABLATE=facts` strips every deterministic fact from the prompt -- filename
event, weapon, health, event time, census -- leaving only the seven frames. Run
on the same 93 split-fight clips, scored on identical pairs.

| folds | | fields | embedding | both |
| ---: | --- | ---: | ---: | ---: |
| 2 | with facts | 66.6% | 69.9% | 67.9% |
| 2 | **no facts** | **74.6%** | 66.0% | **75.2%** |
| 3 | with facts | 60.5% | 66.1% | 63.2% |
| 3 | **no facts** | **70.1%** | 65.5% | **71.3%** |
| 4 | with facts | 65.1% | 68.2% | 67.7% |
| 4 | **no facts** | **70.0%** | 58.8% | 69.6% |
| 5 | with facts | 64.8% | 65.8% | 67.8% |
| 5 | **no facts** | **68.3%** | 65.8% | **70.6%** |

Two findings, both consistent across every fold setting:

**1. The facts do not help the summary.** `embedding` is unchanged with and
without them (66-70% vs 59-66%, overlapping). The prose the taste model actually
ranks on is no better for having been handed OCR output.

**2. The facts actively damage the structured fields.** `fields` is **4-10
points worse** with facts than without, in all four settings, and `both` follows
it. Removing the facts improves the pipeline.

The mechanism is not mysterious. Told "the weapon HUD reads EXTENDING FOCUS
SHOTGUN", the model copies that into `weapon_name` -- including when OCR is
wrong, which `hud_conflict` showed happening on 2 of the first 6 clips. Left
alone, it reads the weapon off the frames and gets it right more often. The
injected facts are laundering OCR errors into the structured output.

This is the outcome flagged as least likely when the ablation was set up:
the deterministic layer is not merely redundant, it is a net negative.

**Consequences, pending a wider replication:**

- The fact block should be removed from the prompt.
- With it goes the justification for most of `facts.py`: weapon timing, health
  boxes, the census, and the banner localizer's precision -- though localization
  still chooses *which frames are sent*, which this ablation did not vary.
- The remaining honest use for OCR is monitoring (read rates, `hud_conflict`) and
  the review UI's seek-to-event, not prompt context.

**Caveat:** one corpus of 93 clips, one prompt variant, one model. The
consistency across folds makes noise unlikely, but this should be replicated on
the full 175 before deleting code.

### Localization ablation: the scan pass can go

`V2_ABLATE=locate` replaces the entire localization stage -- 24 scan frames, OCR
on each, marker detection, run grouping, last-run selection -- with
`event_sec = 0.75 x duration`. It implies `facts`, so it is measured on top of
the no-facts configuration; only one thing differs between it and
`v2.3-nofacts`, namely which frames the model sees.

All 189 clips (175 labelled), across-fight pairs, `embedding` scores:

| folds | v2.3 (facts + OCR locate) | v2.3-nofacts | **v2.3-nofacts-naiveloc** |
| ---: | ---: | ---: | ---: |
| 2 | 74.0% | 72.4% | 73.6% |
| 3 | 71.5% | 68.8% | **72.9%** |
| 4 | 72.2% | 74.2% | 73.8% |
| 5 | 72.6% | 69.9% | 72.6% |
| mean | 72.6% | 71.3% | **73.2%** |

`both` scores, same runs: v2.3 63.7-67.2%, nofacts 64.6-69.1%, **naiveloc
72.7-75.7%** -- the naive variant wins that comparison outright.

**Localization by OCR buys nothing.** Accuracy is flat across all three variants
(71-73%, overlapping in every fold setting), and the naive variant is marginally
the best and the most stable. A multiplication picks frames as good as the
entire scan-and-group stage.

Measured cost of what it replaces:

```text
probe                     0.6s
extract 24 scan frames    1.1s
OCR scan (localize)     113.2s   <- deleted
extract 7 frames          1.3s
OCR window (facts)       57.5s   <- deleted
```

(Absolute times inflated ~5x by CPU contention; the ratio is the point. OCR was
98% of local runtime.) In practice the ablation run processed the same 189 clips
in roughly a sixth of the time.

### What should be deleted

On this evidence:

- the fact block in `prompt.py` and `_fact_lines`
- the localization stage: `locate_event`, `_marker_frames`, `_group`, the marker
  tables, and the 24-frame scan extraction
- weapon attribution: `weapon_from_reads`, `killfeed.py`, the pre-event timing
  logic, the ontology dependency
- the vitals reader and its regions

What survives: filename event parsing (free), engagement grouping (for pair
independence and sibling presentation), `0.75 x duration` frame selection, the
VLM call, the embedding, the taste model.

Two things worth keeping OCR around for, neither in the prompt: monitoring
(read rates as a UI-change alarm) and the review page's seek-to-event.

**Caveat before deleting:** one corpus, one model, one game mode. The result is
consistent across four fold settings and 1400-3800 pairs per setting, which
makes noise unlikely -- but every measurement here comes from one player's
library.

### Final numbers: v3.0 on 334 labelled clips

Corpus grown from 175 to 334 labelled clips (105 liked, 229 not). Folds now split
by **engagement**, not by clip -- see the correction below.

| | centroid | logistic | v1 shipped |
| --- | ---: | ---: | ---: |
| across-fight, embedding | 80.2-80.9% | **80.6-81.1%** | 69.4% |
| across-fight, both | 77.5-79.2% | 78.8-80.2% | -- |
| within-fight, embedding | 60.3% | 60.3% | -- |
| within-fight, both | 61.9% | **63.5%** | -- |

**More data was the biggest lever.** Across-fight embedding went 73% -> 81% purely
from describing 334 clips instead of 175. The ranker change (logistic vs
centroid) is worth under a point; the encoder and corpus mattered, the fitting
method did not.

**v2 beats v1 by ~11 points** on the same clips and pairs.

### Correction: within-fight accuracy is ~62%, not 73-85%

An earlier section reported within-fight accuracy of 73-85% on 26-37 pairs. That
was measured with folds split by clip, which was wrong two ways:

1. **It leaked.** Siblings of one fight landed on both sides of the split, so the
   model trained on a near-duplicate of the clip it was tested on.
2. **It starved the metric.** Splitting by clip scattered siblings across folds,
   leaving 15 within-fight pairs out of 334 clips.

With group-aware folds there are 63 pairs, and within-fight accuracy is
**60-64%**. The earlier figure was optimistic noise. (Notably the across-fight
number *rose* slightly after the fix, 79.1% -> 80.9%, so the leak was not what
was holding it up.)

**What the gap means, and it is the most useful finding here.** The model is
strong at "is this fight worth watching" (81%) and weak at "which clip of this
fight is best" (62%). That is coherent: siblings of one fight produce nearly
identical 40-word summaries, and the difference between them lives in detail the
summary does not capture.

Product consequences:

- Engagement grouping matters *more*, not less. The right interface presents a
  fight once with its siblings available, rather than trusting the ranker to pick
  the best one -- which is also what the 14% dedupe error already implied.
- The lever for within-fight discrimination is richer per-clip description, not a
  better ranker. More frames, or a summary asked to describe execution quality
  rather than event identity, are the things to test.

### Richer description does not fix within-fight (v3.1-exec)

Hypothesis: within-fight accuracy is ~62% because siblings of one fight produce
nearly identical summaries, so adding observable execution detail -- shots fired,
target/player movement, return fire, engagement duration, and a clause on *how*
the kill was done -- should separate them. The extra text is concatenated into
`embed_text` so it reaches the taste model rather than sitting unused in JSON.

Same 93 split-fight clips, both variants, 63 within-fight pairs:

| variant | model | embedding | both |
| --- | --- | ---: | ---: |
| v3.0 | centroid | 65.1% | 65.1% |
| v3.0 | logistic | **68.3%** | 63.5% |
| v3.1-exec | centroid | 61.9% | **68.3%** |
| v3.1-exec | logistic | 63.5% | **68.3%** |

Across-fight on the same clips (651 pairs): v3.0 66.1/71.3%, v3.1-exec
66.5/68.7%.

**No effect.** Best-of-configuration is 68.3% either way, every difference sits
inside the ~±6 point noise band that 63 pairs implies, and across-fight did not
improve either. The extra fields are accurate and readable -- *"Tracks moving
target through trees, fires one shot, then swaps weapons"* -- and they change
nothing about ranking.

Output tokens rose ~115 -> ~181 per clip for that, so v3.1-exec is strictly worse
than v3.0: same accuracy, higher cost. **v3.0 stays the default.**

### What the negative result implies

Two readings, and the second is now more interesting:

1. **Siblings genuinely differ by something not in the frames** -- which frame
   window was captured, luck, context outside the clip. If so, no description
   fixes it, and the answer is the interface: group the fight, show the siblings,
   let the user pick. The 14% dedupe error pointed at the same conclusion from a
   different direction.
2. **The bottleneck is the text channel, not the model's perception.** Everything
   the taste model sees passes through ~40 words of English. That is a very narrow
   pipe for "which of these two near-identical kills is better", and widening it
   with more prose did not help because prose is the constraint.

Reading 2 has a concrete test that has not been run: **embed the frames
directly** (CLIP-style image embeddings of the 7-frame window) instead of, or
alongside, the sentence. That bypasses the text bottleneck entirely, and it is
the idea this project set aside early on for lack of data -- now with 348
described clips and a working metric, it is testable and free of API cost.

### Image embeddings: a small across-fight gain, nothing within-fight

CLIP (`clip-ViT-B-32`) embeddings of the 7-frame window, mean-pooled, computed
locally for all 348 clips. No API cost. Logistic ranker, 334 labelled clips.

| features | across-fight (4730 pairs) | within-fight (63 pairs) |
| --- | ---: | ---: |
| fields | 75.5% | -- |
| embedding (text) | 81.1% | 60.3% |
| both (fields+text) | 80.2% | 63.5% |
| image | 70.3% | 61.9% |
| **image+text** | **82.1%** | 61.9% |
| v1 shipped | 69.4% | -- |

**Image alone is clearly worse than text** (70.3% vs 81.1%). A VLM sentence
carries more ranking signal than pooled CLIP vectors -- the description is doing
real work that raw pixels do not replicate.

**image+text is nominally best at 82.1%**, but the +1.0 point over text alone
should not be treated as established. 4730 pairs come from 334 clips, so pairs
are heavily dependent and the effective sample is far smaller than the pair count
suggests. Directionally it is worth keeping; it is not a result.

**Within-fight did not move.** 60-64% across every feature set tried: text,
richer text, pixels, and both together. Three separate attempts at the same
problem, all null.

### Conclusion on within-fight: it is not a description problem

Tried and failed: more prose (v3.1-exec), image embeddings, image+text, two
ranker families. Nothing moves 62%.

The remaining explanation is that siblings of one fight differ by something the
7-frame window does not contain -- which moment the auto-clipper happened to
capture, what happened before or after the window, or preference that is simply
not stable at that granularity. A wider window is the one untested variable.

**Product answer, arrived at from three directions:** do not ask the ranker to
pick the best clip of a fight. Group the fight, surface it once, show the
siblings, let the user choose. The 14% dedupe error said this, the within-fight
gap says it, and two failed feature experiments say it.

Rank *fights* (82%, comfortably beating v1's 69%). Present *clips* within a
fight rather than ordering them.

### `rank`: the product output

```bash
python pipeline_v2/cli.py rank --top 12
```

Trains on every label available, scores all described clips, groups them into
fights, orders the **fights**, and lists siblings without ordering them --
directly implementing the conclusion above.

The top of a 215-fight inbox, trained on 334 labels:

```text
1. [+1.20] ... aims down sights with a Hunting Rifle, and lands a headshot to knock the opponent.
2. [+1.19] ... fires a single shot that lands a headshot, cracking the enemy's shield...
3. [+1.16] ... aims a Hunting Rifle, and fires a single shot to eliminate them with a headshot.
4. [+1.13] ... lands a long-range headshot, knocking down a distant opponent.
5. [+1.10] ... shoot an opponent across a river from behind a rock, landing a headshot...
6. [+1.10] ... firing a single shot that results in a headshot elimination.
```

**The learned taste is legible**: long-range rifle headshots, single shot, often
from elevation. That is a real preference recovered from keep/skip decisions and
stated in plain English -- something v1's 33-number vector could gesture at
(`sniper_kill`, `no_scope`) but never articulate.

It also exposes the next problem, and it is not accuracy: **the top six are
nearly the same clip**. A pointwise ranker optimises each clip independently, so
a user whose favourite thing is sniper headshots gets an inbox of nothing else.
The fix is a diversity penalty at presentation time -- demote a fight that is too
similar to one already shown, using the embeddings that are already computed --
not a better ranker.

### Frame resolution (v3.3): descriptions got accurate, ranking did not improve

A user-reported error started this: a clip where the player cracked one enemy's
shield with a Hunting Rifle then redirected onto a *second* enemy and headshot
them with the same rifle came back as *"switches to a pistol to secure a headshot
elimination on the downed enemy"*.

Prompt fixes failed. Forcing `changed_target` / `changed_weapon` /
`opponents_engaged` as explicit fields produced confidently wrong answers
(`1 opponent, no target change, weapon changed: yes`).

**Looking at the seven frames explained it.** Frames 1 and 4 both show the same
scoped Hunting Rifle; frames 2-3 are third-person. The model was reading the
**ADS-to-third-person view change as a weapon change**. And the evidence it
needed was on screen -- frame 5 reads `SHIELD CRACKED X2 / KNOCKED / ASSIST /
ELIMINATION` -- but at 960px that text is a few pixels tall.

| frame width | phantom weapon swap | opponents | input tokens |
| --- | --- | ---: | ---: |
| 960 | **yes** | 1 | 4,241 |
| 1440 | **no** | 1 | 8,608 |
| 1920 | no | 1 | 15,013 |

1440 fixes it; 1920 adds nothing for 75% more tokens. Spot checks found a second
class fixed too: `"Strike Pump Shotgun"` became `"Striker Pump Shotgun"`.

**The uncomfortable part.** Re-describing all 348 clips at 1440 did *not* improve
ranking, and may have slightly hurt it:

| folds | v3.0 (960px) | v3.3 (1440px) |
| ---: | ---: | ---: |
| 2 | 79.3% | 79.8% |
| 3 | 81.9% | 79.3% |
| 4 | 81.1% | 80.3% |
| 5 | 82.1% | 80.4% |

Mean 81.1% -> 79.9%. Small, inconsistent in sign at folds=2, and inside the noise
band that dependent pairs from 334 clips imply -- so not established. But it is
not an improvement either, and the direction is worth naming.

One hypothesis worth testing rather than believing: **the hallucination may have
been an accidental feature.** The ADS misread fires on scoped shots, and this
user's taste is long-range rifle headshots -- so "switches to a pistol" was an
unintentional marker for "scoped rifle kill", correlated with keep. Fixing the
error removed a signal the ranker had been using. A systematic error can be a
useful feature, and optimising accuracy can delete it.

**Decision: ship v3.3.** The descriptions are user-facing, a fabricated weapon
swap costs trust, and the ranking difference is not distinguishable from noise.
Accuracy of the thing the user reads wins a tie.

### What resolution did not fix

Entity tracking. `opponents_engaged` is still 1 where the truth is 2, and 1920px
behaves identically to 1440px, so it is not a legibility limit. Tracking two
enemies across scope-out/scope-in is the cross-frame reasoning VLMs are weakest
at, and a 7-frame window may not support object permanence at all. Open.

### The wider lesson from this session

The pipeline spent most of its life compensating for a self-inflicted problem.
OCR existed because "the model cannot read small HUD text" -- true, but only
because the frames were being downscaled to 960px first. The game states its
facts on screen; the pipeline shrank them out of legibility, then rebuilt a lossy
copy of them with a separate OCR stack, and injected that copy back into the
prompt where it measurably hurt.

The sequence that fixed it: inject facts -> measure that they hurt -> delete them
-> find descriptions inaccurate -> discover the cause is resolution -> stop
degrading the input. At no point was the answer more machinery.

### Audio: loudness helps, transcripts do not

The pipeline was deaf. `audio.py` adds two independent things, and they behave
completely differently.

**Discovery first:** 131 of 348 clips carry speech -- far more than the one clip
expected. Mic capture was on for much of the library.

| features | voiced only (130) | all (334, folds 4/5) |
| --- | ---: | ---: |
| text (summary) | 66.8% | 78.7 / 79.2% |
| image+text | 65.7% | 80.3 / 80.4% |
| **transcript alone** | 49.8% | **39.1%** |
| text+transcript | 63.8% | 75.9% |
| **reaction scalars alone** | 60.1% | **67.1%** |
| **text+reaction** | **70.5%** | 80.0 / 79.8% |
| **image+text+reaction** | 68.1% | **81.4 / 81.3%** |

**Transcripts are worthless here, and slightly harmful.** A transcript embedding
is a *topic* embedding: it encodes what the conversation was about, and this
library's speech is mostly ordinary chat ("Mm-hmm", "Nothing else to do", "UK
rapper?"). Reaction words are a small share of the tokens. Below chance on the
full corpus because voiced clips are *less* likely to be keeps -- 24% vs 31% --
so mic-on correlates weakly with casual sessions.

**Three loudness scalars are worth ~1 point overall and ~4 points on voiced
clips.** `peak_ratio` (loudest 50ms window vs the clip's own median),
`loud_fraction`, and a has-speech flag. `image+text+reaction` is the best
configuration measured: **81.4%**, against v1's 69.4%.

**What `peak_ratio` probably measures, stated honestly:** not necessarily human
reaction. A loud moment is gunfire, an explosion, or a shout, and this pipeline
cannot tell them apart. It may be functioning as an *action intensity* detector
rather than a reaction detector. That does not make it less useful, but it does
mean the feature should not be described as "detects when your friends react"
without a test that separates voice from game audio.

**Process note.** The first conclusion recorded here was "audio measures to
nothing", drawn after testing only transcripts. Two of the two tiers had been
built; one had been measured. The scalar tier was tested only after the claim was
challenged, and it reversed the finding.

### The honest state of the metric

The 80.5% is measured almost entirely on **supervised imports** -- 96 of 111
labels came from a bulk import, not from the user reviewing clips. Two readings
fit every number above, and the current data cannot separate them:

1. The taste model works, and the organic subset is too small to measure.
2. The taste model has learned whatever rule generated the *import*, and that
   rule is not the user's taste -- which is why it inverts on the 14 clips that
   record real decisions.

Reading 2 is not the likelier one, but it is not ruled out, and it would
invalidate the headline number entirely. **No amount of additional supervised
data resolves this.** The only fix is organic labels: the user reviewing clips
and pressing keep/skip, a few hundred times.

Until then the defensible claim is narrow: *on imported labels, ranking on the
VLM summary beats both a v1-shaped feature vector and v1's shipped score.* That
is real, and it is not yet the same as "v2 ranks clips the way this user wants".

- **The labels are mostly supervised imports.** Only 15 of these are organic
  keep/skip decisions. The metric may be measuring agreement with whatever rule
  produced the import, not taste.
- **Pairs are not independent.** 985 pairs come from 111 clips, so each clip
  recurs in many comparisons; the effective sample is far smaller than the pair
  count suggests.
- **There is still no v1 baseline.** `fields` is a v1-*shaped* representation
  built from v2's own outputs, not v1's ranking. The head-to-head that decides
  whether to keep building v2 has not been run.

### The next measurement, precisely

Describe ~150 labelled clips (roughly 650k input tokens -- affordable), install
`sentence-transformers`, and rerun. That gives enough pairs for the fold-to-fold
swing to settle, and replaces the weakest component with a real one. Only then
is the fields-vs-embedding question answerable.

## Next: M2 (continued)

Run ~50 clips, read every output by hand, and settle the schema. Specifically
worth watching: the `hud_conflict` rate (it fired on clip one, which may mean
the weapon HUD region is misaligned rather than that the model is right), and
whether summaries stay descriptive rather than drifting evaluative.
