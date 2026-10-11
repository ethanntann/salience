> Historical snapshot, superseded by the root README and pipeline_v2/SCOPE.md.
> Relative links below refer to the original document location.

# Pipeline v2 — scope

A clean-room implementation of the VLM-first flow. It lives beside the existing
`backend/` and does not modify it. The v1 pipeline keeps working throughout;
this one has to earn its place on measured results before anything is retired.

## The flow

```
clip saved (event-containing, not event-centered)
  -> localize the event within the clip           [see below]
  -> extract N frames densely around it
  -> OCR weapon + victory banner locally           [deterministic facts]
  -> one cloud VLM call: frames + OCR facts in the prompt
  -> back: structured JSON + one descriptive sentence + hud_conflict
  -> embed the sentence locally
  -> taste model over embedding + structured fields
  -> rank; below threshold -> quarantine with undo
```

One VLM call per clip. Three artifacts from it: structured fields for control,
the sentence for display, the embedding for ranking.

## Localizing the event

NVIDIA Highlights is an engine integration, not computer vision: the game calls
NVIDIA's SDK to report an event, which is why its labels are exact and why it
only works in titles the studio integrated. What it gives you is a ~30s window
that contains an elimination and a filename saying so. What it does not give you
is **when** inside that window.

This matters directly to cost. Seven frames spread across 30 seconds is one
frame every four seconds; the moment falls between them routinely, and you pay
full VLM price for a description of someone running across a field. Localize
first, then sample densely around the event — same cost per call, far more
information in it.

Three signals, cheapest first:

1. **Filename.** The auto-clipper names the file with the event type
   (`... Elimination ...`). Free, exact, and worth feeding to the VLM as a
   grounded fact alongside the OCR context. Costs nothing, more reliable than
   anything inferrable.
2. **Personal hit feed via OCR.** The centre-screen banner (`ENEMY ELIMINATED`,
   `KNOCKED`, `ENEMY DOWNED`) appears just after the shot and lingers — the
   behaviour v1's reducer had to dedupe, read forwards instead as a clock.
   Deterministic, no model involved.

   Two corrections from M1, both found only by running real clips:

   - **Not the kill feed.** The upper-left feed is *global* and reports every
     player's kills. The first clip tested localized onto a stranger's shotgun
     knock. Only the centre-screen personal feed is the recording player's own
     event.
   - **Not the first appearance.** Clips hold several events — one 22s clip had
     two knocks and two eliminations. Among marker groups matching the
     filename's event type, the **last** group is the trigger; measured
     placement is consistently ~76% through the clip, because the auto-clipper
     keeps mostly preceding footage. The window is therefore biased backwards
     (2.5s before, 1.5s after).

   The count of other events in the clip is passed to the VLM as a fact so a
   multi-kill clip is not described as a single kill.
3. **v1 locator ONNX**, as fallback when OCR finds nothing.

Cheap deterministic signal first, model second — the ordering used elsewhere in
this pipeline.

Risk to check in M1: verify against real clips that the auto-clipper integration
is still active for Fortnite and that filenames still carry event types. This is
a third-party dependency outside your control, and it is one of the arguments for
M6b.

## What is reused and what is new

Rewriting frame extraction and OCR would be pure duplication — both are working,
tested code. v2 imports them and owns everything downstream.

| Component | Decision |
| --- | --- |
| Frame extraction (`clips/keyframes.py`) | **Reuse** via import — proven, batched, has a CFR fast path |
| HUD OCR (`features/hud_ocr.py`, `features/ocr_regions.py`) | **Reuse** via import — region-scoped and fast |
| Media probe (`clips/media_probe.py`) | **Reuse** |
| VLM client | **New** — different prompt, different response contract than the v1 teacher |
| Storage | **New** — own SQLite file, own schema, no shared tables |
| Taste model | **New** — operates on embeddings, trained pairwise |
| Locator ONNX | **Fallback only** — used to localize the event when OCR banner detection finds nothing. Returns as the capture trigger in M6b |
| Student event/context heads | **Not used** in M1-M5 — the VLM replaces them. Kept, not deleted |
| `ranking/highlights.py` reducer | **Not used** — no multi-event reconciliation when the VLM describes the clip as a whole |

The last two rows are the point of the rewrite. If the clip is already centered
on an event, the locator and the multi-event reducer are solving problems that
no longer exist. Dropping them is most of the simplification.

Import path: v2 adds `backend/` to `sys.path` the same way `scripts/` already
does. No packaging changes, no changes to v1 files.

## Budget: $12 of Fireworks credit

This is the binding constraint on the whole plan, so it gets designed for rather
than discovered.

**Measured in M1: ~4,175 input + ~120 output tokens per clip** (seven 960px
frames plus prompt). Multiplied by current provider pricing that puts $12 in the
thousands of clips — comfortably more than this plan needs. Cost is not the
binding constraint the estimate feared, but the guards below stay, because the
expensive mistake was never volume: it is paying twice for the same clip.

One real cost trap did appear, and it was not about volume: the configured model
is a reasoning model, and the first call spent its entire output budget thinking
without producing JSON. Full price, zero result. `reasoning_effort: "none"` fixed
it. Re-check on any model swap.

Four rules, enforced in code, not discipline:

1. **Cache by content hash.** Key on `sha256(clip file) + prompt_version`. The
   same clip is never paid for twice, including across reruns and crashes. This
   matters more than any other optimisation here.
2. **Persist the raw response.** Every request and response is written to disk
   before parsing. A parsing bug must never cost a re-call.
3. **Hard spend ceiling.** A `--max-calls` argument, defaulted low, and a running
   cost estimate printed per call. The process refuses to exceed it.
4. **Measure on clip one.** The first call prints actual token usage and computed
   cost. Everything above is an estimate until that number exists.

Order of spending: 10 clips (does it work at all) -> 50 clips (is the schema
right) -> a few hundred (is the taste model learnable). Do not skip to the third.

## Milestones

### M1 — One clip, end to end, on disk — **done**

Findings are in [README.md](README.md). Three assumptions in this document were
wrong and have been corrected in the sections above: the kill feed is global
rather than personal, clips contain multiple events rather than one, and cost is
not the binding constraint (~4,175 input tokens per clip).


Localize the event, extract frames around it, run OCR, build the prompt, call
Fireworks, persist raw response, parse into the schema, print it. No database,
no ranking, no UI.

Done when: a real clip produces a valid structured record and the true cost per
clip is known.

Also verified here, before anything is built on top of it: that banner-based
localization actually lands on the event, and that filenames still carry event
types.

### M2 — Schema shakedown on ~50 clips

Run a batch. Read every output by hand. This is the only milestone whose main
deliverable is judgement rather than code.

Looking for: fields the model can't fill reliably, fields always the same value
(dead weight), missing fields you keep wishing were there, `hud_conflict` rate,
and whether the sentence is descriptive rather than evaluative.

Done when: the schema is stable and the prompt is versioned. Expect two or three
prompt revisions here — that is what this milestone is for.

### M3 — Storage and embeddings

SQLite schema for clips, VLM records, and embeddings. A local sentence encoder
(small, CPU, ~30MB class) turns each summary into a vector.

Done when: every processed clip has a persisted record and embedding, and
nearest-neighbour lookup over embeddings returns intuitively similar clips.
That last check is cheap and tells you immediately whether the embedding
carries usable signal.

### M4 — Taste model and the honest metric

Train pairwise on feedback: for pairs where one clip was kept and another
skipped, learn to order them correctly. Features are the embedding plus the
structured fields.

**Pairwise accuracy is the headline metric for v2**, replacing teacher
agreement. Report it against a baseline of random ordering and against v1's
ranking on the same pairs. This is the number that decides whether v2 is
actually better, so it exists as early as possible.

Done when: pairwise accuracy is measured on a held-out set of preference pairs
and beats both baselines.

### M5 — Quarantine, not deletion

Below-threshold clips move to a trash folder with a timer and a visible undo.
Every rescue is logged as a labelled mistake and feeds M4.

Done when: no clip is ever unrecoverably deleted by the system.

### M6 — Capture: watch folder now, own recorder later

This is where the product feels instant. Deliberately last: it is scheduling and
plumbing, not intelligence, and it is worthless until the stages above produce
something worth scheduling.

**M6a — watch folder.** A daemon processes clips as the third-party auto-clipper
drops them. Cheap, and enough to validate the whole loop.

**M6b — own recorder.** The intended destination: a desktop app that records
rather than depending on ShadowPlay, Medal, or OBS.

The consequence to plan for: **owning the recorder means owning event
detection.** M1-M5 assume the clip arrives already centered on an event, which is
true only because the third-party auto-clipper is detecting events using game
engine state. Remove it and that assumption goes with it — which is why the v1
locator and student are not deleted, only unused. In this design they return as
the **capture trigger**, which is a better fit for a small fast model than
labelling ever was.

Shape of it:

- Continuous ring buffer holding the last N minutes, hardware-encoded
  (NVENC/AMF/QuickSync) so the cost is near-zero.
- Sampled frames run through the v1 student a few times a second; a positive
  flushes the buffer to a clip file.
- That file enters the M1 pipeline unchanged.

Two hard constraints:

- **OS-level capture only** — Windows Graphics Capture API or DXGI Desktop
  Duplication. Never process injection or hooking: Fortnite runs Easy Anti-Cheat,
  and injection risks user bans and a blacklisted product. Non-negotiable.
- **Drive OBS before writing a capture engine.** OBS already solves replay
  buffering, fullscreen exclusive, multi-GPU, HDR, and audio sync, and exposes
  `obs-websocket` for programmatic control. Ship on top of it; replace it only if
  the encoder ever becomes a real differentiator, which is unlikely — nobody
  chooses a highlight product for its encoder.

Sequencing rationale: the recorder produces input that already exists on disk
today, and capture is table stakes that three free products already offer. The
ranking intelligence is the part nobody else has. Build that first.

## Layout

```text
pipeline_v2/
  SCOPE.md              this document
  README.md             how to run it
  config.py             paths, model id, budget ceiling, prompt version
  frames.py             thin wrapper over v1 keyframe extraction
  facts.py              thin wrapper over v1 OCR -> deterministic facts
  prompt.py             prompt construction and versioning
  vlm.py                Fireworks client, retries, raw-response persistence
  schema.py             Pydantic model of the VLM response contract
  cache.py              content-hash cache and spend ledger
  store.py              SQLite schema and access
  embed.py              sentence -> vector
  taste.py              pairwise ranker
  cli.py                process / rank / report commands
  tests/
  .data/                gitignored: cache, raw responses, db, frames
```

## The response contract

Sketch, to be settled in M2:

```json
{
  "summary": "one descriptive sentence, what happened, never how good it was",
  "outcome": "elimination | knock | death | no_event | other",
  "weapon": "shotgun | sniper | pistol | automatic | other | unknown",
  "player_health_state": "healthy | damaged | critical | unknown",
  "opponent": "human | likely_bot | unknown",
  "mechanics": ["close_range_duel", "build_fight", "..."],
  "hud_conflict": null,
  "confidence": "high | medium | low"
}
```

Two contract rules carried over from the design discussion:

- The summary is **descriptive, never evaluative**. If the model writes
  "impressive", the taste model learns the model's opinion instead of the
  user's. Prompt for what happened; judging is the ranker's job.
- OCR facts enter the prompt as a **strong prior, not law**, gated on the
  existing strong/weak read distinction. High-confidence reads are presented as
  context to prefer; weak reads are presented as hints or omitted. The model can
  disagree by naming the field in `hud_conflict`.

`hud_conflict` earns its place three ways: conflicting clips route to review
instead of silently poisoning the data, its rate is a monitoring signal that
catches a Fortnite UI change breaking OCR regions, and disagreements are the
highest-value labelling targets.

## Risks

| Risk | Mitigation |
| --- | --- |
| Cost per clip is 5x the estimate | M1 measures it before any batch runs; ceiling enforced in code |
| Event localization misses, so the VLM describes empty footage | M1 verifies banner localization on real clips; locator as fallback; wrong-window clips are visible immediately in M2's hand review |
| Auto-clipper integration disappears or changes filenames | Checked in M1; the medium-term answer is M6b |
| Summaries are generic and don't separate clips | Caught at M3 by nearest-neighbour inspection, before building the taste model on top |
| Not enough preference pairs to train on | Bootstrap from existing v1 `feedback_events`; the actions and weights are already recorded |
| Embedding overfits on few examples | Keep structured fields in the feature set; reduce embedding dimensionality if pairwise accuracy is unstable |
| VLM latency or rate limits | Async processing; the user never waits on a call |
| Scope creep back into v1 | v2 writes to its own database and imports v1 read-only |

## Out of scope

UI, multi-user, auth, local VLM, game support beyond Fortnite, retiring v1.

Recording is **deferred, not out of scope** — it is the intended destination
(M6b), sequenced after the intelligence is proven because it produces input that
already exists on disk today.
