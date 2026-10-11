# Salience v2: run and develop

[SCOPE.md](SCOPE.md) owns current decisions. Earlier experiments are preserved
in [the archive](../archive/v2-experiments.md), not used as running instructions.

Current evaluation: [targeted action follow-up](evaluation-action-20260907.md).
All ten action/control moments were reviewed at [localhost:8779](http://127.0.0.1:8779/):
three keeps, seven skips. The report records weapon attribution, build mechanics,
and spectator errors. The 12 background judgments are retained; recording is paused.

Latest: [automatic localization pilot](evaluation-localization-20260908.md).
The opt-in offline prototype found four known own-player events in whole chunks
and generated shot windows without manual times. It missed a spectator kill;
piece control and reliable detailed narration remain unresolved.
`V2_ABLATE=attribution` enables the development variant; v3.3 remains the default.
`V2_ABLATE=killweapon` is the isolated `v3.5-killweapon` arm: it changes only what
`weapon_name` means (the weapon that *caused* the outcome, or `unknown`) and is
byte-identical to v3.3 otherwise, so a measured difference is attributable to that
one change. It has not been scored against ground truth yet.

## Setup

Python 3.11+, FFmpeg/FFprobe on PATH. From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pipeline_v2 --help
python -m pipeline_v2 localize sample-clips --limit 1
```

Shared media utilities live in the neutral `salience_media` package; nothing
here imports the retired v1 backend. No v1 server, model, or database is
required. TF-IDF works offline. Optional semantic embeddings:
`python -m pip install -e ".[semantic]"`; first use downloads the encoder.
TF-IDF results do not reproduce historical neural-embedding metrics.

## Describe and index

Set `FIREWORKS_API_KEY` and `FIREWORKS_MODEL` in the root `.env` or environment.
Choose a vision model available to your account; the historical default ID is
not a current availability guarantee. Configure current per-million-token prices
with `V2_INPUT_PRICE_PER_MTOK` and `V2_OUTPUT_PRICE_PER_MTOK` for cost estimates.

```powershell
python -m pipeline_v2 process sample-clips --limit 1 --max-calls 1
python -m pipeline_v2 index sample-clips --limit 10 --max-calls 1 --encoder tfidf
python -m pipeline_v2 spend
```

`process` prints; `index` stores and rebuilds embeddings. Defaults allow one
new description for `process`, zero for `index`. Cache hits need no key and
consume no call allowance. Cache misses upload selected frames.
`review`, `rank`, `taste`, and `localize` make no paid calls.

`V2_BUDGET_USD` stops new calls when **recorded estimated spend** reaches its
threshold. It is not a prepaid hard cap: the last call may exceed it, unset
prices cannot enforce dollars, concurrent runs are not coordinated, and a
connection failure may hide provider usage. Use small call counts and provider
account limits when a hard dollar ceiling matters.

Cache identity covers clip, prompt version, model, endpoint, prompt, and exact
image bytes. Old clip/version-only entries remain for audit but are not silently
trusted for a different request. Re-indexing an old library can require new
explicitly allowed calls. Invalid responses are retained to avoid paying again
for a parse failure. Crashes between provider completion and persistence remain
ambiguous; exactly-once billing is not guaranteed.

## Moved footage

The store records an absolute path per clip. Move the library, restore it from a
backup, or copy it from another machine and every path goes stale: `review` and
the ground-truth page serve 404s and the video simply will not start.

```powershell
python -m pipeline_v2 relink C:\path\to\clips          # dry run
python -m pipeline_v2 relink C:\path\to\clips --apply
```

Candidates are matched by filename and then **confirmed by content hash**,
because the store is keyed on that hash and a same-named different file would
silently attach a clip's descriptions to the wrong video. Duplicates, misses and
hash mismatches are reported and skipped, never guessed. `--no-verify` trusts
names instead and is only worth it when hashing the library is too slow.

Frozen study plans record clip hashes, not locations, so relinking does not
invalidate one.

## Weapon and actor ground truth

Every weapon claim before 2026-09-11 rested on seven windows picked *because*
they had already failed, then reused to tune the prompt. This collects labels on
a seeded random sample instead, so an arm can be scored on clips that were not
chosen for being wrong. No paid calls: the descriptions are already cached.

```powershell
python -m pipeline_v2.groundtruth prepare .local-data/groundtruth-20260911 --size 50
python -m pipeline_v2.groundtruth serve .local-data/groundtruth-20260911
python -m pipeline_v2.groundtruth report .local-data/groundtruth-20260911
python -m pipeline_v2.groundtruth report .local-data/groundtruth-20260911 --variant v3.5-killweapon
python -m pipeline_v2.groundtruth compare .local-data/groundtruth-20260911
```

Label at [localhost:8780](http://127.0.0.1:8780/): type the weapon that actually
caused the kill (or `unknown`) and say who got it. **The model's answer is never
shown** — revealing it mid-pass would teach the annotator its vocabulary and bias
every later clip. The sample and the claims being scored are frozen in
`plan.json`; re-running `prepare` will not reshuffle it.

Scores are reported per bucket — `exact`, `family`, `mismatch`, `model_declined`
— never as one accuracy number. Naming a different gun and declining to answer
are different outcomes, and moving clips from the first into the second is the
entire point of the v3.5 arm. Actor labels establish prevalence; v3.3 has no
actor field, so they are the reference for later arms rather than a score of it.

## Comparing models

`index --from-plan PLAN` describes exactly the clips a frozen study sampled, so
a new model or prompt is paid for only on the corpus its labels cover:

```powershell
$env:FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p6"
$env:V2_ABLATE = "killweapon"
python -m pipeline_v2 index --from-plan .local-data/groundtruth-20260911/plan.json --limit 10 --max-calls 12
python -m pipeline_v2.groundtruth compare .local-data/groundtruth-20260911
```

A model change gets its own storage variant (`v3.5-killweapon@kimi-k2p6`), so
arms never overwrite each other and `compare` can score them side by side. Only
compare rates between arms with the same `scored` count.

Two request quirks are resolved per model by `config.resolve`, both learned the
expensive way. `glm-5p3-flash` rejects `reasoning_effort="none"` with a 400.
`muse-glimmer-30b` accepts it, ignores it, and spends the whole output budget on
`reasoning_content` while returning `content: null` -- ten calls, no results,
full price. Both need the parameter dropped and a larger `max_tokens`.

Input dominates the bill roughly 70:1 (about 8,500 input tokens per call against
120 output), because the seven 1440px frames are nearly all of it. `V2_FRAME_WIDTH`
is therefore a bigger cost lever than model choice, but 960px is documented in
`frames.py` as causing misreads, so measure it against the labels before using it.

## Does capturing cost the game anything?

Play the same way twice and compare. The gate is SCOPE's: under 5% loss in game
1% lows, under 100ms A/V drift.

```powershell
python -m pipeline_v2.capture.bench baseline .local-data/bench --seconds 120
# arm capture, play the same way
python -m pipeline_v2.capture.bench capture  .local-data/bench --seconds 120 --clips <session>
python -m pipeline_v2.capture.bench report   .local-data/bench
```

A gate with no data reports `null` and the verdict is `inconclusive` -- it never
passes by default. `one_percent_low_fps` is the mean of the slowest 1% of frames
as a rate, because mean FPS can be untouched while 1% lows collapse, which is the
shape of the original complaint.

Frame pacing needs **PresentMon on PATH**; it is the only missing prerequisite
and the harness finds it automatically. GPU/VRAM/power come from `nvidia-smi` and
A/V drift from `ffprobe` regardless.

### Capture backends

Desktop Duplication **failed the gate on 2026-09-16 by eleven times**: it cost
55.2% of the game's 1% lows against a 5% budget, with every encode setting
already tuned. The damage was stalls, not throughput -- GPU utilisation moved
4% while the 99th-percentile frame tripled. DDA copies the whole desktop every
frame whatever is done with the result, so no flag removes that.

`capture/wgc.py` is the replacement: Windows Graphics Capture via the
`windows-capture` wheel (in `.[capture]`, so a shipped build bundles it and the
end user installs nothing). It captures **one window** rather than the desktop,
and `minimum_update_interval` throttles at the **source** -- a 30fps capture
composites half as many frames instead of merely encoding fewer. No injection,
so no anti-cheat exposure.

The frame pump never blocks the capture callback: a frame the encoder cannot
take is dropped and counted. Back-pressure there would stall the compositor,
which is the DDA failure again.

Benchmark it against the recorded DDA baseline with the same paired method:

```powershell
# admin PowerShell, in a match
python -m pipeline_v2.capture.bench baseline .local-data\bench-wgc --seconds 120
python .local-data\tools\wgc_record.py --fps 30          # separate window
$env:SALIENCE_CAPTURE_BACKEND = "wgc"
python -m pipeline_v2.capture.bench capture .local-data\bench-wgc --seconds 120
python -m pipeline_v2.capture.bench report .local-data\bench-wgc
```

### Capture settings

`SALIENCE_CAPTURE_ENCODER`, `_FPS`, `_BITRATE` and `_SCALE` override the
collector without editing code, so an encoder or rate can be A/B'd against the
game's frame times. An explicit encoder outranks both the preference order and
the cached choice in `capture-settings.json` -- that cache pinned `h264_qsv`
on a machine whose panel is wired to the NVIDIA GPU, where Quick Sync cannot
open an MFX session at all. `Display` now carries its adapter's PCI vendor and
the collector never probes an encoder the adapter cannot run.

Keyframes are one per segment (`fps x segment_seconds`). `-g fps` forced an IDR
every second -- five times the original recorder -- and `-force_key_frames`
already guarantees a keyframe per segment boundary. `-maxrate`/`-bufsize` are
gone for the same reason: pinning maxrate to the target makes rate control
effectively CBR.

DDA duplicates the **whole desktop** every frame and copies it again into the
encoder's input surface, on the same GPU the game is using. No flag removes
that. `capture/obs_client.py` is the alternative already written here: OBS
solves exclusive fullscreen, multi-GPU, HDR and audio sync, uses OS-level
capture with no game injection, and its replay buffer makes the event timestamp
*known* rather than guessed at `0.75 x duration`. It needs `.[obs]` and a
driver; nothing calls it yet.

Measured 2026-09-13: all 194 chunks of the September 7 session are **video only**,
so the drift gate is not applicable until the recorder records audio. FFmpeg
cannot supply that here -- `dshow` lists microphones but no loopback device --
so game audio has to come through `soundcard` (already a `capture` extra), and
muxing it means replacing the stdin `q` finalize path in `collection.CaptureRun`.

## Review and rank

```powershell
python -m pipeline_v2 review
python -m pipeline_v2 rank --top 12
python -m pipeline_v2 taste --organic-only --model logistic
python -m pipeline_v2 similar --top 3
```

Review at [localhost:8777](http://localhost:8777/): J skips, K keeps. The
[ranked inbox](http://localhost:8777/ranked) and CLI share one implementation.
Ranking requires ten complete labelled examples and both classes. This permits
fitting; it does not prove generalization. With no labels, review first.

Current defaults: **v3.3 descriptions, text embeddings, logistic ranking**.
Image/audio preprocessing remains experimental. `--mode image+text` is opt-in
when image vectors exist. Their small historical gains need held-out validation.

Legacy labels are explicit: `rank --db .local-data/salience.db` or
`taste --db .local-data/salience.db --model logistic`. Import opens that database
read-only. v2 decisions override imports. Imported human decisions remain useful;
provenance and session coverage determine what they validate.

Data lives in ignored `pipeline_v2/.data/`. Back it up before manual experiments.
Re-indexing preserves preferences, embeddings, and other description versions.

## Code map

| Modules | Responsibility |
| --- | --- |
| `config.py` | Current defaults, credentials, paths |
| `facts.py`, `frames.py`, `prompt.py`, `vlm.py`, `cache.py` | Video to cached description |
| `store.py`, `embed.py` | Durable records and vectors |
| `feedback.py` | Optional read-only v1 import |
| `taste.py`, `ranking.py` | Evaluation and shared fight ranking |
| `review.py`, `cli.py` | Review, inbox, commands |
| `evaluation.py`, `evaluation_review.py`, `evaluation_metrics.py` | Frozen offline pilots and separate user evaluation labels |
| `localization.py`, `evaluation_attribution.py` | Opt-in event proposals and controlled description experiments |
| `freeze.py`, `jsonio.py` | Frozen-runtime checks and atomic JSON, shared by the studies |
| `groundtruth.py` | Blind weapon/actor labels on a frozen random sample |
| `capture/bench.py` | Paired capture-on/off performance gate |
| `capture/`, `audio.py` | Experiments, not prerequisites |

The v1 weapon ontology is gone from v2. It mapped a name to a family via
`weapon_category`, which was a derived property and so never survived
`model_dump()` -- `field_vector` read `"unknown"` for all 348 clips since the
table was created. The module went back to `backend/`, its only remaining user.

`audio.py` and `embed.ClipImageEncoder` are the libraries behind the store's
`audio_features` / `image` rows, but the scripts that populated them are not in
this repo, so `taste`'s image and reaction columns cannot be rebuilt from a
clean checkout. Descriptions are labelled by the prompt that produced them:
auto-clipped files are `v3.3`, uniformly sampled capture windows `v3.3-window`.

Event location is **75% of duration** for the original auto-clipped library.
Finalized collector chunks instead use their explicit `.capture.json` analysis
window. That uniform sample is not event detection and can miss action elsewhere
in a chunk. No OCR or ONNX fallback is active.

The [September 7 pilot](evaluation-20260907.md) tests 60 unseen recording windows,
documents capture FPS/crash problems, and provides a separate local review at
port 8778. The recorder is paused for normal use. Evaluation labels remain out
of training; do not retune on this inspected session and call it untouched data.

Run `python -m pytest` and `python -m ruff check pipeline_v2 salience_media`. Tests isolate data
and mock cloud calls; capture integration uses generated video, not the user's
screen or microphone. Prefer module entry points; direct CLI scripts still work.
