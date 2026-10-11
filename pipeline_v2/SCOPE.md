# Current scope and project assessment

Updated 2026-09-11. This supersedes the original plans and chronological diary.
v2 owns product behavior and new development; the local student/teacher
workflow is retired and v1 is frozen.

## Direction

**Continue v2, focused on a useful personalized fight inbox.** Descriptions
remove much of the fixed-head/OCR reconciliation machinery and provide readable
ranking inputs. That simplification exchanges offline inference for cloud cost,
latency, and frame uploads. It is a promising trade, not a validated product yet.

The seven-frame experiment does not establish live event recall, motion
understanding, or generalization to new players, sessions, games, and seasons.

The [targeted action follow-up](evaluation-action-20260907.md) found coarse
elimination agreement in four manually selected windows, including a victory.
The TF-IDF ranking baseline still placed user-skipped travel above these kills.
The completed user review kept three of ten targeted moments and identified
kill-weapon attribution, build/edit mechanics, and spectator ownership errors.
Continue separating description quality, temporal coverage, and personal ranking;
this curated set cannot establish session recall. These clips are now development
cases; fixes need validation on separately annotated unseen fights.

The [controlled attribution comparison](evaluation-attribution-20260908.md) tested
prompt changes and 7 versus 19 frames. Explicit spectator ownership helped the
known case; weapon errors mostly persisted, and denser input introduced additional
claims. Keep the variant experimental. Increasing frame count globally is not
supported by this result, and piece-control recognition is still unvalidated.

The [focused-context follow-up](evaluation-context-20260908.md) improved weapon
family on three manually centered shot windows. Longer cross-chunk context still
missed piece control. Four additional spectator windows were correctly labeled
spectating, but all four still had incorrect pregame-lobby narration. Separate
shot evidence, build context, viewpoint, and scene state; these development
results do not establish automatic localization or justify default promotion.

An [automatic localization pilot](evaluation-localization-20260908.md) now proposes
events from complete saved chunks. It found four known own-player events and
automatically generated three shot windows with matching weapon-family outputs,
but missed the known spectator event. The seven-chunk cohort is development data,
not full-session validation. Keep it opt-in; no automatic keep/export decision or
capture reactivation follows.

## Evidence and limits

| Question | Historical evidence | Interpretation |
| --- | --- | --- |
| Useful representation? | Text roughly 79–81% across-fight accuracy; historical v1 score about 69% | Promising offline evidence, not an untouched test |
| Discard lesser siblings? | Around 60–64% within-fight across feature families | Keep siblings; grouping is presentation, not deletion |
| Higher resolution? | 1440px fixed fabricated weapon swaps; no clear ranking gain | Keep v3.3 for description fidelity |
| Transcripts? | Harmed measured ranking | Deprioritize transcription/semantic audio |
| Loudness? | Roughly one point overall; +0.237 keep correlation on saved clips | Cheap candidate, uncertain incremental benefit |
| Audio as live detector? | First session has only a few marks and was used to tune | Unproven; selected-clip correlation does not measure live misses |

The inspected store has **348 v3.3 descriptions and zero v2 preference rows**.
Human labels may exist in the separate legacy import. The v2 store alone cannot
support personalization. Re-indexing's cascading deletion bug is fixed; an empty
table does not prove it previously removed labels. No lost labels were reconstructed.

The likelier explanation for the empty table is simpler than a deletion bug:
until 2026-09-13 the store pointed at `C:\Users\Ethan\Videos\NVIDIAFortnite`, a
path from the machine it was built on, so 334 of 348 clips had no playable video
here. `review` is the only source of organic labels and it serves video from that
column, so no keep/skip decision could be recorded. The footage was restored and
`relink` repointed all 334, each confirmed by content hash. Every ranking number
reported so far therefore rests on imported v1 labels alone.

Replaying the available `session-1.csv` during this review found 19.4 minutes and
three clip marks. The +10dB sustained detector fired 15 times and matched two
marks; the periodic baseline also matched two but fired 46 times. This tiny,
tuned-on session is neither a matched-rate comparison nor proof of saved-event
recall. The sweep matches timestamps within a tolerance; it does not inspect
whether the saved video contains the event.

62% is not literally a coin flip: chance is 50%, and uncertainty needs measuring.
Thousands of pairs from hundreds of clips are dependent. Folds currently split
by inferred fight; session separation and confidence intervals remain needed.
Repeated experiments on the same folds make them validation data. Historical
v1 fixed scores use a different training/evaluation procedure from cross-validated
v2. Imported human decisions are not automatically weak because they were imported.

## Desktop Duplication

**The current recording setup failed gameplay acceptance and is paused.** The
September 7 DDA/Intel Quick Sync session saved valid files but caused FPS drops
and game crashes reported by the user. A game access violation during capture is
confirmed in local logs; root cause is unresolved. The old NVENC feasibility
result does not clear this hybrid-GPU laptop setup. See the
[unseen-session pilot](evaluation-20260907.md) for evidence and limits.

Desktop Duplication remains a candidate architecture. Microsoft documents GPU-accessible
DXGI surfaces; FFmpeg's `ddagrab` produces D3D11 hardware frames. The architecture
remains viable, but that does not establish its impact on a running game.
[Microsoft DDA](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api),
[FFmpeg ddagrab](https://ffmpeg.org/ffmpeg-filters.html#ddagrab).

WGC is relevant for per-window capture and HDR requirements, not an automatic
performance upgrade. Microsoft covers window/display selection and HDR formats;
relative overhead requires your own benchmark.
[Microsoft screen capture](https://learn.microsoft.com/en-us/windows/apps/develop/media-authoring-processing/screen-capture).

The notebook's 4.3% of one CPU core is feasibility evidence, not proof of negligible
impact. Measure game frame times, 1% lows, GPU/VRAM use, dropped frames, and A/V
drift. Exercise alt-tab, resolution changes, sleep/wake, and display loss. DDA
interfaces can become invalid and require recreation. The daemon reports failure
but automatic recovery is unfinished.
[Microsoft error contract](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-releaseframe).

## Audio priority

1. **Usable recorded audio is required before recorder release.** Preserve game
   and optional mic audio; separate tracks preserve future options. Measure sync.
2. **Keep intensity features experimental.** Compare loudness plus text against
   text alone on the same held-out clips. Loudness does not establish laughter,
   excitement, or even the recording player's action.
3. **Do not rely on audio alone to decide what footage survives.** Compare peak
   and sustained detectors to a periodic baseline at matched saves/hour on whole
   sessions, including quiet wanted events. Count warmup/cooldown misses. Tune
   on different sessions from the test. Add visual corroboration if measured
   misses justify it; do not rebuild v1's full classifier preemptively.

## Next work, in order

1. Restore acceptable game performance and stability before re-enabling the
   recorder. Benchmark against no capture; file validity is not this gate.
   Nothing measures this yet. `nvidia-smi` and `typeperf` are available for GPU
   and system counters; frame times and 1% lows need PresentMon or an equivalent,
   which is not installed. The A/B requires a live match with capture on and off.
2. Complete the frozen unseen-session review and fix spectator/own-player
   attribution. The 60-window pilot misidentified 14/18 spectator windows as
   lobby footage. It does not validate full-session highlight selection.
3. Close the review loop on multiple real sessions. Preserve imported label
   provenance and freeze untouched sessions before more tuning. Report preference
   accuracy and keeps in the top ten against recency/random baselines. Bootstrap
   uncertainty by session.
4. Validate capture independently while saved-file ingestion remains available.
   Compare at a fixed save rate and inspect false negatives carefully.
5. Complete recorder audio, display/device recovery, source selection, timestamp
   mapping into saved video, and pipeline handoff only within the measured
   performance budget. Audio transcription remains deprioritized.
6. If the inbox repeats one type of fight, test a modest diversity penalty and
   measure the top-ten tradeoff before making it the default.

Suggested pilot gates, set before collecting test results: five untouched
sessions and fifty verified wanted events; at least 90% recall at a chosen,
affordable saves/hour budget; less than 5% deterioration in game 1% lows;
less than 100ms A/V drift over an hour. These are proposed thresholds, not
achievements, and do not certify every hardware configuration.

## Debt and boundaries

This pass consolidates docs, packages v2, shares ranking, aligns defaults, removes
dead scan options, makes legacy imports explicit/read-only, preserves child
records on re-index, handles absent research features, and adds regression tests.
Cache identity includes the request. Recorder lifecycle, active-segment exclusion,
post-roll and nonblocking saves are hardened.

**The local student/teacher workflow is retired.** v1 is frozen: reference,
optional label imports, and the historical Docker demo only. The shared
utilities moved to the neutral `salience_media/` package, so v2 imports nothing
from `backend/` and deleting it is now a self-contained decision.

The 2026-09-11 debt pass additionally: replaced four copies of the frozen-runtime
check with `freeze.py`, which names the field that moved; moved `utc_now`/
`write_json` out of `capture/` into `jsonio.py` so the frozen studies no longer
import the paused capture subsystem; made `capture/windows.py` importable off
Windows so Linux CI can run; gave the `preferences` table one owner; aligned
`schema.MECHANICS` with the tags the prompt actually offers; gave the capture
window prompt its own storage label; and scored tied pairs at 0.5 instead of 0.

Remaining: ambiguous cloud billing after crashes; estimated dollar caps;
recorder-specific localization; prototype review server; embedding/preprocessing
versioning for rigorous comparisons; recorder audio, recovery, and timestamp
mapping. Automatic quarantine/deletion, new games, student retraining, and native
encoding are deferred.

**Orphaned producers.** The store holds 348 `audio_features` rows, 348
`peak_kind` rows, and 348 each of `image` and `audio-only` embeddings, and
`taste` reports 79-82% using them. The scripts that produced them are not in the
repo; `audio.py` and `embed.ClipImageEncoder` are the libraries but have no entry
point, and nothing here can regenerate `peak_kind` at all. Those measurements
cannot currently be reproduced from a clean checkout.

**Direction: the capture-chunk path is the target.** The 348 stored clips came
from third-party auto-capture, not this project's recorder. Own-gameplay
auto-clips are the interim corpus, not the destination, so spectator/actor
attribution is on the critical path -- it is deferred behind the recorder's
performance gate, not dismissed. Note that the saved-clip library contains no
spectator footage (all 348 are `event_source='fraction'`, and only 2 of 348
descriptions miscredit a kill to a teammate), so that failure is unmeasurable
until the recorder produces chunks again.

**First held-out weapon result (2026-09-13).** 50 randomly sampled clips,
blind-labelled via `groundtruth.py`: of 42 with a weapon to name, **7 wrong
(16.7%)**, 35 right at family level or better. "Exact" (12) measures how specific
the label was, not model accuracy, and should not be quoted. Two findings matter
more than the rate:

* **It cannot abstain.** On the 8 clips where nothing of the player's killed
  anything -- their own death, or an opponent bleeding out from an unseen knock
  -- it named a weapon on 7, including a Rare Shield Potion. One said unknown.
* **The error is systematic.** Claiming pistol it was never wrong (10/10);
  claiming shotgun it was wrong 5 of 13, and 4 of those were actually pistols.

Actor prevalence: 46 recording_player, 4 no_event, zero teammate or spectated --
the saved-clip library holds no spectator footage to get wrong.

**Annotation scope, measured rather than assumed.** The page seeks to
`event - 4s` and does not auto-pause, so the annotator saw 9.2s on average
against the model's 4.0s: the model's window sits wholly inside it, with 1.5s of
lead-in before and 3.7s of trailing aftermath after. The annotator did not review
whole clips. 16.7% is therefore the measured rate, not a ceiling inflated by
scope. A residual 1.5s lead-in could still hide the earlier kill of a multi-frag
from the model while the annotator saw it, so a few disagreements may not be
model errors -- but that is a small effect and the rate should not be discounted
for it. The page now bounds playback to the model's window with context opt-in,
so later passes have no gap at all.

**v3.5 result (2026-09-13): the prompt change halved the weapon error rate.**
Three arms on the same 42 scored clips from the frozen random sample:

| arm | wrong |
| --- | --- |
| v3.3, qwen3p7-plus (original corpus) | 7 / 42 = 16.7% |
| v3.3, qwen3p8-max | 6 / 42 = 14.3% |
| v3.5-killweapon, qwen3p8-max | **3 / 42 = 7.1%** |

The matched baseline separates the two causes: upgrading the model bought one
clip (16.7 -> 14.3%), redefining the weapon question bought three more
(14.3 -> 7.1%). v3.5 never wrongly abstained -- zero `model_declined` on clips
that had a real answer. Cost: $0.542 for the 30 calls that completed arm 2.

**v3.6-ownkills is the default (2026-09-13).** v3.5 asked for "each weapon that
eliminated or knocked someone", so when the recording player was the one killed
the model correctly named the weapon that killed *them* -- literal, and not what
the annotator meant. v3.6 scopes the list to the recording player's own kills.

| arm | wrong / 42 | deaths claiming a kill |
| --- | --- | --- |
| v3.6-ownkills · kimi-k2p6 | 1 (2.4%) | **0 / 5** |
| v3.5-killweapon · kimi-k2p6 | 1 (2.4%) | 3 / 5 |
| v3.3 · qwen3p7-plus | 7 (16.7%) | 4 / 5 |

The weapon rate is unchanged because scoping only affects clips with no ground
truth weapon, which that column excludes. The gain is the contradiction: a clip
whose outcome is "death" cannot also list a kill by the player. `report` now
checks that invariant, which needs **no labels at all** and so runs against any
future arm or corpus for free. The three remaining non-empty death-adjacent
answers are bleed-out clips where the player really did knock someone, so the
prompt and the labels now differ only where they genuinely disagree.

Matched ranking check on the same 46 labelled clips, 73 pairs: v3.3 83.6% vs
v3.6-era descriptions 86.3% pairwise. No regression, so re-describing the library
is safe whenever there is credit for it.

**Model bake-off (2026-09-13).** `qwen3p7-plus` was undeployed mid-project, so
the corpus model no longer exists. Five image-capable models were scored at v3.5
against the same labels; the ones with full coverage, all 42 scored:

| arm | wrong | 95% CI | 348 clips |
| --- | --- | --- | --- |
| v3.5 · kimi-k2p6 | 1 (2.4%) | [0.4%, 12.3%] | $2.97 |
| v3.5 · qwen3p8-max | 3 (7.1%) | [2.5%, 19.0%] | $6.15 |
| v3.3 · qwen3p8-max | 6 (14.3%) | [6.7%, 27.8%] | $6.15 |
| v3.5 · deepseek-v4p1-flash | 6 (14.3%) | [6.7%, 27.8%] | $0.68 |
| v3.3 · qwen3p7-plus | 7 (16.7%) | [8.3%, 30.6%] | $6.15 |

1 versus 3 on n=42 is **not** significant -- those intervals overlap almost
entirely. What the data supports is that v3.5 beats v3.3 on a fixed model, and
that kimi-k2p6 is no worse than qwen3p8-max at 52% of the price. Defaults are now
kimi-k2p6 + v3.5. `deepseek-v4p1-flash` gave the most exact names (17) but 6
wrong and 2 abstentions: cheaper, and it returns the whole v3.5 gain. glm-5p3-flash
and muse-glimmer-30b are thinking-only and delivered 7/10 and 3/10; both now have
their request quirks handled but neither is recommended.

**Two regressions were found and fixed by re-testing after changing things.**
Model-tagged storage variants were written by `storage_variant` but read back
with the bare prompt version, so configuring any non-corpus model made `review`,
`rank` and `taste` look for rows nothing had written; reads now go through
`prompt.active_variant`. Separately, `relink` moved every stored path, which
broke the join to the v1 labels -- they record the original machine's paths, so
a path-keyed lookup matched **0 of 348** and `rank` silently reported "not enough
labelled clips". `feedback.PreferenceIndex` now prefers an exact path and falls
back to a unique filename, counting the weaker matches. Ranking reproduces its
previous numbers exactly.

**Abstention improved only slightly, and the stated target was wrong.** On the 8
clips whose annotation says no weapon of the player's killed anything, v3.3 named
one on 7 and v3.5 on 6. The "7/8 -> 0/8" target assumed both sides asked the same
question; they do not. v3.5 asks for weapons that eliminated *or knocked*, while
the labels answer "the weapon that caused the kill", and on three of those clips
a knock genuinely happened in-window with a visible weapon. The clean subset is
the "i was eliminated" clips, where v3.5 still names a weapon on 3 of 5. Either
re-ask the annotator v3.5's question, or narrow the prompt back to eliminations.

**v3.5 answers with a list, not one weapon.** A single field was the wrong shape:
the frames can hold no kill, one, or several with different weapons, and the
recorder's own chunks pick their window at random with no event anchor, so
"the weapon that caused the outcome" has no single answer. `kill_weapons` is a
list and `[]` is how the model declines. Run, then superseded by v3.6.

**Capture debt audit, 2026-09-22.** The capture code grew fast across one
session and accumulated real debt; this pass removed it.

*Duplication.* `wgc.encoder_command` and `audio_pump.mux_command` were the same
builder differing only by audio inputs, so encode settings had to be kept in
step by hand -- exactly how `-g` drifted out of step with the original recorder
before. Now one builder, audio flags emitted only when there is audio.
`GAME_EXE` and the `QueryFullProcessImageNameW` lookup existed twice; process
identity now lives once in `windows.py` (`window_is_game`, `game_window`), and
`foreground_game` reuses it instead of repeating it.

*Dead code.* The pacer redesign left a `_writing` lock guarding nothing, an
unused `_thread`, and a `dropped` counter nothing incremented. `start_tracks`
in `audio_pump` was superseded by `WgcRecorder._audio_command` and referenced an
attribute that never existed. All removed. `dropped` was replaced by
`superseded` -- frames WGC delivered that were overwritten before the pacer
consumed them, which is real and measured at 7.5% on a live run, meaning the
source throttle is looser than the output rate.

*A crash, found by testing the refactor against real hardware rather than
trusting a green suite.* Starting a second WGC session in one process
**segfaults** once audio is involved: the native capture thread and
`soundcard`'s COM state do not both survive a restart. `CaptureControl.wait()`
was missing and adding it did not fix it. Since a segfault has no traceback and
"start recording, stop, start again" is an ordinary user action, the crash is
now converted into `WgcSessionReused` naming the fix: **one recording session
per process**, which `wgc_record.py` already does and which a launcher must do
by spawning a subprocess rather than hosting captures in-process. The root
cause inside the Rust extension is unresolved.

Two previously write-only attributes were made to do real work rather than
deleted: `_sinks` now closes pipe handles orphaned when FFmpeg fails to launch,
and `_closed` is surfaced as `source_closed`, which is how a window-targeted
recording learns the game exited.

218 v2 tests pass; the v1 backend suite is unchanged apart from two optional
`torchvision` imports.

**The recorder has audio, 2026-09-22.** `capture/audio_pump.py` records WASAPI
loopback (game) and the microphone as **two separate AAC tracks** alongside WGC
video, verified live: 48kHz stereo, zero dropped blocks, both non-silent. Not a
mix -- summing them makes the voice-versus-effects ranking question
unanswerable, which is exactly the state the mixed-stereo NVIDIA clips left the
corpus in.

Mechanics worth keeping: FFmpeg cannot inherit a `pipe:N` descriptor across the
Windows process boundary ("Bad file descriptor"), so each track is served as a
**named pipe** FFmpeg opens like a file. And `soundcard` raises
`RuntimeError: Error 0x100000001` when called from WGC's native callback
thread, so devices are resolved on the main thread in `start()`.

**A/V drift is now evaluable for the first time, and nearly passes.** Fixing it
uncovered two real bugs:

| | duration drift |
| --- | --- |
| audio started before the encoder existed | 877ms |
| + video declared 30fps but WGC delivers only on change | 191ms |
| + producers joined one at a time instead of all signalled first | **106ms** |

The middle fix is a **wall-clock pacer**: WGC emits a frame only when the
surface changes, so feeding those to a `-framerate 30` input makes the video
shorter than real time while audio runs at real time. The pacer writes exactly
`fps` frames per second, repeating the last one -- which is what `ddagrab` did
internally, and the origin of its "More than 1000 frames duplicated" notice. It
also moved all encoder I/O off the capture callback, so WGC can never be
back-pressured by a slow write.

The residual is mostly a fixed shutdown tail, not progressive drift: 10s run
127ms, 40s run 190ms -- 4x the duration for 1.5x the drift. **`start_offset_ms`
is 0.0 in every run**, so the streams are aligned where playback sync is
decided; the difference is at the tail. The gate measures duration difference,
which conflates the two, and at ~106ms it sits just over a 100ms limit.

**A coarse visual kill trigger does not work, measured 2026-09-23.** The idea
was that v1 rejected vision by measuring *full OCR* at 98% of runtime, whereas a
32x24 brightness grid is microseconds. Tested on 80 elimination clips whose
filenames give the label for free, the region **discovered** from data rather
than guessed.

First design compared the event frame against one 25% into the clip: max
Cohen's d 0.72, cells scattered across the screen. That design was wrong -- two
frames ten seconds apart differ mostly by scenery, so any banner signal drowns.

Second design was the fair one: event frame against a frame **1.5s earlier**,
same scene and camera, so the banner is close to the only difference, with the
same gap taken during a quiet stretch as the null. The effect got *weaker*:

* max |d| = **0.58**, one cell above 0.5, **none above 0.8**
* top cells span rows 6-22 -- a real HUD element would be a tight cluster

There is a faint hint of something (rows 6-10, columns 6-9 all darkening) but at
d ~= 0.4-0.5 it is not a detector. **Coarse brightness is rejected.**

One confound worth stating: `event_sec` is the `0.75 x duration` estimate, whose
median offset is +0.20s with 77% inside +/-1.0s. With a 1.5s gap, a clip whose
event is misplaced by more than about a second puts the banner in both frames or
neither, which would blunt a real signal. So this rejects *brightness at the
estimated event time*, and a sharper label or a colour/text-aware feature could
still differ. It does not rescue the approach as tested.

**Trigger signals: the game logs some of the answer for free (2026-09-23).**
Fortnite writes `FortniteGame.log` in real time, and reading it is completely
passive -- no injection, no memory access. `capture/game_log.py` parses it;
validated across three real session logs (5 match starts, 3 match ends, 5
deaths), timestamps UTC to the millisecond.

Confirmed available at default verbosity:

* `MatchState changed previous=X current=InProgress` -- match start
* `MatchState changed previous=InProgress current=...` -- match end
* `LogFortDeathCameraMode: Going to use death report` -- the local player's
  death, with world coordinates

**Verbose game logging is impossible, settled 2026-09-26.** `[Core.Log]` in
`Engine.ini` had no effect: across a full session the target categories moved
only as much as extra gameplay explains (`LogPlayerKills` 6 -> 26,
`LogAbilitySystem` 437 -> 574, `LogFortAbility` 9 -> **8**), and not one new line
*type* appeared -- the same `HandlePlayerStateChanged` and Warning-level lines as
before. Fortnite ships as an Unreal **Shipping** build, where Verbose and
VeryVerbose `UE_LOG` calls are compiled out of the binary; no configuration can
enable code that does not exist. The file was reverted. **The kill trigger cannot
come from the log.**

**First real detector measurement, 2026-09-26.** A 24.8-minute solo session with
4 keep marks, mic and input journalled alongside:

| detector | setting | fires | recall | 95% CI |
| --- | --- | --- | --- | --- |
| voice reaction | rise 4.0 dB, sustain 0.40s | 31 | **4/4** | 51-100% |
| input burst | 2.5x median, 4/s floor | 44 | 3/4 | 30-95% |
| metronome, 60s | -- | 24 | 2/4 | 15-85% |
| metronome, 30s | -- | 49 | 2/4 | 15-85% |

**Both beat no-skill**, which is the first time an audio detector in this project
has. Voice did it at *fewer* firings than the 30s metronome and double the
recall. The previously guessed defaults (8.0 dB) caught only 1 of 4, so the
defaults are now measured rather than chosen.

**Four marks cannot establish any of this.** The intervals overlap heavily, and
100% of four is four. What the session does establish is that the harness works
end to end -- levels, input rates and marks all line up and sweep offline -- so
the cost of the next session is only the playing.

One analysis bug worth recording: concatenating the session chunks with
`-c copy` silently dropped the **microphone** track, because concat carries only
the first stream of each type without `-map 0`. The recording was correct; the
first extraction reported an empty mic track and would have been read as a
recorder failure.

**Input activity covers the silent solo session (2026-09-26).** Voice detection
says nothing about someone playing alone without talking, which is probably the
most common session there is. `capture/input_activity.py` counts key-down
**transitions** per half-second window -- transitions, not held state, because
holding W across the map would otherwise outrank a build fight.

`InputBurstDetector` fires on a rise over the player's own rolling median, for
the same reason the voice detector does: a builder, a controller player and a
passive player have different normals and the product cannot ask anyone to
configure a threshold. Confidence 0.4, below an exact game event, because it is
the weakest inference of the three.

`GetAsyncKeyState` polling again -- no hook, no injection, no text recorded. The
watched set is 19 fixed game-relevant keys (movement, fire, aim, build, slots);
it deliberately excludes letters that spell words, so it cannot resemble a
keylogger even by accident.

The universal set is now voice **and** input, because they cover disjoint
sessions: voice is silent on solo play, input is silent on a funny moment where
nobody moves. Both journal per window, so a hard kill loses nothing.

**Thresholds are guesses and nothing is validated.** The recorder now writes
`input.tsv` alongside `marks.tsv`, so the next recorded session with F9 marks is
the dataset that judges both detectors against a metronome.

**Audio capture is per-device, not per-application (2026-09-26).** The game
track is WASAPI loopback on the default output, which carries **everything** the
machine plays: the game, Spotify, a podcast, a browser tab, Discord including
its music bots. A background podcast therefore lands in the "game" track and can
fire a loudness detector that has no way to know it was not gameplay.

The microphone track is unaffected, which is the part that matters most: voice
reaction detection reads the mic and is immune to background audio. The exposure
is loudness- and transcript-based detection on the *game* track.

Per-process capture is possible on this machine but not from Python. Windows
Application Loopback (`ActivateAudioInterfaceAsync` with
`AUDIOCLIENT_ACTIVATION_PARAMS`) needs build 20348+; this machine is 26200. It
is what OBS's "Application Audio Capture" uses. Neither `soundcard` nor
`pyaudiowpatch` exposes it -- both are device-level -- so isolating the game from
Spotify needs native code.

Two mitigations exist today. Windows can assign a different output device per
application, and this machine has three (Realtek Speakers, Realtek Headphones,
NVIDIA HD Audio), so the game can be routed to one device and music to another
with loopback taken from the game's. And because the mic is already separate,
the reaction signal -- the one the product is most interested in -- is already
clean. Neither is a substitute for per-process capture in a shipped product.

**What was said beats how loud it was -- but only for the right words
(2026-09-24).** Tested on the 334 labelled clips whose transcripts were already
stored, at zero cost:

| speech in the clip | clips | kept |
| --- | --- | --- |
| strict excitement ("oh my god", "no way", "let's go") | 21 | **42.9%** |
| laughter | 2 | 50.0% |
| silent | 204 | 36.3% |
| **baseline** | 334 | **31.4%** |
| filler only ("bro", "damn", "yo") | 73 | 20.5% |
| other speech | 32 | 18.8% |
| verbalised disinterest ("boring", "bad video") | 2 | 0.0% |

The direction **reverses** depending on the words. Filler tracks downtime -- you
chat while rotating -- and lands 11 points below baseline; genuine excitement
lands 11 above. A loose "reaction" pattern mixing both scored 19.7%, *worse*
than silence. That is precisely why embedding whole transcripts harmed ranking:
averaging two opposite effects destroys them.

`capture/speech.py` therefore classifies into separate features rather than
embedding, with a fixed inspectable vocabulary rather than a model -- 21
examples cannot train anything.

**Neither result is established.** The excitement lift is not significant
(z = 1.16, 95% CI 24.5-63.5%, n=21). More importantly the corpus is **entirely
auto-saved kills**, so a funny moment without a kill was never recorded: this
data can say that excitement correlates with keeping *among kill clips*, and
cannot say whether reaction-only moments are worth clipping at all. That is the
question the user actually wants answered, and it needs footage the recorder
has not yet captured.

Whisper (`faster-whisper`, already installed) belongs in a **cascade**: a
loudness gate decides when to listen, transcription runs only on those few
seconds. Running it continuously would put it in competition with the game for
the CPU that every capture measurement this month has been fighting over.

**The voice detector can now be tuned on recordings, not sessions
(2026-09-24).** `capture/levels.py` turns any audio track of any recording the
recorder has made into the dBFS series a detector replays against, and
`capture/tune_voice.py` sweeps thresholds over it, scores recall against marked
moments, and prints a metronome baseline.

This exists because the one audio result the project has was produced on a
single 19.4-minute session that was also used to choose the threshold, and it
still could not beat firing blindly on a timer. Every recording made from here
is a tuning dataset with no extra instrumentation.

Separate tracks are what make it meaningful. Measured on a real 2-track
recording: game audio median **-75.2 dBFS**, microphone median **-36.9 dBFS** --
a ~38 dB gap that a mixed track would have destroyed, and with it any ability to
say whether a reaction or an explosion fired the detector.

Scoring is deliberately asymmetric: a firing up to 12s *before* a mark counts,
up to 3s after does not, because the buffer holds the preceding seconds. Recall
is reported as undefined rather than zero when nothing was marked, so a firing
rate can never be mistaken for an accuracy.

Thresholds remain **unvalidated**. The defaults (8 dB rise, 0.4s sustain) are
guesses, and the only recordings with audio so far are 15-second desktop smoke
tests with no reactions in them. One real session with marks would settle it.

**Detection is split so the product is not a Fortnite tool (2026-09-24).**
The log is exact and free and does not generalise: Valorant writes no
equivalent, and no log finds the moment a friend says something funny on a call.
`capture/triggers.py` therefore separates:

* **Universal detectors** -- work in any title, observe the player and the
  machine rather than the game. `VoiceReactionDetector` fires on a sustained
  rise over the speaker's own rolling median, so a quiet talker and a loud one
  both work with no configuration.
* **Game adapters** -- a bonus layer. `GameEventDetector` reports Fortnite's
  exact death and match-end events at confidence 1.0; where a title exposes
  nothing, the universal detectors carry the session alone.

`TriggerBus` runs all of them under one shared cooldown, because two detectors
firing seconds apart describe the same moment and the buffer would hand back
overlapping footage -- the duplicate coverage `engagements.py` exists to clean up
after. An exact game event outranks an inference for the same moment.

Voice deserves a fair test now that it can have one. The earlier finding that
voice did not predict a keep came from `peak_is_voice`, which asked whether the
single *loudest* moment of a clip was a voice -- in a gunfight a shotgun beats a
person shouting, so it was structurally blind to exactly the clips where someone
reacts. Separate mic and game tracks now exist.

**Kill-trigger verbosity experiment is armed but unrun.** `[Core.Log]` was added
to `Saved/Config/WindowsClient/Engine.ini` with `LogPlayerKills`,
`LogFortAbility`, `LogAbilitySystem`, `LogFortGameStateAthena` and `LogAthenaHUD`
at Verbose; these are categories that exist in this build and sit plausibly on
the kill path. The original file is backed up in `.local-data/`. One match will
say whether any of them logs an elimination.

**The log is written continuously during play.** Measured over one 4.3-minute
match: **4,482 lines**, median gap between them 0.00s, 90th percentile 0.13s,
largest gap 3.4s, none over 10s. At session end the file's modification time
matched its final line's own timestamp to the second, so nothing sizeable was
sitting in a buffer.

That is strong but not conclusive: dense timestamps prove when the game
*recorded* an event, not when the line became *readable*. A buffered log would
look identical from the outside. The decisive test is watching the file grow
during play and measuring the delay between a line's timestamp and its
appearance, which `.local-data/tools/watch_game_log.py` does.

Either way the latency budget is generous. The recorder holds a rolling buffer,
so a trigger arriving even several seconds late still saves footage covering the
moment it names.

**`LogPlayerKills` is a spectator signal, not a kill signal.** The category name
is misleading and cost real investigation to settle. Counts vary per match (9,
5, 1.5), which ruled out UI initialisation; timing settled it -- a death at
19.22.58 is followed four seconds later by the same line carrying *another
player's* name. It fires when the kill-feed UI rebinds to whoever is being
watched.

That makes it valuable for a different problem. **Viewpoint is now exact and
free**: `local_player` infers the account name from the first view of a match
(never hardcoded, so it works for any user) and `spectating_spans` reports when
the view was on someone else. Validated on three logs: owner `BabyRico.YT`,
spectating 2, 1 and 0 other players respectively, non-ASCII names intact. This
is the attribution the pilot got wrong on **14 of 18 spectator windows**, and
the attribution arm spent paid calls trying to fix.

Raising log verbosity via `Engine.ini` was considered and proved unnecessary --
the needed signal was already there. The file was backed up, left unmodified,
and the backup removed.

**Still not available: your own eliminations.** No kill is written at default
verbosity, so the kill trigger -- the one most worth clipping -- remains
unsolved.

That reframes the trigger problem. `sweep.py` already states the right
objective: a firing is permission to spend work, not a commitment to keep, so
**recall matters and precision barely does**. Against that, the candidates are:

| signal | cost | status |
| --- | --- | --- |
| game log: match start/end, death | free, exact | **works today** |
| audio loudness (`trigger.py`) | free, live | built; on the one replayed session the peak detector could not beat a metronome |
| input activity (click/key rate) | free, passive | unexplored |
| cheap visual on the HUD region | local CPU | ablated in v1 as 98% of runtime, but that was full OCR, not a region check |
| VLM | seconds and cents per call | far too slow to trigger on |

The audio result is the one worth re-testing rather than trusting: it came from
a single 19.4-minute session with three marks that was also used to tune the
detector. The visual option deserves a fresh look because the v1 ablation
measured *full OCR*, not a narrow fixed-region check, and 348 auto-clipped clips
with known event positions are already on disk to test against for free.

**The frame-cost measurement is parked as unresolved, 2026-09-23.** Two further
paired runs (`--fps 15`, and `--fps 15 --scale 1280x800`) came back with capture
measuring *faster* than baseline in 2/2 and 1/2 scored rounds, which capture
cannot cause. Both were reported PASS: `paired_runs` computed its own verdict
and never applied the impossible-result guard already written into `compare`.
Fixed, with those exact runs as regression cases.

The deeper problem is that the method cannot resolve the effect. Nine baseline
1% lows recorded with **nothing capturing** span 27.35 to 95.76 fps -- a **71%**
spread -- because the 1% low is driven by the worst ~1% of frames, which depends
entirely on what the match was doing. Measuring a ~15% effect inside 71% noise
does not work at three 90-second rounds, which means **the 15.4% window-capture
figure should be treated as unconfirmed**: rounds 2 and 3 agreeing was as likely
coincidence as signal.

What would settle it is holding the content constant -- a fixed Creative or
lobby scene with the camera still, so the only difference between phases is
capture -- or phases long enough to average over scenarios (5+ minutes, 5+
rounds). The ordering of the three backends (DDA worst, WGC monitor better, WGC
window best) is consistent across every run, but only the DDA 55.2% result had a
margin large enough to exceed this noise floor.

**Window capture measured properly, 2026-09-23.** Three alternating
baseline/capture pairs, 90s each, PresentMon, one command (`bench paired`):

| round | baseline 1% low | capture 1% low | loss |
| --- | --- | --- | --- |
| 1 (warm-up) | 35.19 | 84.52 | -140% (impossible) |
| 2 | 88.92 | 76.32 | 14.2% |
| 3 | 95.76 | 79.96 | 16.5% |

Round 1 is a warm-up artifact, not capture behaviour: its baseline 1% low is
35.2 fps against 88.9 and 95.8 after it, and its worst frame 47ms against ~25ms,
while **mean fps barely moved (157 vs 163)** -- a mean-based gate would have
seen nothing. The first 90 seconds hold the drop, shader compilation and asset
streaming.

Discarding it is now a **rule declared in the code** (`--no-warmup` opts out),
not a choice made after seeing the numbers; scoring the run that way gives:

* **loss 15.4%** (median baseline 92.34 -> capture 78.14), gate 5%
* baseline spread **7.1%**, comfortably under the effect, so the pairing holds

Rounds 2 and 3 agree independently at 14.2% and 16.5%, which is what makes
15.4% believable where the earlier single pairs were not.

| backend | frame cost |
| --- | --- |
| Desktop Duplication | 55.2% |
| WGC, whole monitor | 23.8% |
| **WGC, game window** | **15.4%** |
| gate | 5% |

Each architectural step roughly halves the cost and none reaches the gate. The
remaining ~10 points are not obviously reachable by configuration: WGC still
composites through DWM and the frame still crosses into Python. Whether 5% is
the right threshold for a laptop under load is now a product decision with
numbers behind it rather than an open question.

**A window-capture run reported PASS and was wrong, 2026-09-22.** Baseline 1%
low 26.60, capture 35.16 -- capture measured 32% *faster* than no capture, on
every metric including p99 (34.0ms to 10.8ms). Capture cannot speed a game up,
so the two phases were not comparable.

The cause is the method, not the backend: in Fortnite the **match stage
dominates framerate**. A baseline recorded on the bus with 100 players and a
capture phase recorded late-game differ far more than any capture overhead.
Baseline 1% lows across three sessions on this machine were **37.28 and 26.60**
-- a 40% spread with nothing recording. That noise floor is wider than the
effect being measured, which makes single 2-minute pairings unreliable.

`bench.compare` now refuses such a pair: a capture faster than baseline by more
than `NOISE_TOLERANCE` reports `pass: null` with an explanation instead of a
pass. Writing that guard exposed a second flaw -- the verdict called a run
"pass" when one gate passed and the other was unevaluated. A pass now requires
**every** gate evaluated and passing; anything less is inconclusive.

So the 23.8% WGC figure stands as the best measurement, and `--window` remains
**unmeasured**. A trustworthy comparison needs either matched match stages or
several repeats per arm, not one pair.

**WGC more than halves the cost, and still fails, measured 2026-09-22.**

| backend | 1% low | loss vs baseline |
| --- | --- | --- |
| baseline (no capture) | 37.28 fps | -- |
| Desktop Duplication | 16.69 fps | **55.2%** |
| Windows Graphics Capture | 28.42 fps | **23.8%** |
| gate | | 5% |

`capture/wgc.py` captures one window instead of the whole desktop and throttles
at the source via `minimum_update_interval`, so a 30fps capture composites half
as many frames rather than encoding fewer. That removes over half the damage,
confirming the desktop-wide copy as the dominant cost. It needs no OBS -- the
`windows-capture` wheel bundles into a shipped build -- and never injects.

It is still nearly five times over the gate. Two levers are untested: `--window`
(this run captured the monitor, not the game window) and `--fps 15`. Neither is
guaranteed to close a 19-point gap on a DWM-based capture under laptop load.

Caveat on the pairing: the baseline is the 2026-09-16 recording, not a fresh one
from the same session. Same machine, same settings, same game, different match --
so 23.8% is approximately right rather than exact.

**The gate FAILED, measured 2026-09-16.** Paired 120-second Fortnite sessions,
same activity, PresentMon 2.5.1, with every 2026-09-14 encode fix already in
place (NVENC, 30fps, 900-frame GOP, no CBR pinning, no Quick Sync):

| | baseline | capture | change |
| --- | --- | --- | --- |
| mean fps | 158.26 | 141.98 | -10.3% |
| **1% low fps** | **37.28** | **16.69** | **-55.2%** |
| p95 frame ms | 7.95 | 8.73 | +9.8% |
| p99 frame ms | 10.47 | 33.17 | **+216.7%** |
| gpu util | 36.0% | 37.6% | +4.4% |
| power | 38.0W | 44.8W | +17.9% |

The gate allows 5% and the cost is 55%: **eleven times over**. The user's
subjective "FPS drops" report from September 7 is confirmed exactly.

Two things the numbers say that a mean would not. Mean fps fell only 10%, so
average framerate would have called this acceptable -- this is precisely why the
gate is on 1% lows. And GPU utilisation barely moved while p99 tripled and power
rose 18%: the GPU is not saturated, so this is **stalls**, not throughput. That
is the signature of DDA's per-frame desktop copy contending with the game's
present, plus NVENC's package power stealing clocks on a TGP-limited laptop.

**Configuration is exhausted.** Every cheap lever was already applied in this
run. The remaining cost is structural to Desktop Duplication, which copies the
whole desktop every frame regardless of what is done with the result.

Note the 514ms worst frame appears in *both* runs, so this machine has a
half-second hitch unrelated to capture. It does not affect the comparison.

**Anti-cheat is not why the project is on DDA.** Injection (swapchain hooking)
is correctly ruled out -- it is a ban risk for end users. But Windows Graphics
Capture is equally injection-free and captures one window rather than the whole
desktop. DDA was chosen because **FFmpeg exposes `ddagrab` and has no WGC
input**, so it was the only capture API reachable from Python without native
code. That is a tooling constraint, not a safety one. `capture/obs_client.py`
already routes around it: OBS implements Game Capture and WGC natively, is
GPL and redistributable, and is driven over websocket with no injection by us.
It needs `.[obs]`, and nothing calls it yet.

**The performance gate now has a harness (`capture/bench.py`).** It records one
labelled phase at a time -- play the same way with capture off, then on -- and
applies the gates to the pair. Frame pacing comes from a PresentMon trace, GPU /
VRAM / power from nvidia-smi on a sampling thread, A/V drift from ffprobe. A gate
with no data reports `null` and the verdict is `inconclusive`; it can never pass
by default, which is how "the files were valid" once stood in for acceptance.
`one_percent_low_fps` is the mean of the slowest 1% of frames as a rate, because
mean FPS can sit unchanged while 1% lows collapse -- the shape of the original
complaint.

Gate status measured 2026-09-13, not assumed:

* **1% low frame time: not yet evaluable.** PresentMon is not installed. It is
  the only missing prerequisite; the harness picks it up from PATH automatically.
  Needs one paired gaming session.
* **A/V drift: not applicable.** All **194** chunks of the September 7 session
  (95.8 minutes) carry a video track and no audio track at all. The gate cannot
  pass on silence and the harness reports it as unevaluated.

**What recorder audio would take, assessed 2026-09-13.** FFmpeg alone cannot do
it on this machine: `-f dshow` lists only two microphones (HyperX SoloCast, Intel
Smart Sound array) and no loopback or Stereo Mix device, and this build has no
`wasapi` input. Game audio therefore has to come from `soundcard`, which is
already a declared `capture` extra and installed, and which does WASAPI loopback
natively. Muxing it is the awkward part: `collection.CaptureRun` finalizes a run
by writing `q` to FFmpeg's **stdin**, so stdin is unavailable for a PCM pipe.
Getting audio into the same segmented output -- which is what keeps it in sync --
means feeding it through a named pipe and replacing the stdin finalize path.
Optional mic can come from dshow directly and map to a second track, which is
what "separate tracks preserve future options" asks for. Note `soundfile` is not
installed, so `audio.py`'s reaction features cannot run today either.

**Order matters here.** Audio capture costs CPU and adds a second real-time
producer to a recorder that already crashed the game, so it should be added
*after* the frame-time gate has a baseline, not before -- otherwise the next
measurement cannot separate the two causes.

**Build-mode footage exists but is mis-sampled.** The September 7 session is
build-mode play, and its 22 targeted windows mention building, editing or piece
control in **59%** of summaries ("builds upward to gain a height advantage",
"shooting an opponent through wooden walls"). Its 60 *uniformly sampled* windows
are 54/60 `no_event`, and their 27% build mentions are mostly scenery and
traversal. So the phenomenon is present and random sampling misses it: reaching
~30 build fights that way needs roughly 300 windows, about $2.55. The cheap
sequence is to label build mechanics on the 22 already-described windows for
nothing, then spend ~$0.19 on a tagged arm over the same 22 to answer whether the
model can tag it at all. That is development data, hand-picked, and cannot
validate anything. More build footage needs a working recorder first.

**Build/edit mechanics are unclaimable by default.** `editing`, `edit_reset` and
`piece_control` appear only in the attribution variant's prompt, so the default
v3.3 path cannot emit them -- zero occurrences across all 1356 descriptions.
Priority 2 below needs a new prompt version, and re-describing at a new version
costs new paid calls; the v3.3 prompt text is now digest-locked by a test so it
cannot be edited by accident.
