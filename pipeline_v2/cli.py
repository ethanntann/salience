"""Salience v2 command interface. Run python -m pipeline_v2 --help."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
import shutil
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_v2 import cache  # noqa: E402
from pipeline_v2 import config  # noqa: E402
from pipeline_v2 import embed  # noqa: E402
from pipeline_v2 import engagements  # noqa: E402
from pipeline_v2 import facts as facts_mod  # noqa: E402
from pipeline_v2 import feedback as feedback_mod  # noqa: E402
from pipeline_v2 import frames as frames_mod  # noqa: E402
from pipeline_v2 import prompt as prompt_mod  # noqa: E402
from pipeline_v2 import review as review_mod  # noqa: E402
from pipeline_v2 import schema  # noqa: E402
from pipeline_v2 import store  # noqa: E402
from pipeline_v2 import taste as taste_mod  # noqa: E402
from pipeline_v2 import vlm  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}


def _clips(target: Path, limit: int | None) -> list[Path]:
    def ready(path: Path) -> bool:
        return (
            not (path.parent / "capture-run.json").exists()
            or path.with_suffix(".capture.json").exists()
        )

    if target.is_file():
        if not ready(target):
            raise ValueError(f"Capture chunk is not finalized: {target}")
        return [target]
    found = sorted(
        p for p in target.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES and ready(p)
    )
    return found[:limit] if limit else found


def _engagement_primaries(clips: list[Path]) -> tuple[list[Path], int]:
    """Collapse clips covering the same fight down to one clip each."""
    groups = engagements.group(clips, event_of=facts_mod.event_from_filename)
    primaries = [g.primary for g in groups if g.primary is not None]
    skipped = sum(len(g.duplicates) for g in groups)
    return primaries, skipped


def _prepare(clip: Path):
    """Everything before the model: probe, place the event, cut seven frames.

    No decoding of a scan pass and no OCR. Ablation showed both cost 98% of
    local runtime and bought no ranking accuracy.
    """
    duration = frames_mod.probe_duration(clip)
    clip_facts = facts_mod.gather(clip, duration)
    timing = {}
    if clip_facts.analysis_start_sec is not None:
        timing = {
            "pre_sec": clip_facts.event_sec - clip_facts.analysis_start_sec,
            "post_sec": clip_facts.analysis_end_sec - clip_facts.event_sec,
        }
    window = frames_mod.window_frames(
        clip,
        duration,
        clip_facts.event_sec,
        out_dir=config.FRAME_DIR / cache.clip_sha(clip) / "window",
        **timing,
    )
    return duration, clip_facts, window


def cmd_localize(args: argparse.Namespace) -> int:
    """Inspect the estimated event and extract its window without a cloud call."""
    for clip in _clips(Path(args.target), args.limit):
        duration, clip_facts, window = _prepare(clip)
        print("\n" + clip.name)
        print(f"  duration        {duration:.1f}s" if duration else "  duration    ?")
        print(f"  filename event  {clip_facts.source_event}")
        print(
            f"  event at        {clip_facts.event_sec:.2f}s ({clip_facts.event_source})"
        )
        print(f"  window frames   {len(window)}")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    cfg = config.load_config()

    clips = _clips(Path(args.target), args.limit)
    if args.dedupe:
        clips, skipped = _engagement_primaries(clips)
        if skipped:
            print(
                f"{skipped} clip(s) cover a fight already represented; "
                "describing one per engagement (omit --dedupe to retain all clips)\n"
            )
    calls_left = args.max_calls

    for clip in clips:
        _, clip_facts, window = _prepare(clip)
        if not window:
            print(f"{clip.name}: no frames extracted, skipping")
            continue
        try:
            result = vlm.describe(
                cfg,
                clip,
                clip_facts,
                [f.path for f in window],
                allow_call=calls_left > 0,
            )
        except vlm.BudgetExceeded as exc:
            print(f"\nstopped: {exc}")
            break
        if not result.cached:
            calls_left -= 1

        description = result.description
        print(f"\n{clip.name}")
        print(f"  event at   {clip_facts.event_sec:.2f}s ({clip_facts.event_source})")
        print(f"  summary    {description.summary}")
        print(
            f"  outcome    {description.outcome}"
            f" | weapon {description.weapon_name or '?'}"
            f" | health {description.player_health_state}"
        )
        print(f"  mechanics  {', '.join(description.mechanics) or '-'}")
        print(f"  conflict   {description.hud_conflict or '-'}")
        print(
            f"  tokens     in {result.input_tokens} out {result.output_tokens}"
            f" | cost ${result.cost_usd:.5f}"
            + ("  (cached, free)" if result.cached else "")
        )
        if args.json:
            print("  json       " + json.dumps(description.model_dump()))

    summary = cache.spend_summary()
    print(
        f"\ntotal: {summary['calls']} paid calls, "
        f"{summary['input_tokens']} in / {summary['output_tokens']} out tokens, "
        f"${summary['spent']:.4f} of ${cfg.budget_usd:.2f}"
    )
    if summary["calls"] and not summary["spent"]:
        print(
            "cost shows $0 because prices are unset -- put the current per-Mtok "
            "prices in V2_INPUT_PRICE_PER_MTOK / V2_OUTPUT_PRICE_PER_MTOK. "
            "Token counts above are real."
        )
    return 0


# Calls in flight at once. One at a time, with a 1.5s pause after each, was
# 4.6s per window: about 20 minutes for a 40-minute session. The pause was
# there because back-to-back calls tripped Cloudflare's 1010 block, which is
# the block for a client signature (urllib's default User-Agent), not for
# rate; `vlm` now sends its own, and a real rate limit comes back as a 429,
# which `vlm` already retries with backoff.
WORKERS = int(os.environ.get("V2_WORKERS", "16") or 16)


def describe_many(cfg, clips: list[Path], max_calls: int | None, workers: int = WORKERS):
    """Yield (clip, outcome) for each clip, in order, describing `workers` at once.

    `outcome` is (duration, facts, VlmResult), None when no frames could be cut,
    or the exception a clip raised (`BudgetExceeded`, or a parse error). Each
    call is budgeted before it starts, so at most `max_calls` are ever paid.
    Callers store as results arrive, so a run that dies keeps what it got.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    lock = threading.Lock()
    left = [max_calls if max_calls is not None else len(clips)]
    noted = getattr(cfg, "prompt_version", None) in schema.NOTES_PROMPT_VERSIONS

    def one(clip: Path):
        duration, clip_facts, window = _prepare(clip)
        if not window:
            return None
        try:
            return described(clip, duration, clip_facts, window)
        finally:
            # The frames exist only to be sent; kept, they were 2.8 GB of JPEGs
            # (2026-10-23) that every later describe extracts again anyway.
            frames_dir = Path(window[0].path).parent.parent
            if frames_dir.parent == config.FRAME_DIR:
                shutil.rmtree(frames_dir, ignore_errors=True)

    def described(clip: Path, duration, clip_facts, window):
        if noted:
            from dataclasses import replace

            from pipeline_v2 import notes as notes_mod

            variant = prompt_mod.storage_variant(cfg.prompt_version, clip_facts, cfg.model)
            with closing(store.connect()) as conn:
                clip_facts = replace(clip_facts, notes=notes_mod.window_notes(
                    conn, clip, cache.clip_sha(clip), variant))
        frames = [f.path for f in window]
        try:
            # A cached answer costs nothing, so it never touches the allowance.
            return duration, clip_facts, vlm.describe(
                cfg, clip, clip_facts, frames, allow_call=False)
        except vlm.BudgetExceeded:
            pass
        except (ValueError, TypeError) as exc:
            return exc
        with lock:
            allowed = left[0] > 0
            left[0] -= allowed
        try:
            return duration, clip_facts, vlm.describe(
                cfg, clip, clip_facts, frames, allow_call=allowed)
        except (vlm.BudgetExceeded, ValueError, TypeError) as exc:
            return exc

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        yield from zip(clips, pool.map(one, clips))
    finally:
        # A caller that stops early (the recorder stopping) waits for the calls
        # in flight, never for the queue behind them.
        pool.shutdown(wait=True, cancel_futures=True)


def anonymised(text: str, player: str | None) -> str:
    """`text` with the player's name written as "the recording player".

    v3.8 tells the model the player's name so it can tell their kill-feed
    lines from everyone else's, and it then uses the name in summaries. The
    taste model learns from words: a name present in one session's training
    text and absent from another's is a difference between sessions, not
    between moments.
    """
    if not player or player not in text:
        return text
    import re

    text = re.sub(rf"(^|(?<=[.!?]\s)){re.escape(player)}", "The recording player", text)
    return text.replace(player, "the recording player")


def store_described(conn, cfg, clip: Path, duration, clip_facts, result) -> str:
    """Store one described clip and commit; returns the variant it landed in."""
    engagement_at = engagements.clip_time(clip)
    variant = prompt_mod.storage_variant(cfg.prompt_version, clip_facts, cfg.model)
    store.upsert_clip(
        conn,
        clip_sha=cache.clip_sha(clip),
        path=clip,
        engagement=engagement_at.isoformat() if engagement_at else None,
        duration_sec=duration,
        event_sec=clip_facts.event_sec,
        event_source=clip_facts.event_source,
        facts={
            "source_event": clip_facts.source_event,
            "capture_session": clip_facts.capture_session,
            "analysis_start_sec": clip_facts.analysis_start_sec,
            "analysis_end_sec": clip_facts.analysis_end_sec,
        },
        summary=anonymised(result.description.embed_text, clip_facts.player_name),
        payload=result.description.model_dump(),
        prompt_version=variant,
    )
    # Commit per clip, not per batch. A 150-clip run inside one transaction
    # holds a write lock throughout and loses everything if it dies partway.
    conn.commit()
    return variant


def cmd_index(args: argparse.Namespace) -> int:
    """Describe clips (cache-first), store them, then rebuild embeddings."""
    # Without this the command that describes the whole library is the one
    # recording every call as $0, so spend under-reports and the budget cannot
    # stop anything. The study modules already did this; index did not.
    cfg = config.resolve(config.load_config())

    if args.from_plan:
        # Describe exactly the clips a frozen study sampled, so a new prompt arm
        # is scored on the same corpus its labels cover and nothing else is paid
        # for. Paths come from the store, so a relink cannot strand the plan.
        from pipeline_v2 import groundtruth
        from pipeline_v2.jsonio import read_json

        plan = read_json(Path(args.from_plan))
        clips = list(groundtruth.current_paths(plan).values())
        if args.limit:
            clips = clips[: args.limit]
        skipped = 0
        print(
            f"re-describing {len(clips)} clips from {Path(args.from_plan).parent.name} "
            f"at {cfg.prompt_version}"
        )
    elif args.restore_stored:
        # Re-describe exactly the clips already in the store. An ablation must
        # run on the same corpus as the thing it is compared against, and any
        # sampler would drift from it.
        conn0 = store.connect()
        clips = [
            Path(row["path"])
            for row in conn0.execute("select path from clips order by path")
        ]
        conn0.close()
        skipped = 0
        print(f"re-describing {len(clips)} stored clips at {cfg.prompt_version}")
    elif args.from_feedback:
        prefs = feedback_mod.load(Path(args.from_feedback))
        if args.split_fights:
            pilot = feedback_mod.conflict_engagement_clips(
                prefs,
                group_fn=engagements.group,
                event_of=facts_mod.event_from_filename,
            )[: args.limit or 200]
        else:
            pilot = feedback_mod.pilot_sample(prefs, limit=args.limit or 30)
        clips, skipped = [p.path for p in pilot], 0
        print(
            f"pilot: {len(pilot)} labelled clips "
            f"({sum(p.label for p in pilot)} liked, "
            f"{sum(p.organic for p in pilot)} organic)"
        )
    else:
        clips, skipped = (_clips(Path(args.target), args.limit), 0)
    if getattr(args, "only_new", False):
        # Looking up a cached answer still extracts its frames (they are part
        # of the cache key): 300s on a 235-window session that needed nothing.
        with closing(store.connect()) as conn0:
            described = {r[0] for r in conn0.execute("select clip_sha from descriptions")}
        clips = [c for c in clips if cache.clip_sha(c) not in described]
    if skipped:
        print(f"{skipped} duplicate clip(s) skipped\n")

    # A run can describe auto-clipped files and capture windows together; they
    # are different prompts and must be embedded as separate corpora.
    written: set[str] = set()
    unparsed: list[str] = []
    conn = store.connect()
    try:
        for clip, outcome in describe_many(
            cfg, clips, args.max_calls, getattr(args, "workers", None) or WORKERS
        ):
            if outcome is None:
                continue
            if isinstance(outcome, vlm.BudgetExceeded):
                print(f"stopped: {outcome}")
                break
            if isinstance(outcome, (ValueError, TypeError)):
                # A malformed or unexpected response is one clip's problem. It
                # once aborted a 50-clip run at clip 20, losing the embedding
                # step and leaving 30 clips undescribed. The raw text is already
                # cached, so a later retry costs nothing.
                unparsed.append(clip.name)
                print(f"  unparsed {clip.name}: {type(outcome).__name__} {outcome}")
                continue
            duration, clip_facts, result = outcome
            written.add(store_described(conn, cfg, clip, duration, clip_facts, result))
            print(f"  stored {clip.name}" + ("" if result.cached else "  (paid)"))

    finally:
        conn.close()

    if unparsed:
        print(
            f"\n{len(unparsed)} response(s) could not be parsed and were skipped; "
            "their raw text is cached, so re-running costs no new calls:"
        )
        for name in unparsed[:10]:
            print(f"  {name}")

    status = 0
    for variant in sorted(written) or [cfg.prompt_version]:
        status |= _rebuild_embeddings(args.encoder, variant)
    return status


def _rebuild_embeddings(
    prefer: str | None, variant: str = config.DEFAULT_PROMPT_VERSION
) -> int:
    """Re-embed every stored summary at once.

    All vectors are rebuilt together rather than appended one at a time, because
    the TF-IDF baseline is corpus-dependent: adding a clip changes every other
    clip's vector, and mixing generations would silently corrupt similarity.
    """
    conn = store.connect()
    with conn:
        clips = store.all_clips(conn, variant)
        if not clips:
            print(f"nothing stored at {variant}")
            return 0
        encoder = embed.load_encoder(prefer)
        vectors = encoder.fit_transform([clip.summary for clip in clips])
        for clip, vector in zip(clips, vectors, strict=True):
            store.put_embedding(conn, clip.clip_sha, encoder.name, vector, variant)
        spoken = _rebuild_speech_embeddings(conn)
    print(f"\nembedded {len(clips)} clips with {encoder.name} ({len(vectors[0])} dims)"
          + (f", speech in {spoken}" if spoken else ""))
    return 0


def _rebuild_speech_embeddings(conn) -> int:
    """What was said, as its own vectors -- never mixed into the description's.

    Appending transcripts to the description text was measured on session 1:
    held-out taste fell from 92.6% to 83.3%, because the player's callouts
    drowned out what the picture said. Kept apart, `ranking` can weight speech
    by how much it actually predicts this player's ratings, down to zero.
    """
    from pipeline_v2.ranking import SPEECH_VARIANT

    said = store.transcripts(conn)
    conn.execute("delete from embeddings where prompt_ver = ?", (SPEECH_VARIANT,))
    if not said:
        return 0
    encoder = embed.TfidfEncoder()
    shas = sorted(said)
    for sha, vector in zip(shas, encoder.fit_transform([said[s] for s in shas]), strict=True):
        store.put_embedding(conn, sha, encoder.name, vector, SPEECH_VARIANT)
    return len(shas)


def cmd_similar(args: argparse.Namespace) -> int:
    """Nearest neighbours by summary. The M3 sanity check, free of API calls.

    If neighbours are not intuitively similar, the summaries do not separate
    clips and no taste model built on them will work either.
    """
    variant = prompt_mod.active_variant(config.load_config())
    conn = store.connect()
    clips = store.all_clips(conn, variant)
    vectors = store.embeddings(conn, variant)
    if len(clips) < 2:
        print("need at least two indexed clips")
        return 1

    for clip in clips:
        if args.query and args.query.lower() not in clip.path.name.lower():
            continue
        mine = vectors.get(clip.clip_sha)
        if mine is None:
            continue
        scored = sorted(
            (
                (embed.cosine(mine, vectors[other.clip_sha]), other)
                for other in clips
                if other.clip_sha != clip.clip_sha and other.clip_sha in vectors
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        print(f"\n{clip.path.name}")
        print(f'  "{clip.summary}"')
        for score, other in scored[: args.top]:
            print(f"  {score:.3f}  {other.summary}")
    return 0


def cmd_taste(args: argparse.Namespace) -> int:
    """Pairwise accuracy, three ways. The number that judges v2. No API calls."""
    from pipeline_v2 import schema

    prefs = feedback_mod.PreferenceIndex(
        feedback_mod.load(Path(args.db) if args.db else None, require_media=False)
    )
    conn = store.connect()
    # Labels recorded in v2's own review page override anything imported from
    # v1. They are the only decisions made while actually watching the clip,
    # and they are the labels the whole metric is waiting on.
    own = {
        row["clip_sha"]: int(row["label"])
        for row in conn.execute("select clip_sha, label from preferences")
    }
    stored = store.all_clips(conn, args.variant)
    # Descriptions and embeddings are keyed separately: an embedding variant can
    # be built from the same descriptions plus extra text (a transcript), so it
    # must not be looked up as if it were its own description set.
    vectors = store.embeddings(conn, args.embeddings or args.variant)
    images = store.embeddings(conn, "image")
    audios = store.embeddings(conn, "audio-only")
    peak_splits = {}
    for row in store.optional_rows(conn, "peak_kind"):
        scaled = min(float(row["peak_ratio"]), 40.0) / 40.0
        voice = int(row["peak_is_voice"])
        peak_splits[row["clip_sha"]] = [
            scaled if voice else 0.0,  # loudness when a voice caused it
            0.0 if voice else scaled,  # loudness when an effect caused it
            float(row["voice_frac"]),
        ]

    reactions = {}
    for row in store.optional_rows(conn, "audio_features"):
        reactions[row["clip_sha"]] = [
            min(float(row["peak_ratio"]), 40.0) / 40.0,
            float(row["loud_fraction"]),
            1.0 if (row["transcript"] or "").strip() else 0.0,
        ]
    if args.same_clips_as:
        shared = {c.clip_sha for c in store.all_clips(conn, args.same_clips_as)}
        stored = [c for c in stored if c.clip_sha in shared]

    # Group by fight, not by folder. An earlier version keyed on the parent
    # directory, which made every clip the "same engagement" and silently
    # rejected every pair -- a zero-pair result rather than a wrong one, which
    # is at least a loud failure.
    groups = engagements.group(
        [clip.path for clip in stored], event_of=facts_mod.event_from_filename
    )
    engagement_of = {
        clip: f"eng{index}"
        for index, group in enumerate(groups)
        for clip in group.clips
    }

    v1_scores_by_name = feedback_mod.v1_final_scores(Path(args.db) if args.db else None)
    v1_scores: dict[str, float] = {}

    examples = []
    for clip in stored:
        pref = prefs.get(clip.path)
        vector = vectors.get(clip.clip_sha)
        own_label = own.get(clip.clip_sha)
        if vector is None or (pref is None and own_label is None):
            continue
        label = own_label if own_label is not None else pref.label
        weight = 1.0 if own_label is not None else pref.weight
        examples.append(
            taste_mod.Example(
                clip_sha=clip.clip_sha,
                engagement=clip.facts.get("capture_session")
                or engagement_of.get(clip.path, clip.clip_sha),
                label=label,
                weight=weight,
                fields=taste_mod.field_vector(
                    clip.payload, clip.facts, schema.mechanics_for(args.variant)
                ),
                embedding=vector,
                image=images.get(clip.clip_sha),
                audio=audios.get(clip.clip_sha),
                reaction=reactions.get(clip.clip_sha),
                peak_split=peak_splits.get(clip.clip_sha),
            )
        )
        if clip.path.name in v1_scores_by_name:
            v1_scores[clip.clip_sha] = v1_scores_by_name[clip.path.name]

    if args.voiced_only:
        voiced = {
            row["clip_sha"]
            for row in store.optional_rows(conn, "audio_features")
            if (row["transcript"] or "").strip()
        }
        examples = [e for e in examples if e.clip_sha in voiced]

    if args.organic_only:
        organic_names = {
            p.path.name
            for p in feedback_mod.load(
                Path(args.db) if args.db else None, require_media=False
            )
            if p.organic
        }
        keep = set(own) | {
            c.clip_sha for c in stored if c.path.name in organic_names
        }
        examples = [e for e in examples if e.clip_sha in keep]

    liked = sum(e.label for e in examples)
    print(
        f"labelled clips indexed: {len(examples)} ({liked} liked, {len(examples) - liked} not)"
    )
    if liked < 2 or len(examples) - liked < 2:
        print("need at least two of each class to measure anything")
        return 1

    modes = (
        "embedding",
        "image+text",
        "image+text+reaction",
        "peak_split",
        "text+peak_split",
        "image+text+peak_split",
    )
    # Widen to the longest mode name rather than a fixed 12, which the
    # combined modes overflowed and pushed the numeric columns out of line.
    width = max(len("features"), *(len(mode) for mode in modes)) + 2
    print("\n" + f"{'features':{width}}{'pairwise acc':>14}{'pairs':>8}")
    for mode in modes:
        if not all(taste_mod.has_features(e, mode) for e in examples):
            print(f"{mode}: skipped (missing features; compare on a shared corpus)")
            continue
        accuracy, total = taste_mod.pairwise_accuracy(
            examples,
            mode,
            folds=args.folds,
            within=args.within_fight,
            model=args.model,
        )
        print(f"{mode:{width}}{accuracy:>13.1%}{total:>8}")
    accuracy, total = taste_mod.fixed_score_accuracy(examples, v1_scores)
    print(f"{'v1 score':{width}}{accuracy:>13.1%}{total:>8}   (shipped ranking, untrained)")
    print("\nchance = 50.0%. Below that is worse than shuffling.")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    """Use the same ranking path as the browser inbox."""
    from pipeline_v2.ranking import ranked_fights

    with closing(store.connect()) as conn:
        fights, labels = ranked_fights(
            conn,
            db_path=Path(args.db) if args.db else None,
            variant=args.variant,
            mode=args.mode,
        )
    if not fights:
        # Two different failures used to print this one line. An empty variant
        # (a typo, or capture-window clips whose variant carries a `-window`
        # suffix that reads deliberately omit) looked identical to a populated
        # variant with nothing labelled yet -- and the first one wasted an
        # afternoon once already, when relinked paths silently matched 0 of 348.
        wanted = args.variant or prompt_mod.active_variant(config.load_config())
        with closing(store.connect()) as conn:
            present = dict(
                conn.execute(
                    "select prompt_ver, count(*) from descriptions group by 1"
                )
            )
        if not present.get(wanted):
            print(f"no clips described in variant {wanted!r}.")
            near = sorted(
                (v for v in present if wanted.split("@")[0] in v or v in wanted),
                key=lambda v: -present[v],
            )
            for name in (near or sorted(present, key=lambda v: -present[v])[:5]):
                print(f"  {present[name]:5d}  {name}")
            print("Pass --variant with one of these.")
        else:
            print(
                f"{present[wanted]} clip(s) described in {wanted!r} but "
                f"{labels} are labelled; ranking needs labelled examples to "
                "train on."
            )
        return 1
    print(f"{len(fights)} fights, trained on {labels} labels ({args.mode})\n")
    for index, fight in enumerate(fights[: args.top], start=1):
        print(f"{index:3d}. [{fight.score:+.2f}] {fight.best.summary}")
        print(f"      {fight.best.path.name}")
        for sibling in fight.siblings:
            print(f"      + sibling  {sibling.path.name}")
    return 0


def cmd_engagements(args: argparse.Namespace) -> int:
    """Show which clips cover the same fight. Free -- no API calls."""
    clips = _clips(Path(args.target), args.limit)
    groups = engagements.group(clips, event_of=facts_mod.event_from_filename)
    duplicates = sum(len(g.duplicates) for g in groups)
    for index, engagement in enumerate(groups, start=1):
        print(f"\nengagement {index}")
        for clip in engagement.clips:
            marker = "*" if clip == engagement.primary else " "
            stamp = engagements.clip_time(clip)
            print(
                f"  {marker} {stamp:%H:%M:%S}  {clip.name}"
                if stamp
                else f"  {marker} {clip.name}"
            )
    print(
        f"\n{len(clips)} clips -> {len(groups)} engagements "
        f"({duplicates} duplicate, {duplicates / len(clips):.0%} saved)"
        if clips
        else "\nno clips"
    )
    return 0


def relink_plan(conn, root: Path, *, verify: bool = True, progress=None) -> dict:
    """Match store rows whose video is gone to files under ``root``.

    Matching is by filename, but a filename is not identity -- the store is keyed
    by content hash, and attaching 348 descriptions to the wrong videos would be
    invisible until someone watched one. So a candidate is confirmed by hashing
    it, and anything that disagrees is reported rather than written.
    """
    rows = [
        (row["clip_sha"], Path(row["path"]))
        for row in conn.execute("select clip_sha, path from clips order by path")
    ]
    missing = [(sha, path) for sha, path in rows if not path.exists()]

    candidates: dict[str, list[Path]] = {}
    for found in root.rglob("*.mp4"):
        candidates.setdefault(found.name, []).append(found)

    relinked, ambiguous, absent, mismatched = [], [], [], []
    for index, (sha, path) in enumerate(missing, start=1):
        options = candidates.get(path.name, [])
        if not options:
            absent.append(path.name)
            continue
        if len(options) > 1:
            ambiguous.append(path.name)
            continue
        if progress:
            progress(index, len(missing), path.name)
        if verify and cache.clip_sha(options[0]) != sha:
            mismatched.append(path.name)
            continue
        relinked.append((sha, options[0]))

    return {
        "present": len(rows) - len(missing),
        "missing": len(missing),
        "relinked": relinked,
        "ambiguous": ambiguous,
        "absent": absent,
        "mismatched": mismatched,
    }


def cmd_relink(args: argparse.Namespace) -> int:
    """Repoint the store at moved footage. Dry run unless --apply is given."""
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1

    def progress(index: int, total: int, name: str) -> None:
        if index == 1 or index % 25 == 0 or index == total:
            print(f"  verifying {index}/{total}  {name}", flush=True)

    with closing(store.connect()) as conn:
        if args.verify:
            print(f"hashing candidates under {root} to confirm identity")
        plan = relink_plan(
            conn, root, verify=args.verify, progress=progress if args.verify else None
        )
        print(f"\n{plan['present']} clip(s) already resolve; {plan['missing']} missing")
        print(f"  matched and confirmed  {len(plan['relinked'])}")
        for label, key in (
            ("no file of that name", "absent"),
            ("several files match  ", "ambiguous"),
            ("content differs      ", "mismatched"),
        ):
            names = plan[key]
            if names:
                print(f"  {label}  {len(names)}")
                for name in names[:5]:
                    print(f"      {name}")

        if not args.apply:
            print("\ndry run -- pass --apply to write these paths")
            return 0
        with conn:
            for sha, path in plan["relinked"]:
                conn.execute(
                    "update clips set path = ? where clip_sha = ?", (str(path), sha)
                )
        print(f"\nupdated {len(plan['relinked'])} path(s)")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Serve the local rating page. The only source of organic labels."""
    review_mod.serve(port=args.port, variant=args.variant, only=args.only)
    return 0


def cmd_spend(_: argparse.Namespace) -> int:
    summary = cache.spend_summary()
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Salience v2: describe, review, and rank gameplay"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("process", cmd_process), ("localize", cmd_localize)):
        cmd = sub.add_parser(name)
        cmd.add_argument("target", help="clip file or folder")
        cmd.add_argument("--limit", type=int, default=None)
        if name == "process":
            cmd.add_argument("--max-calls", type=int, default=1)
            cmd.add_argument("--json", action="store_true")
            cmd.add_argument(
                "--dedupe",
                action="store_true",
                help="describe only one clip per fight (saves calls, loses clips)",
            )
        cmd.set_defaults(func=handler)

    rank_cmd = sub.add_parser("rank")
    rank_cmd.add_argument(
        "--db", default=None, help="optional legacy feedback database (read-only)"
    )
    rank_cmd.add_argument(
        "--variant", default=prompt_mod.active_variant(config.load_config())
    )
    rank_cmd.add_argument(
        "--mode",
        default=config.DEFAULT_RANK_MODE,
        choices=("embedding", "image+text", "fields", "both"),
    )
    rank_cmd.add_argument("--top", type=int, default=12)
    rank_cmd.set_defaults(func=cmd_rank)

    relink_cmd = sub.add_parser(
        "relink", help="repoint the store at moved footage (dry run by default)"
    )
    relink_cmd.add_argument("root", help="folder to search for the clip files")
    relink_cmd.add_argument(
        "--apply", action="store_true", help="write the matched paths"
    )
    relink_cmd.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="trust filenames instead of confirming each file's content hash",
    )
    relink_cmd.set_defaults(func=cmd_relink, verify=True)

    review_cmd = sub.add_parser("review")
    review_cmd.add_argument("--port", type=int, default=8777)
    review_cmd.add_argument(
        "--variant", default=None,
        help="storage variant to rate and rank, e.g. "
             "v3.6-ownkills-window@minimax-m3 for native-recorder clips",
    )
    review_cmd.add_argument(
        "--only", type=Path, default=None,
        help="queue only clips under this folder, e.g. one recording session",
    )
    review_cmd.set_defaults(func=cmd_review)

    sub.add_parser("spend").set_defaults(func=cmd_spend)

    index_cmd = sub.add_parser("index")
    index_cmd.add_argument("target", nargs="?", default=".")
    index_cmd.add_argument(
        "--from-plan",
        default=None,
        help="describe exactly the clips listed in a frozen study plan.json",
    )
    index_cmd.add_argument(
        "--restore-stored",
        action="store_true",
        help="re-describe exactly the clips already stored",
    )
    index_cmd.add_argument(
        "--split-fights",
        action="store_true",
        help="index clips from fights the user split kept/skipped",
    )
    index_cmd.add_argument(
        "--from-feedback",
        default=None,
        help="path to v1 salience.db; indexes a labelled pilot set",
    )
    index_cmd.add_argument("--limit", type=int, default=None)
    index_cmd.add_argument("--max-calls", type=int, default=0)
    index_cmd.add_argument("--workers", type=int, default=None,
                           help=f"calls in flight at once (default {WORKERS}, V2_WORKERS)")
    index_cmd.add_argument(
        "--encoder", default=None, help="tfidf to force the baseline"
    )
    index_cmd.set_defaults(func=cmd_index)

    taste_cmd = sub.add_parser("taste")
    taste_cmd.add_argument(
        "--db", default=None, help="optional legacy feedback database (read-only)"
    )
    taste_cmd.add_argument("--folds", type=int, default=5)
    taste_cmd.add_argument(
        "--embeddings",
        default=None,
        help="embedding variant to score (default: --variant)",
    )
    taste_cmd.add_argument(
        "--voiced-only",
        action="store_true",
        help="only clips whose audio contains speech",
    )
    taste_cmd.add_argument(
        "--model", default="logistic", choices=("centroid", "logistic")
    )
    taste_cmd.add_argument(
        "--variant",
        default=prompt_mod.active_variant(config.load_config()),
        help="prompt version to score (e.g. v2.3-nofacts)",
    )
    taste_cmd.add_argument(
        "--same-clips-as",
        default=None,
        help="restrict to clips also present in this variant",
    )
    taste_cmd.add_argument(
        "--within-fight",
        action="store_true",
        help="score only pairs from the same fight",
    )
    taste_cmd.add_argument(
        "--organic-only",
        action="store_true",
        help="use only deliberate keep/skip labels, not bulk imports",
    )
    taste_cmd.set_defaults(func=cmd_taste)

    similar_cmd = sub.add_parser("similar")
    similar_cmd.add_argument("--query", default=None, help="substring of a clip name")
    similar_cmd.add_argument("--top", type=int, default=3)
    similar_cmd.set_defaults(func=cmd_similar)

    groups_cmd = sub.add_parser("engagements")
    groups_cmd.add_argument("target")
    groups_cmd.add_argument("--limit", type=int, default=None)
    groups_cmd.set_defaults(func=cmd_engagements)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
