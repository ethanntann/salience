"""From a finished recording to the handful of clips worth keeping.

The capture side used to stop at "here is 1.8 GB of footage and a list of
moments a key-press counter flagged". Every step after that -- cutting,
describing, ranking -- was a command someone had to know to run, and the last
one could not run at all on the recorder's own footage. This is that chain as
one call, run automatically when a recording stops:

    segments -> coverage.jsonl -> windows/ -> mic transcribed -> described
             + stored -> ranked -> highlights/

**Coverage, not triggers, decides what is looked at.** The first version cut
only where a detector fired (`triggers.jsonl -> clips/`). Session 1 showed the
gate dropped most of what mattered: the input-burst trigger overlapped 5 of the
14 moments the player kept. So by default every 8 seconds of the session is
described and taste picks; `coverage=False` keeps the trigger-only path for
comparison. At deepseek-v4p1-flash prices that is about $0.002 per window, ~$0.45 for
half an hour.

**Ranking by taste needs labels on footage like this.** The ranker learns only
from labelled clips in the same storage variant, and native-recorder clips live
in their own (`...-window@<model>`). Until some are labelled -- `review
--variant <that variant>` -- there is nothing of the player's to rank by. Rather
than pretend, the pipeline then falls back to an explicit, non-taste ordering
(what the vision model saw happen, then trigger confidence) and says so in the
manifest under `basis`. A highlights folder that claims taste it does not have
is worse than one that admits it.

Raw segments are kept unless `prune_raw` is asked for. Deleting them is the
point of a clipping product, but it is irreversible, and it removes the ability
to re-cut with a different window or re-run a control.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline_v2.capture import clips as clips_mod
from pipeline_v2.capture import gameplay as gameplay_mod

# What the vision model saw, as a fallback order when there is no taste model.
# Not a judgement of quality: a knock the player found boring still outranks a
# rotation they loved. That is exactly why it is labelled as a fallback.
_OUTCOME_ORDER = {"elimination": 2, "knock": 1}
VICTORY_TIER = 3


def event_tier(clip) -> int:
    """What happened, coarsely: a win, then eliminations, knocks, the rest.

    Ranking goes by this first and by taste within a tier. Measured out of
    sample (taste trained on session 1, scored on session 2's 320 ratings),
    taste alone agreed with the player 86.9% of the time, the tier alone
    92.3%, the tier then taste 96.5%. A win is its own tier because the model
    files the Victory Royale banner under `other`/`no_event`, and the player
    kept it; that moved one more kept moment into the top 10.
    """
    if "victory royale" in (clip.summary or "").lower():
        return VICTORY_TIER
    return _OUTCOME_ORDER.get((clip.payload or {}).get("outcome"), 0)

# What an exported highlight adds around its 8s coverage window: the approach
# matters to a viewer even though the model only needed the window itself.
HIGHLIGHT_LEAD = 8.0
HIGHLIGHT_LAG = 2.0
EXPORT_WORKERS = 5


def _sidecar(clip_path) -> dict:
    """The trigger that produced a clip. Not in the store: `index` keeps only the
    timing facts, so the detector's name and confidence live beside the file."""
    import json

    try:
        return json.loads(Path(clip_path).with_suffix(".capture.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class Highlight:
    rank: int
    clip: str
    score: float
    summary: str
    outcome: str
    trigger: str
    said: str = ""
    asked: bool = False  # the player said "clip that" for it


def describe(clip_dir: Path, max_calls: int) -> None:
    """Describe, store and embed through the same path `index` uses.

    Driven rather than reimplemented: that path carries the per-clip commits,
    the unparsed-response handling and the pacing that keeps Cloudflare from
    blocking a batch, all learned the hard way.
    """
    from pipeline_v2 import cli

    cli.cmd_index(argparse.Namespace(
        target=str(clip_dir), from_plan=None, restore_stored=False,
        split_fights=False, from_feedback=None, limit=None,
        max_calls=max_calls, encoder=None, only_new=True,
    ))


def find_gameplay(session: Path) -> dict:
    """When the player was playing, from Fortnite's log (`gameplay`)."""
    segments = clips_mod.read_segments(session)
    if not segments:
        return {"spans": None, "why": "no footage"}
    return gameplay_mod.spans(session, max(s.end for s in segments))


def listen(session: Path, clip_dir: Path) -> int:
    """Transcribe the mic and attach what was said to each clip (`voice`).

    Optional: without faster-whisper installed the session is ranked on what
    was seen alone, which is all a player who never talks loses anyway.
    """
    from pipeline_v2.capture import voice

    files = [p for p in clip_dir.glob("*") if p.suffix in (".mkv", ".mp4")]
    print("  speech: transcribing what the mic heard...", flush=True)
    try:
        return voice.attach(session, sorted(files))
    except ImportError:
        print("  speech: skipped (pip install -e .[audio] to rank by what was said)")
        return 0


def undescribed(clip_dir: Path) -> int:
    """How many clips in `clip_dir` have content with no stored description."""
    from pipeline_v2 import cache, store

    files = [p for p in clip_dir.glob("*") if p.suffix in (".mkv", ".mp4")]
    conn = store.connect()
    try:
        described = {row[0] for row in conn.execute("select clip_sha from descriptions")}
    finally:
        conn.close()
    return sum(cache.clip_sha(p) not in described for p in files)


def unembedded(clip_dir: Path) -> int:
    """How many of this folder's described clips have no embedding yet.

    Non-zero after a live session (`live`): it stores descriptions as they
    arrive and leaves embedding to this run, which needs the whole corpus.
    """
    from pipeline_v2 import store

    conn = store.connect()
    try:
        return conn.execute(
            "select count(*) from descriptions d join clips c using (clip_sha)"
            " left join embeddings e on e.clip_sha = d.clip_sha"
            " and e.prompt_ver = d.prompt_ver"
            " where c.path like ? and e.clip_sha is null",
            (str(clip_dir) + "%",),
        ).fetchone()[0]
    finally:
        conn.close()


def reembed(clip_dir: Path) -> None:
    """Rebuild embeddings when only the transcripts changed (nothing re-described)."""
    from pipeline_v2 import cli, store

    conn = store.connect()
    try:
        variant, _ = _session_clips(conn, clip_dir)
    finally:
        conn.close()
    if variant:
        cli._rebuild_embeddings(None, variant)


def _configured_variant() -> str | None:
    """The variant a capture window described with the current config lands in."""
    from pipeline_v2 import config, prompt
    from pipeline_v2.facts import ClipFacts

    try:
        cfg = config.load_config()
    except Exception:  # noqa: BLE001 - a missing config falls back to the majority
        return None
    return prompt.storage_variant(
        cfg.prompt_version, ClipFacts(event_source="capture_window"), cfg.model)


def _session_clips(conn, clip_dir: Path) -> tuple[str | None, list]:
    """This session's stored clips and the variant they were described under.

    The variant is read back from the store rather than recomputed, so it cannot
    drift from whatever `index` actually wrote -- a drift of exactly this kind
    (`-window` on writes, not on reads) is what made these clips invisible.
    """
    from pipeline_v2 import store

    root = str(clip_dir.resolve())
    rows = conn.execute(
        "select d.prompt_ver, count(*) from descriptions d "
        "join clips c on c.clip_sha = d.clip_sha "
        "where c.path like ? group by 1 order by 2 desc",
        (root + "%",),
    ).fetchall()
    if not rows:
        return None, []
    # A folder described by several models or prompts (a bake-off) holds one
    # variant per arm; the configured one is what the product ranks by.
    configured = _configured_variant()
    variant = configured if configured in {r[0] for r in rows} else rows[0][0]
    # A re-cut (a changed window) writes new content to the same filename, so
    # the store can hold two clips for one path. Only the newest describes the
    # file now on disk; the older row's description is of footage that is gone.
    newest = {
        row[0]: row[1] for row in conn.execute(
            "select path, clip_sha from clips where path like ? "
            "order by described_at", (root + "%",))
    }
    inside = [
        c for c in store.all_clips(conn, variant)
        if str(Path(c.path).resolve()).startswith(root)
        and newest.get(str(c.path)) == c.clip_sha
    ]
    return variant, inside


# How much the player's hands count beside taste, within an event tier. Kills
# the player kept came with more key presses (building, editing, swapping) than
# kills they skipped: ordering kills, taste alone 37% / 69% on sessions 2 / 3,
# with this 48% / 72%. Both scores are standardised within the session.
ACTIVITY_WEIGHT = 1.0
# The player's voice, as loudness rather than words: reactions are mostly not
# words, and Whisper drops them. Session 3: kept windows were louder on the mic
# 86% of the time, kept kills over skipped ones 70%. Half weight, because on
# session 2 it carried nothing within kills (47%).
VOICE_WEIGHT = 0.5
MIC_LEVEL_KEY = "mic_p95_db"


def voice(clips: list) -> dict[str, float]:
    """The mic's loudness in each clip, cached in its sidecar."""
    from pipeline_v2.capture import levels, voice as voice_mod
    from pipeline_v2.jsonio import write_json

    found: dict[str, float] = {}
    for clip in clips:
        sidecar = Path(clip.path).with_suffix(".capture.json")
        record = _sidecar(clip.path)
        if MIC_LEVEL_KEY not in record and record:
            record[MIC_LEVEL_KEY] = levels.loudness(Path(clip.path), voice_mod.MIC_TRACK)
            try:
                write_json(sidecar, record)
            except OSError:
                pass
        if record.get(MIC_LEVEL_KEY) is not None:
            found[clip.clip_sha] = float(record[MIC_LEVEL_KEY])
    # A silent track says nothing about any window; leave it out entirely.
    return found if len(set(found.values())) > 1 else {}


def activity(session: Path, clips: list) -> dict[str, float]:
    """Key presses during each clip's window, from the recorder's input log."""
    from pipeline_v2.capture.input_activity import ActivitySampler

    try:
        samples = ActivitySampler.read_journal(session / "input.tsv")
    except OSError:
        return {}
    found: dict[str, float] = {}
    for clip in clips:
        span = _span(clip)
        if span is not None and samples:
            found[clip.clip_sha] = float(sum(
                s.presses for s in samples if span[0] <= s.at < span[1]))
    return found


def _standardised(values: dict[str, float]) -> dict[str, float]:
    import statistics

    if len(values) < 2:
        return {k: 0.0 for k in values}
    mean, spread = statistics.mean(values.values()), statistics.pstdev(values.values())
    return {k: (v - mean) / (spread or 1.0) for k, v in values.items()}


def rank(conn, clip_dir: Path) -> tuple[str, list]:
    """Clips best-first, and the basis of the order: "taste" or a fallback."""
    from pipeline_v2.ranking import ranked_fights

    variant, inside = _session_clips(conn, clip_dir)
    if not inside:
        return "none", []
    wanted = {c.clip_sha for c in inside}

    info: dict = {}
    fights, labels = ranked_fights(conn, variant=variant, info=info)
    # Once per clip: `ranked_fights` groups by path, so a stale row sharing a
    # re-cut window's filename yields a second "fight" whose best is that clip.
    best: dict[str, tuple[float, object]] = {}
    for fight in fights:
        sha = fight.best.clip_sha
        if sha in wanted and (sha not in best or fight.score > best[sha][0]):
            best[sha] = (fight.score, fight.best)
    scored = list(best.values())
    hands = _standardised(activity(clip_dir.parent, inside))
    voiced = _standardised(voice(inside))

    def extra(sha: str) -> float:
        return ACTIVITY_WEIGHT * hands.get(sha, 0.0) + VOICE_WEIGHT * voiced.get(sha, 0.0)

    if scored:
        taste = _standardised({clip.clip_sha: score for score, clip in scored})
        scored = [(taste[c.clip_sha] + extra(c.clip_sha), c) for _, c in scored]
        scored.sort(key=lambda pair: (-event_tier(pair[1]), -pair[0]))
        weight = info.get("speech_weight", 0.0)
        tried = info.get("speech_cv") or {}
        if weight:
            heard = f", speech weighted {weight:g}"
        elif tried:
            heard = ", speech off: it hurt this player's ratings"
        else:
            heard = ""
        used = "".join(name for name, on in ((", key presses", hands), (", mic level", voiced)) if on)
        return (f"what happened, then taste{used} ({labels} labels, {variant}{heard})",
                scored)

    def fallback(clip) -> tuple[float, float]:
        # With no ratings yet, the player's hands are the best tie-break there
        # is (and better than a detector's confidence, which measured weak).
        return (float(event_tier(clip)), extra(clip.clip_sha))

    ordered = sorted(inside, key=fallback, reverse=True)
    return (
        "fallback: what happened, then key presses and mic level -- no labels on "
        f"this footage yet (review --variant {variant})",
        [(float(fallback(c)[0]), c) for c in ordered],
    )


def _span(clip) -> tuple[float, float] | None:
    """Where a clip sits on the session timeline, from its sidecar."""
    sidecar = _sidecar(clip.path)
    try:
        return float(sidecar["segment_start_sec"]), float(sidecar["segment_end_sec"])
    except (KeyError, TypeError, ValueError):
        return None


# A highlight grows over neighbouring windows that are still the same fight,
# up to this long. A build fight runs longer than one 8s window, and cutting
# it at the window edge (the player's complaint on session 3) loses the end.
MAX_HIGHLIGHT = 45.0
FIGHT_TAGS = frozenset({
    "build_fight", "box_fight", "piece_control", "editing", "shield_crack",
    "third_party", "multi_opponent", "multi_elimination", "finish_downed", "close_range",
})


def in_fight(clip) -> bool:
    """Whether a window is part of a fight: a kill or knock, or fight mechanics."""
    payload = clip.payload or {}
    return event_tier(clip) > 0 or bool(FIGHT_TAGS & set(payload.get("mechanics") or []))


def fight_extent(clip, by_start: dict, pad: tuple[float, float]) -> tuple[float, float] | None:
    """The span to export for `clip`: its window grown over the neighbouring
    windows of the same fight, plus `pad`, at most `MAX_HIGHLIGHT` long.

    `by_start` maps each window's start time to its clip. Growth stops at the
    first window that is not in the fight, or at a gap (a skipped window).
    """
    span = _span(clip)
    if span is None:
        return None
    start, end = span
    stride = end - start

    def neighbour(at):
        found = by_start.get(round(at, 1))
        return found if found is not None and in_fight(found) else None

    if in_fight(clip):
        while end - start + stride + pad[0] + pad[1] <= MAX_HIGHLIGHT:
            before, after = neighbour(start - stride), neighbour(end)
            if after is not None:
                end += stride
            elif before is not None:
                start -= stride
            else:
                break
        # The win is the payoff of the fight before it: worth one window over
        # the cap rather than a second highlight of the same moment (session 3
        # ranked the final kill and its Victory Royale banner #1 and #2).
        after = by_start.get(round(end, 1))
        if after is not None and event_tier(after) == VICTORY_TIER:
            end += stride
    return start - pad[0], end + pad[1]


def window_index(ordered: list) -> dict:
    """Window start time -> clip, for `fight_extent`."""
    index = {}
    for _, clip in ordered:
        span = _span(clip)
        if span is not None:
            index[round(span[0], 1)] = clip
    return index


# How long before "clip that" the moment it means can have been: it is said
# after the kill, often after a breath or a few seconds of looting.
ASK_LOOKBACK = 12.0


def asked_for(ordered: list, times: list[float]) -> list:
    """The clip each "clip that" means: of the windows in the `ASK_LOOKBACK`
    seconds up to it, the one ranked best. Commands with no window there
    (the lobby) are `unwindowed`'s."""
    found: list = []
    for at in times:
        for _, clip in ordered:
            span = _span(clip)
            if span is not None and span[0] < at and at - ASK_LOOKBACK < span[1]:
                if clip not in found:
                    found.append(clip)
                break
    return found


# What is cut for "clip that" said where no window was described (the lobby,
# a loading screen): the moment it means, its lead-in, and the words.
ASK_SPAN = (ASK_LOOKBACK + HIGHLIGHT_LEAD, HIGHLIGHT_LAG)


def unwindowed(ordered: list, times: list[float]) -> list[tuple[float, float]]:
    """Spans to cut straight from the footage, one per "clip that" no window
    covers. Lobby windows are never described (they are no gameplay), but a
    lobby can still be the funny moment; asks close together are one span."""
    spans: list[list[float]] = []
    for at in sorted(times):
        if any(span[0] < at and at - ASK_LOOKBACK < span[1]
               for _, clip in ordered if (span := _span(clip)) is not None):
            continue
        start, end = at - ASK_SPAN[0], at + ASK_SPAN[1]
        if spans and start < spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return [(start, end) for start, end in spans]


def pick(ordered: list, top: int, pad: tuple[float, float] = (0.0, 0.0),
         extent=None, pinned: list = ()) -> list:
    """The best `top` clips, never two whose exported spans overlap.

    `pinned` clips (the player asked for them) come first and are never
    dropped for `top`; a reel asked for twelve times holds twelve.

    `pad` is the lead and lag an export adds. The second-best window is usually
    the same fight 8 seconds later; padded, the two exports would overlap, and
    keeping both fills a short reel with one moment twice. `extent(clip)`, when
    given, is the whole span exported for a clip (`fight_extent`).
    """
    chosen: list = []
    taken: list[tuple[float, float]] = []
    first = [(score, clip) for score, clip in ordered if clip in pinned]
    first.sort(key=lambda pair: pinned.index(pair[1]))
    rest = [(score, clip) for score, clip in ordered if clip not in pinned]
    for score, clip in first + rest:
        if len(chosen) >= max(top, len(first)) and clip not in pinned:
            break
        if extent is not None:
            span = extent(clip)
        else:
            span = _span(clip)
            if span is not None:
                span = (span[0] - pad[0], span[1] + pad[1])
        if span is not None and any(
            span[0] < end and start < span[1] for start, end in taken
        ):
            continue
        chosen.append((score, clip))
        if span is not None:
            taken.append(span)
    return chosen


def _export(session: Path, clip, destination: Path, span) -> bool:
    """A coverage window re-cut from the raw footage over `span` (its lead-in
    and the rest of its fight).

    False when the raw segments are gone (pruned) or the cut fails; the caller
    then exports the bare window, which is shorter but still the moment.
    """
    if span is None:
        return False
    return clips_mod.cut_span(
        session, span[0], span[1], destination
    ) is not None


def run(
    session: Path,
    *,
    top: int = 10,
    prune_raw: bool = False,
    coverage: bool = True,
    speech: bool = True,
    describe_fn=describe,
    harvest_fn=clips_mod.harvest,
    export_fn=None,
    listen_fn=None,
    reembed_fn=None,
    undescribed_fn=None,
    unembedded_fn=None,
) -> dict:
    """Cut, describe, rank and keep the best `top` clips from one recording."""
    from pipeline_v2 import store
    from pipeline_v2.jsonio import write_json

    if coverage:
        journalled = clips_mod.write_coverage(session)
        clip_dir = session / "windows"
        why = "no recorded footage"
    else:
        journalled = clips_mod.read_journal(session / clips_mod.JOURNAL_NAME)
        clip_dir = session / "clips"
        why = "no triggers fired"
    if not journalled:
        return {"session": str(session), "highlights": [],
                "basis": "none", "why": why}

    played: dict = {"spans": None}
    if coverage:
        played = find_gameplay(session)
        half = clips_mod.COVERAGE_STRIDE / 2

        def outside(request) -> bool:
            # Lobby, loading and spectating: not described, never a highlight.
            return played["spans"] is not None and not gameplay_mod.overlaps(
                played["spans"], request.at - half, request.at + half)

        cut = harvest_fn(
            session, clip_dir, journal_name=clips_mod.COVERAGE_JOURNAL,
            analysis_span=clips_mod.COVERAGE_STRIDE,
            lead=half, lag=half, skip=outside,
        )
        if played["spans"] is not None:
            # Who the player is and the match phase, for the prompt (v3.8).
            segments = clips_mod.read_segments(session)
            gameplay_mod.annotate(session, clip_dir, max(s.end for s in segments))
    else:
        cut = harvest_fn(session, clip_dir)
    # Before describing, so the embeddings built at its end include speech.
    heard = (listen_fn or listen)(session, clip_dir) if speech else 0
    # `cut` includes windows already cut on an earlier run. Describing those
    # again is free (cached) but not fast: 300s of frame extraction on
    # session 1 to look up 235 answers it already had.
    fresh = (undescribed_fn or undescribed)(clip_dir) if cut else 0
    if fresh:
        describe_fn(clip_dir, max_calls=fresh)
    elif heard or (unembedded_fn or unembedded)(clip_dir):
        (reembed_fn or reembed)(clip_dir)

    conn = store.connect()
    try:
        basis, ordered = rank(conn, clip_dir)
        said = store.transcripts(conn)
    finally:
        conn.close()
    if played["spans"] is not None:
        # Windows cut before gating existed are still on disk and in the store.
        ordered = [
            (score, clip) for score, clip in ordered
            if (span := _span(clip)) is None
            or gameplay_mod.overlaps(played["spans"], *span)
        ]

    out = session / "highlights"
    if out.exists():
        shutil.rmtree(out)  # derived output only; rebuilt from clips/ each run
    out.mkdir(parents=True)

    pad = (HIGHLIGHT_LEAD, HIGHLIGHT_LAG) if coverage else (0.0, 0.0)
    export = export_fn or _export
    index = window_index(ordered)

    def extent(clip):
        return fight_extent(clip, index, pad) if coverage else None

    from pipeline_v2.capture import voice as voice_mod

    utterances = voice_mod.heard(session) if speech else []
    asks = voice_mod.clip_requests(utterances)
    pinned = asked_for(ordered, asks)
    picked = [pair for pair in pick(ordered, top, pad, extent if coverage else None, pinned)
              if Path(pair[1].path).is_file()]
    # Asked for, then asked for where nothing was looked at, then the rest.
    jobs = ([("window", score, clip) for score, clip in picked if clip in pinned]
            + [("raw", 0.0, span) for span in unwindowed(ordered, asks)]
            + [("window", score, clip) for score, clip in picked if clip not in pinned])

    def name(position, job) -> str:
        kind, _, item = job
        if kind == "raw":
            return f"{position:02d}-asked-{item[1]:.0f}s.mkv"
        return f"{position:02d}-{Path(item.path).name}"

    def export_one(numbered) -> bool:
        position, job = numbered
        kind, _, item = job
        destination = out / name(position, job)
        if kind == "raw":
            return clips_mod.cut_span(session, *item, destination) is not None
        if not (coverage and export(session, item, destination, extent(item))):
            shutil.copy2(Path(item.path), destination)
        return True

    # Side by side: each is a stream copy waiting on the disk, and one after
    # another they were 3.4s of the wait after a session.
    from concurrent.futures import ThreadPoolExecutor

    numbered = list(enumerate(jobs, start=1))
    with ThreadPoolExecutor(EXPORT_WORKERS) as pool:
        exported = list(pool.map(export_one, numbered))

    kept: list[Highlight] = []
    for (position, (kind, score, item)), ok in zip(numbered, exported):
        if not ok:
            continue
        if kind == "raw":
            kept.append(Highlight(
                rank=position, clip=name(position, (kind, score, item)), score=0.0,
                summary="", outcome="not_described", trigger="clip_command",
                said=voice_mod.said_between(utterances, *item), asked=True,
            ))
            continue
        source = Path(item.path)
        payload = item.payload or {}
        kept.append(Highlight(
            rank=position, clip=source.name, score=round(score, 4),
            summary=item.summary or "", outcome=str(payload.get("outcome")),
            trigger=str(_sidecar(source).get("trigger_source", "")),
            said=said.get(item.clip_sha, ""), asked=item in pinned,
        ))

    manifest = {
        "session": str(session),
        "mode": "coverage" if coverage else "triggers",
        "basis": basis,
        "journalled": len(journalled),
        "cut": len(cut),
        "ranked": len(ordered),
        "gameplay": _gameplay_note(played, journalled),
        "highlights": [asdict(h) for h in kept],
    }
    write_json(out / "highlights.json", manifest)

    # Only once highlights exist, and only for an exact segment set: never a
    # glob broad enough to touch clips/, highlights/ or a journal.
    if prune_raw and kept and len(cut) == len(journalled):
        for segment in clips_mod.read_segments(session):
            segment.path.unlink(missing_ok=True)
        manifest["pruned_raw"] = True
        write_json(out / "highlights.json", manifest)
    return manifest


def _gameplay_note(played: dict, journalled: list) -> dict:
    if played.get("spans") is None:
        return {"source": "none", "why": played.get("why", "not looked up")}
    half = clips_mod.COVERAGE_STRIDE / 2
    excluded = sum(
        not gameplay_mod.overlaps(played["spans"], r.at - half, r.at + half)
        for r in journalled
    )
    return {"source": "fortnite log", "matches": played.get("matches"),
            "excluded_windows": excluded, "origin": played.get("origin_source")}


def report(manifest: dict) -> None:
    source = "windows" if manifest.get("mode") == "coverage" else "triggers"
    print(f"\nhighlights: {len(manifest['highlights'])} kept "
          f"({manifest.get('cut', 0)} cut from {manifest.get('journalled', 0)} {source})")
    print(f"  ranked by {manifest['basis']}")
    played = manifest.get("gameplay") or {}
    if played.get("source") == "fortnite log":
        print(f"  gameplay from the Fortnite log: {played['excluded_windows']} lobby, "
              "loading or spectating windows skipped")
    for h in manifest["highlights"]:
        asked = " (you said clip it)" if h.get("asked") else ""
        if h.get("outcome") == "not_described":
            print(f"  {h['rank']:2d}. (you said clip it, outside a match) \"{h['said'][:100]}\"")
            continue
        print(f"  {h['rank']:2d}. [{h['outcome']}]{asked} {h['summary'][:100]}")
        if h.get("said"):
            print(f"      said: \"{h['said'][:100]}\"")
    if manifest.get("pruned_raw"):
        print("  raw segments deleted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cut, describe and rank a finished recording into highlights."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--prune-raw", action="store_true",
        help="delete the raw 30s segments once highlights exist (irreversible)",
    )
    args = parser.parse_args(argv)
    report(run(args.session, top=args.top, prune_raw=args.prune_raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
