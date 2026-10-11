"""From a finished recording to the handful of clips worth keeping.

The capture side used to stop at "here is 1.8 GB of footage and a list of
moments a key-press counter flagged". Every step after that -- cutting,
describing, ranking -- was a command someone had to know to run, and the last
one could not run at all on the recorder's own footage. This is that chain as
one call, run automatically when a recording stops:

    segments -> coverage.jsonl -> windows/ -> described + stored -> ranked
             -> highlights/

**Coverage, not triggers, decides what is looked at.** The first version cut
only where a detector fired (`triggers.jsonl -> clips/`). Session 1 showed the
gate dropped most of what mattered: the input-burst trigger overlapped 5 of the
14 moments the player kept. So by default every 8 seconds of the session is
described and taste picks; `coverage=False` keeps the trigger-only path for
comparison. At minimax-m3 prices that is about $0.0018 per window, ~$0.45 for
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

# What the vision model saw, as a fallback order when there is no taste model.
# Not a judgement of quality: a knock the player found boring still outranks a
# rotation they loved. That is exactly why it is labelled as a fallback.
_OUTCOME_ORDER = {"elimination": 2, "knock": 1}

# What an exported highlight adds around its 8s coverage window: the approach
# matters to a viewer even though the model only needed the window itself.
HIGHLIGHT_LEAD = 8.0
HIGHLIGHT_LAG = 2.0


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
        max_calls=max_calls, encoder=None,
    ))


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
    variant = rows[0][0]
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


def rank(conn, clip_dir: Path) -> tuple[str, list]:
    """Clips best-first, and the basis of the order: "taste" or a fallback."""
    from pipeline_v2.ranking import ranked_fights

    variant, inside = _session_clips(conn, clip_dir)
    if not inside:
        return "none", []
    wanted = {c.clip_sha for c in inside}

    fights, labels = ranked_fights(conn, variant=variant)
    # Once per clip: `ranked_fights` groups by path, so a stale row sharing a
    # re-cut window's filename yields a second "fight" whose best is that clip.
    best: dict[str, tuple[float, object]] = {}
    for fight in fights:
        sha = fight.best.clip_sha
        if sha in wanted and (sha not in best or fight.score > best[sha][0]):
            best[sha] = (fight.score, fight.best)
    scored = list(best.values())
    if scored:
        scored.sort(key=lambda pair: -pair[0])
        return f"taste ({labels} labels, {variant})", scored

    def fallback(clip) -> tuple[float, float]:
        outcome = _OUTCOME_ORDER.get((clip.payload or {}).get("outcome"), 0)
        confidence = float(_sidecar(clip.path).get("trigger_confidence") or 0)
        return (outcome, confidence)

    ordered = sorted(inside, key=fallback, reverse=True)
    return (
        "fallback: what happened, then trigger confidence -- no labels on "
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


def pick(ordered: list, top: int, pad: tuple[float, float] = (0.0, 0.0)) -> list:
    """The best `top` clips, never two whose exported spans overlap.

    `pad` is the lead and lag an export adds. The second-best window is usually
    the same fight 8 seconds later; padded, the two exports would overlap, and
    keeping both fills a short reel with one moment twice.
    """
    chosen: list = []
    taken: list[tuple[float, float]] = []
    for score, clip in ordered:
        if len(chosen) >= top:
            break
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


def _export(session: Path, clip, destination: Path, pad) -> bool:
    """A coverage window re-cut from the raw footage with its lead-in.

    False when the raw segments are gone (pruned) or the cut fails; the caller
    then exports the bare window, which is shorter but still the moment.
    """
    span = _span(clip)
    if span is None:
        return False
    return clips_mod.cut_span(
        session, span[0] - pad[0], span[1] + pad[1], destination
    ) is not None


def run(
    session: Path,
    *,
    top: int = 10,
    prune_raw: bool = False,
    coverage: bool = True,
    describe_fn=describe,
    harvest_fn=clips_mod.harvest,
    export_fn=None,
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

    if coverage:
        cut = harvest_fn(
            session, clip_dir, journal_name=clips_mod.COVERAGE_JOURNAL,
            analysis_span=clips_mod.COVERAGE_STRIDE,
            lead=clips_mod.COVERAGE_STRIDE / 2, lag=clips_mod.COVERAGE_STRIDE / 2,
        )
    else:
        cut = harvest_fn(session, clip_dir)
    if cut:
        describe_fn(clip_dir, max_calls=len(cut))

    conn = store.connect()
    try:
        basis, ordered = rank(conn, clip_dir)
    finally:
        conn.close()

    out = session / "highlights"
    if out.exists():
        shutil.rmtree(out)  # derived output only; rebuilt from clips/ each run
    out.mkdir(parents=True)

    pad = (HIGHLIGHT_LEAD, HIGHLIGHT_LAG) if coverage else (0.0, 0.0)
    export = export_fn or _export
    kept: list[Highlight] = []
    for position, (score, clip) in enumerate(pick(ordered, top, pad), start=1):
        source = Path(clip.path)
        if not source.is_file():
            continue
        destination = out / f"{position:02d}-{source.name}"
        if not (coverage and export(session, clip, destination, pad)):
            shutil.copy2(source, destination)
        payload = clip.payload or {}
        kept.append(Highlight(
            rank=position, clip=source.name, score=round(score, 4),
            summary=clip.summary or "", outcome=str(payload.get("outcome")),
            trigger=str(_sidecar(source).get("trigger_source", "")),
        ))

    manifest = {
        "session": str(session),
        "mode": "coverage" if coverage else "triggers",
        "basis": basis,
        "journalled": len(journalled),
        "cut": len(cut),
        "ranked": len(ordered),
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


def report(manifest: dict) -> None:
    source = "windows" if manifest.get("mode") == "coverage" else "triggers"
    print(f"\nhighlights: {len(manifest['highlights'])} kept "
          f"({manifest.get('cut', 0)} cut from {manifest.get('journalled', 0)} {source})")
    print(f"  ranked by {manifest['basis']}")
    for h in manifest["highlights"]:
        print(f"  {h['rank']:2d}. [{h['outcome']}] {h['summary'][:100]}")
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
    parser.add_argument(
        "--triggers-only", dest="coverage", action="store_false",
        help="describe only where a detector fired (cheaper, misses most keeps)",
    )
    args = parser.parse_args(argv)
    report(run(args.session, top=args.top, prune_raw=args.prune_raw,
               coverage=args.coverage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
