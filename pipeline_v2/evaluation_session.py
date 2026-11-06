"""Freeze and evaluate a continuous saved-session stretch, with blind human review."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import random
import subprocess

from pipeline_v2 import cache, config, localization, vlm
from pipeline_v2 import freeze
from pipeline_v2.jsonio import read_json, utc_now, write_json


def _fingerprint(cfg) -> dict:
    """The session study runs localization, which overrides the prompt version.

    Recording the description version here would make an unrelated V2_ABLATE
    setting look like drift and reject a valid frozen plan.
    """
    from dataclasses import replace

    from pipeline_v2 import frames, prompt

    return freeze.fingerprint(
        replace(cfg, prompt_version=localization.PROMPT_VERSION),
        prompt.SYSTEM + localization.INSTRUCTIONS,
        frames.FRAME_WIDTH,
    )


def select_stretch(session: Path, excluded: set[Path], count: int = 10, seed: int = 0):
    if count < 1:
        raise ValueError("At least one chunk is required")
    rows = []
    for path in sorted(session.rglob("*.capture.json")):
        m = read_json(path)
        clip = (path.parent / m["video_file"]).resolve()
        rows.append(
            {
                "clip": str(clip),
                "run": str(path.parent),
                "segment_start": m["segment_start_sec"],
                "segment_end": m["segment_end_sec"],
                "duration": m["duration_sec"],
            }
        )
    candidates = []
    for i in range(len(rows) - count + 1):
        group = rows[i : i + count]
        if any(
            Path(w["clip"]) in excluded or not 29.9 <= w["duration"] <= 30.1
            for w in group
        ):
            continue
        if any(
            a["run"] != b["run"] or abs(a["segment_end"] - b["segment_start"]) > 0.1
            for a, b in zip(group, group[1:])
        ):
            continue
        candidates.append(group)
    if not candidates:
        raise ValueError("No consecutive unevaluated full chunks remain")
    return random.Random(seed).choice(candidates), len(candidates)


def prepare(session: Path, out: Path, prior: Path, count: int = 10):
    if (out / "plan.json").exists():
        raise ValueError("Session study already frozen; run or review it instead")
    excluded = set()

    def collect(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ("clip", "source_clip") and isinstance(v, str):
                    excluded.add(Path(v).resolve())
                collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)

    prior_plans = sorted(prior.glob("evaluation*/**/plan.json"))
    for path in prior_plans:
        collect(read_json(path))
    group, candidates = select_stretch(session, excluded, count)
    out.mkdir(parents=True, exist_ok=True)
    offset = 0
    for i, w in enumerate(group):
        w.update(
            id=f"chunk-{i:02d}", offset=offset, clip_sha=cache.clip_sha(Path(w["clip"]))
        )
        offset += w["duration"]
    cfg = config.load_config()
    write_json(
        out / "plan.json",
        {
            "study_kind": "continuous_localization",
            "created_utc": utc_now(),
            "session": str(session),
            "session_id": session.name,
            "windows": group,
            "duration": offset,
            **_fingerprint(cfg),
            "selection": f"Seed 0 choice among {candidates} eligible consecutive {count}-chunk stretches; exclude every clip/source_clip in prior evaluation plans.",
            "prior_plans": [str(p) for p in prior_plans],
            "limitations": "Not previously model-evaluated or user-labeled according to saved study plans. Midpoint contact sheets of the whole session were previously viewed. Same recording as development, not an untouched session. No event-based selection.",
        },
    )
    concat = out / "concat.txt"
    concat.write_text(
        "\n".join(
            "file '" + w["clip"].replace("\\", "/").replace("'", "'\\''") + "'"
            for w in group
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            str(out / "continuous.mp4"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    write_json(out / "review-data.json", {"duration": offset, "windows": []})
    write_json(
        out / "user-labels.json",
        {
            "annotator": "user",
            "labels": {},
            "events": [],
            "timeline_review_complete": False,
        },
    )
    print(
        f"Frozen {len(group)} chunks, {offset:.1f}s: {group[0]['clip']} through {group[-1]['clip']}",
        flush=True,
    )


def run(out: Path, max_calls: int):
    plan = read_json(out / "plan.json")
    cfg = config.load_config()
    freeze.check(plan, _fingerprint(cfg), "session study")
    budget = vlm.CallBudget(max_calls)
    try:
        for w in plan["windows"]:
            if cache.clip_sha(Path(w["clip"])) != w["clip_sha"]:
                raise ValueError("Frozen session media changed")
            events = localization.locate(
                Path(w["clip"]), out / w["id"], max_calls, call_budget=budget
            )
            print(w["id"], f"{len(events)} proposals", flush=True)
    finally:
        write_json(
            out / "last-run.json",
            {
                "saved_utc": utc_now(),
                "call_limit": max_calls,
                "reserved_requests": budget.attempted,
            },
        )
        report(out)


def report(out: Path):
    plan = read_json(out / "plan.json")
    proposals = []
    missing = []
    cost = 0
    requests = 0
    for w in plan["windows"]:
        path = out / w["id"] / "proposals.json"
        if not path.exists():
            missing.append(w["id"])
        else:
            proposals.extend(
                {
                    **e,
                    "start": w["offset"] + e["start"],
                    "end": w["offset"] + e["end"],
                    "chunk_id": w["id"],
                }
                for e in read_json(path)["refined"]
            )
        for p in (out / w["id"]).glob("*.json"):
            if p.stem == "coarse" or (
                p.stem.startswith("refine-") and p.stem[7:].isdigit()
            ):
                cost += read_json(p)["cost_usd"]
                requests += 1
    user = read_json(out / "user-labels.json")
    metrics = None
    if user.get("timeline_review_complete") and not missing:
        from pipeline_v2.evaluation_metrics import event_metrics

        own = [
            (e["start"], e["end"])
            for e in proposals
            if e["actor"] == "recording_player"
        ]
        metrics = event_metrics(own, user["events"])
        metrics["note"] = (
            "Temporal overlap with user-marked wanted moments, not a rating of clip boundaries, description correctness, or final ranking. Adjacent proposals merged."
        )
        metrics["wanted_moments"] = len(user["events"])
        if not user["events"]:
            metrics["precision"] = 0.0 if own else None
        metrics["unwanted_proposals_per_hour"] = sum(
            not any(max(a, e["start"]) < min(b, e["end"]) for e in user["events"])
            for a, b in metrics["proposals"]
        ) / (plan["duration"] / 3600)
    result = {
        "study_kind": "continuous_localization",
        "duration": plan["duration"],
        "missing_chunks": missing,
        "proposals": proposals,
        "requests_with_valid_results": requests,
        "estimated_cost_usd": cost,
        "cost_note": "Valid stage files only; invalid or ambiguous requests may add provider charges. Historical conservative price assumptions.",
        "user_highlight_metrics": metrics,
        "human_review_complete": user.get("timeline_review_complete", False),
    }
    write_json(out / "metrics.json", result)
    return result


def review(out: Path, port: int):
    from pipeline_v2.evaluation_review import Handler
    from pipeline_v2.session_review import HTML

    (out / "index.html").write_text(HTML, encoding="utf-8")
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(Handler, directory=str(out))
    )
    print(f"Review: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("session", type=Path)
    prep.add_argument("out", type=Path)
    prep.add_argument("--prior", type=Path, default=Path(".local-data"))
    prep.add_argument("--chunks", type=int, default=10)
    execute = sub.add_parser("run")
    execute.add_argument("out", type=Path)
    execute.add_argument("--max-calls", type=int, default=0)
    rev = sub.add_parser("review")
    rev.add_argument("out", type=Path)
    rev.add_argument("--port", type=int, default=8780)
    rep = sub.add_parser("report")
    rep.add_argument("out", type=Path)
    args = p.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        prepare(args.session.resolve(), out, args.prior.resolve(), args.chunks)
    elif args.command == "run":
        run(out, args.max_calls)
    elif args.command == "review":
        review(out, args.port)
    else:
        print(report(out))
