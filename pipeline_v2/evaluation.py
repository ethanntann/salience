"""Frozen, offline session pilots, kept separate from the preference database.

prepare performs no inference; describe requires an explicit call allowance.
The continuous sweep tests a candidate offline extension, not a live detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

from pipeline_v2 import (
    cache,
    config,
    embed,
    facts,
    feedback,
    frames,
    prompt,
    store,
    taste,
    vlm,
)
from pipeline_v2 import freeze
from pipeline_v2.jsonio import read_json, utc_now, write_json


def _fingerprint(cfg) -> dict:
    return freeze.fingerprint(
        cfg,
        prompt.SYSTEM
        + prompt.build_prompt(facts.ClipFacts(event_source="capture_window")),
        frames.FRAME_WIDTH,
    )


def select_windows(session: Path, sample_count: int = 12) -> list[dict]:
    clips = sorted(session.rglob("*.capture.json"))
    if len(clips) < sample_count:
        raise ValueError("Session has too few finalized chunks")
    windows = []
    for i in range(sample_count):
        sidecar = clips[int((i + 0.5) * len(clips) / sample_count)]
        m = read_json(sidecar)
        windows.append(
            {
                "id": f"sample-{i:02d}",
                "cohort": "uniform",
                "clip": str(sidecar.with_suffix(".mkv").with_name(m["video_file"])),
                "start": m["analysis_start_sec"],
                "end": m["analysis_end_sec"],
            }
        )
    runs = sorted({p.parent for p in clips})
    longest = max(
        runs, key=lambda r: sum(read_json(p)["duration_sec"] for p in clips if p.parent == r)
    )
    full = [p for p in clips if p.parent == longest and read_json(p)["duration_sec"] >= 29.9]
    if len(full) < 6:
        raise ValueError("Need six consecutive full chunks for the continuous pilot")
    begin = len(full) // 2 - 3
    stretch = full[begin : begin + 6]
    previous_end = None
    elapsed = 0.0
    for sidecar in stretch:
        m = read_json(sidecar)
        if (
            previous_end is not None
            and abs(m["segment_start_sec"] - previous_end) > 0.1
        ):
            raise ValueError("Continuous pilot has a gap in capture timestamps")
        previous_end = m["segment_end_sec"]
        duration = m["duration_sec"]
        start = 0.0
        while start < duration - 0.1:
            end = min(start + 4, duration)
            windows.append(
                {
                    "id": f"sweep-{sum(w['cohort'] == 'continuous' for w in windows):02d}",
                    "cohort": "continuous",
                    "clip": str(sidecar.parent / m["video_file"]),
                    "start": start,
                    "end": end,
                    "timeline_start": elapsed + start,
                    "timeline_end": elapsed + end,
                }
            )
            start = end
        elapsed += duration
    return windows


def prepare(session: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if not plan_path.exists():
        windows = select_windows(session)
        cfg = config.load_config()
        write_json(
            plan_path,
            {
                "schema": "salience.evaluation/1",
                "created_utc": utc_now(),
                "session": str(session),
                "session_id": session.name,
                **_fingerprint(cfg),
                "frames_per_window": 7,
                "selection": "12 equally spaced chunk indices using their frozen uniform windows; six middle full chunks of longest run, nonoverlapping <=4s windows",
                "ranking": "TF-IDF/LSA trained only on historical labelled descriptions + v2 logistic; top 3 continuous windows, adjacent selections merged; chronological and seeded random baselines",
                "limitations": [
                    "one session",
                    "continuous sweep is offline and not shipped live detection",
                    "assistant annotations are provisional, never user preference labels",
                ],
                "windows": windows,
            },
        )
    plan = read_json(plan_path)
    if plan["session"] != str(session):
        raise ValueError("Output already belongs to another session")
    prepared = []
    previews = out / "previews"
    previews.mkdir(exist_ok=True)
    for w in plan["windows"]:
        clip = Path(w["clip"])
        directory = out / "frames" / w["id"]
        record = directory / "frames.json"
        if not record.exists():
            duration = frames.probe_duration(clip)
            midpoint = (w["start"] + w["end"]) / 2
            images = frames.window_frames(
                clip,
                duration,
                midpoint,
                pre_sec=midpoint - w["start"],
                post_sec=w["end"] - midpoint,
                out_dir=directory,
            )
            if len(images) != 7:
                raise ValueError(f"{w['id']}: expected seven frames, got {len(images)}")
            write_json(
                record,
                {
                    "clip_sha": cache.clip_sha(clip),
                    "images": [
                        {
                            "path": str(f.path.relative_to(out)),
                            "timestamp": f.timestamp_sec,
                        }
                        for f in images
                    ],
                },
            )
        preview = previews / f"{clip.parent.name}-{clip.stem}.mp4"
        if not preview.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(clip),
                    "-map",
                    "0:v:0",
                    "-c:v",
                    "copy",
                    "-an",
                    "-movflags",
                    "+faststart",
                    str(preview),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        prepared.append({**w, **read_json(record), "preview": str(preview.relative_to(out))})
        print(f"prepared {w['id']}", flush=True)
    write_json(out / "prepared.json", {"windows": prepared})


def describe(out: Path, max_calls: int) -> None:
    plan = read_json(out / "plan.json")
    cfg = config.load_config()
    freeze.check(plan, _fingerprint(cfg), "evaluation plan")
    # Conservative estimate from the provider's published model card (2026-09-07).
    # It lists conflicting $0.40/$1.60 and $0.50/$3.00 prices; use the higher pair.
    # This is an estimate, never the account invoice or a hard dollar ceiling.
    cfg = config.resolve(cfg)
    results_dir = out / "results"
    results_dir.mkdir(exist_ok=True)
    attempted = 0
    for w in read_json(out / "prepared.json")["windows"]:
        target = results_dir / f"{w['id']}.json"
        if target.exists():
            continue
        clip = Path(w["clip"])
        if cache.clip_sha(clip) != w["clip_sha"]:
            raise ValueError("Source video changed after sample freeze")
        clip_facts = facts.ClipFacts(
            event_sec=(w["start"] + w["end"]) / 2,
            event_source="capture_window",
            analysis_start_sec=w["start"],
            analysis_end_sec=w["end"],
            capture_session=plan["session_id"],
        )
        started = time.monotonic()
        # Stop on any failure: do not automatically replay ambiguous paid requests.
        result = vlm.describe(
            cfg,
            clip,
            clip_facts,
            [out / f["path"] for f in w["images"]],
            allow_call=attempted < max_calls,
        )
        attempted += not result.cached
        write_json(
            target,
            {
                "id": w["id"],
                "created_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "model": cfg.model,
                "prompt_version": cfg.prompt_version,
                **result.model_dump(),
            },
        )
        print(f"described {w['id']}: cached={result.cached}", flush=True)
        if not result.cached:
            time.sleep(1.5)


def rank(out: Path, legacy_db: Path) -> None:
    plan = read_json(out / "plan.json")
    prefs = feedback.PreferenceIndex(feedback.load(legacy_db, require_media=False))
    conn = store.connect()
    try:
        stored = store.all_clips(conn, plan["prompt_version"])
        own = {
            r["clip_sha"]: r["label"]
            for r in conn.execute("select clip_sha,label from preferences")
        }
    finally:
        conn.close()
    training, labels, weights = [], [], []
    for c in stored:
        if c.facts.get("capture_session") == plan["session_id"]:
            continue
        p = prefs.get(c.path)
        if c.clip_sha not in own and p is None:
            continue
        training.append(c)
        labels.append(own.get(c.clip_sha, p.label if p else 0))
        weights.append(1.0 if c.clip_sha in own else p.weight)
    if len(training) < 10 or len(set(labels)) < 2:
        raise ValueError(
            "Insufficient historical labels; cannot claim personalized ranking"
        )
    encoder = embed.TfidfEncoder()
    vectors = encoder.fit_transform([c.summary for c in training])
    examples = [
        taste.Example(c.clip_sha, None, y, weight, [], v)
        for c, y, weight, v in zip(training, labels, weights, vectors, strict=True)
    ]
    model = taste.LogisticRanker().fit(examples, "embedding")
    predictions = [read_json(out / "results" / f"{w['id']}.json") for w in plan["windows"]]
    test_vectors = encoder.transform([p["description"]["summary"] for p in predictions])
    scores = [
        {
            "id": p["id"],
            "score": model.score(
                taste.Example(p["id"], plan["session_id"], 0, 1, [], v), "embedding"
            ),
        }
        for p, v in zip(predictions, test_vectors, strict=True)
    ]
    write_json(
        out / "ranking.json",
        {
            "model": "v2 logistic / train-only TF-IDF-LSA baseline",
            "historical_semantic_encoder_reproduced": False,
            "training_examples": len(training),
            "training_positives": sum(labels),
            "training_ids": [c.clip_sha for c in training],
            "test_labels_used": 0,
            "scores": scores,
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("session", type=Path)
    prep.add_argument("out", type=Path)
    desc = sub.add_parser("describe")
    desc.add_argument("out", type=Path)
    desc.add_argument("--max-calls", type=int, default=0)
    scoring = sub.add_parser("rank")
    scoring.add_argument("out", type=Path)
    scoring.add_argument("--legacy-db", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.session.resolve(), args.out.resolve())
    elif args.command == "describe":
        describe(args.out.resolve(), args.max_calls)
    else:
        rank(args.out.resolve(), args.legacy_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
