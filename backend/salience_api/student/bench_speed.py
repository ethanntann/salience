"""Benchmark local student inference latency on real clips.

Does not write teacher labels or touch the eval UI / live assignments.

Usage::

    python -m salience_api.student.bench_speed \\
        --db ../.local-data/salience.db \\
        --artifacts ../.local-data/student/artifacts \\
        --limit 20 \\
        --report ../.local-data/student/speed-bench.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

from salience_api.clips.keyframes import (
    cleanup_keyframes,
    extract_event_keyframes,
    extract_timeline_keyframes,
)
from salience_api.features.fireworks_teacher import ClipTeacherInput, merge_event_labels
from salience_api.features.hud_ocr import RapidHudOcr
from salience_api.features.teacher_labels import TEACHER_LABEL_KEYS
from salience_api.student.local_teacher import LocalTeacherClient


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _load_clips(db_path: Path, *, limit: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        select id, path, filename, duration_sec, width, height, fps
        from clips
        where path not like 'demo://%'
        order by id
        limit ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    clips = []
    for row in rows:
        path = Path(str(row["path"]))
        if path.is_file():
            clips.append(dict(row))
    return clips


def bench_clip(
    client: LocalTeacherClient,
    clip: dict[str, Any],
    *,
    ocr: RapidHudOcr,
) -> dict[str, Any]:
    video_path = Path(str(clip["path"]))
    started = time.perf_counter()
    phases: dict[str, float] = {}

    mark = time.perf_counter()
    coarse = extract_timeline_keyframes(video_path, clip["duration_sec"])
    phases["timeline_extract_sec"] = time.perf_counter() - mark

    coarse_input = ClipTeacherInput(
        filename=str(clip["filename"]),
        duration_sec=clip["duration_sec"],
        width=clip["width"],
        height=clip["height"],
        fps=clip["fps"],
        tags=[],
        image_paths=[frame.path for frame in coarse],
        image_timestamps=[frame.timestamp_sec for frame in coarse],
        image_views=[frame.view for frame in coarse],
    )

    mark = time.perf_counter()
    base_labels = client.label_clip(coarse_input, ocr=ocr)
    phases["context_sec"] = time.perf_counter() - mark

    mark = time.perf_counter()
    event_timestamps = (
        client.locate_event_timestamps(coarse_input) if coarse else []
    )
    phases["locator_sec"] = time.perf_counter() - mark
    cleanup_keyframes(coarse)

    event_count = 0
    attribution = None
    if event_timestamps:
        mark = time.perf_counter()
        event_frames = extract_event_keyframes(
            video_path,
            clip["duration_sec"],
            event_timestamps,
        )
        phases["event_extract_sec"] = time.perf_counter() - mark
        if event_frames:
            mark = time.perf_counter()
            ocr_observations = ocr.recognize_event_frames(event_frames)
            phases["ocr_sec"] = time.perf_counter() - mark

            mark = time.perf_counter()
            attribution = client.label_weapon_event(
                ClipTeacherInput(
                    filename=str(clip["filename"]),
                    duration_sec=clip["duration_sec"],
                    width=clip["width"],
                    height=clip["height"],
                    fps=clip["fps"],
                    tags=[],
                    image_paths=[frame.path for frame in event_frames],
                    image_timestamps=[frame.timestamp_sec for frame in event_frames],
                    image_views=[frame.view for frame in event_frames],
                    image_event_indices=[frame.event_index for frame in event_frames],
                    image_event_centers=[
                        frame.event_center_sec for frame in event_frames
                    ],
                    ocr_observations=ocr_observations,
                )
            )
            phases["event_heads_sec"] = time.perf_counter() - mark
            event_count = len({frame.event_index for frame in event_frames})
            cleanup_keyframes(event_frames)
        else:
            phases["event_extract_sec"] = phases.get("event_extract_sec", 0.0)
    else:
        phases["event_extract_sec"] = 0.0
        phases["ocr_sec"] = 0.0
        phases["event_heads_sec"] = 0.0

    labels = merge_event_labels(base_labels, attribution)
    label_dict = {
        key: str(getattr(labels, key))
        for key in TEACHER_LABEL_KEYS
    }
    events = []
    if attribution is not None:
        for event in attribution.events:
            if not isinstance(event, dict):
                continue
            events.append(
                {
                    "event_index": event.get("event_index"),
                    "status": event.get("status"),
                    "event_kind": event.get("event_kind"),
                    "finish_timestamp": event.get("finish_timestamp"),
                    "resolved_weapon": event.get("resolved_weapon"),
                    "target_was_active": event.get("target_was_active"),
                    "target_was_downed": event.get("target_was_downed"),
                    "damage_aim_state": event.get("damage_aim_state"),
                    "visual_action_supported": event.get("visual_action_supported"),
                    "summary": event.get("summary"),
                }
            )
    total = time.perf_counter() - started
    phases["total_sec"] = total
    yes_labels = sorted(key for key, value in label_dict.items() if value == "yes")
    return {
        "clip_id": int(clip["id"]),
        "filename": str(clip["filename"]),
        "duration_sec": float(clip["duration_sec"] or 0.0),
        "locator_timestamps": [float(value) for value in event_timestamps],
        "locator_events": len(event_timestamps),
        "event_windows": event_count,
        "attribution_status": attribution.status if attribution else "no_event",
        "labels": label_dict,
        "yes_labels": yes_labels,
        "events": events,
        "phases": phases,
        "total_sec": total,
        "video_url": f"/clips/{int(clip['id'])}/video",
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [float(item["total_sec"]) for item in results]
    phase_names = [
        "timeline_extract_sec",
        "context_sec",
        "locator_sec",
        "event_extract_sec",
        "ocr_sec",
        "event_heads_sec",
        "total_sec",
    ]
    phase_summary = {}
    for name in phase_names:
        values = [
            float(item["phases"].get(name, 0.0))
            for item in results
            if name in item["phases"] or name == "total_sec"
        ]
        if name == "total_sec":
            values = totals
        if not values:
            continue
        phase_summary[name] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p90": _percentile(values, 0.90),
            "max": max(values),
        }
    return {
        "clips": len(results),
        "median_sec": statistics.median(totals) if totals else 0.0,
        "mean_sec": statistics.fmean(totals) if totals else 0.0,
        "p90_sec": _percentile(totals, 0.90),
        "max_sec": max(totals) if totals else 0.0,
        "phases": phase_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time local student inference without writing teacher labels"
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    clips = _load_clips(args.db, limit=args.limit)
    if not clips:
        raise SystemExit(f"no local video clips found in {args.db}")

    client = LocalTeacherClient.from_artifacts(args.artifacts)
    ocr = RapidHudOcr(enabled=True)
    results = []
    for index, clip in enumerate(clips, start=1):
        result = bench_clip(client, clip, ocr=ocr)
        results.append(result)
        print(
            f"[{index}/{len(clips)}] clip {result['clip_id']} "
            f"{result['total_sec']:.1f}s "
            f"events={result['locator_events']}",
            flush=True,
        )

    report = {
        "model": client.model,
        "provider": client.provider,
        "limit": args.limit,
        "summary": summarize(results),
        "clips": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        json.dumps(
            {
                "model": client.model,
                "clips": summary["clips"],
                "median_sec": round(summary["median_sec"], 2),
                "mean_sec": round(summary["mean_sec"], 2),
                "p90_sec": round(summary["p90_sec"], 2),
                "max_sec": round(summary["max_sec"], 2),
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
