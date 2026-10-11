"""Per-stage latency benchmark for the local clip-processing pipeline.

Every optimisation in this pipeline has to be argued against real numbers -- the
neural student is a small fraction of wall time, and OCR has historically been
the largest single cost while being absent from ad-hoc measurements. Run this
before and after any change that touches decoding, OCR, or inference.

Usage::

    python scripts/bench_pipeline.py --clips sample-clips --limit 3
    python scripts/bench_pipeline.py --clips sample-clips --json out.json

Note: on hybrid-core CPUs (Intel P/E) ONNX Runtime throughput has been observed
to vary ~5x between processes depending on core placement. Compare runs taken
back to back, and treat a single run as indicative rather than authoritative.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import cv2  # noqa: E402

from salience_api.clips.keyframes import (  # noqa: E402
    cleanup_keyframes,
    extract_event_keyframes,
    extract_timeline_keyframes,
)
from salience_media.probe import safe_probe_media  # noqa: E402
from salience_api.features.hud_ocr import RapidHudOcr  # noqa: E402
from salience_api.student.schema import select_window_indices  # noqa: E402

STAGES = (
    "probe",
    "timeline_extract",
    "event_extract",
    "weapon_hud_ocr",
    "screen_state_ocr",
)


def _time(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def benchmark_clip(path: Path, ocr: RapidHudOcr) -> dict:
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}

    media, timings["probe"] = _time(lambda: safe_probe_media(path))
    duration = getattr(media, "duration_sec", None) or 20.0

    coarse, timings["timeline_extract"] = _time(
        lambda: extract_timeline_keyframes(path, duration)
    )
    counts["timeline_frames"] = len(coarse)
    if not coarse:
        return {"clip": path.name, "error": "no timeline frames"}

    # Locator output is not needed for timing; use evenly spaced stand-ins so the
    # event stage does the same amount of work on every clip.
    events = [duration * fraction for fraction in (0.2, 0.4, 0.6, 0.85)]
    event_frames, timings["event_extract"] = _time(
        lambda: extract_event_keyframes(path, duration, events)
    )
    counts["event_frames"] = len(event_frames)

    observations, timings["weapon_hud_ocr"] = _time(
        lambda: ocr.recognize_event_frames(event_frames)
    )
    counts["weapon_observations"] = len(observations)

    total = len(coarse)
    sample_indices = list(
        dict.fromkeys(
            [
                *range(max(0, total - 3), total),
                *select_window_indices(total, min(5, total)),
            ]
        )
    )
    texts, timings["screen_state_ocr"] = _time(
        lambda: ocr.recognize_frames_text(
            [coarse[index].path for index in sample_indices]
        )
    )
    counts["screen_state_frames"] = len(sample_indices)
    counts["screen_state_texts"] = sum(len(v) for v in texts.values())

    cleanup_keyframes(coarse)
    cleanup_keyframes(event_frames)

    timings["TOTAL"] = sum(timings[stage] for stage in STAGES)
    return {"clip": path.name, "duration_sec": round(duration, 1),
            "timings": timings, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", default="sample-clips", help="folder of .mp4 clips")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-ocr", action="store_true", help="skip OCR stages")
    args = parser.parse_args()

    folder = (REPO_ROOT / args.clips) if not Path(args.clips).is_absolute() else Path(args.clips)
    clips = sorted(folder.glob("*.mp4"))[: args.limit]
    if not clips:
        print(f"no clips found in {folder}")
        return 1

    ocr = RapidHudOcr(enabled=not args.no_ocr)
    if ocr.available:
        # Warm the models so the first clip is not charged for graph setup.
        import numpy as np

        warm = np.full((120, 480, 3), 20, dtype=np.uint8)
        cv2.putText(warm, "SHOTGUN 30", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (255, 255, 255), 3)
        engine = ocr._get_engine()
        if engine is not None:
            engine(warm)

    results = []
    for clip in clips:
        result = benchmark_clip(clip, ocr)
        results.append(result)
        if "error" in result:
            print(f"{clip.name[:44]:46} ERROR: {result['error']}")
        else:
            print(f"{clip.name[:44]:46} {result['timings']['TOTAL']:6.2f}s")

    ok = [r for r in results if "error" not in r]
    if not ok:
        return 1

    print()
    print(f"{'stage':22}{'mean':>9}{'median':>9}{'% total':>10}")
    print("-" * 50)
    total_mean = statistics.mean(r["timings"]["TOTAL"] for r in ok)
    for stage in (*STAGES, "TOTAL"):
        values = [r["timings"][stage] for r in ok]
        mean = statistics.mean(values)
        share = 100 * mean / total_mean if total_mean else 0.0
        print(f"{stage:22}{mean:>8.2f}s{statistics.median(values):>8.2f}s{share:>9.0f}%")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
