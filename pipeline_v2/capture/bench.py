"""Does capturing cost the game anything? Measure it instead of assuming.

The recorder has been paused since 2026-09-07 on the strength of "FPS drops and
game crashes", and SCOPE proposes a gate of under 5% deterioration in game 1%
lows and under 100ms A/V drift over an hour. Nothing measured any of it, so the
gate could neither pass nor fail and the direction stayed blocked.

This records one labelled run at a time. Play the same way twice -- once with
capture off, once with it on -- then compare. A paired comparison is the whole
point: absolute frame rates say nothing without the matching baseline, and
"it felt fine" is what the September session also reported before it crashed.

Frame times need a present-mon style tracer, which is the one piece not already
on this machine. Everything else (GPU, VRAM, power, A/V drift) is collected from
nvidia-smi and ffprobe regardless, and the report says plainly which gates it
could evaluate and which it could not.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pipeline_v2.config import REPO_ROOT
from pipeline_v2.jsonio import read_json, utc_now, write_json

# SCOPE's proposed gates. They are thresholds to argue about, not achievements.
MAX_ONE_PERCENT_LOW_LOSS = 0.05
MAX_AV_DRIFT_MS = 100.0
# How much better-than-baseline is explainable by ordinary variation before the
# pairing itself is suspect. Observed baseline 1% lows across three sessions on
# one machine: 37.28, 26.60 -- a 40% spread with nothing recording at all.
NOISE_TOLERANCE = 0.05

GAME_PROCESS = "FortniteClient-Win64-Shipping.exe"


# ---------------------------------------------------------------- frame timing


def frame_metrics(frame_times_ms: list[float]) -> dict:
    """Frame pacing from a list of present intervals in milliseconds.

    `one_percent_low_fps` is the average of the slowest 1% of frames expressed
    as a rate -- the usual benchmarking meaning, and a stutter measure rather
    than a throughput one. Mean FPS can be untouched while the 1% low collapses,
    which is exactly the complaint the recorder drew.
    """
    times = sorted(t for t in frame_times_ms if t and t > 0)
    if not times:
        return {"frames": 0}
    worst_count = max(1, len(times) // 100)
    worst = times[-worst_count:]
    return {
        "frames": len(times),
        "seconds": round(sum(times) / 1000.0, 2),
        "mean_fps": round(1000.0 / statistics.fmean(times), 2),
        "mean_frame_ms": round(statistics.fmean(times), 3),
        "p95_frame_ms": round(times[int(len(times) * 0.95) - 1], 3),
        "p99_frame_ms": round(times[int(len(times) * 0.99) - 1], 3),
        "one_percent_low_fps": round(1000.0 / statistics.fmean(worst), 2),
        "worst_frame_ms": round(times[-1], 3),
    }


def read_present_csv(path: Path) -> list[float]:
    """Present intervals from a PresentMon-style CSV, whatever its column case."""
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    wanted = {"msbetweenpresents", "msbetweenpresent"}
    column = next(
        (k for k in rows[0] if k and k.strip().lower() in wanted),
        None,
    )
    if column is None:
        raise ValueError(
            f"no present-interval column in {path.name}; saw {list(rows[0])[:8]}"
        )
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (TypeError, ValueError):
            continue
    return values


# Where `download` puts it, so no system PATH change is needed.
BUNDLED_TRACER = Path(".local-data/tools/PresentMon.exe")


def present_tracer() -> str | None:
    """PresentMon, from the project folder or PATH."""
    from pipeline_v2.config import REPO_ROOT

    local = REPO_ROOT / BUNDLED_TRACER
    if local.is_file():
        return str(local)
    for name in ("PresentMon", "presentmon"):
        found = shutil.which(name)
        if found:
            return found
    return None


def tracer_command(tracer: str, csv_path: Path, seconds: float) -> list[str]:
    """PresentMon 2.x argument form.

    2.x takes double-dashed options; 1.x took single. `--v1_metrics` pins the
    CSV to the older column set so `read_present_csv` has a stable name to look
    for rather than tracking whichever schema the tracer defaults to.
    """
    return [
        tracer,
        "--process_name", GAME_PROCESS,
        "--output_file", str(csv_path),
        "--timed", str(int(seconds)),
        "--terminate_after_timed",
        "--stop_existing_session",
        "--no_console_stats",
        "--v1_metrics",
    ]


# -------------------------------------------------------------------- sampling


@dataclass
class GpuSampler:
    """Polls nvidia-smi on a thread. Absent hardware is recorded, not fatal."""

    interval_sec: float = 1.0
    samples: list[dict] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"

    def _poll_once(self) -> dict | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self.QUERY}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        keys = ("gpu_util_pct", "vram_used_mb", "vram_total_mb", "temp_c", "power_w")
        sample = {}
        for key, raw in zip(keys, parts):
            try:
                sample[key] = float(raw)
            except ValueError:
                sample[key] = None
        return sample

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._poll_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_sec)

    def start(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0, "note": "nvidia-smi produced no readings"}
        def stat(key):
            values = [s[key] for s in self.samples if s.get(key) is not None]
            if not values:
                return None
            return {"mean": round(statistics.fmean(values), 1),
                    "max": round(max(values), 1)}
        return {
            "samples": len(self.samples),
            "gpu_util_pct": stat("gpu_util_pct"),
            "vram_used_mb": stat("vram_used_mb"),
            "temp_c": stat("temp_c"),
            "power_w": stat("power_w"),
        }


# ------------------------------------------------------------------- a/v drift


def _seconds(value: object) -> float | None:
    """Parse either a plain seconds string or Matroska's HH:MM:SS.nnnnnnnnn."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, secs = (float(p) for p in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + secs


def av_drift_ms(clip: Path) -> dict:
    """How far apart the audio and video streams end.

    Drift is the recorder's own defect, visible without a baseline: a file whose
    audio runs 400ms short of its video is out of sync however the game felt.

    Matroska keeps per-track length in a `DURATION` tag rather than the stream
    `duration` field that MP4 uses, so reading only the latter made every
    recorder chunk look like it had no video at all.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,duration,start_time:stream_tags=DURATION",
         "-of", "json", str(clip)],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        return {"clip": clip.name, "error": "ffprobe failed"}
    streams = json.loads(out.stdout or "{}").get("streams", [])

    def span(kind):
        for s in streams:
            if s.get("codec_type") != kind:
                continue
            length = _seconds(s.get("duration")) or _seconds(
                (s.get("tags") or {}).get("DURATION")
            )
            if length is not None:
                return length, float(s.get("start_time") or 0.0)
        return None, None

    video, video_start = span("video")
    audio, audio_start = span("audio")
    if video is None:
        return {"clip": clip.name, "error": "no video stream"}
    if audio is None:
        return {"clip": clip.name, "video_sec": round(video, 3),
                "error": "no audio stream -- recorder audio is unfinished"}
    return {
        "clip": clip.name,
        "video_sec": round(video, 3),
        "audio_sec": round(audio, 3),
        "start_offset_ms": round((audio_start - video_start) * 1000, 1),
        "duration_drift_ms": round((audio - video) * 1000, 1),
    }


# ------------------------------------------------------------------ comparison


def compare(baseline: dict, capture: dict) -> dict:
    """Apply SCOPE's gates to a paired pair of runs.

    Every gate reports `null` when its input is missing rather than defaulting to
    pass. A gate that silently passes because nothing measured it is worse than
    no gate: that is how the recorder got declared acceptable the first time.
    """
    gates: dict = {}
    base_low = (baseline.get("frames") or {}).get("one_percent_low_fps")
    cap_low = (capture.get("frames") or {}).get("one_percent_low_fps")
    if base_low and cap_low:
        loss = (base_low - cap_low) / base_low
        gate = {
            "baseline_fps": base_low,
            "capture_fps": cap_low,
            "loss": round(loss, 4),
            "limit": MAX_ONE_PERCENT_LOW_LOSS,
            "pass": loss <= MAX_ONE_PERCENT_LOW_LOSS,
        }
        # Capture cannot make a game faster. A negative loss means the two
        # phases were not comparable -- in Fortnite the match stage dominates
        # framerate, so a baseline recorded on the bus and a capture phase
        # recorded late-game differ far more than any capture overhead. Such a
        # pair must not report a pass; it has measured nothing.
        if loss < -NOISE_TOLERANCE:
            gate["pass"] = None
            gate["why"] = (
                f"capture measured {-loss:.1%} FASTER than baseline, which "
                "capture cannot cause. The two phases were not comparable -- "
                "re-record both in the same match stage, doing the same thing."
            )
        gates["one_percent_low"] = gate
    else:
        gates["one_percent_low"] = {
            "pass": None,
            "why": "no frame times; install PresentMon and re-run both phases",
        }

    drifts = [
        d["duration_drift_ms"]
        for d in capture.get("av_drift", [])
        if d.get("duration_drift_ms") is not None
    ]
    if drifts:
        worst = max(drifts, key=abs)
        gates["av_drift"] = {
            "worst_ms": worst,
            "limit_ms": MAX_AV_DRIFT_MS,
            "clips": len(drifts),
            "pass": abs(worst) <= MAX_AV_DRIFT_MS,
        }
    else:
        gates["av_drift"] = {
            "pass": None,
            "why": "no clip carried both an audio and a video stream",
        }

    evaluated = [g for g in gates.values() if g.get("pass") is not None]
    return {
        "created_utc": utc_now(),
        "baseline_backend": baseline.get("backend"),
        "capture_backend": capture.get("backend"),
        "gates": gates,
        "gates_evaluated": len(evaluated),
        "gates_total": len(gates),
        # A pass requires EVERY gate to have been evaluated and passed. One
        # measured gate must never certify an unmeasured one: a run whose frame
        # timing could not be compared has not shown capture to be acceptable,
        # however well it did on drift.
        "verdict": (
            "fail"
            if any(g.get("pass") is False for g in gates.values())
            else "pass"
            if len(evaluated) == len(gates) and all(g["pass"] for g in evaluated)
            else "inconclusive"
        ),
        "note": (
            "A paired result. It describes this machine, this session and this "
            "game build, and says nothing about any other configuration."
        ),
    }


# ------------------------------------------------------------------- recording


def measure(label: str, seconds: float, out: Path, *, clips: Path | None = None) -> dict:
    """Record one labelled phase. Run once with capture off, once with it on."""
    from pipeline_v2.capture import windows

    out.mkdir(parents=True, exist_ok=True)
    try:
        running = windows.game_running()
    except (OSError, RuntimeError):
        running = None

    tracer = present_tracer()
    present_csv = out / f"{label}-present.csv"
    process = tracer_note = None
    if tracer:
        process = subprocess.Popen(
            tracer_command(tracer, present_csv, seconds),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    sampler = GpuSampler().start()
    print(f"measuring '{label}' for {seconds:.0f}s"
          + ("" if tracer else "  (no PresentMon: frame times will be missing)"))
    time.sleep(seconds)
    gpu = sampler.stop()
    if process is not None:
        try:
            chatter = process.communicate(timeout=30)[0] or ""
        except subprocess.TimeoutExpired:
            process.kill()
            chatter = ""
        # PresentMon exits 0 even when it could not start, so the message is the
        # only signal that the run produced nothing.
        if "access denied" in chatter.lower() or "elevated" in chatter.lower():
            tracer_note = (
                "PresentMon needs an elevated terminal (it opens an ETW trace "
                "session). Re-run this from PowerShell started as Administrator."
            )

    frames = {}
    if present_csv.exists():
        try:
            frames = frame_metrics(read_present_csv(present_csv))
        except (ValueError, OSError) as exc:
            frames = {"error": str(exc)}

    # Which backend produced the footage, so a verdict is never read against
    # the wrong one. DDA failed this gate at 55.2%; WGC is the replacement.
    backend = os.environ.get("SALIENCE_CAPTURE_BACKEND", "dda")

    drift = []
    if clips and clips.is_dir():
        for clip in sorted(clips.rglob("*.mkv"))[:10]:
            drift.append(av_drift_ms(clip))

    record = {
        "label": label,
        "created_utc": utc_now(),
        "seconds": seconds,
        "game_running": running,
        "backend": backend,
        "present_tracer": tracer,
        "tracer_note": tracer_note,
        "frames": frames,
        "gpu": gpu,
        "av_drift": drift,
    }
    write_json(out / f"{label}.json", record)
    return record


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def paired_runs(
    out: Path,
    rounds: int,
    seconds: float,
    *,
    window: bool,
    warmup: bool = True,
    fps: int = 30,
    scale: str | None = None,
) -> dict:
    """Alternate baseline and capture `rounds` times, then compare medians.

    One pair cannot answer this. Baseline 1% lows on this machine ranged 26.6
    to 37.3 fps with **nothing recording** -- in Fortnite the match stage
    dominates framerate, so a single pair measures whichever stages it happened
    to land in. Alternating and taking medians is what makes a 20-point
    difference distinguishable from that noise.

    Each capture round spawns the recorder as a **subprocess**: a second WGC
    session in one process segfaults, so sessions cannot be hosted in-process.
    """
    import sys

    out.mkdir(parents=True, exist_ok=True)
    recorder_script = REPO_ROOT / ".local-data" / "tools" / "wgc_record.py"
    footage = out / "footage"
    results: list[dict] = []

    for index in range(1, rounds + 1):
        print(f"\n--- round {index} of {rounds} ---")
        print("  BASELINE: play normally, nothing is recording")
        base = measure(f"baseline-{index}", seconds, out)
        _print_phase(base)

        print("  CAPTURE: recorder starting, keep playing the same way")
        command = [
            sys.executable, str(recorder_script),
            "--fps", str(fps), "--out", str(footage),
        ]
        if window:
            command.append("--window")
        # Scale reaches the recorder through the environment, the same way a
        # manual run sets it, so both paths build an identical command.
        environment = dict(os.environ)
        if scale:
            environment["SALIENCE_CAPTURE_SCALE"] = scale
        recorder = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=environment,
        )
        time.sleep(4)  # let it attach and start writing before measuring
        if recorder.poll() is not None:
            print("  recorder exited early:")
            print("   ", (recorder.communicate()[0] or "").strip()[:400])
            return {"error": "recorder would not start"}
        try:
            cap = measure(f"capture-{index}", seconds, out, clips=footage)
        finally:
            recorder.terminate()
            try:
                recorder.wait(timeout=20)
            except subprocess.TimeoutExpired:
                recorder.kill()
        _print_phase(cap)
        results.append({"baseline": base, "capture": cap})

    lows = [
        (r["baseline"]["frames"].get("one_percent_low_fps"),
         r["capture"]["frames"].get("one_percent_low_fps"))
        for r in results
    ]
    usable = [(b, c) for b, c in lows if b and c]
    # Discarding the first round is declared here, before any data exists --
    # doing it after seeing the numbers would be choosing the answer. The
    # reason is mechanical: the first 90 seconds include shader compilation,
    # asset streaming and the drop, which is stutter belonging to the session
    # rather than to capture. Measured on 2026-09-23 the warm-up baseline had a
    # 1% low of 35.2 fps against 88.9 and 95.8 for the rounds that followed,
    # while mean fps barely moved (157 vs 163) -- a mean would have hidden it.
    discarded = None
    if warmup and len(usable) > 2:
        discarded, usable = usable[0], usable[1:]
    summary = {
        "created_utc": utc_now(),
        "rounds": rounds,
        "seconds_per_phase": seconds,
        # Recorded so a result can never be read against the wrong settings.
        "target": "game window" if window else "monitor",
        "fps": fps,
        "scale": scale,
        "pairs": [{"baseline_1pct": b, "capture_1pct": c} for b, c in lows],
        "baseline_median": median([b for b, _ in usable]),
        "capture_median": median([c for _, c in usable]),
        "warmup_discarded": (
            {"baseline_1pct": discarded[0], "capture_1pct": discarded[1]}
            if discarded
            else None
        ),
        "scored_rounds": len(usable),
    }
    if usable:
        base_med, cap_med = summary["baseline_median"], summary["capture_median"]
        loss = (base_med - cap_med) / base_med
        summary["loss"] = round(loss, 4)
        summary["limit"] = MAX_ONE_PERCENT_LOW_LOSS
        summary["pass"] = bool(loss <= MAX_ONE_PERCENT_LOW_LOSS)
        # Spread across baselines is the noise floor; a difference smaller than
        # it is not a measurement.
        baselines = [b for b, _ in usable]
        # The same guard `compare` applies, which this path originally skipped
        # and so reported PASS on runs where capture measured *faster* than
        # baseline in every single round. Capture cannot speed a game up; such
        # a run measured gameplay variation, not capture cost.
        faster = sum(1 for b, c in usable if c > b)
        if loss < -NOISE_TOLERANCE:
            summary["pass"] = None
            summary["why"] = (
                f"capture measured {-loss:.1%} FASTER than baseline, which "
                "capture cannot cause -- this run measured gameplay variation."
            )
        elif faster:
            summary["pass"] = None
            summary["why"] = (
                f"{faster} of {len(usable)} scored rounds had capture faster "
                "than baseline, so the pairs are not comparable."
            )
        summary["rounds_capture_faster"] = faster
        summary["baseline_spread"] = (
            round((max(baselines) - min(baselines)) / max(baselines), 4)
        )
    write_json(out / "paired.json", summary)
    return summary


def _print_phase(record: dict) -> None:
    frames = record.get("frames") or {}
    if frames.get("frames"):
        print(f"    {frames['mean_fps']} fps mean, {frames['one_percent_low_fps']} fps 1% low")
    else:
        print(f"    no frame times: {record.get('tracer_note') or 'unknown'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("baseline", "capture"):
        run = sub.add_parser(name, help=f"measure the '{name}' phase")
        run.add_argument("out", type=Path)
        run.add_argument("--seconds", type=float, default=120.0)
        run.add_argument(
            "--clips", type=Path, default=None,
            help="folder of recorder output, for A/V drift",
        )
    paired = sub.add_parser(
        "paired", help="alternate baseline/capture N times and compare medians"
    )
    paired.add_argument("out", type=Path)
    paired.add_argument("--rounds", type=int, default=3)
    paired.add_argument("--seconds", type=float, default=90.0)
    paired.add_argument("--fps", type=int, default=30)
    paired.add_argument(
        "--scale", default=None, help="downscale before encoding, e.g. 1280x800"
    )
    paired.add_argument(
        "--no-warmup", dest="warmup", action="store_false",
        help="score the first round too (it normally holds the drop and "
             "shader compilation, which is session stutter, not capture cost)",
    )
    paired.add_argument(
        "--window", action="store_true",
        help="capture only the game window instead of the whole monitor",
    )

    verdict = sub.add_parser("report", help="compare the two phases")
    verdict.add_argument("out", type=Path)

    args = parser.parse_args()
    # Anchor a relative path to the repo, not the shell's cwd. The elevated
    # PowerShell used for PresentMon starts in system32, and a relative path
    # there silently became a second, separate results directory -- the baseline
    # landed in the project and the capture phase did not, so `report` could not
    # find its pair.
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out = out.resolve()
    if args.command == "paired":
        result = paired_runs(
            out, args.rounds, args.seconds, window=args.window,
            warmup=args.warmup, fps=args.fps, scale=args.scale,
        )
        if result.get("error"):
            print(f"\n{result['error']}")
            return 1
        print("\n================ RESULT ================")
        print(
            f"  target {result['target']}  fps {result['fps']}"
            + (f"  scale {result['scale']}" if result.get("scale") else "")
        )
        for index, pair in enumerate(result["pairs"], start=1):
            note = ""
            if result.get("warmup_discarded") and index == 1:
                note = "   (warm-up, not scored)"
            print(f"  round {index}: baseline {pair['baseline_1pct']} "
                  f"-> capture {pair['capture_1pct']} fps 1% low{note}")
        if result.get("pass") is None:
            reason = result.get("why")
            if reason:
                print(f"\n  INCONCLUSIVE: {reason}")
                print(f"  baseline spread {result['baseline_spread']:.1%} "
                      "-- run longer phases, or hold the scene constant.")
            else:
                print("\n  no frame times captured -- run from an elevated "
                      "PowerShell")
            return 1
        print(f"\n  median baseline  {result['baseline_median']} fps")
        print(f"  median capture   {result['capture_median']} fps")
        print(f"  loss             {result['loss']:.1%}  (gate {result['limit']:.0%})")
        print(f"  baseline spread  {result['baseline_spread']:.1%}  <- the noise floor")
        print(f"\n  {'PASS' if result['pass'] else 'FAIL'}")
        if result["baseline_spread"] > abs(result["loss"]):
            print("  NOTE: the baselines vary more than the measured effect; "
                  "run more rounds before trusting this.")
        return 0

    if args.command in ("baseline", "capture"):
        record = measure(args.command, args.seconds, out, clips=args.clips)
        frames = record["frames"]
        if frames.get("frames"):
            print(f"  mean {frames['mean_fps']} fps, 1% low {frames['one_percent_low_fps']} fps")
        else:
            print(f"  no frame times captured: {record.get('tracer_note') or 'see above'}")
        print(f"  gpu {record['gpu']}")
        return 0

    baseline_path, capture_path = out / "baseline.json", out / "capture.json"
    missing = [p.name for p in (baseline_path, capture_path) if not p.exists()]
    if missing:
        print(f"missing {', '.join(missing)} -- record both phases first")
        return 1
    result = compare(read_json(baseline_path), read_json(capture_path))
    write_json(out / "verdict.json", result)
    print(json.dumps(result["gates"], indent=2))
    print(f"\nverdict: {result['verdict'].upper()}  "
          f"({result['gates_evaluated']}/{result['gates_total']} gates evaluated)")
    print(result["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
