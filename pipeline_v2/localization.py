"""Opt-in offline VLM event proposals from complete saved chunks; never starts capture."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import time

from pipeline_v2 import cache, config, frames, prompt, vlm
from pipeline_v2 import freeze
from pipeline_v2.jsonio import read_json, utc_now, write_json

PROMPT_VERSION = "localization-v1-dev"

INSTRUCTIONS = """Locate visible completed events in these chronologically ordered, timestamped gameplay frames.
Return JSON: {"events": [{"start": <seconds>, "end": <seconds>, "kind": "elimination|death|victory", "actor": "recording_player|spectated_player|unknown", "evidence": "<short visible evidence>"}]}.
Return an empty events list when no completed event is visible. Maximum three events.
Use start/end to bracket the decisive action and first visible confirmation, NOT the entire fight or a later persistent banner. Respect the supplied timestamps; never invent sub-frame precision. Merge victory and its final elimination into one victory event. Do not repeat a persistent banner as a new kill.
Require opponent dissolution, matching personal elimination confirmation, own death, or victory. Gunfire, edits, shield cracks, loot alone, and unrelated kill-feed entries are insufficient. Do not assume every recording contains an event.
The visible avatar can be a spectated player. Spectator/placement/queue overlays indicate ownership; Finding Next Match does not mean the watched match is a pregame lobby. Explicitly distinguish recording_player from spectated_player; use unknown if not supported.
"""


def validate_events(payload: dict, start: float, end: float) -> list[dict]:
    events = payload.get("events")
    if not isinstance(events, list) or len(events) > 3:
        raise ValueError("Invalid event list")
    for e in events:
        if not isinstance(e, dict):
            raise ValueError("Invalid event")
        a, b = e.get("start"), e.get("end")
        if (
            any(type(t) not in (int, float) or not math.isfinite(t) for t in (a, b))
            or not start <= a < b <= end
        ):
            raise ValueError("Event interval outside supplied frames")
        if e.get("kind") not in ("elimination", "death", "victory") or e.get(
            "actor"
        ) not in ("recording_player", "spectated_player", "unknown"):
            raise ValueError("Invalid event type or actor")
        if not isinstance(e.get("evidence"), str) or not e["evidence"].strip():
            raise ValueError("Missing evidence")
    return events


def shot_window(event: dict, duration: float) -> tuple[float, float] | None:
    """Propose evidence, never a keep decision; broad or spectator events require review."""
    if (
        event.get("kind") != "elimination"
        or event.get("timing_adjustment")
        == "confirmation_already_visible_at_first_frame"
        or event.get("actor") != "recording_player"
        or event["end"] - event["start"] > 0.75
    ):
        return None
    return max(0.1, event["start"] - 1.0), min(duration - 0.1, event["end"] + 0.15)


def bracket_point_events(payload: dict, timestamps: list[float]) -> dict:
    """Retain point confirmations as sampled brackets, without inventing precision."""
    result = copy.deepcopy(payload)
    if not isinstance(result, dict) or not isinstance(result.get("events"), list):
        return result
    for event in result["events"]:
        if not isinstance(event, dict):
            continue
        a, b = event.get("start"), event.get("end")
        if type(a) not in (int, float) or not math.isfinite(a) or a != b:
            continue
        index = next((i for i, t in enumerate(timestamps) if abs(t - a) < 0.001), None)
        if index is None or len(timestamps) < 2:
            continue  # Off-grid/out-of-range points remain invalid.
        if index:
            event.update(
                start=timestamps[index - 1],
                end=timestamps[index],
                timing_adjustment="point_confirmation_bracketed_by_previous_sample",
            )
        else:
            event.update(
                start=timestamps[0],
                end=timestamps[1],
                timing_adjustment="confirmation_already_visible_at_first_frame",
            )
    return result


def locate(
    clip: Path,
    out: Path,
    max_calls: int = 0,
    *,
    call_budget: vlm.CallBudget | None = None,
):
    out.mkdir(parents=True, exist_ok=True)
    if (clip.parent / "capture-run.json").exists() and not clip.with_suffix(
        ".capture.json"
    ).exists():
        raise ValueError("Only finalized chunks can be analyzed")
    cfg = replace(config.load_config(), prompt_version=PROMPT_VERSION)
    cfg = config.resolve(cfg)
    fingerprint = freeze.fingerprint(
        cfg,
        prompt.SYSTEM + INSTRUCTIONS,
        frames.FRAME_WIDTH,
        clip=str(clip),
        clip_sha=cache.clip_sha(clip),
    )
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        freeze.check(plan, fingerprint, "localization")
    else:
        duration = frames.probe_duration(clip)
        if duration is None or duration <= 0.2 or duration > 60:
            raise ValueError("Pilot requires a finalized chunk of 0.2–60 seconds")
        plan = {
            **fingerprint,
            "created_utc": utc_now(),
            "duration": duration,
            "design": "16 uniformly spaced full-chunk frames, then 19 frames around each proposed interval expanded by 1 second; no manual event time input.",
        }
        write_json(plan_path, plan)
    used = 0

    def stage(name, start, end, count):
        nonlocal used
        target = out / (name + ".json")
        if target.exists():
            inputs = read_json(out / (name + "-inputs.json"))["images"]
            if any(
                hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() != f["sha256"]
                for f in inputs
            ):
                raise ValueError("Frozen localization frames changed")
            return validate_events(
                read_json(target), inputs[0]["timestamp"], inputs[-1]["timestamp"]
            )
        images = frames.window_frames(
            clip,
            plan["duration"],
            (start + end) / 2,
            pre_sec=(end - start) / 2,
            post_sec=(end - start) / 2,
            samples=count,
            out_dir=out / name,
        )
        if len(images) != count:
            raise ValueError("Incomplete frame extraction")
        content = [{"type": "text", "text": INSTRUCTIONS}]
        for i, f in enumerate(images):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Frame {i + 1}, time {f.timestamp_sec:.3f} seconds:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": prompt._data_url(f.path)},
                    },
                ]
            )
        write_json(
            out / (name + "-inputs.json"),
            {
                "images": [
                    {
                        "path": str(f.path),
                        "timestamp": f.timestamp_sec,
                        "sha256": hashlib.sha256(f.path.read_bytes()).hexdigest(),
                    }
                    for f in images
                ]
            },
        )
        result, cached = vlm.request_raw(
            cfg, clip, content, allow_call=used < max_calls, call_budget=call_budget
        )
        used += not cached
        # Raw transport persists before validation, so parsing never re-bills.
        events = validate_events(
            bracket_point_events(
                vlm._extract_json(result.raw), [f.timestamp_sec for f in images]
            ),
            images[0].timestamp_sec,
            images[-1].timestamp_sec,
        )
        write_json(
            target,
            {
                **asdict(result),
                "cached": cached,
                "events": events,
                "response_interpretation": "sample-bracket-v1",
            },
        )
        if not cached:
            time.sleep(1.5)
        return events

    coarse = stage("coarse", 0.1, plan["duration"] - 0.1, 16)
    refined = []
    for i, e in enumerate(coarse):
        refined.extend(
            stage(
                f"refine-{i}",
                max(0.1, e["start"] - 1),
                min(plan["duration"] - 0.1, e["end"] + 1),
                19,
            )
        )
    write_json(
        out / "proposals.json",
        {
            "clip": str(clip),
            "coarse": coarse,
            "refined": refined,
            "shot_windows": [
                {"event": e, "start": bounds[0], "end": bounds[1]}
                for e in refined
                if (bounds := shot_window(e, plan["duration"])) is not None
            ],
            "new_calls_this_run": used,
            "note": "Model proposals, not verified events or wanted highlights. Empty is valid. No automatic clip export or capture.",
        },
    )
    return refined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    print(locate(args.clip.resolve(), args.out.resolve(), args.max_calls))
