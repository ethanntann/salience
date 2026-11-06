"""Frozen development comparison: attribution instructions versus frame density.

Uses already reviewed footage. Never trains, overwrites source judgments, or
claims a held-out result. Existing seven-frame baseline responses are reused.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import time

from pipeline_v2 import cache, config, facts, frames, prompt, vlm
from pipeline_v2 import freeze
from pipeline_v2.jsonio import read_json, utc_now, write_json


@contextmanager
def variant(name: str):
    previous = os.environ.get("V2_ABLATE")
    try:
        if name == "attribution":
            os.environ["V2_ABLATE"] = name
        else:
            os.environ.pop("V2_ABLATE", None)
        yield
    finally:
        if previous is None:
            os.environ.pop("V2_ABLATE", None)
        else:
            os.environ["V2_ABLATE"] = previous


def prompt_sha():
    return freeze.sha256_text(
        prompt.SYSTEM
        + prompt.build_prompt(facts.ClipFacts(event_source="capture_window"))
    )


def _fingerprint(cfg) -> dict:
    return freeze.fingerprint(
        cfg,
        prompt.SYSTEM
        + prompt.build_prompt(facts.ClipFacts(event_source="capture_window")),
        frames.FRAME_WIDTH,
    )


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(arm: dict):
    """Protect both exact model images and any additional source chunks."""
    if cache.clip_sha(Path(arm["clip"])) != arm["clip_sha"] or any(
        file_sha(Path(f["path"])) != f["sha256"] for f in arm["images"]
    ):
        raise ValueError("Frozen comparison input changed")
    for segment in arm.get("source_segments", []):
        if (
            segment.get("sha256")
            and cache.clip_sha(Path(segment["clip"])) != segment["sha256"]
        ):
            raise ValueError("Additional source chunk changed")


def prepare(source: Path, out: Path):
    if (out / "plan.json").exists():
        raise ValueError("Comparison already frozen; use run to resume")
    original = read_json(source / "plan.json")
    prepared = read_json(source / "prepared.json")["windows"]
    selected = [
        w
        for w in prepared
        if w["cohort"] == "action" or w["id"] in ("sample-00", "sample-07")
    ]
    dense_ids = {"action-00", "action-02", "action-04", "sweep-22"}
    arms = []
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    fingerprints = {}
    for name in ("baseline", "attribution"):
        with variant(name):
            cfg = config.load_config()
            fingerprints[name] = {
                "prompt_version": cfg.prompt_version,
                "prompt_sha": prompt_sha(),
            }
    if (
        fingerprints["baseline"]["prompt_sha"] != original["prompt_sha"]
        or cfg.model != original["model"]
        or cfg.base_url != original["base_url"]
        or frames.FRAME_WIDTH != original["frame_width"]
    ):
        raise ValueError("Source baseline differs from current runtime")
    for w in selected:
        clip = Path(w["clip"])
        if cache.clip_sha(clip) != w["clip_sha"]:
            raise ValueError("Source video changed")
        image_sets = {7: [{**f, "path": str(source / f["path"])} for f in w["images"]]}
        if w["id"] in dense_ids:
            midpoint = (w["start"] + w["end"]) / 2
            extracted = frames.window_frames(
                clip,
                frames.probe_duration(clip),
                midpoint,
                pre_sec=midpoint - w["start"],
                post_sec=w["end"] - midpoint,
                samples=19,
                out_dir=out / "frames" / w["id"],
            )
            if len(extracted) != 19:
                raise ValueError("Dense arm requires 19 frames")
            image_sets[19] = [
                {"path": str(f.path), "timestamp": f.timestamp_sec} for f in extracted
            ]
        for count, images in image_sets.items():
            for name in ("baseline", "attribution"):
                ident = f"{w['id']}--{name}-{count}"
                reused = name == "baseline" and count == 7
                arm = {
                    "id": ident,
                    "case_id": w["id"],
                    "variant": name,
                    "frame_count": count,
                    "clip": w["clip"],
                    "clip_sha": w["clip_sha"],
                    "start": w["start"],
                    "end": w["end"],
                    "images": [
                        {**f, "sha256": file_sha(Path(f["path"]))} for f in images
                    ],
                    "reused": reused,
                }
                arms.append(arm)
                if reused:
                    result = read_json(source / "results" / f"{w['id']}.json")
                    write_json(
                        out / "results" / f"{ident}.json",
                        {
                            **result,
                            "id": ident,
                            "source_result": str(
                                source / "results" / f"{w['id']}.json"
                            ),
                        },
                    )
    write_json(out / "user-reference-snapshot.json", read_json(source / "user-labels.json"))
    write_json(
        out / "plan.json",
        {
            "created_utc": utc_now(),
            "study_kind": "attribution_development",
            "source": str(source),
            "source_labels_sha": file_sha(source / "user-labels.json"),
            "session_id": original["session_id"],
            "model": cfg.model,
            "base_url": cfg.base_url,
            "frame_width": frames.FRAME_WIDTH,
            "prompts": fingerprints,
            "arms": arms,
            "expected_new_calls": sum(not a["reused"] for a in arms),
            "design": "Same-frame prompt comparison on ten reviewed action/control cases and two reviewed background cases; 2x2 prompt/density comparison on four attribution failures. Fixed original time bounds, model, resolution, temperature, output limit; 7 versus 19 stills.",
            "assessment": "Compare decisive-hit weapon family, exact name where supported, viewpoint/actor, uncertainty, and evidence-supported edits/piece control. Inspect positive claims and abstentions separately. User says action-00 Drum Gun, action-02 shotgun, action-04 shotgun/piece control, sweep-22 spectator. Other cases check regressions, not just fixes.",
            "limitations": "Known development cases, one session, one response per arm; no end-to-end recall or independent generalization estimate. Assistant post-prediction assessment is not a new user judgment.",
        },
    )
    print(
        f"Frozen {len(arms)} arms; {sum(not a['reused'] for a in arms)} new calls maximum",
        flush=True,
    )


def run(out: Path, max_calls: int):
    plan = read_json(out / "plan.json")
    attempted = 0
    for arm in plan["arms"]:
        target = out / "results" / f"{arm['id']}.json"
        if target.exists():
            continue
        with variant(arm["variant"]):
            cfg = config.load_config()
            current = _fingerprint(cfg)
            freeze.check(plan, current, f"comparison ({arm['variant']} arm)")
            freeze.check(
                plan["prompts"][arm["variant"]],
                current,
                f"comparison prompt ({arm['variant']} arm)",
            )
            validate_inputs(arm)
            cfg = config.resolve(cfg)
            result = vlm.describe(
                cfg,
                Path(arm["clip"]),
                facts.ClipFacts(
                    event_source="capture_window",
                    event_sec=(arm["start"] + arm["end"]) / 2,
                    analysis_start_sec=arm["start"],
                    analysis_end_sec=arm["end"],
                    capture_session=plan["session_id"],
                ),
                [Path(f["path"]) for f in arm["images"]],
                allow_call=attempted < max_calls,
            )
            attempted += not result.cached
            # The client persists raw responses before parsing; never re-pay to fix validation.
            write_json(
                target,
                {"id": arm["id"], "created_utc": utc_now(), **result.model_dump()},
            )
            if any(
                i > arm["frame_count"]
                for i in result.description.weapon_evidence_frames
            ):
                raise ValueError(
                    "Model cited a nonexistent frame; result retained for audit"
                )
            print(f"{arm['id']}: cached={result.cached}", flush=True)
            if not result.cached:
                time.sleep(1.5)


def contract_issues(d: dict, count: int) -> list[str]:
    """Internal consistency only; citations are not verified visual evidence."""
    issues = []
    evidence = d.get("weapon_evidence_frames", [])
    weapon = d.get("elimination_weapon_name", "unknown").strip().lower()
    if any(type(i) is not int or not 1 <= i <= count for i in evidence):
        issues.append("Invalid evidence frame number")
    if d.get("outcome") != "elimination" and (
        weapon not in ("", "unknown") or evidence
    ):
        issues.append(
            "Non-elimination outcome contains decisive-hit weapon attribution"
        )
    if weapon not in ("", "unknown") and not evidence:
        issues.append("Decisive-hit weapon has no cited evidence")
    if (d.get("viewpoint"), d.get("event_actor")) in (
        ("spectating", "recording_player"),
        ("own_gameplay", "spectated_player"),
    ):
        issues.append("Viewpoint and event actor contradict each other")
    return issues


def report(out: Path):
    plan = read_json(out / "plan.json")
    rows, new, missing = [], [], []
    for arm in plan["arms"]:
        path = out / "results" / f"{arm['id']}.json"
        if not path.exists():
            missing.append(arm["id"])
            continue
        result = read_json(path)
        rows.append(
            {
                "id": arm["id"],
                "case_id": arm["case_id"],
                "variant": arm["variant"],
                "frame_count": arm["frame_count"],
                "reused": arm["reused"],
                "description": result["description"],
                "contract_issues": contract_issues(
                    result["description"], arm["frame_count"]
                ),
            }
        )
        if not arm["reused"]:
            new.append(result)
    summary = {
        "responses": len(rows),
        "missing": missing,
        "new_responses": len(new),
        "new_input_tokens": sum(r["input_tokens"] for r in new),
        "new_output_tokens": sum(r["output_tokens"] for r in new),
        "new_estimated_cost_usd": sum(r["cost_usd"] for r in new),
        "contract_issues": {
            r["id"]: r["contract_issues"] for r in rows if r["contract_issues"]
        },
        "visual_accuracy": None,
        "session_recall": None,
        "assessment_note": "Contract checks only. Frame citations are not verified visual evidence. Known development cases, not independent validation.",
    }
    write_json(out / "comparison.json", {**summary, "rows": rows})
    references = read_json(out / "user-reference-snapshot.json")["labels"]
    text = [
        "# Attribution development comparison",
        "",
        plan["design"],
        "",
        plan["limitations"],
        "",
    ]
    for case in dict.fromkeys(r["case_id"] for r in rows):
        text.extend(
            [
                f"## {case}",
                "",
                "User reference: "
                + references.get(case, {}).get("notes", "Not supplied"),
                "",
            ]
        )
        for row in [r for r in rows if r["case_id"] == case]:
            d = row["description"]
            text.extend(
                [
                    f"### {row['variant']}, {row['frame_count']} frames",
                    "",
                    d["summary"],
                    "",
                ]
            )
            for field in (
                "weapon_name",
                "elimination_weapon_name",
                "viewpoint",
                "event_actor",
                "mechanics",
                "confidence",
                "weapon_evidence_frames",
                "attribution_notes",
                "mechanics_evidence",
            ):
                text.extend([f"- {field}: {d.get(field, 'not requested')}", ""])
            text.extend(
                ["Contract issue: " + issue for issue in row["contract_issues"]]
            )
    (out / "comparison.md").write_text("\n".join(text), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("out", type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("out", type=Path)
    execute.add_argument("--max-calls", type=int, default=0)
    summary = sub.add_parser("report")
    summary.add_argument("out", type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.source.resolve(), args.out.resolve())
    elif args.command == "run":
        run(args.out.resolve(), args.max_calls)
    else:
        print(report(args.out.resolve()))


if __name__ == "__main__":
    main()
