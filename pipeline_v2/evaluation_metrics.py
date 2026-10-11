"""Report pilot diagnostics and user-labelled recall without inventing labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import re

from pipeline_v2.jsonio import read_json, write_json


def merged(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1] + 0.001:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def event_metrics(intervals, events):
    """Positive temporal overlap; adjacent candidate windows count only once."""
    proposals = merged(intervals)

    def hits(a, b):
        return max(a[0], b["start"]) < min(a[1], b["end"])

    found = sum(any(hits(p, e) for p in proposals) for e in events)
    useful = sum(any(hits(p, e) for e in events) for p in proposals)
    return {
        "proposals": proposals,
        "events": len(events),
        "events_found": found,
        "recall": found / len(events) if events else None,
        "precision": useful / len(proposals) if proposals and events else None,
    }


def report(out: Path):
    plan = read_json(out / "plan.json")
    if plan.get("study_kind") == "continuous_localization":
        from pipeline_v2.evaluation_session import report as session_report

        return session_report(out)
    if plan.get("study_kind") == "curated_action":
        return curated_report(out, plan)
    windows = plan["windows"]
    predictions = {w["id"]: read_json(out / "results" / f"{w['id']}.json") for w in windows}
    ranking = read_json(out / "ranking.json")
    scores = {p["id"]: p["score"] for p in ranking["scores"]}
    reference = read_json(out / "assistant-adjudicated.json")
    spectator = [r for r in reference["labels"] if r["context"] == "spectating"]
    errors = [
        r["id"]
        for r in spectator
        if re.search(
            r"pre[ -]?(?:match|game)|lobby",
            predictions[r["id"]]["description"]["summary"],
            re.I,
        )
    ]
    sweep = [w for w in windows if w["cohort"] == "continuous"]
    top = sorted(sweep, key=lambda w: (-scores[w["id"]], w["id"]))[:3]
    baseline = {
        "ranked": top,
        "chronological": sweep[:3],
        "random_seed_0": random.Random(0).sample(sweep, 3),
    }
    candidates = {
        key: [(w["timeline_start"], w["timeline_end"]) for w in group]
        for key, group in baseline.items()
    }
    sparse = []
    for clip in dict.fromkeys(w["clip"] for w in sweep):
        first = next(w for w in sweep if w["clip"] == clip)
        sidecar = read_json(Path(clip).with_suffix(".capture.json"))
        sparse.append(
            (
                first["timeline_start"] + sidecar["analysis_start_sec"],
                first["timeline_start"] + sidecar["analysis_end_sec"],
            )
        )
    # Diagnostic events include an NPC kill, own death, and a spectator's kill.
    # They are not presumed to be wanted highlights.
    diagnostics = {
        key: event_metrics(value, reference["events"])
        for key, value in candidates.items()
    }
    diagnostics["current_one_uniform_window_per_chunk"] = event_metrics(
        sparse, reference["events"]
    )
    user = (
        read_json(out / "user-labels.json")
        if (out / "user-labels.json").exists()
        else {"labels": {}, "events": []}
    )
    # Only a user explicitly confirming full coverage permits recall calculation.
    user_metrics = (
        {key: event_metrics(value, user["events"]) for key, value in candidates.items()}
        if user.get("timeline_review_complete")
        else None
    )
    human_review = {}
    for cohort in sorted({w["cohort"] for w in windows}):
        labels = [
            user["labels"][w["id"]]
            for w in windows
            if w["cohort"] == cohort and w["id"] in user["labels"]
        ]
        human_review[cohort] = {
            "reviewed": len(labels),
            "preferences": dict(Counter(r["keep"] for r in labels)),
            "description_ratings": dict(
                Counter(r.get("accuracy", "unreviewed") for r in labels)
            ),
        }
    result = {
        "responses": len(predictions),
        "high_confidence_responses": sum(
            p["description"]["confidence"] == "high" for p in predictions.values()
        ),
        "input_tokens": sum(p["input_tokens"] for p in predictions.values()),
        "output_tokens": sum(p["output_tokens"] for p in predictions.values()),
        "estimated_cost_usd": sum(p["cost_usd"] for p in predictions.values()),
        "price_source": "https://fireworks.ai/blog/qwen-3p7-plus",
        "price_note": "Conservative higher published price pair: $0.50/M input, $3/M output; estimate, not invoice.",
        "spectator_windows": len(spectator),
        "explicit_spectator_as_lobby_errors": errors,
        "overall_description_accuracy": None,
        "description_accuracy_note": "User ratings are reported by cohort below. Partial review of predominantly background footage does not establish overall description accuracy or action recall.",
        "human_review_by_cohort": human_review,
        "ranker": {
            k: v for k, v in ranking.items() if k not in ("scores", "training_ids")
        },
        "top_three_continuous": [
            {
                "id": w["id"],
                "score": scores[w["id"]],
                "start": w["timeline_start"],
                "end": w["timeline_end"],
            }
            for w in top
        ],
        "provisional_event_diagnostics": diagnostics,
        "diagnostic_caveat": "Three visually identified events, with post-prediction NPC adjudication. Temporal overlap only, no clip quality or event ownership requirement. Not highlight precision/recall. Three windows per ranking baseline, variable count after adjacent-window merging; sparse sampling has six windows.",
        "human_windows_reviewed": len(user["labels"]),
        "user_highlight_metrics": user_metrics,
        "capture_acceptance": "failed: user reports FPS drops and crashing",
    }
    write_json(out / "metrics.json", result)
    return result


def curated_report(out: Path, plan: dict):
    """Case selection enriches action prevalence; never report session recall."""
    windows = plan["windows"]
    user = (
        read_json(out / "user-labels.json")
        if (out / "user-labels.json").exists()
        else {"labels": {}}
    )
    predictions = {w["id"]: read_json(out / "results" / f"{w['id']}.json") for w in windows}
    ranking = read_json(out / "ranking.json")
    reused = set(plan.get("reused_results", []))
    new = [p for ident, p in predictions.items() if ident not in reused]
    cohorts = {}
    for cohort in dict.fromkeys(w["cohort"] for w in windows):
        group = [w for w in windows if w["cohort"] == cohort]
        labels = [user["labels"][w["id"]] for w in group if w["id"] in user["labels"]]
        cohorts[cohort] = {
            "windows": len(group),
            "reviewed": len(labels),
            "preferences": dict(Counter(r["keep"] for r in labels)),
            "description_ratings": dict(
                Counter(r.get("accuracy", "unreviewed") for r in labels)
            ),
        }
    result = {
        "study_kind": "curated_action",
        "human_review_by_cohort": cohorts,
        "new_responses": len(new),
        "reused_responses": len(reused),
        "new_input_tokens": sum(p["input_tokens"] for p in new),
        "new_output_tokens": sum(p["output_tokens"] for p in new),
        "new_estimated_cost_usd": sum(p["cost_usd"] for p in new),
        "user_highlight_metrics": None,
        "overall_description_accuracy": None,
        "diagnostic_caveat": "Manually selected action cases plus reused controls/background. No automatic event discovery, session recall, natural-prevalence precision, or deployment threshold established. User ratings remain outside training.",
        "ranker": {
            k: v for k, v in ranking.items() if k not in ("scores", "training_ids")
        },
        "ranked_windows": [
            {**row, "user_judgment": user["labels"].get(row["id"])}
            for row in sorted(ranking["scores"], key=lambda r: (-r["score"], r["id"]))
        ],
        "capture_acceptance": "failed: user reports FPS drops and crashing; recording paused",
    }
    write_json(out / "metrics.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    print(report(p.parse_args().out.resolve()))
