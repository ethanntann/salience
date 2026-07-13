"""Repair promoted teacher labels using verified finish events and strong HUD OCR.

Run with ``PYTHONPATH=backend``. The operation is append-only for teacher label
and event rows; live assignments are moved to the reconciled rows atomically.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from salience_api.clips.indexer import (
    record_teacher_events,
    record_teacher_labels,
    rescore_all_clips,
)
from salience_api.features.fireworks_teacher import WeaponAttribution
from salience_api.features.teacher_labels import normalize_teacher_payload
from salience_api.ranking.highlights import attach_event_audit, build_highlight_profile


WEAPON_LABELS = (
    "sniper_kill",
    "shotgun_kill",
    "shotgun_one_pump",
    "pistol_kill",
    "automatic_kill",
    "other_weapon_kill",
    "spray_kill",
    "no_scope",
)
WEAPON_CATEGORIES = {
    "sniper_or_hunting",
    "shotgun",
    "pistol",
    "automatic",
    "other",
}


def _prefer_strong_local_ocr(events: list[dict]) -> list[dict]:
    corrected = deepcopy(events)
    for event in corrected:
        ocr = event.get("local_ocr")
        if event.get("status") != "attributed" or not isinstance(ocr, dict):
            continue
        category = str(ocr.get("category", "unknown"))
        confidence = float(ocr.get("confidence") or 0.0)
        if (
            ocr.get("ambiguous")
            or confidence < 0.9
            or category not in WEAPON_CATEGORIES
        ):
            continue
        event["resolved_weapon"] = category
        event["selected_weapon_name_text"] = str(
            ocr.get("text") or event.get("selected_weapon_name_text") or "unknown"
        )
        event["local_ocr"] = {**ocr, "applied": True, "reason": "strong_pre_finish_hud"}
    return corrected


def _reconciled_labels(label_payload: dict, events: list[dict]) -> dict:
    profile = build_highlight_profile(events)
    active_events = [
        event
        for event in (profile.primary, *profile.secondary)
        if event is not None and event.active
    ]
    downed_events = [
        event
        for event in (profile.primary, *profile.secondary)
        if event is not None and event.downed
    ]
    active_labels = {
        label for event in active_events for label in event.labels
    }
    reconciled = dict(label_payload)
    for key in WEAPON_LABELS:
        reconciled[key] = "yes" if key in active_labels else "no"
    reconciled["elimination_or_knock"] = "yes" if active_events else "no"
    reconciled["downed_finish"] = "yes" if downed_events else "no"
    reconciled["multi_kill"] = "yes" if profile.multi_kill else "no"
    reconciled["high_damage_hit"] = (
        "yes" if "high_damage_hit" in active_labels else "no"
    )
    if reconciled["sniper_kill"] != "yes":
        reconciled["stationary_sniper_target"] = "no"
    return normalize_teacher_payload(reconciled).model_dump()


def reconcile(database: Path, *, dry_run: bool = False) -> dict[str, int]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        select
            live.clip_id,
            tl.provider,
            tl.label_json,
            te.model,
            te.status,
            te.event_json
        from live_teacher_assignments live
        join teacher_labels tl on tl.id = live.label_row_id
        join teacher_events te on te.id = live.event_row_id
        order by live.clip_id
        """
    ).fetchall()
    stats = {
        "examined": len(rows),
        "changed": 0,
        "sniper_removed": 0,
        "multi_weapon_reduced": 0,
    }
    try:
        for row in rows:
            original = json.loads(row["label_json"] or "{}")
            event_data = json.loads(row["event_json"] or "{}")
            raw_events = event_data.get("events")
            if not isinstance(raw_events, list) or not raw_events:
                continue
            events = _prefer_strong_local_ocr(raw_events)
            reconciled = _reconciled_labels(original, events)
            original_normalized = normalize_teacher_payload(original).model_dump()
            if reconciled == original_normalized:
                continue

            stats["changed"] += 1
            if original_normalized.get("sniper_kill") == "yes" and reconciled.get(
                "sniper_kill"
            ) != "yes":
                stats["sniper_removed"] += 1
            original_weapon_count = sum(
                original_normalized.get(key) == "yes" for key in WEAPON_LABELS[:6]
            )
            reconciled_weapon_count = sum(
                reconciled.get(key) == "yes" for key in WEAPON_LABELS[:6]
            )
            if reconciled_weapon_count < original_weapon_count:
                stats["multi_weapon_reduced"] += 1
            if dry_run:
                continue

            labels = normalize_teacher_payload(reconciled)
            model = f"{row['model']}-weapon-reconciled"
            attribution = WeaponAttribution(
                labels={key: reconciled[key] for key in WEAPON_LABELS},
                confidence=labels.confidence,
                evidence=labels.evidence,
                status=str(row["status"]),
                events=events,
                raw_payload={},
            )
            corrected_event_data = attach_event_audit(
                {**event_data, "events": events}, attribution
            )
            record_teacher_labels(
                connection,
                clip_id=int(row["clip_id"]),
                provider=str(row["provider"]),
                labels=labels,
                model=model,
                update_live_assignment=True,
            )
            record_teacher_events(
                connection,
                clip_id=int(row["clip_id"]),
                provider=str(row["provider"]),
                model=model,
                status=str(row["status"]),
                event_data=corrected_event_data,
                update_live_assignment=True,
            )

        if dry_run:
            connection.rollback()
        else:
            rescore_all_clips(connection)
            connection.commit()
        return stats
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reconcile(args.database, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
