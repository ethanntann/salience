from datetime import UTC, datetime
from pathlib import Path
import sqlite3

from salience_api.clips.indexer import index_clip_path
from salience_api.clips.scanner import find_clip_files
from salience_api.feedback.service import record_feedback

TASTE_FEATURES = {
    "shotgun_one_pump",
    "shotgun_kill",
    "pistol_kill",
    "automatic_kill",
    "other_weapon_kill",
    "flick_shot",
    "sniper_kill",
    "no_scope",
    "build_fight",
    "fast_edit",
    "clutch",
    "multi_kill",
    "victory",
    "high_damage_hit",
    "cleanup_kill",
    "downed_finish",
    "spray_kill",
    "competitive_context",
    "stationary_target",
    "stationary_sniper_target",
    "rotation_traversal",
    "looting_or_menu",
    "downtime",
}


def save_taste_preferences(
    conn: sqlite3.Connection, preferences: dict[str, int | float]
) -> int:
    saved = 0
    now = datetime.now(UTC).isoformat()
    for key, value in preferences.items():
        if key not in TASTE_FEATURES:
            continue
        weight = max(-1.0, min(1.0, float(value) / 2.0))
        conn.execute(
            """
            insert into taste_preferences(key, weight, updated_at)
            values (?, ?, ?)
            on conflict(key) do update set
                weight = excluded.weight,
                updated_at = excluded.updated_at
            """,
            (key, weight, now),
        )
        saved += 1
    return saved


def reset_taste_preferences(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("delete from taste_preferences")
    return int(cursor.rowcount)


def import_liked_folder(conn: sqlite3.Connection, folder: Path) -> tuple[int, int]:
    found = find_clip_files(folder)
    imported = 0
    for path in found:
        clip_id = index_clip_path(conn, path)
        record_feedback(conn, clip_id=clip_id, action="keep", label="liked_import")
        imported += 1
    return imported, len(found)
