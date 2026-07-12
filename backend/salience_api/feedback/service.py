from datetime import UTC, datetime
import sqlite3


ACTION_WEIGHTS = {
    "favorite": 1.0,
    "keep": 0.6,
    "tag": 0.4,
    "skip": -0.2,
    "boring": -0.8,
    "delete": -1.0,
    "supervised_keep": 1.0,
    "supervised_skip": -1.0,
}


def feedback_weight(action: str) -> float:
    if action not in ACTION_WEIGHTS:
        raise ValueError(f"Unsupported feedback action: {action}")
    return ACTION_WEIGHTS[action]


def record_feedback(
    conn: sqlite3.Connection,
    *,
    clip_id: int,
    action: str,
    label: str | None,
) -> None:
    conn.execute(
        """
        insert into feedback_events(clip_id, action, label, weight, created_at)
        values (?, ?, ?, ?, ?)
        """,
        (
            clip_id,
            action,
            label,
            feedback_weight(action),
            datetime.now(UTC).isoformat(),
        ),
    )
