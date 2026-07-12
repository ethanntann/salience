from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3

from salience_api.clips.indexer import index_known_clip_path, rescore_all_clips
from salience_api.feedback.service import record_feedback


@dataclass(frozen=True)
class LabeledClipRow:
    path: Path
    label: int


@dataclass(frozen=True)
class ImportLabelsResult:
    imported: int
    keepers: int
    skips: int
    ignored: int


def iter_labeled_clip_rows(path: Path) -> Iterator[LabeledClipRow]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            label = payload.get("label")
            clip_path = payload.get("path")
            if label not in {1, -1} or not clip_path:
                continue
            yield LabeledClipRow(path=Path(str(clip_path)), label=int(label))


def import_labeled_jsonl(
    conn: sqlite3.Connection,
    jsonl_path: Path,
    *,
    source_label: str | None = None,
) -> ImportLabelsResult:
    source = source_label or f"supervised:{jsonl_path.expanduser().resolve()}"
    conn.execute(
        """
        delete from feedback_events
        where label = ? and action in ('supervised_keep', 'supervised_skip')
        """,
        (source,),
    )

    imported = 0
    keepers = 0
    skips = 0
    seen_paths: set[str] = set()
    for row in iter_labeled_clip_rows(jsonl_path):
        key = str(row.path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        clip_id = index_known_clip_path(conn, row.path, source="supervised")
        if row.label == 1:
            action = "supervised_keep"
            keepers += 1
        else:
            action = "supervised_skip"
            skips += 1
        record_feedback(conn, clip_id=clip_id, action=action, label=source)
        imported += 1

    rescore_all_clips(conn)
    return ImportLabelsResult(
        imported=imported,
        keepers=keepers,
        skips=skips,
        ignored=max(
            0, sum(1 for _ in jsonl_path.open("r", encoding="utf-8")) - imported
        ),
    )
