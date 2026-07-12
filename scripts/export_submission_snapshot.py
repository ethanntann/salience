"""Export a path-free, credential-free seed of the promoted teacher inbox."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def export_snapshot(database: Path, output: Path) -> int:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        select
            c.id, c.duration_sec, c.width, c.height, c.fps, c.size_bytes,
            cf.motion_score, cf.audio_peak_score, cf.silence_ratio,
            cf.action_density, cf.extraction_confidence, cf.feature_json,
            cs.base_score, cs.final_score, cs.confidence,
            cs.explanation as score_explanation,
            tl.provider as teacher_provider,
            tl.label_json
        from live_teacher_assignments live
        join clips c on c.id = live.clip_id
        join clip_features cf on cf.clip_id = c.id
        join clip_scores cs on cs.clip_id = c.id
        join teacher_labels tl on tl.id = live.label_row_id
        order by cs.final_score desc, c.id
        """
    ).fetchall()

    snapshot = []
    for rank, row in enumerate(rows, start=1):
        features = json.loads(row["feature_json"] or "{}")
        teacher_labels = json.loads(row["label_json"])
        # Evidence may contain OCR-read player names. Labels and confidence are
        # sufficient for the public ranked inbox, so do not export raw evidence.
        teacher_labels["evidence"] = []
        snapshot.append(
            {
                "filename": f"teacher-ranked-{rank:03d}.mp4",
                "path": f"snapshot://teacher-ranked-{rank:03d}.mp4",
                "source": "supervised",
                "feature_source": "sanitized_submission_snapshot",
                "duration_sec": row["duration_sec"],
                "width": row["width"],
                "height": row["height"],
                "fps": row["fps"],
                "size_bytes": row["size_bytes"],
                "motion_score": row["motion_score"],
                "audio_peak_score": row["audio_peak_score"],
                "silence_ratio": row["silence_ratio"],
                "action_density": row["action_density"],
                "extraction_confidence": row["extraction_confidence"],
                "tags": features.get("tags", []),
                "thumbnail_variant": features.get("thumbnail_variant", "ridge"),
                "teacher_provider": row["teacher_provider"],
                "teacher_model": "precomputed-submission-snapshot",
                "teacher_labels": teacher_labels,
                "base_score": row["base_score"],
                "final_score": row["final_score"],
                "confidence": row["confidence"],
                "score_explanation": row["score_explanation"],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return len(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = export_snapshot(args.database, args.output)
    print(f"Exported {count} sanitized teacher-ranked clips to {args.output}")


if __name__ == "__main__":
    main()
