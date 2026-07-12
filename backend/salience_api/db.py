from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("pragma foreign_keys = on")
        conn.execute("pragma busy_timeout = 5000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists clips (
            id integer primary key autoincrement,
            path text not null unique,
            filename text not null,
            duration_sec real,
            width integer,
            height integer,
            fps real,
            size_bytes integer,
            created_at text,
            indexed_at text not null,
            source text not null default 'local'
        );

        create table if not exists clip_features (
            clip_id integer primary key references clips(id) on delete cascade,
            motion_score real not null default 0,
            audio_peak_score real not null default 0,
            silence_ratio real not null default 0,
            action_density real not null default 0,
            duration_score real not null default 0,
            extraction_confidence real not null default 0,
            feature_json text not null default '{}'
        );

        create table if not exists clip_scores (
            clip_id integer primary key references clips(id) on delete cascade,
            base_score real not null,
            personal_score real not null,
            final_score real not null,
            confidence real not null,
            explanation text not null,
            scored_at text not null
        );

        create table if not exists feedback_events (
            id integer primary key autoincrement,
            clip_id integer not null references clips(id) on delete cascade,
            action text not null,
            label text,
            weight real not null,
            created_at text not null
        );

        create table if not exists teacher_labels (
            id integer primary key autoincrement,
            clip_id integer not null references clips(id) on delete cascade,
            provider text not null,
            schema_version text not null,
            label_json text not null,
            confidence real not null,
            created_at text not null
        );

        create table if not exists teacher_events (
            id integer primary key autoincrement,
            clip_id integer not null references clips(id) on delete cascade,
            provider text not null,
            model text not null,
            schema_version text not null,
            status text not null,
            event_json text not null,
            created_at text not null
        );

        create table if not exists teacher_event_candidates (
            id integer primary key autoincrement,
            clip_id integer not null references clips(id) on delete cascade,
            provider text not null,
            model text not null,
            candidate_version text not null,
            status text not null,
            label_json text not null,
            event_json text not null,
            created_at text not null
        );

        create table if not exists user_ranker_metadata (
            id integer primary key check (id = 1),
            feedback_count integer not null default 0,
            positive_count integer not null default 0,
            negative_count integer not null default 0,
            updated_at text not null
        );

        create table if not exists taste_preferences (
            key text primary key,
            weight real not null,
            updated_at text not null
        );

        create table if not exists teacher_label_reviews (
            id integer primary key autoincrement,
            clip_id integer not null references clips(id) on delete cascade,
            label_key text not null,
            expected_value text not null check (expected_value in ('yes', 'no', 'uncertain')),
            notes text,
            created_at text not null
        );

        create index if not exists idx_teacher_label_reviews_clip_label
            on teacher_label_reviews(clip_id, label_key, id);

        create index if not exists idx_teacher_labels_schema_clip_id
            on teacher_labels(schema_version, clip_id, id);

        create index if not exists idx_teacher_events_schema_clip_id
            on teacher_events(schema_version, clip_id, id);

        create index if not exists idx_teacher_event_candidates_version_clip_id
            on teacher_event_candidates(candidate_version, clip_id, id);

        create table if not exists teacher_pipeline_manifests (
            id integer primary key autoincrement,
            pipeline_key text not null unique,
            provider text not null,
            model text not null,
            prompt_hash text not null,
            revision text not null,
            config_json text not null,
            created_at text not null
        );

        create table if not exists evaluation_batches (
            id integer primary key autoincrement,
            batch_key text not null unique,
            pipeline_manifest_id integer not null references teacher_pipeline_manifests(id),
            candidate_version text not null,
            status text not null,
            created_at text not null,
            updated_at text not null
        );

        create table if not exists prediction_snapshots (
            id integer primary key autoincrement,
            clip_id integer not null references clips(id) on delete cascade,
            batch_id integer references evaluation_batches(id) on delete cascade,
            pipeline_manifest_id integer not null references teacher_pipeline_manifests(id),
            candidate_id integer references teacher_event_candidates(id),
            label_json text not null,
            event_json text not null,
            frames_hash text,
            ocr_hash text,
            frozen_at text not null
        );

        create table if not exists evaluation_batch_items (
            id integer primary key autoincrement,
            batch_id integer not null references evaluation_batches(id) on delete cascade,
            clip_id integer not null references clips(id) on delete cascade,
            candidate_id integer references teacher_event_candidates(id),
            prediction_snapshot_id integer not null references prediction_snapshots(id),
            complete integer not null default 1,
            unique(batch_id, clip_id)
        );

        create table if not exists evaluation_label_reviews (
            id integer primary key autoincrement,
            prediction_snapshot_id integer not null references prediction_snapshots(id) on delete cascade,
            clip_id integer not null references clips(id) on delete cascade,
            label_key text not null,
            expected_value text not null check (expected_value in ('yes', 'no', 'uncertain')),
            notes text,
            created_at text not null
        );

        create table if not exists evaluation_highlight_reviews (
            id integer primary key autoincrement,
            prediction_snapshot_id integer not null references prediction_snapshots(id) on delete cascade,
            clip_id integer not null references clips(id) on delete cascade,
            aspect text not null,
            expected_value text not null check (expected_value in ('yes', 'no', 'uncertain')),
            notes text,
            created_at text not null
        );

        create table if not exists evaluation_promotions (
            id integer primary key autoincrement,
            batch_id integer not null references evaluation_batches(id),
            promoted_at text not null,
            item_count integer not null,
            status text not null,
            details_json text not null
        );

        create table if not exists live_teacher_assignments (
            clip_id integer primary key references clips(id) on delete cascade,
            label_row_id integer,
            event_row_id integer,
            prediction_snapshot_id integer,
            pipeline_manifest_id integer,
            promoted_from_batch_id integer,
            assigned_at text not null
        );

        create table if not exists active_teacher_pipeline (
            id integer primary key check (id = 1),
            pipeline_manifest_id integer not null references teacher_pipeline_manifests(id),
            batch_id integer references evaluation_batches(id),
            activated_at text not null
        );

        create index if not exists idx_prediction_snapshots_batch
            on prediction_snapshots(batch_id, clip_id);

        create index if not exists idx_evaluation_label_reviews_snapshot
            on evaluation_label_reviews(prediction_snapshot_id, label_key, id);

        create index if not exists idx_evaluation_highlight_reviews_snapshot
            on evaluation_highlight_reviews(prediction_snapshot_id, aspect, id);
        """
    )
