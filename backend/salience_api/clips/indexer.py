from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
import json
import sqlite3

from salience_api.clips.media_probe import safe_probe_media
from salience_api.features.basic import (
    BasicFeatures,
    duration_quality,
    estimate_metadata_features,
)
from salience_api.features.bot_detection import opponent_likely_bot_from_evidence
from salience_api.features.quality_modifiers import (
    stationary_sniper_target_from_evidence,
    stationary_target_from_evidence,
)
from salience_api.features.teacher_labels import (
    TEACHER_LABEL_KEYS,
    TeacherClipLabels,
    derive_label_confidences,
    normalize_teacher_payload,
)
from salience_api.ranking.highlights import (
    HighlightProfile,
    build_highlight_profile,
    event_audit_summary,
)
from salience_api.ranking.personal_ranker import train_personal_ranker
from salience_api.ranking.scoring import score_clip

TEACHER_SCHEMA_VERSION = "salience-labels"
TEACHER_FALLBACK_SCHEMA_VERSION = "salience-teacher-v15"
EVENT_VALIDATION_SCHEMA = "event-summary-v2-general"
DECISION_ACTIONS = (
    "favorite",
    "keep",
    "skip",
    "boring",
    "delete",
    "supervised_keep",
    "supervised_skip",
)
RETIRED_TEACHER_TAGS = {"weapon_swap_finish", "sniper_setup_damage"}
TASTE_RELATED_TAGS = {
    "sniper_kill": {"sniper", "sniper_kill", "no_scope"},
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _feature_row(features: BasicFeatures) -> list[float]:
    tags = set(features.tags)
    positive_teacher_labels = [
        1.0 if label in tags else 0.0 for label in TEACHER_LABEL_KEYS
    ]
    return [
        features.motion_score,
        features.audio_peak_score,
        features.silence_ratio,
        duration_quality(features.duration_sec),
        features.action_density,
        *positive_teacher_labels,
    ]


def _taste_preferences(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        "select key, weight from taste_preferences where weight != 0"
    ).fetchall()
    return {str(row["key"]): float(row["weight"]) for row in rows}


def _feedback_training_weight(row: sqlite3.Row, latest_feedback_id: int) -> float:
    weight = float(row["weight"])
    action = str(row["action"])
    label = str(row["label"] or "")
    event_id = int(row["id"])
    distance_from_latest = max(0, latest_feedback_id - event_id)
    recency_multiplier = 0.35 + (0.65 / (1.0 + (distance_from_latest / 160.0)))
    source_multiplier = 1.0
    if action.startswith("supervised_"):
        source_multiplier = 0.45
    elif label == "liked_import":
        source_multiplier = 0.85
    return weight * source_multiplier * recency_multiplier


def _apply_taste_preferences(
    personal_score: float,
    features: BasicFeatures,
    preferences: dict[str, float],
    *,
    feedback_available: bool,
) -> float | None:
    if personal_score < 1e-12:
        personal_score = 0.0
    if not preferences:
        return personal_score if feedback_available else None
    tags = set(features.tags)
    matched_weights = [preferences[tag] for tag in tags if tag in preferences]
    for preference_key, related_tags in TASTE_RELATED_TAGS.items():
        if (
            preference_key in preferences
            and preference_key not in tags
            and tags & related_tags
        ):
            matched_weights.append(preferences[preference_key])
    if not matched_weights:
        return personal_score if feedback_available else None
    adjustment = max(-0.85, min(0.85, sum(matched_weights) * 0.7))
    return max(0.0, min(1.0, personal_score + adjustment))


def _row_value(row: sqlite3.Row, key: str) -> object | None:
    return row[key] if key in row.keys() else None


def _teacher_tags_from_payload(label_json: str | None) -> list[str]:
    if not label_json:
        return []
    payload = json.loads(label_json)
    labels = normalize_teacher_payload(payload)
    tags = labels.yes_labels()
    if "opponent_likely_bot" not in tags and opponent_likely_bot_from_evidence(
        labels.evidence
    ):
        tags.append("opponent_likely_bot")
    if "stationary_target" not in tags and stationary_target_from_evidence(
        labels.evidence
    ):
        tags.append("stationary_target")
    if (
        "stationary_sniper_target" not in tags
        and stationary_sniper_target_from_evidence(labels.evidence, set(tags))
    ):
        tags.append("stationary_sniper_target")
    return tags


def _features_from_row(row: sqlite3.Row) -> BasicFeatures:
    feature_json = json.loads(row["feature_json"] or "{}")
    tags = (
        set(feature_json.get("tags", []))
        - set(TEACHER_LABEL_KEYS)
        - RETIRED_TEACHER_TAGS
    )
    tags.update(_teacher_tags_from_payload(_row_value(row, "teacher_label_json")))
    return BasicFeatures(
        duration_sec=float(row["duration_sec"] or 30.0),
        motion_score=float(row["motion_score"]),
        audio_peak_score=float(row["audio_peak_score"]),
        silence_ratio=float(row["silence_ratio"]),
        extraction_confidence=float(row["extraction_confidence"]),
        action_density=float(row["action_density"]),
        tags=sorted(tags),
    )


def _upsert_clip_features(
    conn: sqlite3.Connection,
    *,
    clip_id: int,
    features: BasicFeatures,
    feature_json: dict,
) -> None:
    conn.execute(
        """
        insert into clip_features(
            clip_id,
            motion_score,
            audio_peak_score,
            silence_ratio,
            action_density,
            duration_score,
            extraction_confidence,
            feature_json
        )
        values (?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(clip_id) do update set
            motion_score = excluded.motion_score,
            audio_peak_score = excluded.audio_peak_score,
            silence_ratio = excluded.silence_ratio,
            action_density = excluded.action_density,
            duration_score = excluded.duration_score,
            extraction_confidence = excluded.extraction_confidence,
            feature_json = excluded.feature_json
        """,
        (
            clip_id,
            features.motion_score,
            features.audio_peak_score,
            features.silence_ratio,
            features.action_density,
            duration_quality(features.duration_sec),
            features.extraction_confidence,
            json.dumps(feature_json),
        ),
    )


def _upsert_score(
    conn: sqlite3.Connection,
    clip_id: int,
    features: BasicFeatures,
    personal_score: float | None,
    *,
    highlight_profile: HighlightProfile | None = None,
) -> None:
    score = score_clip(
        features,
        personal_score=personal_score,
        highlight_profile=highlight_profile,
    )
    conn.execute(
        """
        insert into clip_scores(clip_id, base_score, personal_score, final_score, confidence, explanation, scored_at)
        values (?, ?, ?, ?, ?, ?, ?)
        on conflict(clip_id) do update set
            base_score = excluded.base_score,
            personal_score = excluded.personal_score,
            final_score = excluded.final_score,
            confidence = excluded.confidence,
            explanation = excluded.explanation,
            scored_at = excluded.scored_at
        """,
        (
            clip_id,
            score.base_score,
            score.personal_score,
            score.final_score,
            score.confidence,
            score.explanation,
            _now(),
        ),
    )


def _latest_teacher_labels_by_clip(conn: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    rows = conn.execute(
        """
        select tl.clip_id, tl.provider, tl.label_json, tl.confidence
        from teacher_labels tl
        join clips c on c.id = tl.clip_id
        left join live_teacher_assignments live on live.clip_id = c.id
        where tl.id = coalesce(
            live.label_row_id,
            (
                select fallback.id from teacher_labels fallback
                where fallback.clip_id = c.id
                  and (fallback.schema_version like ? or fallback.schema_version like ?)
                order by case when fallback.schema_version like ? then 1 else 0 end desc,
                         fallback.id desc
                limit 1
            )
        )
        """,
        (
            f"{TEACHER_SCHEMA_VERSION}:%",
            f"{TEACHER_FALLBACK_SCHEMA_VERSION}:%",
            f"{TEACHER_SCHEMA_VERSION}:%",
        ),
    ).fetchall()
    return {int(row["clip_id"]): row for row in rows}


def _latest_teacher_events_by_clip(conn: sqlite3.Connection) -> dict[int, dict]:
    rows = conn.execute(
        """
        select te.clip_id, te.event_json
        from teacher_events te
        join clips c on c.id = te.clip_id
        left join live_teacher_assignments live on live.clip_id = c.id
        where te.id = coalesce(
            live.event_row_id,
            (
                select fallback.id from teacher_events fallback
                where fallback.clip_id = c.id and fallback.schema_version = ?
                order by fallback.id desc limit 1
            )
        )
        """,
        (TEACHER_SCHEMA_VERSION,),
    ).fetchall()
    events: dict[int, dict] = {}
    for row in rows:
        try:
            events[int(row["clip_id"])] = json.loads(row["event_json"] or "{}")
        except json.JSONDecodeError:
            events[int(row["clip_id"])] = {}
    return events


def record_teacher_labels(
    conn: sqlite3.Connection,
    *,
    clip_id: int,
    provider: str,
    labels: TeacherClipLabels,
    model: str,
    label_confidences: dict[str, float] | None = None,
    update_live_assignment: bool = False,
) -> None:
    payload = labels.model_dump()
    confidences = label_confidences or derive_label_confidences(payload)
    payload["label_confidences"] = {
        str(key): max(0.0, min(1.0, float(value)))
        for key, value in confidences.items()
    }
    cursor = conn.execute(
        """
        insert into teacher_labels(clip_id, provider, schema_version, label_json, confidence, created_at)
        values (?, ?, ?, ?, ?, ?)
        """,
        (
            clip_id,
            provider,
            f"{TEACHER_SCHEMA_VERSION}:{model}",
            json.dumps(payload),
            labels.confidence,
            _now(),
        ),
    )
    if update_live_assignment:
        conn.execute(
            "update live_teacher_assignments set label_row_id = ? where clip_id = ?",
            (cursor.lastrowid, clip_id),
        )
    row = conn.execute(
        "select feature_json from clip_features where clip_id = ?", (clip_id,)
    ).fetchone()
    feature_json = json.loads(row["feature_json"] or "{}") if row else {}
    existing_tags = (
        set(feature_json.get("tags", []))
        - set(TEACHER_LABEL_KEYS)
        - RETIRED_TEACHER_TAGS
    )
    feature_json["tags"] = sorted(existing_tags | set(labels.yes_labels()))
    feature_json["teacher_provider"] = provider
    feature_json["teacher_confidence"] = labels.confidence
    feature_json["teacher_evidence"] = labels.evidence
    conn.execute(
        """
        update clip_features
        set feature_json = ?
        where clip_id = ?
        """,
        (json.dumps(feature_json), clip_id),
    )


def record_teacher_events(
    conn: sqlite3.Connection,
    *,
    clip_id: int,
    provider: str,
    model: str,
    status: str,
    event_data: dict,
    update_live_assignment: bool = False,
) -> None:
    cursor = conn.execute(
        """
        insert into teacher_events(
            clip_id,
            provider,
            model,
            schema_version,
            status,
            event_json,
            created_at
        )
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clip_id,
            provider,
            model,
            TEACHER_SCHEMA_VERSION,
            status,
            json.dumps(event_data),
            _now(),
        ),
    )
    if update_live_assignment:
        conn.execute(
            "update live_teacher_assignments set event_row_id = ? where clip_id = ?",
            (cursor.lastrowid, clip_id),
        )


def record_teacher_candidate(
    conn: sqlite3.Connection,
    *,
    clip_id: int,
    provider: str,
    model: str,
    status: str,
    labels: TeacherClipLabels,
    event_data: dict,
) -> None:
    conn.execute(
        """
        insert into teacher_event_candidates(
            clip_id, provider, model, candidate_version, status,
            label_json, event_json, created_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clip_id,
            provider,
            model,
            EVENT_VALIDATION_SCHEMA,
            status,
            json.dumps(labels.model_dump()),
            json.dumps(event_data),
            _now(),
        ),
    )


def index_clip_path(conn: sqlite3.Connection, path: Path) -> int:
    metadata = safe_probe_media(path)
    stat = path.stat()
    indexed_at = _now()
    conn.execute(
        """
        insert into clips(path, filename, duration_sec, width, height, fps, size_bytes, created_at, indexed_at, source)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, 'local')
        on conflict(path) do update set
            filename = excluded.filename,
            duration_sec = coalesce(excluded.duration_sec, clips.duration_sec),
            width = coalesce(excluded.width, clips.width),
            height = coalesce(excluded.height, clips.height),
            fps = coalesce(excluded.fps, clips.fps),
            size_bytes = excluded.size_bytes,
            indexed_at = excluded.indexed_at
        """,
        (
            str(path),
            path.name,
            metadata.duration_sec,
            metadata.width,
            metadata.height,
            metadata.fps,
            metadata.size_bytes or stat.st_size,
            datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
            indexed_at,
        ),
    )
    row = conn.execute("select id from clips where path = ?", (str(path),)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to index clip: {path}")
    clip_id = int(row["id"])
    features = estimate_metadata_features(
        duration_sec=metadata.duration_sec,
        size_bytes=metadata.size_bytes or stat.st_size,
        fps=metadata.fps,
    )
    _upsert_clip_features(
        conn,
        clip_id=clip_id,
        features=features,
        feature_json={
            "tags": [],
            "thumbnail_variant": "ridge",
            "source": "metadata_estimate",
        },
    )
    _upsert_score(conn, clip_id, features, personal_score=None)
    return clip_id


def index_known_clip_path(
    conn: sqlite3.Connection, path: Path, *, source: str = "local"
) -> int:
    if path.exists():
        return index_clip_path(conn, path)

    filename = PureWindowsPath(str(path)).name
    indexed_at = _now()
    conn.execute(
        """
        insert into clips(path, filename, created_at, indexed_at, source)
        values (?, ?, ?, ?, ?)
        on conflict(path) do update set
            filename = excluded.filename,
            indexed_at = excluded.indexed_at,
            source = excluded.source
        """,
        (
            str(path),
            filename,
            indexed_at,
            indexed_at,
            source,
        ),
    )
    row = conn.execute("select id from clips where path = ?", (str(path),)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to index labeled clip: {path}")
    clip_id = int(row["id"])
    existing_features = conn.execute(
        "select 1 from clip_features where clip_id = ?", (clip_id,)
    ).fetchone()
    if existing_features is not None:
        return clip_id
    features = estimate_metadata_features(duration_sec=None, size_bytes=None, fps=None)
    _upsert_clip_features(
        conn,
        clip_id=clip_id,
        features=features,
        feature_json={
            "tags": [],
            "thumbnail_variant": "ridge",
            "source": "supervised_label_placeholder",
        },
    )
    _upsert_score(conn, clip_id, features, personal_score=None)
    return clip_id


def index_demo_clip(conn: sqlite3.Connection, seed: dict) -> int:
    filename = str(seed["filename"])
    source = str(seed.get("source", "demo"))
    seed_path = str(seed.get("path", f"demo://{filename}"))
    features = BasicFeatures(
        duration_sec=float(seed["duration_sec"]),
        motion_score=float(seed["motion_score"]),
        audio_peak_score=float(seed["audio_peak_score"]),
        silence_ratio=float(seed["silence_ratio"]),
        extraction_confidence=float(seed.get("extraction_confidence", 0.86)),
        action_density=float(seed.get("action_density", seed["motion_score"])),
        tags=list(seed.get("tags", [])),
    )
    conn.execute(
        """
        insert into clips(path, filename, duration_sec, width, height, fps, size_bytes, created_at, indexed_at, source)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(path) do update set
            filename = excluded.filename,
            duration_sec = excluded.duration_sec,
            width = excluded.width,
            height = excluded.height,
            fps = excluded.fps,
            size_bytes = excluded.size_bytes,
            indexed_at = excluded.indexed_at,
            source = excluded.source
        """,
        (
            seed_path,
            filename,
            features.duration_sec,
            seed.get("width"),
            seed.get("height"),
            seed.get("fps"),
            seed.get("size_bytes"),
            _now(),
            _now(),
            source,
        ),
    )
    row = conn.execute(
        "select id from clips where path = ?", (seed_path,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to seed demo clip: {filename}")
    clip_id = int(row["id"])
    _upsert_clip_features(
        conn,
        clip_id=clip_id,
        features=features,
        feature_json={
            "tags": list(seed.get("tags", [])),
            "seed_explanation": seed.get("explanation"),
            "thumbnail_variant": seed.get("thumbnail_variant", "ridge"),
            "source": seed.get("feature_source", "demo_seed"),
        },
    )
    teacher_seed = seed.get("teacher_labels")
    if teacher_seed:
        record_teacher_labels(
            conn,
            clip_id=clip_id,
            provider=str(seed.get("teacher_provider", "seeded-fireworks-teacher")),
            labels=normalize_teacher_payload(teacher_seed),
            model=str(seed.get("teacher_model", "demo-precomputed")),
        )
    event_seed = seed.get("teacher_events")
    if isinstance(event_seed, dict):
        record_teacher_events(
            conn,
            clip_id=clip_id,
            provider=str(seed.get("teacher_provider", "seeded-fireworks-teacher")),
            model=str(seed.get("teacher_model", "demo-precomputed")),
            status=str(seed.get("teacher_event_status", "precomputed")),
            event_data=event_seed,
        )
    _upsert_score(conn, clip_id, features, personal_score=None)
    if seed.get("final_score") is not None:
        conn.execute(
            """
            update clip_scores
            set base_score = ?, final_score = ?, confidence = ?,
                explanation = coalesce(?, explanation)
            where clip_id = ?
            """,
            (
                float(seed.get("base_score", seed["final_score"])),
                float(seed["final_score"]),
                float(seed.get("confidence", features.extraction_confidence)),
                seed.get("score_explanation") or seed.get("explanation"),
                clip_id,
            ),
        )
    return clip_id


def seed_demo_clips(conn: sqlite3.Connection, seeds: list[dict]) -> int:
    return sum(1 for seed in seeds if index_demo_clip(conn, seed))


def _effective_feedback_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        with ranked_feedback as (
            select
                fe.*,
                row_number() over (
                    partition by fe.clip_id
                    order by
                        case
                            when fe.action in ('favorite', 'keep', 'skip', 'boring', 'delete')
                              and coalesce(fe.label, '') != 'liked_import'
                            then 2
                            else 1
                        end desc,
                        fe.id desc
                ) as decision_rank
            from feedback_events fe
            where fe.action in (
                'favorite', 'keep', 'skip', 'boring', 'delete',
                'supervised_keep', 'supervised_skip'
            )
        )
        select id, clip_id, action, label, weight, created_at
        from ranked_feedback
        where decision_rank = 1
        order by id
        """
    ).fetchall()


def _training_rows(conn: sqlite3.Connection) -> tuple[list[list[float]], list[float]]:
    latest_feedback_id = int(
        conn.execute("select coalesce(max(id), 0) from feedback_events").fetchone()[0]
        or 0
    )
    rows = conn.execute(
        """
        with ranked_feedback as (
            select
                fe.*,
                row_number() over (
                    partition by fe.clip_id
                    order by
                        case
                            when fe.action in ('favorite', 'keep', 'skip', 'boring', 'delete')
                              and coalesce(fe.label, '') != 'liked_import'
                            then 2
                            else 1
                        end desc,
                        fe.id desc
                ) as decision_rank
            from feedback_events fe
            where fe.action in (
                'favorite', 'keep', 'skip', 'boring', 'delete',
                'supervised_keep', 'supervised_skip'
            )
        )
        select
            fe.id,
            fe.action,
            fe.label,
            fe.created_at,
            cf.motion_score,
            cf.audio_peak_score,
            cf.silence_ratio,
            cf.action_density,
            cf.extraction_confidence,
            cf.feature_json,
            c.duration_sec,
            tl.label_json as teacher_label_json,
            fe.weight
        from ranked_feedback fe
        join clip_features cf on cf.clip_id = fe.clip_id
        join clips c on c.id = fe.clip_id
        left join teacher_labels tl on tl.id = coalesce(
            (select label_row_id from live_teacher_assignments where clip_id = fe.clip_id),
            (
                select id from teacher_labels
                where clip_id = fe.clip_id
                  and (schema_version like ? or schema_version like ?)
                order by case when schema_version like ? then 1 else 0 end desc, id desc
                limit 1
            )
        )
        where fe.decision_rank = 1
        """,
        (
            f"{TEACHER_SCHEMA_VERSION}:%",
            f"{TEACHER_FALLBACK_SCHEMA_VERSION}:%",
            f"{TEACHER_SCHEMA_VERSION}:%",
        ),
    ).fetchall()
    features: list[list[float]] = []
    weights: list[float] = []
    for row in rows:
        feature = _features_from_row(row)
        features.append(_feature_row(feature))
        weights.append(_feedback_training_weight(row, latest_feedback_id))
    return features, weights


def _highlight_profile_from_event_data(
    event_data: dict | None,
) -> HighlightProfile | None:
    if not isinstance(event_data, dict):
        return None
    events = event_data.get("events")
    if not isinstance(events, list) or not events:
        return None
    return build_highlight_profile(events)


def rescore_all_clips(conn: sqlite3.Connection) -> None:
    training_features, training_weights = _training_rows(conn)
    taste_preferences = _taste_preferences(conn)
    events_by_clip = _latest_teacher_events_by_clip(conn)
    ranker = train_personal_ranker(training_features, training_weights)
    rows = conn.execute(
        """
        select
            c.id as clip_id,
            c.path,
            c.duration_sec,
            cf.motion_score,
            cf.audio_peak_score,
            cf.silence_ratio,
            cf.action_density,
            cf.extraction_confidence,
            cf.feature_json,
            tl.label_json as teacher_label_json
        from clips c
        join clip_features cf on cf.clip_id = c.id
        left join teacher_labels tl on tl.id = coalesce(
            (select label_row_id from live_teacher_assignments where clip_id = c.id),
            (
                select id from teacher_labels
                where clip_id = c.id
                  and (schema_version like ? or schema_version like ?)
                order by case when schema_version like ? then 1 else 0 end desc, id desc
                limit 1
            )
        )
        """,
        (
            f"{TEACHER_SCHEMA_VERSION}:%",
            f"{TEACHER_FALLBACK_SCHEMA_VERSION}:%",
            f"{TEACHER_SCHEMA_VERSION}:%",
        ),
    ).fetchall()
    features_by_id = [
        (int(row["clip_id"]), _features_from_row(row))
        for row in rows
        if not str(row["path"]).startswith("snapshot://")
        and json.loads(row["feature_json"] or "{}").get("source")
        != "precomputed_local_student"
    ]
    predictions = ranker.predict(
        [_feature_row(features) for _, features in features_by_id]
    )
    for (clip_id, features), personal_score in zip(
        features_by_id, predictions, strict=True
    ):
        adjusted_score = _apply_taste_preferences(
            personal_score,
            features,
            taste_preferences,
            feedback_available=bool(training_weights),
        )
        profile = _highlight_profile_from_event_data(events_by_clip.get(clip_id))
        _upsert_score(
            conn,
            clip_id,
            features,
            adjusted_score,
            highlight_profile=profile,
        )

    positives = sum(1 for weight in training_weights if weight > 0)
    negatives = sum(1 for weight in training_weights if weight < 0)
    conn.execute(
        """
        insert into user_ranker_metadata(id, feedback_count, positive_count, negative_count, updated_at)
        values (1, ?, ?, ?, ?)
        on conflict(id) do update set
            feedback_count = excluded.feedback_count,
            positive_count = excluded.positive_count,
            negative_count = excluded.negative_count,
            updated_at = excluded.updated_at
        """,
        (len(training_weights), positives, negatives, _now()),
    )


def training_status(conn: sqlite3.Connection) -> dict:
    totals = conn.execute(
        """
        select
            count(*) as clips,
            sum(case when tl.clip_id is not null then 1 else 0 end) as teacher_labeled,
            sum(case when tl.clip_id is null then 1 else 0 end) as teacher_pending
        from clips c
        left join (
            select clip_id, max(id) as id
            from teacher_labels
            where schema_version like ?
            group by clip_id
        ) tl on tl.clip_id = c.id
        """,
        (f"{TEACHER_SCHEMA_VERSION}:%",),
    ).fetchone()
    effective_feedback = _effective_feedback_rows(conn)
    scores_by_clip = {
        int(row["clip_id"]): float(row["personal_score"])
        for row in conn.execute(
            "select clip_id, personal_score from clip_scores"
        ).fetchall()
    }
    positive_scores = [
        scores_by_clip[int(row["clip_id"])]
        for row in effective_feedback
        if float(row["weight"]) > 0 and int(row["clip_id"]) in scores_by_clip
    ]
    negative_scores = [
        scores_by_clip[int(row["clip_id"])]
        for row in effective_feedback
        if float(row["weight"]) < 0 and int(row["clip_id"]) in scores_by_clip
    ]

    def average(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    positive_avg = average(positive_scores)
    negative_avg = average(negative_scores)
    separation = None
    if positive_avg is not None and negative_avg is not None:
        separation = positive_avg - negative_avg

    clips = int(totals["clips"] or 0)
    teacher_labeled = int(totals["teacher_labeled"] or 0)
    return {
        "clips": clips,
        "teacher_labeled": teacher_labeled,
        "teacher_pending": int(totals["teacher_pending"] or 0),
        "teacher_progress": teacher_labeled / clips if clips else 0.0,
        "feedback_count": len(effective_feedback),
        "positive_count": sum(
            1 for row in effective_feedback if float(row["weight"]) > 0
        ),
        "negative_count": sum(
            1 for row in effective_feedback if float(row["weight"]) < 0
        ),
        "positive_avg_personal_score": positive_avg,
        "negative_avg_personal_score": negative_avg,
        "personal_score_separation": separation,
    }


def list_ranked_clips(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        select
            c.id,
            c.path,
            c.filename,
            c.duration_sec,
            c.width,
            c.height,
            c.fps,
            c.size_bytes,
            c.source,
            cs.final_score,
            cs.base_score,
            cs.personal_score,
            cs.confidence,
            cs.explanation,
            cf.feature_json
        from clips c
        left join clip_scores cs on cs.clip_id = c.id
        left join clip_features cf on cf.clip_id = c.id
        order by cs.final_score desc nulls last, c.indexed_at desc
        """
    ).fetchall()
    feedback_rows = conn.execute(
        """
        select clip_id, action, label
        from feedback_events
        where action = 'tag'
        order by created_at desc
        """
    ).fetchall()
    feedback_by_clip: dict[int, list[str]] = {}
    for row in feedback_rows:
        if row["label"]:
            feedback_by_clip.setdefault(int(row["clip_id"]), []).append(
                f"tag:{row['label']}"
            )
    for row in _effective_feedback_rows(conn):
        feedback_by_clip.setdefault(int(row["clip_id"]), []).append(str(row["action"]))
    teacher_by_clip = _latest_teacher_labels_by_clip(conn)
    events_by_clip = _latest_teacher_events_by_clip(conn)

    clips: list[dict] = []
    for row in rows:
        feature_json = json.loads(row["feature_json"] or "{}")
        teacher = teacher_by_clip.get(int(row["id"]))
        teacher_payload = json.loads(teacher["label_json"]) if teacher else {}
        normalized_teacher = (
            normalize_teacher_payload(teacher_payload) if teacher else None
        )
        teacher_labels = (
            {key: getattr(normalized_teacher, key) for key in TEACHER_LABEL_KEYS}
            if normalized_teacher
            else {}
        )
        if normalized_teacher and opponent_likely_bot_from_evidence(
            normalized_teacher.evidence
        ):
            teacher_labels["opponent_likely_bot"] = "yes"
        if normalized_teacher and stationary_target_from_evidence(
            normalized_teacher.evidence
        ):
            teacher_labels["stationary_target"] = "yes"
        if normalized_teacher and stationary_sniper_target_from_evidence(
            normalized_teacher.evidence,
            set(_teacher_tags_from_payload(teacher["label_json"]))
            if teacher
            else set(),
        ):
            teacher_labels["stationary_sniper_target"] = "yes"
        tags = (
            set(feature_json.get("tags", []))
            - set(TEACHER_LABEL_KEYS)
            - RETIRED_TEACHER_TAGS
        )
        if teacher:
            tags.update(_teacher_tags_from_payload(teacher["label_json"]))
        live_audit = event_audit_summary(events_by_clip.get(int(row["id"])))
        clips.append(
            {
                "id": int(row["id"]),
                "path": row["path"],
                "filename": row["filename"],
                "duration_sec": row["duration_sec"],
                "width": row["width"],
                "height": row["height"],
                "fps": row["fps"],
                "size_bytes": row["size_bytes"],
                "source": row["source"],
                "video_url": None
                if str(row["path"]).startswith(("demo://", "snapshot://"))
                else f"/clips/{int(row['id'])}/video",
                "final_score": row["final_score"],
                "base_score": row["base_score"],
                "personal_score": row["personal_score"],
                "confidence": row["confidence"],
                "explanation": row["explanation"]
                or feature_json.get("seed_explanation"),
                "tags": sorted(tags),
                "feedback": feedback_by_clip.get(int(row["id"]), []),
                "thumbnail_variant": feature_json.get("thumbnail_variant", "ridge"),
                "teacher_provider": teacher["provider"] if teacher else None,
                "teacher_confidence": teacher["confidence"] if teacher else None,
                "teacher_labels": teacher_labels,
                "teacher_evidence": list(teacher_payload.get("evidence", [])),
                "highlight_description": live_audit.get("highlight_description"),
            }
        )
    return clips


def clip_exists(conn: sqlite3.Connection, clip_id: int) -> bool:
    row = conn.execute("select 1 from clips where id = ?", (clip_id,)).fetchone()
    return row is not None


def clip_teacher_input_row(
    conn: sqlite3.Connection, clip_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        """
        select c.id, c.path, c.filename, c.duration_sec, c.width, c.height, c.fps, cf.feature_json
        from clips c
        join clip_features cf on cf.clip_id = c.id
        where c.id = ?
        """,
        (clip_id,),
    ).fetchone()


def clip_ids_for_enrichment(
    conn: sqlite3.Connection, *, clip_id: int | None, limit: int
) -> list[int]:
    if clip_id is not None:
        return [clip_id] if clip_exists(conn, clip_id) else []
    rows = conn.execute(
        """
        select c.id
        from clips c
        left join teacher_labels tl on tl.clip_id = c.id and tl.schema_version like ?
        where tl.id is null
        order by c.id desc
        limit ?
        """,
        (f"{TEACHER_SCHEMA_VERSION}:%", limit),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def clip_ids_for_reenrichment(
    conn: sqlite3.Connection, *, limit: int
) -> list[int]:
    """Return assigned clips whose teacher output explicitly needs another pass."""
    rows = conn.execute(
        """
        select c.id, tl.label_json, te.status, te.event_json
        from clips c
        join live_teacher_assignments lta on lta.clip_id = c.id
        left join teacher_labels tl on tl.id = lta.label_row_id
        left join teacher_events te on te.id = lta.event_row_id
        order by c.id desc
        """
    ).fetchall()
    selected: list[int] = []
    for row in rows:
        try:
            labels = json.loads(row["label_json"] or "{}")
        except json.JSONDecodeError:
            labels = {}
        label_unresolved = any(
            str(labels.get(key, "")).lower() == "uncertain"
            for key in TEACHER_LABEL_KEYS
        )
        if label_unresolved:
            selected.append(int(row["id"]))
            if len(selected) >= limit:
                break
    return selected
