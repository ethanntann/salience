from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import sqlite3
from threading import Lock, Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from salience_api.clips.indexer import (
    RETIRED_TEACHER_TAGS,
    EVENT_VALIDATION_SCHEMA,
    TEACHER_FALLBACK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    clip_exists,
    clip_ids_for_enrichment,
    clip_ids_for_reenrichment,
    clip_teacher_input_row,
    list_ranked_clips,
    record_teacher_events,
    record_teacher_candidate,
    record_teacher_labels,
    rescore_all_clips,
    seed_demo_clips,
    training_status,
    index_clip_path,
)
from salience_api.clips.keyframes import (
    cleanup_keyframes,
    extract_event_keyframes,
    extract_timeline_keyframes,
)
from salience_api.student.backbone import ARTIFACT_VERSION, SUPPORTED_ARTIFACT_VERSIONS
from salience_api.clips.scanner import find_clip_files
from salience_api.config import Settings, get_settings
from salience_api.db import connect, init_db
from salience_api.ranking.highlights import event_audit_summary
from salience_api.demo_seed import load_demo_clips
from salience_api.feedback.service import record_feedback
from salience_api.features.fireworks_teacher import (
    AmdDeveloperCloudTeacherClient,
    ClipTeacherInput,
    FireworksTeacherClient,
    OpenAICompatibleTeacherClient,
    WeaponAttribution,
    merge_event_labels,
)
from salience_api.ranking.highlights import attach_event_audit
from salience_api.features.hud_ocr import RapidHudOcr
from salience_api.features.teacher_labels import (
    TEACHER_LABEL_KEYS,
    derive_label_confidences,
    normalize_teacher_payload,
)
from salience_api.evaluation import (
    HIGHLIGHT_REVIEW_ASPECTS,
    batch_metrics,
    create_batch_from_candidates,
    get_batch,
    latest_snapshot_highlight_reviews,
    latest_snapshot_label_reviews,
    list_batch_clips,
    list_batches,
    promote_batch,
    record_highlight_review,
    record_snapshot_label_review,
)
from salience_api.hardware import detect_amd_gpu
from salience_api.schemas import (
    AiStatus,
    ClipRecord,
    CreateEvalBatchRequest,
    EnrichRequest,
    EnrichResponse,
    EvalBatchListResponse,
    EvalBatchMetricsResponse,
    EvalClipResponse,
    EvalSummaryResponse,
    ExportClipsRequest,
    ExportClipsResponse,
    FeedbackRequest,
    HighlightReviewRequest,
    ImportLikedFolderRequest,
    ImportLikedFolderResponse,
    ImportLabelsRequest,
    ImportLabelsResponse,
    PromoteBatchRequest,
    PromoteBatchResponse,
    ScanFolderRequest,
    ScanFolderResponse,
    SnapshotLabelReviewRequest,
    StudentReportsResponse,
    TeacherRunRequest,
    TeacherRunStatus,
    EventValidationRequest,
    TeacherLabelReviewRequest,
    TasteProfileRequest,
    TasteProfileResponse,
    TrainingStatus,
)
from salience_api.student.local_teacher import LocalTeacherClient
from salience_api.student.reports import load_student_reports
from salience_api.taste import (
    import_liked_folder,
    reset_taste_preferences,
    save_taste_preferences,
)
from salience_api.training.labels import import_labeled_jsonl


def _ensure_database(conn: sqlite3.Connection, settings: Settings) -> None:
    init_db(conn)
    if settings.demo_mode:
        count = conn.execute(
            "select count(*) from clips where path like 'demo://%' or path like 'snapshot://%'"
        ).fetchone()[0]
        if count == 0 and settings.demo_data_path.exists():
            seed_demo_clips(conn, load_demo_clips(settings.demo_data_path))
            rescore_all_clips(conn)
    conn.commit()


def _local_teacher_artifacts_ready(artifacts_dir: Path) -> bool:
    root = Path(artifacts_dir)
    meta_path = root / "artifact_meta.json"
    thresholds_path = root / "thresholds.json"
    if not (
        (root / "locator.onnx").is_file()
        and (root / "event_heads.onnx").is_file()
        and meta_path.is_file()
        and thresholds_path.is_file()
    ):
        return False
    try:
        return (
            json.loads(meta_path.read_text(encoding="utf-8")).get("version")
            in SUPPORTED_ARTIFACT_VERSIONS
        )
    except (OSError, ValueError):
        return False


def teacher_provider_configured(settings: Settings) -> bool:
    provider = settings.vlm_provider.lower()
    if provider == "local":
        return _local_teacher_artifacts_ready(settings.student_artifacts_dir)
    if provider == "amd" or provider == "amd-developer-cloud":
        return bool(
            settings.amd_developer_cloud_api_key and settings.amd_developer_cloud_base_url
        )
    return bool(settings.fireworks_api_key)


def build_teacher_client(settings: Settings) -> OpenAICompatibleTeacherClient:
    provider = settings.vlm_provider.lower()
    if provider == "local":
        return LocalTeacherClient.from_artifacts(
            settings.student_artifacts_dir,
            accelerator=settings.accelerator,
        )
    if provider == "amd" or provider == "amd-developer-cloud":
        return AmdDeveloperCloudTeacherClient(settings)
    return FireworksTeacherClient(settings)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(title="Salience API", version="0.1.0")
    teacher_run_lock = Lock()
    local_ocr = RapidHudOcr(enabled=resolved_settings.local_ocr_enabled)
    teacher_client_cache: OpenAICompatibleTeacherClient | None = None
    teacher_run_state = {
        "running": False,
        "requested": 0,
        "enriched": 0,
        "failed": 0,
        "last_error": None,
        "started_at": None,
        "finished_at": None,
    }
    event_validation_run_state = {
        "running": False,
        "requested": 0,
        "enriched": 0,
        "failed": 0,
        "last_error": None,
        "started_at": None,
        "finished_at": None,
    }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @contextmanager
    def db() -> Iterator[sqlite3.Connection]:
        with connect(resolved_settings.resolved_database_path()) as conn:
            _ensure_database(conn, resolved_settings)
            yield conn

    def teacher_configured() -> bool:
        return teacher_provider_configured(resolved_settings)

    def teacher_client() -> OpenAICompatibleTeacherClient:
        nonlocal teacher_client_cache
        if teacher_client_cache is None:
            # Loading ONNX sessions is expensive. Keep one provider/model client
            # for the app lifetime instead of rebuilding it for every clip.
            teacher_client_cache = build_teacher_client(resolved_settings)
        return teacher_client_cache

    def teacher_run_snapshot() -> dict:
        with teacher_run_lock:
            return dict(teacher_run_state)

    def update_teacher_run(**updates: object) -> None:
        with teacher_run_lock:
            teacher_run_state.update(updates)

    def event_validation_run_snapshot() -> dict:
        with teacher_run_lock:
            return dict(event_validation_run_state)

    def update_event_validation_run(**updates: object) -> None:
        with teacher_run_lock:
            event_validation_run_state.update(updates)

    def now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def resolve_within(root: Path, candidate: Path) -> Path | None:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError:
            return None
        return resolved_candidate

    def resolve_container_path(path_text: str) -> Path | None:
        path = Path(path_text)
        if path.exists():
            for trusted_root in resolved_settings.trusted_media_dirs:
                trusted_path = resolve_within(trusted_root, path)
                if trusted_path is not None:
                    return trusted_path
        if resolved_settings.host_videos_dir is None:
            return path
        if path.exists():
            return resolve_within(resolved_settings.host_videos_dir, path)
        try:
            relative = PureWindowsPath(path_text).relative_to(
                PureWindowsPath(resolved_settings.windows_videos_prefix)
            )
        except ValueError:
            return resolve_within(resolved_settings.host_videos_dir, path)
        return resolve_within(
            resolved_settings.host_videos_dir,
            resolved_settings.host_videos_dir.joinpath(*relative.parts),
        )

    def resolve_import_path(path: Path) -> Path:
        if path.exists() or resolved_settings.label_import_dir is None:
            return path
        candidate = resolved_settings.label_import_dir / PureWindowsPath(str(path)).name
        return candidate if candidate.exists() else path

    def export_filename(rank: int, clip: dict) -> str:
        score = int(round(float(clip.get("final_score") or 0) * 100))
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(clip["filename"])).stem).strip(
            "_"
        )
        suffix = Path(str(clip["filename"])).suffix or ".mp4"
        return f"{rank:03d}_{score}_{stem}{suffix}"

    def enrich_clip(conn: sqlite3.Connection, clip_id: int) -> bool:
        row = clip_teacher_input_row(conn, clip_id)
        if row is None:
            return False
        feature_json = json.loads(row["feature_json"] or "{}")
        path_text = str(row["path"])
        client = teacher_client()
        base_row = conn.execute(
            """
            select label_json from teacher_labels
            where clip_id = ? and schema_version like ?
            order by id desc
            limit 1
            """,
            (clip_id, f"{TEACHER_SCHEMA_VERSION}:%"),
        ).fetchone()
        base_labels = (
            normalize_teacher_payload(json.loads(base_row["label_json"]))
            if base_row
            else None
        )
        existing_tags = [
            tag
            for tag in feature_json.get("tags", [])
            if tag not in set(TEACHER_LABEL_KEYS) | RETIRED_TEACHER_TAGS
        ]
        coarse = []
        event_frames = []
        event_timestamps: list[float] = []
        try:
            if not path_text.startswith("demo://"):
                clip_path = resolve_container_path(path_text)
                if clip_path is None:
                    raise FileNotFoundError(path_text)
                coarse = extract_timeline_keyframes(clip_path, row["duration_sec"])
                coarse_input = ClipTeacherInput(
                    filename=row["filename"],
                    duration_sec=row["duration_sec"],
                    width=row["width"],
                    height=row["height"],
                    fps=row["fps"],
                    tags=existing_tags,
                    image_paths=[frame.path for frame in coarse],
                    image_timestamps=[frame.timestamp_sec for frame in coarse],
                    image_views=[frame.view for frame in coarse],
                )
                if base_labels is None:
                    base_labels = client.label_clip(coarse_input)
                if coarse:
                    event_timestamps = client.locate_event_timestamps(coarse_input)
                cleanup_keyframes(coarse)
                coarse = []
                event_frames = extract_event_keyframes(
                    clip_path,
                    row["duration_sec"],
                    event_timestamps,
                )

            if base_labels is None:
                base_labels = client.label_clip(
                    ClipTeacherInput(
                        filename=row["filename"],
                        duration_sec=row["duration_sec"],
                        width=row["width"],
                        height=row["height"],
                        fps=row["fps"],
                        tags=existing_tags,
                        image_paths=[],
                    )
                )

            attribution = None
            if event_frames:
                ocr_observations = local_ocr.recognize_event_frames(event_frames)
                attribution = client.label_weapon_event(
                    ClipTeacherInput(
                        filename=row["filename"],
                        duration_sec=row["duration_sec"],
                        width=row["width"],
                        height=row["height"],
                        fps=row["fps"],
                        tags=existing_tags,
                        image_paths=[frame.path for frame in event_frames],
                        image_timestamps=[
                            frame.timestamp_sec for frame in event_frames
                        ],
                        image_views=[frame.view for frame in event_frames],
                        image_event_indices=[
                            frame.event_index for frame in event_frames
                        ],
                        image_event_centers=[
                            frame.event_center_sec for frame in event_frames
                        ],
                        ocr_observations=ocr_observations,
                    )
                )

            labels = merge_event_labels(base_labels, attribution)

            event_payload = {
                "locator_timestamps": event_timestamps,
                "action_evidence_version": (
                    attribution.raw_payload.get("action_evidence_version", 0)
                    if attribution
                    else 0
                ),
                "events": attribution.events if attribution else [],
                "specialist_confidence": attribution.confidence if attribution else 0.0,
                "specialist_evidence": attribution.evidence if attribution else [],
                "raw_specialist_payload": attribution.raw_payload
                if attribution
                else {},
                "ocr_observations": ocr_observations if event_frames else [],
            }
            record_teacher_events(
                conn,
                clip_id=clip_id,
                provider=client.provider,
                model=client.model,
                status=attribution.status if attribution else "no_event",
                event_data=attach_event_audit(
                    event_payload,
                    attribution
                    or WeaponAttribution(
                        labels={},
                        confidence=0.0,
                        evidence=[],
                        status="no_event",
                        events=[],
                        raw_payload={},
                    ),
                ),
            )
            record_teacher_labels(
                conn,
                clip_id=clip_id,
                provider=client.provider,
                labels=labels,
                model=client.model,
            )
            confidence_row = conn.execute(
                """
                select id, label_json from teacher_labels
                where clip_id = ?
                order by id desc
                limit 1
                """,
                (clip_id,),
            ).fetchone()
            if confidence_row is not None:
                persisted_labels = json.loads(confidence_row["label_json"] or "{}")
                persisted_labels["label_confidences"] = derive_label_confidences(
                    persisted_labels,
                    events=attribution.events if attribution else [],
                )
                conn.execute(
                    "update teacher_labels set label_json = ? where id = ?",
                    (json.dumps(persisted_labels), confidence_row["id"]),
                )
            latest_event = conn.execute(
                "select max(id) from teacher_events where clip_id = ?", (clip_id,)
            ).fetchone()[0]
            latest_label = conn.execute(
                "select max(id) from teacher_labels where clip_id = ?", (clip_id,)
            ).fetchone()[0]
            conn.execute(
                """
                update live_teacher_assignments
                set label_row_id = ?, event_row_id = ?
                where clip_id = ?
                """,
                (latest_label, latest_event, clip_id),
            )
            return True
        finally:
            cleanup_keyframes(coarse)
            cleanup_keyframes(event_frames)

    def enrich_event_validation_candidate(
        conn: sqlite3.Connection, clip_id: int
    ) -> bool:
        row = clip_teacher_input_row(conn, clip_id)
        if row is None:
            return False
        clip_path = resolve_container_path(str(row["path"]))
        if clip_path is None:
            raise FileNotFoundError(str(row["path"]))
        coarse = extract_timeline_keyframes(clip_path, row["duration_sec"])
        event_frames = []
        event_timestamps: list[float] = []
        if not coarse:
            cleanup_keyframes(coarse)
            raise ValueError("Could not extract context frames")
        try:
            feature_json = json.loads(row["feature_json"] or "{}")
            existing_tags = [
                tag
                for tag in feature_json.get("tags", [])
                if tag not in set(TEACHER_LABEL_KEYS) | RETIRED_TEACHER_TAGS
            ]
            client = teacher_client()
            coarse_input = ClipTeacherInput(
                filename=row["filename"],
                duration_sec=row["duration_sec"],
                width=row["width"],
                height=row["height"],
                fps=row["fps"],
                tags=existing_tags,
                image_paths=[frame.path for frame in coarse],
                image_timestamps=[frame.timestamp_sec for frame in coarse],
                image_views=[frame.view for frame in coarse],
            )
            context_labels = client.label_clip(coarse_input)
            event_timestamps = client.locate_event_timestamps(coarse_input)
            event_frames = extract_event_keyframes(
                clip_path, row["duration_sec"], event_timestamps
            )
            if not event_frames:
                raise ValueError("Current locator found no extractable events")
            ocr_observations = local_ocr.recognize_event_frames(event_frames)
            attribution = client.label_weapon_event(
                ClipTeacherInput(
                    filename=row["filename"],
                    duration_sec=row["duration_sec"],
                    width=row["width"],
                    height=row["height"],
                    fps=row["fps"],
                    tags=existing_tags,
                    image_paths=[frame.path for frame in event_frames],
                    image_timestamps=[frame.timestamp_sec for frame in event_frames],
                    image_views=[frame.view for frame in event_frames],
                    image_event_indices=[frame.event_index for frame in event_frames],
                    image_event_centers=[
                        frame.event_center_sec for frame in event_frames
                    ],
                    ocr_observations=ocr_observations,
                )
            )
            labels = merge_event_labels(
                context_labels,
                attribution,
            )
            record_teacher_candidate(
                conn,
                clip_id=clip_id,
                provider=client.provider,
                model=client.model,
                status=attribution.status,
                labels=labels,
                event_data=attach_event_audit(
                    {
                        "locator_timestamps": event_timestamps,
                        "action_evidence_version": 2,
                        "events": attribution.events,
                        "specialist_confidence": attribution.confidence,
                        "specialist_evidence": attribution.evidence,
                        "raw_specialist_payload": attribution.raw_payload,
                        "context_confidence": context_labels.confidence,
                        "context_evidence": context_labels.evidence,
                        "ocr_observations": ocr_observations,
                    },
                    attribution,
                ),
            )
            return True
        finally:
            cleanup_keyframes(coarse)
            cleanup_keyframes(event_frames)

    def run_teacher_background(limit: int, unresolved_only: bool = False) -> None:
        try:
            with connect(resolved_settings.resolved_database_path()) as conn:
                _ensure_database(conn, resolved_settings)
                clip_ids = (
                    clip_ids_for_reenrichment(conn, limit=limit)
                    if unresolved_only
                    else clip_ids_for_enrichment(conn, clip_id=None, limit=limit)
                )
                update_teacher_run(requested=len(clip_ids))
                for current_clip_id in clip_ids:
                    try:
                        if enrich_clip(conn, current_clip_id):
                            conn.commit()
                            with teacher_run_lock:
                                teacher_run_state["enriched"] = (
                                    int(teacher_run_state["enriched"]) + 1
                                )
                        else:
                            conn.rollback()
                            with teacher_run_lock:
                                teacher_run_state["failed"] = (
                                    int(teacher_run_state["failed"]) + 1
                                )
                                teacher_run_state["last_error"] = (
                                    f"clip {current_clip_id}: "
                                    "Clip not found during enrichment"
                                )
                    except Exception as exc:
                        conn.rollback()
                        with teacher_run_lock:
                            teacher_run_state["failed"] = (
                                int(teacher_run_state["failed"]) + 1
                            )
                            teacher_run_state["last_error"] = (
                                f"clip {current_clip_id}: {exc}"
                            )
                rescore_all_clips(conn)
        except Exception as exc:
            with teacher_run_lock:
                unaccounted = max(
                    0,
                    int(teacher_run_state["requested"])
                    - int(teacher_run_state["enriched"])
                    - int(teacher_run_state["failed"]),
                )
                teacher_run_state["failed"] = (
                    int(teacher_run_state["failed"]) + unaccounted
                )
                teacher_run_state["last_error"] = f"run: {exc}"
        finally:
            update_teacher_run(running=False, finished_at=now_iso())

    def run_event_validation_background(clip_ids: list[int]) -> None:
        try:
            with connect(resolved_settings.resolved_database_path()) as conn:
                _ensure_database(conn, resolved_settings)
                for current_clip_id in clip_ids:
                    try:
                        if enrich_event_validation_candidate(conn, current_clip_id):
                            conn.commit()
                            with teacher_run_lock:
                                event_validation_run_state["enriched"] = (
                                    int(event_validation_run_state["enriched"]) + 1
                                )
                        else:
                            with teacher_run_lock:
                                event_validation_run_state["failed"] = (
                                    int(event_validation_run_state["failed"]) + 1
                                )
                                event_validation_run_state["last_error"] = (
                                    f"clip {current_clip_id}: not found"
                                )
                    except Exception as exc:
                        conn.rollback()
                        with teacher_run_lock:
                            event_validation_run_state["failed"] = (
                                int(event_validation_run_state["failed"]) + 1
                            )
                            event_validation_run_state["last_error"] = (
                                f"clip {current_clip_id}: {exc}"
                            )
        except Exception as exc:
            with teacher_run_lock:
                unaccounted = max(
                    0,
                    int(event_validation_run_state["requested"])
                    - int(event_validation_run_state["enriched"])
                    - int(event_validation_run_state["failed"]),
                )
                event_validation_run_state["failed"] = (
                    int(event_validation_run_state["failed"]) + unaccounted
                )
                event_validation_run_state["last_error"] = f"run: {exc}"
        finally:
            update_event_validation_run(running=False, finished_at=now_iso())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ai/status", response_model=AiStatus)
    def ai_status() -> dict:
        amd_detected, amd_name = detect_amd_gpu()
        return {
            "vlm_provider": resolved_settings.vlm_provider,
            "fireworks_configured": bool(resolved_settings.fireworks_api_key),
            "fireworks_model": resolved_settings.fireworks_model,
            "amd_developer_cloud_configured": bool(
                resolved_settings.amd_developer_cloud_api_key
                and resolved_settings.amd_developer_cloud_base_url
            ),
            "amd_developer_cloud_model": resolved_settings.amd_developer_cloud_model,
            "accelerator": resolved_settings.accelerator,
            "amd_gpu_detected": amd_detected,
            "amd_gpu_name": amd_name,
            "local_ocr_enabled": resolved_settings.local_ocr_enabled,
            "local_ocr_available": local_ocr.available,
        }

    @app.get("/clips", response_model=list[ClipRecord])
    def list_clips() -> list[dict]:
        with db() as conn:
            return list_ranked_clips(conn)

    @app.get("/clips/{clip_id}/video")
    def clip_video(clip_id: int) -> FileResponse:
        with db() as conn:
            row = conn.execute(
                "select path from clips where id = ?", (clip_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"Clip not found: {clip_id}"
                )
            path_text = str(row["path"])
            if path_text.startswith(("demo://", "snapshot://")):
                raise HTTPException(
                    status_code=404, detail="This ranked record does not include a local video file"
                )
            video_path = resolve_container_path(path_text)
            if video_path is None or not video_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"Video file not found: {path_text}"
                )
            return FileResponse(
                video_path, media_type="video/mp4", filename=video_path.name
            )

    @app.post("/folders/scan", response_model=ScanFolderResponse)
    def scan_folder(request: ScanFolderRequest) -> dict:
        folder = Path(request.path).expanduser()
        try:
            found = find_clip_files(folder)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NotADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with db() as conn:
            indexed = 0
            for path in found:
                clip_id = index_clip_path(conn, path)
                conn.commit()
                indexed += 1
                if request.enrich and teacher_configured():
                    try:
                        if enrich_clip(conn, clip_id):
                            conn.commit()
                    except Exception:
                        conn.rollback()
            rescore_all_clips(conn)
            clips = list_ranked_clips(conn)
        return {"indexed": indexed, "total_found": len(found), "clips": clips}

    @app.post("/clips/export", response_model=ExportClipsResponse)
    def export_clips(request: ExportClipsRequest) -> dict:
        destination = Path(request.destination).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        with db() as conn:
            clips = list_ranked_clips(conn)
            if request.mode == "keepers":
                selected = [
                    clip
                    for clip in clips
                    if any(
                        item in {"favorite", "keep"}
                        for item in clip.get("feedback", [])
                    )
                ]
            else:
                selected = clips
            selected = selected[: request.limit]

            exported: list[str] = []
            for rank, clip in enumerate(selected, start=1):
                path_text = str(clip["path"])
                if path_text.startswith("demo://"):
                    continue
                source = resolve_container_path(path_text)
                if source is None or not source.exists():
                    continue
                target = destination / export_filename(rank, clip)
                shutil.copy2(source, target)
                exported.append(str(target))
        return {
            "exported": len(exported),
            "destination": str(destination),
            "files": exported,
        }

    @app.post("/taste/profile", response_model=TasteProfileResponse)
    def save_taste_profile(request: TasteProfileRequest) -> dict:
        with db() as conn:
            saved = save_taste_preferences(conn, request.preferences)
            rescore_all_clips(conn)
            return {"saved": saved, "clips": list_ranked_clips(conn)}

    @app.post("/taste/reset", response_model=TasteProfileResponse)
    def reset_taste_profile() -> dict:
        with db() as conn:
            deleted = reset_taste_preferences(conn)
            rescore_all_clips(conn)
            return {"saved": deleted, "clips": list_ranked_clips(conn)}

    @app.post("/taste/import-liked-folder", response_model=ImportLikedFolderResponse)
    def import_liked_clips(request: ImportLikedFolderRequest) -> dict:
        folder = Path(request.path).expanduser()
        try:
            with db() as conn:
                imported, total_found = import_liked_folder(conn, folder)
                rescore_all_clips(conn)
                return {
                    "imported": imported,
                    "total_found": total_found,
                    "clips": list_ranked_clips(conn),
                }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NotADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ai/enrich", response_model=EnrichResponse)
    def enrich(request: EnrichRequest) -> dict:
        if not teacher_configured():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Configure FIREWORKS_API_KEY for Fireworks, or "
                    "AMD_DEVELOPER_CLOUD_API_KEY plus AMD_DEVELOPER_CLOUD_BASE_URL for AMD Developer Cloud."
                ),
            )
        with db() as conn:
            clip_ids = (
                clip_ids_for_reenrichment(conn, limit=request.limit)
                if request.unresolved_only and request.clip_id is None
                else clip_ids_for_enrichment(
                    conn, clip_id=request.clip_id, limit=request.limit
                )
            )
            if request.clip_id is not None and not clip_ids:
                raise HTTPException(
                    status_code=404, detail=f"Clip not found: {request.clip_id}"
                )
            enriched = 0
            failed = 0
            errors = []
            for current_clip_id in clip_ids:
                try:
                    if enrich_clip(conn, current_clip_id):
                        enriched += 1
                        conn.commit()
                    else:
                        conn.rollback()
                        failed += 1
                        errors.append(
                            {
                                "clip_id": current_clip_id,
                                "error": "Clip not found during enrichment",
                            }
                        )
                except Exception as exc:
                    conn.rollback()
                    failed += 1
                    errors.append({"clip_id": current_clip_id, "error": str(exc)})
            rescore_all_clips(conn)
            return {
                "enriched": enriched,
                "failed": failed,
                "errors": errors,
                "clips": list_ranked_clips(conn),
            }

    @app.post("/ai/enrich/background", response_model=TeacherRunStatus)
    def start_background_enrich(request: TeacherRunRequest) -> dict:
        if not teacher_configured():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Configure FIREWORKS_API_KEY for Fireworks, or "
                    "AMD_DEVELOPER_CLOUD_API_KEY plus AMD_DEVELOPER_CLOUD_BASE_URL for AMD Developer Cloud."
                ),
            )
        with teacher_run_lock:
            if teacher_run_state["running"]:
                return dict(teacher_run_state)
            teacher_run_state.update(
                {
                    "running": True,
                    "requested": 0,
                    "enriched": 0,
                    "failed": 0,
                    "last_error": None,
                    "started_at": now_iso(),
                    "finished_at": None,
                }
            )
        Thread(
            target=run_teacher_background,
            args=(request.limit, request.unresolved_only),
            daemon=True,
        ).start()
        return teacher_run_snapshot()

    @app.get("/ai/enrich/background", response_model=TeacherRunStatus)
    def get_background_enrich() -> dict:
        return teacher_run_snapshot()

    @app.post("/ai/event-validation/background", response_model=TeacherRunStatus)
    def start_event_validation(request: EventValidationRequest) -> dict:
        if not teacher_configured():
            raise HTTPException(
                status_code=400, detail="Configure the VLM provider first"
            )
        clip_ids = list(dict.fromkeys(request.clip_ids))
        with teacher_run_lock:
            if event_validation_run_state["running"]:
                return dict(event_validation_run_state)
            event_validation_run_state.update(
                {
                    "running": True,
                    "requested": len(clip_ids),
                    "enriched": 0,
                    "failed": 0,
                    "last_error": None,
                    "started_at": now_iso(),
                    "finished_at": None,
                }
            )
        Thread(
            target=run_event_validation_background, args=(clip_ids,), daemon=True
        ).start()
        return event_validation_run_snapshot()

    @app.get("/ai/event-validation/background", response_model=TeacherRunStatus)
    def get_event_validation() -> dict:
        return event_validation_run_snapshot()

    @app.get("/training/status", response_model=TrainingStatus)
    def get_training_status() -> dict:
        with db() as conn:
            return training_status(conn)

    @app.post("/training/import-labels", response_model=ImportLabelsResponse)
    def import_labels(request: ImportLabelsRequest) -> dict:
        jsonl_path = resolve_import_path(Path(request.path).expanduser())
        if not jsonl_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Label file not found: {jsonl_path}"
            )
        with db() as conn:
            result = import_labeled_jsonl(conn, jsonl_path)
            return {
                "imported": result.imported,
                "keepers": result.keepers,
                "skips": result.skips,
                "ignored": result.ignored,
                "clips": list_ranked_clips(conn),
            }

    @app.post("/feedback", response_model=list[ClipRecord])
    def post_feedback(request: FeedbackRequest) -> list[dict]:
        with db() as conn:
            if not clip_exists(conn, request.clip_id):
                raise HTTPException(
                    status_code=404, detail=f"Clip not found: {request.clip_id}"
                )
            try:
                record_feedback(
                    conn,
                    clip_id=request.clip_id,
                    action=request.action,
                    label=request.label,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            rescore_all_clips(conn)
            return list_ranked_clips(conn)

    def latest_label_reviews(
        conn: sqlite3.Connection, clip_ids: list[int] | None = None
    ) -> dict[int, dict[str, str]]:
        where = ""
        params: list[object] = []
        if clip_ids:
            placeholders = ",".join("?" for _ in clip_ids)
            where = f"where r.clip_id in ({placeholders})"
            params.extend(clip_ids)
        rows = conn.execute(
            f"""
            select r.clip_id, r.label_key, r.expected_value
            from teacher_label_reviews r
            join (
                select clip_id, label_key, max(id) as id
                from teacher_label_reviews
                group by clip_id, label_key
            ) latest on latest.id = r.id
            {where}
            """,
            params,
        ).fetchall()
        reviews: dict[int, dict[str, str]] = {}
        for row in rows:
            reviews.setdefault(int(row["clip_id"]), {})[str(row["label_key"])] = str(
                row["expected_value"]
            )
        return reviews

    def latest_validation_candidates(conn: sqlite3.Connection) -> dict[int, dict]:
        rows = conn.execute(
            """
            select candidate.*
            from teacher_event_candidates candidate
            join (
                select clip_id, max(id) as id
                from teacher_event_candidates
                where candidate_version = ?
                group by clip_id
            ) latest on latest.id = candidate.id
            """,
            (EVENT_VALIDATION_SCHEMA,),
        ).fetchall()
        candidates: dict[int, dict] = {}
        for row in rows:
            raw_labels = json.loads(row["label_json"])
            normalized_labels = normalize_teacher_payload(raw_labels)
            event_data = json.loads(row["event_json"] or "{}")
            evidence = list(
                dict.fromkeys(
                    [
                        *raw_labels.get("evidence", []),
                        *event_data.get("context_evidence", []),
                        *event_data.get("specialist_evidence", []),
                    ]
                )
            )
            candidates[int(row["clip_id"])] = {
                "candidate_labels": {
                    label: getattr(normalized_labels, label)
                    for label in TEACHER_LABEL_KEYS
                },
                "candidate_evidence": evidence,
                "candidate_status": str(row["status"]),
                "candidate_version": str(row["candidate_version"]),
                "candidate_created_at": str(row["created_at"]),
                "candidate_event_audit": event_audit_summary(event_data),
            }
        return candidates

    @app.get("/eval/teacher-clips", response_model=EvalClipResponse)
    def teacher_eval_clips(
        limit: int = 30,
        label_key: str | None = None,
        clip_id: int | None = None,
        mode: str = "live",
    ) -> dict:
        if label_key is not None and label_key not in set(TEACHER_LABEL_KEYS):
            raise HTTPException(status_code=400, detail=f"Unknown label: {label_key}")
        if mode not in {"candidate", "live"}:
            raise HTTPException(status_code=400, detail=f"Unknown eval mode: {mode}")
        with db() as conn:
            clips = list_ranked_clips(conn)
            candidates = latest_validation_candidates(conn)
            for clip in clips:
                clip.update(candidates.get(int(clip["id"]), {}))
            if mode == "candidate":
                clips = [clip for clip in clips if clip.get("candidate_labels")]
            else:
                clips = [clip for clip in clips if clip.get("teacher_provider")]
            requested = max(1, min(limit, 100))
            if clip_id is not None:
                clips = [clip for clip in clips if int(clip["id"]) == clip_id]
                if not clips:
                    description = (
                        "Validation candidate"
                        if mode == "candidate"
                        else "Teacher-labeled clip"
                    )
                    raise HTTPException(
                        status_code=404, detail=f"{description} not found: {clip_id}"
                    )
            elif label_key is not None:
                label_field = (
                    "candidate_labels" if mode == "candidate" else "teacher_labels"
                )
                positive = [
                    clip
                    for clip in clips
                    if clip.get(label_field, {}).get(label_key) == "yes"
                ]
                negative = [
                    clip
                    for clip in clips
                    if clip.get(label_field, {}).get(label_key) != "yes"
                ]
                positive_limit = min(len(positive), max(1, requested // 2))
                clips = (
                    positive[:positive_limit] + negative[: requested - positive_limit]
                )
            else:
                clips = clips[:requested]
            reviews = latest_label_reviews(conn, [int(clip["id"]) for clip in clips])
            for clip in clips:
                clip["label_reviews"] = reviews.get(int(clip["id"]), {})
            return {"mode": mode, "labels": TEACHER_LABEL_KEYS, "clips": clips}

    @app.post("/eval/teacher-review", response_model=EvalSummaryResponse)
    def record_teacher_label_review(
        request: TeacherLabelReviewRequest, mode: str = "live"
    ) -> dict:
        if request.label_key not in set(TEACHER_LABEL_KEYS):
            raise HTTPException(
                status_code=400, detail=f"Unknown label: {request.label_key}"
            )
        if mode not in {"candidate", "live"}:
            raise HTTPException(status_code=400, detail=f"Unknown eval mode: {mode}")
        with db() as conn:
            if not clip_exists(conn, request.clip_id):
                raise HTTPException(
                    status_code=404, detail=f"Clip not found: {request.clip_id}"
                )
            conn.execute(
                """
                insert into teacher_label_reviews(clip_id, label_key, expected_value, notes, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (
                    request.clip_id,
                    request.label_key,
                    request.expected_value,
                    request.notes,
                    now_iso(),
                ),
            )
            if mode == "candidate":
                return event_validation_summary_payload(conn)
            return teacher_eval_summary_payload(conn)

    def eval_summary_from_rows(reviews: list[sqlite3.Row]) -> dict:
        metrics = {
            label: {
                "label_key": label,
                "reviewed": 0,
                "teacher_yes": 0,
                "expected_yes": 0,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_negative": 0,
                "abstention": 0,
                "prediction_abstention": 0,
                "prediction_coverage": None,
            }
            for label in TEACHER_LABEL_KEYS
        }
        for row in reviews:
            label_key = str(row["label_key"])
            if (
                label_key not in metrics
                or str(row["expected_value"]) == "uncertain"
                or not row["label_json"]
            ):
                continue
            teacher_labels = normalize_teacher_payload(json.loads(row["label_json"]))
            metric = metrics[label_key]
            expected = str(row["expected_value"])
            if expected == "uncertain":
                metric["abstention"] += 1
                continue
            prediction = getattr(teacher_labels, label_key)
            if prediction == "uncertain":
                metric["prediction_abstention"] += 1
                continue
            predicted_yes = prediction == "yes"
            expected_yes = expected == "yes"
            metric["reviewed"] += 1
            metric["teacher_yes"] += 1 if predicted_yes else 0
            metric["expected_yes"] += 1 if expected_yes else 0
            if predicted_yes and expected_yes:
                metric["true_positive"] += 1
            elif predicted_yes and not expected_yes:
                metric["false_positive"] += 1
            elif not predicted_yes and expected_yes:
                metric["false_negative"] += 1
            else:
                metric["true_negative"] += 1
        for metric in metrics.values():
            precision_denominator = metric["true_positive"] + metric["false_positive"]
            recall_denominator = metric["true_positive"] + metric["false_negative"]
            reviewed = metric["reviewed"]
            metric["precision"] = (
                metric["true_positive"] / precision_denominator
                if precision_denominator
                else None
            )
            metric["recall"] = (
                metric["true_positive"] / recall_denominator
                if recall_denominator
                else None
            )
            metric["accuracy"] = (
                (metric["true_positive"] + metric["true_negative"]) / reviewed
                if reviewed
                else None
            )
            total_predictions = reviewed + metric["prediction_abstention"]
            metric["prediction_coverage"] = (
                reviewed / total_predictions if total_predictions else None
            )
        return {"labels": list(metrics.values())}

    def teacher_eval_summary_payload(conn: sqlite3.Connection) -> dict:
        reviews = conn.execute(
            """
            select r.clip_id, r.label_key, r.expected_value, tl.label_json
            from teacher_label_reviews r
            join (
                select clip_id, label_key, max(id) as id
                from teacher_label_reviews
                group by clip_id, label_key
            ) latest on latest.id = r.id
            left join teacher_labels tl on tl.id = (
                select id
                from teacher_labels
                where clip_id = r.clip_id
                  and (schema_version like ? or schema_version like ?)
                order by case when schema_version like ? then 1 else 0 end desc, id desc
                limit 1
            )
            """,
            (
                f"{TEACHER_SCHEMA_VERSION}:%",
                f"{TEACHER_FALLBACK_SCHEMA_VERSION}:%",
                f"{TEACHER_SCHEMA_VERSION}:%",
            ),
        ).fetchall()
        return eval_summary_from_rows(reviews)

    @app.get("/eval/teacher-summary", response_model=EvalSummaryResponse)
    def teacher_eval_summary() -> dict:
        with db() as conn:
            return teacher_eval_summary_payload(conn)

    def event_validation_summary_payload(conn: sqlite3.Connection) -> dict:
        reviews = conn.execute(
            """
            select r.clip_id, r.label_key, r.expected_value, candidate.label_json
            from teacher_label_reviews r
            join (
                select clip_id, label_key, max(id) as id
                from teacher_label_reviews
                group by clip_id, label_key
            ) latest_review on latest_review.id = r.id
            left join teacher_event_candidates candidate on candidate.id = (
                select id
                from teacher_event_candidates
                where clip_id = r.clip_id and candidate_version = ?
                order by id desc limit 1
            )
            """,
            (EVENT_VALIDATION_SCHEMA,),
        ).fetchall()
        return eval_summary_from_rows(reviews)

    @app.get("/eval/event-validation-summary", response_model=EvalSummaryResponse)
    def event_validation_summary() -> dict:
        with db() as conn:
            return event_validation_summary_payload(conn)

    @app.get("/eval/batches", response_model=EvalBatchListResponse)
    def eval_batches() -> dict:
        with db() as conn:
            return {"batches": list_batches(conn)}

    @app.get("/eval/student-reports", response_model=StudentReportsResponse)
    def eval_student_reports() -> dict:
        """Offline student speed/agreement reports (separate from teacher batches)."""
        return load_student_reports(resolved_settings.student_artifacts_dir)

    @app.post("/eval/batches", response_model=EvalBatchMetricsResponse)
    def create_eval_batch(request: CreateEvalBatchRequest) -> dict:
        with db() as conn:
            try:
                batch = create_batch_from_candidates(
                    conn,
                    batch_key=request.batch_key.strip(),
                    candidate_version=request.candidate_version
                    or EVENT_VALIDATION_SCHEMA,
                    provider=request.provider,
                    model=request.model,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return batch_metrics(conn, int(batch["id"]))

    @app.get("/eval/batches/{batch_id}", response_model=EvalBatchMetricsResponse)
    def eval_batch_detail(batch_id: int) -> dict:
        with db() as conn:
            try:
                return batch_metrics(conn, batch_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/eval/batches/{batch_id}/clips", response_model=EvalClipResponse)
    def eval_batch_clips(
        batch_id: int,
        limit: int = 100,
        label_key: str | None = None,
        clip_id: int | None = None,
    ) -> dict:
        if label_key is not None and label_key not in set(TEACHER_LABEL_KEYS):
            raise HTTPException(status_code=400, detail=f"Unknown label: {label_key}")
        with db() as conn:
            try:
                batch = get_batch(conn, batch_id)
                snapshots = list_batch_clips(conn, batch_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            live = {int(item["id"]): item for item in list_ranked_clips(conn)}
            snapshot_ids = [int(item["prediction_snapshot_id"]) for item in snapshots]
            reviews = latest_snapshot_label_reviews(conn, snapshot_ids)
            highlight_reviews = latest_snapshot_highlight_reviews(conn, snapshot_ids)
            clips: list[dict] = []
            for snapshot in snapshots:
                item = live.get(int(snapshot["clip_id"]))
                if item is None:
                    continue
                candidate_labels = snapshot["candidate_labels"]
                if clip_id is not None and int(item["id"]) != clip_id:
                    continue
                if label_key is not None and candidate_labels.get(label_key) != "yes":
                    continue
                event_data = snapshot["event_json"]
                enriched = dict(item)
                enriched.update(
                    {
                        "prediction_snapshot_id": snapshot["prediction_snapshot_id"],
                        "candidate_labels": candidate_labels,
                        "candidate_evidence": list(
                            event_data.get("specialist_evidence", [])
                        ),
                        "candidate_status": "complete"
                        if snapshot["complete"]
                        else "incomplete",
                        "candidate_version": batch["candidate_version"],
                        "candidate_event_audit": event_audit_summary(event_data),
                        "label_reviews": reviews.get(
                            int(snapshot["prediction_snapshot_id"]), {}
                        ),
                        "highlight_reviews": highlight_reviews.get(
                            int(snapshot["prediction_snapshot_id"]), {}
                        ),
                    }
                )
                clips.append(enriched)
            return {
                "mode": "candidate",
                "labels": TEACHER_LABEL_KEYS,
                "clips": clips[: max(1, min(limit, 100))],
            }

    @app.post("/eval/snapshot-review", response_model=EvalBatchMetricsResponse)
    def eval_snapshot_review(request: SnapshotLabelReviewRequest) -> dict:
        with db() as conn:
            try:
                record_snapshot_label_review(
                    conn,
                    prediction_snapshot_id=request.prediction_snapshot_id,
                    label_key=request.label_key,
                    expected_value=request.expected_value,
                    notes=request.notes,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            row = conn.execute(
                "select batch_id from prediction_snapshots where id = ?",
                (request.prediction_snapshot_id,),
            ).fetchone()
            if row is None or row["batch_id"] is None:
                raise HTTPException(status_code=404, detail="Snapshot batch not found")
            return batch_metrics(conn, int(row["batch_id"]))

    @app.post("/eval/highlight-review", response_model=dict)
    def eval_highlight_review(request: HighlightReviewRequest) -> dict:
        if request.aspect not in HIGHLIGHT_REVIEW_ASPECTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown highlight aspect: {request.aspect}",
            )
        with db() as conn:
            try:
                record_highlight_review(
                    conn,
                    prediction_snapshot_id=request.prediction_snapshot_id,
                    aspect=request.aspect,
                    expected_value=request.expected_value,
                    notes=request.notes,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"saved": True, "aspect": request.aspect}

    @app.post("/eval/batches/{batch_id}/promote", response_model=PromoteBatchResponse)
    def eval_promote_batch(batch_id: int, request: PromoteBatchRequest) -> dict:
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail="Promotion requires confirm=true",
            )
        with db() as conn:
            try:
                get_batch(conn, batch_id)
                return promote_batch(conn, batch_id, force=request.force)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/demo/reset", response_model=list[ClipRecord])
    def reset_demo() -> list[dict]:
        if not resolved_settings.demo_data_path.exists():
            raise HTTPException(status_code=404, detail="Demo data file not found")
        with db() as conn:
            conn.execute(
                "delete from clips where source = 'demo' or path like 'snapshot://%'"
            )
            seed_demo_clips(conn, load_demo_clips(resolved_settings.demo_data_path))
            rescore_all_clips(conn)
            return list_ranked_clips(conn)

    if resolved_settings.static_dir.exists():
        assets_dir = resolved_settings.static_dir / "assets"
        resolved_assets_dir = resolve_within(resolved_settings.static_dir, assets_dir)
        if resolved_assets_dir is not None and resolved_assets_dir.exists():
            app.mount(
                "/assets", StaticFiles(directory=resolved_assets_dir), name="assets"
            )

        @app.get("/")
        def frontend_index() -> FileResponse:
            index_path = resolve_within(
                resolved_settings.static_dir,
                resolved_settings.static_dir / "index.html",
            )
            if index_path is None:
                raise HTTPException(status_code=404, detail="Static file not found")
            return FileResponse(
                index_path,
                headers={"Cache-Control": "no-cache"},
            )

        @app.get("/{full_path:path}")
        def frontend_fallback(full_path: str) -> FileResponse:
            candidate = resolve_within(
                resolved_settings.static_dir,
                resolved_settings.static_dir / full_path,
            )
            if candidate is None:
                raise HTTPException(status_code=404, detail="Static file not found")
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return frontend_index()

    return app


app = create_app()
