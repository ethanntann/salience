from pydantic import BaseModel, Field


class ClipRecord(BaseModel):
    id: int
    path: str
    filename: str
    duration_sec: float | None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    size_bytes: int | None = None
    source: str = "local"
    video_url: str | None = None
    final_score: float | None = None
    base_score: float | None = None
    personal_score: float | None = None
    confidence: float | None = None
    explanation: str | None = None
    tags: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    thumbnail_variant: str = "ridge"
    teacher_provider: str | None = None
    teacher_confidence: float | None = None
    teacher_labels: dict[str, str] = Field(default_factory=dict)
    teacher_evidence: list[str] = Field(default_factory=list)
    highlight_description: str | None = None


class FeedbackRequest(BaseModel):
    clip_id: int
    action: str
    label: str | None = None


class ScanFolderRequest(BaseModel):
    path: str
    enrich: bool = False


class ScanFolderResponse(BaseModel):
    indexed: int
    total_found: int
    clips: list[ClipRecord]


class AiStatus(BaseModel):
    vlm_provider: str
    fireworks_configured: bool
    fireworks_model: str
    amd_developer_cloud_configured: bool
    amd_developer_cloud_model: str
    accelerator: str
    amd_gpu_detected: bool
    amd_gpu_name: str | None = None
    local_ocr_enabled: bool
    local_ocr_available: bool


class EnrichRequest(BaseModel):
    clip_id: int | None = None
    limit: int = Field(default=10, ge=1, le=100)
    unresolved_only: bool = False


class EnrichError(BaseModel):
    clip_id: int
    error: str


class EnrichResponse(BaseModel):
    enriched: int
    failed: int
    errors: list[EnrichError] = Field(default_factory=list)
    clips: list[ClipRecord]


class TeacherRunRequest(BaseModel):
    limit: int = Field(default=10000, ge=1, le=10000)
    unresolved_only: bool = False


class EventValidationRequest(BaseModel):
    clip_ids: list[int] = Field(min_length=1, max_length=100)


class TeacherRunStatus(BaseModel):
    running: bool
    requested: int
    enriched: int
    failed: int
    last_error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class TrainingStatus(BaseModel):
    clips: int
    teacher_labeled: int
    teacher_pending: int
    teacher_progress: float
    feedback_count: int
    positive_count: int
    negative_count: int
    positive_avg_personal_score: float | None = None
    negative_avg_personal_score: float | None = None
    personal_score_separation: float | None = None


class ImportLabelsRequest(BaseModel):
    path: str


class ImportLabelsResponse(BaseModel):
    imported: int
    keepers: int
    skips: int
    ignored: int
    clips: list[ClipRecord]


class ExportClipsRequest(BaseModel):
    destination: str
    mode: str = Field(default="keepers", pattern="^(keepers|top)$")
    limit: int = Field(default=10, ge=1, le=500)


class ExportClipsResponse(BaseModel):
    exported: int
    destination: str
    files: list[str] = Field(default_factory=list)


class TasteProfileRequest(BaseModel):
    preferences: dict[str, int] = Field(default_factory=dict)


class TasteProfileResponse(BaseModel):
    saved: int
    clips: list[ClipRecord]


class ImportLikedFolderRequest(BaseModel):
    path: str


class ImportLikedFolderResponse(BaseModel):
    imported: int
    total_found: int
    clips: list[ClipRecord]


class EventAuditSummary(BaseModel):
    available: bool = False
    highlight_description: str | None = None
    primary_event: dict | None = None
    secondary_events: list[dict] = Field(default_factory=list)
    rejected_events: list[dict] = Field(default_factory=list)
    multi_kill: bool | None = None
    active_finish_count: int | None = None


class EvalClipRecord(ClipRecord):
    prediction_snapshot_id: int | None = None
    label_reviews: dict[str, str] = Field(default_factory=dict)
    highlight_reviews: dict[str, str] = Field(default_factory=dict)
    candidate_labels: dict[str, str] = Field(default_factory=dict)
    candidate_evidence: list[str] = Field(default_factory=list)
    candidate_status: str | None = None
    candidate_version: str | None = None
    candidate_created_at: str | None = None
    candidate_event_audit: EventAuditSummary = Field(default_factory=EventAuditSummary)


class EvalClipResponse(BaseModel):
    mode: str
    labels: list[str]
    clips: list[EvalClipRecord]


class TeacherLabelReviewRequest(BaseModel):
    clip_id: int
    label_key: str
    expected_value: str = Field(pattern="^(yes|no|uncertain)$")
    notes: str | None = None


class LabelEvalMetric(BaseModel):
    label_key: str
    reviewed: int
    teacher_yes: int
    expected_yes: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    abstention: int = 0
    prediction_abstention: int = 0
    prediction_coverage: float | None = None
    precision: float | None = None
    recall: float | None = None
    accuracy: float | None = None


class EvalSummaryResponse(BaseModel):
    labels: list[LabelEvalMetric]


class CreateEvalBatchRequest(BaseModel):
    batch_key: str = Field(min_length=1, max_length=120)
    candidate_version: str | None = None
    provider: str = "manual"
    model: str = "manual"


class SnapshotLabelReviewRequest(BaseModel):
    prediction_snapshot_id: int
    label_key: str
    expected_value: str = Field(pattern="^(yes|no|uncertain)$")
    notes: str | None = None


class HighlightReviewRequest(BaseModel):
    prediction_snapshot_id: int
    aspect: str
    expected_value: str = Field(pattern="^(yes|no|uncertain)$")
    notes: str | None = None


class PromoteBatchRequest(BaseModel):
    confirm: bool = False
    force: bool = False


class EvalBatchRecord(BaseModel):
    id: int
    batch_key: str
    pipeline_manifest_id: int
    candidate_version: str
    status: str
    created_at: str
    updated_at: str
    item_count: int
    incomplete_count: int
    incomplete_rate: float | None = None
    promotable: bool = False


class EvalBatchListResponse(BaseModel):
    batches: list[EvalBatchRecord]


class EvalBatchMetricsResponse(BaseModel):
    batch: EvalBatchRecord
    labels: list[dict]
    gates: dict
    required_labels: list[str]
    highlights: list[dict] = Field(default_factory=list)


class PromoteBatchResponse(BaseModel):
    batch_id: int
    status: str
    promoted: int
    skipped_incomplete: int
    force: bool = False
    gates: dict


class StudentPhaseTiming(BaseModel):
    mean: float
    median: float
    p90: float
    max: float


class StudentSpeedSummary(BaseModel):
    clips: int
    median_sec: float
    mean_sec: float
    p90_sec: float
    max_sec: float
    phases: dict[str, StudentPhaseTiming] = Field(default_factory=dict)


class StudentAgreementLabel(BaseModel):
    label_key: str
    agree: float
    precision: float
    recall: float
    teacher_yes: int
    student_yes: int
    uncertain_rate: float
    clips: int


class StudentExampleEvent(BaseModel):
    event_index: int | None = None
    status: str | None = None
    event_kind: str | None = None
    finish_timestamp: float | None = None
    resolved_weapon: str | None = None
    target_was_active: bool | None = None
    target_was_downed: bool | None = None
    damage_aim_state: str | None = None
    visual_action_supported: bool | None = None
    summary: str | None = None


class StudentExampleClip(BaseModel):
    clip_id: int
    filename: str
    duration_sec: float
    total_sec: float
    locator_timestamps: list[float] = Field(default_factory=list)
    locator_events: int = 0
    attribution_status: str | None = None
    yes_labels: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    events: list[StudentExampleEvent] = Field(default_factory=list)
    video_url: str | None = None


class StudentReportsResponse(BaseModel):
    available: bool
    model: str | None = None
    speed: StudentSpeedSummary | None = None
    agreement_labels: list[StudentAgreementLabel] = Field(default_factory=list)
    example_clips: list[StudentExampleClip] = Field(default_factory=list)
    speed_report_path: str | None = None
    agreement_report_path: str | None = None
    message: str | None = None
