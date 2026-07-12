export type Clip = {
  id: number;
  filename: string;
  path: string;
  duration_sec: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  size_bytes: number | null;
  source: string;
  video_url: string | null;
  final_score: number | null;
  base_score: number | null;
  personal_score: number | null;
  confidence: number | null;
  explanation: string | null;
  tags: string[];
  feedback: string[];
  thumbnail_variant: string;
  teacher_provider: string | null;
  teacher_confidence: number | null;
  teacher_labels: Record<string, string>;
  teacher_evidence: string[];
  highlight_description: string | null;
};

export type ScanResponse = {
  indexed: number;
  total_found: number;
  clips: Clip[];
};

export type AiStatus = {
  vlm_provider: string;
  fireworks_configured: boolean;
  fireworks_model: string;
  amd_developer_cloud_configured: boolean;
  amd_developer_cloud_model: string;
  accelerator: string;
  amd_gpu_detected: boolean;
  amd_gpu_name: string | null;
};

export type EnrichResponse = {
  enriched: number;
  failed: number;
  errors: Array<{ clip_id: number; error: string }>;
  clips: Clip[];
};

export type TrainingStatus = {
  clips: number;
  teacher_labeled: number;
  teacher_pending: number;
  teacher_progress: number;
  feedback_count: number;
  positive_count: number;
  negative_count: number;
  positive_avg_personal_score: number | null;
  negative_avg_personal_score: number | null;
  personal_score_separation: number | null;
};

export type TeacherRunStatus = {
  running: boolean;
  requested: number;
  enriched: number;
  failed: number;
  last_error: string | null;
  started_at: string | null;
  finished_at: string | null;
};

export type ImportLabelsResponse = {
  imported: number;
  keepers: number;
  skips: number;
  ignored: number;
  clips: Clip[];
};

export type ExportResponse = {
  exported: number;
  destination: string;
  files: string[];
};

export type TasteProfileResponse = {
  saved: number;
  clips: Clip[];
};

export type ImportLikedFolderResponse = {
  imported: number;
  total_found: number;
  clips: Clip[];
};

export type EventAuditSummary = {
  available: boolean;
  highlight_description: string | null;
  primary_event: Record<string, unknown> | null;
  secondary_events: Array<Record<string, unknown>>;
  rejected_events: Array<Record<string, unknown>>;
  multi_kill: boolean | null;
  active_finish_count: number | null;
};

export type EvalClip = Clip & {
  prediction_snapshot_id: number | null;
  label_reviews: Record<string, string>;
  highlight_reviews: Record<string, string>;
  candidate_labels: Record<string, string>;
  candidate_evidence: string[];
  candidate_status: string | null;
  candidate_version: string | null;
  candidate_created_at: string | null;
  candidate_event_audit: EventAuditSummary;
};

export type EvalClipResponse = {
  mode: EvalMode;
  labels: string[];
  clips: EvalClip[];
};

export type EvalMode = "candidate" | "live";

export type LabelEvalMetric = {
  label_key: string;
  reviewed: number;
  teacher_yes: number;
  expected_yes: number;
  true_positive: number;
  false_positive: number;
  false_negative: number;
  true_negative: number;
  abstention: number;
  prediction_abstention: number;
  prediction_coverage: number | null;
  precision: number | null;
  recall: number | null;
  accuracy: number | null;
};

export type EvalSummaryResponse = {
  labels: LabelEvalMetric[];
};

export type EvalBatch = {
  id: number;
  batch_key: string;
  pipeline_manifest_id: number;
  candidate_version: string;
  status: string;
  created_at: string;
  updated_at: string;
  item_count: number;
  incomplete_count: number;
  incomplete_rate: number | null;
  promotable: boolean;
};

export type EvalBatchGates = {
  precision_min: number;
  recall_min: number;
  min_positives: number;
  min_negatives: number;
  max_incomplete_rate: number;
  incomplete_rate: number;
  incomplete_pass: boolean;
  required_labels_pass: boolean;
  required_highlights_pass: boolean;
  batch_status_ok: boolean;
  promotable: boolean;
};

export type EvalBatchMetricsResponse = {
  batch: EvalBatch;
  labels: Array<LabelEvalMetric & Record<string, unknown>>;
  gates: EvalBatchGates;
  required_labels: string[];
};

export type PromoteBatchResponse = {
  batch_id: number;
  status: string;
  promoted: number;
  skipped_incomplete: number;
  gates: EvalBatchGates;
};

export type StudentPhaseTiming = {
  mean: number;
  median: number;
  p90: number;
  max: number;
};

export type StudentSpeedSummary = {
  clips: number;
  median_sec: number;
  mean_sec: number;
  p90_sec: number;
  max_sec: number;
  phases: Record<string, StudentPhaseTiming>;
};

export type StudentAgreementLabel = {
  label_key: string;
  agree: number;
  precision: number;
  recall: number;
  teacher_yes: number;
  student_yes: number;
  uncertain_rate: number;
  clips: number;
};

export type StudentExampleEvent = {
  event_index: number | null;
  status: string | null;
  event_kind: string | null;
  finish_timestamp: number | null;
  resolved_weapon: string | null;
  target_was_active: boolean | null;
  target_was_downed: boolean | null;
  damage_aim_state: string | null;
  visual_action_supported: boolean | null;
  summary: string | null;
};

export type StudentExampleClip = {
  clip_id: number;
  filename: string;
  duration_sec: number;
  total_sec: number;
  locator_timestamps: number[];
  locator_events: number;
  attribution_status: string | null;
  yes_labels: string[];
  labels: Record<string, string>;
  events: StudentExampleEvent[];
  video_url: string | null;
};

export type StudentReportsResponse = {
  available: boolean;
  model: string | null;
  speed: StudentSpeedSummary | null;
  agreement_labels: StudentAgreementLabel[];
  example_clips: StudentExampleClip[];
  speed_report_path: string | null;
  agreement_report_path: string | null;
  message: string | null;
};
