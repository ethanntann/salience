import type {
  AiStatus,
  Clip,
  EvalBatch,
  EvalBatchMetricsResponse,
  EvalClipResponse,
  EvalSummaryResponse,
  EnrichResponse,
  ExportResponse,
  ImportLabelsResponse,
  ImportLikedFolderResponse,
  PromoteBatchResponse,
  ScanResponse,
  StudentReportsResponse,
  TeacherRunStatus,
  TasteProfileResponse,
  TrainingStatus
} from "./types";

const devBase = window.location.port === "5173" ? "http://localhost:8000" : "";
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? devBase).replace(/\/+$/, "");

export function resolveApiUrl(url: string): string {
  if (!API_BASE || !url.startsWith("/") || url.startsWith("//")) {
    return url;
  }
  return `${API_BASE}${url}`;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchClips(): Promise<Clip[]> {
  return parseJson<Clip[]>(await fetch(`${API_BASE}/clips`));
}

export async function fetchAiStatus(): Promise<AiStatus> {
  return parseJson<AiStatus>(await fetch(`${API_BASE}/ai/status`));
}

export async function scanFolder(path: string, enrich = false): Promise<ScanResponse> {
  return parseJson<ScanResponse>(
    await fetch(`${API_BASE}/folders/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, enrich })
    })
  );
}

export async function sendFeedback(clipId: number, action: string, label?: string): Promise<Clip[]> {
  return parseJson<Clip[]>(
    await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_id: clipId, action, label: label ?? null })
    })
  );
}

export async function resetDemo(): Promise<Clip[]> {
  return parseJson<Clip[]>(
    await fetch(`${API_BASE}/demo/reset`, {
      method: "POST"
    })
  );
}

export async function enrichClips(clipId?: number, limit = 25): Promise<EnrichResponse> {
  return parseJson<EnrichResponse>(
    await fetch(`${API_BASE}/ai/enrich`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_id: clipId ?? null, limit })
    })
  );
}

export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  return parseJson<TrainingStatus>(await fetch(`${API_BASE}/training/status`));
}

export async function startTeacherRun(limit = 10000): Promise<TeacherRunStatus> {
  return parseJson<TeacherRunStatus>(
    await fetch(`${API_BASE}/ai/enrich/background`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit })
    })
  );
}

export async function fetchTeacherRunStatus(): Promise<TeacherRunStatus> {
  return parseJson<TeacherRunStatus>(await fetch(`${API_BASE}/ai/enrich/background`));
}

export async function importLabels(path: string): Promise<ImportLabelsResponse> {
  return parseJson<ImportLabelsResponse>(
    await fetch(`${API_BASE}/training/import-labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    })
  );
}

export async function exportClips(destination: string, mode: "keepers" | "top", limit: number): Promise<ExportResponse> {
  return parseJson<ExportResponse>(
    await fetch(`${API_BASE}/clips/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ destination, mode, limit })
    })
  );
}

export async function saveTasteProfile(preferences: Record<string, number>): Promise<TasteProfileResponse> {
  return parseJson<TasteProfileResponse>(
    await fetch(`${API_BASE}/taste/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preferences })
    })
  );
}

export async function resetTasteProfile(): Promise<TasteProfileResponse> {
  return parseJson<TasteProfileResponse>(
    await fetch(`${API_BASE}/taste/reset`, {
      method: "POST"
    })
  );
}

export async function importLikedFolder(path: string): Promise<ImportLikedFolderResponse> {
  return parseJson<ImportLikedFolderResponse>(
    await fetch(`${API_BASE}/taste/import-liked-folder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    })
  );
}

export async function fetchEvalClips(
  limit = 30,
  labelKey?: string,
  clipId?: number,
  mode: "candidate" | "live" = "candidate"
): Promise<EvalClipResponse> {
  const params = new URLSearchParams({ limit: String(limit), mode });
  if (labelKey) {
    params.set("label_key", labelKey);
  }
  if (clipId) {
    params.set("clip_id", String(clipId));
  }
  return parseJson<EvalClipResponse>(
    await fetch(`${API_BASE}/eval/teacher-clips?${params.toString()}`, { cache: "no-store" })
  );
}

export async function fetchEvalSummary(mode: "candidate" | "live" = "candidate"): Promise<EvalSummaryResponse> {
  return parseJson<EvalSummaryResponse>(
    await fetch(
      `${API_BASE}${mode === "candidate" ? "/eval/event-validation-summary" : "/eval/teacher-summary"}`,
      { cache: "no-store" }
    )
  );
}

export async function saveTeacherLabelReview(
  clipId: number,
  labelKey: string,
  expectedValue: "yes" | "no" | "uncertain",
  mode: "candidate" | "live" = "candidate"
): Promise<EvalSummaryResponse> {
  return parseJson<EvalSummaryResponse>(
    await fetch(`${API_BASE}/eval/teacher-review?mode=${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_id: clipId, label_key: labelKey, expected_value: expectedValue })
    })
  );
}

export async function fetchEvalBatches(): Promise<EvalBatch[]> {
  const payload = await parseJson<{ batches: EvalBatch[] }>(
    await fetch(`${API_BASE}/eval/batches`, { cache: "no-store" })
  );
  return payload.batches;
}

export async function createEvalBatch(batchKey: string): Promise<EvalBatchMetricsResponse> {
  return parseJson<EvalBatchMetricsResponse>(
    await fetch(`${API_BASE}/eval/batches`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch_key: batchKey })
    })
  );
}

export async function fetchEvalBatchMetrics(batchId: number): Promise<EvalBatchMetricsResponse> {
  return parseJson<EvalBatchMetricsResponse>(
    await fetch(`${API_BASE}/eval/batches/${batchId}`, { cache: "no-store" })
  );
}

export async function fetchEvalBatchClips(
  batchId: number,
  limit = 100,
  labelKey?: string,
  clipId?: number
): Promise<EvalClipResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (labelKey) params.set("label_key", labelKey);
  if (clipId) params.set("clip_id", String(clipId));
  return parseJson<EvalClipResponse>(
    await fetch(`${API_BASE}/eval/batches/${batchId}/clips?${params.toString()}`, {
      cache: "no-store"
    })
  );
}

export async function promoteEvalBatch(batchId: number): Promise<PromoteBatchResponse> {
  return parseJson<PromoteBatchResponse>(
    await fetch(`${API_BASE}/eval/batches/${batchId}/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: true })
    })
  );
}

export async function fetchStudentReports(): Promise<StudentReportsResponse> {
  return parseJson<StudentReportsResponse>(
    await fetch(`${API_BASE}/eval/student-reports`, { cache: "no-store" })
  );
}

export async function saveSnapshotLabelReview(
  predictionSnapshotId: number,
  labelKey: string,
  expectedValue: "yes" | "no" | "uncertain"
): Promise<EvalBatchMetricsResponse> {
  return parseJson<EvalBatchMetricsResponse>(
    await fetch(`${API_BASE}/eval/snapshot-review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_snapshot_id: predictionSnapshotId,
        label_key: labelKey,
        expected_value: expectedValue
      })
    })
  );
}

export async function saveHighlightReview(
  predictionSnapshotId: number,
  aspect: string,
  expectedValue: "yes" | "no" | "uncertain"
): Promise<{ saved: boolean; aspect: string }> {
  return parseJson<{ saved: boolean; aspect: string }>(
    await fetch(`${API_BASE}/eval/highlight-review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_snapshot_id: predictionSnapshotId,
        aspect,
        expected_value: expectedValue
      })
    })
  );
}
