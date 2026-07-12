import { useEffect, useMemo, useRef, useState } from "react";
import {
  createEvalBatch,
  fetchEvalBatches,
  fetchEvalBatchClips,
  fetchEvalBatchMetrics,
  fetchEvalClips,
  fetchEvalSummary,
  fetchStudentReports,
  promoteEvalBatch,
  resolveApiUrl,
  saveSnapshotLabelReview,
  saveHighlightReview,
  saveTeacherLabelReview
} from "../api";
import type {
  EvalBatch,
  EvalBatchGates,
  EvalClip,
  EvalMode,
  LabelEvalMetric,
  StudentReportsResponse
} from "../types";

type ExpectedValue = "yes" | "no" | "uncertain";
type Status = "loading" | "ready" | "error";

function formatMetric(value: number | null): string {
  return value === null ? "-" : `${Math.round(value * 100)}%`;
}

function labelName(label: string): string {
  return label.replaceAll("_", " ");
}

function predictionClass(value: string | undefined): string {
  if (value === "yes") {
    return "positive";
  }
  if (value === "no") {
    return "negative";
  }
  return "neutral";
}

function formatTimestamp(seconds: unknown): string {
  if (typeof seconds !== "number" || Number.isNaN(seconds)) {
    return "?:??";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function auditEventSummary(event: Record<string, unknown> | null | undefined): string | null {
  if (!event) {
    return null;
  }
  const summary = typeof event.summary === "string" ? event.summary : null;
  if (summary) {
    return summary;
  }
  const highlightType =
    typeof event.highlight_type === "string" ? event.highlight_type.replaceAll("_", " ") : "event";
  return `${formatTimestamp(event.timestamp_sec)} - ${highlightType}`;
}

function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(1)}s`;
}

function EvalClipCard({
  clip,
  selectedLabel,
  mode,
  disabled,
  onReview
  ,onHighlightReview
}: {
  clip: EvalClip;
  selectedLabel: string;
  mode: EvalMode;
  disabled: boolean;
  onReview: (clipId: number, labelKey: string, expectedValue: ExpectedValue) => Promise<void>;
  onHighlightReview: (snapshotId: number, aspect: string, expectedValue: ExpectedValue) => Promise<void>;
}) {
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [saving, setSaving] = useState<ExpectedValue | null>(null);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const livePrediction = clip.teacher_labels[selectedLabel] ?? "uncertain";
  const candidatePrediction = clip.candidate_labels[selectedLabel] ?? "unavailable";
  const review = clip.label_reviews[selectedLabel];
  const activeLabels = mode === "candidate" ? clip.candidate_labels : clip.teacher_labels;
  const activeEvidence = mode === "candidate" ? clip.candidate_evidence : clip.teacher_evidence;
  const positiveLabels = Object.entries(activeLabels)
    .filter(([, value]) => value === "yes")
    .map(([key]) => key);
  const audit = clip.candidate_event_audit;
  const primarySummary = auditEventSummary(audit?.primary_event);
  const highlightReviews = clip.highlight_reviews ?? {};
  const [highlightSaving, setHighlightSaving] = useState<string | null>(null);

  async function reviewAs(expectedValue: ExpectedValue) {
    setSaving(expectedValue);
    try {
      await onReview(clip.id, selectedLabel, expectedValue);
    } finally {
      setSaving(null);
    }
  }

  async function reviewHighlight(aspect: string, expectedValue: ExpectedValue) {
    if (!clip.prediction_snapshot_id) {
      return;
    }
    const key = `${aspect}:${expectedValue}`;
    setHighlightSaving(key);
    try {
      await onHighlightReview(clip.prediction_snapshot_id, aspect, expectedValue);
    } finally {
      setHighlightSaving(null);
    }
  }

  return (
    <article className="evalClip">
      <div className="evalVideo">
        {clip.video_url && videoLoaded ? <video src={resolveApiUrl(clip.video_url)} controls preload="metadata" /> : null}
        {clip.video_url && !videoLoaded ? (
          <button className="loadVideoButton" onClick={() => setVideoLoaded(true)}>
            Load video
          </button>
        ) : null}
        {!clip.video_url ? <span>No local video</span> : null}
      </div>
      <div className="evalBody">
        <div className="evalClipHeader">
          <div>
            <h2>{clip.filename}</h2>
            <p className="meta">
              Score {Math.round((clip.final_score ?? 0) * 100)} - VLM confidence{" "}
              {clip.teacher_confidence === null ? "unknown" : `${Math.round(clip.teacher_confidence * 100)}%`}
              {clip.candidate_status ? ` - Candidate ${clip.candidate_status}` : ""}
            </p>
          </div>
          <a href={`/#clip-${clip.id}`} className="ghostLink">
            Clip #{clip.id}
          </a>
        </div>

        <div className="evalPrediction">
          <span>New validation candidate</span>
          <strong className={predictionClass(candidatePrediction)}>{candidatePrediction}</strong>
          <span>Promoted live label</span>
          <strong className={predictionClass(livePrediction)}>{livePrediction}</strong>
          <span>Human expected</span>
          <strong className={predictionClass(review)}>{review ?? "unreviewed"}</strong>
        </div>

        {mode === "candidate" ? (
          <div className="evalAudit">
            {audit?.available ? (
              <>
                <p className="evalHighlightDescription">
                  {audit.highlight_description ?? "Structured audit present without description."}
                </p>
                {primarySummary ? (
                  <p className="evalAuditPrimary">
                    Primary: {primarySummary}
                    {audit.multi_kill ? " · multi-kill" : ""}
                  </p>
                ) : null}
                {audit.secondary_events.length ? (
                  <ul className="evalAuditSecondary">
                    {audit.secondary_events.map((event, index) => (
                      <li key={`${String(event.event_index ?? index)}-${index}`}>
                        Secondary: {auditEventSummary(event)}
                      </li>
                    ))}
                  </ul>
                ) : null}
                {audit.rejected_events.length ? (
                  <details className="evalRejected">
                    <summary>Rejected events ({audit.rejected_events.length})</summary>
                    <ul>
                      {audit.rejected_events.map((event, index) => (
                        <li key={`${String(event.event_index ?? index)}-rej-${index}`}>
                          {auditEventSummary(event)}
                        </li>
                      ))}
                    </ul>
                  </details>
                ) : null}
              </>
            ) : (
              <p className="evalAuditUnavailable">Structured audit unavailable</p>
            )}
            <details
              className="evalDiagnostics"
              open={diagnosticsOpen}
              onToggle={(event) => setDiagnosticsOpen((event.target as HTMLDetailsElement).open)}
            >
              <summary>Raw candidate evidence</summary>
              {activeEvidence.length ? (
                <p className="evalEvidence">{activeEvidence.slice(0, 6).join(" ")}</p>
              ) : (
                <p className="evalEvidence">No candidate evidence saved.</p>
              )}
            </details>
          </div>
        ) : activeEvidence.length ? (
          <p className="evalEvidence">{activeEvidence.slice(0, 3).join(" ")}</p>
        ) : (
          <p className="evalEvidence">No live teacher evidence saved.</p>
        )}

        <p className="evalTagCaption">
          Positive {mode === "candidate" ? "candidate" : "promoted live"} labels
        </p>
        <div className="evalTags" aria-label="Positive labels for active evaluation mode">
          {positiveLabels.map((label) => (
            <span key={label}>{labelName(label)}</span>
          ))}
        </div>

        <div className="evalReviewButtons">
          {(["yes", "no", "uncertain"] as ExpectedValue[]).map((value) => (
            <button
              key={value}
              className={review === value ? "selected" : ""}
              disabled={disabled || saving !== null}
              onClick={() => void reviewAs(value).catch(() => undefined)}
            >
              {saving === value ? "Saving..." : `Expected ${value}`}
            </button>
          ))}
        </div>
        {mode === "candidate" && clip.prediction_snapshot_id ? (
          <div className="evalHighlightChecks">
            {(["primary", "timestamp", "weapon", "target_state", "description"] as const).map(
              (aspect) => {
                const marked = highlightReviews[aspect];
                return (
                  <div key={aspect}>
                    <span>
                      {labelName(aspect)}
                      {marked ? ` · ${marked}` : " · unmarked"}
                    </span>
                    {(["yes", "no", "uncertain"] as ExpectedValue[]).map((value) => (
                      <button
                        key={value}
                        className={marked === value ? "selected" : ""}
                        disabled={disabled || highlightSaving !== null}
                        onClick={() => void reviewHighlight(aspect, value).catch(() => undefined)}
                      >
                        {highlightSaving === `${aspect}:${value}` ? "..." : value}
                      </button>
                    ))}
                  </div>
                );
              }
            )}
          </div>
        ) : null}
      </div>
    </article>
  );
}

export function EvalDashboard() {
  const [status, setStatus] = useState<Status>("loading");
  const [message, setMessage] = useState<string | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [clips, setClips] = useState<EvalClip[]>([]);
  const [metrics, setMetrics] = useState<LabelEvalMetric[]>([]);
  const [mode, setMode] = useState<EvalMode>("candidate");
  const [selectedLabel, setSelectedLabel] = useState("sniper_kill");
  const [limit, setLimit] = useState(30);
  const [clipLookup, setClipLookup] = useState("");
  const [batches, setBatches] = useState<EvalBatch[]>([]);
  const [selectedBatchId, setSelectedBatchId] = useState<number | null>(null);
  const [batchGates, setBatchGates] = useState<EvalBatchGates | null>(null);
  const [requiredLabels, setRequiredLabels] = useState<string[]>([]);
  const [batchKeyDraft, setBatchKeyDraft] = useState("");
  const [promoteOpen, setPromoteOpen] = useState(false);
  const [batchBusy, setBatchBusy] = useState(false);
  const [studentReports, setStudentReports] = useState<StudentReportsResponse | null>(null);
  const loadRequestRef = useRef(0);
  const reviewQueueRef = useRef<Promise<void>>(Promise.resolve());

  async function loadStudentReports() {
    try {
      setStudentReports(await fetchStudentReports());
    } catch {
      setStudentReports(null);
    }
  }

  async function loadBatches(preferredId?: number | null) {
    const nextBatches = await fetchEvalBatches();
    setBatches(nextBatches);
    const chosen =
      preferredId ??
      selectedBatchId ??
      null;
    setSelectedBatchId(chosen);
    if (chosen !== null) {
      const detail = await fetchEvalBatchMetrics(chosen);
      setBatchGates(detail.gates);
      setRequiredLabels(detail.required_labels);
    } else {
      setBatchGates(null);
      setRequiredLabels([]);
    }
  }

  async function load(
    nextLimit = limit,
    labelKey = selectedLabel,
    clipId?: number,
    nextMode: EvalMode = mode
  ) {
    const requestId = ++loadRequestRef.current;
    const pendingReviews = reviewQueueRef.current;
    setStatus("loading");
    try {
      await pendingReviews;
      const [clipPayload, summaryPayload] = await Promise.all([
        fetchEvalClips(nextLimit, labelKey, clipId, nextMode),
        fetchEvalSummary(nextMode)
      ]);
      if (requestId !== loadRequestRef.current) {
        return;
      }
      setLabels(clipPayload.labels);
      setClips(clipPayload.clips);
      setMetrics(summaryPayload.labels);
      if (!clipPayload.labels.includes(labelKey)) {
        setSelectedLabel(clipPayload.labels[0] ?? "sniper_kill");
      }
      try {
        await loadBatches();
      } catch {
        // Batch endpoints are additive; legacy eval still works without them.
      }
      await loadStudentReports();
      setStatus("ready");
    } catch (error) {
      if (requestId !== loadRequestRef.current) {
        return;
      }
      setMessage(error instanceof Error ? error.message : String(error));
      setStatus("error");
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedMetric = useMemo(
    () => metrics.find((metric) => metric.label_key === selectedLabel) ?? null,
    [metrics, selectedLabel]
  );
  const reviewedCount = useMemo(
    () => clips.filter((clip) => clip.label_reviews[selectedLabel]).length,
    [clips, selectedLabel]
  );

  function handleReview(clipId: number, labelKey: string, expectedValue: ExpectedValue) {
    const loadRequestId = loadRequestRef.current;
    const reviewMode = mode;
    const operation = reviewQueueRef.current.then(async () => {
      const clip = clips.find((item) => item.id === clipId);
      if (selectedBatchId !== null && clip?.prediction_snapshot_id) {
        const summary = await saveSnapshotLabelReview(
          clip.prediction_snapshot_id,
          labelKey,
          expectedValue
        );
        if (loadRequestId !== loadRequestRef.current) {
          return;
        }
        setMetrics(summary.labels);
        setBatchGates(summary.gates);
        setRequiredLabels(summary.required_labels);
      } else {
        const summary = await saveTeacherLabelReview(
          clipId,
          labelKey,
          expectedValue,
          reviewMode
        );
        if (loadRequestId !== loadRequestRef.current) {
          return;
        }
        setMetrics(summary.labels);
      }
      setClips((current) =>
        current.map((item) =>
          item.id === clipId
            ? { ...item, label_reviews: { ...item.label_reviews, [labelKey]: expectedValue } }
            : item
        )
      );
    });
    reviewQueueRef.current = operation.catch(() => undefined);
    return operation;
  }

  async function handleHighlightReview(
    snapshotId: number,
    aspect: string,
    expectedValue: ExpectedValue
  ) {
    await saveHighlightReview(snapshotId, aspect, expectedValue);
    setClips((current) =>
      current.map((clip) =>
        clip.prediction_snapshot_id === snapshotId
          ? {
              ...clip,
              highlight_reviews: {
                ...(clip.highlight_reviews ?? {}),
                [aspect]: expectedValue
              }
            }
          : clip
      )
    );
    if (selectedBatchId !== null) {
      const detail = await fetchEvalBatchMetrics(selectedBatchId);
      setBatchGates(detail.gates);
    }
  }

  function chooseLabel(label: string) {
    setSelectedLabel(label);
    if (selectedBatchId !== null && mode === "candidate") {
      void handleSelectBatch(selectedBatchId, label);
    } else {
      void load(limit, label);
    }
  }

  function chooseMode(nextMode: EvalMode) {
    setMode(nextMode);
    setMessage(null);
    void load(limit, selectedLabel, undefined, nextMode);
  }

  function findClip() {
    const clipId = Number(clipLookup.replace(/\D/g, ""));
    if (!Number.isInteger(clipId) || clipId <= 0) {
      setMessage("Enter a valid clip number.");
      return;
    }
    setMessage(null);
    if (selectedBatchId !== null && mode === "candidate") {
      void handleSelectBatch(selectedBatchId, selectedLabel, clipId);
    } else {
      void load(limit, selectedLabel, clipId);
    }
  }

  async function handleCreateBatch() {
    const key = batchKeyDraft.trim() || `batch-${new Date().toISOString().slice(0, 19)}`;
    setBatchBusy(true);
    setMessage(null);
    try {
      const created = await createEvalBatch(key);
      setBatchKeyDraft("");
      await loadBatches(created.batch.id);
      const frozen = await fetchEvalBatchClips(created.batch.id, limit, selectedLabel);
      setClips(frozen.clips);
      setMetrics(created.labels);
      setMessage(`Frozen batch ${created.batch.batch_key} with ${created.batch.item_count} clips.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBatchBusy(false);
    }
  }

  async function handleSelectBatch(
    batchId: number,
    labelKey = selectedLabel,
    clipId?: number
  ) {
    setSelectedBatchId(batchId);
    setBatchBusy(true);
    try {
      const detail = await fetchEvalBatchMetrics(batchId);
      const frozen = await fetchEvalBatchClips(batchId, limit, labelKey, clipId);
      setBatchGates(detail.gates);
      setRequiredLabels(detail.required_labels);
      setClips(frozen.clips);
      setMetrics(detail.labels);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBatchBusy(false);
    }
  }

  async function handlePromote() {
    if (selectedBatchId === null) {
      return;
    }
    setBatchBusy(true);
    setMessage(null);
    try {
      const result = await promoteEvalBatch(selectedBatchId);
      setPromoteOpen(false);
      await loadBatches(selectedBatchId);
      setMessage(
        `Promoted ${result.promoted} clips (skipped incomplete: ${result.skipped_incomplete}).`
      );
      void load(limit, selectedLabel, undefined, "live");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBatchBusy(false);
    }
  }

  if (status === "error") {
    return (
      <main className="shell">
        <section className="notice">
          <h1>Teacher eval</h1>
          <p>{message}</p>
        </section>
      </main>
    );
  }

  return (
    <main className="shell evalShell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Internal validation</p>
          <h1>Teacher Eval</h1>
        </div>
        <div className="topbarActions">
          <label className="clipLookup">
            <span>Find clip #</span>
            <input
              value={clipLookup}
              onChange={(event) => setClipLookup(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  findClip();
                }
              }}
              inputMode="numeric"
              placeholder="e.g. 492"
              aria-label="Find evaluation clip by number"
            />
          </label>
          <button className="ghost" onClick={findClip}>
            Find clip
          </button>
          <a className="ghostLink" href="/">
            Back to app
          </a>
        </div>
      </header>

      <section className="evalToolbar">
        <label>
          Review source
          <select value={mode} onChange={(event) => chooseMode(event.target.value as EvalMode)}>
            <option value="candidate">Validation candidates</option>
            <option value="live">Promoted live labels</option>
          </select>
        </label>
        <label>
          Label
          <select value={selectedLabel} onChange={(event) => chooseLabel(event.target.value)}>
            {labels.map((label) => (
              <option key={label} value={label}>
                {labelName(label)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Clips
          <input
            type="number"
            min="5"
            max="100"
            value={limit}
            onChange={(event) => setLimit(Number(event.target.value))}
          />
        </label>
        <button
          onClick={() => {
            if (selectedBatchId !== null) {
              void handleSelectBatch(selectedBatchId, selectedLabel);
            } else {
              void load(limit);
            }
          }}
        >
          {status === "loading" || batchBusy ? "Loading..." : "Refresh sample"}
        </button>
      </section>

      <section className="evalBatchBar">
        <label>
          Versioned batch
          <select
            value={selectedBatchId ?? ""}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isInteger(value) && value > 0) {
                void handleSelectBatch(value);
              } else {
                setSelectedBatchId(null);
                setBatchGates(null);
                void load(limit, selectedLabel, undefined, mode);
              }
            }}
          >
            <option value="">No frozen batch</option>
            {batches.map((batch) => (
              <option key={batch.id} value={batch.id}>
                {batch.batch_key} - {batch.status} - {batch.item_count} clips
              </option>
            ))}
          </select>
        </label>
        <label>
          New batch key
          <input
            value={batchKeyDraft}
            onChange={(event) => setBatchKeyDraft(event.target.value)}
            placeholder="e.g. event-summary-2026-07-11"
            aria-label="New evaluation batch key"
          />
        </label>
        <button disabled={batchBusy} onClick={() => void handleCreateBatch()}>
          Freeze candidates
        </button>
        <button
          disabled={batchBusy || !batchGates?.promotable}
          onClick={() => setPromoteOpen(true)}
        >
          Promote batch
        </button>
      </section>

      <section className="evalStudentBar" aria-label="Local student reports">
        <div className="evalStudentHeader">
          <div>
            <p className="eyebrow">Local student</p>
            <h2>{studentReports?.model ?? "Student reports"}</h2>
            <p>
              Offline speed + teacher-agreement reports. Separate from versioned teacher
              batches — does not write live labels.
            </p>
          </div>
          <button className="ghost" onClick={() => void loadStudentReports()}>
            Refresh student reports
          </button>
        </div>
        {!studentReports?.available ? (
          <p className="evalStudentEmpty">
            {studentReports?.message ?? "No student reports loaded yet."}
          </p>
        ) : (
          <>
            {studentReports.speed ? (
              <div className="evalStudentSpeed">
                <div>
                  <span className="summaryValue">
                    {formatSeconds(studentReports.speed.median_sec)}
                  </span>
                  <span className="summaryLabel">median / clip</span>
                </div>
                <div>
                  <span className="summaryValue">
                    {formatSeconds(studentReports.speed.mean_sec)}
                  </span>
                  <span className="summaryLabel">mean</span>
                </div>
                <div>
                  <span className="summaryValue">
                    {formatSeconds(studentReports.speed.p90_sec)}
                  </span>
                  <span className="summaryLabel">p90</span>
                </div>
                <div>
                  <span className="summaryValue">{studentReports.speed.clips}</span>
                  <span className="summaryLabel">bench clips</span>
                </div>
                <div>
                  <span className="summaryValue">
                    {formatSeconds(studentReports.speed.phases.event_extract_sec?.median)}
                  </span>
                  <span className="summaryLabel">event extract</span>
                </div>
                <div>
                  <span className="summaryValue">
                    {formatSeconds(studentReports.speed.phases.ocr_sec?.median)}
                  </span>
                  <span className="summaryLabel">ocr</span>
                </div>
                <div>
                  <span className="summaryValue">
                    {formatSeconds(studentReports.speed.phases.event_heads_sec?.median)}
                  </span>
                  <span className="summaryLabel">event heads</span>
                </div>
              </div>
            ) : null}
            {studentReports.agreement_labels.length > 0 ? (
              <div className="evalStudentAgreement">
                {studentReports.agreement_labels
                  .filter((label) => label.teacher_yes > 0 || label.student_yes > 0)
                  .slice(0, 12)
                  .map((label) => (
                    <button
                      key={label.label_key}
                      type="button"
                      className={label.label_key === selectedLabel ? "selected" : ""}
                      onClick={() => chooseLabel(label.label_key)}
                    >
                      <strong>{labelName(label.label_key)}</strong>
                      <span>
                        agree {formatMetric(label.agree)} · P {formatMetric(label.precision)} · R{" "}
                        {formatMetric(label.recall)}
                      </span>
                      <span>
                        teacher yes {label.teacher_yes} / student yes {label.student_yes}
                      </span>
                    </button>
                  ))}
              </div>
            ) : null}
            {studentReports.example_clips.length > 0 ? (
              <div className="evalStudentExamples">
                <h3>Example clip records</h3>
                {studentReports.example_clips.slice(0, 8).map((clip) => (
                  <article key={clip.clip_id} className="evalStudentExample">
                    <div className="evalStudentExampleVideo">
                      {clip.video_url ? (
                        <video src={resolveApiUrl(clip.video_url)} controls preload="metadata" />
                      ) : (
                        <span>No local video</span>
                      )}
                    </div>
                    <div className="evalStudentExampleBody">
                      <div className="evalClipHeader">
                        <strong>
                          #{clip.clip_id} {clip.filename}
                        </strong>
                        <span>
                          {formatSeconds(clip.total_sec)} · {clip.locator_events} events ·{" "}
                          {clip.attribution_status ?? "n/a"}
                        </span>
                      </div>
                      <p className="evalTagCaption">Student yes labels</p>
                      <div className="evalTags">
                        {clip.yes_labels.length > 0 ? (
                          clip.yes_labels.map((label) => (
                            <span key={label} className="tag positive">
                              {labelName(label)}
                            </span>
                          ))
                        ) : (
                          <span className="tag neutral">none</span>
                        )}
                      </div>
                      {clip.events.length > 0 ? (
                        <ul className="evalStudentEvents">
                          {clip.events.map((event, index) => (
                            <li key={`${clip.clip_id}-${index}`}>
                              {formatTimestamp(event.finish_timestamp)} ·{" "}
                              {event.event_kind ?? "unknown"} · {event.resolved_weapon ?? "unknown"} ·{" "}
                              {event.status ?? "n/a"}
                              {event.target_was_downed ? " · downed" : ""}
                              {event.target_was_active ? " · active" : ""}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="evalEvidence">No attributed events</p>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            ) : null}
          </>
        )}
      </section>

      {batchGates ? (
        <section className="evalGates">
          <div>
            <span className="summaryValue">{batchGates.promotable ? "PASS" : "BLOCKED"}</span>
            <span className="summaryLabel">promotion gates</span>
          </div>
          <div>
            <span className="summaryValue">{batchGates.required_labels_pass ? "yes" : "no"}</span>
            <span className="summaryLabel">required labels</span>
          </div>
          <div>
            <span className="summaryValue">
              {Math.round((batchGates.incomplete_rate ?? 0) * 100)}%
            </span>
            <span className="summaryLabel">incomplete</span>
          </div>
          <div>
            <span className="summaryValue">{requiredLabels.map(labelName).join(", ") || "-"}</span>
            <span className="summaryLabel">gate labels</span>
          </div>
        </section>
      ) : null}

      {message ? <section className="notice">{message}</section> : null}

      {promoteOpen ? (
        <section className="evalPromoteDialog notice" role="dialog" aria-label="Confirm promotion">
          <h2>Promote versioned batch?</h2>
          <p>
            This atomically copies complete frozen snapshots into live teacher assignments and
            rescores. Incomplete clips stay on their prior live assignment. Unversioned legacy
            candidates are not promotable.
          </p>
          <div className="evalReviewButtons">
            <button disabled={batchBusy} onClick={() => void handlePromote()}>
              {batchBusy ? "Promoting..." : "Confirm promote"}
            </button>
            <button className="ghost" disabled={batchBusy} onClick={() => setPromoteOpen(false)}>
              Cancel
            </button>
          </div>
        </section>
      ) : null}

      <section className="evalSummary">
        <div>
          <span className="summaryValue">{selectedMetric?.reviewed ?? 0}</span>
          <span className="summaryLabel">reviewed total</span>
        </div>
        <div>
          <span className="summaryValue">{reviewedCount}</span>
          <span className="summaryLabel">in current sample</span>
        </div>
        <div>
          <span className="summaryValue">{formatMetric(selectedMetric?.precision ?? null)}</span>
          <span className="summaryLabel">precision</span>
        </div>
        <div>
          <span className="summaryValue">{formatMetric(selectedMetric?.recall ?? null)}</span>
          <span className="summaryLabel">recall</span>
        </div>
        <div>
          <span className="summaryValue">{formatMetric(selectedMetric?.accuracy ?? null)}</span>
          <span className="summaryLabel">accuracy</span>
        </div>
      </section>

      <section className="evalMetricTable">
        {metrics
          .filter((metric) => metric.reviewed > 0)
          .map((metric) => (
            <button
              key={metric.label_key}
              className={metric.label_key === selectedLabel ? "selected" : ""}
              onClick={() => chooseLabel(metric.label_key)}
            >
              <strong>{labelName(metric.label_key)}</strong>
              <span>P {formatMetric(metric.precision)} / R {formatMetric(metric.recall)}</span>
            </button>
          ))}
      </section>

      <section className="evalList">
        {clips.map((clip) => (
          <EvalClipCard
            key={clip.id}
            clip={clip}
            selectedLabel={selectedLabel}
            mode={mode}
            disabled={status === "loading"}
            onReview={handleReview}
            onHighlightReview={handleHighlightReview}
          />
        ))}
        {status === "ready" && clips.length === 0 ? (
          <section className="notice">No clips are available for this evaluation mode and filter.</section>
        ) : null}
      </section>
    </main>
  );
}
