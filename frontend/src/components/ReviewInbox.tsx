import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchClips,
  fetchTrainingStatus,
  resetTasteProfile,
  scanFolder,
  sendFeedback,
  saveTasteProfile
} from "../api";
import type { Clip, TrainingStatus } from "../types";
import { ClipCard } from "./ClipCard";
import { ProcessClipsPanel } from "./ProcessClipsPanel";
import { TasteOnboarding } from "./TasteOnboarding";

type Status = "loading" | "ready" | "error";
type ClipFilter = "teacher" | "sample" | "demo";

function normalizedPath(clip: Clip): string {
  return clip.path.replace(/\\/g, "/").toLowerCase();
}

export function clipMatchesFilter(clip: Clip, filter: ClipFilter): boolean {
  const path = normalizedPath(clip);
  const isSample = path.includes("/sample-clips/");
  const isDemo = clip.source === "demo" || path.includes("/demo-video/");

  if (filter === "sample") return isSample;
  if (filter === "demo") return isDemo;
  return !isSample && !isDemo && Boolean(clip.teacher_provider && clip.teacher_provider !== "local");
}

function scoreLabel(score: number | null): string {
  if (score === null) {
    return "pending";
  }
  if (score >= 0.8) {
    return "strong";
  }
  if (score >= 0.65) {
    return "worth checking";
  }
  if (score >= 0.5) {
    return "mixed";
  }
  return "low";
}

export function ReviewInbox() {
  const [clips, setClips] = useState<Clip[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [visibleCount, setVisibleCount] = useState(30);
  const [clipFilter, setClipFilter] = useState<ClipFilter>("teacher");
  const feedbackQueueRef = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    Promise.all([fetchClips(), fetchTrainingStatus()])
      .then(([items, training]) => {
        setClips(items);
        setTrainingStatus(training);
        setStatus("ready");
      })
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : String(error));
        setStatus("error");
      });
  }, []);

  const filteredClips = useMemo(
    () => clips.filter((clip) => clipMatchesFilter(clip, clipFilter)),
    [clips, clipFilter]
  );
  const filterCounts = useMemo(
    () => ({
      teacher: clips.filter((clip) => clipMatchesFilter(clip, "teacher")).length,
      sample: clips.filter((clip) => clipMatchesFilter(clip, "sample")).length,
      demo: clips.filter((clip) => clipMatchesFilter(clip, "demo")).length
    }),
    [clips]
  );
  const reviewedCount = useMemo(
    () => filteredClips.filter((clip) => clip.feedback.some((item) => ["favorite", "keep", "skip", "boring", "delete"].includes(item))).length,
    [filteredClips]
  );
  const keeperCount = useMemo(
    () => filteredClips.filter((clip) => clip.feedback.some((item) => item === "favorite" || item === "keep")).length,
    [filteredClips]
  );
  const topScore = filteredClips[0]?.final_score ?? null;
  const displayedClips = filteredClips.slice(0, visibleCount);

  async function refreshTrainingStatus() {
    setTrainingStatus(await fetchTrainingStatus());
  }

  async function refreshClips() {
    setClips(await fetchClips());
  }

  function handleFeedback(clipId: number, action: string, label?: string) {
    const operation = feedbackQueueRef.current.then(async () => {
      try {
        const updatedClips = await sendFeedback(clipId, action, label);
        setClips(updatedClips);
        await refreshTrainingStatus();
      } catch (error) {
        setMessage(error instanceof Error ? error.message : String(error));
        throw error;
      }
    });
    feedbackQueueRef.current = operation.catch(() => undefined);
    return operation;
  }

  async function handleSaveTasteProfile(preferences: Record<string, number>) {
    setBusy(true);
    setMessage("Applying taste profile...");
    try {
      const response = await saveTasteProfile(preferences);
      setClips(response.clips);
      await refreshTrainingStatus();
      setMessage(`Saved ${response.saved} taste preferences. Rankings updated.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleProcessClips(path: string) {
    setBusy(true);
    setMessage("Processing clips with the local student model...");
    try {
      const response = await scanFolder(path, true);
      setClips(response.clips);
      if (path.replace(/\\/g, "/").toLowerCase().includes("/sample-clips")) {
        setClipFilter("sample");
      } else if (path.replace(/\\/g, "/").toLowerCase().includes("/demo-video")) {
        setClipFilter("demo");
      }
      setVisibleCount(30);
      await refreshTrainingStatus();
      setMessage(`Processed ${response.indexed} of ${response.total_found} clip(s) from ${path}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleResetTasteProfile() {
    setBusy(true);
    setMessage("Resetting taste profile...");
    try {
      const response = await resetTasteProfile();
      setClips(response.clips);
      await refreshTrainingStatus();
      setMessage("Taste profile reset. Your clip labels are still kept.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  if (status === "loading") {
    return (
      <main className="shell">
        <p className="loading">Loading Salience...</p>
      </main>
    );
  }

  if (status === "error") {
    return (
      <main className="shell">
        <section className="notice">
          <h1>Salience</h1>
          <p>{message}</p>
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Fast local Fortnite clip ranking</p>
          <h1>Salience</h1>
        </div>
        <button className="ghost" onClick={refreshClips}>
          Refresh
        </button>
      </header>

      <section className="summary" aria-label="Clip ranking summary">
        <div>
          <span className="summaryValue">{filteredClips.length}</span>
          <span className="summaryLabel">clips ranked</span>
        </div>
        <div>
          <span className="summaryValue">{topScore === null ? "-" : Math.round(topScore * 100)}</span>
          <span className="summaryLabel">top score - {scoreLabel(topScore)}</span>
        </div>
        <div>
          <span className="summaryValue">{keeperCount}</span>
          <span className="summaryLabel">{reviewedCount} reviewed</span>
        </div>
      </section>

      {message ? <p className="statusMessage">{message}</p> : null}

      <section className="workflow">
        <TasteOnboarding
          onSave={handleSaveTasteProfile}
          onReset={handleResetTasteProfile}
          busy={busy}
        />
        <ProcessClipsPanel onProcess={handleProcessClips} busy={busy} />
      </section>

      <section className="rankerPanel">
        <div>
          <h2>Personalization</h2>
          <p>
            Mark a few clips as favorite, keep, boring, or skip. The local ranker updates immediately and learns your taste.
          </p>
        </div>
        <div className="trainingStats">
          <span>{trainingStatus?.positive_count ?? 0} keep signals</span>
          <span>{trainingStatus?.negative_count ?? 0} skip signals</span>
          <span>
            Taste signal{" "}
            {trainingStatus?.personal_score_separation == null
              ? "pending"
              : trainingStatus.personal_score_separation.toFixed(2)}
          </span>
        </div>
      </section>

      <section className="clipFilters" aria-label="Filter ranked clips">
        <div>
          <h2>Clip set</h2>
          <p>Choose which ranked collection to review.</p>
        </div>
        <div className="filterButtons">
          {([
            ["teacher", "Teacher model"],
            ["sample", "Sample clips"],
            ["demo", "Demo clips"]
          ] as const).map(([value, label]) => (
            <button
              key={value}
              className={clipFilter === value ? "selected" : "ghost"}
              aria-pressed={clipFilter === value}
              onClick={() => {
                setClipFilter(value);
                setVisibleCount(30);
              }}
            >
              {label} <span>{filterCounts[value]}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="inbox" aria-label="Ranked clip inbox">
        {displayedClips.map((clip, index) => (
          <ClipCard key={clip.id} clip={clip} rank={index + 1} onFeedback={handleFeedback} />
        ))}
        {displayedClips.length === 0 ? <p className="noResults">No clips in this set yet.</p> : null}
        {visibleCount < filteredClips.length ? (
          <button className="loadMore" onClick={() => setVisibleCount((count) => Math.min(count + 30, filteredClips.length))}>
            Show 30 more
          </button>
        ) : null}
      </section>
    </main>
  );
}
