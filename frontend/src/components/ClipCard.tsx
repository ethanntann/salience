import { useState } from "react";
import { resolveApiUrl } from "../api";
import type { Clip } from "../types";

function formatPercent(value: number | null): string {
  if (value === null) {
    return "Pending";
  }
  return `${Math.round(value * 100)}`;
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) {
    return "Unknown length";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

export function ClipCard({
  clip,
  rank,
  onFeedback
}: {
  clip: Clip;
  rank: number;
  onFeedback: (clipId: number, action: string, label?: string) => Promise<void>;
}) {
  const [tag, setTag] = useState("");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);

  async function act(action: string, label?: string) {
    setBusyAction(action);
    try {
      await onFeedback(clip.id, action, label);
      if (action === "tag") {
        setTag("");
      }
    } finally {
      setBusyAction(null);
    }
  }

  function runAction(action: string, label?: string) {
    void act(action, label).catch(() => undefined);
  }

  return (
    <article className="clipCard">
      <div className={`thumb ${clip.video_url ? "thumbVideo" : `thumb-${clip.thumbnail_variant}`}`} aria-label={`${clip.filename} preview`}>
        {clip.video_url && videoLoaded ? <video src={resolveApiUrl(clip.video_url)} controls preload="metadata" /> : null}
        {clip.video_url && !videoLoaded ? (
          <button className="loadVideoButton" onClick={() => setVideoLoaded(true)}>
            Load video
          </button>
        ) : null}
        <span className="rank">#{rank}</span>
        <span className="hud hudTop">{clip.teacher_provider ? "VLM" : "LOCAL"}</span>
        <span className="hud hudBottom">{formatDuration(clip.duration_sec)}</span>
      </div>

      <div className="clipBody">
        <div className="clipHeader">
          <div>
            <h2>{clip.filename}</h2>
            <p className="meta">
              {clip.width && clip.height ? `${clip.width}x${clip.height}` : "Demo metadata"}
              {clip.fps ? ` · ${Math.round(clip.fps)} fps` : ""}
            </p>
          </div>
          <div className="scoreBlock">
            <span className="score">{formatPercent(clip.final_score)}</span>
            <span className="scoreLabel">score</span>
          </div>
        </div>

        <div className="meter" aria-label={`Score ${formatPercent(clip.final_score)}`}>
          <span style={{ width: `${Math.round((clip.final_score ?? 0) * 100)}%` }} />
        </div>

        <p className="highlightDescription">
          {clip.highlight_description ?? "Verified highlight description unavailable"}
        </p>
        {clip.explanation ? <p className="explanation secondaryExplanation">{clip.explanation}</p> : null}

        <div className="teacherBox">
          <strong>
            {clip.teacher_provider ? `VLM: ${clip.teacher_provider}` : "VLM: pending"}
            {clip.teacher_confidence !== null ? ` · ${Math.round(clip.teacher_confidence * 100)}%` : ""}
          </strong>
          {clip.teacher_evidence.length ? <p>{clip.teacher_evidence.slice(0, 2).join("; ")}</p> : null}
        </div>

        <div className="tags">
          {clip.tags.map((item) => (
            <span key={item}>{item.replaceAll("_", " ")}</span>
          ))}
          {clip.feedback.slice(0, 3).map((item) => (
            <span className="feedbackTag" key={item}>
              {item.replace("tag:", "+").replaceAll("_", " ")}
            </span>
          ))}
        </div>

        <div className="actions" aria-label={`Feedback for ${clip.filename}`}>
          <button disabled={busyAction !== null} onClick={() => runAction("favorite")}>
            Favorite
          </button>
          <button disabled={busyAction !== null} onClick={() => runAction("keep")}>
            Keep
          </button>
          <button disabled={busyAction !== null} onClick={() => runAction("boring")}>
            Boring
          </button>
          <button disabled={busyAction !== null} onClick={() => runAction("skip")}>
            Skip
          </button>
          <button className="danger" disabled={busyAction !== null} onClick={() => runAction("delete")}>
            Delete
          </button>
        </div>

        <form
          className="tagForm"
          onSubmit={(event) => {
            event.preventDefault();
            const cleanTag = tag.trim();
            if (cleanTag) {
              runAction("tag", cleanTag);
            }
          }}
        >
          <input
            value={tag}
            onChange={(event) => setTag(event.target.value)}
            placeholder="Add taste tag"
            aria-label="Add taste tag"
          />
          <button disabled={!tag.trim() || busyAction !== null}>Add</button>
        </form>
      </div>
    </article>
  );
}
