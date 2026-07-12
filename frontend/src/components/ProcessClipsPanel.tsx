import { useState } from "react";

const CLIP_SOURCES = [
  { label: "Sample clips (judges)", path: "/app/sample-clips" },
  { label: "Demo video clips (testing)", path: "/app/demo-video" }
] as const;

export function ProcessClipsPanel({
  onProcess,
  busy
}: {
  onProcess: (path: string) => Promise<void>;
  busy: boolean;
}) {
  const [selectedPath, setSelectedPath] = useState<string>(CLIP_SOURCES[0].path);

  return (
    <section className="processClipsPanel">
      <div>
        <h2>Process new clips</h2>
        <p>Run the local student model on a fixed set of unseen clips baked into this container.</p>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void onProcess(selectedPath);
        }}
      >
        <select
          value={selectedPath}
          onChange={(event) => setSelectedPath(event.target.value)}
          aria-label="Clip source"
        >
          {CLIP_SOURCES.map((source) => (
            <option key={source.path} value={source.path}>
              {source.label}
            </option>
          ))}
        </select>
        <button disabled={busy}>{busy ? "Processing..." : "Process clips"}</button>
      </form>
    </section>
  );
}
