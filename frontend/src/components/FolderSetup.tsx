import { FormEvent, useState } from "react";

export function FolderSetup({
  onScan,
  message
}: {
  onScan: (path: string) => Promise<void>;
  message: string | null;
}) {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!path.trim()) {
      return;
    }
    setBusy(true);
    try {
      await onScan(path.trim());
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="folderPanel">
      <div>
        <h2>Scan clips</h2>
        <p>Paste the folder where your recorder saves Fortnite clips. Scanning is local and fast.</p>
      </div>
      <form onSubmit={submit}>
        <input
          value={path}
          onChange={(event) => setPath(event.target.value)}
          placeholder="C:\\Users\\YourName\\Videos\\Fortnite"
          aria-label="Clips folder path"
        />
        <button disabled={busy || !path.trim()}>{busy ? "Scanning..." : "Scan"}</button>
      </form>
      {message ? <p className="statusMessage">{message}</p> : null}
    </section>
  );
}
