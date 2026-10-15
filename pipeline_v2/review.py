"""Local review and ranked inbox.
Explicit keep/skip decisions populate the v2 preference store."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pipeline_v2 import store
from pipeline_v2 import prompt
from pipeline_v2 import config
from pipeline_v2.http_media import serve_video

PAGE = """<!doctype html>
<meta charset="utf-8"><title>Salience review</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:15px/1.5 system-ui,sans-serif; background:#0d0f13; color:#e6e8eb;
         display:flex; flex-direction:column; align-items:center; gap:14px; padding:22px; }
  video { width:min(94vw,900px); border-radius:10px; background:#000; }
  #summary { max-width:min(94vw,900px); font-size:17px; }
  #meta { color:#94a0ae; font-size:13px; }
  #said { max-width:min(94vw,900px); color:#c9b98a; font-style:italic; }
  #progress { color:#94a0ae; font-size:13px; letter-spacing:.03em; }
  .keys { display:flex; gap:10px; }
  button { font:inherit; padding:9px 20px; border-radius:8px; border:1px solid #2a313a;
           background:#161b22; color:#e6e8eb; cursor:pointer; }
  button:hover { background:#1e242c; }
  kbd { background:#222933; border-radius:4px; padding:1px 6px; font-size:12px; }
  #done { font-size:19px; }
  #wrong.on { background:#5a1d1d; border-color:#a33; }
</style>
<div id="progress"></div>
<video id="v" controls autoplay muted></video>
<div id="summary"></div>
<div id="said"></div>
<div id="meta"></div>
<div class="keys">
  <button onclick="rate(0)">Skip &nbsp;<kbd>J</kbd></button>
  <button onclick="rate(1)">Keep &nbsp;<kbd>K</kbd></button>
  <button id="wrong" onclick="toggleWrong()">Description wrong &nbsp;<kbd>W</kbd></button>
</div>
<script>
let current = null;
let rating = false;
let wrong = false;
function toggleWrong() {
  wrong = !wrong;
  document.getElementById('wrong').classList.toggle('on', wrong);
}
async function load() {
  const r = await fetch('/api/next');
  const d = await r.json();
  if (d.done) {
    document.body.innerHTML = '<div id="done">All rated \\u2014 ' + d.rated +
      ' decisions. Run <code>taste --organic-only</code>.</div>';
    return;
  }
  current = d;
  wrong = false;
  document.getElementById('wrong').classList.remove('on');
  document.getElementById('progress').textContent =
    d.rated + ' rated \\u00b7 ' + d.remaining + ' left';
  const v = document.getElementById('v');
  v.src = '/media/' + d.sha;
  // Start a little before the located event: the interesting part is the
  // approach, and rating the same three seconds every time is what makes a
  // few hundred decisions bearable.
  v.currentTime = Math.max(0, (d.event_sec || 0) - 4);
  v.onloadedmetadata = () => { v.currentTime = Math.max(0, (d.event_sec || 0) - 4); };
  document.getElementById('summary').textContent = d.summary;
  document.getElementById('said').textContent = d.said ? 'Said: “' + d.said + '”' : '';
  document.getElementById('meta').textContent = d.name + (d.analysis_start_sec == null ? '' :
    ' · Description window: ' + d.analysis_start_sec.toFixed(1) + '–' + d.analysis_end_sec.toFixed(1) + 's');
}
async function rate(label) {
  if (!current || rating) return;
  rating = true;
  try {
    const r = await fetch('/api/rate', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sha: current.sha, label, wrong})});
    if (!r.ok) throw new Error('Decision was not saved. Please retry.');
    await load();
  } catch (error) {
    document.getElementById('meta').textContent = error.message;
  } finally { rating = false; }
}
addEventListener('keydown', e => {
  if (e.key === 'j' || e.key === 'J') rate(0);
  if (e.key === 'k' || e.key === 'K') rate(1);
  if (e.key === 'w' || e.key === 'W') toggleWrong();
});
load();
</script>
"""


# Which storage variant this server rates and ranks. Set once by `serve`. Reads
# otherwise resolve to the bare variant, which deliberately omits the `-window`
# suffix capture clips are stored under -- so without this, the only page that
# writes v2 labels could not show a single clip from the native recorder.
_VARIANT: str | None = None
# Folder whose clips are queued for rating, or None for all. Set by `serve`, so
# the session just played is not queued behind every older unrated window.
_ONLY: str | None = None


def _variant() -> str:
    return _VARIANT or prompt.active_variant(config.load_config())


# A re-cut window writes new content under the same filename; only the newest
# row for a path describes the file now on disk, so only it is worth rating.
_CURRENT = (
    " and c.described_at = (select max(c2.described_at) from clips c2"
    " where c2.path = c.path)"
)


def _said(conn: sqlite3.Connection, sha: str) -> str:
    """What the mic heard during the clip, if it was transcribed."""
    row = conn.execute("select text from transcripts where clip_sha = ?", (sha,)).fetchone()
    return row[0] if row else ""


def _within() -> str:
    return " and c.path like ?" if _ONLY else ""


def _within_args() -> tuple:
    return (_ONLY + "%",) if _ONLY else ()


def _next_clip(conn: sqlite3.Connection) -> dict:
    variant = _variant()
    rated = conn.execute("select count(*) from preferences").fetchone()[0]
    row = conn.execute(
        "select c.clip_sha, c.path, c.event_sec, c.facts_json, d.summary from clips c"
        " join descriptions d on d.clip_sha = c.clip_sha"
        " where c.clip_sha not in (select clip_sha from preferences)"
        " and d.prompt_ver = ?" + _CURRENT + _within() +
        " order by c.described_at, c.clip_sha limit 1",
        (variant, *_within_args()),
    ).fetchone()
    if row is None:
        return {"done": True, "rated": rated}
    remaining = conn.execute(
        "select count(*) from clips c join descriptions d on d.clip_sha = c.clip_sha"
        " where c.clip_sha not in (select clip_sha from preferences)"
        " and d.prompt_ver = ?" + _CURRENT + _within(),
        (variant, *_within_args()),
    ).fetchone()[0]
    return {
        "done": False,
        "sha": row["clip_sha"],
        "name": Path(row["path"]).name,
        "summary": row["summary"],
        "said": _said(conn, row["clip_sha"]),
        "event_sec": row["event_sec"],
        "analysis_start_sec": json.loads(row["facts_json"]).get("analysis_start_sec"),
        "analysis_end_sec": json.loads(row["facts_json"]).get("analysis_end_sec"),
        "rated": rated,
        "remaining": remaining,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/ranked":
            body = RANKED_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/ranked":
            self._json(_ranked_payload())
            return

        if self.path == "/api/next":
            conn = store.connect()
            self._json(_next_clip(conn))
            conn.close()
            return

        if self.path.startswith("/media/"):
            self._serve_media(self.path[len("/media/") :])
            return

        self.send_error(404)

    def _serve_media(self, sha: str) -> None:
        conn = store.connect()
        row = conn.execute(
            "select path from clips where clip_sha = ?", (sha,)
        ).fetchone()
        conn.close()
        if row is None:
            self.send_error(404)
            return
        path = Path(row["path"])
        if not path.exists():
            self.send_error(404)
            return

        serve_video(self, path)

    def do_POST(self) -> None:
        if self.path != "/api/rate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 4096:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or type(payload.get("label")) is not int:
                raise ValueError("label must be 0 or 1")
            if payload["label"] not in (0, 1) or not isinstance(
                payload.get("sha"), str
            ):
                raise ValueError("provide a clip sha and a 0/1 label")
        except (ValueError, TypeError):
            self._json({"error": "invalid preference"}, 400)
            return
        conn = store.connect()
        try:
            if not conn.execute(
                "select 1 from clips where clip_sha=?", (payload["sha"],)
            ).fetchone():
                self._json({"error": "clip not found"}, 404)
                return
            with conn:
                conn.execute(
                    "insert or replace into preferences (clip_sha, label, decided_at)"
                    " values (?, ?, strftime('%s','now'))",
                    (payload["sha"], payload["label"]),
                )
                # Rating again replaces the flag too, so un-flagging sticks.
                conn.execute(
                    "delete from description_flags where clip_sha=? and prompt_ver=?",
                    (payload["sha"], _variant()),
                )
                if payload.get("wrong") is True:
                    conn.execute(
                        "insert into description_flags (clip_sha, prompt_ver, decided_at)"
                        " values (?, ?, strftime('%s','now'))",
                        (payload["sha"], _variant()),
                    )
        finally:
            conn.close()
        self._json({"ok": True})


def serve(port: int = 8777, variant: str | None = None, only: Path | None = None) -> None:
    global _VARIANT, _ONLY
    _VARIANT = variant
    _ONLY = str(Path(only).resolve()) if only else None
    conn = store.connect()
    pending = _next_clip(conn)
    conn.close()
    print(f"variant:       {_variant()}")
    print(f"ranked inbox:  http://localhost:{port}/ranked")
    if not pending.get("done"):
        print(
            f"rate clips:    http://localhost:{port}/   "
            f"({pending['remaining']} waiting, J = skip, K = keep, W = description wrong)"
        )
    print("Ctrl+C to stop")
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        server.serve_forever()


RANKED_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Salience ranked</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:14px/1.55 system-ui,sans-serif; background:#0d0f13; color:#e6e8eb;
         display:grid; grid-template-columns: 420px 1fr; height:100vh; }
  #list { overflow-y:auto; border-right:1px solid #1e242c; }
  #head { padding:14px 16px; border-bottom:1px solid #1e242c; color:#94a0ae; font-size:13px; }
  .row { padding:11px 16px; border-bottom:1px solid #161b22; cursor:pointer; }
  .row:hover { background:#141920; }
  .row.on { background:#182029; border-left:3px solid #4c8dff; padding-left:13px; }
  .rk { color:#6f7c8c; font-variant-numeric:tabular-nums; margin-right:8px; }
  .sc { color:#4c8dff; font-variant-numeric:tabular-nums; float:right; }
  .sib { color:#7f8b99; font-size:12px; margin-top:4px; }
  #pane { padding:22px; display:flex; flex-direction:column; gap:14px; align-items:center;
          justify-content:center; }
  video { width:100%; max-width:900px; border-radius:10px; background:#000; }
  #sum { max-width:900px; font-size:16px; }
  #meta { max-width:900px; color:#94a0ae; font-size:12px; }
  #sibs { max-width:900px; display:flex; gap:8px; flex-wrap:wrap; }
  #sibs button { font:inherit; font-size:12px; padding:5px 11px; border-radius:7px;
                 border:1px solid #2a313a; background:#161b22; color:#c7d0da; cursor:pointer; }
</style>
<div id="list"><div id="head">loading…</div></div>
<div id="pane">
  <video id="v" controls></video>
  <div id="sum"></div>
  <div id="sibs"></div>
  <div id="meta"></div>
</div>
<script>
let fights = [];
function play(sha, name, summary, eventSec) {
  const v = document.getElementById('v');
  v.src = '/media/' + sha;
  v.onloadedmetadata = () => { v.currentTime = Math.max(0, (eventSec || 0) - 4); };
  v.play().catch(() => {});
  document.getElementById('sum').textContent = summary;
  document.getElementById('meta').textContent = name;
}
function show(i) {
  document.querySelectorAll('.row').forEach((r, j) => r.classList.toggle('on', i === j));
  const f = fights[i];
  play(f.sha, f.name, f.summary, f.event_sec);
  const sibs = document.getElementById('sibs');
  sibs.innerHTML = '';
  f.siblings.forEach(s => {
    const b = document.createElement('button');
    b.textContent = 'sibling: ' + s.name.replace(/^Fortnite /, '');
    b.onclick = () => play(s.sha, s.name, s.summary, s.event_sec);
    sibs.appendChild(b);
  });
}
fetch('/api/ranked').then(r => r.json()).then(d => {
  fights = d.fights;
  const list = document.getElementById('list');
  list.innerHTML = '<div id="head">' + d.fights.length + ' fights \u00b7 trained on ' +
    d.labels + ' labels \u00b7 ranked best first</div>';
  fights.forEach((f, i) => {
    const el = document.createElement('div');
    el.className = 'row';
    const score = document.createElement('span');
    score.className = 'sc'; score.textContent = f.score.toFixed(2);
    const rank = document.createElement('span');
    rank.className = 'rk'; rank.textContent = i + 1;
    el.append(score, rank, document.createTextNode(f.summary));
    if (f.siblings.length) {
      const siblings = document.createElement('div');
      siblings.className = 'sib';
      siblings.textContent = '+' + f.siblings.length + ' more clip(s) of this fight';
      el.appendChild(siblings);
    }
    el.onclick = () => show(i);
    list.appendChild(el);
  });
  if (fights.length) show(0);
});
</script>
"""


def _ranked_payload() -> dict:
    from pipeline_v2 import ranking

    conn = store.connect()
    fights, labels = ranking.ranked_fights(conn, variant=_variant())
    conn.close()

    def brief(clip) -> dict:
        return {
            "sha": clip.clip_sha,
            "name": clip.path.name,
            "summary": clip.summary,
            "event_sec": clip.event_sec,
        }

    return {
        "labels": labels,
        "fights": [
            {
                "score": fight.score,
                **brief(fight.best),
                "siblings": [brief(s) for s in fight.siblings],
            }
            for fight in fights
        ],
    }
