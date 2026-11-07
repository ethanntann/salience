"use strict";

// The desktop app: four views on one local server (`server.py`).
//   #/                      library: every session's highlights
//   #/session/<id>[/<file>] one session, one highlight playing
//   #/rate                  rating, every session's windows in one queue
//   #/settings              the recording settings that are the player's

const page = document.getElementById("page");
const HEART = '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" role="img" aria-label="Loved"><path d="M12 21s-7.5-4.6-9.5-9.2C1.1 8.4 3.2 5 6.6 5c2 0 3.4 1.1 4.2 2.4h2.4C14 6.1 15.4 5 17.4 5c3.4 0 5.5 3.4 4.1 6.8C19.5 16.4 12 21 12 21z"></path></svg>';
const TICK = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M5 12l4 4L19 6"></path></svg>';
const BIN = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"></path></svg>';

const PLAY = '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"></path></svg>';
const PAUSE = '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"></rect><rect x="14" y="5" width="4" height="14" rx="1"></rect></svg>';
const SOUND = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9H4z"></path><path d="M16 9a4 4 0 0 1 0 6"></path></svg>';
const MUTED = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9H4z"></path><path d="M16 9l5 6M21 9l-5 6"></path></svg>';
const FULL = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"></path></svg>';
const clockOf = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const tenths = (t) => { const r = Math.round(t * 10) / 10; return `${clockOf(r)}.${Math.round((r % 1) * 10) % 10}`; };

let libraryCache = null;
let leaving = null; // the current view's clean-up (key handlers)

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const enc = encodeURIComponent;

// Every request to the app carries its launch secret (`server.SECRET_HEADER`):
// the server refuses any that does not, so other websites cannot use it.
const nativeFetch = window.fetch.bind(window);
window.fetch = (input, init = {}) => nativeFetch(input, {
  ...init, headers: { ...(init.headers || {}), "X-Salience": window.SALIENCE_SECRET || "" } });

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function library(fresh = false) {
  if (!libraryCache || fresh) libraryCache = await api("/api/library");
  return libraryCache;
}

function when(iso) {
  const date = new Date(iso);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const day = new Date(date); day.setHours(0, 0, 0, 0);
  const days = Math.round((today - day) / 86400000);
  const time = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (days === 0) return `Today, ${time}`;
  if (days === 1) return `Yesterday, ${time}`;
  const name = date.toLocaleDateString([], days < 7
    ? { weekday: "long" } : { weekday: "long", month: "short", day: "numeric" });
  return `${name}, ${time}`;
}

function facts(s) {
  const bits = [];
  if (s.minutes) bits.push(`${s.minutes} min`);
  if (s.matches) bits.push(`${s.matches} match${s.matches === 1 ? "" : "es"}`);
  bits.push(`${s.highlights.length} highlight${s.highlights.length === 1 ? "" : "s"}`);
  return bits.join(" · ");
}

const info = (h) => [h.weapon, h.at].filter(Boolean).join(" · ");
const thumb = (s, h) => `/thumb/${enc(s.id)}/${enc(h.file)}`;

function card(s, h) {
  return `<a class="card" href="#/session/${enc(s.id)}/${enc(h.file)}">
    <div class="thumb">
      <img src="${thumb(s, h)}" alt="" loading="lazy">
      <span class="tag rank">${esc(h.rank)}</span>
      ${h.length ? `<span class="tag length">${esc(h.length)}</span>` : ""}
      ${h.added ? '<span class="tag added">From rating</span>' : h.laughed ? '<span class="tag added">Laughter</span>' : ""}
      ${h.loved ? `<span class="heart">${HEART}</span>` : ""}
    </div>
    <div>
      <div class="name">${esc(h.title)}</div>
      <div class="info">${esc(info(h))}</div>
      ${h.said ? `<div class="said">“${esc(h.said)}”</div>` : ""}
    </div>
  </a>`;
}

function setNav(which) {
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === which));
}

async function refreshChrome() {
  try {
    const [lib, set] = await Promise.all([library(), api("/api/settings")]);
    const left = lib.sessions.reduce((n, s) => n + s.unrated, 0);
    const badge = document.getElementById("rate-badge");
    badge.hidden = left === 0;
    badge.textContent = `${left} left`;
    document.getElementById("status-quality").textContent = qualityLine(set);
  } catch (error) { /* the views say what failed */ }
}

function qualityLine(set) {
  const res = set.resolutions[set.resolution] || "Full size";
  return `${res.replace(" (your screen's)", "")} · ${set.fps} fps · ${set.mbps} Mbps · game + mic`;
}

// The background recorder (the tray process), in the sidebar

let recorder = null;
let lastState = null;

async function postJson(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  return { status: response.status, data: await response.json().catch(() => ({})) };
}

// Nothing on this page changes on a timer. A ticking recording clock and a
// blinking dot redrew the window every second behind a fullscreen game, and
// each redraw took the game out of independent flip and back: it froze once
// a second while this window was open (2026-11-06). The window's focus is no
// guide: behind the game it still had it, and still drew every second.
const still = () => document.hidden || !document.hasFocus();
function setText(id, text) {
  const node = document.getElementById(id);
  if (node.textContent !== text) node.textContent = text;
}
function setClass(id, name) {
  const node = document.getElementById(id);
  if (node.className !== name) node.className = name;
}
function quieten() { document.body.classList.toggle("still", still()); paintRecorder(); }
window.addEventListener("focus", quieten);
window.addEventListener("blur", quieten);
document.addEventListener("visibilitychange", quieten);

function paintRecorder() {
  const r = recorder;
  if (!r) return;
  const on = r.autoclip;
  const sw = document.getElementById("autoclip");
  if (sw.getAttribute("aria-checked") !== String(on)) sw.setAttribute("aria-checked", String(on));
  let dot = "", state, detail;
  if (!on && r.running) { state = "Turning off…"; detail = "Finishing this session first."; }
  else if (!on) { state = "Auto-clip is off"; detail = "Nothing is recorded. Turn it on to clip your games."; }
  else if (!r.has_key) { state = "Needs your key"; detail = "Add your Fireworks key in Settings to start."; }
  else if (!r.running && r.state === "closed") { state = "Recorder closed"; detail = "You quit it from the tray. Click the switch to start it again."; }
  else if (!r.running) { state = "Starting…"; detail = "The recorder is starting in the background."; }
  else if (r.state === "recording") {
    dot = "recording";
    const since = r.started ? new Date(r.started * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
    state = since ? `Recording since ${since}` : "Recording";
    detail = "Fortnite is open. Highlights are made when you finish.";
  } else if (r.state === "highlights") {
    dot = "highlights"; state = "Making highlights…"; detail = "They open here when ready.";
  } else { dot = "ready"; state = "Ready"; detail = "Starts recording on its own when Fortnite opens. Nothing to press."; }
  setClass("rec-dot", `dot ${dot}`);
  setText("rec-state", state);
  setText("rec-detail", detail);
}

async function pollRecorder() {
  try { recorder = await api("/api/status"); } catch (error) { return; }
  if (recorder.goto) {  // the tray asked for a session (it just finished)
    libraryCache = null;
    if (location.hash === recorder.goto) route(); else location.hash = recorder.goto;
  }
  // A session just finished: its highlights are new.
  if (lastState === "highlights" && recorder.state !== "highlights") {
    libraryCache = null;
    if (!location.hash.startsWith("#/rate") && !location.hash.startsWith("#/settings")) route();
    refreshChrome();
  }
  lastState = recorder.state;
  paintRecorder();
}

function matchTime(seconds) {
  const h = Math.floor(seconds / 3600), m = Math.round((seconds % 3600) / 60);
  return h ? `${h} h${m ? ` ${m} min` : ""}` : `${m} min`;
}

// Clips the player likes. Asked about only while an import is under way,
// and the box is redrawn only when what it says changes.
let importState = null;
let importShown = "";
function importText(job) {
  const n = job.windows, files = `${job.files} file${job.files === 1 ? "" : "s"}`;
  const moments = (k) => `${k} moment${k === 1 ? "" : "s"}`;
  switch (job.stage.replace(/-waiting$/, "")) {
    case "cutting": return job.stage.endsWith("waiting") ? "Waiting for Fortnite to close before getting your clips ready."
      : `Getting ${files} ready…`;
    case "ready": return `Found ${moments(n)} in ${files}. Describing ${n === 1 ? "it" : "them"} takes ${n} call${n === 1 ? "" : "s"} to your Fireworks key.`;
    case "describing": return job.stage.endsWith("waiting") ? "Waiting for Fortnite to close before describing."
      : `Describing ${moments(n)}…`;
    case "done": return `Added ${moments(job.liked)}. They now count toward your taste.`;
    case "empty": return "No moments long enough in those files. Each needs at least 3 seconds without an edit.";
    case "error": return `That didn't work: ${job.error}`;
    default: return job.total ? `${moments(job.total)} from your clips count toward your taste.` : "No clips added yet.";
  }
}
function paintImport(page) {
  const box = page.querySelector("#import-box");
  if (!box) return;
  if (!importState) { api("/api/import").then((job) => { importState = job; paintImport(page); }).catch(() => {}); return; }
  const job = importState, stage = job.stage.replace(/-waiting$/, "");
  const shown = JSON.stringify([job.stage, job.windows, job.liked, job.total, job.error]);
  if (shown !== importShown || !box.firstChild) {
    importShown = shown;
    const buttons = stage === "ready"
      ? `<button class="button primary" id="import-go">Describe</button><button class="button" id="import-cancel">Cancel</button>`
      : stage === "cutting" || stage === "describing" ? ""
      : `<button class="button primary" id="import-add">Add clips…</button>`;
    box.innerHTML = `<div class="line"><div><span id="import-note">${esc(importText(job))}</span></div>
      <div style="display:flex;gap:8px">${buttons}</div></div>`;
    const on = (id, fn) => { const b = box.querySelector(id); if (b) b.addEventListener("click", fn); };
    on("#import-add", async () => { importState = await api("/api/import/choose", {}); paintImport(page); followImport(page); });
    on("#import-cancel", async () => { importState = await api("/api/import/discard", {}); paintImport(page); });
    on("#import-go", async () => {
      const { status, data } = await postJson("/api/import/describe", {});
      importState = data.stage ? data : importState;
      if (status === 409 && data.need_key) { setText("import-note", "Add your Fireworks key below first."); return; }
      paintImport(page); followImport(page);
    });
  }
}
function followImport(page) {
  const stage = (importState && importState.stage || "").replace(/-waiting$/, "");
  if (stage !== "cutting" && stage !== "describing") return;
  setTimeout(async () => {
    if (!document.body.contains(page.querySelector("#import-box"))) return;
    try { importState = await api("/api/import"); } catch (error) { return; }
    paintImport(page); followImport(page);
  }, 2000);
}

async function setAutoclip(on) {
  const { status, data } = await postJson("/api/autoclip", { on });
  recorder = data.autoclip === undefined ? recorder : data;
  paintRecorder();
  if (status === 409 && data.need_key) {
    location.hash = "#/settings";
    setTimeout(() => {
      const key = document.getElementById("key");
      if (key) { key.focus(); document.getElementById("key-note").textContent = "Add your key, then turn Auto-clip on."; }
    }, 300);
  }
  if (location.hash.startsWith("#/settings")) route();
}

document.getElementById("autoclip").addEventListener("click", () => {
  // On but closed (quit from the tray): the switch starts it again rather than turning auto-clip off.
  if (recorder && recorder.autoclip && !recorder.running && recorder.state === "closed") return setAutoclip(true);
  setAutoclip(!(recorder && recorder.autoclip));
});
setInterval(pollRecorder, 2000);
pollRecorder();
quieten();

// The sidebar: its theme and whether it is collapsed, kept on this PC

const store = {
  get: (key) => { try { return localStorage.getItem(key); } catch (error) { return null; } },
  set: (key, value) => {
    try { localStorage.setItem(key, value); } catch (error) { /* blocked: the app's copy still keeps it */ }
    // And on disk, through the app: this page's own storage is gone with the port at the next launch.
    if (!window.SALIENCE_PREVIEW && (window.SALIENCE_LOOK || {})[key] !== value) {
      (window.SALIENCE_LOOK = window.SALIENCE_LOOK || {})[key] = value;
      fetch("/api/appearance", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, value }) }).catch(() => {});
    }
  },
};

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;  // auto: Windows decides
  document.querySelectorAll(".theme [data-theme]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.theme === theme)));
  document.getElementById("theme-cycle").setAttribute("aria-label", `Theme: ${theme}. Change it`);
  store.set("salience.theme", theme);
}
document.querySelectorAll(".theme [data-theme]").forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.theme)));
document.getElementById("theme-cycle").addEventListener("click", () => {
  const order = ["auto", "light", "dark"];
  applyTheme(order[(order.indexOf(store.get("salience.theme") || "auto") + 1) % order.length]);
});
applyTheme(store.get("salience.theme") || "auto");

// Palettes and patterns (themes.py writes static/themes.json): chosen in Settings.
function applyPalette(name) {
  document.documentElement.dataset.palette = name;
  store.set("salience.palette", name);
}
function applyPattern(name) {
  document.documentElement.dataset.pattern = name;
  store.set("salience.pattern", name);
}
let themesCache = null;
const themes = async () => themesCache || (themesCache = await (await fetch("/static/themes.json")).json());

function applyRail(collapsed) {
  if (collapsed) document.documentElement.dataset.rail = "collapsed";
  else delete document.documentElement.dataset.rail;
  const button = document.getElementById("collapse");
  button.setAttribute("aria-expanded", String(!collapsed));
  button.setAttribute("aria-label", collapsed ? "Expand the sidebar" : "Collapse the sidebar");
  store.set("salience.rail", collapsed ? "collapsed" : "open");
}
document.getElementById("collapse").addEventListener("click", () => applyRail(store.get("salience.rail") !== "collapsed"));
applyRail(store.get("salience.rail") === "collapsed");

// Library

// At once from what is loaded, then again only if a fresh copy differs: waiting
// on the fresh one first left the last page up for a second (2026-10-25).
let libraryShown = 0;
async function showLibrary() {
  setNav("library");
  const mine = ++libraryShown;
  const stale = libraryCache;
  if (stale) drawLibrary(stale); else page.innerHTML = SKELETON.library;
  const lib = await library(true);
  if (mine !== libraryShown || !location.hash.replace(/^#\/?/, "").match(/^$|^library/)) return;  // gone elsewhere
  if (!stale || JSON.stringify(stale) !== JSON.stringify(lib)) drawLibrary(lib);
}

function drawLibrary(lib) {
  if (!lib.sessions.length) {
    page.innerHTML = `<div class="empty"><h2>No highlights yet</h2>
      <p>Play a match of Fortnite. Salience records on its own, and your highlights appear here when you finish.</p></div>`;
    return;
  }
  const state = { filter: "all", text: "" };
  const total = lib.sessions.reduce((n, s) => n + s.highlights.length, 0);
  const match = (h) => (state.filter === "all" || (state.filter === "loved" && h.loved)
      || (state.filter === "asked" && h.asked))
    && (!state.text || `${h.title} ${h.summary} ${h.said} ${h.weapon}`.toLowerCase().includes(state.text));

  const draw = () => {
    const filtering = state.filter !== "all" || state.text;
    const sessions = lib.sessions.map((s) => ({ ...s, shown: s.highlights.filter(match) }));
    // The top spot is the newest session with highlights; empty ones (a lobby, say) wait below.
    const latest = sessions.find((x) => x.highlights.length) || sessions[0];
    const earlier = sessions.filter((x) => x !== latest);
    const body = filtering
      ? sessions.filter((s) => s.shown.length).map((s) => `<section class="section">
          <div class="section-head"><h2>${esc(when(s.started))}</h2></div>
          <div class="grid">${s.shown.map((h) => card(s, h)).join("")}</div></section>`).join("")
        || '<div class="empty"><h2>Nothing matches</h2><p>Try another word, or clear the filter.</p></div>'
      : `<section class="section">
          <div class="section-head">
            <h2>${esc(when(latest.started))} <span class="meta">${esc(facts(latest))}</span></h2>
            <a href="#/session/${enc(latest.id)}" class="button ghost">Open session</a>
          </div>
          <div class="grid">${latest.highlights.map((h) => card(latest, h)).join("")}</div>
        </section>
        ${earlier.length ? `<section class="section"><h2>Earlier</h2>
          ${earlier.map((s) => `<a class="row" href="#/session/${enc(s.id)}">
            <div class="stills">${s.highlights.slice(0, 3).map((h) => `<img src="${thumb(s, h)}" alt="" loading="lazy">`).join("")}</div>
            <div class="what"><b>${esc(when(s.started))}</b><span>${esc(facts(s))}</span></div>
            <span class="side">${!s.highlights.length ? "No highlights" : s.unrated ? `${s.unrated} to rate` : "All rated"}</span>
          </a>`).join("")}</section>` : ""}`;
    page.querySelector("#results").innerHTML = body;
  };

  const typed = page.querySelector("#search")?.value || "";  // a redraw keeps what was typed
  state.text = typed.trim().toLowerCase();
  page.innerHTML = `<header class="head">
      <div><h1>Library</h1><p class="sub">${lib.sessions.length} session${lib.sessions.length === 1 ? "" : "s"} · ${total} highlights</p></div>
      <div class="chips">
        <label for="search" class="sr">Search highlights</label>
        <input id="search" class="search" type="search" placeholder="Search: sniper, no-scope, what you said">
        <button class="chip on" data-filter="all">All</button>
        <button class="chip" data-filter="loved">Loved</button>
        <button class="chip" data-filter="asked">Asked for</button>
      </div>
    </header><div id="results" style="display:flex;flex-direction:column;gap:36px"></div>`;
  page.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => {
    state.filter = b.dataset.filter;
    page.querySelectorAll("[data-filter]").forEach((x) => x.classList.toggle("on", x === b));
    draw();
  }));
  page.querySelector("#search").value = typed;
  page.querySelector("#search").addEventListener("input", (e) => { state.text = e.target.value.trim().toLowerCase(); draw(); });
  draw();
}

// Session

async function showSession(id, file) {
  setNav("library");
  const lib = await library();
  const s = lib.sessions.find((x) => x.id === id);
  if (!s) { page.innerHTML = '<div class="empty"><h2>That session is gone</h2><a href="#/">Back to the library</a></div>'; return; }
  const current = s.highlights.find((h) => h.file === file) || s.highlights[0];
  const rated = s.windows - s.unrated;
  const share = s.windows ? Math.round((rated / s.windows) * 100) : 100;
  const missing = s.unrated
    ? `<section class="missing" aria-labelledby="missing">
        <div class="mark">${TICK}</div>
        <div class="words">
          <h2 id="missing">Did we miss a moment?</h2>
          <p>${s.unrated} moment${s.unrated === 1 ? "" : "s"} from this session ${s.unrated === 1 ? "isn't" : "aren't"} rated yet. A keep joins your highlights right away, and every rating makes the next session's picks more yours.</p>
          <div class="bar"><div class="track"><div class="fill" style="width:${share}%"></div></div><span>${rated} of ${s.windows} rated</span></div>
        </div>
        <a class="button primary big" href="#/rate">Rate the rest</a>
      </section>`
    : `<section class="missing done"><div class="words"><h2>Every moment rated</h2>
        <p>All ${s.windows} moments from this session are rated. Your next session is ranked with them.</p></div></section>`;

  page.innerHTML = `<a class="back" href="#/">← Library</a>
    <header class="head">
      <div><h1>${esc(when(s.started))}</h1><p class="sub">${esc(facts(s))} · ranked by your taste from ${lib.ratings.toLocaleString()} ratings</p></div>
      <button class="button ghost" id="folder">Show in folder</button>
    </header>
    ${current ? `<div class="split">
      <section class="stage" aria-label="Now playing">
        <div class="player" id="player-box">
          <div class="screen skel" id="screen">
            <video id="player" autoplay playsinline src="/video/${enc(s.id)}/${enc(current.file)}?v=${enc(current.length)}"></video>
            <button class="big-play" id="big-play" aria-label="Play" hidden>${PLAY}</button>
          </div>
          <div class="player-bar">
            <div class="timeline" id="timeline" role="slider" tabindex="0" aria-label="Seek" aria-valuemin="0">
              <div class="tl-rail"></div>
              <div class="tl-played" id="played"></div>
              <div class="tl-cut" id="cut-left"></div><div class="tl-cut" id="cut-right"></div>
              <div class="tl-keep" id="keep"></div>
              <button class="tl-handle" id="h-start" aria-label="Trim start"></button>
              <button class="tl-handle" id="h-end" aria-label="Trim end"></button>
              <span id="kills"></span>
              <span class="tl-head" id="head"></span>
            </div>
            <div class="controls">
              <button class="ctl" id="play" aria-label="Play">${PLAY}</button>
              <span class="time"><span id="now">0:00</span> <span class="faint">/ <span id="total">0:00</span></span></span>
              <span class="killnote" id="killnote" hidden></span>
              <span class="grow"></span>
              <span class="trimnote" id="trimnote" hidden></span>
              <button class="ctl" id="mute" aria-label="Mute">${SOUND}</button>
              <button class="ctl" id="full" aria-label="Full screen">${FULL}</button>
            </div>
          </div>
        </div>
        <div class="caption">
          <div class="words">
            <div class="title"><h2>${esc(current.title)}</h2>${current.loved ? HEART : ""}</div>
            <p>${esc(current.summary)}</p>
            <span class="when">${current.laughed ? "Someone laughed · " : ""}${current.said ? `You said: <i style="color:var(--ink)">“${esc(current.said)}”</i> · ` : ""}${current.at ? `${esc(current.at)} into the session` : ""}</span>
          </div>
          <div class="actions" id="watch-actions">
            ${current.reel ? '<button class="button" id="reel">9:16 reel</button>' : ""}
            <button class="button" id="trim">Trim</button>
            <button class="button" id="export">Export MP4</button>
            ${current.sha ? `<button class="button ${current.rated === 1 ? "primary" : ""}" id="keeper" aria-pressed="${current.rated === 1}">${current.rated === 1 ? "Kept" : "Good one"}</button>
            <button class="button" id="not-keeper">Not a keeper</button>` : ""}
            <button class="button icon" id="delete" aria-label="Delete this highlight">${BIN}</button>
          </div>
          <div class="actions" id="trim-actions" hidden>
            ${current.trimmed ? '<button class="button ghost" id="restore">Restore original</button>' : ""}
            <button class="button ghost" id="trim-cancel">Cancel</button>
            <button class="button primary" id="trim-save">Save trim</button>
          </div>
        </div>
        <div class="saved" id="saved" hidden></div>
      </section>
      <section class="list" aria-labelledby="ranked">
        <h3 id="ranked">${s.highlights.length} highlights</h3>
        ${s.highlights.map((h) => `<a class="item ${h === current ? "on" : ""}" href="#/session/${enc(s.id)}/${enc(h.file)}">
          <span class="n">${esc(h.rank)}</span>
          <span class="mini"><img src="${thumb(s, h)}" alt="" loading="lazy">${h.length ? `<span class="tag">${esc(h.length)}</span>` : ""}</span>
          <span class="words"><b>${esc(h.title)}</b><span>${esc(info(h))}</span></span>
        </a>`).join("")}
      </section>
    </div>` : '<div class="empty"><h2>No highlights in this session</h2></div>'}
    ${missing}`;

  page.querySelector("#folder").addEventListener("click", () => api("/api/open-folder", { session: s.id }));
  if (current) playerControls(s, current);
  const exporter = page.querySelector("#export");
  if (exporter) {
    exporter.addEventListener("click", async () => {
      exporter.disabled = true;
      const done = await api("/api/export", { session: s.id, file: current.file }).catch(() => ({ ok: false }));
      exporter.disabled = false;
      const saved = page.querySelector("#saved");
      if (done.cancelled) return;
      saved.hidden = false;
      saved.innerHTML = done.ok
        ? `Saved <b>${esc(done.path.split(/[\/]/).pop())}</b> <button class="button ghost" id="reveal">Show in folder</button>`
        : "That export could not be made.";
      const reveal = saved.querySelector("#reveal");
      if (reveal) reveal.addEventListener("click", () => api("/api/reveal", { path: done.path }));
    });
  }
  const keeper = page.querySelector("#keeper");
  if (keeper) {
    keeper.addEventListener("click", async () => {
      await api("/api/rate", { sha: current.sha, label: 1 });
      current.rated = 1; keeper.textContent = "Kept"; keeper.classList.add("primary");
      keeper.setAttribute("aria-pressed", "true");
      libraryCache = null;
    });
  }
  const notKeeper = page.querySelector("#not-keeper");
  if (notKeeper) {
    notKeeper.addEventListener("click", async () => {
      if (!confirm(`Not a keeper? “${current.title}” and its reel are removed, and Salience learns you'd skip kills like it.`)) return;
      await api("/api/rate", { sha: current.sha, label: 0 });
      await api("/api/delete", { session: s.id, file: current.file });
      await library(true);
      go(`#/session/${s.id}`);
      refreshChrome();
    });
  }
  const remove = page.querySelector("#delete");
  if (remove) {
    remove.addEventListener("click", async () => {
      if (!confirm(`Delete “${current.title}” and its reel? This can't be undone.`)) return;
      await api("/api/delete", { session: s.id, file: current.file });
      await library(true);
      go(`#/session/${s.id}`);
      refreshChrome();
    });
  }
}

// The player: one timeline for watching and trimming

function playerControls(s, current) {
  const box = page.querySelector("#player-box");
  const video = page.querySelector("#player");
  const line = page.querySelector("#timeline");
  const el = (id) => page.querySelector(`#${id}`);
  const MIN = 1.0; // seconds: the shortest trim the server accepts
  let length = 0, a = 0, b = 0, trimming = false, dragging = null, vertical = false, shownPaused = null;

  const pct = (t) => `${length ? (t / length) * 100 : 0}%`;
  // The marker and its note: shown once the kill is known, which can be after
  // the page is (markers are found on the session page, `/api/kills`).
  // Every kill and knock in the clip, each a diamond and a word in the note (2026-10-25).
  const marks = () => (current.kills && current.kills.length ? current.kills
    : current.kill != null ? [{ at: current.kill, event: current.event }] : []);
  const showKill = () => {
    const list = marks();
    el("killnote").hidden = !list.length;
    el("kills").innerHTML = list.map((k, i) => `<button class="tl-kill" data-kill="${i}"
      title="Jump to the ${esc(k.event || "moment")} at ${esc(clockOf(k.at))}"
      aria-label="Jump to the ${esc(k.event || "moment")} at ${esc(clockOf(k.at))}" style="left:${pct(k.at)}"></button>`).join("");
    el("killnote").innerHTML = "● " + list.map((k, i) => `<button class="killjump" data-kill="${i}">${esc(k.event || "moment")} ${esc(clockOf(k.at))}</button>`).join(" · ");
  };
  if (s.markers_pending) {
    api(`/api/kills?session=${enc(s.id)}`).then((found) => {
      for (const h of s.highlights) if (found[h.file]) Object.assign(h, found[h.file]);
      s.markers_pending = false;
      if (page.contains(video)) showKill();
    }).catch(() => {});
  }
  const paint = () => {
    const t = video.currentTime || 0;
    el("played").style.width = pct(t);
    el("head").style.left = pct(t);
    el("now").textContent = clockOf(t);
    line.setAttribute("aria-valuenow", t.toFixed(1));
    if (trimming) {
      el("cut-left").style.width = pct(a);
      el("cut-right").style.left = pct(b);
      el("keep").style.left = pct(a);
      el("keep").style.width = pct(b - a);
      el("h-start").style.left = pct(a);
      el("h-end").style.left = pct(b);
      el("trimnote").textContent = `Keeps ${tenths(a)} to ${tenths(b)} · ${(Math.round(b * 10) / 10 - Math.round(a * 10) / 10).toFixed(1)}s`;
    }
    // Only when it changes: replacing the icon every frame swapped the element
    // under the pointer between press and release, and the click was lost.
    const paused = video.paused;
    if (paused !== shownPaused) {
      shownPaused = paused;
      el("play").innerHTML = paused ? PLAY : PAUSE;
      el("play").setAttribute("aria-label", paused ? "Play" : "Pause");
      el("big-play").hidden = !paused;
    }
  };
  const loop = () => {
    if (!page.contains(video)) return;
    if (still() && video.paused) { setTimeout(loop, 250); return; }  // see `still`
    paint();
    requestAnimationFrame(loop);
  };
  const ready = () => {
    length = video.duration || 0;
    el("total").textContent = clockOf(length);
    line.setAttribute("aria-valuemax", length.toFixed(1));
    showKill();
    if (!b) b = length;
    paint();
  };
  video.addEventListener("loadedmetadata", ready);
  for (const done of ["loadeddata", "error"]) video.addEventListener(done, () => el("screen").classList.remove("skel"));
  if (video.readyState >= 1) ready();
  requestAnimationFrame(loop);

  const KILL_LEAD_IN = 1.5;  // seconds of the fight before the kill
  const toggle = () => {
    if (video.paused) {
      if (trimming && (video.currentTime < a || video.currentTime >= b - 0.05)) video.currentTime = a;
      video.play();
    } else video.pause();
  };
  el("play").addEventListener("click", toggle);
  el("big-play").addEventListener("click", toggle);
  video.addEventListener("click", toggle);
  video.addEventListener("timeupdate", () => {
    if (trimming && !dragging && video.currentTime >= b) video.currentTime = a; // a trim previews on a loop
  });
  el("mute").addEventListener("click", () => {
    video.muted = !video.muted;
    el("mute").innerHTML = video.muted ? MUTED : SOUND;
    el("mute").setAttribute("aria-label", video.muted ? "Unmute" : "Mute");
  });
  el("full").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen(); else box.requestFullscreen();
  });

  // Scrubbing and the trim handles share the timeline.
  const at = (e) => {
    const r = line.getBoundingClientRect();
    return Math.min(length, Math.max(0, ((e.clientX - r.left) / r.width) * length));
  };
  const move = (e) => {
    const t = at(e);
    if (dragging === "start") a = Math.min(t, b - MIN);
    else if (dragging === "end") b = Math.max(t, a + MIN);
    if (dragging) video.currentTime = dragging === "start" ? a : dragging === "end" ? b : t;
    paint();
  };
  // The marker, or its note: watch the kill from a moment before it, to see the shot.
  const toKill = (i) => {
    const k = marks()[i];
    if (!length || !k) return;
    video.currentTime = Math.max(trimming ? a : 0, k.at - KILL_LEAD_IN);
    video.play();
  };
  el("killnote").addEventListener("click", (e) => {
    const b = e.target.closest("[data-kill]");
    if (b) toKill(Number(b.dataset.kill));
  });
  line.addEventListener("pointerdown", (e) => {
    if (!length) return;
    const mark = e.target.closest(".tl-kill");
    if (mark) { e.preventDefault(); toKill(Number(mark.dataset.kill)); return; }
    dragging = e.target.id === "h-start" ? "start" : e.target.id === "h-end" ? "end" : "seek";
    if (dragging !== "seek") video.pause();
    line.setPointerCapture(e.pointerId);
    move(e);
  });
  line.addEventListener("pointermove", (e) => { if (dragging) move(e); });
  const drop = () => { dragging = null; };
  line.addEventListener("pointerup", drop);
  line.addEventListener("pointercancel", drop);
  line.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 5 : 1;
    if (e.key === "ArrowLeft") { e.preventDefault(); video.currentTime = Math.max(0, video.currentTime - step); }
    if (e.key === "ArrowRight") { e.preventDefault(); video.currentTime = Math.min(length, video.currentTime + step); }
  });
  const keys = (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === " " || e.key === "k") { e.preventDefault(); toggle(); }
  };
  document.addEventListener("keydown", keys);
  leaving = () => document.removeEventListener("keydown", keys);

  const setTrimming = (on) => {
    trimming = on;
    box.classList.toggle("trimming", on);
    el("trimnote").hidden = !on;
    el("watch-actions").hidden = on;
    el("trim-actions").hidden = !on;
    if (on) { a = 0; b = length; video.pause(); }
    paint();
  };
  el("trim").addEventListener("click", () => setTrimming(true));
  el("trim-cancel").addEventListener("click", () => setTrimming(false));
  el("trim-save").addEventListener("click", async (e) => {
    e.target.disabled = true; e.target.textContent = "Saving…";
    const done = await api("/api/trim", { session: s.id, file: current.file, start: a, end: b }).catch(() => ({ ok: false }));
    if (!done.ok) { e.target.disabled = false; e.target.textContent = "Save trim"; alert("That trim could not be saved."); return; }
    await library(true);
    route();
  });
  const restore = el("restore");
  if (restore) restore.addEventListener("click", async () => {
    await api("/api/restore", { session: s.id, file: current.file });
    await library(true);
    route();
  });
  const reel = el("reel");
  if (reel) reel.addEventListener("click", () => {
    vertical = !vertical;
    const t = video.currentTime;
    el("screen").classList.toggle("reel", vertical);
    video.src = vertical ? `/reel/${enc(s.id)}/${enc(current.reel)}` : `/video/${enc(s.id)}/${enc(current.file)}?v=${enc(current.length)}`;
    video.addEventListener("loadedmetadata", () => { video.currentTime = t; }, { once: true });
    reel.textContent = vertical ? "16:9 original" : "9:16 reel";
    el("trim").disabled = vertical;
  });
}

// Rating

// Shown at once while a view's data comes in, in the shape of what will fill it.
const SKELETON = {
  rate: `<header class="head"><div><h1>Rate clips</h1><p class="sub skel skel-line" style="width:340px"></p></div>
      <div class="skel" style="width:240px;height:6px;border-radius:3px"></div></header>
    <div class="split" aria-busy="true" aria-label="Loading the next moment">
      <section class="stage"><div class="skel" style="aspect-ratio:16/9;border-radius:14px"></div>
        <div class="skel skel-line" style="width:80%"></div><div class="skel skel-line" style="width:40%"></div></section>
      <section class="calls"><div class="skel" style="height:56px;border-radius:12px"></div>
        <div class="skel" style="height:56px;border-radius:12px"></div><div class="skel" style="height:56px;border-radius:12px"></div>
        <div class="skel" style="height:44px;border-radius:999px;width:70%"></div></section></div>`,
  library: `<header class="head"><div><h1>Library</h1><p class="sub skel skel-line" style="width:200px"></p></div></header>
    <div class="grid" aria-busy="true" aria-label="Loading your highlights">${'<div class="card"><div class="skel" style="aspect-ratio:16/9;border-radius:12px"></div><div class="skel skel-line" style="width:70%"></div><div class="skel skel-line" style="width:45%"></div></div>'.repeat(8)}</div>`,
};

async function showRate(id) {
  setNav("rate");
  page.innerHTML = SKELETON.rate;
  // The library as already loaded: a fresh one is a round trip the player waits on.
  const lib = await library();
  if (!lib.sessions.length) { page.innerHTML = '<div class="empty"><h2>Nothing to rate yet</h2><p>Play a session first.</p></div>'; return; }
  if (id) { history.replaceState(null, "", "#/rate"); }
  // One queue across every session: a session at a time said "all rated"
  // while other sessions still had moments to rate (2026-11-04).
  const total = lib.sessions.reduce((n, x) => n + x.windows, 0);
  let kept = 0;
  let wrong = false;
  let clip = null;

  const shell = () => {
    page.innerHTML = `<header class="head">
        <div><h1>Rate clips</h1><p class="sub">Kills, laughs and "clip that" from every session · every rating teaches the ranking what you like</p></div>
        <div class="progress"><div class="track"><div class="fill" id="fill"></div></div><span id="count"></span></div>
      </header><div id="body">${SKELETON.rate.slice(SKELETON.rate.indexOf('<div class="split"'))}</div>`;
  };

  // Rated this sitting, newest last: Back takes the last one back and offers it again.
  const done = [];

  const next = async () => show(await api("/api/rate/next"));

  const back = async () => {
    const last = done.pop();
    if (!last) return;
    clip = null;
    const undone = await api("/api/unrate", { sha: last.sha });
    if (last.label) kept = Math.max(0, kept - 1);
    libraryCache = null;
    await show(undone && undone.clip && !undone.clip.done ? undone.clip : await api("/api/rate/next"));
    refreshChrome();
  };

  const show = async (offered) => {
    clip = offered;
    wrong = false;
    const left = clip.done ? 0 : clip.remaining;
    page.querySelector("#fill").style.width = `${total ? Math.round(((total - left) / total) * 100) : 100}%`;
    page.querySelector("#count").textContent = clip.done ? `${total} of ${total} rated` : `${left} left`;
    const body = page.querySelector("#body");
    if (clip.done) {
      body.innerHTML = `<div class="empty"><h2>All rated</h2>
        <p>${kept ? `${kept} keep${kept === 1 ? "" : "s"} added to highlights. ` : ""}Nothing left to rate in any session. Your next session is ranked with these.</p>
        <a class="button primary" href="#/">Back to the library</a>
        ${done.length ? `<button class="wrong" id="back">← Back to the last one (B)</button>` : ""}</div>`;
      const backButton = body.querySelector("#back");
      if (backButton) backButton.addEventListener("click", back);
      return;
    }
    body.innerHTML = `<div class="split">
      <section class="stage" aria-label="Moment">
        <div class="screen skel" id="clip-screen"><video id="clip" src="/media/${enc(clip.sha)}" autoplay loop controls muted></video></div>
        <div class="caption"><div class="words">
          <p style="font-size:16px;color:var(--ink)">${esc(clip.summary)}</p>
          <span class="when">${clip.said ? `You said: <i style="color:var(--ink)">“${esc(clip.said)}”</i>` : "Nothing said on the mic"}</span>
          ${clip.pc_said ? `<span class="when">Friends or the game said: <i style="color:var(--ink)">“${esc(clip.pc_said)}”</i></span>` : ""}
          <span class="when">${clip.laughed && clip.laughed.length
            ? `Laughter heard: ${clip.laughed.map((t) => (t === "mic" ? "you" : "your PC's audio")).join(" and ")}`
            : "No laughter heard"}</span>
          ${clip.unsure ? `<span class="when">Salience can't tell if you'd keep this kill: your call teaches it most.</span>` : ""}
        </div></div>
      </section>
      <section class="calls" aria-label="Your call">
        <button class="call keep" data-call="keep">Keep <kbd>K</kbd></button>
        <button class="call love" data-call="love">Love it <kbd>L</kbd></button>
        <button class="call" data-call="skip">Skip <kbd>J</kbd></button>
        <div style="font-size:13px;color:var(--muted);margin-top:8px">Skip because…</div>
        <div class="reasons">
          <button class="reason" data-call="spray">1 · Spray damage</button>
          <button class="reason" data-call="bot">2 · Bot or NPC</button>
          <button class="reason" data-call="dull">3 · Nothing happens</button>
        </div>
        <button class="wrong" id="wrong" aria-pressed="false">Description is wrong (W)</button>
        ${done.length ? `<button class="wrong" id="back">← Back to the last one (B)</button>` : ""}
        <div class="note"><b>${kept ? `${kept} keep${kept === 1 ? "" : "s"} added to highlights.` : "Nothing kept yet."}</b><br>
          A keep joins its session's highlights right away, with a 9:16 reel.</div>
      </section></div>`;
    const video = body.querySelector("#clip");
    // The moment may be re-cut from the raw footage first (about a second).
    for (const done of ["loadeddata", "error"]) {
      video.addEventListener(done, () => body.querySelector("#clip-screen").classList.remove("skel"), { once: true });
    }
    video.muted = false; // muted only so autoplay is allowed
    body.querySelectorAll("[data-call]").forEach((b) => b.addEventListener("click", () => rate(b.dataset.call)));
    body.querySelector("#wrong").addEventListener("click", toggleWrong);
    const backButton = body.querySelector("#back");
    if (backButton) backButton.addEventListener("click", back);
  };

  const toggleWrong = () => {
    wrong = !wrong;
    const button = page.querySelector("#wrong");
    if (button) { button.classList.toggle("on", wrong); button.setAttribute("aria-pressed", String(wrong)); }
  };

  const rate = async (call) => {
    if (!clip || clip.done) return;
    const label = call === "keep" || call === "love" ? 1 : 0;
    const body = { sha: clip.sha, label, wrong };
    if (call === "love") body.love = true;
    if (["spray", "bot", "dull"].includes(call)) body.reason = call;
    clip = null;
    await api("/api/rate", body);
    done.push({ sha: body.sha, label });
    if (label) kept += 1;
    libraryCache = null;
    await next();
    refreshChrome();
  };

  const keys = (e) => {
    if (e.target.closest("input, select, textarea")) return;
    const call = { k: "keep", l: "love", j: "skip", 1: "spray", 2: "bot", 3: "dull" }[e.key.toLowerCase()];
    if (call) { e.preventDefault(); rate(call); }
    else if (e.key.toLowerCase() === "w") { e.preventDefault(); toggleWrong(); }
    else if (e.key.toLowerCase() === "b") { e.preventDefault(); back(); }
  };
  document.addEventListener("keydown", keys);
  leaving = () => document.removeEventListener("keydown", keys);
  shell();
  await next();
}

// Settings

async function showSettings() {
  setNav("settings");
  let set = await api("/api/settings");
  const looks = await themes();
  const names = { standard: "Standard", high: "High", max: "Max" };

  const draw = () => {
    const custom = set.quality === "custom";
    const perHour = (set.mbps * 0.45).toFixed(set.mbps * 0.45 < 10 ? 1 : 0);
    page.innerHTML = `<header><h1>Settings</h1><p class="sub">Changes apply from the next recording.</p></header>
      <section class="panel" aria-labelledby="rec">
        <h2 id="rec">Recording</h2>
        <div class="presets" role="radiogroup" aria-label="Quality preset">
          ${Object.entries(set.presets).map(([id, p]) => `<button class="preset ${set.quality === id ? "on" : ""}" role="radio"
            aria-checked="${set.quality === id}" data-preset="${id}"><b>${names[id] || id}</b><span>${p.fps} fps · ${p.mbps} Mbps</span></button>`).join("")}
          <button class="preset ${custom ? "on" : ""}" role="radio" aria-checked="${custom}" data-preset="custom">
            <b>Custom</b><span>${custom ? `${set.fps} fps · ${set.mbps} Mbps` : "Set your own below"}</span></button>
        </div>
        <div class="fields">
          <div class="field"><span id="res-label">Resolution</span>
            <div class="segmented" role="radiogroup" aria-labelledby="res-label">
              ${Object.entries(set.resolutions).map(([id, label]) => `<button role="radio" aria-checked="${set.resolution === id}"
                class="${set.resolution === id ? "on" : ""}" data-res="${id}">${esc(label.replace(" (your screen's)", ""))}</button>`).join("")}
            </div><small>Full size is your screen's own. Never scaled up.</small></div>
          <div class="field"><span id="fps-label">Frame rate</span>
            <div class="segmented" role="radiogroup" aria-labelledby="fps-label">
              ${set.frame_rates.slice().reverse().map((f) => `<button role="radio" aria-checked="${set.fps === f}"
                class="${set.fps === f ? "on" : ""}" data-fps="${f}">${f} fps</button>`).join("")}
            </div></div>
        </div>
        <div class="slider">
          <div class="top"><label for="bitrate" style="font-size:13px;color:var(--muted)">Bitrate</label><b id="mbps">${set.mbps} Mbps</b></div>
          <input id="bitrate" type="range" min="${set.bitrate_range[0]}" max="${set.bitrate_range[1]}" step="1" value="${set.mbps}">
          <div class="ends"><span>${set.bitrate_range[0]} · smaller files</span><span>${set.bitrate_range[1]} · sharper motion</span></div>
        </div>
        <div class="summary"><b>${custom ? "Custom" : names[set.quality]}: ${esc((set.resolutions[set.resolution] || "").replace(" (your screen's)", ""))}, ${set.fps} fps, ${set.mbps} Mbps</b>
          <span>About ${perHour} GB of raw footage an hour</span></div>
      </section>
      <section class="panel" aria-labelledby="look">
        <h2 id="look">Appearance</h2>
        <div class="field"><span id="mode-label">Mode</span>
          <div class="segmented" role="radiogroup" aria-labelledby="mode-label" style="max-width:420px">
            ${["auto", "light", "dark"].map((m) => `<button role="radio" data-mode="${m}" aria-checked="${(store.get("salience.theme") || "auto") === m}"
              class="${(store.get("salience.theme") || "auto") === m ? "on" : ""}">${m === "auto" ? "Follow Windows" : m === "light" ? "Light" : "Dark"}</button>`).join("")}
          </div></div>
        <div class="field"><span id="palette-label">Theme</span>
          <div class="palettes" role="radiogroup" aria-labelledby="palette-label">
            ${looks.palettes.map((t) => {
              const on = (store.get("salience.palette") || looks.default_palette) === t.name;
              const half = (c) => `<span class="swatch-half" style="background:${c.ground}"><span style="background:${c.select};color:${c.ink}">Aa</span><i style="background:${c.strong}"></i></span>`;
              return `<button class="palette ${on ? "on" : ""}" role="radio" aria-checked="${on}" data-palette="${t.name}">
                <span class="swatch">${half(t.light)}${half(t.dark)}</span><span class="palette-name">${esc(t.label)}</span></button>`;
            }).join("")}
          </div></div>
        <div class="field"><span id="pattern-label">Background</span>
          <div class="patterns" role="radiogroup" aria-labelledby="pattern-label">
            ${looks.patterns.map((t) => {
              const on = (store.get("salience.pattern") || looks.default_pattern) === t.name;
              return `<button class="pattern ${on ? "on" : ""}" role="radio" aria-checked="${on}" data-pattern-pick="${t.name}">
                <span class="pattern-preview" style="background-image:${t.image.replace(/"/g, "&quot;")};background-size:${t.size}"></span>${esc(t.label)}</button>`;
            }).join("")}
          </div></div>
      </section>
      <section class="panel" aria-labelledby="gen">
        <h2 id="gen">General</h2>
        <div class="line"><div><div id="auto-label"><b>Auto-clip my games</b></div><span>Runs the recorder in the background. It records when Fortnite opens and makes highlights when you finish. Closing this window doesn't stop it.</span></div>
          <button class="switch" id="set-autoclip" role="switch" aria-checked="${set.autoclip}" aria-labelledby="auto-label"><span></span></button></div>
        <div class="line"><div><div id="boot-label"><b>Start with Windows</b></div><span>Starts the recorder when you sign in, so it's ready before your first match. Only the small tray icon starts, not this window.</span></div>
          <button class="switch" id="set-startup" role="switch" aria-checked="${set.starts_with_windows}" aria-labelledby="boot-label"><span></span></button></div>
        <div class="line"><div><div id="obs-label"><b>Use my OBS recordings</b></div><span>If you already record or stream with OBS. Salience records nothing itself; when you close Fortnite it takes that game from your OBS recording. Keep OBS recording while you play.${set.uses_obs ? ` Looking in ${esc(set.obs_folder || "your Videos folder")}.` : ""}</span>
          ${set.uses_obs ? `<div style="margin-top:8px"><button class="button" id="obs-folder">Change folder…</button></div>` : ""}</div>
          <button class="switch" id="set-obs" role="switch" aria-checked="${set.uses_obs}" aria-labelledby="obs-label"><span></span></button></div>
        <div class="line"><div><b>Export my stats</b><span id="stats-note">A small file of counts: sessions, highlights, what you kept, how long highlights took, and whether the game stuttered. No video, audio, words or names. Nothing is sent: you choose who gets it.</span></div>
          <button class="button" id="export-stats">Export…</button></div>
        <div class="line"><div><div id="pc-label"><b>Transcribe friends</b></div><span>Turns what your PC plays (friends on Discord or party chat, and the game) into text, so funny moments and their "clip that" count. It's kept on this PC and never sent anywhere. Your friends' voices are recorded either way.</span></div>
          <button class="switch" id="set-pc" role="switch" aria-checked="${set.transcribe_pc}" aria-labelledby="pc-label"><span></span></button></div>
      </section>
      <section class="panel" aria-labelledby="liked">
        <h2 id="liked">Clips you like</h2>
        <div class="line"><div><b>Teach Salience your taste</b><span>Add clips you saved or montages you made. Salience splits them into moments and learns what you like from them, before you've rated anything. Describing them uses your Fireworks key, one call per moment.</span></div></div>
        <div id="import-box"></div>
      </section>
      <section class="panel" aria-labelledby="ai">
        <h2 id="ai">Describer</h2>
        ${set.relay ? `<div class="line"><div><b>Beta: ${matchTime(set.relay.left_seconds)} of match time left this month</b>
          <span>Of ${matchTime(set.relay.allowance_seconds)}. It comes back on ${new Date(set.relay.resets_at).toLocaleDateString(undefined, { month: "long", day: "numeric" })}. Highlights from games past it are made then.</span></div></div>` : ""}
        ${set.invites_open && !set.relay ? `<div class="field">
          <label for="invite">Invite code</label>
          <div style="display:flex;flex-wrap:wrap;gap:8px">
            <input id="invite" class="search" style="flex:1 1 280px;width:auto" autocomplete="off" placeholder="From the person who invited you">
            <button class="button primary" id="save-invite">Use code</button>
          </div>
          <small id="invite-note">No Fireworks account needed: the beta describes your games for you, 5 hours of matches a month.</small>
        </div>` : ""}
        <div class="field">
          <label for="key">Fireworks API key</label>
          <div style="display:flex;flex-wrap:wrap;gap:8px">
            <input id="key" class="search" style="flex:1 1 280px;width:auto" type="password" autocomplete="off"
              placeholder="${set.has_key ? "Saved. Paste a new key to replace it" : "Paste your key from fireworks.ai"}">
            <button class="button primary" id="save-key">Save</button>
          </div>
          <small id="key-note">${set.has_key ? "Saved on this PC and sent only to Fireworks." : "Highlights can't be described without it."}</small>
        </div>
      </section>`;

    page.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", async () => {
      const id = b.dataset.preset;
      set = id === "custom"
        ? await api("/api/settings", { custom: { fps: set.fps, mbps: set.mbps } })
        : await api("/api/settings", { quality: id });
      draw(); refreshChrome();
    }));
    page.querySelectorAll("[data-res]").forEach((b) => b.addEventListener("click", async () => {
      set = await api("/api/settings", { resolution: b.dataset.res }); draw(); refreshChrome();
    }));
    page.querySelectorAll("[data-fps]").forEach((b) => b.addEventListener("click", async () => {
      set = await api("/api/settings", { custom: { fps: Number(b.dataset.fps), mbps: set.mbps } }); draw(); refreshChrome();
    }));
    page.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { applyTheme(b.dataset.mode); draw(); }));
    page.querySelectorAll(".palette").forEach((b) => b.addEventListener("click", () => { applyPalette(b.dataset.palette); draw(); }));
    page.querySelectorAll("[data-pattern-pick]").forEach((b) => b.addEventListener("click", () => { applyPattern(b.dataset.patternPick); draw(); }));
    paintImport(page);
    page.querySelector("#set-autoclip").addEventListener("click", () => setAutoclip(!set.autoclip));
    page.querySelector("#set-pc").addEventListener("click", async () => {
      set = await api("/api/transcribe-pc", { on: !set.transcribe_pc }); draw();
    });
    page.querySelector("#set-obs").addEventListener("click", async () => {
      set = await api("/api/obs", { on: !set.uses_obs }); draw();
    });
    const obsFolder = page.querySelector("#obs-folder");
    if (obsFolder) obsFolder.addEventListener("click", async () => {
      set = await api("/api/obs", { choose: true }); draw();
    });
    page.querySelector("#set-startup").addEventListener("click", async () => {
      set = await api("/api/startup", { on: !set.starts_with_windows }); draw();
    });
    page.querySelector("#export-stats").addEventListener("click", async (e) => {
      e.target.disabled = true; e.target.textContent = "Counting…";
      const { status, data } = await postJson("/api/stats-export", {});
      e.target.disabled = false; e.target.textContent = "Export…";
      setText("stats-note", status === 200 ? `Saved to ${data.path}. It's open in File Explorer.` : "That didn't work.");
    });
    const invite = page.querySelector("#save-invite");
    if (invite) invite.addEventListener("click", async (e) => {
      const code = page.querySelector("#invite").value.trim();
      e.target.disabled = true; e.target.textContent = "Checking…";
      const { status, data } = await postJson("/api/invite", { code });
      if (status === 200) { set = data; draw(); pollRecorder(); return; }
      e.target.disabled = false; e.target.textContent = "Use code";
      page.querySelector("#invite-note").textContent = data.error || "That code didn't work.";
    });
    page.querySelector("#save-key").addEventListener("click", async (e) => {
      const key = page.querySelector("#key").value.trim();
      e.target.disabled = true; e.target.textContent = "Checking…";
      const { status, data } = await postJson("/api/key", { key });
      if (status === 200) { set = data; draw(); pollRecorder();
        page.querySelector("#key-note").textContent = "Saved. Fireworks accepted it."; return; }
      e.target.disabled = false; e.target.textContent = "Save";
      page.querySelector("#key-note").textContent = data.error || "That key could not be saved.";
    });
    const slider = page.querySelector("#bitrate");
    slider.addEventListener("input", () => { page.querySelector("#mbps").textContent = `${slider.value} Mbps`; });
    slider.addEventListener("change", async () => {
      set = await api("/api/settings", { custom: { fps: set.fps, mbps: Number(slider.value) } }); draw(); refreshChrome();
    });
  };
  draw();
}

// Routing

// Somewhere, drawn afresh even when already there: a deleted highlight stayed
// on screen when the address did not change (2026-11-05).
function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

async function route() {
  if (leaving) { leaving(); leaving = null; }
  const [view, id, file] = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  try {
    if (view === "session" && id) await showSession(id, file);
    else if (view === "rate") await showRate(id);
    else if (view === "settings") await showSettings();
    else await showLibrary();
  } catch (error) {
    page.innerHTML = `<div class="empty"><h2>Something went wrong</h2><p>${esc(error.message)}</p>
      <button class="button" onclick="location.reload()">Try again</button></div>`;
  }
  page.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
route();
refreshChrome();
