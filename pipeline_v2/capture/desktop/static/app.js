"use strict";

// The desktop app: four views on one local server (`server.py`).
//   #/                      library: every session's highlights
//   #/session/<id>[/<file>] one session, one highlight playing
//   #/rate[/<id>]           rating a session's windows
//   #/settings              the recording settings that are the player's

const page = document.getElementById("page");
const HEART = '<svg width="20" height="20" viewBox="0 0 24 24" fill="#FF5C93" role="img" aria-label="Loved"><path d="M12 21s-7.5-4.6-9.5-9.2C1.1 8.4 3.2 5 6.6 5c2 0 3.4 1.1 4.2 2.4h2.4C14 6.1 15.4 5 17.4 5c3.4 0 5.5 3.4 4.1 6.8C19.5 16.4 12 21 12 21z"></path></svg>';
const TICK = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0D0F13" stroke-width="2.2" aria-hidden="true"><path d="M5 12l4 4L19 6"></path></svg>';
const BIN = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"></path></svg>';

let libraryCache = null;
let leaving = null; // the current view's clean-up (key handlers)

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const enc = encodeURIComponent;

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
      ${h.added ? '<span class="tag added">From rating</span>' : ""}
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

// Library

async function showLibrary() {
  setNav("library");
  const lib = await library(true);
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
    const [latest, ...earlier] = sessions;
    const body = filtering
      ? sessions.filter((s) => s.shown.length).map((s) => `<section class="section">
          <div class="section-head"><h2>${esc(when(s.started))}</h2></div>
          <div class="grid">${s.shown.map((h) => card(s, h)).join("")}</div></section>`).join("")
        || '<div class="empty"><h2>Nothing matches</h2><p>Try another word, or clear the filter.</p></div>'
      : `<section class="section">
          <div class="section-head">
            <div><h2 style="display:inline">${esc(when(latest.started))}</h2><span class="meta">${esc(facts(latest))}</span></div>
            <a href="#/session/${enc(latest.id)}" class="button ghost">Open session</a>
          </div>
          <div class="grid">${latest.highlights.map((h) => card(latest, h)).join("")}</div>
        </section>
        ${earlier.length ? `<section class="section"><h2>Earlier</h2>
          ${earlier.map((s) => `<a class="row" href="#/session/${enc(s.id)}">
            <div class="stills">${s.highlights.slice(0, 3).map((h) => `<img src="${thumb(s, h)}" alt="" loading="lazy">`).join("")}</div>
            <div class="what"><b>${esc(when(s.started))}</b><span>${esc(facts(s))}</span></div>
            <span class="side">${s.unrated ? `${s.unrated} to rate` : "All rated"}</span>
          </a>`).join("")}</section>` : ""}`;
    page.querySelector("#results").innerHTML = body;
  };

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
        <a class="button primary big" href="#/rate/${enc(s.id)}">Rate the rest</a>
      </section>`
    : `<section class="missing done"><div class="words"><h2>Every moment rated</h2>
        <p>All ${s.windows} moments from this session are rated. Your next session is ranked with them.</p></div></section>`;

  page.innerHTML = `<a class="back" href="#/">← Library</a>
    <header class="head">
      <div><h1>${esc(when(s.started))}</h1><p class="sub">${esc(facts(s))} · ranked by your taste from ${lib.ratings.toLocaleString()} ratings</p></div>
      <button class="button ghost" id="folder">Show in folder</button>
    </header>
    ${missing}
    ${current ? `<div class="split">
      <section class="stage" aria-label="Now playing">
        <div class="screen" id="screen"><video id="player" controls autoplay src="/video/${enc(s.id)}/${enc(current.file)}?v=${enc(current.length)}"></video></div>
        <div class="caption">
          <div class="words">
            <div class="title"><h2>${esc(current.title)}</h2>${current.loved ? HEART : ""}</div>
            <p>${esc(current.summary)}</p>
            <span class="when">${current.said ? `You said: <i style="color:var(--ink)">“${esc(current.said)}”</i> · ` : ""}${current.at ? `${esc(current.at)} into the session` : ""}</span>
          </div>
          <div class="actions">
            ${current.reel ? '<button class="button" id="reel">9:16 reel</button>' : ""}
            <button class="button" id="trim">Trim</button>
            <button class="button" id="export">Export MP4</button>
            <button class="button icon" id="delete" aria-label="Delete this highlight">${BIN}</button>
          </div>
        </div>
        <div class="trimmer" id="trimmer" hidden>
          <div class="trim-track" id="track">
            <div class="trim-keep" id="keep"></div>
            <input type="range" id="trim-start" min="0" max="1" step="0.1" value="0" aria-label="Trim start">
            <input type="range" id="trim-end" min="0" max="1" step="0.1" value="1" aria-label="Trim end">
          </div>
          <div class="trim-row">
            <span id="trim-label"></span>
            <div class="actions">
              ${current.trimmed ? '<button class="button ghost" id="restore">Restore original</button>' : ""}
              <button class="button ghost" id="trim-cancel">Cancel</button>
              <button class="button" id="trim-preview">Preview</button>
              <button class="button primary" id="trim-save">Save trim</button>
            </div>
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
    </div>` : '<div class="empty"><h2>No highlights in this session</h2></div>'}`;

  page.querySelector("#folder").addEventListener("click", () => api("/api/open-folder", { session: s.id }));
  const reel = page.querySelector("#reel");
  if (reel) {
    let vertical = false;
    reel.addEventListener("click", () => {
      vertical = !vertical;
      const screen = page.querySelector("#screen");
      screen.classList.toggle("reel", vertical);
      page.querySelector("#player").src = vertical
        ? `/reel/${enc(s.id)}/${enc(current.reel)}` : `/video/${enc(s.id)}/${enc(current.file)}`;
      reel.textContent = vertical ? "16:9 original" : "9:16 reel";
    });
  }
  const player = page.querySelector("#player");
  const trimmer = page.querySelector("#trimmer");
  if (trimmer) {
    const first = page.querySelector("#trim-start");
    const last = page.querySelector("#trim-end");
    const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}.${Math.floor((t % 1) * 10)}`;
    let previewing = false;
    const paint = () => {
      const length = Number(last.max) || 1;
      let a = Number(first.value), b = Number(last.value);
      if (b - a < 1) { if (document.activeElement === first) first.value = a = Math.max(0, b - 1); else last.value = b = Math.min(length, a + 1); }
      const keep = page.querySelector("#keep");
      keep.style.left = `${(a / length) * 100}%`;
      keep.style.width = `${((b - a) / length) * 100}%`;
      page.querySelector("#trim-label").textContent = `${clock(a)} to ${clock(b)} · keeps ${(b - a).toFixed(1)}s of ${length.toFixed(1)}s`;
    };
    const ready = () => {
      const length = Math.floor(player.duration * 10) / 10;
      first.max = last.max = String(length);
      if (Number(last.value) <= 1) last.value = String(length);
      paint();
    };
    page.querySelector("#trim").addEventListener("click", () => {
      trimmer.hidden = !trimmer.hidden;
      if (!trimmer.hidden) { if (player.readyState >= 1) ready(); else player.addEventListener("loadedmetadata", ready, { once: true }); }
    });
    first.addEventListener("input", () => { paint(); player.currentTime = Number(first.value); });
    last.addEventListener("input", () => { paint(); player.currentTime = Number(last.value); });
    player.addEventListener("timeupdate", () => {
      if (previewing && player.currentTime >= Number(last.value)) { player.pause(); previewing = false; }
    });
    page.querySelector("#trim-preview").addEventListener("click", () => {
      previewing = true; player.currentTime = Number(first.value); player.play();
    });
    page.querySelector("#trim-cancel").addEventListener("click", () => { trimmer.hidden = true; previewing = false; });
    page.querySelector("#trim-save").addEventListener("click", async (e) => {
      e.target.disabled = true; e.target.textContent = "Saving…";
      const done = await api("/api/trim", { session: s.id, file: current.file,
        start: Number(first.value), end: Number(last.value) }).catch(() => ({ ok: false }));
      if (!done.ok) { e.target.disabled = false; e.target.textContent = "Save trim"; alert("That trim could not be saved."); return; }
      await library(true);
      route();
    });
    const restore = page.querySelector("#restore");
    if (restore) restore.addEventListener("click", async () => {
      await api("/api/restore", { session: s.id, file: current.file });
      await library(true);
      route();
    });
  }
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
  const remove = page.querySelector("#delete");
  if (remove) {
    remove.addEventListener("click", async () => {
      if (!confirm(`Delete “${current.title}” and its reel? This can't be undone.`)) return;
      await api("/api/delete", { session: s.id, file: current.file });
      await library(true);
      location.hash = `#/session/${s.id}`;
      refreshChrome();
    });
  }
}

// Rating

async function showRate(id) {
  setNav("rate");
  const lib = await library(true);
  const s = lib.sessions.find((x) => x.id === id)
    || lib.sessions.find((x) => x.unrated > 0) || lib.sessions[0];
  if (!s) { page.innerHTML = '<div class="empty"><h2>Nothing to rate yet</h2><p>Play a session first.</p></div>'; return; }
  if (!id || id !== s.id) { history.replaceState(null, "", `#/rate/${s.id}`); }
  let kept = 0;
  let wrong = false;
  let clip = null;

  const shell = () => {
    page.innerHTML = `<header class="head">
        <div><h1>Rate clips</h1><p class="sub">${esc(when(s.started))} · every rating teaches the ranking what you like</p></div>
        <div class="progress"><div class="track"><div class="fill" id="fill"></div></div><span id="count"></span></div>
      </header><div id="body"></div>`;
  };

  const next = async () => {
    clip = await api(`/api/rate/next?session=${enc(s.id)}`);
    wrong = false;
    const total = s.windows;
    const left = clip.done ? 0 : clip.remaining;
    page.querySelector("#fill").style.width = `${total ? Math.round(((total - left) / total) * 100) : 100}%`;
    page.querySelector("#count").textContent = clip.done ? `${total} of ${total} rated` : `${left} left`;
    const body = page.querySelector("#body");
    if (clip.done) {
      body.innerHTML = `<div class="empty"><h2>All rated</h2>
        <p>${kept ? `${kept} keep${kept === 1 ? "" : "s"} added to this session's highlights. ` : ""}Your next session is ranked with these.</p>
        <a class="button primary" href="#/session/${enc(s.id)}">Back to the session</a></div>`;
      return;
    }
    body.innerHTML = `<div class="split">
      <section class="stage" aria-label="Moment">
        <div class="screen"><video id="clip" src="/media/${enc(clip.sha)}" autoplay loop controls muted></video></div>
        <div class="caption"><div class="words">
          <p style="font-size:16px;color:var(--ink)">${esc(clip.summary)}</p>
          <span class="when">${clip.said ? `You said: <i style="color:var(--ink)">“${esc(clip.said)}”</i>` : "Nothing said on the mic"}</span>
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
        <div class="note"><b>${kept ? `${kept} keep${kept === 1 ? "" : "s"} added to highlights.` : "Nothing kept yet."}</b><br>
          Keeps join this session's highlights right away, with a 9:16 reel.</div>
      </section></div>`;
    const video = body.querySelector("#clip");
    video.muted = false; // muted only so autoplay is allowed
    body.querySelectorAll("[data-call]").forEach((b) => b.addEventListener("click", () => rate(b.dataset.call)));
    body.querySelector("#wrong").addEventListener("click", toggleWrong);
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
      <section class="panel" aria-labelledby="ai">
        <h2 id="ai">Describer</h2>
        <div class="line"><div><div>Fireworks API key</div><span>${set.has_key ? "Set. Kept on this PC and sent only to Fireworks." : "Not set: highlights can't be described without it."}</span></div>
          <span>Change it from the tray menu</span></div>
      </section>
      <section class="panel" aria-labelledby="gen">
        <h2 id="gen">General</h2>
        <div class="line"><div>Start Salience with Windows</div><span>${set.starts_with_windows ? "On" : "Off"} · change it from the tray menu</span></div>
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
    const slider = page.querySelector("#bitrate");
    slider.addEventListener("input", () => { page.querySelector("#mbps").textContent = `${slider.value} Mbps`; });
    slider.addEventListener("change", async () => {
      set = await api("/api/settings", { custom: { fps: set.fps, mbps: Number(slider.value) } }); draw(); refreshChrome();
    });
  };
  draw();
}

// Routing

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
