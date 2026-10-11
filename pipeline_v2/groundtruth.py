"""Blind weapon and actor ground truth on a frozen random sample.

Every weapon claim in this project so far rests on seven windows that were
hand-picked *because* they had already failed, then reused to tune the prompt.
Nothing can be concluded from that in either direction. This collects labels on
a seeded random sample instead, so a prompt arm can be scored on clips that were
not chosen for being wrong.

Three rules make the result usable:

* The sample is frozen at prepare time and recorded with the model's claims, so
  a later re-index cannot quietly change what is being scored.
* The labelling page never shows the model's answer. Revealing it mid-pass would
  teach the annotator the model's vocabulary and bias every later label.
* Scoring is reported per bucket, never as a single accuracy number. "Named a
  different gun" and "declined to answer" are different outcomes and the whole
  point of the v3.5 arm is to move clips from the first into the second.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import re

from pipeline_v2 import prompt
from pipeline_v2 import config, freeze, store
from pipeline_v2.http_media import serve_video
from pipeline_v2.jsonio import read_json, utc_now, write_json

ACTORS = ("recording_player", "teammate", "spectated_player", "no_event", "unclear")

# Rarity is not weapon identity: "Epic Hunting Rifle" and "hunting rifle" are the
# same answer, and an annotator will not type the tier.
_RARITY = re.compile(
    r"\b(common|uncommon|rare|epic|legendary|mythic|exotic)\b", re.IGNORECASE
)


def normalise_weapon(text: object) -> str:
    cleaned = _RARITY.sub(" ", str(text or ""))
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned.lower())
    return " ".join(cleaned.split())


# Weapon classes, for the case where the annotator typed a class rather than a
# name ("shotgun" for a Striker Pump Shotgun). Only what is needed to decide
# family-level agreement -- this is not a revived ontology and nothing outside
# scoring reads it.
#
# A hunting rifle is a SNIPER-class weapon, not an assault rifle. Substring
# matching alone would call "Warforged Assault Rifle" a match for a label of
# "rifle", and would miss that "sniper" and "Hunting Rifle" are the same answer.
# Longest phrases first so "hunting rifle" wins over a bare "rifle".
_CLASS_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sniper", ("hunting rifle", "marksman rifle", "sniper", "dmr")),
    ("shotgun", ("shotgun",)),
    # "Baba Yaga" carries no class word; confirmed a pistol by the annotator.
    ("pistol", ("pistol", "revolver", "dualies", "hand cannon", "baba yaga")),
    ("assault_rifle", ("assault rifle", "burst rifle", "smg", "submachine")),
)


def weapon_class(text: object) -> str | None:
    """The class a name or label belongs to, or None when it is ambiguous.

    A bare "rifle" is deliberately unclassified: it could be an assault rifle or
    a hunting rifle, and guessing would turn a real disagreement into a match.
    """
    name = normalise_weapon(text)
    for label, terms in _CLASS_TERMS:
        if any(term in name for term in terms):
            return label
    return None


def parse_weapons(text: object) -> list[str]:
    """Split an annotator's answer into separate weapons.

    A window can hold several kills, so "pistol, shotgun" is two answers rather
    than one weapon with a comma in its name.
    """
    parts = re.split(r"[,/;]| and | then ", str(text or ""))
    return [p.strip() for p in parts if p.strip()]


def asks_for_kill_weapons(variant: str) -> bool:
    """Did the prompt behind this variant ask for the kill_weapons list?"""
    from pipeline_v2 import schema

    return (variant or "").startswith(
        (schema.KILL_WEAPON_PROMPT_VERSION, schema.OWN_KILLS_PROMPT_VERSION)
    )


def claimed_weapons(payload: dict, variant: str) -> list[str]:
    """What a stored description claims killed someone, for its prompt version.

    Keyed on the VARIANT, never on which fields the payload happens to carry.
    `kill_weapons` is a pydantic field with a default factory, so `model_dump()`
    writes `[]` into every payload including v3.3 ones that were never asked for
    it -- reading that shape as an abstention scored a whole arm at 100%
    declined when the model had in fact answered in `weapon_name` every time.
    """
    if asks_for_kill_weapons(variant):
        # Here an empty list is the model genuinely declining.
        listed = payload.get("kill_weapons")
        listed = listed if isinstance(listed, list) else []
        return [str(w).strip() for w in listed if str(w).strip()]
    single = str(payload.get("weapon_name") or "").strip()
    return [single] if single else []


def compare_claims(claimed: list[str], actual: list[str]) -> str:
    """Bucket a set of claims against a set of answers, best match wins.

    Set-based because a window can hold several kills and neither side is ordered
    reliably, so agreeing on *a* weapon is the strongest claim this data supports.
    """
    truths = [a for a in actual if normalise_weapon(a) not in ("", "unknown")]
    if not truths:
        return "no_ground_truth"
    claims = [c for c in claimed if normalise_weapon(c) not in ("", "unknown")]
    if not claims:
        return "model_declined"
    ranking = ("exact", "family", "mismatch")
    best = "mismatch"
    for claim in claims:
        for truth in truths:
            bucket = compare_weapon(claim, truth)
            if ranking.index(bucket) < ranking.index(best):
                best = bucket
    return best


def compare_weapon(claimed: str, actual: str) -> str:
    """Bucket one weapon claim against the annotator's answer."""
    model, user = normalise_weapon(claimed), normalise_weapon(actual)
    if not user or user == "unknown":
        return "no_ground_truth"
    if not model or model == "unknown":
        return "model_declined"
    if model == user:
        return "exact"

    model_class, user_class = weapon_class(model), weapon_class(user)
    if model_class and user_class:
        # Both sides are classifiable, so the classes settle it. This overrides
        # substring: an assault rifle sharing the word "rifle" with a hunting
        # rifle is a different weapon, not a partial match.
        return "family" if model_class == user_class else "mismatch"
    if model in user or user in model:
        return "family"
    return "mismatch"


def prepare(out: Path, size: int, seed: int, variant: str | None = None) -> dict:
    """Freeze a seeded random sample of described clips."""
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        return read_json(plan_path)

    variant = variant or prompt.active_variant(config.load_config())
    conn = store.connect()
    try:
        clips = store.all_clips(conn, variant)
    finally:
        conn.close()
    if len(clips) < size:
        raise ValueError(f"{variant} has {len(clips)} clips, need {size}")

    sample = random.Random(seed).sample(sorted(clips, key=lambda c: c.clip_sha), size)
    plan = {
        "schema": "salience.groundtruth/1",
        "created_utc": utc_now(),
        "variant": variant,
        "seed": seed,
        "size": size,
        "selection": (
            f"random.Random({seed}).sample over all {len(clips)} clips described at "
            f"{variant}, ordered by clip_sha. No filtering on outcome, weapon, or "
            "whether the description looked wrong."
        ),
        "purpose": (
            "Held-out weapon and actor ground truth. Labels never enter ranking "
            "training and are not keep/skip preferences."
        ),
        "clips": [
            {
                "clip_sha": c.clip_sha,
                "path": str(c.path),
                "name": c.path.name,
                "event_sec": c.event_sec,
            }
            for c in sample
        ],
        # Pin what is being scored. A re-describe at the same version would
        # otherwise silently change the claims without changing the plan.
        "claims_sha": freeze.sha256_text(
            json.dumps(
                {c.clip_sha: c.payload.get("weapon_name", "") for c in sample},
                sort_keys=True,
            )
        ),
    }
    write_json(plan_path, plan)
    if not (out / "labels.json").exists():
        write_json(out / "labels.json", {"annotator": "user", "labels": {}})
    return plan


def validate_label(payload: dict, known: set[str]) -> tuple[str, dict]:
    if not isinstance(payload, dict):
        raise ValueError("expected an object")
    sha = payload.get("clip_sha")
    if sha not in known:
        raise ValueError("unknown clip")
    weapon = payload.get("weapon", "")
    notes = payload.get("notes", "")
    if not isinstance(weapon, str) or len(weapon) > 120:
        raise ValueError("weapon must be short text")
    if not isinstance(notes, str) or len(notes) > 2000:
        raise ValueError("notes too long")
    if payload.get("actor") not in ACTORS:
        raise ValueError(f"actor must be one of {ACTORS}")
    return sha, {
        "weapon": weapon.strip(),
        "actor": payload["actor"],
        "notes": notes.strip(),
        "saved_utc": utc_now(),
    }


def current_paths(plan: dict) -> dict[str, Path]:
    """Where the sampled clips live *now*.

    The plan freezes which clips are in the sample, not where the files sit. A
    `relink` after the freeze moves every path, and re-resolving from the store
    by clip_sha keeps the frozen sample valid instead of stranding it.
    """
    conn = store.connect()
    try:
        known = {c.clip_sha: c.path for c in store.all_clips(conn, plan["variant"])}
    finally:
        conn.close()
    return {
        c["clip_sha"]: known.get(c["clip_sha"], Path(c["path"])) for c in plan["clips"]
    }


def model_window(event_sec: float | None, duration_sec: float | None) -> tuple[float, float]:
    """The seconds the VLM was actually shown, matching `frames.window_frames`.

    Annotating a wider span than the model saw lets a label name a kill that was
    never in its input. The first pass seeked to `event - 4s` and never paused,
    so the annotator saw 9.2s against the model's 4.0s.
    """
    from pipeline_v2 import frames

    event = event_sec or 0.0
    start = max(0.1, event - frames.WINDOW_PRE_SEC)
    end = event + frames.WINDOW_POST_SEC
    if duration_sec:
        end = min(end, duration_sec - 0.1)
    return start, max(start, end)


def _handler(out: Path):
    plan = read_json(out / "plan.json")
    by_sha = {c["clip_sha"]: c for c in plan["clips"]}
    paths = current_paths(plan)
    order = [c["clip_sha"] for c in plan["clips"]]

    conn = store.connect()
    try:
        durations = {
            row["clip_sha"]: row["duration_sec"]
            for row in conn.execute("select clip_sha, duration_sec from clips")
        }
    finally:
        conn.close()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
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
            if self.path == "/api/next":
                labels = read_json(out / "labels.json")["labels"]
                remaining = [s for s in order if s not in labels]
                if not remaining:
                    self._json({"done": True, "labelled": len(labels)})
                    return
                clip = by_sha[remaining[0]]
                start, end = model_window(
                    clip.get("event_sec"), durations.get(clip["clip_sha"])
                )
                self._json(
                    {
                        "done": False,
                        **clip,
                        "window_start": start,
                        "window_end": end,
                        "labelled": len(labels),
                        "total": len(order),
                    }
                )
                return
            if self.path.startswith("/media/"):
                path = paths.get(self.path[len("/media/") :])
                if path is None or not path.exists():
                    self.send_error(404)
                    return
                serve_video(self, path)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/label":
                self.send_error(404)
                return
            if self.headers.get("Origin") not in (
                None,
                f"http://{self.headers.get('Host')}",
            ):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8192:
                    raise ValueError("invalid request length")
                sha, record = validate_label(
                    json.loads(self.rfile.read(length)), set(by_sha)
                )
                stored = read_json(out / "labels.json")
                stored["labels"][sha] = record
                write_json(out / "labels.json", stored)
            except (ValueError, TypeError, OSError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json({"ok": True})

    return Handler


def serve(out: Path, port: int = 8780) -> int:
    plan = read_json(out / "plan.json")
    labelled = len(read_json(out / "labels.json")["labels"])

    # Fail loudly here rather than as a 404 the browser renders as a dead
    # player. The first run of this page served a sample whose footage was on
    # another machine, and the only symptom was video that would not start.
    missing = [sha for sha, path in current_paths(plan).items() if not path.exists()]
    if missing:
        print(f"{len(missing)} of {plan['size']} clips have no video file.")
        print("Point the store at the footage first:")
        print("  python -m pipeline_v2 relink <folder> --apply")
        return 1

    print(f"labelling {plan['size']} clips at {plan['variant']} ({labelled} done)")
    print(f"open http://127.0.0.1:{port}/   (Ctrl+C to stop)")
    print("The model's answer stays hidden until you run `report`.")
    with ThreadingHTTPServer(("127.0.0.1", port), _handler(out)) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def report(out: Path, variant: str | None = None) -> dict:
    """Score a variant's weapon claims against the frozen labels."""
    plan = read_json(out / "plan.json")
    labels = read_json(out / "labels.json")["labels"]
    variant = variant or plan["variant"]

    conn = store.connect()
    try:
        payloads = {c.clip_sha: c.payload for c in store.all_clips(conn, variant)}
    finally:
        conn.close()

    rows, buckets, actors, missing = [], {}, {}, []
    for clip in plan["clips"]:
        sha = clip["clip_sha"]
        label = labels.get(sha)
        if label is None:
            continue
        payload = payloads.get(sha)
        if payload is None:
            missing.append(clip["name"])
            continue
        claimed = claimed_weapons(payload, variant)
        truths = parse_weapons(label["weapon"])
        bucket = compare_claims(claimed, truths)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        actors[label["actor"]] = actors.get(label["actor"], 0) + 1
        rows.append(
            {
                "name": clip["name"],
                "actor": label["actor"],
                "claimed": ", ".join(claimed) if claimed else "",
                "claimed_count": len(claimed),
                "actual": label["weapon"],
                "actual_count": len(truths),
                "bucket": bucket,
                "notes": label["notes"],
            }
        )

    # A self-consistency check needing no labels: if the recording player was the
    # one eliminated then they killed nobody, so the list must be empty. v3.5
    # violated this on 3 of 5 death clips because it asked about any kill in
    # frame, which includes the player's own.
    contradictions = []
    for clip in plan["clips"]:
        payload = payloads.get(clip["clip_sha"])
        if payload is None or payload.get("outcome") != "death":
            continue
        named = [
            w
            for w in claimed_weapons(payload, variant)
            if normalise_weapon(w) != "unknown"
        ]
        if named:
            contradictions.append({"name": clip["name"], "claimed": named})

    scored = sum(v for k, v in buckets.items() if k != "no_ground_truth")
    result = {
        "variant": variant,
        "sample_size": plan["size"],
        "labelled": len(labels),
        "scored": scored,
        "weapon_buckets": buckets,
        "exact_rate": (buckets.get("exact", 0) / scored) if scored else None,
        "wrong_rate": (buckets.get("mismatch", 0) / scored) if scored else None,
        "declined_rate": (buckets.get("model_declined", 0) / scored) if scored else None,
        "deaths_claiming_a_kill": contradictions,
        "actor_ground_truth": actors,
        "actor_note": (
            "v3.3 has no actor field, so actor cannot be scored against it. These "
            "counts establish prevalence and are the reference for later arms."
        ),
        "missing_from_variant": missing,
        "limits": (
            "One session, one player, one season. A seeded random sample supports "
            "a rate for this library; it is not a generalization estimate. "
            "Annotators watched the whole clip; the model saw roughly 4s of a ~22s "
            "file (about 19%), so a label may name a kill outside the frames the "
            "model was given. Scoring is set-based to blunt that, but a clip whose "
            "kills used different weapons can still score as a mismatch when the "
            "model correctly described its own window."
        ),
        "rows": rows,
    }
    write_json(out / f"report-{variant}.json", result)
    return result


def variants_covering(plan: dict) -> list[str]:
    """Every stored prompt variant that describes at least one sampled clip."""
    shas = tuple(c["clip_sha"] for c in plan["clips"])
    conn = store.connect()
    try:
        rows = conn.execute(
            "select distinct prompt_ver from descriptions where clip_sha in "
            f"({','.join('?' * len(shas))}) order by prompt_ver",
            shas,
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def compare(out: Path, variants: list[str] | None = None) -> list[dict]:
    """Score every arm against one set of labels.

    Arms rarely cover the same clips -- a bake-off describes ten, a full run
    fifty -- so `scored` is reported per arm and the rate is over what that arm
    actually answered. Comparing rates across arms with different coverage is
    only fair when their `scored` counts match.
    """
    plan = read_json(out / "plan.json")
    return [report(out, v) for v in (variants or variants_covering(plan))]


def _print_comparison(results: list[dict]) -> None:
    width = max([len("variant")] + [len(r["variant"]) for r in results]) + 2
    print(f"\n{'variant':{width}}{'scored':>7}{'exact':>7}{'family':>8}"
          f"{'wrong':>7}{'declined':>10}{'wrong %':>9}")
    for r in sorted(results, key=lambda x: (x["wrong_rate"] is None, x["wrong_rate"])):
        b = r["weapon_buckets"]
        rate = f"{r['wrong_rate']:.1%}" if r["wrong_rate"] is not None else "-"
        print(
            f"{r['variant']:{width}}{r['scored']:>7}{b.get('exact', 0):>7}"
            f"{b.get('family', 0):>8}{b.get('mismatch', 0):>7}"
            f"{b.get('model_declined', 0):>10}{rate:>9}"
        )
    print("\nA rate is only comparable between arms with the same `scored` count.")


def _print(result: dict) -> None:
    print(f"\nvariant {result['variant']}   "
          f"labelled {result['labelled']}/{result['sample_size']}   "
          f"scored {result['scored']}")
    if not result["scored"]:
        print("\nnothing scored yet -- run `serve` and label some clips first")
        return
    print("\nweapon claim vs ground truth")
    for bucket in ("exact", "family", "mismatch", "model_declined", "no_ground_truth"):
        count = result["weapon_buckets"].get(bucket, 0)
        if count:
            share = count / result["scored"] if bucket != "no_ground_truth" else None
            suffix = f"{share:>8.1%}" if share is not None else "       -"
            print(f"  {bucket:18}{count:>5}{suffix}")
    contradictions = result.get("deaths_claiming_a_kill") or []
    if contradictions:
        print(
            f"\n{len(contradictions)} clip(s) say the player died yet still name a "
            "kill weapon (no labels needed to call this wrong):"
        )
        for row in contradictions[:5]:
            print(f"  {row['claimed']}  {row['name']}")

    print("\nactor prevalence (ground truth, not a score)")
    for actor, count in sorted(result["actor_ground_truth"].items()):
        print(f"  {actor:18}{count:>5}")
    wrong = [r for r in result["rows"] if r["bucket"] == "mismatch"]
    if wrong:
        print(f"\n{len(wrong)} wrong weapon claim(s):")
        for row in wrong[:20]:
            print(f"  claimed {row['claimed']!r} -- actual {row['actual']!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="freeze a seeded random sample")
    prep.add_argument("out", type=Path)
    prep.add_argument("--size", type=int, default=50)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--variant", default=None)

    page = sub.add_parser("serve", help="label the sample in a browser")
    page.add_argument("out", type=Path)
    page.add_argument("--port", type=int, default=8780)

    score = sub.add_parser("report", help="score a variant against the labels")
    score.add_argument("out", type=Path)
    score.add_argument("--variant", default=None)

    versus = sub.add_parser("compare", help="score every arm side by side")
    versus.add_argument("out", type=Path)
    versus.add_argument(
        "--variants", nargs="*", default=None, help="default: every arm found"
    )

    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        plan = prepare(out, args.size, args.seed, args.variant)
        print(f"froze {plan['size']} clips at {plan['variant']} (seed {plan['seed']})")
        print(f"next: python -m pipeline_v2.groundtruth serve {args.out}")
    elif args.command == "serve":
        return serve(out, args.port)
    elif args.command == "compare":
        _print_comparison(compare(out, args.variants))
    else:
        _print(report(out, args.variant))
    return 0


PAGE = """<!doctype html>
<meta charset="utf-8"><title>Salience ground truth</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:15px/1.55 system-ui,sans-serif; background:#0d0f13; color:#e6e8eb;
         display:flex; flex-direction:column; align-items:center; gap:12px; padding:22px; }
  video { width:min(94vw,900px); border-radius:10px; background:#000; }
  .panel { width:min(94vw,900px); display:flex; flex-direction:column; gap:10px; }
  label { color:#94a0ae; font-size:13px; display:flex; flex-direction:column; gap:4px; }
  input, select, textarea, button { font:inherit; padding:9px 11px; border-radius:8px;
         border:1px solid #2a313a; background:#161b22; color:#e6e8eb; }
  button { cursor:pointer; } button:hover { background:#1e242c; }
  #progress, #status { color:#94a0ae; font-size:13px; }
  #status { min-height:18px; } .err { color:#f3a0a0 !important; }
  #hint { color:#6f7c8c; font-size:12px; max-width:min(94vw,900px); }
  #done { font-size:19px; padding:40px; }
</style>
<div id="progress"></div>
<video id="v" controls></video>
<div class="panel">
  <p id="hint">Judge <b>only the window below</b> &mdash; it is exactly the frames
  the model was given. Playback stops at its end. List every weapon that killed or
  knocked someone <i>in that window</i>, comma separated; type <code>unknown</code>
  if nothing died or you cannot tell. The model's answer is hidden on purpose.</p>
  <div><button id="replay" type="button">Replay window</button>
       <button id="context" type="button">Play with context</button>
       <span id="bounds"></span></div>
  <label>Weapon(s) that killed or knocked, in this window
    <input id="weapon" autocomplete="off" placeholder="e.g. Drum Gun, or: pistol, shotgun"></label>
  <label>Who got it?
    <select id="actor">
      <option value="recording_player">Me (the recording player)</option>
      <option value="teammate">A teammate</option>
      <option value="spectated_player">A player I was spectating</option>
      <option value="no_event">Nothing was eliminated</option>
      <option value="unclear">Cannot tell</option>
    </select></label>
  <label>Notes (optional)<textarea id="notes" rows="2"></textarea></label>
  <button id="save">Save and next</button>
  <div id="status" role="status"></div>
</div>
<script>
let current = null, saving = false, playbackEnd = 0;
const $ = id => document.getElementById(id);
function status(text, bad) { $('status').textContent = text; $('status').className = bad ? 'err' : ''; }
async function load() {
  const d = await (await fetch('/api/next')).json();
  if (d.done) {
    document.body.innerHTML = '<div id="done">All ' + d.labelled +
      ' clips labelled. Run <code>python -m pipeline_v2.groundtruth report OUT</code>.</div>';
    return;
  }
  current = d;
  $('progress').textContent = d.labelled + ' / ' + d.total + ' labelled \\u00b7 ' + d.name;
  const v = $('v');
  v.src = '/media/' + d.clip_sha;
  playbackEnd = d.window_end;
  v.onloadedmetadata = () => { v.currentTime = d.window_start; };
  $('bounds').textContent = 'model window ' + d.window_start.toFixed(1) +
    '–' + d.window_end.toFixed(1) + 's';
  $('weapon').value = ''; $('notes').value = ''; $('actor').value = 'recording_player';
  $('weapon').focus();
  status('');
}
async function save() {
  if (!current || saving) return;
  if (!$('weapon').value.trim()) { status('Type a weapon, or "unknown".', true); return; }
  saving = true;
  try {
    const r = await fetch('/api/label', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({clip_sha: current.clip_sha, weapon: $('weapon').value,
                            actor: $('actor').value, notes: $('notes').value})});
    if (!r.ok) throw new Error((await r.json()).error || 'save failed');
    await load();
  } catch (e) { status(e.message, true); } finally { saving = false; }
}
$('save').onclick = save;
// Stop at the window edge so what is judged is what the model was given.
$('v').ontimeupdate = () => { if (playbackEnd && $('v').currentTime >= playbackEnd && !$('v').paused) $('v').pause(); };
$('replay').onclick = () => { playbackEnd = current.window_end; $('v').currentTime = current.window_start; $('v').play(); };
$('context').onclick = () => { playbackEnd = 0; $('v').currentTime = Math.max(0, current.window_start - 4); $('v').play(); };
addEventListener('keydown', e => { if (e.key === 'Enter' && e.target.id !== 'notes') save(); });
load();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
