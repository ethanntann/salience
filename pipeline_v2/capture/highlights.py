"""From a finished recording to the handful of clips worth keeping.

The capture side used to stop at "here is 1.8 GB of footage and a list of
moments a key-press counter flagged". Every step after that -- cutting,
describing, ranking -- was a command someone had to know to run, and the last
one could not run at all on the recorder's own footage. This is that chain as
one call, run automatically when a recording stops:

    segments -> coverage.jsonl -> windows/ -> mic transcribed -> described
             + stored -> ranked -> highlights/

**Coverage, not triggers, decides what is looked at.** The first version cut
only where a detector fired (`triggers.jsonl -> clips/`). Session 1 showed the
gate dropped most of what mattered: the input-burst trigger overlapped 5 of the
14 moments the player kept. So by default every 8 seconds of the session is
described and taste picks; `coverage=False` keeps the trigger-only path for
comparison. At deepseek-v4p1-flash prices that is about $0.002 per window, ~$0.45 for
half an hour.

**Ranking by taste needs labels on footage like this.** The ranker learns only
from labelled clips in the same storage variant, and native-recorder clips live
in their own (`...-window@<model>`). Until some are labelled -- `review
--variant <that variant>` -- there is nothing of the player's to rank by. Rather
than pretend, the pipeline then falls back to an explicit, non-taste ordering
(what the vision model saw happen, then trigger confidence) and says so in the
manifest under `basis`. A highlights folder that claims taste it does not have
is worse than one that admits it.

Raw segments are kept unless `prune_raw` is asked for. Deleting them is the
point of a clipping product, but it is irreversible, and it removes the ability
to re-cut with a different window or re-run a control.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline_v2.capture import clips as clips_mod
from pipeline_v2.capture import reactions
from pipeline_v2.capture import gameplay as gameplay_mod

# What the vision model saw, as a fallback order when there is no taste model.
# Not a judgement of quality: a knock the player found boring still outranks a
# rotation they loved. That is exactly why it is labelled as a fallback.
_OUTCOME_ORDER = {"elimination": 2, "knock": 1}


def event_tier(clip) -> int:
    """What happened, coarsely: a win or an elimination, then knocks, the rest.

    Ranking goes by this first and by taste within a tier. Measured out of
    sample (taste trained on session 1, scored on session 2's 320 ratings),
    taste alone agreed with the player 86.9% of the time, the tier alone
    92.3%, the tier then taste 96.5%. The model files the Victory Royale banner
    under `other`/`no_event`, so a win is read from the summary.

    A win used to be a tier above kills. The player kept 6 of 17 win windows,
    the same rate as kills (42 of 121), and none of the 6 on sessions 5 and 6.
    Each session ranked by the other four (2026-10-22): wins first put 21 of
    32 keeps in the top 10, wins among the kills 24, better or equal on every
    session.
    """
    if is_victory(clip):
        return _OUTCOME_ORDER["elimination"]
    return _OUTCOME_ORDER.get((clip.payload or {}).get("outcome"), 0)


def is_victory(clip) -> bool:
    return "victory royale" in (clip.summary or "").lower()

# What an exported highlight adds around its 8s coverage window: the approach
# matters to a viewer even though the model only needed the window itself.
HIGHLIGHT_LEAD = 8.0
HIGHLIGHT_LAG = 2.0
EXPORT_WORKERS = 5

# Every highlight is also made as a 9:16 reel for TikTok, Shorts and Reels: a
# centre crop, because the crosshair (and so the action) sits at the centre,
# scaled to 1080x1920, with the game and mic tracks mixed into the one track
# a phone plays. NVENC, so 4.6s for a 42s highlight.
REEL_DIR = "reels"
REEL_SIZE = (1080, 1920)


# The game and the mic as one track, at the levels they were recorded at; the
# limiter only catches a shout over gunfire summing past full scale.
MIXED_TITLE = "game + mic"
_MIX = "amix=inputs={n}:normalize=0,alimiter=limit=0.97"


def _audio_tracks(source: Path, runner=subprocess.run) -> list[str]:
    """Each audio track's title ("" when untitled), in order."""
    probe = runner(["ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=index:stream_tags=title", "-of", "csv=p=0",
                    str(source)], capture_output=True, text=True)
    if probe.returncode != 0:
        return []
    return [(line.split(",", 1) + [""])[1].strip() for line in (probe.stdout or "").splitlines()
            if line.strip()]


def mix_tracks(path: Path, runner=subprocess.run) -> bool:
    """Put the game and the mic, mixed, first and default in an exported highlight.

    The recording keeps them as separate tracks, the game's the default, and a
    player plays the default alone: the player heard none of their own mic in
    their highlights (2026-10-23). The separate tracks stay after the mix, for
    editing. Video is copied; only the mix is encoded. False leaves `path` as it was.
    """
    tracks = _audio_tracks(path, runner)
    if len(tracks) < 2 or tracks[0] == MIXED_TITLE:
        return False
    inputs = "".join(f"[0:a:{i}]" for i in range(len(tracks)))
    temporary = path.with_name(path.stem + ".mixing" + path.suffix)
    done = runner(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path),
                   "-filter_complex", f"{inputs}{_MIX.format(n=len(tracks))}[mix]",
                   "-map", "0:v", "-map", "[mix]", "-map", "0:a",
                   "-c", "copy", "-c:a:0", "aac", "-b:a:0", "192k",
                   "-metadata:s:a:0", f"title={MIXED_TITLE}",
                   "-metadata:s:a:1", "title=game", "-metadata:s:a:2", "title=mic",
                   "-disposition:a", "0", "-disposition:a:0", "default",
                   str(temporary)], capture_output=True)
    if done.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(path)
    return True


def make_reel(source: Path, destination: Path, runner=subprocess.run) -> bool:
    """A 9:16 copy of one exported highlight. False if it could not be made."""
    tracks = _audio_tracks(source, runner)
    width, height = REEL_SIZE
    video = (f"[0:v]crop=trunc(ih*{width}/{height}/2)*2:ih,"
             f"scale={width}:{height}:flags=lanczos,setsar=1[v]")
    if len(tracks) > 1 and tracks[0] != MIXED_TITLE:
        inputs = "".join(f"[0:a:{i}]" for i in range(len(tracks)))
        graph = f"{video};{inputs}{_MIX.format(n=len(tracks))}[a]"
        audio = ["-map", "[a]"]
    else:
        graph, audio = video, (["-map", "0:a:0"] if tracks else [])
    destination.parent.mkdir(parents=True, exist_ok=True)
    done = runner(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
                   "-filter_complex", graph, "-map", "[v]", *audio,
                   "-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "12M",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                   str(destination)], capture_output=True)
    return done.returncode == 0 and destination.is_file()


def reels(session: Path, reel_fn=None) -> int:
    """`_reels`, with the manifest to itself: the rating page adds highlights too."""
    with _HIGHLIGHTS_LOCK:
        return _reels(session, reel_fn)


def _reels(session: Path, reel_fn=None) -> int:
    """A 9:16 reel of every highlight `run` kept; how many were made.

    Its own step, run after the highlights folder has opened: 10 reels take
    11s on the GPU encoder (its own limit, measured: parallel or not, GPU
    crop or CPU), which the player should not wait on to see their clips.
    """
    from concurrent.futures import ThreadPoolExecutor

    from pipeline_v2.jsonio import write_json

    out = session / "highlights"
    try:
        manifest = json.loads((out / "highlights.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    kept = [h for h in manifest.get("highlights", []) if (out / h.get("file", "")).is_file()
            and h.get("file")]

    def one(highlight) -> str:
        reel = Path(highlight["file"]).with_suffix(".mp4").name
        made = (reel_fn or make_reel)(out / highlight["file"], out / REEL_DIR / reel)
        return reel if made else ""

    with ThreadPoolExecutor(EXPORT_WORKERS) as pool:
        for highlight, reel in zip(kept, pool.map(one, kept)):
            highlight["reel"] = reel
    write_json(out / "highlights.json", manifest)
    return sum(1 for h in kept if h["reel"])


def release_windows(session: Path) -> int:
    """Delete the copies of the session's described windows; how many went.

    Run once the session's highlights and reels are made: from then on a window
    is only ever watched again on the rating page, which re-cuts it from the raw
    footage (`clips.recut_window`). Its mic level is measured first, being the
    one thing ranking reads from the copy itself. A copy whose raw footage is
    gone, or that the rating page has open, is kept.
    """
    from types import SimpleNamespace

    from pipeline_v2 import cache, store

    clip_dir = session / "windows"
    copies = sorted(clip_dir.glob("*.mkv")) if clip_dir.is_dir() else []
    if not copies:
        return 0
    conn = store.connect()
    try:
        described = {row[0] for row in conn.execute("select clip_sha from descriptions")}
    finally:
        conn.close()
    hashed = [(clip, cache.clip_sha(clip)) for clip in copies]
    done = [(clip, sha) for clip, sha in hashed if sha in described]
    voice([SimpleNamespace(path=str(clip), clip_sha=sha) for clip, sha in done])
    freed = 0
    for clip, sha in done:
        try:
            freed += clips_mod.release_window(clip, sha)
        except OSError:
            pass  # open in the rating page; the next session's release takes it
    return freed


# A window the player kept on the rating page becomes a highlight there and
# then, the same way the ranked ones were made (lead-in, the rest of its fight,
# game and mic mixed, a reel), so a moment the ranking missed is one key away.
RATED_TRIGGER = "rating"
_HIGHLIGHTS_LOCK = threading.Lock()  # one manifest writer at a time


def _manifest(out: Path) -> dict | None:
    try:
        return json.loads((out / "highlights.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def add_kept(clip_sha: str, reel_fn=None) -> str | None:
    """Export a window kept on the rating page into its session's highlights.

    The new file's name, or None: not a recorded session's window, already a
    highlight, or not cuttable.
    """
    from pipeline_v2 import store
    from pipeline_v2.jsonio import write_json

    with _HIGHLIGHTS_LOCK:
        conn = store.connect()
        try:
            row = conn.execute("select path from clips where clip_sha = ?", (clip_sha,)).fetchone()
            if row is None:
                return None
            path = Path(row[0])
            session = path.parent.parent
            _, inside = _session_clips(conn, path.parent)
            said = store.transcripts(conn).get(clip_sha, "")
        finally:
            conn.close()
        clip = next((c for c in inside if c.clip_sha == clip_sha), None)
        if clip is None or not (session / "segments.csv").is_file():
            return None
        out = session / "highlights"
        manifest = _manifest(out) or {"session": str(session), "highlights": []}
        entries = manifest.setdefault("highlights", [])
        if any(h.get("clip") == path.name for h in entries):
            return None
        position = max((int(h.get("rank") or 0) for h in entries), default=0) + 1
        name = f"{position:02d}-{path.name}"
        destination = out / name
        span = fight_extent(clip, window_index([(0.0, c) for c in inside]),
                            (HIGHLIGHT_LEAD, HIGHLIGHT_LAG))
        if not _export(session, clip, destination, span):
            out.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                shutil.copy2(path, destination)
            elif clips_mod.recut_window(path, destination) is None:
                return None
        mix_tracks(destination)
        reel = Path(name).with_suffix(".mp4").name
        made = (reel_fn or make_reel)(destination, out / REEL_DIR / reel)
        payload = clip.payload or {}
        entries.append(asdict(Highlight(
            rank=position, clip=path.name, score=0.0, summary=clip.summary or "",
            outcome=str(payload.get("outcome")), trigger=RATED_TRIGGER, said=said,
            file=name, reel=reel if made else "")))
        write_json(out / "highlights.json", manifest)
        return name


def drop_kept(clip_sha: str) -> bool:
    """Undo `add_kept` when the player re-rates that window a skip. Highlights
    the ranking chose are the player's to delete, never this."""
    from pipeline_v2 import store
    from pipeline_v2.jsonio import write_json

    with _HIGHLIGHTS_LOCK:
        conn = store.connect()
        try:
            row = conn.execute("select path from clips where clip_sha = ?", (clip_sha,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return False
        path = Path(row[0])
        out = path.parent.parent / "highlights"
        manifest = _manifest(out)
        if manifest is None:
            return False
        entries = manifest.get("highlights", [])
        gone = [h for h in entries if h.get("clip") == path.name
                and h.get("trigger") == RATED_TRIGGER]
        if not gone:
            return False
        for h in gone:
            (out / h["file"]).unlink(missing_ok=True)
            if h.get("reel"):
                (out / REEL_DIR / h["reel"]).unlink(missing_ok=True)
        manifest["highlights"] = [h for h in entries if h not in gone]
        write_json(out / "highlights.json", manifest)
        return True


def _sidecar(clip_path) -> dict:
    """The trigger that produced a clip. Not in the store: `index` keeps only the
    timing facts, so the detector's name and confidence live beside the file."""
    import json

    try:
        return json.loads(Path(clip_path).with_suffix(".capture.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class Highlight:
    rank: int
    clip: str
    score: float
    summary: str
    outcome: str
    trigger: str
    said: str = ""
    asked: bool = False  # the player said "clip that" for it
    laughed: bool = False  # laughter in it or just after, on the mic or the PC's audio
    file: str = ""  # its name in highlights/
    reel: str = ""  # its 9:16 copy in reels/, once `reels` has made it


def describe(clip_dir: Path, max_calls: int) -> None:
    """Describe, store and embed through the same path `index` uses.

    Driven rather than reimplemented: that path carries the per-clip commits,
    the unparsed-response handling and the pacing that keeps Cloudflare from
    blocking a batch, all learned the hard way.
    """
    from pipeline_v2 import cli

    cli.cmd_index(argparse.Namespace(
        target=str(clip_dir), from_plan=None, restore_stored=False,
        split_fights=False, from_feedback=None, limit=None,
        max_calls=max_calls, encoder=None, only_new=True,
    ))


def find_gameplay(session: Path) -> dict:
    """When the player was playing, from Fortnite's log (`gameplay`)."""
    segments = clips_mod.read_segments(session)
    if not segments:
        return {"spans": None, "why": "no footage"}
    return gameplay_mod.spans(session, max(s.end for s in segments))


def listen(session: Path, clip_dir: Path) -> int:
    """Transcribe the mic and attach what was said to each clip (`voice`).

    Optional: without the speech model installed (`.[audio]`) the session is ranked on what
    was seen alone, which is all a player who never talks loses anyway.
    """
    from pipeline_v2.capture import voice

    print("  speech: transcribing what the mic heard...", flush=True)
    try:
        return voice.attach(session, window_files(clip_dir))
    except ImportError:
        print("  speech: skipped (pip install -e .[audio] to rank by what was said)")
        return 0


def window_files(clip_dir: Path) -> list[Path]:
    """Every window of a folder: copies on disk and released ones (sidecar only)."""
    files = {p for p in clip_dir.glob("*") if p.suffix in (".mkv", ".mp4")}
    files |= {clip for sidecar in clip_dir.glob("*.capture.json")
              if clips_mod.released(clip := sidecar.with_name(
                  sidecar.name[:-len(".capture.json")] + ".mkv"))}
    return sorted(files)


def hear_laughs(session: Path) -> list:
    """Laughter on the mic and the PC's audio (`reactions`). Never in the way of
    the highlights: without the model, or if it fails, there is none."""
    print("  laughter: listening to the mic and the PC's audio...", flush=True)
    try:
        return reactions.listen(session)
    except ImportError:
        return []
    except Exception as exc:  # noqa: BLE001 -- highlights are made without it
        print(f"  laughter: skipped ({type(exc).__name__}: {exc})", flush=True)
        return []


def undescribed(clip_dir: Path) -> int:
    """How many clips in `clip_dir` have content with no stored description."""
    from pipeline_v2 import cache, store

    files = [p for p in clip_dir.glob("*") if p.suffix in (".mkv", ".mp4")]
    conn = store.connect()
    try:
        described = {row[0] for row in conn.execute("select clip_sha from descriptions")}
    finally:
        conn.close()
    return sum(cache.clip_sha(p) not in described for p in files)


def unembedded(clip_dir: Path) -> int:
    """How many of this folder's described clips have no embedding yet.

    Non-zero after a live session (`live`): it stores descriptions as they
    arrive and leaves embedding to this run, which needs the whole corpus.
    """
    from pipeline_v2 import store

    conn = store.connect()
    try:
        return conn.execute(
            "select count(*) from descriptions d join clips c using (clip_sha)"
            " left join embeddings e on e.clip_sha = d.clip_sha"
            " and e.prompt_ver = d.prompt_ver"
            " where c.path like ? and e.clip_sha is null",
            (str(clip_dir) + "%",),
        ).fetchone()[0]
    finally:
        conn.close()


def reembed(clip_dir: Path) -> None:
    """Rebuild embeddings when only the transcripts changed (nothing re-described)."""
    from pipeline_v2 import cli, store

    conn = store.connect()
    try:
        variant, _ = _session_clips(conn, clip_dir)
    finally:
        conn.close()
    if variant:
        cli._rebuild_embeddings(None, variant)


def _configured_variant() -> str | None:
    """The variant a capture window described with the current config lands in."""
    from pipeline_v2 import config, prompt
    from pipeline_v2.facts import ClipFacts

    try:
        cfg = config.load_config()
    except Exception:  # noqa: BLE001 - a missing config falls back to the majority
        return None
    return prompt.storage_variant(
        cfg.prompt_version, ClipFacts(event_source="capture_window"), cfg.model)


def _session_clips(conn, clip_dir: Path) -> tuple[str | None, list]:
    """This session's stored clips and the variant they were described under.

    The variant is read back from the store rather than recomputed, so it cannot
    drift from whatever `index` actually wrote -- a drift of exactly this kind
    (`-window` on writes, not on reads) is what made these clips invisible.
    """
    from pipeline_v2 import store

    root = str(clip_dir.resolve())
    rows = conn.execute(
        "select d.prompt_ver, count(*) from descriptions d "
        "join clips c on c.clip_sha = d.clip_sha "
        "where c.path like ? group by 1 order by 2 desc",
        (root + "%",),
    ).fetchall()
    if not rows:
        return None, []
    # A folder described by several models or prompts (a bake-off) holds one
    # variant per arm; the configured one is what the product ranks by.
    configured = _configured_variant()
    variant = configured if configured in {r[0] for r in rows} else rows[0][0]
    # A re-cut (a changed window) writes new content to the same filename, so
    # the store can hold two clips for one path. Only the newest describes the
    # file now on disk; the older row's description is of footage that is gone.
    newest = {
        row[0]: row[1] for row in conn.execute(
            "select path, clip_sha from clips where path like ? "
            "order by described_at", (root + "%",))
    }
    # `newest` holds only this folder's paths, so membership places a clip
    # here. Resolving every stored path on disk to check was 10,000 disk
    # lookups a library load in the app (1.6s of its 2.6s, 2026-10-24).
    inside = [c for c in store.all_clips(conn, variant) if newest.get(str(c.path)) == c.clip_sha]
    return variant, inside


# How much the player's hands count beside taste, within an event tier. Kills
# the player kept came with more key presses (building, editing, swapping) than
# kills they skipped: ordering kills, taste alone 37% / 69% on sessions 2 / 3,
# with this 48% / 72%. Both scores are standardised within the session.
ACTIVITY_WEIGHT = 1.0
# The player's voice, as loudness rather than words: reactions are mostly not
# words, and speech models drop them. Session 3: kept windows were louder on the mic
# 86% of the time, kept kills over skipped ones 70%. Half weight, because on
# session 2 it carried nothing within kills (47%).
VOICE_WEIGHT = 0.5
MIC_LEVEL_KEY = "mic_p95_db"
LEVEL_WORKERS = 8
# What kind of kill, as the player put it: they keep sniper kills, headshots,
# no-scopes and piece control, and skip sprays (an SMG, rifle or pistol), bots
# and finishing someone already down. Taste learns this from the summaries too
# slowly; this says it outright, within an event tier. Each session ranked by
# the other three (v3.11): 17 of 27 keeps in the top 10 without, 18 with, and
# a kept kill above a skipped one 64% of the time without, 75% with.
# (`kill_traits` says why the words are now averaged with the kill fields.)
KILL_TRAIT_WEIGHT = 1.0
_SNIPER = re.compile(r"sniper|hunting rifle|reaper|no-scope")
_SPRAY = re.compile(r"assault rifle|pistol")
_NOT_SPRAY = re.compile(r"shotgun|sniper|hunting rifle")


def kill_traits(clip) -> float:
    """What kind of kill, from the words and the kill fields averaged.

    The words alone swung with the describer's phrasing. Re-describing 51
    windows (only the note of what was said changed) turned a kept "shotguns
    X through a build... headshot" into "finishes a downed opponent", and
    two kept kills dropped out of the top 10: 24 of 32 keeps before, 22
    after. The fields alone held 24 on both, and the average 24 on both with
    the best kill ordering (a kept kill above a skipped one 78% and 77%,
    against the words' 77% and 74%). Each session ranked by the other four.
    """
    return (_trait_words(clip) + _trait_fields(clip)) / 2


def _trait_words(clip) -> float:
    """+1 for each trait the player keeps, -1 or -2 for each they skip, as said."""
    payload = clip.payload or {}
    text = " ".join([clip.summary or "", *(payload.get("kill_weapons") or [])]).lower()
    tags = set(payload.get("mechanics") or [])
    score = float("headshot" in text or "headshot" in tags)
    score += float(bool(_SNIPER.search(text)))
    score += float(bool({"piece_control", "box_fight"} & tags) or "piece control" in text)
    score -= float(bool(_SPRAY.search(text)) and not _NOT_SPRAY.search(text))
    score -= float("finish_downed" in tags or "downed" in text)
    score -= 2.0 * (payload.get("opponent_kind") in ("bot", "npc"))
    return score


def _trait_fields(clip) -> float:
    """The same traits from v3.11's kill fields (zero on older descriptions)."""
    payload = clip.payload or {}
    weapon = str(payload.get("kill_weapon_class") or "none")
    damage = payload.get("kill_damage") or 0
    score = float(payload.get("kill_headshot") == "yes")
    score += float(weapon == "sniper")
    score += float(payload.get("boxed_opponent") == "yes")
    # A one-pump. The damage is only trusted on a shotgun's single hit: on a
    # pistol, SMG or rifle the number on screen is several hits stacked.
    score += float(weapon == "shotgun" and payload.get("kill_hits") == 1 and damage >= 150)
    score -= float(weapon in ("assault_rifle", "pistol"))  # SMG kills were kept 10 of 23
    score -= 2.0 * (payload.get("opponent_kind") in ("bot", "npc"))
    return score


def _step(what: str) -> None:
    """A timestamped line in the log, so a slow wait says where it went."""
    import time

    print(f"  [{time.strftime('%H:%M:%S')}] {what}", flush=True)


def voice(clips: list) -> dict[str, float]:
    """The mic's loudness in each clip, cached in its sidecar."""
    from pipeline_v2.capture import levels, voice as voice_mod
    from pipeline_v2.jsonio import write_json

    from concurrent.futures import ThreadPoolExecutor

    def measured(clip) -> dict:
        record = _sidecar(clip.path)
        if MIC_LEVEL_KEY not in record and record and Path(clip.path).is_file():
            record[MIC_LEVEL_KEY] = levels.loudness(Path(clip.path), voice_mod.MIC_TRACK)
            try:
                write_json(Path(clip.path).with_suffix(".capture.json"), record)
            except OSError:
                pass
        return record

    found: dict[str, float] = {}
    # Side by side: one ffmpeg each, and one after another they were 18s of
    # session 5's wait (155 windows).
    with ThreadPoolExecutor(LEVEL_WORKERS) as pool:
        for clip, record in zip(clips, pool.map(measured, clips)):
            if record.get(MIC_LEVEL_KEY) is not None:
                found[clip.clip_sha] = float(record[MIC_LEVEL_KEY])
    # A silent track says nothing about any window; leave it out entirely.
    return found if len(set(found.values())) > 1 else {}


def activity(session: Path, clips: list) -> dict[str, float]:
    """Key presses during each clip's window, from the recorder's input log."""
    from pipeline_v2.capture.input_activity import ActivitySampler

    try:
        samples = ActivitySampler.read_journal(session / "input.tsv")
    except OSError:
        return {}
    offset = input_offset(session)
    found: dict[str, float] = {}
    for clip in clips:
        span = _span(clip)
        if span is not None and samples:
            found[clip.clip_sha] = float(sum(
                s.presses for s in samples if span[0] <= s.at - offset < span[1]))
    return found


def input_offset(session: Path) -> float:
    """Seconds from the input logs' zero to the video's; 0 before it was kept."""
    try:
        stats = json.loads((session / "capture_stats.json").read_text(encoding="utf-8"))
        return float(stats.get("input_offset_sec") or 0.0)
    except (OSError, ValueError, TypeError):
        return 0.0


def _standardised(values: dict[str, float]) -> dict[str, float]:
    import statistics

    if len(values) < 2:
        return {k: 0.0 for k in values}
    mean, spread = statistics.mean(values.values()), statistics.pstdev(values.values())
    return {k: (v - mean) / (spread or 1.0) for k, v in values.items()}


def rank(conn, clip_dir: Path) -> tuple[str, list]:
    """Clips best-first, and the basis of the order: "taste" or a fallback."""
    from pipeline_v2.ranking import ranked_fights

    variant, inside = _session_clips(conn, clip_dir)
    if not inside:
        return "none", []
    wanted = {c.clip_sha for c in inside}

    info: dict = {}
    fights, labels = ranked_fights(conn, variant=variant, info=info)
    # Once per clip: `ranked_fights` groups by path, so a stale row sharing a
    # re-cut window's filename yields a second "fight" whose best is that clip.
    best: dict[str, tuple[float, object]] = {}
    for fight in fights:
        sha = fight.best.clip_sha
        if sha in wanted and (sha not in best or fight.score > best[sha][0]):
            best[sha] = (fight.score, fight.best)
    scored = list(best.values())
    hands = _standardised(activity(clip_dir.parent, inside))
    voiced = _standardised(voice(inside))
    kinds = _standardised({c.clip_sha: kill_traits(c) for c in inside})

    def extra(sha: str) -> float:
        return (ACTIVITY_WEIGHT * hands.get(sha, 0.0) + VOICE_WEIGHT * voiced.get(sha, 0.0)
                + KILL_TRAIT_WEIGHT * kinds.get(sha, 0.0))

    if scored:
        taste = _standardised({clip.clip_sha: score for score, clip in scored})
        scored = [(taste[c.clip_sha] + extra(c.clip_sha), c) for _, c in scored]
        scored.sort(key=lambda pair: (-event_tier(pair[1]), -pair[0]))
        weight = info.get("speech_weight", 0.0)
        tried = info.get("speech_cv") or {}
        if weight:
            heard = f", speech weighted {weight:g}"
        elif tried:
            heard = ", speech off: it hurt this player's ratings"
        else:
            heard = ""
        used = "".join(name for name, on in ((", key presses", hands), (", mic level", voiced),
                                                  (", kill type", kinds)) if on)
        return (f"what happened, then taste{used} ({labels} labels, {variant}{heard})",
                scored)

    def fallback(clip) -> tuple[float, float]:
        # With no ratings yet, the player's hands are the best tie-break there
        # is (and better than a detector's confidence, which measured weak).
        return (float(event_tier(clip)), extra(clip.clip_sha))

    ordered = sorted(inside, key=fallback, reverse=True)
    return (
        "fallback: what happened, then key presses and mic level -- no labels on "
        f"this footage yet (review --variant {variant})",
        [(float(fallback(c)[0]), c) for c in ordered],
    )


def _span(clip) -> tuple[float, float] | None:
    """Where a clip sits on the session timeline, from its sidecar."""
    sidecar = _sidecar(clip.path)
    try:
        return float(sidecar["segment_start_sec"]), float(sidecar["segment_end_sec"])
    except (KeyError, TypeError, ValueError):
        return None


# A highlight grows over neighbouring windows that are still the same fight,
# up to this long. A build fight runs longer than one 8s window, and cutting
# it at the window edge (the player's complaint on session 3) loses the end.
MAX_HIGHLIGHT = 45.0
FIGHT_TAGS = frozenset({
    "build_fight", "box_fight", "piece_control", "editing", "shield_crack",
    "third_party", "multi_opponent", "multi_elimination", "finish_downed", "close_range",
})


def in_fight(clip) -> bool:
    """Whether a window is part of a fight: a kill or knock, or fight mechanics."""
    payload = clip.payload or {}
    return event_tier(clip) > 0 or bool(FIGHT_TAGS & set(payload.get("mechanics") or []))


# Quiet moments (no kill, knock or fight: a funny one, an emote, a glitch)
# become highlights once the player has kept this many on the rating page,
# and the taste ranking places them. Loudness could not find them: each
# session's three loudest quiet windows were the bus, deaths and a video
# playing nearby, and 0 of 18 were kept (2026-10-24).
QUIET_KEEPS_TO_LEARN = 5


def quiet_keeps(conn) -> int:
    """How many quiet moments the player has kept, across every session."""
    from pipeline_v2 import store

    kept = {row[0] for row in conn.execute("select clip_sha from preferences where label = 1")}
    return sum(1 for clip in store.all_clips(conn, _configured_variant())
               if clip.clip_sha in kept and event_tier(clip) == 0 and not in_fight(clip))


def fight_extent(clip, by_start: dict, pad: tuple[float, float]) -> tuple[float, float] | None:
    """The span to export for `clip`: its window grown over the neighbouring
    windows of the same fight, plus `pad`, at most `MAX_HIGHLIGHT` long.

    `by_start` maps each window's start time to its clip. Growth stops at the
    first window that is not in the fight, or at a gap (a skipped window).
    """
    span = _span(clip)
    if span is None:
        return None
    start, end = span
    stride = end - start

    def neighbour(at):
        found = by_start.get(round(at, 1))
        return found if found is not None and in_fight(found) else None

    if in_fight(clip):
        while end - start + stride + pad[0] + pad[1] <= MAX_HIGHLIGHT:
            before, after = neighbour(start - stride), neighbour(end)
            if after is not None:
                end += stride
            elif before is not None:
                start -= stride
            else:
                break
        # The win is the payoff of the fight before it: worth one window over
        # the cap rather than a second highlight of the same moment (session 3
        # ranked the final kill and its Victory Royale banner #1 and #2).
        after = by_start.get(round(end, 1))
        if after is not None and is_victory(after):
            end += stride
    return start - pad[0], end + pad[1]


def window_index(ordered: list) -> dict:
    """Window start time -> clip, for `fight_extent`."""
    index = {}
    for _, clip in ordered:
        span = _span(clip)
        if span is not None:
            index[round(span[0], 1)] = clip
    return index


# How long before "clip that" the moment it means can have been: it is said
# after the kill, often after a breath or a few seconds of looting.
ASK_LOOKBACK = 12.0


def asked_for(ordered: list, times: list[float]) -> list:
    """The clip each "clip that" means: of the windows in the `ASK_LOOKBACK`
    seconds up to it, the one ranked best. Commands with no window there
    (the lobby) are `unwindowed`'s."""
    found: list = []
    for at in times:
        for _, clip in ordered:
            span = _span(clip)
            if span is not None and span[0] < at and at - ASK_LOOKBACK < span[1]:
                if clip not in found:
                    found.append(clip)
                break
    return found


# What is cut for "clip that" said where no window was described (the lobby,
# a loading screen): the moment it means, its lead-in, and the words.
ASK_SPAN = (ASK_LOOKBACK + HIGHLIGHT_LEAD, HIGHLIGHT_LAG)


def unwindowed(ordered: list, times: list[float]) -> list[tuple[float, float]]:
    """Spans to cut straight from the footage, one per "clip that" no window
    covers. Lobby windows are never described (they are no gameplay), but a
    lobby can still be the funny moment; asks close together are one span."""
    spans: list[list[float]] = []
    for at in sorted(times):
        if any(span[0] < at and at - ASK_LOOKBACK < span[1]
               for _, clip in ordered if (span := _span(clip)) is not None):
            continue
        start, end = at - ASK_SPAN[0], at + ASK_SPAN[1]
        if spans and start < spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return [(start, end) for start, end in spans]


def pick(ordered: list, top: int, pad: tuple[float, float] = (0.0, 0.0),
         extent=None, pinned: list = ()) -> list:
    """The best `top` clips, never two whose exported spans overlap.

    `pinned` clips (the player asked for them) come first and are never
    dropped for `top`; a reel asked for twelve times holds twelve.

    `pad` is the lead and lag an export adds. The second-best window is usually
    the same fight 8 seconds later; padded, the two exports would overlap, and
    keeping both fills a short reel with one moment twice. `extent(clip)`, when
    given, is the whole span exported for a clip (`fight_extent`).
    """
    chosen: list = []
    taken: list[tuple[float, float]] = []
    first = [(score, clip) for score, clip in ordered if clip in pinned]
    first.sort(key=lambda pair: pinned.index(pair[1]))
    rest = [(score, clip) for score, clip in ordered if clip not in pinned]
    for score, clip in first + rest:
        if len(chosen) >= max(top, len(first)) and clip not in pinned:
            break
        if extent is not None:
            span = extent(clip)
        else:
            span = _span(clip)
            if span is not None:
                span = (span[0] - pad[0], span[1] + pad[1])
        if span is not None and any(
            span[0] < end and start < span[1] for start, end in taken
        ):
            continue
        chosen.append((score, clip))
        if span is not None:
            taken.append(span)
    return chosen


def _export(session: Path, clip, destination: Path, span) -> bool:
    """A coverage window re-cut from the raw footage over `span` (its lead-in
    and the rest of its fight).

    False when the raw segments are gone (pruned) or the cut fails; the caller
    then exports the bare window, which is shorter but still the moment.
    """
    if span is None:
        return False
    return clips_mod.cut_span(
        session, span[0], span[1], destination
    ) is not None


def run(
    session: Path,
    *,
    top: int = 10,
    prune_raw: bool = False,
    coverage: bool = True,
    speech: bool = True,
    describe_fn=describe,
    harvest_fn=clips_mod.harvest,
    export_fn=None,
    listen_fn=None,
    reembed_fn=None,
    undescribed_fn=None,
    unembedded_fn=None,
) -> dict:
    """Cut, describe, rank and keep the best `top` clips from one recording."""
    from pipeline_v2 import store
    from pipeline_v2.jsonio import write_json

    if coverage:
        journalled = clips_mod.write_coverage(session)
        clip_dir = session / "windows"
        why = "no recorded footage"
    else:
        journalled = clips_mod.read_journal(session / clips_mod.JOURNAL_NAME)
        clip_dir = session / "clips"
        why = "no triggers fired"
    if not journalled:
        return {"session": str(session), "highlights": [],
                "basis": "none", "why": why}

    played: dict = {"spans": None}
    if coverage:
        played = find_gameplay(session)
        half = clips_mod.COVERAGE_STRIDE / 2

        def outside(request) -> bool:
            # Lobby, loading and spectating: not described, never a highlight.
            return played["spans"] is not None and not gameplay_mod.overlaps(
                played["spans"], request.at - half, request.at + half)

        cut = harvest_fn(
            session, clip_dir, journal_name=clips_mod.COVERAGE_JOURNAL,
            analysis_span=clips_mod.COVERAGE_STRIDE,
            lead=half, lag=half, skip=outside,
        )
        if played["spans"] is not None:
            # Who the player is and the match phase, for the prompt (v3.8).
            segments = clips_mod.read_segments(session)
            gameplay_mod.annotate(session, clip_dir, max(s.end for s in segments))
    else:
        cut = harvest_fn(session, clip_dir)
    _step(f"cut {len(cut)} windows")
    # Before describing, so the embeddings built at its end include speech.
    heard = (listen_fn or listen)(session, clip_dir) if speech else 0
    found_laughs = hear_laughs(session) if speech else []
    # `cut` includes windows already cut on an earlier run. Describing those
    # again is free (cached) but not fast: 300s of frame extraction on
    # session 1 to look up 235 answers it already had.
    fresh = (undescribed_fn or undescribed)(clip_dir) if cut else 0
    _step(f"transcribed; {fresh} windows left to describe")
    if fresh:
        describe_fn(clip_dir, max_calls=fresh)
    elif heard or (unembedded_fn or unembedded)(clip_dir):
        (reembed_fn or reembed)(clip_dir)

    _step("described")
    conn = store.connect()
    try:
        basis, ordered = rank(conn, clip_dir)
        said = store.transcripts(conn)
        learned_quiet = quiet_keeps(conn) >= QUIET_KEEPS_TO_LEARN
    finally:
        conn.close()
    if played["spans"] is not None:
        # Windows cut before gating existed are still on disk and in the store.
        ordered = [
            (score, clip) for score, clip in ordered
            if (span := _span(clip)) is None
            or gameplay_mod.overlaps(played["spans"], *span)
        ]

    out = session / "highlights"
    if out.exists():
        shutil.rmtree(out)  # derived output only; rebuilt from clips/ each run
    out.mkdir(parents=True)

    pad = (HIGHLIGHT_LEAD, HIGHLIGHT_LAG) if coverage else (0.0, 0.0)
    export = export_fn or _export
    index = window_index(ordered)

    def extent(clip):
        return fight_extent(clip, index, pad) if coverage else None

    _step("ranked")
    from pipeline_v2.capture import voice as voice_mod

    utterances = voice_mod.heard(session) if speech else []
    asks = voice_mod.clip_requests(utterances)
    pinned = asked_for(ordered, asks)
    # A highlight is a moment: a kill, a knock or a fight, or one the player
    # asked for. Of 59 keeps, 1 had none of these (2026-10-24), and a session
    # spent in a lobby made four idle lobby screens its "highlights".
    def laughed(clip) -> bool:
        span = _span(clip)
        return span is not None and bool(reactions.laughed_in(found_laughs, *span))

    # Laughter makes a moment one, as a kill does: a funny moment has no kill.
    moments = [(score, clip) for score, clip in ordered
               if clip in pinned or learned_quiet or event_tier(clip) > 0 or in_fight(clip)
               or laughed(clip)]
    picked = [pair for pair in pick(moments, top, pad, extent if coverage else None, pinned)
              if Path(pair[1].path).is_file() or clips_mod.released(Path(pair[1].path))]
    # Asked for, then asked for where nothing was looked at, then the rest.
    jobs = ([("window", score, clip) for score, clip in picked if clip in pinned]
            + [("raw", 0.0, span) for span in unwindowed(ordered, asks)]
            + [("window", score, clip) for score, clip in picked if clip not in pinned])

    def name(position, job) -> str:
        kind, _, item = job
        if kind == "raw":
            return f"{position:02d}-asked-{item[1]:.0f}s.mkv"
        return f"{position:02d}-{Path(item.path).name}"

    def export_one(numbered) -> bool:
        position, job = numbered
        kind, _, item = job
        destination = out / name(position, job)
        if kind == "raw":
            if clips_mod.cut_span(session, *item, destination) is None:
                return False
        elif not (coverage and export(session, item, destination, extent(item))):
            if Path(item.path).is_file():
                shutil.copy2(Path(item.path), destination)
            elif clips_mod.recut_window(Path(item.path), destination) is None:
                return False  # released, and its raw footage would not cut
        mix_tracks(destination)
        return True

    # Side by side: each is a stream copy waiting on the disk, and one after
    # another they were 3.4s of the wait after a session.
    from concurrent.futures import ThreadPoolExecutor

    numbered = list(enumerate(jobs, start=1))
    with ThreadPoolExecutor(EXPORT_WORKERS) as pool:
        exported = list(pool.map(export_one, numbered))

    _step("exported")
    kept: list[Highlight] = []
    for (position, (kind, score, item)), ok in zip(numbered, exported):
        if not ok:
            continue
        if kind == "raw":
            kept.append(Highlight(
                rank=position, clip=name(position, (kind, score, item)), score=0.0,
                file=name(position, (kind, score, item)),
                summary="", outcome="not_described", trigger="clip_command",
                said=voice_mod.said_between(utterances, *item), asked=True,
            ))
            continue
        source = Path(item.path)
        payload = item.payload or {}
        kept.append(Highlight(
            rank=position, clip=source.name, score=round(score, 4),
            summary=item.summary or "", outcome=str(payload.get("outcome")),
            trigger=str(_sidecar(source).get("trigger_source", "")),
            said=said.get(item.clip_sha, ""), asked=item in pinned, laughed=laughed(item),
            file=name(position, (kind, score, item)),
        ))

    manifest = {
        "session": str(session),
        "mode": "coverage" if coverage else "triggers",
        "basis": basis,
        "journalled": len(journalled),
        "cut": len(cut),
        "ranked": len(ordered),
        "gameplay": _gameplay_note(played, journalled),
        "highlights": [asdict(h) for h in kept],
    }
    if not kept:
        manifest["why"] = "No kills, knocks or fights this session, so no highlights."
    write_json(out / "highlights.json", manifest)

    # Only once highlights exist, and only for an exact segment set: never a
    # glob broad enough to touch clips/, highlights/ or a journal.
    # Never under a released window: the raw footage is then its only copy.
    if (prune_raw and kept and len(cut) == len(journalled)
            and not any(clips_mod.released(p) for p in window_files(clip_dir))):
        for segment in clips_mod.read_segments(session):
            segment.path.unlink(missing_ok=True)
        manifest["pruned_raw"] = True
        write_json(out / "highlights.json", manifest)
    return manifest


def _gameplay_note(played: dict, journalled: list) -> dict:
    if played.get("spans") is None:
        return {"source": "none", "why": played.get("why", "not looked up")}
    half = clips_mod.COVERAGE_STRIDE / 2
    excluded = sum(
        not gameplay_mod.overlaps(played["spans"], r.at - half, r.at + half)
        for r in journalled
    )
    return {"source": "fortnite log", "matches": played.get("matches"),
            "excluded_windows": excluded, "origin": played.get("origin_source")}


def report(manifest: dict) -> None:
    source = "windows" if manifest.get("mode") == "coverage" else "triggers"
    print(f"\nhighlights: {len(manifest['highlights'])} kept "
          f"({manifest.get('cut', 0)} cut from {manifest.get('journalled', 0)} {source})")
    print(f"  ranked by {manifest['basis']}")
    played = manifest.get("gameplay") or {}
    if played.get("source") == "fortnite log":
        print(f"  gameplay from the Fortnite log: {played['excluded_windows']} lobby, "
              "loading or spectating windows skipped")
    for h in manifest["highlights"]:
        asked = " (you said clip it)" if h.get("asked") else ""
        if h.get("outcome") == "not_described":
            print(f"  {h['rank']:2d}. (you said clip it, outside a match) \"{h['said'][:100]}\"")
            continue
        print(f"  {h['rank']:2d}. [{h['outcome']}]{asked} {h['summary'][:100]}")
        if h.get("said"):
            print(f"      said: \"{h['said'][:100]}\"")
    if manifest.get("pruned_raw"):
        print("  raw segments deleted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cut, describe and rank a finished recording into highlights."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--prune-raw", action="store_true",
        help="delete the raw 30s segments once highlights exist (irreversible)",
    )
    args = parser.parse_args(argv)
    report(run(args.session, top=args.top, prune_raw=args.prune_raw))
    print(f"  reels : {reels(args.session)} made, 9:16, in {args.session / 'highlights' / REEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
