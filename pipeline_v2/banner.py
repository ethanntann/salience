"""When a kill happened in a Fortnite clip: the ELIMINATION! banner under the crosshair.

The describer says a window holds a kill but not when in its 8 seconds, and
the game log never names the player's eliminations, so the app's kill marker
sat on the window's centre: up to 4s off (2026-10-24, "the kill was at like
13", the marker at 12). The game says it on screen instead. Fortnite puts
ELIMINATION! in the same place after every kill, and matching that word in
the frames (bright pixels against a stencil, correlated) finds it.

The banner slides in, and the match reaches `MATCH` about half a second after
the shot that killed: on session 7's highlights the kill frame was at the
onset less `LEAD`, by eye, to a frame or two (01: 13.4s, 08: 7.8s, 09: 11.1s).
One decode of the span, 10 frames a second, about a second a highlight.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np

# The words, as bright pixels, at `HEIGHT`. ELIMINATION! is the average of
# 8 solo kills' banners; in squads a kill is KNOCKED! first, and half of a
# squads session's highlights had no marker without it (2026-10-25). Each has
# its own cut: KNOCKED!'s letters fade to grey below the top rows.
STENCIL = (
    "..#####..#....##........###..##......##...####..######.##....###........##..##..",
    "..#####.##....##..####.####..##..##..##...####..######.##...#####...##..##..##..",
    ".##.....##....##..###..####.##..###..#...##.##...###...##..##..##..###..#..##...",
    ".##.....#.....##..#.#..#.#..##..###..#...#..#.....#....#...##..##..###..#..##...",
    ".####..##.....#...#.#.#..#..##..#.#.##..##..#....##...##..##...##..#.#.##..##...",
    ".#.....##....##..##.##...#..#...#..##...#...#....##...##..##...##..#..###.......",
    ".#.....##........#..##..........#.......#...#.........#...##...#...#............",
    "........#.#..#...#......#........................#....#...##..#...#...##........",
)
KNOCKED = (
    ".###.###.###..##...####....#####..##..###.#####.######..##",
    ".##..##..###..##..######..######..##.###..#####.######..##",
    ".##.###..###..#..##..###.###..##.###.##..###....##..###.##",
    ".#####...###.##..##..###.###.....#####...#####..##..##..##",
    "######...######.###..###.##......#####...#####..##..##..##",
    "######..##.####.##...##..##..###.#####...##....##...##....",
    "###.##..##..##..##..###..##..##..##.##...##....##..##..##.",
    "##..##..##..##...#####...#####..##..##..#####..#####...##.",
)
# The ELIM! of DOUBLE ELIM!, TRIPLE ELIM! and the rest: a squad's last
# player falling, stacked under ENEMY TEAM WIPED! or another banner. Its top
# rows only: a skull sits over the lower middle (2026-10-25).
ELIM = (
    "..#####....#####.##....##....#..###..##",
    "..#####....#####.##....##..###..###..##",
    ".###.......##....##....##..###.####..#.",
    ".#####....###....##....#..####.#.##.##.",
    ".#####....####..##....##..##.###.#..##.",
    "######....##....##....##..##.##..#.....",
)
BRIGHT = 185  # ELIMINATION!'s white, in grey levels
# word: (stencil, the grey level its letters are above), first match first
WORDS = {"kill": (STENCIL, BRIGHT), "knock": (KNOCKED, 140), "elim": (ELIM, 170)}
KINDS = {"kill": "kill", "knock": "knock", "elim": "kill"}  # what each word means happened
HEIGHT = 600  # frames are scaled to this height, the stencils'
# x0, x1, y0, y1 of the frame: where the words sit, and room for a banner
# stacked under another (ENEMY TEAM WIPED! above DOUBLE ELIM!, say).
REGION = (0.43, 0.57, 0.70, 0.84)
FPS = 10
MATCH = 0.5  # correlation that is the word, not a white wall
LEAD = 0.5  # seconds from the killing shot to the banner's match
GAP = 1.0  # seconds without the word that make the next match a new kill
# A banner already matched is still up while its match stays above this: on
# screen it flickers about `MATCH` (0.4 to 0.5 for two seconds), and one
# ELIMINATION! came back as kills at 0:17 and 0:18 (2026-11-03).
HOLD = 0.35

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _stencil(rows=STENCIL) -> np.ndarray:
    s = np.array([[c == "#" for c in row] for row in rows], np.float32)
    return s - s.mean()


def scores(frames: np.ndarray, word: str = "kill") -> np.ndarray:
    """Each frame's best match for `word` anywhere in the region, -1..1."""
    return all_scores(frames, (word,))[word]


def all_scores(frames: np.ndarray, words=tuple(WORDS)) -> dict[str, np.ndarray]:
    """Each frame's best match for each word, -1..1, every frame and word at once.

    Normalised correlation of the bright pixels with each stencil: a stencil
    sums to zero, so the window's mean drops out of the product, and a bright
    mask's squares are itself, so each window's spread is a box sum. One
    position at a time was 7 CPU-seconds a 30s segment, which a match being
    played cannot spare; each frame's transform is taken once per cut and
    shared by the words that use it.
    """
    from scipy import fft

    out = {}
    if not len(frames):
        return {w: np.zeros(0, np.float32) for w in words}
    h, w = frames.shape[1:]
    by_cut: dict[int, list[str]] = {}
    for word in words:
        by_cut.setdefault(WORDS[word][1], []).append(word)
    for cut, named in by_cut.items():
        bright = (frames > cut).astype(np.float32)
        stencils = {word: _stencil(WORDS[word][0]) for word in named}
        th = max(st.shape[0] for st in stencils.values())
        tw = max(st.shape[1] for st in stencils.values())
        shape = (fft.next_fast_len(h + th), fft.next_fast_len(w + tw))
        spectrum = fft.rfft2(bright, s=shape, axes=(1, 2))
        table = np.pad(bright.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
        for word, stencil in stencils.items():
            sh, sw = stencil.shape
            if h < sh or w < sw:
                out[word] = np.zeros(len(frames), np.float32)
                continue
            kernel = fft.rfft2(stencil[::-1, ::-1], s=shape)
            full = fft.irfft2(spectrum * kernel, s=shape, axes=(1, 2))
            num = full[:, sh - 1:h, sw - 1:w]  # the "valid" part: the stencil wholly inside
            box = table[:, sh:, sw:] - table[:, :-sh, sw:] - table[:, sh:, :-sw] + table[:, :-sh, :-sw]
            spread = np.clip(box - box ** 2 / (sh * sw), 0.0, None)
            ncc = num / (np.sqrt(spread * float((stencil ** 2).sum())) + 1e-6)
            ncc[spread < 1e-3] = 0.0  # an even wall (all dark, all bright) matches nothing
            out[word] = ncc.reshape(len(frames), -1).max(axis=1).astype(np.float32)
    return out


def onsets(series: np.ndarray, fps: float = FPS) -> list[float]:
    """Seconds into the series where the word arrives, once per banner."""
    return [round(t, 2) for t in _onset_times(np.arange(len(series)) / fps, series)]


def _size(path: Path, runner) -> tuple[int, int] | None:
    out = runner(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                  "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW)
    try:
        width, height = (int(v) for v in out.stdout.strip().split(",")[:2])
        return width, height
    except ValueError:
        return None


def frames(path: Path, start: float, end: float, runner=subprocess.run, fps: float = FPS,
           gpu: bool = True) -> np.ndarray:
    """The region, grey, `fps` a second, from `start` to `end` seconds into the file.

    Decoded on the graphics card's video decoder where there is one (NVDEC,
    apart from the 3D engine a game draws with): 1.1 CPU-seconds a 30s
    segment, against 25 decoding on the CPU, which is most of a core beside a
    match. The CPU is the fallback, or the choice with `gpu` false.
    """
    size = _size(path, runner)
    if size is None:
        return np.zeros((0, 1, 1), np.uint8)
    width = round(size[0] * HEIGHT / size[1] / 2) * 2
    x0, x1, y0, y1 = REGION
    w, h = int(width * (x1 - x0)), int(HEIGHT * (y1 - y0))
    start = max(0.0, start)
    # Frames dropped to `fps` before they are scaled, not after: the same
    # frames, a tenth less time on the CPU (2026-11-05).
    cut = f"crop={w}:{h}:{int(width * x0)}:{int(HEIGHT * y0)},format=gray"
    span = ["-ss", f"{start:.2f}", "-t", f"{max(0.1, end - start):.2f}"]
    tries = (["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", *span, "-i", str(path),
              "-an", "-vf", f"fps={fps},scale_cuda={width}:{HEIGHT},hwdownload,format=nv12,{cut}"],
             [*span, "-i", str(path), "-an", "-vf", f"fps={fps},scale={width}:{HEIGHT},{cut}"])
    if not gpu:
        tries = tries[1:]
    raw = b""
    for middle in tries:
        done = runner(["ffmpeg", "-v", "error", *middle, "-f", "rawvideo", "-"],
                      capture_output=True, timeout=120, creationflags=_NO_WINDOW)
        raw = done.stdout or b""
        if len(raw) >= w * h and getattr(done, "returncode", 0) == 0:
            break
    count = len(raw) // (w * h)
    return np.frombuffer(raw[:count * w * h], np.uint8).reshape(count, h, w)


def found(path: Path, start: float, end: float, runner=subprocess.run) -> list[tuple[float, str]]:
    """(seconds into the file, "kill" or "knock") for each banner between `start` and `end`."""
    start = max(0.0, start)
    try:
        stack = frames(path, start, end + LEAD + 0.5, runner)
    except (OSError, subprocess.SubprocessError):
        return []
    if not len(stack):
        return []
    # A banner already up at `start` is an earlier kill's.
    matched = all_scores(stack)
    hits = sorted((round(start + t - LEAD, 2), KINDS[word]) for word in WORDS for t in onsets(matched[word])
                  if 0 < t and start + t - LEAD <= end)
    # One banner, one kill: ELIM! is the start of ELIMINATION! too.
    kept: list[tuple[float, str]] = []
    for at, kind in hits:
        if kept and kept[-1][1] == kind and at - kept[-1][0] < GAP:
            continue
        kept.append((at, kind))
    return kept


def kills(path: Path, start: float, end: float, runner=subprocess.run) -> list[float]:
    """Seconds into the file where kills or knocks happened between `start` and `end`."""
    return [t for t, _ in found(path, start, end, runner)]


SIDECAR_KEY = "banners"


# Each recorded segment's match scores, once: beside the session.
SESSION_CACHE = "banners.json"


def _segments(session: Path) -> list[tuple[Path, float, float]]:
    """(file, start, end) of each finished segment the recorder listed
    (`segments.csv`), read here: the pipeline does not import the recorder."""
    import csv

    found = []
    try:
        rows = list(csv.reader((Path(session) / "segments.csv").read_text(encoding="utf-8").splitlines()))
    except OSError:
        return []
    for row in rows:
        try:
            path, start, end = Path(session) / Path(row[0]).name, float(row[1]), float(row[2])
        except (IndexError, ValueError):
            continue
        if end > start and path.is_file() and path.stat().st_size > 0:
            found.append((path, start, end))
    return sorted(found, key=lambda segment: segment[1])


def _onset_times(times: np.ndarray, series: np.ndarray) -> list[float]:
    """`onsets` over frames at `times` (seconds), which may span segments."""
    found, last = [], None
    for t, value in zip(times, series):
        if value >= MATCH:
            if last is None or t - last > GAP:
                found.append(float(t))
            last = float(t)
        elif value >= HOLD and last is not None and t - last <= GAP:
            last = float(t)  # the same banner, flickering under the match
    return found


SCAN_WORKERS = 2

_SCANNING: dict[str, threading.Lock] = {}
_SCANNING_GUARD = threading.Lock()


def session_scores(session: Path, until: float, runner=subprocess.run) -> tuple[np.ndarray, dict]:
    """`_session_scores`, one caller at a time per session: windows planned
    side by side each found no scores yet and each scanned the whole session,
    eight times over, 3:13 for 33 windows' frames (2026-11-05)."""
    key = str(Path(session).resolve())
    with _SCANNING_GUARD:
        lock = _SCANNING.setdefault(key, __import__("threading").Lock())
    with lock:
        return _session_scores(session, until, runner)


def _session_scores(session: Path, until: float, runner=subprocess.run) -> tuple[np.ndarray, dict]:
    """(frame times, word -> scores) over the session's segments up to
    `until` seconds, each segment decoded once and its scores kept.

    A window at a time started the GPU's decoder for every 8s: 2.2
    CPU-seconds a window, mid-match, against 1.2 a whole 30s segment
    (2026-11-01). Scores, not hits, are kept, so a banner across two
    segments is one banner, as it is on screen.
    """
    import json

    from pipeline_v2.jsonio import write_json

    path = Path(session) / SESSION_CACHE
    words = "+".join(WORDS)
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    if cache.get("words") != words or cache.get("fps") != FPS:
        cache = {"words": words, "fps": FPS, "segments": {}}
    segments = [s for s in _segments(session) if s[1] < until]
    missing = [s for s in segments if s[0].name not in cache["segments"]]

    def scan(item):
        index, (file, begins, ends) = item
        try:
            stack = frames(file, 0.0, ends - begins + 1.0, runner, gpu=workers == 1 or index % 2 == 0)
        except (OSError, subprocess.SubprocessError):
            return file, None
        matched = all_scores(stack)
        return file, {w: [round(float(v), 3) for v in matched[w]] for w in WORDS}

    from concurrent.futures import ThreadPoolExecutor

    from salience_media import ffmpeg

    # After the game, every other segment on the CPU, as frames are taken
    # (`live.bank_frames`): 2.1s a segment on the GPU's decoder, 2.5 on the
    # CPU, 44.5s one at a time for an 18-segment session (2026-11-05).
    # During play, the GPU alone, one at a time.
    workers = 1 if ffmpeg.LOW_PRIORITY or len(missing) < 2 else SCAN_WORKERS
    with ThreadPoolExecutor(workers) as pool:
        scanned = dict(pool.map(scan, enumerate(missing)))
    times, series, changed = [], {w: [] for w in WORDS}, False
    for file, begins, ends in segments:
        kept = cache["segments"].get(file.name)
        if kept is None:
            kept = scanned.get(file)
            if kept is None:
                break  # a later segment's frames would follow a gap
            cache["segments"][file.name] = kept
            changed = True
        count = min(len(kept[w]) for w in WORDS)
        times += [begins + i / FPS for i in range(count)]
        for w in WORDS:
            series[w] += kept[w][:count]
    if changed:
        try:
            write_json(path, cache)
        except OSError:
            pass
    return np.array(times, np.float64), {w: np.array(v, np.float32) for w, v in series.items()}


def session_found(session: Path, start: float, end: float, runner=subprocess.run) -> list[tuple[float, str]]:
    """`found` for the span [start, end] of the session, from `session_scores`:
    (seconds after `start`, kind), as the window cut from there would give."""
    times, matched = session_scores(session, end + LEAD + 0.5, runner)
    hits = sorted((round(t - LEAD - start, 2), KINDS[word]) for word in WORDS
                  for t in _onset_times(times, matched[word]) if start < t and t - LEAD <= end)
    kept: list[tuple[float, str]] = []
    for at, kind in hits:
        if kept and kept[-1][1] == kind and at - kept[-1][0] < GAP:
            continue
        kept.append((at, kind))
    return kept


def _in_session(clip: Path, record: dict | None) -> tuple[Path, float] | None:
    """The session a window was cut from and where it starts, while that
    session's footage is still on disk."""
    session = Path(clip).parent.parent
    try:
        start = float((record or {})["segment_start_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    covering = [s for s in _segments(session) if s[1] <= start < s[2]]
    return (session, start) if covering else None


def cached_found(clip: Path, length: float, runner=subprocess.run) -> list[tuple[float, str]]:
    """`found` over a whole window. From its session's footage, a segment
    decoded once for all its windows (`session_found`), while that is on
    disk; otherwise the window itself, kept in its sidecar."""
    import json

    from pipeline_v2.jsonio import write_json

    sidecar = Path(clip).with_suffix(".capture.json")
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    placed = _in_session(Path(clip), record) if length else None
    if placed is not None:
        return session_found(placed[0], placed[1], placed[1] + length, runner)
    words = "+".join(WORDS)
    kept = (record or {}).get(SIDECAR_KEY) or {}
    if kept.get("words") == words:
        return [(float(t), str(kind)) for t, kind in kept.get("hits") or []]
    hits = found(Path(clip), 0.0, length, runner) if length else []
    if record is not None:
        record[SIDECAR_KEY] = {"words": words, "hits": [[t, kind] for t, kind in hits]}
        try:
            write_json(sidecar, record)
        except OSError:
            pass
    return hits
