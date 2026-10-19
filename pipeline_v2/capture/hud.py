"""Reading Fortnite's HUD from our own footage: the player's eliminations and knocks.

The vision model sees 7 frames 1.3s apart and has to judge from them whether
the player killed someone. It gets it wrong both ways: other people's
kill-feed lines credited to the player, kills it never saw. But the game says
so outright. Every one of the player's eliminations puts an "ELIMINATION!"
banner with the opponent's name in the centre of the screen, and we have
every frame of it.

So this reads the recording, not the game: no injection, nothing the
anti-cheat could see. The word "ELIMINATION!" is drawn identically every
time, which makes it a template match (a millisecond a frame) rather than
text recognition (half a second); OCR runs only on the few frames that match,
to read the opponent's name.

Positions are fractions of the frame for Fortnite's default HUD, from a
1920x1200 recording; the templates are scaled with the frame height.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "games" / "assets"
# Banner kinds and the templates that identify each. "ELIMINATION!" is matched
# as two halves around the skull icon drawn over its middle letters;
# "KNOCKED!" (duos, trios, squads) has nothing over it.
KINDS = {
    "elimination": ("fortnite_elim_left.png", "fortnite_elim_right.png"),
    "knock": ("fortnite_knocked.png",),
}
REFERENCE_HEIGHT = 1200
# The centre banner, and within it the name plate under the word.
BANNER = (0.38, 0.72, 0.62, 0.86)
# The name plate, relative to where the word's left half matched, in pixels at
# REFERENCE_HEIGHT. Relative, because the banner is not always drawn in the
# same place: session 3 had it shifted left and down on some kills.
PLATE_FROM_WORD = (-15, 22, 175, 52)  # x0, y0, x1, y1
# Every frame. At 10 fps the banner's sharpest frame was skipped often enough
# to miss kills (session 2: stevens_land scored 0.61 sampled, 0.73 at 30 fps).
SCAN_FPS = None
# The banner pops in with a scaling animation, so the word is matched at a few
# sizes: Santi192424 scored 0.61 at the template's size and 0.88 at 0.9/1.1.
# Banner-free footage stayed at or under 0.62 at all three sizes.
SCALES = (0.9, 1.0, 1.1)
# A frame is matched at all sizes only if it already looks like the word at
# its usual size: banner-free frames sit far below this, and it saves 2/3 of
# the matching.
PREFILTER = 0.45
# Both halves of the word must match this well. The skull icon is drawn over
# the middle letters, so the word is matched as two halves around it. Session
# 3, every match checked by eye: real banners scored 0.66-0.92, everything
# else (floors, an item pickup, a leg) 0.60-0.62.
THRESHOLD = 0.65
# One banner stays up about two seconds and can dip below the threshold for a
# frame or two (the skull animates). Matches with no longer gap than this
# between them are one banner; a double kill 1.5s apart is still two.
GAP_SECONDS = 1.0
# The same banner seen twice: it drops out for a moment mid-display (1.6s
# after it appears, every time on session 3), and one straddling a segment
# boundary is seen by both scans. Within this, the same name is one kill.
MERGE_SECONDS = 2.5


def _same_name(a: str, b: str) -> bool:
    """Two reads of one name plate (OCR differs a little between frames)."""
    import difflib

    def clean(name):
        return "".join(ch for ch in name.lower() if ch.isalnum())

    a, b = clean(a), clean(b)
    if not a or not b:
        return True  # an unreadable plate cannot tell two kills apart
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.6


# A weak match with no name on its plate is not a banner: on session 2 a black
# loading screen and the "INPUT CODE" lobby menu each scored 0.65. Every real
# banner on sessions 2 and 3 scored 0.70 or more, or had a name read.
UNNAMED_MINIMUM = 0.70


def plausible(event: "Elimination") -> bool:
    return bool(event.name) or event.score >= UNNAMED_MINIMUM


def merge(events: list["Elimination"]) -> list["Elimination"]:
    """One event per banner: repeats of a name within `MERGE_SECONDS` dropped,
    keeping the best-matched read of it."""
    kept: list[Elimination] = []
    for event in sorted(events, key=lambda e: e.at):
        twin = next((k for k, o in enumerate(kept) if o.kind == event.kind
                     and event.at - o.at < MERGE_SECONDS and _same_name(o.name, event.name)),
                    None)
        if twin is None:
            kept.append(event)
        elif event.score > kept[twin].score:
            kept[twin] = Elimination(at=kept[twin].at, score=event.score,
                                     name=event.name or kept[twin].name, kind=event.kind)
    return kept
EVENTS_NAME = "hud_events.json"


@dataclass(frozen=True)
class Elimination:
    at: float  # seconds on the session timeline
    score: float  # the weakest template's match
    name: str = ""  # the opponent, as read off the name plate
    kind: str = "elimination"  # or "knock"


def _crop(region, width: int, height: int) -> tuple[int, int, int, int]:
    """ffmpeg crop (w, h, x, y) for a fractional region, even-sized."""
    x0, y0, x1, y1 = region
    w = round((x1 - x0) * width) // 2 * 2
    h = round((y1 - y0) * height) // 2 * 2
    return w, h, round(x0 * width), round(y0 * height)


def _templates(height: int, names: tuple[str, ...], scales=SCALES):
    """Per template (each half of the word), its image at each of `scales`,
    the usual size first."""
    import cv2

    base = height / REFERENCE_HEIGHT
    ordered = sorted(scales, key=lambda k: abs(k - 1.0))
    found = []
    for name in names:
        image = cv2.imread(str(ASSETS / name), cv2.IMREAD_GRAYSCALE)
        found.append([image if base * k == 1.0 else
                      cv2.resize(image, None, fx=base * k, fy=base * k, interpolation=cv2.INTER_AREA)
                      for k in ordered])
    return found


def _frame_rate(segment: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", str(segment)],
        capture_output=True, text=True)
    num, _, den = probe.stdout.strip().partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 30.0


def frame_size(segment: Path) -> tuple[int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(segment)],
        capture_output=True, text=True)
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    return width, height


def match_scores(frames, templates) -> list[tuple[float, tuple[int, int, float]]]:
    """Per grayscale frame: the weakest template's best match over its sizes,
    and where and at what size the first template matched (x, y, scale)."""
    import cv2

    ordered = sorted(SCALES, key=lambda k: abs(k - 1.0))

    def best(frame, sizes):
        results = [(cv2.minMaxLoc(cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)), k)
                   for t, k in zip(sizes, ordered)]
        (_, score, _, (x, y)), scale = max(results, key=lambda r: r[0][1])
        return float(score), (x, y, scale)

    found = []
    for frame in frames:
        quick = [best(frame, sizes[:1]) for sizes in templates]
        if min(q[0] for q in quick) < PREFILTER:
            found.append((min(q[0] for q in quick), quick[0][1]))
            continue
        full = [best(frame, sizes) for sizes in templates]
        found.append((min(f[0] for f in full), full[0][1]))
    return found


def peaks(times: list[float], scores: list[float], threshold: float = THRESHOLD,
          gap: float = GAP_SECONDS) -> list[tuple[float, float, int]]:
    """(time, score, index) of the best frame of each run of matches."""
    found: list[tuple[float, float, int]] = []
    last_hit = None
    for index, (at, score) in enumerate(zip(times, scores)):
        if score < threshold:
            continue
        if found and last_hit is not None and at - last_hit <= gap:
            if score > found[-1][1]:
                found[-1] = (at, score, index)
        else:
            found.append((at, score, index))
        last_hit = at
    return found


def scan(session: Path, runner=subprocess.run, segments=None) -> list[Elimination]:
    """Every banner in a recorded session (or in `segments` of it), named."""
    import numpy as np

    from pipeline_v2.capture import clips

    segments = clips.read_segments(session) if segments is None else segments
    if not segments:
        return []
    width, height = frame_size(segments[0].path)
    cw, ch, cx, cy = _crop(BANNER, width, height)
    templates = {kind: _templates(height, names) for kind, names in KINDS.items()}
    times: list[float] = []
    scores: dict[str, list[float]] = {kind: [] for kind in KINDS}
    # Only frames that matched are kept: all of them would be ~1GB a session.
    kept: dict[str, dict[int, object]] = {kind: {} for kind in KINDS}
    for segment in segments:
        rate = f"fps={SCAN_FPS}," if SCAN_FPS else ""
        raw = runner(
            ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", str(segment.path), "-vf",
             f"{rate}crop={cw}:{ch}:{cx}:{cy},format=gray", "-f", "rawvideo", "-"],
            capture_output=True).stdout
        batch = np.frombuffer(raw, np.uint8).reshape(-1, ch, cw)
        step = 1 / (SCAN_FPS or _frame_rate(segment.path))
        for kind, kind_templates in templates.items():
            matched = match_scores(batch, kind_templates)
            for offset, (score, where) in enumerate(matched):
                if score >= THRESHOLD:
                    kept[kind][len(times) + offset] = (batch[offset], where)
            scores[kind] += [score for score, _ in matched]
        times += [segment.start + i * step for i in range(len(batch))]
    events = []
    for kind in KINDS:
        found = peaks(times, scores[kind])
        names = read_names(kept[kind], [index for _, _, index in found], height)
        events += [Elimination(at=round(at, 2), score=round(score, 3), name=name, kind=kind)
                   for (at, score, _), name in zip(found, names)]
    return [e for e in merge(events) if plausible(e)]


def read_names(frames: dict, indices: list[int], height: int) -> list[str]:
    """OCR the name plate on each banner frame ('' when unreadable).

    `frames[index]` is (banner crop, where the word's left half matched).
    """
    if not indices:
        return []
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return [""] * len(indices)
    ocr = RapidOCR()
    names = []
    for index in indices:
        crop, (wx, wy, matched) = frames[index]
        # The plate grows with the word as the banner animates in.
        scale = height / REFERENCE_HEIGHT * matched
        x0, y0, x1, y1 = (round(v * scale) for v in PLATE_FROM_WORD)
        plate = crop[max(0, wy + y0): wy + y1, max(0, wx + x0): wx + x1]
        result, _ = ocr(plate) if plate.size else (None, None)
        names.append(" ".join(r[1] for r in (result or [])).strip())
    return names


def write_events(session: Path, events: list[Elimination], scanned: list[str] | None = None) -> Path:
    path = session / EVENTS_NAME
    path.write_text(json.dumps({"scanned": sorted(scanned or []),
                                "eliminations": [asdict(e) for e in events]}, indent=1),
                    encoding="utf-8")
    return path


def _load(session: Path) -> dict:
    try:
        return json.loads((session / EVENTS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_events(session: Path) -> list[Elimination] | None:
    """The session's banners, or None if the HUD was never read."""
    data = _load(session)
    if "eliminations" not in data:
        return None
    return [Elimination(**e) for e in data["eliminations"]]


def update(session: Path, runner=subprocess.run, scan_fn=None) -> int:
    """Scan the segments not scanned yet and add their banners; how many new.

    Called by `live` as segments finish and by the final run for the rest, so
    each 30s of footage is read once. A banner straddling two segments is seen
    by both scans and kept once.
    """
    from pipeline_v2.capture import clips

    data = _load(session)
    scanned = set(data.get("scanned", []))
    events = [Elimination(**e) for e in data.get("eliminations", [])]
    todo = [s for s in clips.read_segments(session) if s.path.name not in scanned]
    if not todo:
        return 0
    combined = merge(events + list((scan_fn or scan)(session, runner=runner, segments=todo)))
    write_events(session, combined, sorted(scanned | {s.path.name for s in todo}))
    return len(combined) - len(events)


def tiers(events: list[Elimination] | None) -> dict | None:
    """Window start -> what the HUD says happened, for `highlights.event_tier`.

    None unless the HUD was read AND at least one banner was found: a session
    with no banner may be a HUD layout this reader does not know, and then the
    describer's judgement is the better one.
    """
    if not events:
        return None
    return {"eliminations": [e.at for e in events if e.kind == "elimination"],
            "knocks": [e.at for e in events if e.kind == "knock"]}
