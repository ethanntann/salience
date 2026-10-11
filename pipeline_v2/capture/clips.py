"""Turning a trigger into a clip on disk.

The missing half of the capture system. Detectors produced `Trigger` objects that
nothing consumed, so a session left behind 30-second segments and no answer to
"which moment mattered".

Two facts about the recording shape the design.

**One keyframe per segment.** The encoder runs with `-g fps * segment_seconds`,
one forced IDR per 30s chunk, because a keyframe per second cost bitrate during
play. Measured on a real segment: exactly one keyframe, at 0.000s. So a stream
copy can only cut at a chunk boundary, and any window tighter than 30s has to be
re-encoded.

**Re-encoding during play is the one thing we cannot afford.** Every capture
measurement this month has been a fight over the CPU and GPU the game wants. So
nothing here runs while the player is playing: a trigger appends one line to a
journal, which is free, and the cutting happens afterwards against footage
already on disk.

Deferring has a second benefit. A trigger fires *during* a moment, but the clip
needs the seconds after it, and those have not been recorded yet. Cutting later
removes any need to wait for them.

What a triggered clip gives the ranking pipeline that a raw segment cannot is
`event_sec`. The uniform path leaves it `None` and picks a random 4-second
window to describe, which on a 30-second chunk usually misses the event
entirely. Here the event time is known, so the window is placed on it.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# The buffer holds the seconds *before* the trigger, because a detector fires
# only once its evidence has arrived -- the shout comes after the shot. The same
# asymmetry the threshold tuner scores against: early is a hit, late is a miss.
LEAD_SECONDS = 12.0
LAG_SECONDS = 4.0

# What the vision model is asked to describe. Narrower than the clip on purpose:
# the surrounding footage is context for a human watching, not for the model.
ANALYSIS_SPAN = 4.0

JOURNAL_NAME = "triggers.jsonl"

# Full coverage: the whole session as back-to-back analysis windows, so nothing
# depends on a detector having fired. Session 1 measured why: the input-burst
# trigger overlapped only 5 of the 14 moments the player kept, and 7 of their
# 17 keeps came from the *random* control arm. A gate that drops two thirds of
# what the player likes cannot be rescued by ranking what is left.
#
# Each window's analysis span is the stride itself, so the spans tile the
# session with no gap; 7 frames over 8s is one every 1.3s, inside the few
# seconds a kill-feed line stays up. Each window is cut to exactly its span, so
# the windows together are the size of the recording, not twice it (16s clips
# on an 8s stride measured 3.8 GB for half an hour). The lead-in a human wants
# is added only to the few windows exported as highlights (`cut_span`).
COVERAGE_JOURNAL = "coverage.jsonl"
COVERAGE_STRIDE = 8.0


@dataclass(frozen=True)
class ClipRequest:
    """A trigger expressed in the run's own timeline.

    `at` is seconds from the start of the run, not `time.monotonic()`. A trigger
    carries a monotonic clock that means nothing once the process exits, so the
    conversion happens while the run start is still known -- at journalling time.
    """

    at: float
    kind: str
    source: str
    confidence: float = 0.5
    detail: str = ""

    @classmethod
    def from_trigger(cls, trigger, run_started: float) -> "ClipRequest":
        return cls(
            at=round(trigger.at - run_started, 3),
            kind=trigger.kind,
            source=trigger.source,
            confidence=trigger.confidence,
            detail=trigger.detail,
        )


def append(journal: Path, request: ClipRequest) -> None:
    """One line per trigger, written as it fires.

    Journalled rather than held in memory for the same reason marks and input
    samples are: on Windows a hard kill is `TerminateProcess`, so no catchable
    signal arrives, no cleanup runs, and anything held until the end of a
    session dies with it.
    """
    try:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as handle:
            print(json.dumps(asdict(request)), file=handle)
    except OSError:
        pass  # never interrupt a recording to write telemetry


def read_journal(path: Path) -> list[ClipRequest]:
    found: list[ClipRequest] = []
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            found.append(ClipRequest(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue  # a torn final line is normal after a kill
    return found


@dataclass(frozen=True)
class Segment:
    path: Path
    start: float
    end: float


def read_segments(run_dir: Path) -> list[Segment]:
    """The chunk listing ffmpeg's segment muxer writes as it closes each file."""
    listing = run_dir / "segments.csv"
    if not listing.is_file():
        return []
    found: list[Segment] = []
    for row in csv.reader(listing.read_text(encoding="utf-8").splitlines()):
        if len(row) != 3:
            continue
        path = run_dir / Path(row[0]).name
        try:
            start, end = float(row[1]), float(row[2])
        except ValueError:
            continue  # the muxer may not have finished writing this row
        if path.is_file() and path.stat().st_size > 0 and end > start:
            found.append(Segment(path, start, end))
    return sorted(found, key=lambda segment: segment.start)


def segments_covering(
    segments: list[Segment], start: float, end: float
) -> list[Segment]:
    """Every chunk overlapping the window, in order and with no gap between them.

    A gap means a chunk is still being written or was lost, and concatenating
    across one splices unrelated footage together without complaint. Refusing is
    better than producing a clip that misrepresents what happened.
    """
    touching = [s for s in segments if s.end > start and s.start < end]
    for earlier, later in zip(touching, touching[1:]):
        if later.start > earlier.end + 0.001:
            return []
    return touching


def available_window(
    segments: list[Segment], at: float, lead: float, lag: float
) -> tuple[float, float]:
    """The requested window clipped to footage that actually exists.

    A trigger three seconds into a run cannot have twelve seconds of lead, and
    one near the end cannot have its lag. A shorter clip is the right answer;
    padding to the requested length with whatever happens to be adjacent is not.
    """
    if not segments:
        return (0.0, 0.0)
    return (
        max(segments[0].start, at - lead),
        min(segments[-1].end, at + lag),
    )


def concat_list(segments: list[Segment], destination: Path) -> Path:
    """The input file ffmpeg's concat demuxer reads.

    Its parser takes single quotes around each path, so an embedded quote has to
    be broken out of the quoted run and re-escaped. Windows paths make that
    vanishingly unlikely, but a silently truncated path would produce a clip of
    the wrong moment rather than an error.
    """
    quote = chr(39)
    escaped = quote + chr(92) + quote + quote
    lines = [
        "file " + quote + str(s.path.resolve()).replace(quote, escaped) + quote
        for s in segments
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def cut_command(
    list_file: Path,
    first_segment_start: float,
    start: float,
    end: float,
    destination: Path,
    *,
    encoder: str = "h264_nvenc",
    bitrate: str = "8M",
    copy: bool = False,
) -> list[str]:
    """Re-encode the video window; copy the audio tracks untouched.

    With `copy`, nothing is re-encoded: the caller has placed `start` on a
    keyframe, and output-side seeking then begins exactly there. Off the grid
    the same command silently starts at the *next* keyframe (measured: a
    request for 23s began with a second of nothing), which is why snapping is
    the caller's job and not optional.

    `-ss` sits *after* `-i` so the seek is frame-accurate instead of snapping
    back to the single keyframe at the head of the chunk.

    `-map 0` is not decoration. Without it ffmpeg keeps only the first stream of
    each type, which silently dropped the microphone track once already and
    looked like a recorder fault rather than a missing muxing flag.
    """
    codec = (
        ["-c", "copy", "-avoid_negative_ts", "make_zero"] if copy
        else ["-c:v", encoder, "-b:v", bitrate, "-c:a", "copy"]
    )
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-ss", f"{start - first_segment_start:.3f}",
        "-t", f"{end - start:.3f}",
        "-map", "0",
        *codec,
        str(destination),
    ]


def keyframe_interval(segment: Path, runner=subprocess.run) -> float | None:
    """The regular keyframe spacing of a recording, or None if it has none
    short enough to cut on.

    Read from the first seconds of one segment (`-skip_frame nokey` decodes
    keyframes only). Recordings made before the 2s interval have one keyframe
    per 30s chunk and come back None, so they are re-encoded as before.
    """
    try:
        probe = runner(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-skip_frame", "nokey", "-read_intervals", "%+7",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(segment)],
            capture_output=True, text=True, timeout=30,
        )
        times = [float(x) for x in probe.stdout.split() if x.strip()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    gaps = [b - a for a, b in zip(times, times[1:])]
    if len(gaps) < 2 or max(gaps) - min(gaps) > 0.05 or max(gaps) > 4.0:
        return None
    return round(sum(gaps) / len(gaps), 3)


def snap_down(at: float, origin: float, interval: float) -> float:
    """The keyframe at or before `at`, on a grid starting at `origin`."""
    steps = int((at - origin) / interval + 1e-6)
    return round(origin + steps * interval, 3)


def metadata(
    request: ClipRequest,
    clip: Path,
    start: float,
    end: float,
    *,
    session_id: str,
    run_id: str,
    run_started_utc: str | None,
    encoder: str,
    audio: bool = True,
    analysis_span: float = ANALYSIS_SPAN,
) -> dict:
    """The sidecar the ranking pipeline reads, with the event located in it.

    `sampling` reads "trigger" rather than "uniform" so a later evaluation can
    separate clips whose analysis window was chosen from clips whose window was
    a guess. Mixing the two makes any comparison between them meaningless.
    """
    event_at = request.at - start  # where the event sits inside the clip
    duration = end - start
    half = analysis_span / 2
    latest = max(0.0, duration - analysis_span)
    analysis_start = min(max(0.0, event_at - half), latest)
    return {
        "schema": "salience.capture/1",
        "session_id": session_id,
        "run_id": run_id,
        "sampling": "coverage" if request.source == "coverage" else "trigger",
        "video_file": clip.name,
        "audio": audio,
        "run_started_utc": run_started_utc,
        "segment_start_sec": start,
        "segment_end_sec": end,
        "duration_sec": duration,
        "analysis_start_sec": round(analysis_start, 3),
        "analysis_end_sec": round(min(analysis_start + analysis_span, duration), 3),
        "event_sec": round(event_at, 3),
        "trigger_kind": request.kind,
        "trigger_source": request.source,
        "trigger_confidence": request.confidence,
        "trigger_detail": request.detail,
        "encoder": encoder,
    }


def harvest(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    lead: float = LEAD_SECONDS,
    lag: float = LAG_SECONDS,
    encoder: str = "h264_nvenc",
    bitrate: str = "8M",
    journal_name: str = JOURNAL_NAME,
    analysis_span: float = ANALYSIS_SPAN,
    runner=subprocess.run,
    probe_keyframes=None,
) -> list[dict]:
    """Cut every journalled trigger in a finished run. Never call during play.

    `journal_name` selects which journal to cut from, so a control set of
    randomly-timed windows can be produced by the same code that cuts real
    triggers. A control built by a different path would differ in clip length,
    analysis window or encoder settings, and any of those would bias the
    comparison it exists to make.
    """
    from pipeline_v2.jsonio import write_json

    requests = read_journal(run_dir / journal_name)
    segments = read_segments(run_dir)
    if not requests or not segments:
        return []

    out_dir = out_dir or (run_dir / "clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = run_dir.parent.name
    started_utc = _run_started(run_dir)
    interval = (probe_keyframes or keyframe_interval)(segments[0].path)

    saved: list[dict] = []
    for index, request in enumerate(sorted(requests, key=lambda r: r.at)):
        start, end = available_window(segments, request.at, lead, lag)
        if interval:
            # A little extra lead, never a clip that starts on nothing.
            start = snap_down(start, segments[0].start, interval)
        if end - start < min(analysis_span, ANALYSIS_SPAN):
            continue  # too little footage to describe; not worth a paid call
        covering = segments_covering(segments, start, end)
        if not covering:
            continue  # a gap in the footage; refuse rather than splice
        clip = out_dir / f"{index:03d}-{request.kind}.mkv"
        sidecar = clip.with_suffix(".capture.json")
        if clip.is_file() and sidecar.is_file():
            # Already cut. Resumable because a long harvest gets interrupted --
            # this one was, at index 47 of 73 -- and re-encoding an hour of
            # footage again to reach the tail is the expensive way to recover.
            # Keyed on the *sidecar*, which is written after a successful cut, so
            # a clip left half-written by a kill is redone rather than trusted.
            #
            # The name alone is not proof. `{index}-{kind}` is reused whenever a
            # journal changes, a control is redrawn with another seed, or the
            # lead and lag change -- and trusting it then reports a clip of one
            # moment as another. So the sidecar must record this exact window.
            try:
                previous = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = None  # unreadable sidecar: cut it again
            if previous is not None and _same_window(previous, request, start, end):
                saved.append(previous)
                continue
        listing = concat_list(covering, out_dir / f"{index:03d}.concat")
        result = runner(
            cut_command(
                listing, covering[0].start, start, end, clip,
                encoder=encoder, bitrate=bitrate, copy=bool(interval),
            ),
            capture_output=True,
            text=True,
        )
        listing.unlink(missing_ok=True)
        if getattr(result, "returncode", 1) != 0 or not clip.is_file():
            continue
        record = metadata(
            request, clip, start, end,
            session_id=session_id, run_id=run_dir.name,
            run_started_utc=started_utc, encoder=encoder,
            audio=_has_audio(clip), analysis_span=analysis_span,
        )
        write_json(clip.with_suffix(".capture.json"), record)
        saved.append(record)
    return saved


def cut_span(
    run_dir: Path,
    start: float,
    end: float,
    destination: Path,
    *,
    encoder: str = "h264_nvenc",
    bitrate: str = "8M",
    runner=subprocess.run,
    probe_keyframes=None,
) -> tuple[float, float] | None:
    """Cut one arbitrary span of a finished run; the span actually cut, or None.

    Clipped to footage that exists, snapped back to a keyframe when the run has
    short ones (then stream-copied), refused across a gap.
    """
    segments = read_segments(run_dir)
    if not segments:
        return None
    start = max(start, segments[0].start)
    end = min(end, segments[-1].end)
    interval = (probe_keyframes or keyframe_interval)(segments[0].path)
    if interval:
        start = snap_down(start, segments[0].start, interval)
    covering = segments_covering(segments, start, end)
    if end <= start or not covering:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    listing = concat_list(covering, destination.with_suffix(".concat"))
    result = runner(
        cut_command(listing, covering[0].start, start, end, destination,
                    encoder=encoder, bitrate=bitrate, copy=bool(interval)),
        capture_output=True, text=True,
    )
    listing.unlink(missing_ok=True)
    if getattr(result, "returncode", 1) != 0 or not destination.is_file():
        return None
    return start, end


def coverage_requests(
    segments: list[Segment], stride: float = COVERAGE_STRIDE
) -> list[ClipRequest]:
    """One request per `stride` seconds of recorded footage, centred so the
    analysis spans [k*stride, (k+1)*stride) tile the recording exactly.

    Only footage that exists and has no gap is covered; a trailing remainder
    shorter than half a stride is dropped rather than described half-empty.
    """
    requests: list[ClipRequest] = []
    if not segments:
        return requests
    runs: list[tuple[float, float]] = []
    for segment in segments:
        if runs and segment.start <= runs[-1][1] + 0.001:
            runs[-1] = (runs[-1][0], max(runs[-1][1], segment.end))
        else:
            runs.append((segment.start, segment.end))
    for first, last in runs:
        at = first + stride / 2
        while at <= last:  # at least half a stride of footage left
            requests.append(ClipRequest(round(at, 3), "coverage", "coverage", 0.0))
            at += stride
    return requests


def write_coverage(run_dir: Path, stride: float = COVERAGE_STRIDE) -> list[ClipRequest]:
    """(Re)write the coverage journal for a finished run and return it.

    Rewritten every time rather than appended: it is derived entirely from
    `segments.csv`, and harvest's resume check keeps already-cut windows.
    """
    requests = coverage_requests(read_segments(run_dir), stride)
    journal = run_dir / COVERAGE_JOURNAL
    journal.write_text(
        "".join(json.dumps(asdict(r)) + "\n" for r in requests), encoding="utf-8"
    )
    return requests


def _same_window(previous: dict, request: ClipRequest, start: float, end: float) -> bool:
    """Was this sidecar written for this request, cut over this window?"""
    try:
        return (
            abs(float(previous["segment_start_sec"]) - start) < 0.01
            and abs(float(previous["segment_end_sec"]) - end) < 0.01
            and previous.get("trigger_kind") == request.kind
            and previous.get("trigger_source") == request.source
        )
    except (KeyError, TypeError, ValueError):
        return False


def _has_audio(clip: Path) -> bool:
    """Whether the cut actually carries audio, rather than assuming it does.

    A `--no-audio` recording produces video-only clips, and a sidecar claiming
    audio sends the ranking pipeline's audio features after a track that is not
    there. Unknown is reported as False: claiming audio is the costlier mistake.
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(clip)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and bool(probe.stdout.strip())


def _run_started(run_dir: Path) -> str | None:
    """Reuse the wall-clock start an existing sidecar already recorded."""
    for sidecar in sorted(run_dir.glob("*.capture.json")):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("run_started_utc"):
            return payload["run_started_utc"]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut clips for journalled triggers.")
    parser.add_argument("run_dir", type=Path, help="a finished recording run")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lead", type=float, default=LEAD_SECONDS)
    parser.add_argument("--lag", type=float, default=LAG_SECONDS)
    parser.add_argument("--encoder", default="h264_nvenc")
    parser.add_argument(
        "--journal", default=JOURNAL_NAME,
        help="which journal to cut from (default: the live trigger journal)",
    )
    args = parser.parse_args(argv)

    saved = harvest(
        args.run_dir, args.out, lead=args.lead, lag=args.lag,
        encoder=args.encoder, journal_name=args.journal,
    )
    journalled = len(read_journal(args.run_dir / args.journal))
    print(f"{len(saved)} clip(s) cut from {journalled} trigger(s)")
    for record in saved:
        print(
            f"  {record['video_file']}  {record['duration_sec']:.1f}s"
            f"  event at {record['event_sec']:.1f}s"
            f"  [{record['trigger_source']}] {record['trigger_detail']}"
        )
    if journalled and not saved:
        print("  nothing cut: check the run has closed segments covering them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
