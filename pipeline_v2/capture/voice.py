"""What the player said, from the mic track, placed on the session timeline.

The vision model sees seven frames and hears nothing, so a moment that is
funny because of what someone said reads as "player stands still" and ranks
accordingly. The recorder keeps the mic on its own track, so speech can be
transcribed without gunfire mixed into it.

Not a detector and not a filter. The words become their own vectors, beside
the description's, and the player's ratings decide how much they count
(`ranking.choose_speech_weight`): someone who keeps the clips where everyone
was laughing gets speech weighed in, and someone playing in silence, or whose
talk is only callouts, gets the ranking they had before. A fixed list of "funny" phrases was the alternative, and
`speech.py` records why a fixed vocabulary measured badly.

One phrase is the exception, because it is a request, not a guess at what
is funny: "clip it" / "clip that" (`clip_requests`) keeps the moment just
before it as a highlight, whatever the ranking thought of it.

Local (Parakeet on the CPU, `audio.Transcriber`), cached per raw segment in
`mic_transcript.json`, so a re-run and a pruned session both keep what was
heard. `live` transcribes between matches, when the CPU is the game's least;
the session's end does the rest.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pipeline_v2.capture import clips as clips_mod

CACHE_NAME = "mic_transcript.json"
# `wgc` maps the game loopback first and the microphone second.
MIC_TRACK = 1
# Part of the cache: a transcript made another way is redone, not trusted.
SOURCE = "mic/parakeet-tdt-0.6b-v2-int8/norm-novad-words"
# Evens out a quiet mic before the speech model hears it. Per-moment
# gain, so timing is untouched and a shout is not made louder than it was.
LOUDNESS = "dynaudnorm=f=150:g=15"


@dataclass(frozen=True, order=True)
class Utterance:
    start: float  # seconds on the session timeline
    end: float
    text: str


# Segments transcribed per call. Under Whisper one call per 30s segment measured
# 8s each (its fixed per-call cost dominated); one call over 5 minutes, 2s per 30s.
# Chunked rather than whole-session so an interrupted run resumes.
CHUNK = 20


def _mic_wav(segments: list, destination: Path, runner=subprocess.run) -> Path | None:
    """The segments' mic track joined as 16kHz mono, or None without a mic track.

    Each segment is cut to its listed duration (`outpoint`), so position in
    the joined file maps back to session time exactly: the muxer's audio runs
    a few tens of milliseconds long per segment, which over a session would
    drift speech by seconds.
    """
    listing = destination.with_suffix(".txt")
    listing.write_text("".join(
        f"file '{s.path.resolve().as_posix()}'\noutpoint {s.end - s.start:.6f}\n"
        for s in segments
    ), encoding="utf-8")
    done = runner(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-map", f"0:a:{MIC_TRACK}", "-af", LOUDNESS, "-ac", "1", "-ar", "16000",
         str(destination)],
        capture_output=True,
    )
    if done.returncode != 0 or not destination.is_file():
        return None
    return destination


def _read_cache(path: Path) -> dict[str, list]:
    """Segment name -> utterances, or empty if it was made another way."""
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if cache.pop("_source", None) != SOURCE:
        return {}
    return cache


def _write_cache(path: Path, cache: dict[str, list]) -> None:
    path.write_text(json.dumps({"_source": SOURCE, **cache}), encoding="utf-8")


def _place(chunk: list, heard: list) -> dict[str, list]:
    """Utterances in joined-file time, filed under the segment each starts in,
    in session time."""
    placed: dict[str, list] = {s.path.name: [] for s in chunk}
    offsets, at = [], 0.0
    for segment in chunk:
        offsets.append((at, segment))
        at += segment.end - segment.start
    for start, end, text in heard:
        base, segment = next(
            (o for o in reversed(offsets) if o[0] <= start), offsets[0])
        shift = segment.start - base
        placed[segment.path.name].append([start + shift, end + shift, text])
    return placed


def transcribe_session(
    session: Path, *, transcriber=None, runner=subprocess.run, limit: int | None = None
) -> list[Utterance]:
    """Every utterance on the mic, transcribing only segments not yet cached
    (at most `limit` of them, so a caller between matches can stop early)."""
    cache_path = session / CACHE_NAME
    cache = _read_cache(cache_path)
    segments = clips_mod.read_segments(session)
    todo = [s for s in segments if s.path.name not in cache and s.path.is_file()]
    todo = todo[:limit] if limit is not None else todo
    if todo:
        if transcriber is None:
            from pipeline_v2.audio import Transcriber

            transcriber = Transcriber()
        with tempfile.TemporaryDirectory(prefix="salience-mic-") as tmp:
            for first in range(0, len(todo), CHUNK):
                chunk = todo[first:first + CHUNK]
                wav = _mic_wav(chunk, Path(tmp) / "mic.wav", runner)
                # Stored in session time: once raw segments are pruned, the
                # listing no longer says where each one started.
                cache.update(_place(chunk, [] if wav is None else transcriber.timed(wav)))
                _write_cache(cache_path, cache)
    return sorted(
        Utterance(start, end, text)
        for heard in cache.values()
        for start, end, text in heard
    )


def transcribed(session: Path) -> list[tuple[float, float]]:
    """The stretches of the session already transcribed, a raw segment each."""
    cache = _read_cache(session / CACHE_NAME)
    return [(s.start, s.end) for s in clips_mod.read_segments(session) if s.path.name in cache]


def heard(session: Path) -> list[Utterance]:
    """What is already transcribed, without transcribing anything."""
    return sorted(
        Utterance(start, end, text)
        for heard in _read_cache(session / CACHE_NAME).values()
        for start, end, text in heard
    )


# The player asking for the moment just gone: "clip it", "clip that!",
# "clip this". The model may punctuate between the words ("Clip, that.").
CLIP_COMMAND = re.compile(r"\bclip(?:ped)?[\s,.!?-]+(?:it|that|this)\b", re.IGNORECASE)


def clip_requests(utterances: list[Utterance]) -> list[float]:
    """When the player said "clip that": the end of each utterance asking."""
    return [u.end for u in utterances if CLIP_COMMAND.search(u.text)]


def said_between(utterances: list[Utterance], start: float, end: float) -> str:
    """Everything said that overlaps [start, end)."""
    return " ".join(u.text for u in utterances if u.start < end and start < u.end)


def attach(session: Path, clip_files: list[Path], *, transcriber=None,
           runner=subprocess.run, limit: int | None = None) -> int:
    """Store what was said in each clip. Returns how many transcripts changed.

    `limit` as for `transcribe_session`: 0 attaches what is already heard
    without loading the speech model."""
    from pipeline_v2 import cache, store

    placed = []
    for clip in clip_files:
        try:
            sidecar = json.loads(clip.with_suffix(".capture.json").read_text(encoding="utf-8"))
            placed.append((clip, float(sidecar["segment_start_sec"]),
                           float(sidecar["segment_end_sec"])))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not placed:
        return 0
    utterances = transcribe_session(session, transcriber=transcriber, runner=runner,
                                    limit=limit)
    changed = 0
    conn = store.connect()
    try:
        for clip, start, end in placed:
            changed += store.put_transcript(
                conn, cache.clip_sha(clip), said_between(utterances, start, end), SOURCE
            )
        conn.commit()
    finally:
        conn.close()
    return changed
