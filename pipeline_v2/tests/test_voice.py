"""What was said, on the right clip, without re-running Whisper or inventing speech."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pipeline_v2 import cache, store
from pipeline_v2.capture import voice


class FakeWhisper:
    """Answers each call with the next list, in joined-file time."""

    def __init__(self, *answers: list):
        self.answers = list(answers)
        self.calls = 0

    def timed(self, wav: Path):
        self.calls += 1
        return self.answers.pop(0) if self.answers else []


def _session(tmp_path: Path, segments: int = 2, gap: float = 0.0) -> Path:
    """Segments of 30s; `gap` seconds of no footage between them."""
    session = tmp_path / "run"
    session.mkdir()
    rows = []
    for i in range(segments):
        (session / f"part-{i:05d}.mkv").write_bytes(b"x" * (i + 1))
        start = i * (30.0 + gap)
        rows.append(f"part-{i:05d}.mkv,{start},{start + 30.0}\n")
    (session / "segments.csv").write_text("".join(rows), encoding="utf-8")
    return session


def _ffmpeg(command, **_):
    Path(command[-1]).write_bytes(b"wav")
    return subprocess.CompletedProcess(command, 0)


def test_utterances_land_on_the_session_timeline(tmp_path):
    """Joined-file time maps back across a gap in the footage."""
    session = _session(tmp_path, gap=10.0)
    whisper = FakeWhisper([(2.0, 3.0, "hello"), (35.0, 36.5, "no way")])

    heard = voice.transcribe_session(session, transcriber=whisper, runner=_ffmpeg)

    assert heard == [voice.Utterance(2.0, 3.0, "hello"), voice.Utterance(45.0, 46.5, "no way")]


def test_segments_are_joined_into_one_call_per_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "CHUNK", 2)
    session = _session(tmp_path, segments=3)
    whisper = FakeWhisper([(1.0, 2.0, "a")], [(1.0, 2.0, "b")])

    heard = voice.transcribe_session(session, transcriber=whisper, runner=_ffmpeg)

    assert whisper.calls == 2
    assert heard == [voice.Utterance(1.0, 2.0, "a"), voice.Utterance(61.0, 62.0, "b")]


def test_cached_segments_are_not_transcribed_again(tmp_path):
    session = _session(tmp_path)
    whisper = FakeWhisper([(2.0, 3.0, "hello")])
    voice.transcribe_session(session, transcriber=whisper, runner=_ffmpeg)

    again = voice.transcribe_session(session, transcriber=whisper, runner=_ffmpeg)

    assert whisper.calls == 1
    assert again[0].text == "hello"


def test_pruned_segments_keep_what_was_heard(tmp_path):
    session = _session(tmp_path)
    voice.transcribe_session(
        session, transcriber=FakeWhisper([(31.0, 32.0, "lol")]), runner=_ffmpeg)
    for part in session.glob("part-*.mkv"):
        part.unlink()

    heard = voice.transcribe_session(session, transcriber=None, runner=_ffmpeg)

    assert heard == [voice.Utterance(31.0, 32.0, "lol")]


def test_no_mic_track_is_silence_not_an_error(tmp_path):
    session = _session(tmp_path, segments=1)

    def no_track(command, **_):
        return subprocess.CompletedProcess(command, 1)

    assert voice.transcribe_session(
        session, transcriber=FakeWhisper(), runner=no_track) == []
    assert json.loads((session / voice.CACHE_NAME).read_text()) == {
        "_source": voice.SOURCE, "part-00000.mkv": []}


def test_a_transcript_made_the_old_way_is_redone(tmp_path):
    """Before loudness levelling, quiet callouts were lost; trusting that
    cache would keep them lost."""
    session = _session(tmp_path, segments=1)
    (session / voice.CACHE_NAME).write_text(
        json.dumps({"part-00000.mkv": [[1.0, 2.0, "the building bro"]]}), encoding="utf-8")

    def no_track(command, **_):
        return subprocess.CompletedProcess(command, 1)

    assert voice.transcribe_session(
        session, transcriber=FakeWhisper(), runner=no_track) == []


def test_said_between_takes_overlapping_speech_only():
    heard = [voice.Utterance(1, 2, "before"), voice.Utterance(7.5, 9, "across"),
             voice.Utterance(20, 21, "after")]

    assert voice.said_between(heard, 8.0, 16.0) == "across"


def test_attach_stores_speech_per_clip(tmp_path, isolated_data):
    session = _session(tmp_path)
    windows = session / "windows"
    windows.mkdir()
    loud, quiet = windows / "a.mkv", windows / "b.mkv"
    for clip, (start, end) in ((loud, (32.0, 40.0)), (quiet, (0.0, 8.0))):
        clip.write_bytes(clip.name.encode())
        clip.with_suffix(".capture.json").write_text(json.dumps(
            {"segment_start_sec": start, "segment_end_sec": end}), encoding="utf-8")

    changed = voice.attach(session, [loud, quiet],
                           transcriber=FakeWhisper([(34.0, 35.0, "let's go")]),
                           runner=_ffmpeg)

    conn = store.connect()
    assert store.transcripts(conn) == {cache.clip_sha(loud): "let's go"}
    conn.close()
    assert changed == 1  # an empty transcript is recorded but changes nothing


def test_speech_minutes_apart_is_two_utterances_not_one_long_one():
    """Whisper returned "But it's ... walking retard though" as one segment
    from 922s to 1036s, which put it in every window of those two minutes."""
    from types import SimpleNamespace as W

    from pipeline_v2.audio import utterances

    words = [W(start=922.2, end=922.8, word=" But"), W(start=922.8, end=923.4, word=" it's"),
             W(start=1034.4, end=1035.4, word=" walking"), W(start=1035.4, end=1036.6, word=" on.")]

    assert utterances(words) == [(922.2, 923.4, "But it's"), (1034.4, 1036.6, "walking on.")]


@pytest.mark.parametrize("text", ["Clip it!", "clip that", "Oh my god, clip that.",
                                  "CLIP THIS", "Clip, that.", "clipped it"])
def test_asking_for_a_clip_is_heard(text):
    assert voice.clip_requests([voice.Utterance(10.0, 11.5, text)]) == [11.5]


@pytest.mark.parametrize("text", ["he clipped through the wall", "a clip of ammo",
                                  "clipping the edge", "eclipse it"])
def test_other_talk_about_clips_is_not_asking(text):
    assert voice.clip_requests([voice.Utterance(10.0, 11.5, text)]) == []
