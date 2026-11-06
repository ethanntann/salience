"""Turning a trigger into a clip.

The half of the capture system that was missing: detectors produced triggers and
nothing cut footage for them. These cover the parts that can lie quietly --
window placement near the edges of a run, gaps in the footage, and the metadata
that tells the ranking pipeline where the event actually was.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline_v2.capture import clips
from pipeline_v2.capture.clips import ClipRequest, Segment
from pipeline_v2.capture.triggers import Trigger


@dataclass
class FakeResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _segments(tmp_path: Path, count: int, length: float = 30.0):
    made = []
    for i in range(count):
        path = tmp_path / f"part-{i:05d}.mkv"
        path.write_bytes(b"footage")
        made.append(Segment(path, i * length, (i + 1) * length))
    return made


def _write_segments_csv(tmp_path: Path, segments):
    rows = [f"{s.path.name},{s.start:.6f},{s.end:.6f}" for s in segments]
    (tmp_path / "segments.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_a_trigger_is_converted_to_run_relative_time():
    """A monotonic clock means nothing once the process exits."""
    trigger = Trigger(kind="voice_reaction", at=5_000.5, source="voice")

    request = ClipRequest.from_trigger(trigger, run_started=4_900.0)

    assert request.at == 100.5
    assert request.kind == "voice_reaction"


def test_triggers_survive_a_killed_session(tmp_path):
    journal = tmp_path / clips.JOURNAL_NAME
    clips.append(journal, ClipRequest(at=12.0, kind="voice_reaction", source="voice"))
    clips.append(journal, ClipRequest(at=40.0, kind="input_burst", source="input"))

    found = clips.read_journal(journal)

    assert [r.at for r in found] == [12.0, 40.0]
    assert [r.kind for r in found] == ["voice_reaction", "input_burst"]


def test_a_torn_final_line_is_skipped(tmp_path):
    """Normal after a TerminateProcess kill mid-write."""
    journal = tmp_path / clips.JOURNAL_NAME
    clips.append(journal, ClipRequest(at=12.0, kind="voice_reaction", source="voice"))
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"at": 40.0, "kind": "inp')

    assert [r.at for r in clips.read_journal(journal)] == [12.0]


def test_a_journal_failure_does_not_raise(tmp_path):
    """Telemetry must never interrupt a recording."""
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory", encoding="utf-8")

    clips.append(blocked / "triggers.jsonl", ClipRequest(1.0, "k", "s"))


def test_the_window_spans_only_the_chunks_it_touches(tmp_path):
    segments = _segments(tmp_path, 4)

    covering = clips.segments_covering(segments, 44.0, 60.0)

    assert [s.start for s in covering] == [30.0]


def test_a_window_crossing_a_boundary_takes_both_chunks(tmp_path):
    segments = _segments(tmp_path, 4)

    covering = clips.segments_covering(segments, 55.0, 71.0)

    assert [s.start for s in covering] == [30.0, 60.0]


def test_a_gap_in_the_footage_is_refused_rather_than_spliced(tmp_path):
    """Concatenating across a missing chunk yields a clip that lies."""
    segments = _segments(tmp_path, 4)
    without_middle = [s for s in segments if s.start != 30.0]

    assert clips.segments_covering(without_middle, 10.0, 70.0) == []


def test_a_trigger_early_in_a_run_gets_a_shorter_lead(tmp_path):
    """Twelve seconds of lead cannot exist three seconds into a run."""
    segments = _segments(tmp_path, 2)

    start, end = clips.available_window(segments, at=3.0, lead=12.0, lag=4.0)

    assert start == 0.0  # not -9.0
    assert end == 7.0


def test_a_trigger_at_the_end_of_a_run_gets_a_shorter_lag(tmp_path):
    segments = _segments(tmp_path, 2)  # 60s of footage

    start, end = clips.available_window(segments, at=59.0, lead=12.0, lag=4.0)

    assert start == 47.0
    assert end == 60.0  # not 63.0


def test_the_analysis_window_is_placed_on_the_event_not_guessed():
    """The whole point of a triggered clip over a uniform sample."""
    request = ClipRequest(at=100.0, kind="voice_reaction", source="voice")

    record = clips.metadata(
        request, Path("000-voice_reaction.mkv"), start=88.0, end=104.0,
        session_id="s", run_id="r", run_started_utc=None, encoder="h264_nvenc",
    )

    assert record["event_sec"] == 12.0
    assert record["analysis_start_sec"] == 10.0  # centred on the event
    assert record["analysis_end_sec"] == 14.0
    assert record["sampling"] == "trigger"  # separable from uniform samples


def test_the_analysis_window_stays_inside_a_short_clip():
    request = ClipRequest(at=5.0, kind="input_burst", source="input")

    record = clips.metadata(
        request, Path("c.mkv"), start=0.0, end=6.0,
        session_id="s", run_id="r", run_started_utc=None, encoder="h264_nvenc",
    )

    assert record["analysis_start_sec"] >= 0.0
    assert record["analysis_end_sec"] <= record["duration_sec"]


def test_harvest_cuts_a_clip_and_writes_its_sidecar(tmp_path):
    import json

    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    segments = _segments(run, 4)
    _write_segments_csv(run, segments)
    clips.append(
        run / clips.JOURNAL_NAME,
        ClipRequest(at=100.0, kind="voice_reaction", source="voice", detail="+6dB"),
    )

    commands = []

    def runner(command, **_):
        commands.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    saved = clips.harvest(run, runner=runner)

    assert len(saved) == 1
    assert saved[0]["event_sec"] == 12.0
    assert saved[0]["trigger_detail"] == "+6dB"
    sidecar = run / "clips" / "000-voice_reaction.capture.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["sampling"] == "trigger"
    # The concat list is cleaned up; a stale one would be fed to a later cut.
    assert list((run / "clips").glob("*.concat")) == []


def test_harvest_keeps_every_stream_so_the_mic_track_survives(tmp_path):
    """`-c copy` without `-map 0` silently dropped the microphone track once."""
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 4))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))

    commands = []

    def runner(command, **_):
        commands.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    clips.harvest(run, runner=runner)

    assert "-map" in commands[0]
    assert commands[0][commands[0].index("-map") + 1] == "0"


def test_a_failed_cut_writes_no_sidecar(tmp_path):
    """A sidecar without footage would send the ranking pipeline to a dead path."""
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 4))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))

    saved = clips.harvest(run, runner=lambda *a, **k: FakeResult(returncode=1))

    assert saved == []
    assert list(run.glob("clips/*.capture.json")) == []


def test_a_run_with_no_triggers_does_nothing(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 2))

    assert clips.harvest(run, runner=lambda *a, **k: FakeResult()) == []


def test_a_trigger_with_too_little_footage_is_skipped(tmp_path):
    """Below the analysis span there is nothing for the model to describe."""
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    short = [Segment(run / "part-00000.mkv", 0.0, 2.0)]
    short[0].path.write_bytes(b"footage")
    _write_segments_csv(run, short)
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(1.0, "voice_reaction", "v"))

    assert clips.harvest(run, runner=lambda *a, **k: FakeResult()) == []


def test_an_interrupted_harvest_resumes_instead_of_recutting(tmp_path):
    """A long harvest gets interrupted -- one did, at index 47 of 73.

    Re-encoding everything again to reach the tail is the expensive recovery.
    """
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 6))
    for at in (100.0, 160.0):
        clips.append(run / clips.JOURNAL_NAME, ClipRequest(at, "voice_reaction", "v"))

    calls = []

    def runner(command, **_):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    first = clips.harvest(run, runner=runner)
    assert len(first) == 2
    assert len(calls) == 2

    again = clips.harvest(run, runner=runner)

    assert len(again) == 2  # both still reported
    assert len(calls) == 2  # but nothing was re-encoded


def test_a_clip_without_its_sidecar_is_cut_again(tmp_path):
    """The sidecar is written after a successful cut, so it is the proof.

    A clip left half-written by a kill has no sidecar and must not be trusted.
    """
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 4))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))

    calls = []

    def runner(command, **_):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    clips.harvest(run, runner=runner)
    (run / "clips" / "000-voice_reaction.capture.json").unlink()
    clips.harvest(run, runner=runner)

    assert len(calls) == 2  # the half-written clip was redone


def test_an_unreadable_sidecar_is_cut_again(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 4))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))

    calls = []

    def runner(command, **_):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    clips.harvest(run, runner=runner)
    (run / "clips" / "000-voice_reaction.capture.json").write_text("{tor", encoding="utf-8")

    saved = clips.harvest(run, runner=runner)

    assert len(calls) == 2
    assert len(saved) == 1


def test_a_control_set_is_cut_by_the_same_code_as_real_triggers(tmp_path):
    """A control built by a different path would not be a control.

    Clip length, analysis window and encoder settings all have to match, or the
    comparison the control exists to make is biased by the machinery.
    """
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 6))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))
    clips.append(run / "control.jsonl", ClipRequest(100.0, "control", "random"))

    def runner(command, **_):
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    real = clips.harvest(run, run / "real", runner=runner)
    control = clips.harvest(
        run, run / "control", journal_name="control.jsonl", runner=runner
    )

    assert len(real) == len(control) == 1
    for field in ("duration_sec", "analysis_start_sec", "analysis_end_sec", "event_sec"):
        assert real[0][field] == control[0][field]
    assert control[0]["trigger_kind"] == "control"  # but still distinguishable


def test_a_missing_journal_yields_nothing_rather_than_raising(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 2))

    assert clips.harvest(run, journal_name="absent.jsonl",
                         runner=lambda *a, **k: FakeResult()) == []


def test_a_clip_cut_for_a_different_window_is_not_reused(tmp_path):
    """Same filename, different moment: the name alone is not proof.

    Redrawing a control with another seed, or changing lead and lag, reuses
    `{index}-{kind}`. Trusting the old sidecar would report one moment as another.
    """
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 6))
    journal = run / clips.JOURNAL_NAME
    clips.append(journal, ClipRequest(100.0, "voice_reaction", "v"))

    calls = []

    def runner(command, **_):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    clips.harvest(run, runner=runner)
    journal.unlink()
    clips.append(journal, ClipRequest(140.0, "voice_reaction", "v"))  # new moment

    saved = clips.harvest(run, runner=runner)

    assert len(calls) == 2  # re-cut, not reused
    assert saved[0]["segment_start_sec"] == 128.0


def test_changing_the_lead_recuts_rather_than_reusing(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 6))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))

    calls = []

    def runner(command, **_):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    clips.harvest(run, runner=runner)
    saved = clips.harvest(run, lead=20.0, runner=runner)

    assert len(calls) == 2
    assert saved[0]["segment_start_sec"] == 80.0


def test_a_fake_clip_with_no_audio_stream_is_not_claimed_to_have_audio(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 4))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))

    def runner(command, **_):
        Path(command[-1]).write_bytes(b"not a real video")
        return FakeResult()

    [saved] = clips.harvest(run, runner=runner)

    assert saved["audio"] is False


def test_coverage_spans_tile_the_footage_with_no_gap(tmp_path):
    """Full coverage is only full if consecutive analysis spans meet exactly."""
    requests = clips.coverage_requests(_segments(tmp_path, 2), stride=8.0)
    spans = [(r.at - 4.0, r.at + 4.0) for r in requests]

    assert spans[0][0] == 0.0
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert spans[-1][1] >= 60.0 - 4.0  # nothing but a sub-half-stride tail left out
    assert {r.source for r in requests} == {"coverage"}


def test_coverage_does_not_bridge_a_gap_in_the_footage(tmp_path):
    first = tmp_path / "a.mkv"
    second = tmp_path / "b.mkv"
    segments = [Segment(first, 0.0, 30.0), Segment(second, 40.0, 70.0)]

    requests = clips.coverage_requests(segments, stride=8.0)

    assert not any(30.0 < r.at < 40.0 for r in requests)
    assert any(r.at == 44.0 for r in requests)  # restarts on the far side


def test_a_coverage_clip_is_described_across_its_whole_window(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 2))
    clips.write_coverage(run)

    def runner(command, **_):
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    saved = clips.harvest(
        run, run / "windows", journal_name=clips.COVERAGE_JOURNAL,
        analysis_span=clips.COVERAGE_STRIDE, runner=runner,
    )

    middle = saved[3]
    assert middle["sampling"] == "coverage"
    assert middle["analysis_end_sec"] - middle["analysis_start_sec"] == 8.0
    assert middle["event_sec"] - middle["analysis_start_sec"] == 4.0


def test_keyframed_footage_is_copied_from_a_snapped_start(tmp_path):
    """Recordings with a 2s keyframe are cut without re-encoding, and the start
    moves *back* to a keyframe: off the grid, a copy cut starts on nothing."""
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 4))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(at=101.0, kind="k", source="s"))
    commands = []

    def runner(command, **_):
        commands.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    saved = clips.harvest(run, runner=runner, probe_keyframes=lambda _: 2.0)

    assert saved[0]["segment_start_sec"] == 88.0  # 101 - 12 = 89, snapped to 88
    assert saved[0]["event_sec"] == 13.0          # the event did not move
    assert "copy" in commands[0][commands[0].index("-c") + 1]
    assert "h264_nvenc" not in commands[0]


def test_footage_without_short_keyframes_is_still_re_encoded(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 4))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(at=101.0, kind="k", source="s"))
    commands = []

    def runner(command, **_):
        commands.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    saved = clips.harvest(run, runner=runner, probe_keyframes=lambda _: None)

    assert saved[0]["segment_start_sec"] == 89.0
    assert "h264_nvenc" in commands[0]


def test_keyframe_interval_rejects_one_keyframe_per_chunk():
    def probe(times):
        return lambda *a, **k: FakeResult(stdout="\n".join(map(str, times)))

    assert clips.keyframe_interval(Path("x"), runner=probe([0, 2, 4, 6])) == 2.0
    assert clips.keyframe_interval(Path("x"), runner=probe([0.0])) is None
    assert clips.keyframe_interval(Path("x"), runner=probe([0, 2, 7, 9])) is None


def test_a_span_is_cut_from_a_snapped_keyframe_and_clipped_to_footage(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 2))
    commands = []

    def runner(command, **_):
        commands.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    cut = clips.cut_span(run, 41.0, 75.0, run / "h.mkv", runner=runner,
                         probe_keyframes=lambda _: 2.0)

    assert cut == (40.0, 60.0)  # back to a keyframe; no footage past 60s
    assert "copy" in commands[0]


def test_a_span_with_no_footage_is_refused(tmp_path):
    assert clips.cut_span(tmp_path, 0.0, 10.0, tmp_path / "h.mkv") is None


def _released_run(tmp_path: Path):
    """A run with one window cut, its copy then released."""
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 6))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))
    calls = []

    def runner(command, **_):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    [record] = clips.harvest(run, runner=runner)
    clip = run / "clips" / record["video_file"]
    assert clips.release_window(clip, "sha-of-the-copy")
    return run, clip, runner, calls


def test_a_released_window_keeps_its_sidecar_and_its_hash(tmp_path):
    _, clip, _, _ = _released_run(tmp_path)

    assert not clip.is_file() and clip.with_suffix(".capture.json").is_file()
    assert clips.released(clip)
    assert clips.window_sha(clip) == "sha-of-the-copy"


def test_a_released_window_is_never_cut_again(tmp_path):
    """A re-cut is a new hash: a new window, described and paid for twice."""
    run, clip, runner, calls = _released_run(tmp_path)

    again = clips.harvest(run, runner=runner)

    assert len(again) == 1 and len(calls) == 1
    assert not clip.is_file()


def test_a_window_is_kept_while_its_raw_footage_is_not_all_there(tmp_path):
    run = tmp_path / "session" / "run"
    run.mkdir(parents=True)
    _write_segments_csv(run, _segments(run, 6))
    clips.append(run / clips.JOURNAL_NAME, ClipRequest(100.0, "voice_reaction", "v"))

    def runner(command, **_):
        Path(command[-1]).write_bytes(b"clip")
        return FakeResult()

    [record] = clips.harvest(run, runner=runner)
    clip = run / "clips" / record["video_file"]
    (run / "part-00003.mkv").unlink()  # under the window at 100s

    assert not clips.release_window(clip, "sha")
    assert clip.is_file() and not clips.released(clip)


def test_a_released_window_is_recut_from_its_own_span(tmp_path):
    _, clip, runner, calls = _released_run(tmp_path)
    record = clips._read_sidecar(clip)

    out = clips.recut_window(clip, tmp_path / "watch.mkv", runner=runner)

    assert out == tmp_path / "watch.mkv" and out.is_file()
    command = calls[-1]
    start = float(command[command.index("-ss") + 1])
    length = float(command[command.index("-t") + 1])
    assert round(length, 3) == round(record["segment_end_sec"] - record["segment_start_sec"], 3)
    assert start >= 0
