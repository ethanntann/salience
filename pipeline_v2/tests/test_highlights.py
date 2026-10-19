"""From a finished recording to the clips worth keeping.

These cover the ways a highlights folder can mislead: claiming taste it does not
have, leaking clips from another session or the control arm into the ranking,
and deleting raw footage before anything was kept from it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_v2 import store
from pipeline_v2.capture import clips, highlights

VARIANT = "v3.6-ownkills-window@minimax-m3"


@pytest.fixture(autouse=True)
def _deaf(monkeypatch):
    """No Whisper in unit tests; speech has its own tests in test_voice."""
    monkeypatch.setattr(highlights, "listen", lambda session, clip_dir: 0)
    # Nor the developer's real Fortnite log; gating has its own tests.
    monkeypatch.setattr(highlights, "find_gameplay", lambda session: {"spans": None})
    monkeypatch.setattr(highlights, "unembedded", lambda clip_dir: 0)
    monkeypatch.setattr(highlights, "read_hud", lambda session, runner=None: 0)


def _session(tmp_path: Path, outcomes: list[str]) -> Path:
    """A recorded session whose clips are already cut, described and stored."""
    session = tmp_path / "rec" / "run"
    clip_dir = session / "clips"
    clip_dir.mkdir(parents=True)
    for i in range(3):
        (session / f"part-{i:05d}.mkv").write_bytes(b"raw")
    (session / "segments.csv").write_text(
        "".join(f"part-{i:05d}.mkv,{i * 30}.0,{(i + 1) * 30}.0\n" for i in range(3)),
        encoding="utf-8",
    )
    conn = store.connect()
    for i, outcome in enumerate(outcomes):
        clips.append(session / clips.JOURNAL_NAME,
                     clips.ClipRequest(20.0 + i, "input_burst", "input_burst", 0.6))
        clip = clip_dir / f"{i:03d}-input_burst.mkv"
        clip.write_bytes(b"clip")
        clip.with_suffix(".capture.json").write_text(json.dumps(
            {"trigger_source": "input_burst", "trigger_confidence": 0.6 - i / 100}),
            encoding="utf-8")
        _store(conn, f"sha{session.name}{i}", clip, outcome)
    conn.commit()
    conn.close()
    return session


def _store(conn, sha: str, path: Path, outcome: str) -> None:
    store.upsert_clip(
        conn, clip_sha=sha, path=path, engagement=None, duration_sec=16.0,
        event_sec=12.0, event_source="capture_window", facts={},
        summary=f"a clip that ended in {outcome}",
        payload={"outcome": outcome}, prompt_version=VARIANT,
    )


def _run(session, **kwargs):
    return highlights.run(
        session, coverage=False,
        describe_fn=lambda *a, **k: None,
        harvest_fn=lambda s, d: [{} for _ in clips.read_journal(s / clips.JOURNAL_NAME)],
        **kwargs,
    )


def test_without_labels_the_basis_says_fallback_not_taste(tmp_path, isolated_data):
    """Claiming taste it does not have is worse than admitting it."""
    manifest = _run(_session(tmp_path, ["no_event", "elimination", "knock"]))

    assert manifest["basis"].startswith("fallback")
    assert "review --variant" in manifest["basis"]  # and says how to fix it
    assert [h["outcome"] for h in manifest["highlights"]] == [
        "elimination", "knock", "no_event"
    ]


def test_top_limits_what_is_kept(tmp_path, isolated_data):
    session = _session(tmp_path, ["elimination", "knock", "no_event"])

    manifest = _run(session, top=1)

    assert len(manifest["highlights"]) == 1
    kept = sorted(p.name for p in (session / "highlights").glob("*.mkv"))
    assert kept == ["01-000-input_burst.mkv"]  # 000 is the elimination


def test_clips_from_other_folders_never_leak_in(tmp_path, isolated_data):
    """The control arm and other sessions share the variant, not the ranking."""
    session = _session(tmp_path, ["no_event"])
    control = session / "control" / "000-control.mkv"
    control.parent.mkdir()
    control.write_bytes(b"clip")
    conn = store.connect()
    _store(conn, "control0", control, "elimination")
    conn.commit()
    conn.close()

    manifest = _run(session)

    assert [h["clip"] for h in manifest["highlights"]] == ["000-input_burst.mkv"]


def test_raw_footage_is_kept_unless_pruning_is_asked_for(tmp_path, isolated_data):
    session = _session(tmp_path, ["elimination"])

    _run(session)

    assert len(list(session.glob("part-*.mkv"))) == 3


def test_pruning_deletes_only_raw_segments(tmp_path, isolated_data):
    session = _session(tmp_path, ["elimination"])

    manifest = _run(session, prune_raw=True)

    assert manifest["pruned_raw"] is True
    assert list(session.glob("part-*.mkv")) == []
    assert (session / "clips" / "000-input_burst.mkv").is_file()  # untouched
    assert (session / clips.JOURNAL_NAME).is_file()
    assert len(list((session / "highlights").glob("*.mkv"))) == 1


def test_nothing_is_pruned_if_some_triggers_were_not_cut(tmp_path, isolated_data):
    """Deleting raw footage would make the missing clips unrecoverable."""
    session = _session(tmp_path, ["elimination", "knock"])

    manifest = highlights.run(
        session, prune_raw=True, coverage=False,
        describe_fn=lambda *a, **k: None,
        harvest_fn=lambda s, d: [{}],  # only one of two cut
    )

    assert "pruned_raw" not in manifest
    assert len(list(session.glob("part-*.mkv"))) == 3


def test_a_session_with_no_triggers_says_why(tmp_path, isolated_data):
    session = tmp_path / "empty"
    session.mkdir()

    manifest = highlights.run(session, coverage=False, describe_fn=lambda *a, **k: None)

    assert manifest["highlights"] == []
    assert manifest["why"] == "no triggers fired"


def test_coverage_needs_no_trigger_at_all(tmp_path, isolated_data):
    """The default path: every window of the session, whether or not a detector
    fired. Session 1's trigger overlapped only 5 of 14 kept moments."""
    session = _session(tmp_path, [])
    seen = {}

    def harvest(s, d, **kwargs):
        seen.update(kwargs, out=d)
        return []

    manifest = highlights.run(session, describe_fn=lambda *a, **k: None,
                              harvest_fn=harvest)

    assert manifest["mode"] == "coverage"
    assert manifest["journalled"] == len(clips.read_journal(
        session / clips.COVERAGE_JOURNAL)) > 0
    assert seen["journal_name"] == clips.COVERAGE_JOURNAL
    assert seen["analysis_span"] == clips.COVERAGE_STRIDE
    assert seen["out"] == session / "windows"


def test_coverage_with_no_footage_says_why(tmp_path, isolated_data):
    session = tmp_path / "empty"
    session.mkdir()

    manifest = highlights.run(session, describe_fn=lambda *a, **k: None)

    assert manifest["why"] == "no recorded footage"


class _Clip:
    def __init__(self, tmp_path, name, start):
        self.path = tmp_path / f"{name}.mkv"
        self.path.with_suffix(".capture.json").write_text(json.dumps(
            {"segment_start_sec": start, "segment_end_sec": start + 16}),
            encoding="utf-8")


def test_pick_never_keeps_two_views_of_one_moment(tmp_path):
    """Neighbouring coverage clips overlap by design; a reel of the same fight
    shifted by 8 seconds is two highlights of one moment."""
    a, b, c = (_Clip(tmp_path, n, s) for n, s in (("a", 100), ("b", 108), ("c", 300)))

    chosen = highlights.pick([(0.9, a), (0.8, b), (0.7, c)], top=5)

    assert [clip for _, clip in chosen] == [a, c]


def test_pick_stops_at_top(tmp_path):
    found = [(1.0 - i / 10, _Clip(tmp_path, str(i), i * 100)) for i in range(4)]

    assert len(highlights.pick(found, top=2)) == 2


def test_padding_keeps_back_to_back_windows_of_one_fight_apart(tmp_path):
    """Coverage windows do not overlap, but their padded exports would."""
    a, b = _Clip(tmp_path, "a", 100), _Clip(tmp_path, "b", 116)
    ordered = [(0.9, a), (0.8, b)]

    assert len(highlights.pick(ordered, top=5)) == 2
    assert len(highlights.pick(ordered, top=5, pad=(8.0, 2.0))) == 1


def test_coverage_windows_are_cut_to_their_span_only(tmp_path, isolated_data):
    """Windows cut to 16s on an 8s stride measured 3.8 GB for half an hour."""
    session = _session(tmp_path, [])
    seen = {}

    def harvest(s, d, **kwargs):
        seen.update(kwargs)
        return []

    highlights.run(session, describe_fn=lambda *a, **k: None, harvest_fn=harvest)

    assert seen["lead"] + seen["lag"] == clips.COVERAGE_STRIDE


def test_a_recut_window_is_ranked_by_its_new_description_only(tmp_path, isolated_data):
    """Same filename, new content: the old description is of footage that is gone."""
    session = _session(tmp_path, ["elimination"])
    clip = session / "clips" / "000-input_burst.mkv"
    conn = store.connect()
    _store(conn, "recut", clip, "no_event")
    conn.execute("update clips set described_at = described_at + 10 where clip_sha='recut'")
    conn.commit()

    _, inside = highlights._session_clips(conn, session / "clips")
    conn.close()

    assert [c.clip_sha for c in inside] == ["recut"]


def test_each_clip_is_ranked_once_even_if_two_fights_name_it(tmp_path, isolated_data, monkeypatch):
    from types import SimpleNamespace

    from pipeline_v2 import ranking

    session = _session(tmp_path, ["elimination"])
    conn = store.connect()
    _, inside = highlights._session_clips(conn, session / "clips")
    clip = inside[0]
    monkeypatch.setattr(ranking, "ranked_fights", lambda c, variant=None, info=None: (
        [SimpleNamespace(score=0.4, best=clip), SimpleNamespace(score=0.9, best=clip)], 12))

    basis, ordered = highlights.rank(conn, session / "clips")
    conn.close()

    assert "taste" in basis
    # Once, at its better fight's score (standardised: a lone clip scores 0).
    assert [c for _, c in ordered] == [clip]


def test_speech_changes_without_new_cuts_rebuild_embeddings(tmp_path, isolated_data):
    """A re-run that only heard new speech must still re-embed, or taste never sees it."""
    session = _session(tmp_path, ["knock"])
    rebuilt = []

    highlights.run(
        session, coverage=False, describe_fn=lambda *a, **k: None,
        harvest_fn=lambda s, d: [], listen_fn=lambda s, d: 1,
        reembed_fn=rebuilt.append,
    )

    assert rebuilt == [session / "clips"]


def test_no_speech_skips_listening(tmp_path, isolated_data):
    session = _session(tmp_path, ["knock"])
    heard = []

    highlights.run(
        session, coverage=False, speech=False, describe_fn=lambda *a, **k: None,
        harvest_fn=lambda s, d: [], listen_fn=lambda s, d: heard.append(s) or 0,
    )

    assert heard == []


def test_highlight_carries_what_was_said(tmp_path, isolated_data):
    session = _session(tmp_path, ["knock"])
    conn = store.connect()
    store.put_transcript(conn, f"sha{session.name}0", "no way that worked", "test")
    conn.commit()
    conn.close()

    manifest = _run(session)

    assert manifest["highlights"][0]["said"] == "no way that worked"


def test_rerun_describes_only_when_something_is_new(tmp_path, isolated_data):
    """Already-cut windows come back from harvest; re-describing them costs minutes."""
    session = _session(tmp_path, ["knock"])
    calls = []

    for fresh in (0, 2):
        highlights.run(
            session, coverage=False,
            describe_fn=lambda d, max_calls: calls.append(max_calls),
            harvest_fn=lambda s, d: [{}], undescribed_fn=lambda d, n=fresh: n,
        )

    assert calls == [2]


def test_windows_outside_gameplay_are_skipped_but_keep_their_numbers(tmp_path, isolated_data,
                                                                      monkeypatch):
    """Lobby and spectating are not described; ratings stay on the same filenames."""
    session = _session(tmp_path, [])
    monkeypatch.setattr(highlights, "find_gameplay",
                        lambda s: {"spans": [(20.0, 60.0)], "matches": 1})
    seen = {}

    def harvest(s, d, **kwargs):
        seen["skipped"] = [r.at for r in clips.read_journal(s / clips.COVERAGE_JOURNAL)
                           if kwargs["skip"](r)]
        return []

    manifest = highlights.run(session, describe_fn=lambda *a, **k: None, harvest_fn=harvest)

    assert seen["skipped"] == [4.0, 12.0, 68.0, 76.0, 84.0]
    assert manifest["gameplay"]["excluded_windows"] == 5


def test_a_win_outranks_kills_and_kills_outrank_taste():
    """Out of sample, what happened then taste beat taste alone (96.5% vs 86.9%)."""
    from types import SimpleNamespace

    win = SimpleNamespace(summary="A Victory Royale banner appears", payload={"outcome": "other"})
    kill = SimpleNamespace(summary="eliminates an opponent", payload={"outcome": "elimination"})
    walk = SimpleNamespace(summary="runs across a field", payload={"outcome": "no_event"})

    tiers = [highlights.event_tier(c) for c in (win, kill, walk)]

    assert tiers == sorted(tiers, reverse=True) and len(set(tiers)) == 3


def _windows(tmp_path, kinds):
    """Back-to-back 8s windows; each kind is 'kill', 'build' (fight tags) or 'calm'."""
    found = []
    for i, kind in enumerate(kinds):
        clip = tmp_path / f"{i:03d}-coverage.mkv"
        clip.write_bytes(b"x")
        clip.with_suffix(".capture.json").write_text(
            json.dumps({"segment_start_sec": i * 8.0, "segment_end_sec": i * 8.0 + 8}),
            encoding="utf-8")
        payload = {"outcome": "elimination" if kind == "kill" else "no_event",
                   "mechanics": ["build_fight"] if kind == "build" else []}
        found.append((0.0, SimpleNamespace(path=str(clip), payload=payload, summary="")))
    return found


def test_a_highlight_grows_over_the_rest_of_its_fight(tmp_path):
    ordered = _windows(tmp_path, ["calm", "build", "build", "kill", "build", "calm"])
    index = highlights.window_index(ordered)

    span = highlights.fight_extent(ordered[3][1], index, pad=(8.0, 2.0))

    # Windows 1-4 are the fight (8s to 40s); the lead-in reaches back into 0.
    assert span == (8.0 - 8.0, 40.0 + 2.0)


def test_growth_stops_at_the_length_cap(tmp_path):
    ordered = _windows(tmp_path, ["build"] * 10)
    span = highlights.fight_extent(ordered[0][1], highlights.window_index(ordered), (8.0, 2.0))

    assert span[1] - span[0] <= highlights.MAX_HIGHLIGHT


def test_a_calm_window_is_exported_as_it_was(tmp_path):
    ordered = _windows(tmp_path, ["build", "calm", "build"])
    span = highlights.fight_extent(ordered[1][1], highlights.window_index(ordered), (8.0, 2.0))

    assert span == (8.0 - 8.0, 16.0 + 2.0)


def test_two_highlights_never_share_a_fight(tmp_path):
    ordered = _windows(tmp_path, ["kill", "build", "kill", "calm", "calm", "calm", "calm", "kill"])
    index = highlights.window_index(ordered)

    picked = highlights.pick(ordered, 5, (8.0, 2.0),
                             lambda c: highlights.fight_extent(c, index, (8.0, 2.0)))

    names = [Path(c.path).name for _, c in picked]
    assert "000-coverage.mkv" in names and "002-coverage.mkv" not in names


def test_a_win_right_after_a_long_fight_joins_it(tmp_path):
    # A fight long enough to reach the cap at 32s, then the win's banner.
    ordered = _windows(tmp_path, ["build"] * 4 + ["calm"] * 2)
    ordered[4][1].summary = "The Victory Royale banner appears"
    index = highlights.window_index(ordered)

    span = highlights.fight_extent(ordered[0][1], index, (8.0, 2.0))

    assert span == (-8.0, 40.0 + 2.0)  # through the banner window, past the cap
    assert span[1] - span[0] > highlights.MAX_HIGHLIGHT
