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
_MAKE_REEL = highlights.make_reel  # before the fixture stubs it
_MIX_TRACKS = highlights.mix_tracks


@pytest.fixture(autouse=True)
def _deaf(monkeypatch):
    """No Whisper in unit tests; speech has its own tests in test_voice."""
    monkeypatch.setattr(highlights, "listen", lambda session, clip_dir: 0)
    monkeypatch.setattr(highlights, "hear_laughs", lambda session: [])
    # Nor the developer's real Fortnite log; gating has its own tests.
    monkeypatch.setattr(highlights, "find_gameplay", lambda session: {"spans": None})
    monkeypatch.setattr(highlights, "unembedded", lambda clip_dir: 0)
    # Nor NVENC; reels have their own test.
    monkeypatch.setattr(highlights, "make_reel", lambda source, destination: False)
    monkeypatch.setattr(highlights, "mix_tracks", lambda path: False)


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
    assert [h["outcome"] for h in manifest["highlights"]] == ["elimination", "knock"]


def test_top_limits_what_is_kept(tmp_path, isolated_data):
    session = _session(tmp_path, ["elimination", "knock", "no_event"])

    manifest = _run(session, top=1)

    assert len(manifest["highlights"]) == 1
    kept = sorted(p.name for p in (session / "highlights").glob("*.mkv"))
    assert kept == ["01-000-input_burst.mkv"]  # 000 is the elimination


def test_clips_from_other_folders_never_leak_in(tmp_path, isolated_data):
    """The control arm and other sessions share the variant, not the ranking."""
    session = _session(tmp_path, ["knock"])
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


def test_a_win_ranks_with_kills_and_both_outrank_the_rest():
    """A win was kept as often as a kill; above kills it cost 3 of 32 top-10 keeps."""
    from types import SimpleNamespace

    win = SimpleNamespace(summary="A Victory Royale banner appears", payload={"outcome": "other"})
    kill = SimpleNamespace(summary="eliminates an opponent", payload={"outcome": "elimination"})
    walk = SimpleNamespace(summary="runs across a field", payload={"outcome": "no_event"})

    tiers = [highlights.event_tier(c) for c in (win, kill, walk)]

    assert tiers[0] == tiers[1] > tiers[2]


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


def test_clip_that_means_the_best_window_just_before_it(tmp_path):
    ordered = _windows(tmp_path, ["calm", "kill", "calm", "calm", "calm"])
    ordered = [ordered[1], ordered[0], *ordered[2:]]  # the kill ranks first
    kill = ordered[0][1]

    # Said at 22s, in the window after the kill (8-16s): the kill is meant.
    assert highlights.asked_for(ordered, [22.0]) == [kill]
    # Nothing recorded there (lobby): nothing is meant.
    assert highlights.asked_for(ordered, [500.0]) == []


def test_a_clip_asked_for_is_kept_even_below_the_top(tmp_path):
    ordered = _windows(tmp_path, ["kill", "calm", "calm", "kill", "calm", "calm", "calm"])
    asked = ordered[6][1]

    picked = [c for _, c in highlights.pick(ordered, 1, pinned=[asked])]

    assert picked == [asked]
    picked = [c for _, c in highlights.pick(ordered, 2, pinned=[asked])]
    assert picked == [asked, ordered[0][1]]


def test_asking_twice_for_one_moment_keeps_it_once(tmp_path):
    ordered = _windows(tmp_path, ["kill", "calm", "calm", "calm"])

    picked = highlights.pick(ordered, 2, pinned=highlights.asked_for(ordered, [4.0, 6.0]))

    assert [c for _, c in picked].count(ordered[0][1]) == 1


def test_clip_that_in_the_lobby_is_cut_from_the_footage(tmp_path):
    ordered = _windows(tmp_path, ["kill", "calm"])  # gameplay 0-16s

    # Inside gameplay: a window means it. In the lobby at 100s and 104s: one cut.
    spans = highlights.unwindowed(ordered, [10.0, 100.0, 104.0])

    lead, lag = highlights.ASK_SPAN
    assert spans == [(100.0 - lead, 104.0 + lag)]


def test_a_lobby_clip_that_is_kept_first_with_what_was_said(tmp_path, isolated_data, monkeypatch):
    from pipeline_v2.capture import voice

    session = _session(tmp_path, ["elimination", "no_event"])
    (session / voice.CACHE_NAME).write_text(json.dumps(
        {"_source": voice.SOURCE, "part-00001.mkv": [[48.0, 49.5, "Bro look at him, clip that"]]}),
        encoding="utf-8")
    cuts = []

    def cut(run_dir, start, end, destination):
        cuts.append((start, end))
        destination.write_bytes(b"cut")
        return start, end

    monkeypatch.setattr(clips, "cut_span", cut)

    manifest = _run(session)

    first = manifest["highlights"][0]
    assert first["asked"] and first["outcome"] == "not_described"
    assert first["said"] == "Bro look at him, clip that"
    assert cuts == [(49.5 - highlights.ASK_SPAN[0], 49.5 + highlights.ASK_SPAN[1])]
    assert (session / "highlights" / first["clip"]).is_file()
    assert len(manifest["highlights"]) == 2  # the lobby window itself is no moment


def test_a_reel_is_cropped_to_9_16_with_both_tracks_mixed(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="1\n2\n")
        Path(command[-1]).write_bytes(b"reel")
        return SimpleNamespace(returncode=0)

    reel = tmp_path / "reels" / "01-x.mp4"
    assert _MAKE_REEL(tmp_path / "01-x.mkv", reel, runner=runner)

    graph = calls[1][calls[1].index("-filter_complex") + 1]
    assert "crop=trunc(ih*1080/1920/2)*2:ih" in graph and "scale=1080:1920" in graph
    assert "[0:a:0][0:a:1]amix=inputs=2" in graph
    assert reel.is_file()


def test_a_highlight_plays_the_game_and_the_mic_mixed_by_default(tmp_path):
    """A player plays the default track only; the mic was its own, unheard."""
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="1\n2\n")
        Path(command[-1]).write_bytes(b"mixed")
        return SimpleNamespace(returncode=0)

    clip = tmp_path / "01-x.mkv"
    clip.write_bytes(b"two tracks")

    assert _MIX_TRACKS(clip, runner=runner)

    command = calls[1]
    assert "[0:a:0][0:a:1]amix=inputs=2" in command[command.index("-filter_complex") + 1]
    maps = [command[i + 1] for i, arg in enumerate(command) if arg == "-map"]
    assert maps == ["0:v", "[mix]", "0:a"]  # the mix first, the separate tracks kept
    assert "-disposition:a:0" in command and f"title={highlights.MIXED_TITLE}" in command
    assert clip.read_bytes() == b"mixed"


def test_an_already_mixed_highlight_is_left_and_its_reel_uses_the_mix(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=f"1,{highlights.MIXED_TITLE}\n2,game\n3,mic\n")
        Path(command[-1]).write_bytes(b"reel")
        return SimpleNamespace(returncode=0)

    assert not _MIX_TRACKS(tmp_path / "01-x.mkv", runner=runner)
    assert _MAKE_REEL(tmp_path / "01-x.mkv", tmp_path / "reels" / "01-x.mp4", runner=runner)

    reel = calls[-1]
    assert "amix" not in reel[reel.index("-filter_complex") + 1]
    assert reel[reel.index("[v]") + 1:reel.index("[v]") + 3] == ["-map", "0:a:0"]


def test_every_kept_highlight_gets_a_reel_after_the_folder_is_made(tmp_path, isolated_data):
    session = _session(tmp_path, ["elimination", "knock"])
    manifest = _run(session)
    made = []

    def reel(source, destination):
        made.append(source.name)
        return source.name.startswith("01-")  # the second one fails

    assert highlights.reels(session, reel_fn=reel) == 1

    stored = json.loads((session / "highlights" / "highlights.json").read_text(encoding="utf-8"))
    assert sorted(made) == sorted(h["file"] for h in manifest["highlights"])
    assert [bool(h["reel"]) for h in stored["highlights"]] == [True, False]
    assert stored["highlights"][0]["reel"].endswith(".mp4")


def test_kill_traits_score_what_the_player_keeps_above_sprays_and_bots():
    from types import SimpleNamespace as Clip

    sniper = Clip(summary="The recording player no-scopes X with a hunting rifle headshot.",
                  payload={"mechanics": ["headshot"], "opponent_kind": "player"})
    spray = Clip(summary="The recording player lands a stream of SMG and assault rifle hits.",
                 payload={"mechanics": [], "opponent_kind": "player"})
    pistol = Clip(summary="The recording player eliminates X with a pistol.", payload={})
    bot = Clip(summary="The recording player knocks the NPC Chica the Chicken with a shotgun.",
               payload={"opponent_kind": "npc"})
    scores = [highlights._trait_words(c) for c in (sniper, spray, pistol, bot)]
    assert scores == [2.0, -1.0, -1.0, -2.0]


def test_kill_fields_count_beside_the_words_so_one_phrase_cannot_swing_a_keep():
    """A kept headshot kill re-described as "finishes a downed opponent"."""
    from types import SimpleNamespace as Clip

    fields = {"kill_weapon_class": "shotgun", "kill_headshot": "yes", "kill_hits": 1,
              "kill_damage": 180, "opponent_kind": "player"}
    first = Clip(summary="The recording player shotguns X through a build, landing a headshot.",
                 payload=dict(fields))
    again = Clip(summary="The recording player finishes a downed opponent with a shotgun.",
                 payload=dict(fields))

    assert highlights._trait_fields(first) == highlights._trait_fields(again) == 2.0
    assert highlights.kill_traits(first) - highlights.kill_traits(again) == 1.0  # not 2

    pistol = Clip(summary="eliminates X", payload={"kill_weapon_class": "pistol", "kill_hits": 1,
                                                    "kill_damage": 104})
    assert highlights._trait_fields(pistol) == -1.0  # its 104 is stacked hits, not a one-pump


def test_a_window_kept_on_the_rating_page_becomes_a_highlight(tmp_path, isolated_data):
    """The moment the ranking missed is one key away, numbered after the rest."""
    session = _session(tmp_path, ["elimination", "knock", "no_event"])
    _run(session, top=1)

    name = highlights.add_kept(f"sha{session.name}2")

    out = session / "highlights"
    assert name == "02-002-input_burst.mkv" and (out / name).is_file()
    manifest = json.loads((out / "highlights.json").read_text(encoding="utf-8"))
    assert [(h["file"], h["trigger"]) for h in manifest["highlights"]] == [
        ("01-000-input_burst.mkv", "input_burst"), (name, highlights.RATED_TRIGGER)]
    assert highlights.add_kept(f"sha{session.name}2") is None  # once
    assert highlights.add_kept(f"sha{session.name}0") is None  # already ranked in


def test_re_rated_a_skip_only_the_rating_pages_highlight_goes(tmp_path, isolated_data):
    session = _session(tmp_path, ["elimination", "knock"])
    _run(session, top=1)
    name = highlights.add_kept(f"sha{session.name}1")

    assert highlights.drop_kept(f"sha{session.name}1")
    assert not highlights.drop_kept(f"sha{session.name}0")  # the ranking's: the player's call

    out = session / "highlights"
    assert not (out / name).exists() and (out / "01-000-input_burst.mkv").is_file()
    manifest = json.loads((out / "highlights.json").read_text(encoding="utf-8"))
    assert [h["file"] for h in manifest["highlights"]] == ["01-000-input_burst.mkv"]


def test_a_released_window_can_still_be_a_highlight(tmp_path, isolated_data, monkeypatch):
    """Exported from the raw footage; never skipped because its copy is gone."""
    session = _session(tmp_path, ["elimination"])
    clip = session / "clips" / "000-input_burst.mkv"
    record = json.loads(clip.with_suffix(".capture.json").read_text(encoding="utf-8"))
    clip.with_suffix(".capture.json").write_text(json.dumps(
        {**record, "segment_start_sec": 16.0, "segment_end_sec": 24.0,
         clips.RELEASED_KEY: f"sha{session.name}0"}), encoding="utf-8")
    clip.unlink()

    def recut(source, destination, runner=None):
        destination.write_bytes(b"from raw")
        return destination

    monkeypatch.setattr(clips, "recut_window", recut)
    manifest = highlights.run(
        session, coverage=False, describe_fn=lambda *a, **k: None,
        harvest_fn=lambda s, d: [{}])

    assert [h["clip"] for h in manifest["highlights"]] == ["000-input_burst.mkv"]
    assert (session / "highlights" / "01-000-input_burst.mkv").read_bytes() == b"from raw"


def test_a_session_spent_in_the_lobby_has_no_highlights(tmp_path, isolated_data):
    """Not four idle lobby screens (2026-10-24): no kill, knock or fight is no highlight."""
    manifest = _run(_session(tmp_path, ["no_event", "no_event", "no_event"]))

    assert manifest["highlights"] == []
    assert "No kills, knocks or fights" in manifest["why"]


def test_quiet_moments_become_highlights_once_the_player_keeps_them(tmp_path, isolated_data, monkeypatch):
    """Funny moments have no kill to find them; the player's keeps teach it."""
    session = _session(tmp_path, ["no_event", "no_event"])
    assert _run(session)["highlights"] == []

    monkeypatch.setattr(highlights, "quiet_keeps", lambda conn: highlights.QUIET_KEEPS_TO_LEARN)
    assert len(_run(session)["highlights"]) == 2


def test_laughter_makes_a_quiet_moment_a_highlight(tmp_path, isolated_data, monkeypatch):
    """A funny moment has no kill; friends laughing is what marks it."""
    from pipeline_v2.capture import reactions

    session = _session(tmp_path, ["no_event", "no_event"])
    for i, clip in enumerate(sorted((session / "clips").glob("*.mkv"))):
        side = clip.with_suffix(".capture.json")
        record = json.loads(side.read_text(encoding="utf-8"))
        side.write_text(json.dumps({**record, "segment_start_sec": 8.0 * i, "segment_end_sec": 8.0 * i + 8.0}),
                        encoding="utf-8")
    monkeypatch.setattr(highlights, "hear_laughs",
                        lambda session: [reactions.Laugh(13.0, 14.5, 0.99, "game")])  # past the first window's 4s

    manifest = _run(session)

    assert [(h["clip"], h["laughed"]) for h in manifest["highlights"]] == [("001-input_burst.mkv", True)]
