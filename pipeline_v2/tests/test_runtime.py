import argparse
import json
from dataclasses import replace

import pytest

from pipeline_v2 import cli, config, embed, feedback, ranking, store, vlm
from pipeline_v2.facts import ClipFacts
from conftest import add_clip


def test_ranking_uses_v2_labels_without_legacy_database(conn, tmp_path):
    for i in range(12):
        sha = str(i)
        add_clip(conn, tmp_path / f"clip{i}.mp4", sha=sha)
        conn.execute("insert into preferences values (?, ?, 0)", (sha, i % 2))
        store.put_embedding(conn, sha, "test", [float(i % 2), 1.0])
    fights, count = ranking.ranked_fights(conn)
    assert count == 12
    assert len(fights) == 12
    assert fights[0].score > fights[-1].score


def test_rank_cli_no_longer_accesses_nonexistent_embeddings_argument(isolated_data):
    assert (
        cli.cmd_rank(
            argparse.Namespace(db=None, variant="v3.3", mode="embedding", top=12)
        )
        == 1
    )


def test_taste_clean_database_does_not_need_research_tables(isolated_data):
    args = argparse.Namespace(
        db=None,
        variant="v3.3",
        embeddings=None,
        same_clips_as=None,
        voiced_only=False,
        organic_only=True,
    )
    assert cli.cmd_taste(args) == 1


def test_legacy_import_is_opt_in_and_read_only(tmp_path):
    assert feedback.load(None) == []
    assert feedback.v1_final_scores(None) == {}
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(Exception):
        feedback.load(missing)
    assert not missing.exists()


def _pref(path, label=1):
    return feedback.Preference(
        path=path, label=label, weight=1.0, organic=True, action="keep"
    )


def test_an_exact_path_always_wins_over_a_same_named_file(tmp_path):
    here, elsewhere = tmp_path / "new" / "clip.mp4", tmp_path / "old" / "clip.mp4"
    index = feedback.PreferenceIndex([_pref(elsewhere, 0), _pref(here, 1)])

    assert index.get(here).label == 1
    assert index.matched_by_name == 0


def test_a_relocated_library_still_finds_its_labels(tmp_path):
    """`relink` moves every path; a path-only join matched 0 of 348 clips."""
    index = feedback.PreferenceIndex([_pref(tmp_path / "old" / "clip.mp4")])

    found = index.get(tmp_path / "restored" / "clip.mp4")

    assert found is not None
    assert index.matched_by_name == 1


def test_an_ambiguous_filename_is_refused_rather_than_guessed(tmp_path):
    index = feedback.PreferenceIndex(
        [_pref(tmp_path / "a" / "clip.mp4", 1), _pref(tmp_path / "b" / "clip.mp4", 0)]
    )

    assert index.get(tmp_path / "elsewhere" / "clip.mp4") is None


def test_an_unknown_clip_matches_nothing(tmp_path):
    index = feedback.PreferenceIndex([_pref(tmp_path / "a" / "clip.mp4")])

    assert index.get(tmp_path / "a" / "other.mp4") is None


@pytest.mark.parametrize("texts", [["rifle headshot"], ["rifle", "pistol"], ["the"]])
def test_first_clip_can_be_embedded_without_neural_download(texts):
    vectors = embed.TfidfEncoder().fit_transform(texts)
    assert len(vectors) == len(texts)
    assert len({len(v) for v in vectors}) == 1


def test_cache_identity_covers_model_and_frames_and_needs_no_key(
    isolated_data, monkeypatch
):
    clip = isolated_data / "clip.mp4"
    frame = isolated_data / "frame.jpg"
    clip.write_bytes(b"clip")
    frame.write_bytes(b"frame")
    cfg = replace(config.load_config(), api_key="test")
    calls = []

    def post(*args):
        calls.append(1)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"summary": "Player lands a headshot."})
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }

    monkeypatch.setattr(vlm, "_post", post)
    assert not vlm.describe(cfg, clip, ClipFacts(), [frame]).cached
    assert vlm.describe(
        replace(cfg, api_key=""), clip, ClipFacts(), [frame], allow_call=False
    ).cached
    with pytest.raises(vlm.BudgetExceeded):
        vlm.describe(
            replace(cfg, model="different"),
            clip,
            ClipFacts(),
            [frame],
            allow_call=False,
        )
    frame.write_bytes(b"different pixels")
    with pytest.raises(vlm.BudgetExceeded):
        vlm.describe(cfg, clip, ClipFacts(), [frame], allow_call=False)
    assert len(calls) == 1


def test_invalid_paid_response_is_retained(isolated_data, monkeypatch):
    clip = isolated_data / "clip.mp4"
    frame = isolated_data / "frame.jpg"
    clip.write_bytes(b"clip")
    frame.write_bytes(b"frame")
    cfg = replace(config.load_config(), api_key="test")
    calls = []

    def post(*args):
        calls.append(1)
        return {"choices": [], "usage": {"prompt_tokens": 20}}

    monkeypatch.setattr(vlm, "_post", post)
    for _ in range(2):
        with pytest.raises(ValueError):
            vlm.describe(cfg, clip, ClipFacts(), [frame])
    assert len(calls) == 1


def test_extract_json_repairs_an_unquoted_array_word():
    """qwen3p8-max returned one mechanics tag unquoted and killed a 50-clip run."""
    from pipeline_v2 import vlm

    raw = (
        '{"summary": "x", "mechanics": ["mid_range", close_range, "weapon_swap"],'
        ' "hud_conflict": null, "ok": true}'
    )

    parsed = vlm._extract_json(raw)

    assert parsed["mechanics"] == ["mid_range", "close_range", "weapon_swap"]
    assert parsed["hud_conflict"] is None and parsed["ok"] is True


def test_extract_json_leaves_valid_payloads_alone():
    from pipeline_v2 import vlm

    raw = '{"mechanics": ["a"], "flag": false, "none": null, "nums": [1, 2]}'

    assert vlm._extract_json(raw) == {
        "mechanics": ["a"],
        "flag": False,
        "none": None,
        "nums": [1, 2],
    }


def test_extract_json_still_rejects_genuine_garbage():
    import pytest

    from pipeline_v2 import vlm

    with pytest.raises(ValueError):
        vlm._extract_json("no object here at all")
    with pytest.raises(Exception):
        vlm._extract_json('{"a": [1, 2,,]}')
