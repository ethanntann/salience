"""The sample must be frozen and random, and scoring must separate the buckets."""

from __future__ import annotations

import json

import pytest

from pipeline_v2 import groundtruth, store
from pipeline_v2.jsonio import read_json


def _seed_store(count: int) -> None:
    conn = store.connect()
    for index in range(count):
        store.upsert_clip(
            conn,
            clip_sha=f"sha{index:03d}",
            path=store.Path(f"/clips/clip-{index:03d}.mp4"),
            engagement=None,
            duration_sec=20.0,
            event_sec=15.0,
            event_source="fraction",
            facts={},
            summary=f"summary {index}",
            payload={"weapon_name": "Striker Pump Shotgun", "outcome": "elimination"},
            prompt_version="v3.3",
        )
    conn.commit()
    conn.close()


def test_normalisation_ignores_rarity_and_punctuation():
    assert groundtruth.normalise_weapon("Epic Hunting Rifle") == "hunting rifle"
    assert groundtruth.normalise_weapon("  MYTHIC  Drum-Gun ") == "drum gun"
    assert groundtruth.normalise_weapon(None) == ""


@pytest.mark.parametrize(
    "claimed, actual, bucket",
    [
        ("Drum Gun", "drum gun", "exact"),
        ("Epic Hunting Rifle", "Hunting Rifle", "exact"),
        ("Striker Pump Shotgun", "shotgun", "family"),
        ("Tactical Pistol", "Drum Gun", "mismatch"),
        ("unknown", "Drum Gun", "model_declined"),
        ("", "Drum Gun", "model_declined"),
        ("Drum Gun", "unknown", "no_ground_truth"),
    ],
)
def test_weapon_buckets(claimed, actual, bucket):
    assert groundtruth.compare_weapon(claimed, actual) == bucket


def test_declining_is_not_counted_as_a_correct_answer():
    """The v3.5 arm is meant to move clips into `model_declined`, not hide there."""
    assert groundtruth.compare_weapon("unknown", "Drum Gun") != "exact"


def test_prepare_freezes_a_reproducible_sample(tmp_path, isolated_data):
    _seed_store(60)
    first = groundtruth.prepare(tmp_path / "a", size=10, seed=7)
    second = groundtruth.prepare(tmp_path / "b", size=10, seed=7)
    different = groundtruth.prepare(tmp_path / "c", size=10, seed=8)

    shas = [c["clip_sha"] for c in first["clips"]]
    assert len(shas) == len(set(shas)) == 10
    assert shas == [c["clip_sha"] for c in second["clips"]]
    assert shas != [c["clip_sha"] for c in different["clips"]]


def test_prepare_is_idempotent_and_never_reshuffles(tmp_path, isolated_data):
    _seed_store(60)
    out = tmp_path / "gt"
    original = groundtruth.prepare(out, size=10, seed=0)

    # A second call with different arguments must not move the frozen sample.
    again = groundtruth.prepare(out, size=25, seed=99)

    assert again["clips"] == original["clips"]
    assert again["seed"] == 0


def test_prepare_refuses_a_sample_larger_than_the_corpus(tmp_path, isolated_data):
    _seed_store(5)
    with pytest.raises(ValueError, match="need 10"):
        groundtruth.prepare(tmp_path / "gt", size=10, seed=0)


def test_labels_are_validated(tmp_path):
    known = {"sha001"}
    sha, record = groundtruth.validate_label(
        {"clip_sha": "sha001", "weapon": " Drum Gun ", "actor": "teammate"}, known
    )
    assert sha == "sha001"
    assert record["weapon"] == "Drum Gun"
    assert record["actor"] == "teammate"

    for bad in (
        {"clip_sha": "nope", "weapon": "x", "actor": "teammate"},
        {"clip_sha": "sha001", "weapon": "x", "actor": "invented"},
        {"clip_sha": "sha001", "weapon": "x" * 200, "actor": "teammate"},
    ):
        with pytest.raises(ValueError):
            groundtruth.validate_label(bad, known)


def test_report_scores_only_labelled_clips_and_keeps_buckets_separate(tmp_path, isolated_data):
    _seed_store(20)
    out = tmp_path / "gt"
    plan = groundtruth.prepare(out, size=4, seed=0)
    shas = [c["clip_sha"] for c in plan["clips"]]

    # Model claimed "Striker Pump Shotgun" for every seeded clip.
    (out / "labels.json").write_text(
        json.dumps(
            {
                "annotator": "user",
                "labels": {
                    shas[0]: {"weapon": "Striker Pump Shotgun", "actor": "recording_player", "notes": ""},
                    shas[1]: {"weapon": "shotgun", "actor": "recording_player", "notes": ""},
                    shas[2]: {"weapon": "Drum Gun", "actor": "teammate", "notes": ""},
                },
            }
        ),
        encoding="utf-8",
    )

    result = groundtruth.report(out)

    assert result["labelled"] == 3 and result["scored"] == 3
    assert result["weapon_buckets"] == {"exact": 1, "family": 1, "mismatch": 1}
    assert result["exact_rate"] == pytest.approx(1 / 3)
    assert result["wrong_rate"] == pytest.approx(1 / 3)
    assert result["actor_ground_truth"] == {"recording_player": 2, "teammate": 1}
    assert read_json(out / "report-v3.3.json")["scored"] == 3


def test_report_flags_clips_absent_from_the_scored_variant(tmp_path, isolated_data):
    _seed_store(20)
    out = tmp_path / "gt"
    plan = groundtruth.prepare(out, size=3, seed=0)
    (out / "labels.json").write_text(
        json.dumps(
            {
                "annotator": "user",
                "labels": {
                    c["clip_sha"]: {"weapon": "Drum Gun", "actor": "recording_player", "notes": ""}
                    for c in plan["clips"]
                },
            }
        ),
        encoding="utf-8",
    )

    # Nothing has been described at this arm yet.
    result = groundtruth.report(out, variant="v3.5-killweapon")

    assert result["scored"] == 0
    assert len(result["missing_from_variant"]) == 3


def test_sample_survives_a_relink(tmp_path, isolated_data):
    """Freezing the sample must not freeze where the files happen to live."""
    _seed_store(20)
    out = tmp_path / "gt"
    plan = groundtruth.prepare(out, size=3, seed=0)
    assert all("clips" in c["path"] for c in plan["clips"])

    moved = tmp_path / "moved"
    moved.mkdir()
    conn = store.connect()
    with conn:
        for clip in plan["clips"]:
            conn.execute(
                "update clips set path = ? where clip_sha = ?",
                (str(moved / clip["name"]), clip["clip_sha"]),
            )
    conn.close()

    resolved = groundtruth.current_paths(plan)

    assert {p.parent for p in resolved.values()} == {moved}
    assert set(resolved) == {c["clip_sha"] for c in plan["clips"]}


def test_a_hunting_rifle_is_a_sniper_not_an_assault_rifle():
    assert groundtruth.weapon_class("Hunting Rifle") == "sniper"
    assert groundtruth.weapon_class("Mythic Hunting Rifle") == "sniper"
    assert groundtruth.weapon_class("Warforged Assault Rifle") == "assault_rifle"

    # A generic "sniper" label matches the specific weapon.
    assert groundtruth.compare_weapon("Hunting Rifle", "sniper") == "family"
    # Sharing the word "rifle" is not agreement.
    assert groundtruth.compare_weapon("Warforged Assault Rifle", "hunting rifle") == "mismatch"
    assert groundtruth.compare_weapon("Hunting Rifle", "assault rifle") == "mismatch"


def test_a_bare_rifle_label_stays_ambiguous_rather_than_guessing():
    assert groundtruth.weapon_class("rifle") is None
    # Falls back to substring, which is what the annotator meant by typing it.
    assert groundtruth.compare_weapon("Chaos Exploder Rifle", "rifle") == "family"


def test_class_labels_match_their_specific_weapons():
    assert groundtruth.compare_weapon("Striker Pump Shotgun", "shotgun") == "family"
    assert groundtruth.compare_weapon("Lancehead Pistol", "pistol") == "family"
    assert groundtruth.compare_weapon("Wrecker Revolver", "pistol") == "family"
    assert groundtruth.compare_weapon("Striker Pump Shotgun", "pistol") == "mismatch"


def test_baba_yaga_is_a_pistol():
    assert groundtruth.weapon_class("9mm Baba Yaga") == "pistol"
    assert groundtruth.compare_weapon("9mm Baba Yaga", "pistol") == "family"


def test_annotator_answers_split_into_separate_weapons():
    assert groundtruth.parse_weapons("pistol, shotgun") == ["pistol", "shotgun"]
    assert groundtruth.parse_weapons("pistol then shotgun") == ["pistol", "shotgun"]
    assert groundtruth.parse_weapons("Hunting Rifle") == ["Hunting Rifle"]
    assert groundtruth.parse_weapons("") == []


def test_claims_are_read_according_to_the_variant_not_the_payload_shape():
    """`kill_weapons` defaults to [] in EVERY payload, v3.3 ones included."""
    v33 = {"weapon_name": "Drum Gun", "kill_weapons": []}
    v35 = {"weapon_name": "", "kill_weapons": ["Drum Gun", "Pump Shotgun"]}

    # v3.3 never asked for the list, so its empty default must be ignored.
    assert groundtruth.claimed_weapons(v33, "v3.3") == ["Drum Gun"]
    assert groundtruth.claimed_weapons(v33, "v3.3@some-model") == ["Drum Gun"]

    assert groundtruth.claimed_weapons(v35, "v3.5-killweapon") == [
        "Drum Gun",
        "Pump Shotgun",
    ]
    assert groundtruth.claimed_weapons(v35, "v3.5-killweapon@some-model") == [
        "Drum Gun",
        "Pump Shotgun",
    ]

    # Only where the prompt asked is an empty list a real abstention.
    assert groundtruth.claimed_weapons(
        {"weapon_name": "Drum Gun", "kill_weapons": []}, "v3.5-killweapon"
    ) == []


def test_a_multi_kill_window_agrees_when_any_weapon_matches():
    """The annotator watched the clip; the model saw a window inside it."""
    assert groundtruth.compare_claims(["Striker Pump Shotgun"], ["pistol", "shotgun"]) == "family"
    assert groundtruth.compare_claims(["Lancehead Pistol"], ["pistol", "shotgun"]) == "family"
    # Neither of the annotator's weapons -> still a real miss.
    assert groundtruth.compare_claims(["Hunting Rifle"], ["pistol", "shotgun"]) == "mismatch"


def test_an_empty_list_against_a_real_kill_is_declining_not_correct():
    assert groundtruth.compare_claims([], ["pistol"]) == "model_declined"


def test_an_empty_list_where_nothing_died_is_not_scored():
    assert groundtruth.compare_claims([], ["unknown"]) == "no_ground_truth"
    assert groundtruth.compare_claims(["Chug Jug"], ["unknown"]) == "no_ground_truth"


def test_model_window_matches_what_the_frames_module_extracts():
    from pipeline_v2 import frames

    start, end = groundtruth.model_window(15.3, 20.4)

    assert start == 15.3 - frames.WINDOW_PRE_SEC
    assert end == 15.3 + frames.WINDOW_POST_SEC


def test_model_window_is_clamped_inside_the_file():
    start, end = groundtruth.model_window(19.8, 20.4)
    assert end <= 20.4
    assert start < end

    start, end = groundtruth.model_window(1.0, 20.4)
    assert start >= 0.1


def test_model_window_survives_missing_metadata():
    start, end = groundtruth.model_window(None, None)
    assert start >= 0.0 and end >= start


def test_compare_scores_every_arm_that_covers_the_sample(tmp_path, isolated_data):
    import json as _json

    _seed_store(20)
    out = tmp_path / "gt"
    plan = groundtruth.prepare(out, size=3, seed=0)
    shas = [c["clip_sha"] for c in plan["clips"]]

    # A second arm describing only the first two clips.
    conn = store.connect()
    for sha in shas[:2]:
        conn.execute(
            "insert or replace into descriptions (clip_sha, summary, payload, prompt_ver)"
            " values (?, ?, ?, ?)",
            (sha, "s", _json.dumps({"kill_weapons": ["Drum Gun"]}), "v3.5-killweapon"),
        )
    conn.commit()
    conn.close()

    (out / "labels.json").write_text(
        _json.dumps({"annotator": "user", "labels": {
            s: {"weapon": "Drum Gun", "actor": "recording_player", "notes": ""}
            for s in shas}}),
        encoding="utf-8")

    results = {r["variant"]: r for r in groundtruth.compare(out)}

    assert set(results) == {"v3.3", "v3.5-killweapon"}
    # v3.3 claimed a shotgun for all three; the new arm named the right gun twice.
    assert results["v3.3"]["scored"] == 3
    assert results["v3.3"]["wrong_rate"] == 1.0
    assert results["v3.5-killweapon"]["scored"] == 2
    assert results["v3.5-killweapon"]["weapon_buckets"] == {"exact": 2}
