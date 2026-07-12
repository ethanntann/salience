from salience_api.features.hud_ocr import best_weapon_ocr, weapon_category_from_text
from salience_api.clips.keyframes import Keyframe
from salience_api.features.hud_ocr import RapidHudOcr


def test_weapon_category_from_fortnite_hud_text():
    assert weapon_category_from_text("LEGENDARY STRIKER PUMP SHOTGUN") == "shotgun"
    assert weapon_category_from_text("Hunting Rifle") == "sniper_or_hunting"
    assert weapon_category_from_text("Hop Rock Dualies") == "pistol"
    assert weapon_category_from_text("Striker Burst Rifle") == "automatic"
    assert weapon_category_from_text("Thermal DMR") == "sniper_or_hunting"
    assert weapon_category_from_text("Rocket Launcher") == "other"
    assert weapon_category_from_text("Drum Gun") == "automatic"


def test_ocr_frame_selection_prefers_shot_phase_over_finish(monkeypatch, tmp_path):
    seen: list[float] = []
    engine = RapidHudOcr()
    monkeypatch.setattr(engine, "_get_engine", lambda: object())
    monkeypatch.setattr(
        engine,
        "_recognize_weapon_panel",
        lambda _engine, _path, *, event_index, timestamp, dedicated_crop: (
            seen.append(timestamp) or []
        ),
    )
    frames = [
        Keyframe(
            path=tmp_path / f"{timestamp}.jpg",
            timestamp_sec=timestamp,
            event_index=0,
            event_center_sec=10.0,
        )
        for timestamp in (9.25, 9.625, 10.0, 10.375, 10.75)
    ]

    engine.recognize_event_frames(frames)

    assert seen == [9.25, 9.625]


def test_stable_weapon_ocr_wins_across_nearby_pre_event_frames():
    observations = [
        {
            "event_index": 2,
            "timestamp": 14.9,
            "text": "STRIKER PUMP SHOTGUN",
            "confidence": 0.72,
        },
        {
            "event_index": 2,
            "timestamp": 15.3,
            "text": "STRIKER PUMP SHOTGUN",
            "confidence": 0.79,
        },
        {
            "event_index": 2,
            "timestamp": 15.6,
            "text": "STRIKER BURST RIFLE",
            "confidence": 0.99,
        },
    ]

    result = best_weapon_ocr(observations, 2, event_timestamp=15.3)

    assert result is not None
    assert result["ambiguous"] is False
    assert result["category"] == "shotgun"
    assert result["supporting_frames"] == 2


def test_shot_phase_weapon_beats_finish_swap_hud():
    observations = [
        {
            "event_index": 0,
            "timestamp": 14.5,
            "text": "EPIC HUNTING RIFLE",
            "confidence": 0.95,
        },
        {
            "event_index": 0,
            "timestamp": 15.25,
            "text": "EXTENDING FOCUS SHOTGUN",
            "confidence": 0.99,
        },
    ]

    result = best_weapon_ocr(observations, 0, event_timestamp=15.3)

    assert result is not None
    assert result["ambiguous"] is False
    assert result["category"] == "sniper_or_hunting"
    assert "HUNTING" in result["text"].upper()


def test_ambiguous_weapon_ocr_marks_conflict_instead_of_guessing():
    observations = [
        {
            "event_index": 0,
            "timestamp": 5.0,
            "text": "PUMP SHOTGUN",
            "confidence": 0.9,
        },
        {
            "event_index": 0,
            "timestamp": 5.0,
            "text": "BURST RIFLE",
            "confidence": 0.88,
        },
    ]

    result = best_weapon_ocr(observations, 0, event_timestamp=5.0)

    assert result is not None
    assert result["ambiguous"] is True
    assert result["category"] == "unknown"
