from salience_api.features.hud_ocr import best_weapon_ocr, weapon_category_from_text
from salience_api.clips.keyframes import Keyframe
from salience_api.features.hud_ocr import RapidHudOcr


def test_weapon_category_survives_ocr_dropping_the_space():
    """Stylised HUD fonts often read as one word; those are not 'unknown'."""
    assert weapon_category_from_text("HUNTINGRIFLE") == "sniper_or_hunting"
    assert weapon_category_from_text("MYTHIC HUNTINGRIFLE") == "sniper_or_hunting"
    assert weapon_category_from_text("ASSAULTRIFLE") == "automatic"
    assert weapon_category_from_text("HANDCANNON") == "pistol"
    assert weapon_category_from_text("MARKSMANRIFLE") == "sniper_or_hunting"
    # Still nothing to match on.
    assert weapon_category_from_text("143") == "unknown"
    assert weapon_category_from_text("") == "unknown"


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
    # Panels are read in batches, so record what each batch was asked to read.
    monkeypatch.setattr(
        engine,
        "_recognize_panels",
        lambda _engine, batch: [
            (event_index, seen.append(frame.timestamp_sec) or [])
            for event_index, frame in batch
        ],
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


def test_unresolved_events_fall_through_to_their_second_choice(monkeypatch, tmp_path):
    """An event that reads a weapon on its first frame must not read a second."""
    batches: list[list[float]] = []
    engine = RapidHudOcr()
    monkeypatch.setattr(engine, "_get_engine", lambda: object())

    def fake_panels(_engine, batch):
        batches.append([frame.timestamp_sec for _, frame in batch])
        results = []
        for event_index, frame in batch:
            # Event 0 resolves immediately; event 1 never does.
            found = event_index == 0
            results.append((
                event_index,
                [{"weapon_category": "shotgun" if found else "unknown",
                  "confidence": 0.9, "text": "pump shotgun"}],
            ))
        return results

    monkeypatch.setattr(engine, "_recognize_panels", fake_panels)
    frames = []
    for event_index, centre in ((0, 10.0), (1, 20.0)):
        for offset in (-0.75, -0.375, 0.0):
            frames.append(
                Keyframe(
                    path=tmp_path / f"{event_index}-{offset}.jpg",
                    timestamp_sec=centre + offset,
                    event_index=event_index,
                    event_center_sec=centre,
                )
            )

    engine.recognize_event_frames(frames)

    assert len(batches) == 2, "two phases: first choices, then unresolved events"
    assert batches[0] == [9.25, 19.25], "phase one reads both events"
    assert batches[1] == [19.625], "only the unresolved event pays for a second read"


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
