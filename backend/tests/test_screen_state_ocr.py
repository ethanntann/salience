from salience_api.features.screen_state_ocr import matches_menu_screen, matches_victory_banner


def test_matches_victory_banner_detects_known_phrase():
    assert matches_victory_banner(["#1 VICTORY ROYALE", "16 ELIMS"]) is True


def test_matches_victory_banner_ignores_unrelated_text():
    assert matches_victory_banner(["ELIMINATED", "2 REMAINING"]) is False


def test_matches_victory_banner_joins_split_ocr_boxes():
    # The stylized banner usually splits into separate OCR detections.
    assert matches_victory_banner(["VICTORY", "ROYALE", "16 ELIMS"]) is True


def test_matches_victory_banner_fuzzy_matches_garbled_ocr():
    assert matches_victory_banner(["V1CTORY ROYAL"]) is True


def test_matches_victory_banner_rejects_victory_crown_text():
    assert matches_victory_banner(["VICTORY CROWN"]) is False


def test_matches_menu_screen_detects_known_phrase():
    assert matches_menu_screen(["BATTLE PASS", "SEASON 4"]) is True


def test_matches_menu_screen_ignores_gameplay_hud():
    assert matches_menu_screen(["SHOTGUN", "24 HP"]) is False


class _StubResult:
    def __init__(self, texts: list[str]) -> None:
        self.txts = texts
        self.scores = [0.9] * len(texts)


class _StubEngine:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def __call__(self, image):
        return _StubResult(self._texts)


def test_recognize_full_frame_text_returns_detected_strings(tmp_path, monkeypatch):
    from PIL import Image

    from salience_api.features.hud_ocr import RapidHudOcr

    path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(path)

    ocr = RapidHudOcr(enabled=True)
    monkeypatch.setattr(ocr, "_get_engine", lambda: _StubEngine(["#1 VICTORY ROYALE"]))

    assert ocr.recognize_full_frame_text(path) == ["#1 VICTORY ROYALE"]


def test_recognize_full_frame_text_returns_empty_when_engine_unavailable(tmp_path, monkeypatch):
    from PIL import Image

    from salience_api.features.hud_ocr import RapidHudOcr

    path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(path)

    ocr = RapidHudOcr(enabled=True)
    monkeypatch.setattr(ocr, "_get_engine", lambda: None)

    assert ocr.recognize_full_frame_text(path) == []
