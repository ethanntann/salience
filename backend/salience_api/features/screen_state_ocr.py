"""Rule-based UI text matching for labels that are literal on-screen text.

Kept separate from ``hud_ocr.py`` (which crops and reads the weapon HUD
panel specifically) because these matchers read arbitrary full-frame OCR
output and only ever corroborate a "yes" - they never override the model.

OCR returns each detected text box separately, and the stylized victory
banner usually splits into separate boxes ("VICTORY", "ROYALE") or garbles
characters, so matching joins all boxes from a frame and falls back to a
fuzzy word-window comparison.
"""

from __future__ import annotations

from difflib import SequenceMatcher

_VICTORY_PHRASES = ("victory royale", "#1 victory")
_VICTORY_FUZZY_PHRASE = "victory royale"
_VICTORY_FUZZY_RATIO = 0.8
_MENU_PHRASES = (
    "battle pass",
    "locker",
    "quit match",
    "leave match",
    "settings",
    "career",
    "challenges",
)


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _joined(texts: list[str]) -> str:
    return _normalize(" ".join(str(text) for text in texts))


def _fuzzy_contains(haystack: str, phrase: str, *, min_ratio: float) -> bool:
    words = haystack.split()
    span = len(phrase.split())
    for start in range(max(0, len(words) - span + 1)):
        window = " ".join(words[start : start + span])
        if SequenceMatcher(None, window, phrase).ratio() >= min_ratio:
            return True
    return False


def matches_victory_banner(texts: list[str]) -> bool:
    joined = _joined(texts)
    if any(phrase in joined for phrase in _VICTORY_PHRASES):
        return True
    return _fuzzy_contains(
        joined, _VICTORY_FUZZY_PHRASE, min_ratio=_VICTORY_FUZZY_RATIO
    )


def matches_menu_screen(texts: list[str]) -> bool:
    return any(phrase in _joined(texts) for phrase in _MENU_PHRASES)
