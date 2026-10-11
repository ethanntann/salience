"""Region-scoped OCR that recognizes only the text the pipeline actually uses.

RapidOCR's convenience call (``engine(image)``) runs detection and then
recognition on *every* text box it finds. A Fortnite frame carries ~30 of them --
ammo, materials, shields, player count, timers, minimap labels -- and the
pipeline reads two or three. Measured on a 960x540 timeline frame, detection is
~20% of that call and recognition scales with the number of boxes, so the bulk of
the cost is transcribing HUD clutter nobody reads.

This module splits the two halves:

1. detect per frame (unavoidable -- text moves between frames),
2. keep only boxes whose centre lands in a named region of interest,
3. recognize every surviving crop from every frame in **one** batched call.

That turns ``N x (detect + recognize)`` into ``N x detect + 1 x recognize``.

Regions are normalized fractions of frame size, so they survive resolution
changes, and they are per-game data rather than code -- porting to another title
is a new ``REGIONS`` table, not a new pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class TextRegion:
    """A named region of interest as fractions of frame width/height."""

    name: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, cx: float, cy: float, frame_w: int, frame_h: int) -> bool:
        left, top = self.x * frame_w, self.y * frame_h
        return (
            left <= cx <= left + self.width * frame_w
            and top <= cy <= top + self.height * frame_h
        )


@dataclass(frozen=True)
class RegionRead:
    """One recognized string, tagged with where and when it was found."""

    frame_index: int
    region: str
    text: str
    confidence: float


# Fortnite HUD geometry. The kill feed rides the upper-left/centre, the selected
# weapon and ammo sit bottom-right, and the victory banner lands centre-screen.
# Deliberately generous -- a region that is slightly too big costs a few extra
# recognitions, one that is too small silently drops the read.
FORTNITE_REGIONS: tuple[TextRegion, ...] = (
    TextRegion("kill_feed", 0.0, 0.40, 0.55, 0.28),
    TextRegion("weapon_hud", 0.58, 0.74, 0.42, 0.26),
    TextRegion("banner", 0.10, 0.25, 0.80, 0.35),
)

# Everything on screen, for callers that genuinely need arbitrary text.
WHOLE_FRAME: tuple[TextRegion, ...] = (TextRegion("frame", 0.0, 0.0, 1.0, 1.0),)


def _box_centre(box) -> tuple[float, float]:
    import numpy as np

    points = np.asarray(box, dtype=float).reshape(-1, 2)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


class RegionalOcr:
    """Detect per frame, recognize once, for a fixed set of regions."""

    def __init__(self, *, enabled: bool = True, engine=None) -> None:
        self.enabled = enabled
        # Callers that already hold a warm RapidOCR should pass it in; loading a
        # second copy doubles model memory and first-call latency for nothing.
        self._engine = engine

    @property
    def available(self) -> bool:
        return self.enabled and find_spec("rapidocr") is not None

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        if not self.available:
            return None
        from rapidocr import RapidOCR

        self._engine = RapidOCR()
        return self._engine

    @staticmethod
    def _supports_split(engine) -> bool:
        """Whether this engine exposes detection and recognition separately."""
        return all(
            callable(getattr(engine, name, None))
            for name in ("text_det", "crop_text_regions", "recognize_txt")
        )

    @staticmethod
    def _read_unsplit(
        engine, images: Sequence["np.ndarray"], *, min_confidence: float
    ) -> list[RegionRead]:
        """Per-image fallback for engines without the split API.

        Region filtering needs detection boxes, so this path cannot narrow by
        region and returns every read tagged ``frame``. It exists so the module
        keeps working against a stubbed or future RapidOCR whose internals moved.
        """
        reads: list[RegionRead] = []
        for frame_index, image in enumerate(images):
            if image is None:
                continue
            result = engine(image)
            texts = tuple(getattr(result, "txts", ()) or ())
            scores = tuple(getattr(result, "scores", ()) or ())
            for text, score in zip(texts, scores, strict=False):
                normalized = " ".join(str(text).split())
                confidence = max(0.0, min(1.0, float(score)))
                if not normalized or confidence < min_confidence:
                    continue
                reads.append(
                    RegionRead(
                        frame_index=frame_index,
                        region="frame",
                        text=normalized,
                        confidence=confidence,
                    )
                )
        return reads

    def read(
        self,
        images: Sequence["np.ndarray"],
        regions: Sequence[TextRegion] = FORTNITE_REGIONS,
        *,
        min_confidence: float = 0.0,
    ) -> list[RegionRead]:
        """Return every in-region text read across ``images``.

        Detection runs per image; recognition runs once for all surviving crops.
        """
        engine = self._get_engine()
        if engine is None or not len(images):
            return []
        if not self._supports_split(engine):
            return self._read_unsplit(engine, images, min_confidence=min_confidence)

        import numpy as np

        crops: list[np.ndarray] = []
        origins: list[tuple[int, str]] = []

        for frame_index, image in enumerate(images):
            if image is None:
                continue
            height, width = image.shape[:2]
            detection = engine.text_det(image)
            boxes = getattr(detection, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            kept_boxes = []
            kept_regions: list[str] = []
            for box in boxes:
                centre_x, centre_y = _box_centre(box)
                for region in regions:
                    if region.contains(centre_x, centre_y, width, height):
                        kept_boxes.append(box)
                        kept_regions.append(region.name)
                        break
            if not kept_boxes:
                continue

            frame_crops = engine.crop_text_regions(image, np.asarray(kept_boxes))
            for crop, region_name in zip(frame_crops, kept_regions, strict=True):
                crops.append(crop)
                origins.append((frame_index, region_name))

        if not crops:
            return []

        recognized = engine.recognize_txt(crops)
        texts = tuple(getattr(recognized, "txts", ()) or ())
        scores = tuple(getattr(recognized, "scores", ()) or ())

        reads: list[RegionRead] = []
        for (frame_index, region_name), text, score in zip(
            origins, texts, scores, strict=False
        ):
            normalized = " ".join(str(text).split())
            confidence = max(0.0, min(1.0, float(score)))
            if not normalized or confidence < min_confidence:
                continue
            reads.append(
                RegionRead(
                    frame_index=frame_index,
                    region=region_name,
                    text=normalized,
                    confidence=confidence,
                )
            )
        return reads

    def texts_by_frame(
        self,
        images: Sequence["np.ndarray"],
        regions: Sequence[TextRegion] = FORTNITE_REGIONS,
    ) -> dict[int, list[str]]:
        """Group reads by frame, for matchers that consume whole-frame text."""
        grouped: dict[int, list[str]] = {}
        for read in self.read(images, regions):
            grouped.setdefault(read.frame_index, []).append(read.text)
        return grouped
