import numpy as np

from salience_api.features.ocr_regions import (
    FORTNITE_REGIONS,
    WHOLE_FRAME,
    RegionalOcr,
    TextRegion,
)


def _frame(width=200, height=100):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _box(cx, cy, half=5):
    """An axis-aligned quad centred on (cx, cy)."""
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=float,
    )


class _DetOutput:
    def __init__(self, boxes):
        self.boxes = np.asarray(boxes) if len(boxes) else None
        self.scores = [0.9] * len(boxes)


class _RecOutput:
    def __init__(self, texts):
        self.txts = list(texts)
        self.scores = [0.9] * len(texts)


class _SplitEngine:
    """Fake engine exposing the detection/recognition split API."""

    def __init__(self, boxes_per_frame, labels=None):
        self.boxes_per_frame = boxes_per_frame
        self.labels = labels or {}
        self.det_calls = 0
        self.rec_calls = 0
        self.rec_batch_sizes = []

    def text_det(self, image):
        boxes = self.boxes_per_frame[self.det_calls]
        self.det_calls += 1
        return _DetOutput(boxes)

    def crop_text_regions(self, image, det_boxes):
        # One crop per box; content is irrelevant, identity is carried by order.
        return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in det_boxes]

    def recognize_txt(self, crops):
        self.rec_calls += 1
        self.rec_batch_sizes.append(len(crops))
        return _RecOutput([f"text{index}" for index in range(len(crops))])


class _UnsplitEngine:
    """Fake engine offering only the convenience call."""

    def __init__(self, texts_per_frame):
        self.texts_per_frame = texts_per_frame
        self.calls = 0

    def __call__(self, image):
        texts = self.texts_per_frame[self.calls]
        self.calls += 1
        return _RecOutput(texts)


def test_text_region_contains_uses_normalised_coordinates():
    region = TextRegion("right_half", 0.5, 0.0, 0.5, 1.0)

    assert region.contains(150, 50, 200, 100) is True
    assert region.contains(50, 50, 200, 100) is False
    # Boundaries are inclusive so a box sitting exactly on the edge is kept.
    assert region.contains(100, 50, 200, 100) is True


def test_read_issues_one_recognition_batch_for_all_frames():
    # Three frames, two boxes each, all inside the whole-frame region.
    boxes = [[_box(50, 50), _box(150, 50)] for _ in range(3)]
    engine = _SplitEngine(boxes)
    ocr = RegionalOcr(engine=engine)

    reads = ocr.read([_frame(), _frame(), _frame()], WHOLE_FRAME)

    assert engine.det_calls == 3, "detection must still run per frame"
    assert engine.rec_calls == 1, "recognition must be a single batched call"
    assert engine.rec_batch_sizes == [6]
    assert len(reads) == 6
    assert {read.frame_index for read in reads} == {0, 1, 2}


def test_read_drops_boxes_outside_every_region():
    # x=190 lands in the right-hand weapon HUD; x=10 lands in nothing we asked for.
    region = TextRegion("right_edge", 0.8, 0.0, 0.2, 1.0)
    engine = _SplitEngine([[_box(190, 50), _box(10, 50), _box(195, 80)]])
    ocr = RegionalOcr(engine=engine)

    reads = ocr.read([_frame()], [region])

    assert engine.rec_batch_sizes == [2], "only in-region boxes reach recognition"
    assert all(read.region == "right_edge" for read in reads)


def test_read_skips_recognition_entirely_when_nothing_is_in_region():
    engine = _SplitEngine([[_box(10, 10)]])
    ocr = RegionalOcr(engine=engine)

    reads = ocr.read([_frame()], [TextRegion("far_right", 0.9, 0.0, 0.1, 1.0)])

    assert reads == []
    assert engine.rec_calls == 0


def test_read_filters_by_confidence():
    # The fake engine scores every read at 0.9; a fresh engine per call because
    # detection consumes one frame of scripted boxes each time.
    strict = RegionalOcr(engine=_SplitEngine([[_box(50, 50)]]))
    lenient = RegionalOcr(engine=_SplitEngine([[_box(50, 50)]]))

    assert strict.read([_frame()], WHOLE_FRAME, min_confidence=0.95) == []
    assert len(lenient.read([_frame()], WHOLE_FRAME, min_confidence=0.5)) == 1


def test_read_falls_back_when_engine_has_no_split_api():
    engine = _UnsplitEngine([["100 HP"], ["VICTORY", "ROYALE"]])
    ocr = RegionalOcr(engine=engine)

    reads = ocr.read([_frame(), _frame()], WHOLE_FRAME)

    assert engine.calls == 2
    assert [read.text for read in reads] == ["100 HP", "VICTORY", "ROYALE"]
    assert [read.frame_index for read in reads] == [0, 1, 1]
    assert {read.region for read in reads} == {"frame"}


def test_texts_by_frame_groups_reads_and_omits_empty_frames():
    engine = _SplitEngine([[_box(50, 50)], [], [_box(60, 60), _box(70, 70)]])
    ocr = RegionalOcr(engine=engine)

    grouped = ocr.texts_by_frame([_frame(), _frame(), _frame()], WHOLE_FRAME)

    assert sorted(grouped) == [0, 2]
    assert len(grouped[0]) == 1
    assert len(grouped[2]) == 2


def test_fortnite_regions_stay_inside_the_frame():
    for region in FORTNITE_REGIONS:
        assert 0.0 <= region.x < 1.0
        assert 0.0 <= region.y < 1.0
        assert region.x + region.width <= 1.0 + 1e-9, region.name
        assert region.y + region.height <= 1.0 + 1e-9, region.name
