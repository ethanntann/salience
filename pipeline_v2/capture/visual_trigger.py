"""Can a cheap local pixel check tell that an elimination just happened?

The kill trigger is the one signal nothing supplies. The game log gives match
boundaries and the local player's death but never their eliminations; audio
loudness has not beaten a metronome on the one session it was tested against;
a VLM is seconds and cents per call, so it cannot run live.

v1 dismissed vision, but it measured *full OCR* at 98% of local runtime. This is
a different proposition: downscale the frame to a coarse grid and look at how a
few cells change. That is microseconds of work, not OCR.

Nothing here assumes where the banner is. The region is **discovered** by
comparing frames at the known event time against frames from the same clips away
from it -- 348 auto-clipped files whose filenames name the engine event give the
labels for free. A guessed rectangle would be a hypothesis dressed as a result.
"""

from __future__ import annotations

from dataclasses import dataclass

# Coarse enough to be free, fine enough to localise a HUD element. A 1920x1200
# frame becomes 768 cells, which is a few microseconds of arithmetic.
GRID_COLUMNS = 32
GRID_ROWS = 24


@dataclass(frozen=True)
class GridFrame:
    """One frame reduced to a small grid of mean brightness per cell."""

    cells: list[float]  # row-major, GRID_ROWS * GRID_COLUMNS
    width: int
    height: int

    def cell(self, row: int, column: int) -> float:
        return self.cells[row * GRID_COLUMNS + column]


def to_grid(image) -> GridFrame:
    """Reduce a HxWx3 array to per-cell mean brightness.

    Mean brightness rather than colour: a banner is a bright overlay on darker
    gameplay, and one number per cell keeps the live cost trivial.
    """
    import numpy as np

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    height, width = array.shape[:2]
    rows = np.array_split(array, GRID_ROWS, axis=0)
    cells: list[float] = []
    for band in rows:
        for column in np.array_split(band, GRID_COLUMNS, axis=1):
            cells.append(float(column.mean()))
    return GridFrame(cells=cells, width=width, height=height)


def cell_deltas(event: GridFrame, control: GridFrame) -> list[float]:
    """Per-cell brightness change from a control frame to an event frame."""
    return [e - c for e, c in zip(event.cells, control.cells, strict=True)]


def separation(
    event_deltas: list[list[float]], control_deltas: list[list[float]]
) -> list[float]:
    """How well each cell separates event frames from control frames.

    A standardised mean difference per cell (Cohen's d). Sign matters: a cell
    that reliably *brightens* at the event is as useful as one that darkens, but
    a cell whose change is large and inconsistent is noise, and dividing by the
    pooled spread is what tells them apart.
    """
    import numpy as np

    events = np.asarray(event_deltas, dtype=np.float64)
    controls = np.asarray(control_deltas, dtype=np.float64)
    if not len(events) or not len(controls):
        return [0.0] * (GRID_ROWS * GRID_COLUMNS)
    spread = np.sqrt(
        (events.var(axis=0, ddof=1) + controls.var(axis=0, ddof=1)) / 2.0
    )
    spread[spread < 1e-6] = 1e-6
    return list((events.mean(axis=0) - controls.mean(axis=0)) / spread)


def top_cells(scores: list[float], count: int = 12) -> list[tuple[int, int, float]]:
    """The most discriminative cells as (row, column, score), strongest first."""
    ranked = sorted(
        range(len(scores)), key=lambda i: abs(scores[i]), reverse=True
    )[:count]
    return [
        (index // GRID_COLUMNS, index % GRID_COLUMNS, scores[index])
        for index in ranked
    ]


def describe_region(cells: list[tuple[int, int, float]]) -> str:
    """Where on screen the discriminative cells sit, in plain terms."""
    if not cells:
        return "nowhere"
    rows = [r for r, _, _ in cells]
    columns = [c for _, c, _ in cells]
    vertical = ("top", "upper-middle", "lower-middle", "bottom")[
        min(3, int(4 * (sum(rows) / len(rows)) / GRID_ROWS))
    ]
    horizontal = ("left", "centre-left", "centre-right", "right")[
        min(3, int(4 * (sum(columns) / len(columns)) / GRID_COLUMNS))
    ]
    return f"{vertical} {horizontal}"
