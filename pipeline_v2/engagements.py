"""Group clips that cover the same fight.

Auto-clippers save one file per *engine event*, not per engagement. A knock and
the elimination that follows it three seconds later become two files with
overlapping footage. Measured on a 10-clip batch, about 40% were duplicate
coverage of the same fight.

That matters twice over: the inbox shows the same moment twice, and the taste
model trains on rows that are not independent -- one fight the user happened to
like counts double, which quietly inflates whatever it contains.

Grouping is deterministic and free: the filename carries a save timestamp, and
two clips whose recorded spans overlap are the same engagement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

# "Fortnite 2026.07.12 - 04.45.03.62.Elimination.DVR"
_STAMP = re.compile(
    r"(?P<y>\d{4})\.(?P<mo>\d{2})\.(?P<d>\d{2})\s*-\s*"
    r"(?P<h>\d{2})\.(?P<mi>\d{2})\.(?P<s>\d{2})"
)

# Ranked by how completely each event closes out an engagement. When one fight
# produced several clips, the finish is the one worth reviewing.
_EVENT_RANK = {
    "victory": 4,
    "double_elimination": 3,
    "elimination": 2,
    "knock": 1,
    None: 0,
}


def clip_time(path: Path) -> datetime | None:
    match = _STAMP.search(path.stem)
    if match is None:
        return None
    parts = {key: int(value) for key, value in match.groupdict().items()}
    try:
        return datetime(
            parts["y"], parts["mo"], parts["d"], parts["h"], parts["mi"], parts["s"]
        )
    except ValueError:
        return None


"""
MEASURED CONSEQUENCE OF DISCARDING (do not re-enable by default):

Against 553 hand-labelled clips, choosing the "most complete event" as the
primary and dropping its siblings **discards a clip the user kept, in favour of
one they skipped, 14% of the time** (21 of 151 multi-clip engagements).

The hierarchy below is a guess about what matters -- that an elimination beats a
knock. The labels say that guess is wrong often enough to be a product bug:
throwing away the clip someone wanted is the worst failure this pipeline has.

So grouping is now used for what it is actually sound for -- keeping preference
pairs independent, and presenting siblings together -- and not for deletion.
`--dedupe` still exists to save API calls, with that cost understood.
"""


@dataclass
class Engagement:
    """One fight, and every clip the auto-clipper saved of it."""

    clips: list[Path] = field(default_factory=list)
    primary: Path | None = None

    @property
    def duplicates(self) -> list[Path]:
        return [clip for clip in self.clips if clip != self.primary]


def group(
    clips: list[Path],
    *,
    event_of,
    window_sec: float = 25.0,
) -> list[Engagement]:
    """Chain clips saved within ``window_sec`` of each other into engagements.

    ``window_sec`` defaults to slightly more than a clip length, because two
    saves closer together than one clip duration necessarily share footage.
    Chaining is single-linkage: a long fight producing knock, knock, elimination
    stays one engagement rather than splitting on the middle gap.

    ``event_of`` maps a path to its filename event label, so this module does not
    need to know how filenames encode events.
    """
    stamped = sorted(
        ((clip_time(clip), clip) for clip in clips),
        key=lambda pair: (pair[0] is None, pair[0] or datetime.min, str(pair[1])),
    )

    engagements: list[Engagement] = []
    previous: datetime | None = None
    for moment, clip in stamped:
        if (
            moment is None
            or previous is None
            or not engagements
            or (moment - previous).total_seconds() > window_sec
        ):
            engagements.append(Engagement(clips=[clip]))
        else:
            engagements[-1].clips.append(clip)
        previous = moment

    for engagement in engagements:
        engagement.primary = max(
            engagement.clips,
            key=lambda clip: (
                _EVENT_RANK.get(event_of(clip), 0),
                # Tie-break on save time: the later save saw more of the fight.
                clip_time(clip) or datetime.min,
            ),
        )
    return engagements
