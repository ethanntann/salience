"""Read-only, optional import of legacy human preference labels.
v2 review decisions take precedence. Imported labels are not synthetic by definition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

# Positive/negative and how much the action is worth. Mirrors v1's weights so
# the two pipelines are measured against the same notion of "liked".
_ACTIONS: dict[str, tuple[int, float]] = {
    "favorite": (1, 1.0),
    "keep": (1, 0.6),
    "supervised_keep": (1, 0.6),
    "skip": (0, 0.2),
    "boring": (0, 0.8),
    "delete": (0, 1.0),
    "supervised_skip": (0, 0.2),
}

# v1 discounts `supervised_*` to 45%, and v2 copied that without checking why.
# The reason it exists in v1: imported labels there could be stale (rated
# against an older model's output) or made without watching the clip, so they
# are weaker evidence than a click made moments ago in the review UI.
#
# Neither applies to this file. `import_labeled_jsonl` records human keep/skip
# decisions verbatim, and the person who made them confirmed they were
# deliberate. Discounting them was inherited caution, not a measured choice, so
# the weights above are used as written and no multiplier is applied.


@dataclass(frozen=True)
class Preference:
    path: Path
    label: int  # 1 = liked, 0 = not
    weight: float
    organic: bool
    action: str


def load(db_path: Path | None, *, require_media: bool = True) -> list[Preference]:
    """Latest decision per clip, as weighted preferences."""
    if db_path is None:
        return []
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            select c.path as path, f.action as action, f.id as id
            from feedback_events f
            join clips c on c.id = f.clip_id
            order by f.id
            """
        ).fetchall()
    finally:
        conn.close()

    # Later rows win: a clip re-reviewed has changed the user's mind, and only
    # the current opinion is a label.
    latest: dict[str, sqlite3.Row] = {}
    for row in rows:
        latest[row["path"]] = row

    preferences: list[Preference] = []
    for path, row in latest.items():
        action = str(row["action"])
        if action not in _ACTIONS:
            continue
        label, weight = _ACTIONS[action]
        organic = not action.startswith("supervised_")
        preferences.append(
            Preference(
                path=Path(path),
                label=label,
                weight=weight,
                organic=organic,
                action=action,
            )
        )
    # Ranking stored descriptions needs label identity, not the original video.
    # Keep missing media excluded from workflows that need to decode it.
    return [p for p in preferences if not require_media or p.path.exists()]


class PreferenceIndex:
    """Looks up a label for a clip by path, falling back to filename.

    The legacy database records where the footage lived on the machine that
    produced it. Restoring the library elsewhere -- which `relink` does -- makes
    every stored path disagree with it: a path-only join matched 0 of 348 clips
    and `rank` reported "not enough labelled clips".

    Exact path always wins. The filename fallback is deliberately weaker than a
    path match and is counted in `matched_by_name` so a caller can report it,
    because two files sharing a name are not proof of the same recording. A name
    that is ambiguous on either side is refused rather than guessed at.
    """

    def __init__(self, preferences: list[Preference]) -> None:
        self._by_path = {p.path.resolve(): p for p in preferences}
        counts: dict[str, int] = {}
        for pref in preferences:
            counts[pref.path.name] = counts.get(pref.path.name, 0) + 1
        self._by_name = {
            p.path.name: p for p in preferences if counts[p.path.name] == 1
        }
        self.matched_by_name = 0

    def get(self, path: Path) -> "Preference | None":
        exact = self._by_path.get(path.resolve())
        if exact is not None:
            return exact
        named = self._by_name.get(path.name)
        if named is not None:
            self.matched_by_name += 1
        return named


def v1_final_scores(db_path: Path | None) -> dict[str, float]:
    """v1's shipped ranking, keyed by clip filename."""
    if db_path is None:
        return {}
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "select c.path as path, s.final_score as score"
            " from clip_scores s join clips c on c.id = s.clip_id"
        ).fetchall()
    finally:
        conn.close()
    return {Path(row["path"]).name: float(row["score"]) for row in rows}


def conflict_engagement_clips(
    preferences: list["Preference"], *, group_fn, event_of
) -> list["Preference"]:
    """Clips from fights the user split -- kept one, skipped another.

    These are the most informative comparisons in the whole dataset. Same fight,
    same weapon, same map, seconds apart: almost every confound is held constant,
    so an ordering between them isolates taste itself. Across-fight pairs can be
    won by noticing "eliminations beat knocks"; these cannot.
    """
    by_path = {p.path: p for p in preferences}
    chosen: list[Preference] = []
    for group in group_fn([p.path for p in preferences], event_of=event_of):
        labels = {by_path[c].label for c in group.clips if c in by_path}
        if len(labels) > 1:
            chosen.extend(by_path[c] for c in group.clips if c in by_path)
    return chosen


def pilot_sample(
    preferences: list[Preference], *, limit: int, organic_first: bool = True
) -> list[Preference]:
    """Pick a balanced subset to describe first.

    Balanced because pairwise accuracy needs both classes, and organic-first
    because those 20 events are the highest-quality signal in the set -- if the
    metric cannot be measured on the best labels, more of the weaker ones will
    not rescue it.
    """
    ranked = sorted(
        preferences,
        key=lambda p: (0 if (organic_first and p.organic) else 1, -p.weight),
    )
    positives = [p for p in ranked if p.label == 1]
    negatives = [p for p in ranked if p.label == 0]
    half = max(1, limit // 2)
    chosen = positives[:half] + negatives[:half]
    return chosen[:limit]
