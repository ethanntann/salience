"""Shared fight ranking for the CLI and browser.
Keep siblings accessible: reliable within-fight ordering is not established."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline_v2 import engagements
from pipeline_v2 import facts as facts_mod
from pipeline_v2 import feedback as feedback_mod
from pipeline_v2 import store
from pipeline_v2 import taste as taste_mod
from pipeline_v2 import prompt
from pipeline_v2 import config
from pipeline_v2 import schema


SPEECH_VARIANT = "speech"
# Shares speech is tried at, against the description. 0.3 measured neutral on
# a player whose talk was callouts; 1.0 is for a player whose keeps are mostly
# what was said.
SPEECH_WEIGHTS = (0.0, 0.3, 1.0)
# Speech is on unless the player's ratings show it hurts. It used to be off
# unless they showed it helped, which a player who says six things a session
# can never show -- and what was said is the one signal the frames cannot
# carry. 0.3 was neutral where speech did not matter (92.8% vs 92.6%).
DEFAULT_SPEECH_WEIGHT = 0.3
# A difference smaller than this on held-out minutes is noise: it can neither
# switch speech off nor turn it up.
MIN_SPEECH_GAIN = 0.01
# Held-out blocks for choosing the weight: neighbouring windows of one session
# are near-duplicates, so they must fall in the same fold.
CV_BLOCK_SECONDS = 120.0
# A keep the player marked as loved counts as two. Measured on 4 loved keeps
# (2026-10-23, six sessions each ranked by the other five): 1, 2 and 4 all put
# 31 of 39 keeps in the top 10 and the loved ones at #1-#3; nothing to choose yet.
LOVE_WEIGHT = 2.0
# A window of a clip or montage the player brought in as one they like
# (`capture.imports`) counts as half a keep of their own. Simulated
# (2026-11-06), 10 own ratings ranked keeps above skips 78% of the time, 85%
# with 20 liked windows at half weight and 86% at full: half, because a liked
# window is the player's word on a moment, not on one of their own windows.
IMPORT_WEIGHT = 0.5
# From this many own ratings, held out a session at a time, imports must not
# make the ranking worse. 20 random windows passed off as liked took 10
# ratings from 78% to 57%: a bad import is worse than none.
IMPORT_CHECK_MIN = 30
IMPORT_CHECK_FILE = "imports_check.json"


def _cv_group(clip, fallback: str) -> str:
    """The session minute-block a capture clip belongs to, from its sidecar."""
    import json

    try:
        sidecar = json.loads(
            Path(clip.path).with_suffix(".capture.json").read_text(encoding="utf-8"))
        block = int(float(sidecar["segment_start_sec"]) // CV_BLOCK_SECONDS)
    except (OSError, ValueError, KeyError, TypeError):
        return fallback
    return f"{Path(clip.path).parent.parent}:{block}"


def choose_speech_weight(
    training: list, speech: dict, dims: int, groups: dict, mode: str
) -> tuple[float, dict]:
    """How much what was said should count, judged on this player's own ratings.

    Returns (weight, held-out accuracy per weight tried). `DEFAULT_SPEECH_WEIGHT`
    unless the ratings say otherwise: off if it measurably hurts, full if full
    measurably beats it. Without enough speech to judge, the default stands.
    """
    from dataclasses import replace

    if not dims:
        return 0.0, {}
    voiced = sum(e.clip_sha in speech for e in training)
    if voiced < 10:
        return DEFAULT_SPEECH_WEIGHT, {}
    regrouped = [replace(e, engagement=groups.get(e.clip_sha, e.engagement)) for e in training]
    tried: dict[float, float] = {}
    for weight in SPEECH_WEIGHTS:
        examples = [
            replace(e, embedding=taste_mod.with_speech(
                e.embedding, speech.get(e.clip_sha), weight, dims))
            for e in regrouped
        ]
        accuracy, pairs = taste_mod.pairwise_accuracy(
            examples, mode, folds=5, model="logistic")
        if not pairs:
            return DEFAULT_SPEECH_WEIGHT, {}
        tried[weight] = accuracy
    default = DEFAULT_SPEECH_WEIGHT
    if tried[0.0] > tried[default] + MIN_SPEECH_GAIN:
        return 0.0, tried
    if tried[1.0] > tried[default] + MIN_SPEECH_GAIN:
        return 1.0, tried
    return default, tried


# The speech weight is chosen again only once this many more labels exist:
# chosen every run, it was 15 fits and 16 of the 18s ranking took after a
# session (2026-11-05), for an answer that moves only as the player rates.
SPEECH_WEIGHT_RECHOOSE = 25
SPEECH_WEIGHT_FILE = "speech_weight.json"


def _remembered_speech_weight(training, speech, dims, groups, mode, variant) -> tuple[float, dict]:
    import json

    path = config.DATA_DIR / SPEECH_WEIGHT_FILE
    key = f"{variant}|{mode}|{dims}"
    try:
        kept = json.loads(path.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError):
        kept = None
    if kept and abs(len(training) - kept["labels"]) < SPEECH_WEIGHT_RECHOOSE:
        return float(kept["weight"]), {float(k): v for k, v in kept.get("tried", {}).items()}
    weight, tried = choose_speech_weight(training, speech, dims, groups, mode)
    try:
        everything = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        everything[key] = {"labels": len(training), "weight": weight, "tried": {str(k): v for k, v in tried.items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(everything), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return weight, tried


@dataclass(frozen=True)
class RankedFight:
    score: float
    best: store.StoredClip
    siblings: list[store.StoredClip]


def _examples(conn, variant: str):
    """(stored clips, engagement groups, example(clip, label, weight)) for `variant`."""
    mechanics = schema.mechanics_for(variant)
    stored = store.all_clips(conn, variant)
    vectors = store.embeddings(conn, variant)
    images = store.embeddings(conn, "image")
    groups = engagements.group(
        [c.path for c in stored], event_of=facts_mod.event_from_filename
    )
    engagement_of = {
        clip: f"eng{index}"
        for index, group in enumerate(groups)
        for clip in group.clips
    }

    def example(clip, label: int, weight: float) -> taste_mod.Example:
        return taste_mod.Example(
            clip_sha=clip.clip_sha,
            engagement=engagement_of.get(clip.path, clip.clip_sha),
            label=label,
            weight=weight,
            fields=taste_mod.field_vector(clip.payload, clip.facts, mechanics),
            embedding=vectors.get(clip.clip_sha, []),
            image=images.get(clip.clip_sha),
        )

    return stored, groups, example


def _imported(conn) -> set[str]:
    return {row[0] for row in conn.execute("select clip_sha from imported")}


def _session_of(clip) -> str:
    return str(Path(clip.path).parent.parent)


def _pairwise(ranker, examples, mode) -> tuple[float, int]:
    """(share of keep/skip pairs in order, pairs) for `examples`."""
    kept = [ranker.score(e, mode) for e in examples if e.label]
    skipped = [ranker.score(e, mode) for e in examples if not e.label]
    pairs = len(kept) * len(skipped)
    if not pairs:
        return 0.0, 0
    return sum((k > s) + 0.5 * (k == s) for k in kept for s in skipped) / pairs, pairs


def imports_help(own: list, liked: list, sessions: dict[str, str], mode: str) -> tuple[bool, dict]:
    """Whether the liked windows rank this player's own ratings no worse.

    Each session's ratings are ranked by a model trained on the others', with
    and without the liked windows. The imports are only ever trained on: what
    is being judged is the player's own word. Too few ratings to judge, and
    they are used: that is when they matter most.
    """
    held = sorted({sessions[e.clip_sha] for e in own})
    if len(own) < IMPORT_CHECK_MIN or len(held) < 2 or not liked:
        return True, {}
    totals = {False: [0.0, 0], True: [0.0, 0]}
    for session in held:
        test = [e for e in own if sessions[e.clip_sha] == session]
        train = [e for e in own if sessions[e.clip_sha] != session]
        if len({e.label for e in train}) < 2 or len({e.label for e in test}) < 2:
            continue
        for with_liked in (False, True):
            ranker = taste_mod.LogisticRanker().fit(train + (liked if with_liked else []), mode)
            accuracy, pairs = _pairwise(ranker, test, mode)
            totals[with_liked][0] += accuracy * pairs
            totals[with_liked][1] += pairs
    if not totals[False][1]:
        return True, {}
    tried = {name: round(totals[on][0] / totals[on][1], 4) for name, on in (("without", False), ("with", True))}
    return tried["with"] >= tried["without"] - MIN_SPEECH_GAIN, tried


def _remembered_imports_help(own, liked, sessions, mode, variant) -> tuple[bool, dict]:
    """`imports_help`, judged again only once the ratings or the imports change much."""
    import json

    path = config.DATA_DIR / IMPORT_CHECK_FILE
    key = f"{variant}|{mode}"
    try:
        kept = json.loads(path.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError):
        kept = None
    if (kept and kept["liked"] == len(liked)
            and abs(len(own) - kept["labels"]) < SPEECH_WEIGHT_RECHOOSE):
        return bool(kept["use"]), kept.get("tried", {})
    use, tried = imports_help(own, liked, sessions, mode)
    try:
        everything = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        everything[key] = {"labels": len(own), "liked": len(liked), "use": use, "tried": tried}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(everything), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return use, tried


def _own_labels(conn) -> tuple[dict[str, int], set[str]]:
    own = {
        row["clip_sha"]: int(row["label"])
        for row in conn.execute("select clip_sha, label from preferences")
    }
    loved = {row[0] for row in conn.execute("select clip_sha from rating_notes where love = 1")}
    return own, loved


def ranked_fights(
    conn,
    *,
    db_path: Path | None = None,
    variant: str | None = None,
    mode: str = config.DEFAULT_RANK_MODE,
    info: dict | None = None,
) -> tuple[list[RankedFight], int]:
    """Returns (fights ordered best first, number of labels trained on).

    `info`, when given, is filled with how speech was weighed and why.
    """
    cfg = config.load_config()
    variant = variant or prompt.active_variant(cfg)
    stored, groups, example = _examples(conn, variant)
    prefs = feedback_mod.PreferenceIndex(
        feedback_mod.load(db_path, require_media=False)
    )
    own, loved = _own_labels(conn)
    imported = _imported(conn)

    training = []
    liked = []
    for clip in stored:
        own_label = own.get(clip.clip_sha)
        pref = prefs.get(clip.path)
        if own_label is None and pref is None:
            if clip.clip_sha in imported:
                candidate = example(clip, 1, IMPORT_WEIGHT)
                if taste_mod.has_features(candidate, mode):
                    liked.append(candidate)
            continue
        candidate = example(
            clip,
            own_label if own_label is not None else pref.label,
            (LOVE_WEIGHT if clip.clip_sha in loved else 1.0)
            if own_label is not None else pref.weight,
        )
        if taste_mod.has_features(candidate, mode):
            training.append(candidate)
    by_sha = {clip.clip_sha: clip for clip in stored}
    if liked:
        use, tried = _remembered_imports_help(
            training, liked, {e.clip_sha: _session_of(by_sha[e.clip_sha]) for e in training},
            mode, variant)
        if info is not None:
            info.update(imports=len(liked), imports_used=use, imports_cv=tried)
        if use:
            training = training + liked
    if len(training) < 10 or len({e.label for e in training}) < 2:
        return [], len(training)

    speech = store.embeddings(conn, SPEECH_VARIANT)
    dims = len(next(iter(speech.values()), []))
    # An imported window has no speech of the player's, and silence is not
    # what they like about it: it gets the average of what their own say.
    imported_shas = {e.clip_sha for e in liked}
    own_training = [e for e in training if e.clip_sha not in imported_shas]
    if dims and imported_shas and own_training:
        said = [speech.get(e.clip_sha, [0.0] * dims) for e in own_training]
        average = [sum(column) / len(said) for column in zip(*said)]
        speech = {**speech, **{sha: average for sha in imported_shas}}
    blocks = {
        e.clip_sha: _cv_group(by_sha[e.clip_sha], e.engagement or e.clip_sha)
        for e in own_training
    }
    # Judged on the player's own ratings only, like the imports are.
    weight, tried = _remembered_speech_weight(own_training, speech, dims, blocks, mode, variant)
    if info is not None:
        info.update(speech_weight=weight, speech_cv=tried)

    def heard(e: taste_mod.Example) -> taste_mod.Example:
        if not weight:
            return e
        from dataclasses import replace

        return replace(e, embedding=taste_mod.with_speech(
            e.embedding, speech.get(e.clip_sha), weight, dims))

    training = [heard(e) for e in training]
    ranker = taste_mod.LogisticRanker().fit(training, mode)
    candidates = ((clip, heard(example(clip, 0, 1.0))) for clip in stored)
    scores = {
        clip.clip_sha: ranker.score(candidate, mode)
        for clip, candidate in candidates
        if taste_mod.has_features(candidate, mode)
    }

    by_path = {clip.path: clip for clip in stored}
    fights: list[RankedFight] = []
    for group in groups:
        members = [
            by_path[path]
            for path in group.clips
            if path in by_path and by_path[path].clip_sha in scores
        ]
        if not members:
            continue
        # The highest score represents the fight; siblings keep their original
        # chronological order. There is no validated within-fight ordering.
        best = max(members, key=lambda c: scores[c.clip_sha])
        fights.append(
            RankedFight(
                score=scores[best.clip_sha],
                best=best,
                siblings=[c for c in members if c.clip_sha != best.clip_sha],
            )
        )
    fights.sort(key=lambda fight: fight.score, reverse=True)
    return fights, len(training)


# Kills the player has rated, keeps and skips, from at least two sessions,
# before a kill can be judged one they would skip.
KEEP_MIN_KILLS = 30
# Below this chance of a keep, a kill is not a highlight. Held out a session
# at a time (2026-11-01, 167 rated kills, 58 kept): 83% of kept kills shown,
# 56% of skipped ones left out; the taste ranking alone was 64% and 67% at
# its best, and at the same 83% left out a third as many.
KEEP_CHANCE = 0.4


def keep_chances(conn, candidates: list, is_kill, extras, *, variant: str | None = None,
                 mode: str = config.DEFAULT_RANK_MODE) -> dict[str, float]:
    """The chance the player keeps each of `candidates` (kills), from their
    rated kills; empty until there are `KEEP_MIN_KILLS` of them.

    Two steps. Taste trained on kills alone: trained on every window, it had
    learned kill-or-not, and told a kept kill from a skipped one 65% of the
    time; on kills, 73%. Then that with `extras(clip)` (the player's keys,
    mic and kill type, each standardised within its session) in a small
    model, its taste a session at a time out of sample: 78%.
    """
    from sklearn.linear_model import LogisticRegression

    variant = variant or prompt.active_variant(config.load_config())
    stored, _, example = _examples(conn, variant)
    own, loved = _own_labels(conn)
    speech = store.embeddings(conn, SPEECH_VARIANT)
    dims = len(next(iter(speech.values()), []))

    def heard(e: taste_mod.Example) -> taste_mod.Example:
        if not dims:
            return e
        from dataclasses import replace

        return replace(e, embedding=taste_mod.with_speech(
            e.embedding, speech.get(e.clip_sha), DEFAULT_SPEECH_WEIGHT, dims))

    rated = [(clip, heard(example(clip, own[clip.clip_sha],
                                  LOVE_WEIGHT if clip.clip_sha in loved else 1.0)))
             for clip in stored if clip.clip_sha in own and is_kill(clip)]
    rated = [(clip, e) for clip, e in rated if taste_mod.has_features(e, mode)]
    session = {clip.clip_sha: str(Path(clip.path).parent.parent) for clip, _ in rated}
    sessions = set(session.values())
    if len(rated) < KEEP_MIN_KILLS or len(sessions) < 2 or len({e.label for _, e in rated}) < 2:
        return {}
    taste: dict[str, float] = {}
    for held in sessions:
        train = [e for clip, e in rated if session[clip.clip_sha] != held]
        if len({e.label for e in train}) < 2:
            return {}
        ranker = taste_mod.LogisticRanker().fit(train, mode)
        for clip, e in rated:
            if session[clip.clip_sha] == held:
                taste[clip.clip_sha] = ranker.score(e, mode)
    model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000).fit(
        [[taste[clip.clip_sha], *extras(clip)] for clip, _ in rated], [e.label for _, e in rated])
    ranker = taste_mod.LogisticRanker().fit([e for _, e in rated], mode)
    chances = {}
    for clip in candidates:
        e = heard(example(clip, 0, 1.0))
        if taste_mod.has_features(e, mode):
            row = [ranker.score(e, mode), *extras(clip)]
            chances[clip.clip_sha] = float(model.predict_proba([row])[0][1])
    return chances
