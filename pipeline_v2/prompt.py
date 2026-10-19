"""Prompt construction: frames in, one description out.

This file used to inject OCR facts as context the model should prefer. Ablation
killed that: the facts left summary ranking unchanged and made the structured
output 4-10 points worse, because the model copied OCR errors it would not have
made unaided. The frames are now the only input.

`hud_conflict` survives in the schema with nothing to conflict against. It stays
because it costs one token and would be the first signal that a future variant
reintroduces bad context -- but the UI-drift alarm it used to provide now needs
a standalone OCR read-rate check instead.
"""

from __future__ import annotations

import base64
from pathlib import Path

from pipeline_v2 import schema
from pipeline_v2.facts import ClipFacts

SYSTEM = (
    "You describe short Fortnite gameplay clips. "
    "You output only one parseable JSON object, no markdown, no commentary."
)

_INSTRUCTIONS = """\
The frames below are in chronological order and are centred on one event.

Return exactly this JSON shape:

{
  "summary": "<one sentence, max 40 words, describing what happens>",
  "outcome": "elimination | knock | death | no_event | other",
  "weapon_name": "<the weapon's in-game name as shown, e.g. 'Lancehead Pistol'>",
  "player_health_state": "healthy | damaged | critical | unknown",
  "mechanics": ["<zero or more tags from the list below, nothing else>"],
  "hud_conflict": null,
  "confidence": "high | medium | low"
}

Allowed "mechanics" values -- use these exact strings or omit the tag:
close_range, mid_range, long_range, headshot, build_fight, high_ground,
vertical_movement, finish_downed, weapon_swap, multi_opponent, low_health

"player_health_state" comes from the player's own health and shield bars at the
bottom-left of the screen, not from the opponent's. Read the numbers if they are
legible: full or near-full health is "healthy", visibly reduced is "damaged",
a small fraction remaining is "critical". Use "unknown" if the bars are not
readable in these frames.

Rules for "summary":
- Describe only what you can see: who does what, at what range, in what state.
- Do NOT judge quality. Never write "impressive", "great", "clean", "lucky",
  "satisfying", or any similar word. Something else ranks these clips; your job
  is to say what happened, accurately and specifically.
- Prefer concrete detail over general phrasing. "Finishes a downed opponent at
  point-blank range while taking fire from a second player" is useful.
  "A player gets a kill" is not.

Use "unknown" whenever the frames do not support an answer. An honest "unknown"
is more useful than a guess. Leave "hud_conflict" null.
"""


# v3.1 asks for *execution* detail rather than event identity. Siblings of one
# fight produce nearly identical summaries under v3.0, and within-fight accuracy
# sits at ~62% against ~81% across fights. What separates two clips of one kill is
# how it was executed, and none of that appears in "eliminates an opponent at
# close range with a shotgun".
#
# Every added field is an observable count or state -- shots, movement, return
# fire, duration. Deliberately not quality judgements: the describe-never-judge
# rule stands, because a model that rates clips teaches the taste model its own
# opinion instead of the user's.
_EXECUTION = """
Also report, as plain observation and never as praise or criticism:

  "shots_fired": <integer or null>       shots the player took at this target
  "target_moving": "yes | no | unknown"
  "player_moving": "yes | no | unknown"
  "return_fire": "yes | no | unknown"    was the player being shot at
  "engagement_seconds": <number or null> roughly how long the exchange lasted
  "execution_notes": "<one short clause on HOW it was done: aim corrections,
                      missed shots, hesitation, repositioning, range change.
                      Observable actions only, no evaluation.>"
"""


# v3.2 forces target/weapon disambiguation.
#
# Observed failure: a clip where the player cracked one enemy's shield with a
# Hunting Rifle, then redirected onto a *second* enemy and headshot them with the
# same rifle, came back as "switches to a pistol to secure a headshot elimination
# on the downed enemy". Two errors from one cause -- a target change narrated as
# a weapon change, and two opponents collapsed into one.
#
# The root weakness is entity tracking across frames, which VLMs are poor at.
# Asking the two questions *separately* stops one being substituted for the
# other, and `opponents_engaged` turns a prose claim into a countable field.
#
# This matters for ranking, not only for reading: redirecting onto a fresh
# opponent is a better clip than finishing someone already downed, so descriptions
# that collapse the two rank good clips as mundane ones.
# The fields go INSIDE the JSON template, not appended after it. A first attempt
# appended them and the model returned every one as null -- it had been told to
# return "exactly this JSON shape" and did precisely that.
_TRACKING_FIELDS = """  "opponents_engaged": <integer>,
  "changed_target": "yes | no | unknown",
  "changed_weapon": "yes | no | unknown",
"""

_TRACKING = """
Tracking rules -- these three fields matter more than the summary:

  "opponents_engaged": how many DISTINCT enemies the player shot at
  "changed_target": did the player redirect onto a different enemy
  "changed_weapon": did the weapon in hand visibly change

Rules, in order of importance:
- Changing target is NOT changing weapon. Report a weapon change only if the
  weapon actually in hand visibly differs between frames. If the player swings
  aim onto a new enemy while holding the same gun, that is "changed_target".
- Do not describe an opponent as downed, knocked, or finished unless you saw
  that opponent go down in these frames. A different enemy is a different enemy.
- If two enemies appear, say so in the summary rather than merging them.
"""


_ATTRIBUTION_FIELDS = """  "viewpoint": "own_gameplay | spectating | non_gameplay | unknown",
  "event_actor": "recording_player | spectated_player | other | unknown",
  "elimination_weapon_name": "<weapon used for the decisive hit, or unknown>",
  "weapon_evidence_frames": [<one-based frame numbers supporting decisive-hit attribution>],
  "attribution_notes": "<brief chronological evidence for weapon attribution, or why uncertain>",
  "mechanics_evidence": "<brief visible placement/edit/reset sequence, or unknown>",
"""

_ATTRIBUTION = """
Establish viewpoint first. SPECTATING, a watched player's name, spectator controls,
or equivalent HUD evidence indicate spectating, not a pre-match lobby. The avatar
on screen is not automatically the recording player. If unclear, use unknown.
Outcome describes the visible event; event_actor identifies whose action/death it
is. Explicitly say "the spectated player" in the summary when appropriate. Health
bars in spectator footage belong to the watched player, not the recording player.

Track the visible order: engagement, damage, first elimination evidence, aftermath.
weapon_name describes the weapon used in the engagement, not a later held weapon.
elimination_weapon_name requires evidence of the decisive hit. A weapon selected
after a kill, while looting, or while an elimination banner persists is NOT evidence
that it caused the kill. Check the held weapon and selected inventory slot at the
shot/damage frames; a rapid swap may already have happened when the banner appears.
Readable kill-feed evidence must match this actor and opponent, not another fight.
If the decisive action falls between frames or the name is unreadable, use unknown
or a supported weapon family in the summary instead of guessing an exact name.
For outcomes other than elimination, set elimination_weapon_name to unknown and
weapon_evidence_frames to []. Cite only supplied frame numbers. Never invent a
shot, weapon switch, or uninterrupted sequence across an unobserved gap.

Additional allowed mechanics: editing, edit_reset, piece_control.
editing requires a visible edit grid/selection or a supported shape-change sequence;
edit_reset requires evidence of restoring the edited piece. piece_control requires
evidence that the viewed player places/owns pieces that restrict an opponent's
space or routes; nearby builds alone are insufficient. Distinguish placing new
pieces, editing owned pieces, shooting structures, and moving through existing
builds. Describe supported mechanics concretely in the summary; otherwise stay
with build_fight or omit the claim. Transparent construction previews are not
automatically glass material. Do not infer ownership solely from color.

Confidence must account for unresolved actor, weapon, and mechanics attribution,
not just whether an elimination banner is visible. An unknown field is acceptable.
"""


# v3.5 changes one thing from v3.3: what the weapon question asks for.
#
# v3.3 asks for "the weapon's in-game name as shown" -- a claim about what is
# held, not about what killed anyone -- and the model answered it literally.
# Measured on 50 randomly sampled clips: it named a weapon on 7 of the 8 clips
# where nothing of the player's killed anything (their own death, or an opponent
# bleeding out), including a Rare Shield Potion. It never once abstained across
# all 348 descriptions.
#
# A single weapon field is also the wrong shape. The model sees ~4s of a ~22s
# auto-clip, and the recorder's own chunks choose their window at random, so the
# frames can hold no kill, one, or several with different weapons. Asking for one
# forced it to pick, and the annotator's "first kill pistol, second kill shotgun"
# could not agree with either answer. A list removes that false choice, and an
# empty list is how the model declines.
_V33_WEAPON_FIELD = (
    '  "weapon_name": "<the weapon\'s in-game name as shown, '
    "e.g. 'Lancehead Pistol'>\","
)
_KILL_WEAPON_FIELD = (
    '  "kill_weapons": ["<in-game name of each weapon that eliminated or '
    'knocked someone in these frames, earliest first>"],'
)

_KILL_WEAPON_RULES = """
Rules for "kill_weapons":
- List one entry per elimination or knock you can actually see happen in these
  frames, naming the weapon that landed the decisive hit, earliest first.
- Judge each from the frames showing the shot or the damage, not from what is in
  hand afterwards. A weapon picked up, swapped to, or held while looting AFTER a
  kill did not cause it, and an elimination banner stays on screen across a
  weapon swap -- a visible banner says nothing about the weapon held beside it.
- Return [] when no elimination or knock happens in these frames. That includes
  the recording player being eliminated, an opponent bleeding out from a knock
  you cannot see, and frames holding only looting, building or travel. An empty
  list is the correct answer, not a failure to find something.
- Consumables and utility items -- a shield potion, a Chug Jug, a launch pad --
  belong here only if one visibly caused a kill. Normally none did.
- Name a weapon only where the decisive moment is visible. If someone clearly
  dies but the cause is off-screen or unreadable, write "unknown" for that entry
  rather than guessing from what is on screen.
"""


def active_variant(cfg) -> str:
    """The storage variant a read should default to for the current runtime.

    Writes go through `storage_variant`, which appends a model tag. Reads used
    the bare prompt version, so configuring any model other than the corpus one
    made `review`, `rank` and `taste` look for rows nothing had written. The
    per-clip `-window` suffix is deliberately absent: that splits a corpus by
    source and a caller wanting it asks for it explicitly.
    """
    from pipeline_v2 import config

    if cfg.model and cfg.model != config.CORPUS_MODEL:
        return f"{cfg.prompt_version}@{model_tag(cfg.model)}"
    return cfg.prompt_version


def model_tag(model: str) -> str:
    """Short stable name for a model id, for use inside a storage variant."""
    return (model or "").rsplit("/", 1)[-1]


_OWN_KILL_FIELD = (
    '  "kill_weapons": ["<in-game name of each weapon THE RECORDING PLAYER used '
    'to eliminate or knock an opponent in these frames, earliest first>"],'
)

_OWN_KILL_RULES = """
Rules for "kill_weapons":
- List only weapons THE RECORDING PLAYER used to eliminate or knock an opponent
  in these frames, naming the one that landed the decisive hit, earliest first.
- These are the recording player's kills and nobody else's. A teammate's kill, a
  spectated player's kill, and an opponent's kill all contribute nothing here.
- If the recording player is the one eliminated, they killed nobody: return [].
  An "outcome" of "death" and a non-empty "kill_weapons" contradict each other,
  so if you set outcome to "death", kill_weapons must be [].
- Judge each from the frames showing the shot or the damage, not from what is in
  hand afterwards. A weapon picked up, swapped to, or held while looting AFTER a
  kill did not cause it, and an elimination banner stays on screen across a
  weapon swap -- a visible banner says nothing about the weapon held beside it.
- Return [] when the recording player eliminates or knocks nobody in these
  frames. That includes an opponent bleeding out from a knock you cannot see,
  and frames holding only looting, building or travel. An empty list is the
  correct answer, not a failure to find something.
- Consumables and utility items -- a shield potion, a Chug Jug, a launch pad --
  belong here only if one visibly caused a kill. Normally none did.
- Name a weapon only where the decisive moment is visible. If the recording
  player clearly downs someone but the weapon is off-screen or unreadable, write
  "unknown" for that entry rather than guessing from what is on screen.
"""


def storage_variant(
    prompt_version: str, clip_facts: ClipFacts, model: str | None = None
) -> str:
    """The label a description built from these facts must be stored under.

    ``build_prompt`` emits materially different instructions for a uniformly
    sampled capture window ("there may be no combat") than for an auto-clipped
    file centred on an event. Both used to be stored as plain ``v3.3``, so
    `taste --variant v3.3` would have averaged two prompts into one measurement.
    The cache was never affected: its key covers the exact request bytes.

    The attribution variant replaces the whole template regardless of the event
    source, so it needs no window suffix.
    """
    from pipeline_v2 import config

    variant = prompt_version
    if (
        prompt_version != schema.ATTRIBUTION_PROMPT_VERSION
        and clip_facts.event_source == "capture_window"
    ):
        variant = f"{prompt_version}-window"
    # A different model answering the same question is a different corpus, not
    # more of the existing one.
    if model and model != config.CORPUS_MODEL:
        variant = f"{variant}@{model_tag(model)}"
    return variant


_DEFAULT_TAGS = """close_range, mid_range, long_range, headshot, build_fight, high_ground,
vertical_movement, finish_downed, weapon_swap, multi_opponent, low_health
"""

_CENTRED = "The frames below are in chronological order and are centred on one event."
_WINDOWED = (
    "The frames below are in chronological order from a uniformly sampled gameplay window. "
    "There may be no combat or completed event. Describe only what is visible; "
    "use no_event when there is no visible outcome. Do not infer an elimination "
    "merely because this window was recorded."
)


def build_prompt(clip_facts: ClipFacts, version: str | None = None) -> str:
    """Build the prompt for a version. Branch order preserves every frozen digest."""
    if version is None:
        from pipeline_v2 import config

        version = config.load_config().prompt_version

    if version == schema.ATTRIBUTION_PROMPT_VERSION:
        template = _INSTRUCTIONS.replace(
            "The frames below are in chronological order and are centred on one event.",
            "The frames below are in chronological order from a gameplay window. "
            "There may be no combat or completed event. Do not infer an elimination.",
        ).replace(
            '  "hud_conflict": null,', _ATTRIBUTION_FIELDS + '  "hud_conflict": null,'
        )
        return template + _ATTRIBUTION

    # v3.5 is the only variant that follows the capture-window wording, because
    # it is the arm meant to run on recorder chunks as well as auto-clipped files.
    if version in (
        schema.KILL_WEAPON_PROMPT_VERSION,
        schema.OWN_KILLS_PROMPT_VERSION,
        schema.WORD_BANK_PROMPT_VERSION,
        schema.GAME_CONTEXT_PROMPT_VERSION,
    ):
        own = version != schema.KILL_WEAPON_PROMPT_VERSION
        template = _INSTRUCTIONS
        if clip_facts.event_source == "capture_window":
            template = template.replace(_CENTRED, _WINDOWED)
        if _V33_WEAPON_FIELD not in template:
            raise ValueError("could not find the v3.3 weapon_name line to replace")
        field = _OWN_KILL_FIELD if own else _KILL_WEAPON_FIELD
        rules = _OWN_KILL_RULES if own else _KILL_WEAPON_RULES
        built = template.replace(_V33_WEAPON_FIELD, field) + rules
        if version in (schema.WORD_BANK_PROMPT_VERSION, schema.GAME_CONTEXT_PROMPT_VERSION):
            from pipeline_v2.games import fortnite

            if _DEFAULT_TAGS not in built:
                raise ValueError("could not find the mechanics list to extend")
            tags = _DEFAULT_TAGS.rstrip("\n") + ",\n" + ", ".join(fortnite.MECHANICS) + "\n"
            built = built.replace(_DEFAULT_TAGS, tags)
            if version == schema.WORD_BANK_PROMPT_VERSION:
                return built + fortnite.WORD_BANK
            built += fortnite.WORD_BANK_V38
            # What the game's log knows, when it covers this window.
            if clip_facts.player_name:
                built += fortnite.CONTEXT.format(player=clip_facts.player_name)
                phase = fortnite.PHASES.get(clip_facts.match_phase or "")
                if phase:
                    built += fortnite.CONTEXT_PHASE.format(phase=phase)
        return built

    if clip_facts.event_source == "capture_window":
        return _INSTRUCTIONS.replace(_CENTRED, _WINDOWED)
    if version == "v3.2b-track":
        template = _INSTRUCTIONS.replace(
            '  "hud_conflict": null,', _TRACKING_FIELDS + '  "hud_conflict": null,'
        )
        return template + _TRACKING
    if version == "v3.1-exec":
        return _INSTRUCTIONS + _EXECUTION
    return _INSTRUCTIONS


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_content(clip_facts: ClipFacts, frame_paths: list[Path]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": build_prompt(clip_facts)}]
    for index, path in enumerate(frame_paths):
        content.append({"type": "text", "text": f"Frame {index + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
    return content
