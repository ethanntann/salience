"""The VLM response contract.

Deliberately small. M2 exists to find out which of these fields the model can
actually fill reliably, and every field that survives has to earn its place by
being either rankable or controllable.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Outcome = Literal["elimination", "knock", "death", "no_event", "other"]
Health = Literal["healthy", "damaged", "critical", "unknown"]
Confidence = Literal["high", "medium", "low"]

# Fixed vocabulary. Left free, the model produced `high_ground` and
# `high_ground_advantage` for one concept, which is useless for filtering. The
# embedding carries nuance; this field exists to be queried.
#
# This must list exactly the tags the active prompt offers, in that order.
# `test_mechanics_vocabulary` asserts that against the prompt text; the two
# previously disagreed, and `field_vector` spent three dimensions on tags the
# default prompt never mentioned, which were zero across all 1356 descriptions.
MECHANICS = (
    "close_range",
    "mid_range",
    "long_range",
    "headshot",
    # The new session includes builds; the old library's mode is not a contract.
    "build_fight",
    "high_ground",
    "vertical_movement",
    "finish_downed",
    "weapon_swap",
    "multi_opponent",
    "low_health",
)

# The attribution variant is the only prompt that asks for build/edit mechanics,
# so it is the only one whose descriptions can carry them. Offering them by
# default would change the v3.3 prompt text, invalidating every cached response.
ATTRIBUTION_MECHANICS = MECHANICS + ("editing", "edit_reset", "piece_control")

DEFAULT_PROMPT_VERSION = "v3.3"
ATTRIBUTION_PROMPT_VERSION = "v3.4-attribution-dev"

# Isolated single-change arm: redefines weapon_name as the weapon that CAUSED
# the outcome. Everything else is byte-identical to v3.3, so a difference in
# weapon accuracy is attributable to that one change. The attribution variant
# altered six things at once and could not support that claim.
KILL_WEAPON_PROMPT_VERSION = "v3.5-killweapon"

# v3.6 keeps v3.5's list and scopes it to the RECORDING PLAYER's kills. v3.5 asked
# for "each weapon that eliminated or knocked someone", and when the player was
# the one eliminated the model correctly named the weapon that killed *them* --
# literal, and not what the annotator meant by "the weapon that got the kill".
OWN_KILLS_PROMPT_VERSION = "v3.6-ownkills"

# v3.7 is v3.6 plus the game's word bank (`games.fortnite`): its terms, each
# with the visible cue that earns it, extra mechanics tags, and a ban on
# scenery. One change against v3.6, so a difference is attributable to it.
WORD_BANK_PROMPT_VERSION = "v3.7-wordbank"
# v3.7 plus what Fortnite's log knows (who the player is, the match phase) and
# sharper weapon and build-piece rules (`games.fortnite.WORD_BANK_V38`).
GAME_CONTEXT_PROMPT_VERSION = "v3.8-context"
# v3.8 plus two lines about the window (`notes`): what the player said, and
# what happened in the 8 seconds before. Beat v3.8 19 vs 17 of 33 kept
# moments in the top 10; HUD banner facts added nothing on top.
SAID_PROMPT_VERSION = "v3.9-said"
NOTES_PROMPT_VERSIONS = (SAID_PROMPT_VERSION,)

# One registry for V2_ABLATE. config maps the variable to a version; prompt
# branches on the version. Reading the environment in both places let the two
# disagree about which prompt was being built.
ABLATIONS = {
    "exec": "v3.1-exec",
    "track": "v3.2b-track",
    "attribution": ATTRIBUTION_PROMPT_VERSION,
    "killweapon": KILL_WEAPON_PROMPT_VERSION,
    "ownkills": OWN_KILLS_PROMPT_VERSION,
    "wordbank": WORD_BANK_PROMPT_VERSION,
    "context": GAME_CONTEXT_PROMPT_VERSION,
    "said": SAID_PROMPT_VERSION,
}


def mechanics_for(prompt_version: str) -> tuple[str, ...]:
    """The tag vocabulary a description at this version could legally contain."""
    if prompt_version == ATTRIBUTION_PROMPT_VERSION:
        return ATTRIBUTION_MECHANICS
    if prompt_version.startswith((WORD_BANK_PROMPT_VERSION, GAME_CONTEXT_PROMPT_VERSION, "v3.9")):
        from pipeline_v2.games import fortnite

        return MECHANICS + fortnite.MECHANICS
    return MECHANICS


class ClipDescription(BaseModel):
    """One VLM call's output: a sentence to embed, plus fields to control on."""

    # The sentence is the whole point -- it carries meaning no fixed label set
    # can, and it is what gets embedded for the taste model. It must describe,
    # never judge: if the model writes "impressive", the ranker learns the
    # model's taste instead of the user's.
    summary: str = Field(min_length=10, max_length=400)

    outcome: Outcome = "other"

    # Free text, not an enum. Measured over 10 clips: the model's *category* was
    # unstable (one gun came back "other" then "automatic" on consecutive clips)
    # while the in-game *name* was right every time. So it reports the name and
    # `weapon_ontology` -- deterministic, versioned, already written -- maps it.
    weapon_name: str = ""

    # v3.5. One weapon cannot describe these windows. The model is shown ~4s of
    # a ~22s auto-clip, and the recorder's own chunks pick their window at
    # random with no event anchor at all, so the frames may contain no kill, one,
    # or several with different weapons. A single field forced a choice between
    # them and scored as wrong whichever one the annotator did not mean.
    # Empty is the correct answer when no kill or knock is visible.
    kill_weapons: list[str] = Field(default_factory=list, max_length=6)

    player_health_state: Health = "unknown"
    mechanics: list[str] = Field(default_factory=list, max_length=6)

    # Names the field where the frames contradict the HUD context it was given.
    # Silence here is a design failure, not a success: a model forced to agree
    # with a bad OCR read produces a confident wrong answer and destroys the
    # only signal that anything went wrong.
    hud_conflict: str | None = None
    confidence: Confidence = "medium"

    # v3.1 execution fields. Optional, so v3.0 payloads still validate.
    shots_fired: int | None = None
    target_moving: str = "unknown"
    player_moving: str = "unknown"
    return_fire: str = "unknown"
    engagement_seconds: float | None = None
    execution_notes: str = ""

    # v3.2 entity-tracking fields.
    opponents_engaged: int | None = None
    changed_target: str = "unknown"
    changed_weapon: str = "unknown"

    # Opt-in attribution experiment. Old descriptions stay unknown, never self.
    viewpoint: Literal["own_gameplay", "spectating", "non_gameplay", "unknown"] = (
        "unknown"
    )
    event_actor: Literal["recording_player", "spectated_player", "other", "unknown"] = (
        "unknown"
    )
    elimination_weapon_name: str = "unknown"
    weapon_evidence_frames: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        default_factory=list, max_length=32
    )
    attribution_notes: str = Field(default="", max_length=1000)
    mechanics_evidence: str = Field(default="", max_length=1000)

    @property
    def embed_text(self) -> str:
        # What the taste model actually sees. Execution detail is appended to the
        # summary because the summary alone cannot separate two clips of one kill
        # -- within-fight accuracy is ~62% against ~81% across fights.
        return " ".join(part for part in (self.summary, self.execution_notes) if part)


class VlmResult(BaseModel):
    """A parsed description plus what the call cost."""

    description: ClipDescription
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
