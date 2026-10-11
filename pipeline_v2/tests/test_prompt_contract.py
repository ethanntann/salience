"""The prompt text and the structured vocabulary must not drift apart again.

`schema.MECHANICS` once carried three tags (editing, edit_reset, piece_control)
that the default prompt never offered, so `field_vector` reserved three
dimensions that were zero across all 1356 stored descriptions.
"""

from __future__ import annotations

import re

from pipeline_v2 import prompt, schema
from pipeline_v2.facts import ClipFacts


def _default_allowed_tags() -> tuple[str, ...]:
    body = prompt._INSTRUCTIONS.split(
        'Allowed "mechanics" values -- use these exact strings or omit the tag:'
    )[1]
    listing = body.split("\n\n")[0]
    return tuple(tag.strip() for tag in listing.replace("\n", " ").split(",") if tag.strip())


def test_default_prompt_offers_exactly_the_default_vocabulary():
    assert _default_allowed_tags() == schema.MECHANICS


def test_attribution_prompt_adds_exactly_the_build_and_edit_tags():
    line = re.search(r"Additional allowed mechanics: ([^.]+)\.", prompt._ATTRIBUTION)
    assert line is not None
    extra = tuple(tag.strip() for tag in line.group(1).split(","))

    assert set(schema.ATTRIBUTION_MECHANICS) - set(schema.MECHANICS) == set(extra)
    assert schema.mechanics_for(schema.ATTRIBUTION_PROMPT_VERSION) == (
        schema.ATTRIBUTION_MECHANICS
    )
    assert schema.mechanics_for("v3.3") == schema.MECHANICS


def test_auto_clipped_and_capture_window_prompts_get_separate_labels():
    clipped = ClipFacts(event_source="fraction")
    window = ClipFacts(event_source="capture_window")

    assert prompt.build_prompt(clipped) != prompt.build_prompt(window)
    assert prompt.storage_variant("v3.3", clipped) == "v3.3"
    assert prompt.storage_variant("v3.3", window) == "v3.3-window"


def test_the_attribution_template_needs_no_window_suffix(monkeypatch):
    monkeypatch.setenv("V2_ABLATE", "attribution")
    clipped = ClipFacts(event_source="fraction")
    window = ClipFacts(event_source="capture_window")

    # One template replaces both, so one label is correct.
    assert prompt.build_prompt(clipped) == prompt.build_prompt(window)
    version = schema.ATTRIBUTION_PROMPT_VERSION
    assert prompt.storage_variant(version, window) == version


def test_v35_changes_only_the_weapon_field_and_its_rules():
    """The whole point of this arm is that one change is attributable to it."""
    import difflib

    clipped = ClipFacts(event_source="fraction")
    before = prompt.build_prompt(clipped, "v3.3").splitlines()
    after = prompt.build_prompt(
        clipped, schema.KILL_WEAPON_PROMPT_VERSION
    ).splitlines()

    removed = [
        line
        for line in difflib.unified_diff(before, after, n=0, lineterm="")
        if line.startswith("-") and not line.startswith("---")
    ]

    # Exactly one line removed: the v3.3 weapon_name definition, replaced by the
    # kill_weapons list. Everything else is additive, so any measured difference
    # traces to that one question changing.
    assert len(removed) == 1
    assert "weapon_name" in removed[0]
    assert "as shown" in removed[0]
    assert "kill_weapons" in prompt.build_prompt(clipped, schema.KILL_WEAPON_PROMPT_VERSION)


def test_v35_follows_the_capture_window_wording():
    """It is the arm intended to run on recorder chunks, not only auto-clips."""
    window = prompt.build_prompt(
        ClipFacts(event_source="capture_window"), schema.KILL_WEAPON_PROMPT_VERSION
    )

    assert "uniformly sampled gameplay window" in window
    assert "kill_weapons" in window
    assert prompt.storage_variant(
        schema.KILL_WEAPON_PROMPT_VERSION, ClipFacts(event_source="capture_window")
    ) == "v3.5-killweapon-window"


def test_every_registered_ablation_builds_a_distinct_prompt():
    clipped = ClipFacts(event_source="fraction")
    versions = {"v3.3", *schema.ABLATIONS.values()}
    built = {v: prompt.build_prompt(clipped, v) for v in versions}

    assert len(set(built.values())) == len(versions)


def test_the_shipped_prompt_text_is_unchanged():
    """A changed byte here invalidates every cached description at this version.

    The cache key covers the exact request, so editing the prompt means paying
    again for all 348 clips. Update these digests only alongside a new version.
    """
    from pipeline_v2 import freeze

    clipped = prompt.SYSTEM + prompt.build_prompt(ClipFacts(event_source="fraction"))
    window = prompt.SYSTEM + prompt.build_prompt(
        ClipFacts(event_source="capture_window")
    )

    assert freeze.sha256_text(clipped) == (
        "9279b643ddf593dabae9b12e537a14c0f89b44c113cb5d090b4f4b2a9b110118"
    )
    assert freeze.sha256_text(window) == (
        "c6451c56c64972dda01e4c9c1f52b64a55980b2e275e5d9872a9ae200d9aeb67"
    )


def test_a_different_model_gets_its_own_storage_variant():
    """A model change must not overwrite the corpus in place."""
    from pipeline_v2 import config

    clipped = ClipFacts(event_source="fraction")
    other = "accounts/fireworks/models/some-other-vision-model"

    assert prompt.storage_variant("v3.3", clipped, config.CORPUS_MODEL) == "v3.3"
    assert prompt.storage_variant("v3.3", clipped, None) == "v3.3"
    assert (
        prompt.storage_variant("v3.3", clipped, other)
        == "v3.3@some-other-vision-model"
    )


def test_the_model_tag_composes_with_the_capture_window_suffix():
    from pipeline_v2 import config

    window = ClipFacts(event_source="capture_window")
    other = "accounts/fireworks/models/some-other-vision-model"

    assert prompt.storage_variant("v3.3", window, config.CORPUS_MODEL) == "v3.3-window"
    assert (
        prompt.storage_variant("v3.3", window, other)
        == "v3.3-window@some-other-vision-model"
    )


def test_reads_and_writes_agree_on_the_variant_for_any_model():
    """A write that reads back as nothing is the bug this guards against."""
    from dataclasses import replace

    from pipeline_v2 import config

    clipped = ClipFacts(event_source="fraction")
    base = config.Config(
        api_key="", model=config.CORPUS_MODEL, base_url="u",
        input_price_per_mtok=0, output_price_per_mtok=0, budget_usd=1,
        prompt_version="v3.3",
    )

    # Corpus model: both sides plain.
    assert prompt.active_variant(base) == "v3.3"
    assert prompt.storage_variant(base.prompt_version, clipped, base.model) == "v3.3"

    # Any other model: both sides carry the same tag.
    other = replace(base, model="accounts/fireworks/models/kimi-k2p6")
    assert prompt.active_variant(other) == "v3.3@kimi-k2p6"
    assert (
        prompt.storage_variant(other.prompt_version, clipped, other.model)
        == "v3.3@kimi-k2p6"
    )

    arm = replace(other, prompt_version=schema.KILL_WEAPON_PROMPT_VERSION)
    assert prompt.active_variant(arm) == "v3.5-killweapon@kimi-k2p6"
    assert (
        prompt.storage_variant(arm.prompt_version, clipped, arm.model)
        == "v3.5-killweapon@kimi-k2p6"
    )


def test_v36_scopes_the_kill_list_to_the_recording_player():
    clipped = ClipFacts(event_source="fraction")
    v35 = prompt.build_prompt(clipped, schema.KILL_WEAPON_PROMPT_VERSION)
    v36 = prompt.build_prompt(clipped, schema.OWN_KILLS_PROMPT_VERSION)

    assert v35 != v36
    assert "THE RECORDING PLAYER" in v36
    # The invariant the report checks without labels.
    assert '"death"' in v36 and "kill_weapons must be []" in v36
    # v3.5 counted the player's own death as a kill; v3.6 must not.
    assert "the recording player being eliminated" in v35
    assert "they killed nobody" in v36


def test_both_kill_list_arms_are_read_as_lists():
    from pipeline_v2 import groundtruth

    for version in (
        schema.KILL_WEAPON_PROMPT_VERSION,
        schema.OWN_KILLS_PROMPT_VERSION,
    ):
        assert groundtruth.asks_for_kill_weapons(version)
        assert groundtruth.asks_for_kill_weapons(f"{version}@kimi-k2p6")
    assert not groundtruth.asks_for_kill_weapons("v3.3")
