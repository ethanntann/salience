from dataclasses import replace

import pytest

from pipeline_v2 import config, freeze


def _cfg():
    return config.Config(
        api_key="k",
        model="accounts/fireworks/models/test-vl",
        base_url="https://example.invalid/v1",
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
        budget_usd=1.0,
        prompt_version="v3.3",
    )


def test_fingerprint_records_the_runtime_identity():
    recorded = freeze.fingerprint(_cfg(), "system+instructions", 1440)

    assert recorded["model"] == "accounts/fireworks/models/test-vl"
    assert recorded["prompt_version"] == "v3.3"
    assert recorded["frame_width"] == 1440
    assert recorded["prompt_sha"] == freeze.sha256_text("system+instructions")


def test_matching_runtime_passes():
    recorded = freeze.fingerprint(_cfg(), "prompt", 1440)

    freeze.check(recorded, freeze.fingerprint(_cfg(), "prompt", 1440), "study")


@pytest.mark.parametrize(
    "field, changed",
    [
        ("model", {"model": "accounts/fireworks/models/other"}),
        ("base_url", {"base_url": "https://elsewhere.invalid/v1"}),
        ("prompt_version", {"prompt_version": "v3.4-attribution-dev"}),
    ],
)
def test_any_changed_config_field_is_refused_and_named(field, changed):
    recorded = freeze.fingerprint(_cfg(), "prompt", 1440)
    current = freeze.fingerprint(replace(_cfg(), **changed), "prompt", 1440)

    with pytest.raises(ValueError, match=field):
        freeze.check(recorded, current, "study")


def test_a_changed_prompt_is_refused():
    recorded = freeze.fingerprint(_cfg(), "prompt", 1440)
    current = freeze.fingerprint(_cfg(), "prompt, revised", 1440)

    with pytest.raises(ValueError, match="prompt_sha"):
        freeze.check(recorded, current, "study")


def test_a_changed_frame_width_is_refused():
    recorded = freeze.fingerprint(_cfg(), "prompt", 1440)
    current = freeze.fingerprint(_cfg(), "prompt", 960)

    with pytest.raises(ValueError, match="frame_width"):
        freeze.check(recorded, current, "study")


def test_study_specific_extras_are_compared_too():
    recorded = freeze.fingerprint(_cfg(), "prompt", 1440, clip_sha="abc")
    current = freeze.fingerprint(_cfg(), "prompt", 1440, clip_sha="def")

    with pytest.raises(ValueError, match="clip_sha"):
        freeze.check(recorded, current, "localization")


def test_descriptive_plan_fields_are_ignored():
    """A plan carries selection notes and window lists; those are not identity."""
    recorded = {
        **freeze.fingerprint(_cfg(), "prompt", 1440),
        "created_utc": "2026-09-07T00:00:00+00:00",
        "selection": "twelve equally spaced chunks",
        "windows": [{"id": "sample-00"}],
    }

    freeze.check(recorded, freeze.fingerprint(_cfg(), "prompt", 1440), "study")


def test_a_plan_written_before_a_field_existed_still_loads():
    recorded = freeze.fingerprint(_cfg(), "prompt", 1440)
    del recorded["prompt_version"]

    freeze.check(recorded, freeze.fingerprint(_cfg(), "prompt", 1440), "study")


def test_the_error_names_every_field_that_moved():
    recorded = freeze.fingerprint(_cfg(), "prompt", 1440)
    current = freeze.fingerprint(replace(_cfg(), model="other"), "revised", 960)

    with pytest.raises(ValueError) as excinfo:
        freeze.check(recorded, current, "evaluation plan")

    message = str(excinfo.value)
    assert "evaluation plan" in message
    for field in ("model", "prompt_sha", "frame_width"):
        assert field in message
