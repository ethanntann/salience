from contextlib import closing

import pytest

from pipeline_v2 import config, store


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Redirect the data directory for EVERY test, requested or not.

    This used to be opt-in, and a test that forgot to ask for it wrote 60 rows
    straight into the real store in `pipeline_v2/.data/`. Tests must never be
    able to reach it, so isolation is the default and cannot be forgotten.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "FRAME_DIR", tmp_path / "frames")
    monkeypatch.setattr(config, "CACHE_DB", tmp_path / "cache.sqlite")
    monkeypatch.delenv("V2_ABLATE", raising=False)
    # Tests must not inherit the developer's .env. The model decides the storage
    # variant, so a real FIREWORKS_MODEL silently retags every row a test writes.
    monkeypatch.setenv("FIREWORKS_MODEL", config.CORPUS_MODEL)
    for name in (
        "V2_FRAME_WIDTH",
        "V2_BUDGET_USD",
        "V2_REASONING_EFFORT",
        "SALIENCE_CAPTURE_ENCODER",
        "SALIENCE_CAPTURE_FPS",
        "SALIENCE_CAPTURE_BITRATE",
    ):
        monkeypatch.delenv(name, raising=False)
    # Module-level capture state must not leak between tests: the WGC
    # one-session-per-process guard would make the second test to touch it fail.
    try:
        from pipeline_v2.capture import wgc

        monkeypatch.setattr(wgc, "_SESSION_STARTED", False)
    except ImportError:  # pragma: no cover - capture extra not installed
        pass
    return tmp_path


@pytest.fixture
def conn(isolated_data):
    with closing(store.connect()) as connection:
        yield connection


def add_clip(conn, path, sha="one", variant=config.DEFAULT_PROMPT_VERSION):
    store.upsert_clip(
        conn,
        clip_sha=sha,
        path=path,
        engagement=None,
        duration_sec=20,
        event_sec=15,
        event_source="fraction",
        facts={},
        summary=f"Player lands a rifle headshot in clip {sha}.",
        payload={},
        prompt_version=variant,
    )
