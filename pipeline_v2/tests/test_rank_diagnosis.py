"""Telling apart the two reasons ranking can come back empty.

One line used to cover both: a variant nothing was ever written to, and a
populated variant with nothing labelled yet. They need opposite fixes -- pass a
different `--variant`, versus go and label some clips -- and the silent version
of this already cost an afternoon once, when relinked paths matched 0 of 348
labels and ranking just said "not enough labelled clips".

Capture-window clips make it routine rather than rare: writes tag them with a
`-window` suffix that reads deliberately omit, so describing a whole session and
then running `rank` lands in the empty-variant case every time.
"""

from __future__ import annotations

import argparse

import pytest

from pipeline_v2 import cli


@pytest.fixture
def rank_args():
    return argparse.Namespace(db=None, variant=None, mode="embedding", top=10)


def _no_fights(monkeypatch, labels=0):
    monkeypatch.setattr(
        "pipeline_v2.ranking.ranked_fights", lambda *a, **k: ([], labels)
    )


def test_an_empty_variant_lists_the_variants_that_exist(
    monkeypatch, capsys, rank_args, isolated_data
):
    _no_fights(monkeypatch)
    rank_args.variant = "v9.9-typo@nonesuch"
    monkeypatch.setattr(
        cli.store, "connect", lambda: _FakeConn({"v3.6-ownkills-window@minimax-m3": 146})
    )

    assert cli.cmd_rank(rank_args) == 1
    out = capsys.readouterr().out
    assert "no clips described in variant" in out
    assert "v3.6-ownkills-window@minimax-m3" in out
    assert "146" in out
    assert "--variant" in out  # the message says what to do about it


def test_a_populated_variant_with_no_labels_says_so_instead(
    monkeypatch, capsys, rank_args, isolated_data
):
    _no_fights(monkeypatch, labels=0)
    rank_args.variant = "v3.6-ownkills-window@minimax-m3"
    monkeypatch.setattr(
        cli.store, "connect", lambda: _FakeConn({"v3.6-ownkills-window@minimax-m3": 146})
    )

    assert cli.cmd_rank(rank_args) == 1
    out = capsys.readouterr().out
    assert "146 clip(s) described" in out
    assert "0 are labelled" in out
    assert "no clips described in variant" not in out  # the other diagnosis


def test_the_window_suffix_mismatch_is_reported_not_silent(
    monkeypatch, capsys, rank_args, isolated_data
):
    """The routine case: describe a session, run rank, get nothing.

    Reads resolve to the bare variant while capture windows were written with a
    `-window` suffix, so the near-match has to be surfaced or the user is left
    guessing at a spelling.
    """
    _no_fights(monkeypatch)
    rank_args.variant = "v3.6-ownkills@minimax-m3"
    monkeypatch.setattr(
        cli.store,
        "connect",
        lambda: _FakeConn({
            "v3.6-ownkills-window@minimax-m3": 146,
            "v2.3-nofacts": 189,
        }),
    )

    cli.cmd_rank(rank_args)
    out = capsys.readouterr().out

    assert "v3.6-ownkills-window@minimax-m3" in out
    # The unrelated variant must not crowd out the near-match.
    assert out.index("v3.6-ownkills-window@minimax-m3") < (
        out.index("v2.3-nofacts") if "v2.3-nofacts" in out else len(out)
    )


class _FakeConn:
    """Just enough connection to answer the variant census and close."""

    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def execute(self, _sql, *_args):
        return iter(self._counts.items())

    def close(self):
        pass
