"""The app's side of the relay: invite codes, busy waits, and a used-up month."""

from __future__ import annotations

import io
import json
import time
import urllib.error
from dataclasses import replace

import pytest

from pipeline_v2 import config, vlm
from pipeline_v2.capture import app


@pytest.fixture
def settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.env"
    monkeypatch.setattr(app, "SETTINGS", path)
    original = app.write_setting
    monkeypatch.setattr(app, "write_setting", lambda key, value, p=path: original(key, value, p))
    for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "FIREWORKS_MODEL", app.RELAY_URL_KEY):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(app.RELAY_URL_KEY, "https://relay.example")
    return path


def test_an_invite_points_the_app_at_the_relay(settings):
    def call(path, body=None, token=None, timeout=15.0):
        assert (path, body) == ("/v1/register", {"invite": "BETA1"})
        return 200, {"token": "slc_abc", "base_url": "https://relay.example/v1",
                     "model": "accounts/fireworks/models/deepseek-v4p1-flash"}

    assert app.redeem_invite("BETA1", call) is None
    saved = app.read_settings(settings)
    assert saved["FIREWORKS_API_KEY"] == "slc_abc"
    assert saved["FIREWORKS_BASE_URL"] == "https://relay.example/v1"
    assert app.uses_relay()


def test_a_bad_invite_says_why(settings):
    def call(path, body=None, token=None, timeout=15.0):
        return 403, {"error": {"code": "invite_invalid", "message": "That invite code isn't valid."}}

    assert app.redeem_invite("NOPE", call) == "That invite code isn't valid."
    assert "FIREWORKS_API_KEY" not in app.read_settings(settings)


def test_first_run_takes_a_key_or_an_invite(settings):
    redeemed, checked = [], []
    asked = iter(["BETA1"])
    assert app.setup_key(ask=lambda m: next(asked, None), check=lambda k: checked.append(k),
                         redeem=lambda c: redeemed.append(c))
    assert (redeemed, checked) == (["BETA1"], [])
    asked = iter(["fw_ownkey"])
    assert app.setup_key(ask=lambda m: next(asked, None), check=lambda k: checked.append(k),
                         redeem=lambda c: redeemed.append(c))
    assert checked == ["fw_ownkey"]


def _http_error(code, body, headers=None):
    return urllib.error.HTTPError("https://relay.example/v1/chat/completions", code, "x",
                                  headers or {}, io.BytesIO(json.dumps(body).encode()))


def test_a_used_up_month_stops_describing_and_is_remembered(monkeypatch):
    vlm.allowance_reset()

    def refuse(request, timeout):
        raise _http_error(402, {"error": {"code": "allowance_used", "message": "Used up.",
                                          "resets_at": "2026-12-01T00:00:00.000Z"}})

    monkeypatch.setattr(vlm.urllib.request, "urlopen", refuse)
    cfg = replace(config.load_config(), api_key="slc_abc", base_url="https://relay.example/v1")
    with pytest.raises(vlm.AllowanceUsed) as raised:
        vlm._post(cfg, [{"type": "text", "text": "x"}])
    assert raised.value.resets_at == "2026-12-01T00:00:00.000Z"
    assert isinstance(raised.value, vlm.BudgetExceeded)  # `index` stops the batch on it
    assert vlm.LAST_ALLOWANCE is raised.value
    vlm.allowance_reset()


def test_busy_waits_as_long_as_asked(monkeypatch):
    waits = []
    answers = iter([_http_error(429, {}, {"retry-after": "7"}), None])

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices": []}'

    def urlopen(request, timeout):
        error = next(answers)
        if error:
            raise error
        return Reply()

    monkeypatch.setattr(vlm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vlm.time, "sleep", waits.append)
    cfg = replace(config.load_config(), api_key="k")
    assert vlm._post(cfg, [{"type": "text", "text": "x"}]) == {"choices": []}
    assert waits == [7.0]
    assert vlm._retry_wait(None, 0) == 3 and vlm._retry_wait("500", 0) == 60.0


def test_a_waiting_session_comes_back_when_the_month_resets(tmp_path):
    session = tmp_path / "20261106-024025"
    session.mkdir()
    (session / "segments.csv").write_text("part-00000.mkv,0,30\n", encoding="utf-8")
    app.wait_for_allowance(session, "2026-12-01T00:00:00Z")
    resets = time.mktime(time.strptime("2026-12-01", "%Y-%m-%d"))

    assert app.unfinished(tmp_path, now=resets - 7 * 86400) == []
    # Weeks later than the usual two days, and still made.
    assert app.unfinished(tmp_path, now=resets + 2 * 86400) == [session]
