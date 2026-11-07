"""The app's local server answers only its own window (security audit, 2026-11-07)."""

from __future__ import annotations

import http.client
import json

import pytest

from pipeline_v2.capture.desktop import server


@pytest.fixture
def running():
    # The handler alone: `server.start` also points the rating page's module
    # at the real recordings, which other tests would then see.
    import threading
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _ask(port, method, path, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
    status = conn.getresponse().status
    conn.close()
    return status


def test_other_websites_are_refused(running):
    body = json.dumps({"key": "mode", "value": "dark"})
    ours = {"X-Salience": server.Handler.secret, "Content-Type": "application/json"}
    # A page elsewhere, posting without the secret (text/plain needs no preflight).
    assert _ask(running, "POST", "/api/appearance", {"Origin": "https://evil.example"}, body) == 403
    assert _ask(running, "POST", "/api/appearance", {"Content-Type": "text/plain"}, body) == 403
    # DNS rebinding: the right secret cannot help a request to another name.
    assert _ask(running, "GET", "/api/settings", {**ours, "Host": f"evil.example:{running}"}) == 403
    assert _ask(running, "GET", "/api/settings") == 403


def test_the_apps_own_page_gets_in(running):
    ours = {"X-Salience": server.Handler.secret, "Content-Type": "application/json",
            "Origin": f"http://127.0.0.1:{running}"}
    assert _ask(running, "GET", "/") == 200
    assert _ask(running, "GET", "/api/settings", {"X-Salience": server.Handler.secret}) == 200
    assert _ask(running, "POST", "/api/appearance", ours, json.dumps({"key": "nope", "value": 1})) == 400


def test_the_page_is_given_the_secret(running):
    conn = http.client.HTTPConnection("127.0.0.1", running, timeout=10)
    conn.request("GET", "/")
    page = conn.getresponse().read().decode()
    conn.close()
    assert f'window.SALIENCE_SECRET = "{server.Handler.secret}"' in page
