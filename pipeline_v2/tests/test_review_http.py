import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pipeline_v2 import review
from conftest import add_clip


@pytest.fixture
def server(conn, tmp_path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"0123456789")
    add_clip(conn, media)
    conn.commit()
    http = ThreadingHTTPServer(("127.0.0.1", 0), review.Handler)
    thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)


def test_review_decision_persists_and_advances(server, conn):
    request = urllib.request.Request(
        server + "/api/rate",
        data=json.dumps({"sha": "one", "label": 1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
    assert conn.execute("select label from preferences").fetchone()[0] == 1
    with urllib.request.urlopen(server + "/api/next", timeout=5) as response:
        assert json.load(response)["done"]


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"sha": "one", "label": 2}, 400),
        ({"sha": "one", "label": True}, 400),
        ({"sha": "missing", "label": 1}, 404),
    ],
)
def test_bad_decisions_do_not_poison_training(server, conn, payload, status):
    request = urllib.request.Request(
        server + "/api/rate", data=json.dumps(payload).encode()
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == status
    error.value.close()
    assert conn.execute("select count(*) from preferences").fetchone()[0] == 0


@pytest.mark.parametrize(
    "range_header,body",
    [("bytes=2-4", b"234"), ("bytes=-3", b"789"), ("bytes=8-999", b"89")],
)
def test_browser_seek_ranges_have_correct_length(server, range_header, body):
    request = urllib.request.Request(
        server + "/media/one", headers={"Range": range_header}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 206
        assert response.read() == body


def test_invalid_seek_is_rejected_without_hanging(server):
    request = urllib.request.Request(
        server + "/media/one", headers={"Range": "bytes=50-100"}
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 416
    error.value.close()


def _rate(server, payload):
    request = urllib.request.Request(
        server + "/api/rate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200


def test_a_wrong_description_is_flagged_with_the_rating(server, conn):
    """The accuracy test set: the ranker reads only description text."""
    _rate(server, {"sha": "one", "label": 0, "wrong": True})

    assert [r[0] for r in conn.execute("select clip_sha from description_flags")] == ["one"]
    assert conn.execute("select label from preferences").fetchone()[0] == 0


def test_rerating_without_the_flag_clears_it(server, conn):
    _rate(server, {"sha": "one", "label": 1, "wrong": True})
    _rate(server, {"sha": "one", "label": 1})

    assert conn.execute("select count(*) from description_flags").fetchone()[0] == 0
