"""The mouse's own movement counts, per window: distance and the fastest 50ms."""

from __future__ import annotations

import struct

from pipeline_v2.capture import mouse


def _record(kind: int, flags: int, dx: int, dy: int) -> bytes:
    body = mouse._MOUSE.pack(flags, 0, 0, dx, dy, 0)
    size = mouse._HEADER + len(body)
    header = struct.pack("<II", kind, size) + bytes(mouse._HEADER - 8)
    record = header + body
    return record + bytes(-len(record) % 8)


def test_a_batch_reads_every_relative_mouse_record():
    buffer = (_record(0, 0, 5, -3) + _record(1, 0, 99, 99)  # a keyboard record
              + _record(0, mouse._MOUSE_MOVE_ABSOLUTE, 30000, 30000) + _record(0, 0, -7, 2))

    assert mouse.motions(buffer, 4) == [(5, -3), (-7, 2)]


def test_a_flick_peaks_where_tracking_the_same_distance_does_not(tmp_path):
    m = mouse.MouseMotion(journal=tmp_path / "mouse.tsv", started=0.0)
    for i in range(5):  # 300 counts within 50ms
        m.add(0.105 + i * 0.01, 60)
    for i in range(50):  # 300 counts over the next window
        m.add(0.505 + i * 0.01, 6)
    m.add(1.2, 0)

    flick, track = m.samples[:2]
    assert flick.distance == track.distance == 300
    assert flick.peak == 300 and track.peak < flick.peak / 5
    assert mouse.MouseMotion.read_journal(tmp_path / "mouse.tsv") == m.samples


def test_still_windows_are_still_written():
    m = mouse.MouseMotion(started=0.0)

    m.add(2.0, 0)

    assert [s.at for s in m.samples] == [0.5, 1.0, 1.5, 2.0]
    assert all(s.distance == 0 for s in m.samples)
