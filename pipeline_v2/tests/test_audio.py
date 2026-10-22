"""Parakeet's tokens into utterances, across the overlapping pieces."""

import wave

import numpy as np

from pipeline_v2 import audio


def test_tokens_join_into_words_with_times_and_offset():
    words = audio.words_of([" No", ",", " what", " the", " fu", "ck"],
                           [9.76, 10.4, 10.72, 10.96, 11.2, 11.36], offset=100.0)

    assert [w.word for w in words] == [" No,", " what", " the", " fuck"]
    assert words[0].start == 109.76 and words[0].end == words[1].start
    assert words[-1].end == words[-1].start + audio.LAST_WORD_SEC


def test_pieces_cover_everything_and_own_each_second_once():
    found = audio.pieces(40.0)

    assert found[0][1] == 0.0 and found[-1][2] == 40.0
    assert all(a[2] == b[1] for a, b in zip(found, found[1:]))  # no gap, no double
    assert all(start + audio.PIECE_SEC >= until for start, _, until in found)


def test_a_word_in_the_overlap_is_heard_once(tmp_path):
    """Both pieces hear the word at 13.5s; only the piece owning it keeps it."""
    path = tmp_path / "mic.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(16000 * 20, dtype=np.int16).tobytes())

    class Model:
        def recognize(self, piece, sample_rate):
            start = 0.0 if not hasattr(self, "seen") else audio.PIECE_SEC - audio.OVERLAP_SEC
            self.seen = True
            at = 13.5 - start  # the same moment, in each piece's own time
            from types import SimpleNamespace

            return SimpleNamespace(tokens=[" clip", " that"], timestamps=[at, at + 0.3])

    transcriber = audio.Transcriber.__new__(audio.Transcriber)
    transcriber._model = Model()

    [(start, end, text)] = transcriber.timed(path)
    assert (round(start, 3), round(end, 3), text) == (13.5, 14.1, "clip that")
