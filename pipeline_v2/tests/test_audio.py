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
        # Not silence: talk-level noise throughout (silence is never heard).
        handle.writeframes(np.random.default_rng(0).integers(-600, 600, 16000 * 20, dtype=np.int16).tobytes())

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


def test_silence_is_never_put_to_the_model(tmp_path):
    """A mic behind noise suppression is digital silence between words; the
    model ran over all of it (2:11 of a 4-minute wait, 2026-11-01)."""
    path = tmp_path / "mic.wav"
    quiet = np.zeros(16000 * 40, dtype=np.int16)
    quiet[16000 * 30:16000 * 31] = np.random.default_rng(0).integers(-3000, 3000, 16000, dtype=np.int16)  # a word at 30s
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(quiet.tobytes())
    heard: list[int] = []

    class Model:
        def recognize(self, piece, sample_rate):
            from types import SimpleNamespace

            heard.append(len(piece))
            return SimpleNamespace(tokens=[], timestamps=[])

    transcriber = audio.Transcriber.__new__(audio.Transcriber)
    transcriber._model = Model()
    transcriber.timed(path)

    pieces = audio.pieces(40.0)
    with_word = [p for p in pieces if p[0] <= 30.0 < p[0] + audio.PIECE_SEC or p[0] < 31.0 <= p[0] + audio.PIECE_SEC]
    assert 0 < len(heard) == len(with_word) < len(pieces)


def test_a_long_run_of_talk_is_split_after_its_sentences():
    """One 16.6s run put a kill's words on the window before the kill (2026-11-04)."""
    words = [audio.Word(0.0 + i * 0.6, 0.5 + i * 0.6, w) for i, w in enumerate(
        " He's just having a good time, bro. I just killed him, bro. I'm so sorry.".split(" ")[1:])]
    words = [audio.Word(w.start, w.end, " " + w.word) for w in words]
    runs = audio.utterances(words)
    assert [r[2] for r in runs] == ["He's just having a good time, bro.", "I just killed him, bro. I'm so sorry."]
    assert runs[1][0] == words[7].start
    short = [audio.Word(0.0, 0.3, " Clip."), audio.Word(0.4, 0.7, " that.")]
    assert [r[2] for r in audio.utterances(short)] == ["Clip. that."]  # under 3s: never split
