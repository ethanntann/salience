"""The audio channel: what was said, and how loudly people reacted.

The visual pipeline is deaf, which makes an entire category of clip invisible to
it. A moment that is funny because of what someone shouted has, so far, been
ranked on whatever happened on screen -- often nothing impressive. That is a
plausible reason the learned taste came out as pure long-range headshots: the
model learned the pattern it could see.

Two levels, both local and free of API cost:

1. **Reaction features** -- loudness envelope over the clip, and how much of it
   is voice. Cheap, and enough to say "people were shouting here".
2. **Transcription** -- Parakeet (`Transcriber`). This is the one that fits the existing
   architecture exactly: the transcript is another string joined into
   `embed_text`, ranked by the same taste model, no new model type required.

The mic is mixed into the same stereo track as game audio in these files, so
speech competes with gunfire. A speech model tolerates that better than a
laughter classifier would, which is part of why transcription is the primary
signal. (The recorder now keeps the mic on its own track.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

_SAMPLE_RATE = 16000  # what the speech model wants; resampling here avoids doing it twice


@dataclass(frozen=True)
class AudioFeatures:
    """Mirrors the columns of the `audio_features` table `taste` reads."""

    transcript: str = ""
    peak_ratio: float = 0.0  # loudest moment vs clip median, a shout detector
    loud_fraction: float = 0.0  # share of the clip well above its own baseline


def extract_wav(clip: Path, out_dir: Path | None = None) -> Path | None:
    """Mono 16kHz wav. Returns None when the clip carries no audio stream."""
    directory = out_dir or Path(tempfile.mkdtemp(prefix="salience-audio-"))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{clip.stem}.wav"
    if target.exists():
        return target
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(clip),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            "-f",
            "wav",
            str(target),
        ],
        capture_output=True,
    )
    if completed.returncode != 0 or not target.exists():
        return None
    return target


def reaction_features(wav: Path) -> tuple[float, float]:
    """(peak_ratio, loud_fraction) from the loudness envelope.

    Measured against the clip's *own* median rather than an absolute threshold,
    because recording levels vary between sessions and an absolute cutoff would
    just rank loud captures above quiet ones.
    """
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(str(wav), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if not len(samples):
        return 0.0, 0.0

    window = max(1, rate // 20)  # 50ms
    usable = len(samples) - (len(samples) % window)
    if usable < window:
        return 0.0, 0.0
    envelope = np.sqrt((samples[:usable].reshape(-1, window) ** 2).mean(axis=1))
    median = float(np.median(envelope)) or 1e-6
    peak_ratio = float(envelope.max() / median)
    loud_fraction = float((envelope > median * 3.0).mean())
    return peak_ratio, loud_fraction


# The speech model: NVIDIA Parakeet TDT 0.6B v2, int8 ONNX (631 MB, CC-BY-4.0),
# on the CPU through onnx-asr. It replaced faster-whisper `small` behind a
# voice detector (2026-10-22). The detector was what lost the player's words:
# on sessions 5 and 6 it passed 0 and 8 seconds of 23 and 38 minutes, and
# dropped "Oh, my God." 3s before session 5's one kept kill, along with "Dead.",
# "Easy." and "Bruh.". Without a detector Whisper invents sign-offs from mic
# hiss ("Thanks for watching!": small 29 times over the two sessions,
# large-v3-turbo ~85 in session 5 alone); Parakeet over every second of both
# invented nothing, heard every line Whisper did, and took 4.3 minutes on all
# cores against Whisper small's 6.
MODEL = "nemo-parakeet-tdt-0.6b-v2"
QUANTIZATION = "int8"
# Audio goes through in pieces this long, overlapping so a word cut at one
# piece's edge is whole in the next. Each word is kept from the piece whose
# own half of the overlap it starts in.
PIECE_SEC = 15.0
OVERLAP_SEC = 2.0
# A word's end is the next token's start; the last one in a piece gets this.
LAST_WORD_SEC = 0.3


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    word: str  # with its leading space, as `utterances` joins them


def read_wav(wav: Path):
    """16-bit PCM wav (what ffmpeg writes here) as float32 samples and rate."""
    import wave

    import numpy as np

    with wave.open(str(wav)) as handle:
        rate, channels = handle.getframerate(), handle.getnchannels()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    samples = samples.astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def words_of(tokens: list[str], times: list[float], offset: float) -> list[Word]:
    """Parakeet's tokens as words: a token starting with a space starts one."""
    found: list[list] = []
    for token, at in zip(tokens, times):
        if token.startswith(" ") or not found:
            found.append([offset + at, token])
        else:
            found[-1][1] += token
    starts = [w[0] for w in found]
    ends = starts[1:] + [starts[-1] + LAST_WORD_SEC] if found else []
    return [Word(start, end, text if text.startswith(" ") else " " + text)
            for (start, text), end in zip(found, ends) if text.strip()]


def pieces(length: float) -> list[tuple[float, float, float]]:
    """(piece start, own from, own until) covering [0, length)."""
    step = PIECE_SEC - OVERLAP_SEC
    found, at = [], 0.0
    while True:
        last = at + PIECE_SEC >= length
        own_from = at + OVERLAP_SEC / 2 if at else 0.0
        found.append((at, own_from, length if last else at + step + OVERLAP_SEC / 2))
        if last:
            return found
        at += step


class Transcriber:
    """Parakeet, loaded once, over all of the audio: no voice detector."""

    def __init__(self, cpu_threads: int = 0) -> None:
        import onnx_asr
        import onnxruntime

        options = onnxruntime.SessionOptions()
        # 0 lets onnxruntime take every core: right after a session, wrong mid-match.
        options.intra_op_num_threads = cpu_threads
        self._model = onnx_asr.load_model(
            MODEL, quantization=QUANTIZATION, sess_options=options).with_timestamps()

    def transcribe(self, wav: Path) -> str:
        return " ".join(text for _, _, text in self.timed(wav))

    def timed(self, wav: Path) -> list[tuple[float, float, str]]:
        """(start, end, text) per utterance, seconds from the start of the file.

        Utterances are cut from word times at `WORD_GAP` pauses, never taken a
        piece at a time: in session 4 (2026-10-19) "But it's" (922s) and
        "walking retard though" (1034s) once came back as one 114-second line,
        said over every window in that stretch.
        """
        samples, rate = read_wav(wav)
        words: list[Word] = []
        for start, own_from, own_until in pieces(len(samples) / rate):
            piece = samples[int(start * rate):int((start + PIECE_SEC) * rate)]
            if len(piece) < rate // 10:
                continue
            result = self._model.recognize(piece, sample_rate=rate)
            words += [w for w in words_of(result.tokens or [], result.timestamps or [], start)
                      if own_from <= w.start < own_until]
        return utterances(words) if words else []


# Seconds of silence between two words that make them separate utterances.
WORD_GAP = 1.0


def utterances(words) -> list[tuple[float, float, str]]:
    """Words, as (start, end, text) runs split at `WORD_GAP` pauses."""
    runs = [[words[0]]]
    for word in words[1:]:
        if word.start - runs[-1][-1].end > WORD_GAP:
            runs.append([word])
        else:
            runs[-1].append(word)
    return [(float(run[0].start), float(run[-1].end), "".join(w.word for w in run).strip())
            for run in runs]


def analyse(
    clip: Path, transcriber: Transcriber | None, out_dir: Path
) -> AudioFeatures:
    wav = extract_wav(clip, out_dir)
    if wav is None:
        return AudioFeatures()
    peak_ratio, loud_fraction = reaction_features(wav)
    transcript = transcriber.transcribe(wav) if transcriber is not None else ""
    return AudioFeatures(
        transcript=transcript,
        peak_ratio=peak_ratio,
        loud_fraction=loud_fraction,
    )
