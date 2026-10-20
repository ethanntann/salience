"""The audio channel: what was said, and how loudly people reacted.

The visual pipeline is deaf, which makes an entire category of clip invisible to
it. A moment that is funny because of what someone shouted has, so far, been
ranked on whatever happened on screen -- often nothing impressive. That is a
plausible reason the learned taste came out as pure long-range headshots: the
model learned the pattern it could see.

Two levels, both local and free of API cost:

1. **Reaction features** -- loudness envelope over the clip, and how much of it
   is voice. Cheap, and enough to say "people were shouting here".
2. **Transcription** -- Whisper. This is the one that fits the existing
   architecture exactly: the transcript is another string joined into
   `embed_text`, ranked by the same taste model, no new model type required.

The mic is mixed into the same stereo track as game audio in these files, so
speech competes with gunfire. Whisper tolerates that better than a laughter
classifier would, which is part of why transcription is the primary signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

_SAMPLE_RATE = 16000  # what Whisper wants; resampling here avoids doing it twice


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


VAD_THRESHOLD = 0.3


class Transcriber:
    """Whisper, loaded once. `small` trades some accuracy for speed on CPU."""

    def __init__(self, model_size: str = "small", cpu_threads: int = 0) -> None:
        from faster_whisper import WhisperModel

        # 0 lets CTranslate2 take every core: right after a session, wrong mid-match.
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8",
                                   cpu_threads=cpu_threads)

    def transcribe(self, wav: Path) -> str:
        segments, _ = self._model.transcribe(
            str(wav),
            language="en",
            vad_filter=True,  # drop silence and most non-speech
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def timed(self, wav: Path) -> list[tuple[float, float, str]]:
        """(start, end, text) per utterance, seconds from the start of the file.

        Segments Whisper itself thinks are not speech are dropped: on a quiet
        mic it otherwise invents stock phrases ("Thank you.") from breathing.

        The voice detector runs at 0.3 rather than its 0.5 default, on audio
        `voice` has already evened out: at the default, a mic recorded around
        -40dB lost almost every callout (3 utterances in a 51-minute session;
        "Why are you hiding in the building bro?" came out "the building bro").

        Utterances are cut from word times, not taken as Whisper's segments.
        The voice detector splices the speech together before Whisper hears
        it, and a segment straddling a splice is mapped back as one span over
        everything between: in session 4 (2026-10-19) "But it's" (922s) and
        "walking retard though" (1034s) came back as one 114-second line, said
        over every window in that stretch. A pause of `WORD_GAP` starts a new
        utterance.
        """
        segments, _ = self._model.transcribe(
            str(wav),
            language="en",
            vad_filter=True,
            vad_parameters={"threshold": VAD_THRESHOLD},
            beam_size=1,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        heard = []
        for segment in segments:
            if not segment.text.strip() or segment.no_speech_prob >= 0.6:
                continue
            words = [w for w in segment.words or [] if w.word.strip()]
            if not words:
                heard.append((float(segment.start), float(segment.end), segment.text.strip()))
                continue
            heard.extend(utterances(words))
        return heard


# Seconds of silence between two words that make them separate utterances.
WORD_GAP = 1.0


def utterances(words) -> list[tuple[float, float, str]]:
    """Whisper's words, as (start, end, text) runs split at `WORD_GAP` pauses."""
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
