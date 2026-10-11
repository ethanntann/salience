"""What was said, not how loud it was.

Loudness cannot tell a shout of delight from a cough, and SCOPE records that
embedding transcripts *harmed* measured ranking. Both facts have the same cause:
speech was treated as one undifferentiated signal.

Measured on 334 labelled clips whose transcripts were already stored:

| speech in the clip | kept |
| --- | --- |
| strict excitement ("oh my god", "no way", "let's go") | **42.9%** (n=21) |
| no speech at all | 36.3% |
| overall baseline | 31.4% |
| any "reaction" word incl. filler ("bro", "damn", "yo") | **19.7%** |

So the direction reverses depending on *which* words. Conversational filler
tracks downtime -- you chat while rotating -- and lands well below baseline,
while genuine excitement lands well above it. Averaging the two together is
exactly what an embedding of the whole transcript does, and why that made
ranking worse.

Two honest limits. The excitement lift is **not statistically significant**
(z = 1.16, 95% CI 24.5-63.5%): 21 clips is too few. And every clip in that
corpus is an auto-saved kill, so a funny moment with no kill was never
recorded -- this data cannot say whether such moments are worth clipping, only
that among kill clips, excitement correlates with keeping.

Classification rather than embedding, and a fixed vocabulary rather than a
model, because 21 examples cannot train anything and a phrase list is
inspectable when it goes wrong.

**Status as of the 2026-09-28 control: not worth building on yet.** The loudness
detector this was designed to refine fired 44 times in a 31-minute session and
2 of those clips held a real event -- 4.5%, against 8.2% for randomly-timed
clips cut from the same footage by the same code. Spending 44 describe calls on
uniform random windows would have found more events than listening did.

So `VoiceReactionDetector` is out of the default set and the Whisper cascade
below is **not** on the path to anything measured. Two readings remain open and
this module is kept for both: the threshold was set from four marks, so the
level gate rather than speech may be what failed; and every clip in the labelled
corpus is an auto-saved kill, so a funny moment with no kill has never been in
any training or evaluation set here. Nothing measured says *what was said* is
useless -- only that *how loud it was* did not beat chance.
"""

from __future__ import annotations

from dataclasses import dataclass

# Unambiguous excitement. Deliberately narrow: every term added that doubles as
# conversational filler drags the group back toward baseline, which is the
# measured failure of the looser pattern.
EXCITEMENT = (
    "oh my god", "oh my days", "no way", "let's go", "lets go", "holy shit",
    "holy crap", "are you kidding", "what the", "insane", "clutch",
    "sheesh", "no shot", "actually cracked",
)

# Words that sound like reactions and behave like filler. Kept explicit so the
# distinction is documented rather than rediscovered.
FILLER = ("bro", "damn", "yo", "nice", "yeah", "okay", "dude")

# Verbalised disinterest -- rare but unambiguous, and a direct negative label
# from the player's own mouth ("this is a whole lot of bad video bro").
DISINTEREST = ("boring", "so bad", "nothing happening", "lame", "mid", "trash")

LAUGHTER = ("haha", "hehe", "lmao", "lol")


def _contains(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = (text or "").lower()
    return [t for t in terms if t in lowered]


@dataclass(frozen=True)
class SpeechSignal:
    """What a transcript says about the moment, as separable counts."""

    excitement: tuple[str, ...] = ()
    laughter: tuple[str, ...] = ()
    disinterest: tuple[str, ...] = ()
    filler: tuple[str, ...] = ()
    words: int = 0

    @property
    def reacted(self) -> bool:
        """Something was said that is not small talk."""
        return bool(self.excitement or self.laughter)

    @property
    def features(self) -> list[float]:
        """Separable inputs for the ranker -- never one merged 'speech' number.

        Kept as counts-normalised flags rather than a single score so that the
        excitement and filler effects, which point in opposite directions, can
        never cancel each other out inside one feature.
        """
        return [
            1.0 if self.excitement else 0.0,
            1.0 if self.laughter else 0.0,
            1.0 if self.disinterest else 0.0,
            1.0 if self.filler else 0.0,
            min(self.words, 60) / 60.0,
        ]


def classify(transcript: str) -> SpeechSignal:
    """Split a transcript into the categories that behave differently."""
    text = transcript or ""
    return SpeechSignal(
        excitement=tuple(_contains(text, EXCITEMENT)),
        laughter=tuple(_contains(text, LAUGHTER)),
        disinterest=tuple(_contains(text, DISINTEREST)),
        filler=tuple(_contains(text, FILLER)),
        words=len(text.split()),
    )


class Transcriber:
    """faster-whisper, loaded once, for the second stage of a cascade.

    Never run continuously. Whisper competes with the game for the same CPU
    that every capture measurement this month has been fighting over, and
    transcribing a whole session to find a handful of reactions is the expensive
    way round. A loudness gate decides *when* to listen; this decides *what was
    said*, over the few seconds around that moment only.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_size, device=device, compute_type="int8"
        )

    def transcribe(self, wav_path) -> str:
        segments, _ = self._model.transcribe(
            str(wav_path),
            language="en",
            vad_filter=True,  # skip silence; most of a clip is not speech
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def would_gate(mic_db: float, baseline_db: float, rise_db: float = 8.0) -> bool:
    """Should the expensive stage run for this moment?

    The cascade's whole point: cheap arithmetic on a level series decides
    whether Whisper is worth waking, so transcription cost scales with the
    number of interesting moments rather than with session length.
    """
    return (mic_db - baseline_db) >= rise_db
