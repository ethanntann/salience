"""Laughter, on the player's mic and on everything their PC plays.

A funny moment has no kill to find it by, and loudness does not find it
either: each session's three loudest quiet windows were the bus, deaths and a
video playing nearby, 0 of 18 kept (2026-10-24). Laughter does. Two tracks:

- the mic: the player laughing;
- the game track: Windows' loopback of everything the PC plays, so friends
  laughing on Discord, any other call, in-game voice chat and proximity chat
  are all in it with no setup, whatever the player uses to talk.

The model is Zencastr's laughter detector (MIT, on EAT, Hugging Face
`zencastr/laughter-detection`), run with onnxruntime. It is quantized to
int8 on first use: 2.6x faster on the CPU (2 minutes of audio in 4.1s on
two threads) for frame probabilities within 0.013 on average. Its own
windowing and decoding are re-implemented here (MIT); none of the
downloaded Python is run.

Calibration on the player's own sessions (2026-10-24): at the model's
`max_f1` operating point, 4.5 hours of game audio produced no laughter at
all and the mic one short event; the permissive `high_recall` point fired
9-39 times a session on game sounds. So `max_f1` it is: what it hears is
very likely laughter. Whether it catches laughter through game audio is
untested (the sessions had none).

Cached per raw segment in `reactions.json`, like `voice`'s transcripts.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pipeline_v2.capture import clips as clips_mod

CACHE_NAME = "reactions.json"
REPO = "zencastr/laughter-detection"
SOURCE = f"{REPO}/int8/max_f1"
# `wgc` maps the game loopback first and the microphone second.
TRACKS = {"game": 0, "mic": 1}
SAMPLE_RATE = 16000
WINDOW_S, HOP_S = 4.0, 2.0
BATCH = 16
# Laughter that has just happened belongs to the moment before it, the way a
# reaction to a kill comes after the kill.
AFTER_S = 4.0
CHUNK = 20  # segments a call, as `voice.CHUNK`


@dataclass(frozen=True)
class Laugh:
    start: float  # session time
    end: float
    score: float
    track: str


class Detector:
    """The model, loaded once (int8, made on first use and kept)."""

    def __init__(self, cpu_threads: int = 0, models: Path | None = None) -> None:
        import onnxruntime
        from huggingface_hub import hf_hub_download

        config = json.loads(Path(hf_hub_download(REPO, "config.json")).read_text(encoding="utf-8"))
        self.point = config["operating_points"]["max_f1"]
        self.frame_s = config["patch_size"] * config["mel_frame_shift_s"]
        self.frame_samples = config["patch_size"] * int(SAMPLE_RATE * config["mel_frame_shift_s"])
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = onnxruntime.InferenceSession(
            str(_int8_model(models)), options, providers=["CPUExecutionProvider"])

    def laughs(self, samples, rate: int) -> list[tuple[float, float, float]]:
        """(start, end, score) of each laugh, seconds from the start of `samples`."""
        import numpy as np

        if rate != SAMPLE_RATE or len(samples) == 0:
            return []
        windows, valid, starts = window_audio(samples.astype(np.float32), rate, WINDOW_S, HOP_S)
        probs = np.concatenate([
            self._session.run(None, {"waveform": windows[i:i + BATCH],
                                     "n_valid": valid[i:i + BATCH].astype(np.int64)})[1]
            for i in range(0, len(windows), BATCH)])
        counts = (np.arange(probs.shape[1]) * self.frame_samples)[None] < valid[:, None]
        probs = np.where(counts, probs, np.nan)
        duration = len(samples) / rate
        return decode(stitch(probs, starts, duration, self.frame_s), self.point, self.frame_s, duration)


def _int8_model(models: Path | None = None) -> Path:
    """The detector quantized to int8, made once from the published float model."""
    from huggingface_hub import hf_hub_download

    from pipeline_v2.capture import app

    target = (models or app.APP_DIR / "models") / "laughter-int8.onnx"
    if target.is_file():
        return target
    source = hf_hub_download(REPO, "model.onnx")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        return Path(source)  # without `onnx`, the float model: slower, same answers
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.onnx")
    quantize_dynamic(source, str(partial), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    partial.replace(target)
    return target


# Windowing, stitching and decoding, as the model's laughter_core.py (MIT)
# does them: the operating points were calibrated with this arithmetic.

def window_audio(wave, sr: int, window_s: float, hop_s: float):
    import numpy as np

    size, hop, n = int(window_s * sr), int(hop_s * sr), len(wave)
    starts = list(range(0, max(n - size, 0) + 1, hop))
    if starts[-1] + size < n:
        starts.append(n - size)
    windows = np.stack([np.pad(wave[s:s + size], (0, max(0, size - len(wave[s:s + size]))))
                        for s in starts])
    valid = np.array([min(size, n - s) for s in starts])
    return windows, valid, np.array(starts) / sr


def stitch(probs, starts_s, duration_s: float, frame_s: float):
    import numpy as np

    grid = np.full(int(duration_s / frame_s) + 2, np.nan)
    for j, start in enumerate(starts_s):
        for k in range(probs.shape[1]):
            g = int((start + k * frame_s) / frame_s)
            if g < len(grid) and not np.isnan(probs[j, k]):
                grid[g] = probs[j, k] if np.isnan(grid[g]) else max(grid[g], probs[j, k])
    return grid


def decode(grid, point: dict, frame_s: float, duration_s: float) -> list[tuple[float, float, float]]:
    import numpy as np

    times = np.arange(len(grid)) * frame_s
    hot = grid >= point["threshold"]  # NaN compares False
    spans, start = [], None
    for i, h in enumerate(hot):
        if h and start is None:
            start = times[i]
        elif not h and start is not None:
            spans.append([start, times[i]])
            start = None
    if start is not None:
        spans.append([start, times[-1] + frame_s])
    merged: list[list[float]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + point["merge_gap_s"]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    found = []
    for s, e in merged:
        if e - s < point["min_duration_s"]:
            continue
        a = int(round(s / frame_s))
        b = max(a + 1, int(round(e / frame_s)))
        found.append((round(float(s), 3), round(float(min(e, duration_s)), 3),
                      round(float(np.nanmax(grid[a:b])), 4)))
    return found


# A session's laughs, cached per raw segment.

def _read_cache(path: Path) -> dict[str, dict[str, list]]:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if cache.pop("_source", None) != SOURCE:
        return {}
    return cache


def _write_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps({"_source": SOURCE, **cache}), encoding="utf-8")


def _track_wav(segments: list, track: int, destination: Path, runner=subprocess.run) -> Path | None:
    """`track` of `segments`, joined as 16kHz mono, each cut to its listed length."""
    listing = destination.with_suffix(".txt")
    listing.write_text("".join(
        f"file '{s.path.resolve().as_posix()}'\noutpoint {s.end - s.start:.6f}\n" for s in segments),
        encoding="utf-8")
    done = runner(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                   "-map", f"0:a:{track}", "-ac", "1", "-ar", str(SAMPLE_RATE), str(destination)],
                  capture_output=True)
    return destination if done.returncode == 0 and destination.is_file() else None


def listen(session: Path, *, detector=None, runner=subprocess.run,
           limit: int | None = None) -> list[Laugh]:
    """Every laugh in the session, detecting only segments not yet cached
    (at most `limit` of them, so a caller between matches can stop early)."""
    from pipeline_v2.audio import read_wav

    cache_path = session / CACHE_NAME
    cache = _read_cache(cache_path)
    segments = clips_mod.read_segments(session)
    todo = [s for s in segments if s.path.name not in cache and s.path.is_file()]
    todo = todo[:limit] if limit is not None else todo
    if todo:
        detector = detector or Detector()
        with tempfile.TemporaryDirectory(prefix="salience-laughs-") as tmp:
            for first in range(0, len(todo), CHUNK):
                chunk = todo[first:first + CHUNK]
                found: dict[str, dict[str, list]] = {s.path.name: {} for s in chunk}
                for name, track in TRACKS.items():
                    wav = _track_wav(chunk, track, Path(tmp) / f"{name}.wav", runner)
                    heard = [] if wav is None else detector.laughs(*read_wav(wav))
                    _place(chunk, heard, name, found)
                cache.update(found)
                _write_cache(cache_path, cache)
    return laughs(session)


def _place(chunk: list, heard: list, track: str, found: dict) -> None:
    offsets, at = [], 0.0
    for segment in chunk:
        offsets.append((at, segment))
        at += segment.end - segment.start
    for segment in chunk:
        found[segment.path.name].setdefault(track, [])
    for start, end, score in heard:
        base, segment = next((o for o in reversed(offsets) if o[0] <= start), offsets[0])
        shift = segment.start - base
        found[segment.path.name][track].append([start + shift, end + shift, score])


def laughs(session: Path) -> list[Laugh]:
    """What is already detected, without detecting anything."""
    return sorted(Laugh(start, end, score, track)
                  for by_track in _read_cache(session / CACHE_NAME).values()
                  for track, found in by_track.items()
                  for start, end, score in found)


def laughed_in(found: list[Laugh], start: float, end: float) -> list[Laugh]:
    """The laughs in a moment, or just after it (`AFTER_S`): a reaction follows."""
    return [laugh for laugh in found if laugh.start < end + AFTER_S and start < laugh.end]
