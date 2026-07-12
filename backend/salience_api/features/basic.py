from dataclasses import dataclass, field


@dataclass(frozen=True)
class BasicFeatures:
    duration_sec: float
    motion_score: float
    audio_peak_score: float
    silence_ratio: float
    extraction_confidence: float
    action_density: float = 0.0
    tags: list[str] = field(default_factory=list)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def duration_quality(duration_sec: float) -> float:
    if 8 <= duration_sec <= 45:
        return 1.0
    if 45 < duration_sec <= 75:
        return 0.6
    if 4 <= duration_sec < 8:
        return 0.5
    return 0.2


def estimate_metadata_features(
    *,
    duration_sec: float | None,
    size_bytes: int | None,
    fps: float | None,
) -> BasicFeatures:
    duration = duration_sec or 30.0
    bytes_per_second = (size_bytes or 0) / max(duration, 1.0)
    bitrate_signal = clamp(bytes_per_second / 4_000_000)
    fps_signal = clamp((fps or 30.0) / 120.0)
    motion_score = clamp((0.55 * bitrate_signal) + (0.25 * fps_signal) + 0.15)
    audio_peak_score = clamp(0.25 + (0.45 * bitrate_signal))
    silence_ratio = clamp(0.65 - (0.45 * bitrate_signal))
    action_density = clamp(
        (motion_score + audio_peak_score + (1.0 - silence_ratio)) / 3.0
    )
    return BasicFeatures(
        duration_sec=duration,
        motion_score=motion_score,
        audio_peak_score=audio_peak_score,
        silence_ratio=silence_ratio,
        extraction_confidence=0.5 if size_bytes else 0.35,
        action_density=action_density,
    )
