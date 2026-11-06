"""Experimental audio-intensity detectors over dBFS samples.
Offline selected-clip correlations motivate testing; they do not validate live
recall. The active daemon supplies WASAPI loopback levels, independently of OBS."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time


@dataclass
class TriggerConfig:
    # Rolling window that defines "normal" for this session. Long enough that a
    # firefight cannot quietly become the baseline.
    baseline_seconds: float = 90.0
    # How far above the rolling median counts as an event. 3.0 comes from the
    # measured envelope work: `loud_fraction` used the same multiple, and 20-25%
    # of a typical clip sat above it.
    threshold_multiple: float = 3.0
    # Minimum gap between saves. The replay buffer covers the preceding N
    # seconds, so two saves closer than this return overlapping footage -- which
    # is exactly the duplicate-coverage problem the engagement grouping exists to
    # clean up after. Better not to create it.
    cooldown_seconds: float = 25.0
    # Refuse to fire until the baseline is actually established, otherwise the
    # first loud noise of a session always triggers.
    warmup_seconds: float = 20.0
    # dB above the rolling median that counts as an event. 9.5 dB is 3x linear
    # amplitude -- the same multiple `loud_fraction` used in the clip analysis.
    # Exposed because it is inherited from clip-level envelopes, not from live
    # meters, and is the first thing a real session will disagree with.
    threshold_db: float = 9.5


@dataclass
class LevelTrigger:
    """Rolling-baseline peak detector over a stream of dB samples.

    dB rather than linear amplitude because OBS reports dB and because loudness
    perception is logarithmic; a linear threshold would fire constantly at high
    volume and never at low.
    """

    config: TriggerConfig = field(default_factory=TriggerConfig)
    _samples: deque = field(default_factory=lambda: deque(maxlen=4096))
    # Anchored to the first observation rather than to construction time, so the
    # clock the caller uses is the only clock that matters. Mixing a wall clock
    # with caller-supplied timestamps made warmup never complete.
    _started: float | None = None
    _last_fire: float | None = None

    def observe(self, level_db: float, now: float | None = None) -> bool:
        """Feed one meter reading. Returns True when a clip should be saved."""
        moment = now if now is not None else time.monotonic()
        if self._started is None:
            self._started = moment
        # OBS reports -inf (or very low) for silence; keep it out of the baseline
        # so a muted session does not make any noise look like an event.
        if level_db > -60.0:
            self._samples.append((moment, level_db))

        cutoff = moment - self.config.baseline_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if moment - self._started < self.config.warmup_seconds:
            return False
        if (
            self._last_fire is not None
            and moment - self._last_fire < self.config.cooldown_seconds
        ):
            return False
        if len(self._samples) < 20:
            return False

        levels = sorted(value for _, value in self._samples)
        median = levels[len(levels) // 2]
        # 3x linear amplitude is ~9.5 dB, so the measured multiple carries over.
        if level_db - median >= self.config.threshold_db:
            self._last_fire = moment
            return True
        return False

    @property
    def ready(self) -> bool:
        if self._started is None:
            return False
        return time.monotonic() - self._started >= self.config.warmup_seconds


@dataclass
class SustainedConfig(TriggerConfig):
    # Averaging window. A gunshot is ~50ms; a firefight is seconds. Averaging
    # over 1.5s is what separates the two, and separating them is the whole
    # point -- the peak detector fires on individual shots, which is why it
    # saturated its own cooldown on the first real session.
    window_seconds: float = 1.5
    # Measured on session-1: both hand-marked moments peaked at 10.8 dB
    # sustained, around the 93rd percentile of that session. Low confidence --
    # two marks is not a sample -- so this is a starting point, not a finding.
    threshold_db: float = 10.0


@dataclass
class SustainedTrigger:
    """Rolling-baseline detector over *averaged* level, not instantaneous peak.

    Same baseline machinery as `LevelTrigger`; the only difference is that the
    value compared against the median is a short moving average rather than the
    latest sample. Kept as a separate class rather than a flag on the original
    because the two answer different questions and should be measurable
    side by side.
    """

    config: SustainedConfig = field(default_factory=SustainedConfig)
    _samples: deque = field(default_factory=lambda: deque(maxlen=4096))
    _window: deque = field(default_factory=lambda: deque(maxlen=4096))
    _started: float | None = None
    _last_fire: float | None = None

    def level(self) -> float | None:
        """Current windowed mean, or None before the window fills."""
        if not self._window:
            return None
        return sum(v for _, v in self._window) / len(self._window)

    def observe(self, level_db: float, now: float | None = None) -> bool:
        moment = now if now is not None else time.monotonic()
        if self._started is None:
            self._started = moment

        # The averaging window includes silence deliberately: a quiet gap in the
        # middle of a burst should pull the average down, because a burst with
        # gaps is not a sustained firefight.
        self._window.append((moment, level_db))
        while self._window and self._window[0][0] < moment - self.config.window_seconds:
            self._window.popleft()

        if level_db > -60.0:
            self._samples.append((moment, level_db))
        cutoff = moment - self.config.baseline_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if moment - self._started < self.config.warmup_seconds:
            return False
        if (
            self._last_fire is not None
            and moment - self._last_fire < self.config.cooldown_seconds
        ):
            return False
        if len(self._samples) < 20:
            return False

        levels = sorted(value for _, value in self._samples)
        median = levels[len(levels) // 2]
        windowed = self.level()
        if windowed is None:
            return False
        if windowed - median >= self.config.threshold_db:
            self._last_fire = moment
            return True
        return False

    def delta(self) -> float:
        """Windowed mean minus rolling median, for logging and meters."""
        windowed = self.level()
        if windowed is None or len(self._samples) < 20:
            return 0.0
        levels = sorted(value for _, value in self._samples)
        return windowed - levels[len(levels) // 2]
