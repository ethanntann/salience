"""How hard the player is working, from keyboard and mouse state alone.

Every other universal signal assumes the player makes a noise. Playing solo in
silence is probably the most common case there is, and voice detection has
nothing to say about it: no talking, no reaction, no trigger.

Input does. A fight is mechanically dense -- clicking, strafing, building,
swapping -- and downtime is not. Walking across the map is a held key; a build
fight is dozens of discrete actions a second. That difference is visible without
knowing anything about the game, which makes it the one universal signal that
works in Valorant, in Fortnite, and in a title nobody has adapted yet.

`GetAsyncKeyState` is a **polling** API. It reads keyboard state on demand,
installs no hook, injects into no process and records no text -- only how many
of a fixed set of keys changed state. That distinction matters for software
shipped to players whose anti-cheat is watching, and it is the same mechanism
`marks.py` already uses.

Two channels, because one was not enough. Counting key-down transitions is
discrete and misses continuous work: 40% of windows in a real 24.8-minute
session read zero presses, and 313 of the 485 zero runs were a single 0.5s
window sitting between presses rather than idle time. Holding W to run counts
once, and mouse movement -- most of the mechanical work in a shooter -- was not
counted at all. `GetLastInputInfo` reports the last moment input of any kind
arrived, so polling it fills exactly that gap and is just as passive.

First measurement, 24.8-minute session with 4 marks, presses channel only:
2.5x over the rolling median with a 4/s floor caught 3 of 4 in 44 firings,
against a 60s metronome's 2 of 4 in 24. Better than no skill, on four marks, so
encouraging and not established. The movement channel is recorded but **not yet
swept against marks** -- no session exists that contains it.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_PRESSED = 0x8000

# Movement, firing, building and weapon swaps -- the actions that separate a
# fight from a walk. Deliberately a fixed list of game-relevant keys rather than
# every key on the keyboard: this must never resemble a keylogger, and chat
# typing is activity that is not gameplay.
WATCHED: tuple[tuple[int, str], ...] = (
    (0x01, "mouse_left"),    # firing
    (0x02, "mouse_right"),   # aiming
    (0x57, "w"), (0x41, "a"), (0x53, "s"), (0x44, "d"),  # movement
    (0x20, "space"),         # jump
    (0x10, "shift"),         # sprint / edit modifier
    (0x11, "ctrl"),          # crouch
    (0x51, "q"), (0x45, "e"), (0x46, "f"), (0x47, "g"),  # build / interact
    (0x31, "1"), (0x32, "2"), (0x33, "3"), (0x34, "4"), (0x35, "5"),  # slots
)


class _LastInputInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]


@dataclass(frozen=True)
class ActivitySample:
    at: float  # seconds from the start of the recording
    presses: int  # key-down transitions in this window
    distinct: int  # how many different keys were involved
    # Polls where the system's last-input tick advanced. Key transitions alone
    # were blind to most of what a player does: holding W to run counts once and
    # then reads as nothing, and **mouse movement was not counted at all**,
    # which is most of the work in a shooter. Measured on a real session, 40% of
    # windows came back as zero presses, and 313 of 485 zero runs were a single
    # 0.5s window scattered between presses rather than genuine idle.
    # `GetLastInputInfo` advances on any input, aiming included.
    active_polls: int = 0
    # Polls where at least one watched key was *down*. The cheapest and most
    # direct answer to the blind spot: measured mid-match, W was held in 45% of
    # polls, D in 40%, space in 15%, while the presses channel reported one or
    # two transitions for the same window. Continuous engagement was already
    # being polled at 100Hz and discarded.
    held_polls: int = 0
    polls: int = 0

    @property
    def rate(self) -> float:
        """Presses per second, the comparable number across window sizes."""
        return self.presses / ActivitySampler.WINDOW_SECONDS

    @property
    def occupancy(self) -> float:
        """Share of the window with a movement or action key down, 0.0 to 1.0.

        Separates the three cases transitions alone conflate: standing in a menu
        (near 0), walking a straight line (high occupancy, one distinct key, few
        transitions) and a build fight (high on all three).
        """
        return self.held_polls / self.polls if self.polls else 0.0

    @property
    def activity(self) -> float:
        """Share of the window carrying input of any kind, 0.0 to 1.0.

        Continuous where `rate` is discrete: a player holding a key and moving
        the mouse scores high here and zero there.
        """
        return self.active_polls / self.polls if self.polls else 0.0


class ActivitySampler:
    """Counts key-down transitions and input presence per window, and journals both.

    Transitions, not held state: holding W to cross the map is one press, while
    a build fight is a stream of them. Counting held keys would rank running in
    a straight line as the busiest thing a player does.

    Presence covers what transitions cannot. A player aiming through a fight
    generates almost no transitions and near-continuous input, so on the presses
    channel alone that reads as idle.
    """

    WINDOW_SECONDS = 0.5
    POLL_SECONDS = 0.01  # 100Hz; fast enough not to miss a tap

    def __init__(self, journal: Path | None = None, started: float | None = None):
        self.journal = journal
        self.started = started if started is not None else time.monotonic()
        self.samples: list[ActivitySample] = []
        self._down: dict[int, bool] = {}
        self._presses = 0
        self._keys: set[int] = set()
        self._polls = 0
        self._active = 0
        self._held = 0
        self._last_tick: int | None = None
        self._window_end = self.WINDOW_SECONDS
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _input_tick(self, user) -> int | None:
        """Millisecond tick of the last input of any kind, or None if unavailable.

        A seam, so the presence channel can be driven from a test without
        constructing win32 structures.
        """
        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(info)
        if not user.GetLastInputInfo(ctypes.byref(info)):
            return None
        return int(info.dwTime)

    def _read(self, user) -> None:
        self._polls += 1
        tick = self._input_tick(user)
        if tick is not None:
            # Counts *changes*, not a nonzero reading: the tick only moves when
            # input actually arrived, so an idle window leaves it standing still.
            if self._last_tick is not None and tick != self._last_tick:
                self._active += 1
            self._last_tick = tick
        any_down = False
        for key, _ in WATCHED:
            down = bool(user.GetAsyncKeyState(key) & _PRESSED)
            if down:
                any_down = True
            if down and not self._down.get(key, False):
                self._presses += 1
                self._keys.add(key)
            self._down[key] = down
        if any_down:
            self._held += 1

    def _close_window(self, moment: float) -> ActivitySample:
        sample = ActivitySample(
            at=round(self._window_end, 2),
            presses=self._presses,
            distinct=len(self._keys),
            active_polls=self._active,
            held_polls=self._held,
            polls=self._polls,
        )
        self.samples.append(sample)
        self._append(sample)
        self._presses, self._keys = 0, set()
        self._polls, self._active, self._held = 0, 0, 0
        self._window_end += self.WINDOW_SECONDS
        return sample

    def poll_once(self, now: float | None = None, user=None) -> ActivitySample | None:
        """One poll. Returns a sample only when a window closes."""
        client = user if user is not None else ctypes.WinDLL(
            "user32", use_last_error=True
        )
        moment = (now if now is not None else time.monotonic()) - self.started
        self._read(client)
        if moment >= self._window_end:
            return self._close_window(moment)
        return None

    def _append(self, sample: ActivitySample) -> None:
        """Journalled per window, for the same reason marks are.

        A hard kill on Windows runs no cleanup, so anything held until the end
        of a session is lost with it.
        """
        if self.journal is None:
            return
        try:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{sample.at}\t{sample.presses}\t{sample.distinct}"
                    f"\t{sample.active_polls}\t{sample.held_polls}"
                    f"\t{sample.polls}\n"
                )
        except OSError:
            pass  # telemetry must never interrupt a recording

    @staticmethod
    def read_journal(path: Path) -> list[ActivitySample]:
        found: list[ActivitySample] = []
        if not path.is_file():
            return found
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            # Three columns is the original format, written before movement
            # and occupancy were counted; five is the short-lived middle one.
            # Both stay readable so already-recorded sessions remain usable, and
            # `polls == 0` distinguishes "never measured" from "measured idle".
            if len(parts) not in (3, 5, 6):
                continue
            try:
                columns = [float(parts[0])] + [int(v) for v in parts[1:]]
            except ValueError:
                continue
            at, presses, distinct = columns[0], int(columns[1]), int(columns[2])
            if len(columns) == 6:
                active, held, polls = (int(v) for v in columns[3:])
            elif len(columns) == 5:
                active, held, polls = int(columns[3]), 0, int(columns[4])
            else:
                active, held, polls = 0, 0, 0
            found.append(
                ActivitySample(at, presses, distinct, active, held, polls)
            )
        return found

    def _loop(self) -> None:
        user = ctypes.WinDLL("user32", use_last_error=True)
        while not self._stop.is_set():
            self.poll_once(user=user)
            self._stop.wait(self.POLL_SECONDS)

    def start(self) -> "ActivitySampler":
        self._thread = threading.Thread(
            target=self._loop, name="input-activity", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> list[ActivitySample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return self.samples


@dataclass
class InputBurstDetector:
    """Fires when input rate rises well above the player's own recent normal.

    Relative to a rolling baseline, not an absolute rate, because a controller
    player, a keyboard-and-mouse player and a builder all have different
    normals. The same reasoning as the voice detector: the product cannot ask
    every user to configure a threshold.

    Thresholds here are **guesses**. They exist to be swept against a recorded
    session with marks, exactly as the voice thresholds are.
    """

    name = "input_burst"
    # Swept on the same session: 2.5x over the rolling median with a 4/s floor
    # caught 3/4 marks in 44 firings. Raising the multiple to 4.0 caught none,
    # so the burst is modest rather than dramatic. Four marks; not validated.
    multiple: float = 2.5  # of the rolling median press rate
    minimum_rate: float = 4.0  # presses/sec floor, so idle noise cannot fire
    cooldown_seconds: float = 20.0
    baseline_seconds: float = 60.0
    _history: list[tuple[float, float]] = field(default_factory=list)
    _last_fire: float | None = None

    def observe(self, *, press_rate: float | None = None,
                now: float | None = None, **_):
        from pipeline_v2.capture.triggers import Trigger

        if press_rate is None:
            return None
        moment = now if now is not None else time.monotonic()
        self._history.append((moment, press_rate))
        cutoff = moment - self.baseline_seconds
        self._history = [(t, v) for t, v in self._history if t >= cutoff]
        if len(self._history) < 10:
            return None  # no baseline yet

        values = sorted(v for _, v in self._history)
        median = values[len(values) // 2]
        if press_rate < self.minimum_rate:
            return None
        if median > 0 and press_rate < median * self.multiple:
            return None
        if median == 0 and press_rate < self.minimum_rate:
            return None
        if self._last_fire is not None and (
            moment - self._last_fire < self.cooldown_seconds
        ):
            return None

        self._last_fire = moment
        return Trigger(
            kind="input_burst",
            at=moment,
            source=self.name,
            # Above the voice detector's 0.5 and below an exact game event's
            # 1.0. Raised from 0.4 once measured: input bursts hit 24.1% against
            # a random baseline of 8.2%, while voice hit 4.5%. At 0.4 the worse
            # channel won every tie and its cooldown then suppressed this one.
            confidence=0.6,
            detail=f"{press_rate:.1f} presses/s vs median {median:.1f}",
        )
