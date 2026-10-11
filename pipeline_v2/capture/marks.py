"""A hotkey that records "that was worth keeping", aligned to the recording.

Every detector in this project needs marks to be scored against, and the only
audio result the project has was tuned on a session whose three marks were also
used to choose the threshold. Marks have to be cheap to produce or there will
never be enough of them.

Pressed during play, so they are the player's own judgement at the moment it
happened rather than a reconstruction afterwards. Offsets are relative to the
start of the recording, which is what makes them line up with the footage and
with the level series `levels.py` extracts from it.

`GetAsyncKeyState` is a **polling** API: it reads keyboard state on demand and
installs no hook, injects nothing, and touches no other process. That matters
for a product shipped to players whose anti-cheat is watching.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# F9/F10: outside Fortnite's defaults, and not something a hand lands on.
VK_MARK = 0x78  # F9  -- something worth keeping just happened
VK_DULL = 0x79  # F10 -- explicitly nothing happened, a negative example

_PRESSED = 0x8000


@dataclass
class Mark:
    at: float  # seconds from the start of the recording
    kind: str  # keep | dull


@dataclass
class MarkRecorder:
    """Polls two keys and timestamps presses against a recording start.

    Edge-triggered: a held key marks once, not sixty times a second.
    """

    started: float = field(default_factory=time.monotonic)
    interval: float = 0.05
    marks: list[Mark] = field(default_factory=list)
    # Appended as each press happens, not written at exit. On Windows
    # `Popen.terminate()` is `TerminateProcess`: no catchable signal, so no
    # cleanup ever runs and an end-of-session write loses the whole session.
    # Same reasoning as persisting a paid API response before parsing it.
    journal: Path | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _down: dict[int, bool] = field(default_factory=dict)

    def _poll_once(self, now: float | None = None) -> Mark | None:
        user = ctypes.WinDLL("user32", use_last_error=True)
        moment = (now if now is not None else time.monotonic()) - self.started
        for key, kind in ((VK_MARK, "keep"), (VK_DULL, "dull")):
            down = bool(user.GetAsyncKeyState(key) & _PRESSED)
            was = self._down.get(key, False)
            self._down[key] = down
            if down and not was:
                mark = Mark(at=round(moment, 2), kind=kind)
                self.marks.append(mark)
                self._append(mark)
                return mark
        return None

    def _append(self, mark: Mark) -> None:
        """One line per mark. A partial file is still usable; a lost one is not."""
        if self.journal is None:
            return
        try:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(f"{mark.at}\t{mark.kind}\n")
        except OSError:
            pass  # a marking failure must never stop the recording

    @staticmethod
    def read_journal(path: Path) -> list[Mark]:
        """Recover marks from a session that was killed rather than stopped."""
        found: list[Mark] = []
        if not path.is_file():
            return found
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            try:
                found.append(Mark(at=float(parts[0]), kind=parts[1]))
            except ValueError:
                continue
        return found

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def start(self) -> "MarkRecorder":
        self._thread = threading.Thread(
            target=self._loop, name="marks", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> list[Mark]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return self.marks

    def save(self, path: Path) -> Path:
        """Two files: the full record, and the plain list the tuner reads."""
        from pipeline_v2.jsonio import write_json

        write_json(
            path,
            {
                "marks": [{"at": m.at, "kind": m.kind} for m in self.marks],
                "keep_seconds": [m.at for m in self.marks if m.kind == "keep"],
                "dull_seconds": [m.at for m in self.marks if m.kind == "dull"],
            },
        )
        return path
