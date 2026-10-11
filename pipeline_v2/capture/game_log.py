"""Gameplay events read from Fortnite's own log file.

Every other trigger idea in this project estimates what happened -- audio
loudness guesses at a fight, OCR guesses at a banner, a VLM costs money and
seconds. The game already writes some of the answer to disk, timestamped and
exact, and reading a log file is completely passive: no injection, no memory
reads, nothing an anti-cheat could object to.

What the default verbosity actually provides, confirmed against a real session:

* ``MatchState changed previous=X current=Y`` -- match start and end
* ``LogFortDeathCameraMode: Going to use death report`` -- the local player's
  death, with world coordinates

* ``LogPlayerKills: HandlePlayerStateChanged : <name>`` -- despite the category
  name this is **not** a kill. It fires when the kill-feed UI rebinds to a
  different player, which is the view changing to whoever is being watched.
  Timing proves it: a death is followed seconds later by this line carrying
  another player's name. That makes it the **spectator signal** -- exactly the
  viewpoint the descriptions got wrong on 14 of 18 windows in the September
  pilot, available here exactly and for free.

What the log does **not** provide is your own eliminations. No kill is written
at default verbosity, so the kill trigger still has to come from elsewhere.

Timestamps in the log are UTC despite looking local.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = (
    Path.home()
    / "AppData/Local/FortniteGame/Saved/Logs/FortniteGame.log"
)

# [2026.09.23-20.18.22:135][236]LogFort: MatchState changed previous=...
_LINE = re.compile(
    r"\[(?P<stamp>\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}):(?P<ms>\d{3})\]"
    r"\[\s*\d*\](?P<body>.*)"
)
_MATCH_STATE = re.compile(r"MatchState changed previous=(\w+) current=(\w+)")
_DEATH = re.compile(r"LogFortDeathCameraMode: Going to use death report")
# Despite the category name this is not a kill. It fires when the kill-feed UI
# rebinds to a different player state -- i.e. when the view changes to whoever
# is being watched. Confirmed by timing: a death at 19.22.58 is followed four
# seconds later by this line carrying another player's name.
_VIEWED = re.compile(r"LogPlayerKills: HandlePlayerStateChanged : (.+?)\s*$")


@dataclass(frozen=True)
class GameEvent:
    kind: str  # match_start | match_end | death | viewing
    at: datetime  # UTC
    detail: str = ""
    # For `viewing`: the player whose feed is on screen. Equal to the local
    # player during own gameplay, someone else while spectating.
    player: str = ""


def parse_timestamp(stamp: str, milliseconds: str = "0") -> datetime:
    moment = datetime.strptime(stamp, "%Y.%m.%d-%H.%M.%S")
    return moment.replace(
        microsecond=int(milliseconds) * 1000, tzinfo=timezone.utc
    )


def parse_line(line: str) -> GameEvent | None:
    """One log line to an event, or None for the overwhelming majority."""
    found = _LINE.match(line)
    if found is None:
        return None
    body = found.group("body")

    state = _MATCH_STATE.search(body)
    if state is not None:
        previous, current = state.groups()
        if current == "InProgress":
            kind = "match_start"
        elif previous == "InProgress":
            kind = "match_end"
        else:
            return None
        return GameEvent(
            kind=kind,
            at=parse_timestamp(found.group("stamp"), found.group("ms")),
            detail=f"{previous}->{current}",
        )

    viewed = _VIEWED.search(body)
    if viewed is not None:
        name = viewed.group(1).strip()
        return GameEvent(
            kind="viewing",
            at=parse_timestamp(found.group("stamp"), found.group("ms")),
            detail=f"viewing {name}",
            player=name,
        )

    if _DEATH.search(body):
        return GameEvent(
            kind="death",
            at=parse_timestamp(found.group("stamp"), found.group("ms")),
            detail="local player died",
        )
    return None


def read_events(path: Path | None = None) -> list[GameEvent]:
    """Every event in a log file, oldest first."""
    log = path or DEFAULT_LOG
    events: list[GameEvent] = []
    if not log.is_file():
        return events
    with log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            event = parse_line(line)
            if event is not None:
                events.append(event)
    return events


def local_player(events: list[GameEvent]) -> str | None:
    """Whose gameplay this log belongs to.

    The first `viewing` of a match is the local player watching themselves, so
    the name that appears at match start is the account's own. Inferred rather
    than configured, because a hardcoded name would be wrong for every other
    user of the product.
    """
    for index, event in enumerate(events):
        if event.kind != "match_start":
            continue
        for later in events[index + 1 : index + 4]:
            if later.kind == "viewing":
                return later.player
    return None


def spectating_spans(
    events: list[GameEvent], owner: str | None = None
) -> list[tuple[datetime, str]]:
    """When the view was on somebody else, and who.

    This is the viewpoint signal the descriptions could not get right on their
    own: 14 of 18 spectator windows in the September pilot were narrated as
    pre-match lobby footage. The log states it exactly and for free.
    """
    owner = owner or local_player(events)
    return [
        (event.at, event.player)
        for event in events
        if event.kind == "viewing" and owner and event.player != owner
    ]


class LogWatcher:
    """Yields new events as the game appends them.

    Polled rather than notified: the file is opened, read to the end, and
    closed each time, so the game is never blocked and a log rotation between
    polls cannot leave a stale handle open.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_LOG
        self._offset = 0
        self._inode: tuple | None = None

    def seek_to_end(self) -> "LogWatcher":
        """Ignore everything already written; only report what happens next."""
        if self.path.is_file():
            self._offset = self.path.stat().st_size
            self._inode = self._identity()
        return self

    def _identity(self) -> tuple | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_ctime_ns, stat.st_ino)

    def poll(self) -> list[GameEvent]:
        if not self.path.is_file():
            return []
        identity = self._identity()
        if identity != self._inode:
            # A new match can rotate the log; start over rather than seeking
            # into a file that is no longer the one the offset referred to.
            self._offset, self._inode = 0, identity
        events = []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._offset)
            for line in handle:
                event = parse_line(line)
                if event is not None:
                    events.append(event)
            self._offset = handle.tell()
        return events
