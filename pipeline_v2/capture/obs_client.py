"""Drive OBS's replay buffer over obs-websocket.

Why OBS rather than a capture engine of our own: OBS already solves fullscreen
exclusive capture, multi-GPU laptops, HDR, variable refresh, audio sync, and
hardware encoding. None of that is a differentiator for a clip-ranking product,
and all of it is months of Windows-specific support burden. We drive it and
replace it later only if the encoder ever becomes the interesting part -- which
it will not.

Two things this buys that a watched folder cannot:

1. **The event timestamp is known.** Saving the buffer at moment T means the
   event sits a known distance from the end of the file, instead of being guessed
   at `0.75 x duration`. The localization problem that consumed a day of
   measurement simply does not exist here.
2. **Separate audio tracks.** OBS can record desktop and microphone to different
   tracks, which is what the mixed stereo in the NVIDIA clips made impossible --
   and what any future voice-versus-effects work needs.

Anti-cheat note: OBS uses OS-level capture (Game Capture / Windows Graphics
Capture). Nothing here injects into a game process, which is the line that must
not be crossed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 4455


class ObsUnavailable(RuntimeError):
    """OBS is not running, or obs-websocket is not enabled/authenticated."""


@dataclass
class ObsConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    password: str = ""

    @classmethod
    def from_env(cls) -> "ObsConfig":
        return cls(
            host=os.environ.get("OBS_HOST", DEFAULT_HOST),
            port=int(os.environ.get("OBS_PORT", str(DEFAULT_PORT))),
            password=os.environ.get("OBS_PASSWORD", ""),
        )


class ObsReplayBuffer:
    """Thin wrapper over the handful of requests this product needs."""

    def __init__(self, config: ObsConfig | None = None) -> None:
        self.config = config or ObsConfig.from_env()
        self._client = None

    def connect(self):
        import obsws_python as obs

        try:
            self._client = obs.ReqClient(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                timeout=5,
            )
        except Exception as exc:  # connection refused, bad password, wrong port
            raise ObsUnavailable(
                f"cannot reach obs-websocket at {self.config.host}:{self.config.port} "
                f"({exc}). In OBS: Tools > WebSocket Server Settings > Enable, "
                f"then set OBS_PASSWORD."
            ) from exc
        return self

    @property
    def client(self):
        if self._client is None:
            self.connect()
        return self._client

    def version(self) -> str:
        info = self.client.get_version()
        return f"OBS {info.obs_version} / websocket {info.obs_web_socket_version}"

    def buffer_active(self) -> bool:
        return bool(self.client.get_replay_buffer_status().output_active)

    def ensure_buffer(self) -> bool:
        """Start the replay buffer if it is not already running.

        Returns True if it was started here. The buffer is what makes capture
        retroactive -- without it, a trigger can only record the future.
        """
        if self.buffer_active():
            return False
        self.client.start_replay_buffer()
        # OBS reports active before the encoder is truly ready; a short settle
        # avoids a save request landing on a buffer with nothing in it.
        for _ in range(20):
            time.sleep(0.25)
            if self.buffer_active():
                return True
        raise ObsUnavailable("replay buffer did not start")

    def save(self) -> Path | None:
        """Flush the buffer to disk and return the file OBS wrote.

        `save_replay_buffer` is asynchronous, so the path is polled rather than
        assumed: asking immediately returns the *previous* save.
        """
        previous = self._last_path()
        self.client.save_replay_buffer()
        for _ in range(40):  # up to ~10s; encoding a 30s buffer takes a moment
            time.sleep(0.25)
            current = self._last_path()
            if current and current != previous:
                return current
        return None

    def _last_path(self) -> Path | None:
        try:
            path = self.client.get_last_replay_buffer_replay().saved_replay_path
        except Exception:
            return None
        return Path(path) if path else None
