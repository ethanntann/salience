"""Recording, the tray app, highlights and the desktop window."""

import os as _os
import sys as _sys
from pathlib import Path as _Path

# An installed copy (`installer/`) carries its own FFmpeg beside its Python:
# found first, and by every process this one starts.
_BUNDLED_FFMPEG = _Path(_sys.executable).resolve().parent.parent / "ffmpeg"
if (_BUNDLED_FFMPEG / "ffmpeg.exe").is_file() and str(_BUNDLED_FFMPEG) not in _os.environ.get("PATH", ""):
    _os.environ["PATH"] = f"{_BUNDLED_FFMPEG}{_os.pathsep}{_os.environ.get('PATH', '')}"
