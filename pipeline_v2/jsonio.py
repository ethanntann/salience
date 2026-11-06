"""Atomic JSON writes and a UTC stamp, shared by capture and the frozen studies.

These lived in ``capture/collection.py``, which drags in a Windows-only ctypes
module. That made every frozen-evaluation entry point import the paused capture
subsystem, and made the package unimportable off Windows.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for attempt in range(4):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                # Windows readers can briefly prevent replacing an open file.
                time.sleep(0.02 * (attempt + 1))
    finally:
        # Cleanup must never replace the real failure. When a scanner holds the
        # temp file, `replace` raises and then this `unlink` raises too, and the
        # second exception is the one the caller sees -- an "access denied" on a
        # path nothing asked about, instead of the write that actually failed.
        # A leaked dot-prefixed temp is the cheaper outcome.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
