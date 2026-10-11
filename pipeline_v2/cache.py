"""Request cache and estimated-spend ledger.
Raw outputs survive validation failures; crash-safe exactly-once billing is not guaranteed."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from pipeline_v2 import config

_SCHEMA = """
create table if not exists responses (
    key           text primary key,
    clip_path     text not null,
    clip_sha      text not null,
    prompt_ver    text not null,
    raw           text not null,
    input_tokens  integer not null default 0,
    output_tokens integer not null default 0,
    cost_usd      real    not null default 0,
    created_at    real    not null
);
"""


@dataclass(frozen=True)
class CachedResponse:
    raw: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def clip_sha(path: Path) -> str:
    """Hash file contents, not the name -- renaming a clip must not re-bill it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.CACHE_DB)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def cache_key(sha: str, prompt_version: str, request: dict | None = None) -> str:
    key = f"{sha}:{prompt_version}"
    if request is not None:
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        key += f":{digest}"
    return key


def get(key: str) -> CachedResponse | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "select raw, input_tokens, output_tokens, cost_usd from responses"
            " where key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return None
    return CachedResponse(
        raw=row["raw"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=row["cost_usd"],
    )


def put(
    key: str,
    *,
    clip_path: Path,
    sha: str,
    prompt_version: str,
    raw: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    # Raw text lands on disk as well as in SQLite: if the schema ever changes or
    # the database is deleted, the paid-for responses survive as plain files.
    config.ensure_dirs()
    (config.RAW_DIR / f"{key.replace(':', '_')}.json").write_text(
        json.dumps(
            {"clip": str(clip_path), "prompt_version": prompt_version, "raw": raw},
            indent=2,
        ),
        encoding="utf-8",
    )
    with closing(_connect()) as conn, conn:
        conn.execute(
            "insert or replace into responses"
            " (key, clip_path, clip_sha, prompt_ver, raw, input_tokens,"
            "  output_tokens, cost_usd, created_at)"
            " values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                str(clip_path),
                sha,
                prompt_version,
                raw,
                input_tokens,
                output_tokens,
                cost_usd,
                time.time(),
            ),
        )


def spend_summary() -> dict:
    with closing(_connect()) as conn:
        row = conn.execute(
            "select count(*) as calls, coalesce(sum(cost_usd), 0) as spent,"
            " coalesce(sum(input_tokens), 0) as input_tokens,"
            " coalesce(sum(output_tokens), 0) as output_tokens from responses"
        ).fetchone()
    return dict(row)
