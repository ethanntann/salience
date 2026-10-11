"""SQLite storage for described clips.

Separate database from v1. The two pipelines share no tables, so v2 can change
its schema freely and a mistake here cannot damage the labels v1 depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time

from pipeline_v2 import config

_SCHEMA = """
create table if not exists clips (
    clip_sha     text primary key,
    path         text not null,
    engagement   text,
    duration_sec real,
    event_sec    real,
    event_source text,
    facts_json   text not null,
    described_at real not null
);

-- Keyed by (clip, prompt version), not by clip. An ablation is a different
-- description of the same clip, and a clip-only key silently overwrote 93
-- originals the first time a variant was run -- destroying the comparison the
-- variant existed to make.
create table if not exists descriptions (
    clip_sha    text not null references clips(clip_sha) on delete cascade,
    summary     text not null,
    payload     text not null,
    prompt_ver  text not null,
    primary key (clip_sha, prompt_ver)
);

create table if not exists embeddings (
    clip_sha   text not null references clips(clip_sha) on delete cascade,
    prompt_ver text not null default 'v2.3',
    model      text not null,
    dims       integer not null,
    vector     blob not null,
    primary key (clip_sha, prompt_ver)
);

-- Declared here and nowhere else. The review server used to carry its own copy
-- of this DDL without the label check; because `if not exists` lets the first
-- writer win and this module always connects first, that copy silently never
-- applied. One owner means the constraint cannot drift again.
create table if not exists preferences (
    clip_sha   text primary key references clips(clip_sha) on delete cascade,
    label      integer not null check(label in (0, 1)),
    decided_at real not null
);

-- A reviewer saying "that description is wrong", per description (clip and
-- prompt version), because the ranker reads only description text: an
-- inaccurate one is a ranking error. The flags are the accuracy test set any
-- prompt or model change is scored against.
create table if not exists description_flags (
    clip_sha   text not null references clips(clip_sha) on delete cascade,
    prompt_ver text not null,
    decided_at real not null,
    primary key (clip_sha, prompt_ver)
);
"""


@dataclass(frozen=True)
class StoredClip:
    clip_sha: str
    path: Path
    summary: str
    payload: dict
    facts: dict
    event_sec: float | None


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DATA_DIR / "v2.sqlite", timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma busy_timeout = 30000")
    # WAL lets a reader (a `taste` run) work while a long index run writes.
    # Best-effort: switching journal mode needs exclusive access, so a stale
    # journal from a crashed run makes it fail -- and a failed optimisation must
    # not take down the run it was meant to help.
    try:
        conn.execute("pragma journal_mode = wal")
    except sqlite3.OperationalError:
        pass
    conn.execute("pragma foreign_keys = on")
    conn.executescript(_SCHEMA)
    return conn


def upsert_clip(
    conn: sqlite3.Connection,
    *,
    clip_sha: str,
    path: Path,
    engagement: str | None,
    duration_sec: float | None,
    event_sec: float | None,
    event_source: str,
    facts: dict,
    summary: str,
    payload: dict,
    prompt_version: str,
) -> None:
    conn.execute(
        "insert into clips (clip_sha, path, engagement, duration_sec,"
        " event_sec, event_source, facts_json, described_at)"
        " values (?, ?, ?, ?, ?, ?, ?, ?)"
        " on conflict(clip_sha) do update set path=excluded.path,"
        " engagement=excluded.engagement, duration_sec=excluded.duration_sec,"
        " event_sec=excluded.event_sec, event_source=excluded.event_source,"
        " facts_json=excluded.facts_json, described_at=excluded.described_at",
        (
            clip_sha,
            str(path.resolve()),
            engagement,
            duration_sec,
            event_sec,
            event_source,
            json.dumps(facts),
            time.time(),
        ),
    )
    conn.execute(
        "insert or replace into descriptions (clip_sha, summary, payload, prompt_ver)"
        " values (?, ?, ?, ?)",
        (clip_sha, summary, json.dumps(payload), prompt_version),
    )


def put_embedding(
    conn: sqlite3.Connection,
    clip_sha: str,
    model: str,
    vector: list[float],
    prompt_ver: str = config.DEFAULT_PROMPT_VERSION,
) -> None:
    import numpy as np

    blob = np.asarray(vector, dtype=np.float32).tobytes()
    conn.execute(
        "insert or replace into embeddings"
        " (clip_sha, prompt_ver, model, dims, vector) values (?, ?, ?, ?, ?)",
        (clip_sha, prompt_ver, model, len(vector), blob),
    )


def embeddings(
    conn: sqlite3.Connection, prompt_ver: str = config.DEFAULT_PROMPT_VERSION
) -> dict[str, list[float]]:
    import numpy as np

    rows = conn.execute(
        "select clip_sha, vector from embeddings where prompt_ver = ?",
        (prompt_ver,),
    ).fetchall()
    return {
        row["clip_sha"]: np.frombuffer(row["vector"], dtype=np.float32).tolist()
        for row in rows
    }


def all_clips(
    conn: sqlite3.Connection, prompt_ver: str = config.DEFAULT_PROMPT_VERSION
) -> list[StoredClip]:
    rows = conn.execute(
        "select c.clip_sha, c.path, c.facts_json, c.event_sec,"
        " d.summary, d.payload from clips c"
        " join descriptions d on d.clip_sha = c.clip_sha"
        " where d.prompt_ver = ? order by c.path",
        (prompt_ver,),
    ).fetchall()
    return [
        StoredClip(
            clip_sha=row["clip_sha"],
            path=Path(row["path"]),
            summary=row["summary"],
            payload=json.loads(row["payload"]),
            facts=json.loads(row["facts_json"]),
            event_sec=row["event_sec"],
        )
        for row in rows
    ]


def optional_rows(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    """Read optional research features without requiring research-only tables."""
    if table not in {"audio_features", "peak_kind"}:
        raise ValueError(f"unsupported research table: {table}")
    if not conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone():
        return []
    return conn.execute(f"select * from {table}").fetchall()
