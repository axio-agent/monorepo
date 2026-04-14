"""TUI-specific config helpers: ProjectConfig."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

GLOBAL_PROJECT = "<global>"

__all__ = ["GLOBAL_PROJECT", "ProjectConfig", "connect_config"]


async def connect_config(db_path: str | Path) -> aiosqlite.Connection:
    """Open (or create) the config database. Caller owns the connection."""
    path = Path(db_path)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS axio_config ("
        "  project TEXT NOT NULL,"
        "  key TEXT NOT NULL,"
        "  value TEXT NOT NULL,"
        "  PRIMARY KEY(project, key)"
        ")"
    )
    await conn.commit()
    return conn


class ProjectConfig:
    """Per-project key-value config backed by SQLite.

    Accepts either an already-open ``aiosqlite.Connection`` (caller owns it,
    ``close()`` is a no-op) or a ``str | Path`` to a ``.db`` file (connection
    is opened lazily and closed by ``close()``).
    """

    def __init__(self, db: str | Path | aiosqlite.Connection, project: str | None = None) -> None:
        if isinstance(db, aiosqlite.Connection):
            self._conn: aiosqlite.Connection | None = db
            self._db_path: Path | None = None
            self._owns_conn = False
        else:
            self._conn = None
            self._db_path = Path(db)
            self._owns_conn = True
        self._project = project or str(Path.cwd().resolve())

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            assert self._db_path is not None
            self._conn = await connect_config(self._db_path)
        return self._conn

    async def get(self, key: str, default: str | None = None) -> str | None:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT value FROM axio_config WHERE project = ? AND key = ?",
            (self._project, key),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else default

    async def set(self, key: str, value: str) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            "INSERT INTO axio_config (project, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(project, key) DO UPDATE SET value = excluded.value",
            (self._project, key, value),
        )
        await conn.commit()

    async def delete(self, key: str) -> None:
        conn = await self._ensure_conn()
        await conn.execute("DELETE FROM axio_config WHERE project = ? AND key = ?", (self._project, key))
        await conn.commit()

    async def get_prefix(self, prefix: str) -> dict[str, str]:
        """Return all keys matching a prefix, e.g. ``transport.nebius.``."""
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT key, value FROM axio_config WHERE project = ? AND key LIKE ?",
            (self._project, prefix + "%"),
        ) as cursor:
            rows = await cursor.fetchall()
        return {str(k): str(v) for k, v in rows}

    async def delete_prefix(self, prefix: str) -> None:
        """Delete all keys matching a prefix."""
        conn = await self._ensure_conn()
        await conn.execute(
            "DELETE FROM axio_config WHERE project = ? AND key LIKE ?",
            (self._project, prefix + "%"),
        )
        await conn.commit()

    async def all(self) -> dict[str, str]:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT key, value FROM axio_config WHERE project = ?",
            (self._project,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {str(k): str(v) for k, v in rows}

    async def close(self) -> None:
        if self._conn is not None and self._owns_conn:
            await self._conn.close()
            self._conn = None
