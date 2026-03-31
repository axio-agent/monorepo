"""SQLiteContextStore: persistent conversation storage + ProjectConfig."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import aiosqlite
from axio.context import ContextStore, SessionInfo
from axio.messages import Message

GLOBAL_PROJECT = "<global>"


async def _connect(db_path: Path) -> aiosqlite.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id TEXT NOT NULL,"
        "  project TEXT NOT NULL,"
        "  position INTEGER NOT NULL,"
        "  role TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  created_at TEXT NOT NULL DEFAULT (datetime('now')),"
        "  UNIQUE(session_id, position)"
        ")"
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project)")
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS config ("
        "  project TEXT NOT NULL,"
        "  key TEXT NOT NULL,"
        "  value TEXT NOT NULL,"
        "  PRIMARY KEY(project, key)"
        ")"
    )
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS context_tokens ("
        "  session_id TEXT NOT NULL,"
        "  project TEXT NOT NULL,"
        "  input_tokens INTEGER NOT NULL DEFAULT 0,"
        "  output_tokens INTEGER NOT NULL DEFAULT 0,"
        "  PRIMARY KEY(session_id, project)"
        ")"
    )
    return conn


def _extract_preview(content_json: str, max_len: int = 80) -> str:
    """Extract text preview from serialized content JSON."""
    try:
        blocks = json.loads(content_json)
        for b in blocks:
            if b.get("type") == "text":
                text: str = b["text"]
                return text[:max_len] + ("..." if len(text) > max_len else "")
    except (json.JSONDecodeError, KeyError):
        pass
    return "(no preview)"


class SQLiteContextStore(ContextStore):
    """Persistent conversation storage backed by SQLite."""

    def __init__(self, db_path: str | Path, session_id: str, project: str | None = None) -> None:
        self._db_path = Path(db_path)
        self._session_id = session_id
        self._project = project or str(Path.cwd().resolve())
        self._conn: aiosqlite.Connection | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await _connect(self._db_path)
        return self._conn

    async def append(self, message: Message) -> None:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT COALESCE(MAX(position), -1) FROM messages WHERE session_id = ?",
            (self._session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        pos = (row[0] if row else -1) + 1
        content_json = json.dumps(message.to_dict()["content"])
        await conn.execute(
            "INSERT INTO messages (session_id, project, position, role, content) VALUES (?, ?, ?, ?, ?)",
            (self._session_id, self._project, pos, message.role, content_json),
        )
        await conn.commit()

    async def get_history(self) -> list[Message]:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY position",
            (self._session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [Message.from_dict({"role": role, "content": json.loads(content)}) for role, content in rows]

    async def clear(self) -> None:
        conn = await self._ensure_conn()
        await conn.execute("DELETE FROM messages WHERE session_id = ?", (self._session_id,))
        await conn.execute(
            "DELETE FROM context_tokens WHERE session_id = ? AND project = ?",
            (self._session_id, self._project),
        )
        await conn.commit()

    async def fork(self) -> SQLiteContextStore:
        new_id = uuid4().hex
        conn = await self._ensure_conn()
        await conn.execute(
            "INSERT INTO messages (session_id, project, position, role, content) "
            "SELECT ?, project, position, role, content FROM messages WHERE session_id = ?",
            (new_id, self._session_id),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO context_tokens (session_id, project, input_tokens, output_tokens) "
            "SELECT ?, project, input_tokens, output_tokens FROM context_tokens "
            "WHERE session_id = ? AND project = ?",
            (new_id, self._session_id, self._project),
        )
        await conn.commit()
        return SQLiteContextStore(self._db_path, new_id, self._project)

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            "INSERT INTO context_tokens (session_id, project, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id, project) DO UPDATE SET input_tokens=?, output_tokens=?",
            (self._session_id, self._project, input_tokens, output_tokens, input_tokens, output_tokens),
        )
        await conn.commit()

    async def add_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            "INSERT INTO context_tokens (session_id, project, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id, project) DO UPDATE "
            "SET input_tokens = input_tokens + excluded.input_tokens, "
            "    output_tokens = output_tokens + excluded.output_tokens",
            (self._session_id, self._project, input_tokens, output_tokens),
        )
        await conn.commit()

    async def get_context_tokens(self) -> tuple[int, int]:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT input_tokens, output_tokens FROM context_tokens WHERE session_id = ? AND project = ?",
            (self._session_id, self._project),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return 0, 0
        return int(row[0]), int(row[1])

    async def list_sessions(self) -> list[SessionInfo]:
        """List all sessions for a project, newest first."""
        if not self._db_path.exists():
            return []
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT m.session_id, COUNT(*) as cnt, "
            "(SELECT content FROM messages WHERE session_id = m.session_id "
            "AND role = 'user' ORDER BY position LIMIT 1) as first_content, "
            "MIN(m.created_at) as created, "
            "COALESCE(ct.input_tokens, 0), COALESCE(ct.output_tokens, 0) "
            "FROM messages m "
            "LEFT JOIN context_tokens ct ON ct.session_id = m.session_id AND ct.project = m.project "
            "WHERE m.project = ? "
            "GROUP BY m.session_id ORDER BY created DESC",
            (self._project,),
        ) as cursor:
            rows = await cursor.fetchall()
        result: list[SessionInfo] = []
        for session_id, count, first_content, created_at, in_tok, out_tok in rows:
            preview = _extract_preview(first_content) if first_content else "(no preview)"
            result.append(
                SessionInfo(
                    session_id=session_id,
                    message_count=count,
                    preview=preview,
                    created_at=created_at,
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                )
            )
        return result

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


class ProjectConfig:
    """Per-project key-value config backed by SQLite."""

    def __init__(self, db_path: str | Path, project: str | None = None) -> None:
        self._db_path = Path(db_path)
        self._project = project or str(Path.cwd().resolve())
        self._conn: aiosqlite.Connection | None = None

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await _connect(self._db_path)
        return self._conn

    async def get(self, key: str, default: str | None = None) -> str | None:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT value FROM config WHERE project = ? AND key = ?",
            (self._project, key),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else default

    async def set(self, key: str, value: str) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            "INSERT INTO config (project, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(project, key) DO UPDATE SET value = excluded.value",
            (self._project, key, value),
        )
        await conn.commit()

    async def delete(self, key: str) -> None:
        conn = await self._ensure_conn()
        await conn.execute("DELETE FROM config WHERE project = ? AND key = ?", (self._project, key))
        await conn.commit()

    async def get_prefix(self, prefix: str) -> dict[str, str]:
        """Return all keys matching a prefix, e.g. ``transport.nebius.``."""
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT key, value FROM config WHERE project = ? AND key LIKE ?",
            (self._project, prefix + "%"),
        ) as cursor:
            rows = await cursor.fetchall()
        return {str(k): str(v) for k, v in rows}

    async def delete_prefix(self, prefix: str) -> None:
        """Delete all keys matching a prefix."""
        conn = await self._ensure_conn()
        await conn.execute(
            "DELETE FROM config WHERE project = ? AND key LIKE ?",
            (self._project, prefix + "%"),
        )
        await conn.commit()

    async def all(self) -> dict[str, str]:
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT key, value FROM config WHERE project = ?",
            (self._project,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {str(k): str(v) for k, v in rows}

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
