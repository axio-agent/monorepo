"""Tests for AutoCompactStore wrapping SQLiteContextStore."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
import pytest
from axio.blocks import TextBlock
from axio.compaction import AutoCompactStore
from axio.events import StreamEvent
from axio.messages import Message
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool

from axio_context_sqlite import SQLiteContextStore, connect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str = "user", text: str = "x") -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


class FailTransport:
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "autocompact_test.db"


@pytest.fixture
async def conn(db_path: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    c = await connect(db_path)
    yield c
    await c.close()


@pytest.fixture
async def sqlite_store(conn: aiosqlite.Connection) -> SQLiteContextStore:
    return SQLiteContextStore(conn, uuid4().hex, "test-project")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoCompactWithSQLite:
    async def test_history_written_back_to_sqlite(self, sqlite_store: SQLiteContextStore) -> None:
        """After compaction, messages in the DB are the compacted ones."""
        for i in range(14):
            role = "user" if i % 2 == 0 else "assistant"
            await sqlite_store.append(_msg(role, f"msg-{i}"))

        transport = StubTransport([make_text_response("Summary of earlier work")])
        store = AutoCompactStore(sqlite_store, transport, max_tokens=1, keep_recent=4)

        await store.add_context_tokens(2, 0)

        history = await sqlite_store.get_history()
        # 2 (summary pair) + 4 recent
        assert len(history) == 6
        assert history[0].role == "user"
        block0 = history[0].content[0]
        assert isinstance(block0, TextBlock)
        assert "Summary of earlier work" in block0.text
        assert history[1].role == "assistant"

    async def test_cumulative_tokens_persisted_in_sqlite(self, sqlite_store: SQLiteContextStore) -> None:
        """Cumulative token counts survive compaction in the SQLite table."""
        for i in range(14):
            role = "user" if i % 2 == 0 else "assistant"
            await sqlite_store.append(_msg(role, f"msg-{i}"))
        await sqlite_store.set_context_tokens(400_000, 15_000)

        transport = StubTransport([make_text_response("summary")])
        store = AutoCompactStore(sqlite_store, transport, max_tokens=1, keep_recent=4)

        await store.add_context_tokens(2, 0)

        in_tok, out_tok = await sqlite_store.get_context_tokens()
        assert in_tok == 400_002
        assert out_tok == 15_000

    async def test_sqlite_session_id_preserved(self, sqlite_store: SQLiteContextStore) -> None:
        """session_id of wrapper matches the inner SQLiteContextStore."""
        store = AutoCompactStore(sqlite_store, StubTransport([]), max_tokens=999_999)
        assert store.session_id == sqlite_store.session_id

    async def test_compact_does_not_corrupt_on_failure(self, sqlite_store: SQLiteContextStore) -> None:
        """If compaction fails, the original history in the DB is untouched."""
        msgs = [_msg("user" if i % 2 == 0 else "assistant", f"msg-{i}") for i in range(14)]
        for m in msgs:
            await sqlite_store.append(m)

        store = AutoCompactStore(sqlite_store, FailTransport(), max_tokens=1, keep_recent=4)
        await store.add_context_tokens(2, 0)

        history = await sqlite_store.get_history()
        assert len(history) == 14

    async def test_fork_creates_independent_sqlite_session(self, sqlite_store: SQLiteContextStore) -> None:
        """fork() returns an AutoCompactStore with a new session_id; mutations are independent."""
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            await sqlite_store.append(_msg(role, f"msg-{i}"))

        store = AutoCompactStore(sqlite_store, StubTransport([]), max_tokens=999_999)
        forked = await store.fork()

        assert forked.session_id != store.session_id

        # append to fork - parent unaffected
        await forked.append(_msg(text="extra"))
        assert len(await store.get_history()) == 6
        assert len(await forked.get_history()) == 7

    async def test_compact_uses_fork_as_snapshot(self, sqlite_store: SQLiteContextStore) -> None:
        """compact_context receives a fork of the store, not the live store itself."""
        for i in range(14):
            role = "user" if i % 2 == 0 else "assistant"
            await sqlite_store.append(_msg(role, f"msg-{i}"))

        seen_session_ids: list[str] = []

        import axio.compaction as _mod

        original = _mod.compact_context

        async def _capturing(ctx, transport, **kw):  # type: ignore[no-untyped-def]
            seen_session_ids.append(ctx.session_id)
            return await original(ctx, StubTransport([make_text_response("summary")]), **kw)

        _mod.compact_context = _capturing  # type: ignore[assignment]
        try:
            store = AutoCompactStore(
                sqlite_store, StubTransport([make_text_response("summary")]), max_tokens=1, keep_recent=4
            )
            await store.add_context_tokens(2, 0)
        finally:
            _mod.compact_context = original

        assert len(seen_session_ids) == 1
        assert seen_session_ids[0] != sqlite_store.session_id
