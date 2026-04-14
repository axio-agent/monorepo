"""Tests for SQLiteContextStore."""

from collections.abc import AsyncGenerator
from pathlib import Path

import aiosqlite
import pytest
from axio.blocks import TextBlock
from axio.messages import Message

from axio_context_sqlite import SQLiteContextStore, connect
from axio_context_sqlite.store import COMPRESS_THRESHOLD, compress_payload, decompress_payload


def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
async def conn(db_path: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    c = await connect(db_path)
    yield c
    await c.close()


@pytest.fixture
async def store(conn: aiosqlite.Connection) -> SQLiteContextStore:
    return SQLiteContextStore(conn, "session-1", "test-project")


async def test_append_and_get_history(store: SQLiteContextStore) -> None:
    await store.append(_msg("user", "Hello"))
    await store.append(_msg("assistant", "Hi!"))
    history = await store.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


async def test_clear(store: SQLiteContextStore) -> None:
    await store.append(_msg("user", "Hello"))
    await store.clear()
    assert await store.get_history() == []


async def test_fork(store: SQLiteContextStore) -> None:
    await store.append(_msg("user", "Hello"))
    forked = await store.fork()
    await forked.append(_msg("assistant", "Hi!"))
    assert len(await store.get_history()) == 1
    assert len(await forked.get_history()) == 2


async def test_set_get_context_tokens(store: SQLiteContextStore) -> None:
    await store.set_context_tokens(100, 50)
    inp, out = await store.get_context_tokens()
    assert inp == 100
    assert out == 50


async def test_add_context_tokens(store: SQLiteContextStore) -> None:
    await store.set_context_tokens(100, 50)
    await store.add_context_tokens(20, 10)
    inp, out = await store.get_context_tokens()
    assert inp == 120
    assert out == 60


async def test_list_sessions(db_path: Path) -> None:
    c = await connect(db_path)
    try:
        s1 = SQLiteContextStore(c, "sess-a", "proj")
        s2 = SQLiteContextStore(c, "sess-b", "proj")
        await s1.append(_msg("user", "First session"))
        await s2.append(_msg("user", "Second session"))
        sessions = await s1.list_sessions()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"sess-a", "sess-b"}
        previews = {s.preview for s in sessions}
        assert "First session" in previews
        assert "Second session" in previews
    finally:
        await c.close()


class TestCompressPayload:
    def test_small_stays_plain(self) -> None:
        data = "hello"
        result = compress_payload(data)
        assert result.startswith("plain:")
        assert decompress_payload(result) == data

    def test_large_gets_compressed(self) -> None:
        data = "x" * COMPRESS_THRESHOLD
        result = compress_payload(data)
        assert result.startswith("gzip:")
        assert decompress_payload(result) == data

    def test_legacy_no_prefix(self) -> None:
        assert decompress_payload('{"foo": 1}') == '{"foo": 1}'

    def test_roundtrip(self) -> None:
        data = "[" + '{"type":"text","text":"a"},' * 50 + "]"
        assert decompress_payload(compress_payload(data)) == data


async def test_large_message_compressed_on_disk(db_path: Path) -> None:
    """Large content is stored compressed; get_history still returns original."""
    big_text = "word " * 200  # well above threshold

    c = await connect(db_path)
    try:
        s = SQLiteContextStore(c, "big-session", "proj")
        await s.append(_msg("user", big_text))
    finally:
        await c.close()

    # Inspect raw bytes on disk — should start with gzip:
    async with aiosqlite.connect(str(db_path)) as raw:
        async with raw.execute("SELECT content FROM axio_context_messages") as cur:
            row = await cur.fetchone()
    assert row is not None
    assert row[0].startswith("gzip:")

    # But get_history transparently decompresses
    c2 = await connect(db_path)
    try:
        s2 = SQLiteContextStore(c2, "big-session", "proj")
        history = await s2.get_history()
    finally:
        await c2.close()
    assert len(history) == 1
    block = history[0].content[0]
    assert isinstance(block, TextBlock)
    assert block.text == big_text


async def test_close_and_reopen(db_path: Path) -> None:
    """Data persists after close and reopen."""
    c = await connect(db_path)
    try:
        s = SQLiteContextStore(c, "persist-session", "proj")
        await s.append(_msg("user", "Persistent message"))
        await s.set_context_tokens(42, 7)
    finally:
        await c.close()

    c2 = await connect(db_path)
    try:
        s2 = SQLiteContextStore(c2, "persist-session", "proj")
        history = await s2.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        inp, out = await s2.get_context_tokens()
        assert inp == 42
        assert out == 7
    finally:
        await c2.close()
