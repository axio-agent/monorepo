"""Tests for SQLiteContextStore and ProjectConfig."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import SessionInfo
from axio.messages import Message
from axio_context_sqlite import SQLiteContextStore, connect

from axio_tui.sqlite_config import ProjectConfig


class TestSQLiteContextStore:
    async def test_append_and_get_history(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        msg = Message(role="user", content=[TextBlock(text="hi")])
        await store.append(msg)
        history = await store.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content[0] == TextBlock(text="hi")

    async def test_ordering(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        await store.append(Message(role="assistant", content=[TextBlock(text="2")]))
        history = await store.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    async def test_clear(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        await store.clear()
        assert await store.get_history() == []

    async def test_fork_returns_copy(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        child = await store.fork()
        child_history = await child.get_history()
        assert len(child_history) == 1
        assert child_history[0].content[0] == TextBlock(text="hi")

    async def test_fork_isolation_child_to_parent(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        child = await store.fork()
        await child.append(Message(role="user", content=[TextBlock(text="2")]))
        assert len(await store.get_history()) == 1

    async def test_fork_isolation_parent_to_child(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        child = await store.fork()
        await store.append(Message(role="user", content=[TextBlock(text="2")]))
        assert len(await child.get_history()) == 1

    async def test_all_block_types(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        blocks = [
            TextBlock(text="hello"),
            ImageBlock(media_type="image/png", data=b"\x89PNG"),
            ToolUseBlock(id="call_1", name="echo", input={"msg": "hi"}),
            ToolResultBlock(tool_use_id="call_1", content="result"),
        ]
        await store.append(Message(role="assistant", content=blocks))
        history = await store.get_history()
        assert len(history) == 1
        restored = history[0].content
        assert restored[0] == TextBlock(text="hello")
        assert restored[1] == ImageBlock(media_type="image/png", data=b"\x89PNG")
        assert restored[2] == ToolUseBlock(id="call_1", name="echo", input={"msg": "hi"})
        assert restored[3] == ToolResultBlock(tool_use_id="call_1", content="result")

    async def test_tool_result_nested_content(self, db_conn: aiosqlite.Connection) -> None:
        nested = ToolResultBlock(
            tool_use_id="call_2",
            content=[TextBlock(text="inner"), ImageBlock(media_type="image/jpeg", data=b"\xff\xd8")],
            is_error=True,
        )
        store = SQLiteContextStore(db_conn, "s1")
        await store.append(Message(role="user", content=[nested]))
        history = await store.get_history()
        restored = history[0].content[0]
        assert isinstance(restored, ToolResultBlock)
        assert restored.is_error is True
        assert restored.content == [TextBlock(text="inner"), ImageBlock(media_type="image/jpeg", data=b"\xff\xd8")]

    async def test_session_isolation(self, db_conn: aiosqlite.Connection) -> None:
        s1 = SQLiteContextStore(db_conn, "session_a")
        s2 = SQLiteContextStore(db_conn, "session_b")
        await s1.append(Message(role="user", content=[TextBlock(text="a")]))
        await s2.append(Message(role="user", content=[TextBlock(text="b")]))
        assert len(await s1.get_history()) == 1
        assert (await s1.get_history())[0].content[0] == TextBlock(text="a")
        assert len(await s2.get_history()) == 1
        assert (await s2.get_history())[0].content[0] == TextBlock(text="b")

    async def test_persistence(self, tmp_path: Path) -> None:
        conn1 = await connect(tmp_path / "test.db")
        try:
            store = SQLiteContextStore(conn1, "persist")
            await store.append(Message(role="user", content=[TextBlock(text="saved")]))
        finally:
            await conn1.close()

        conn2 = await connect(tmp_path / "test.db")
        try:
            store2 = SQLiteContextStore(conn2, "persist")
            history = await store2.get_history()
            assert len(history) == 1
            assert history[0].content[0] == TextBlock(text="saved")
        finally:
            await conn2.close()


class TestProjectConfig:
    async def test_config_get_set(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(tmp_path / "cfg.db", project="/test")
        await cfg.set("model", "gpt-4")
        assert await cfg.get("model") == "gpt-4"
        await cfg.close()

    async def test_config_default(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(tmp_path / "cfg.db", project="/test")
        assert await cfg.get("missing") is None
        assert await cfg.get("missing", "fallback") == "fallback"
        await cfg.close()

    async def test_config_delete(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(tmp_path / "cfg.db", project="/test")
        await cfg.set("key", "val")
        await cfg.delete("key")
        assert await cfg.get("key") is None
        await cfg.close()

    async def test_config_all(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(tmp_path / "cfg.db", project="/test")
        await cfg.set("a", "1")
        await cfg.set("b", "2")
        result = await cfg.all()
        assert result == {"a": "1", "b": "2"}
        await cfg.close()

    async def test_config_project_isolation(self, tmp_path: Path) -> None:
        db = tmp_path / "cfg.db"
        c1 = ProjectConfig(db, project="/proj1")
        c2 = ProjectConfig(db, project="/proj2")
        await c1.set("key", "val1")
        await c2.set("key", "val2")
        assert await c1.get("key") == "val1"
        assert await c2.get("key") == "val2"
        await c1.close()
        await c2.close()


class TestListSessions:
    async def test_list_sessions(self, tmp_path: Path) -> None:
        proj = "/test/project"
        conn = await connect(tmp_path / "test.db")
        try:
            s1 = SQLiteContextStore(conn, "s1", project=proj)
            await s1.append(Message(role="user", content=[TextBlock(text="hello world")]))
            await s1.append(Message(role="assistant", content=[TextBlock(text="hi")]))

            s2 = SQLiteContextStore(conn, "s2", project=proj)
            await s2.append(Message(role="user", content=[TextBlock(text="second session")]))

            sessions = await s1.list_sessions()
        finally:
            await conn.close()

        assert len(sessions) == 2
        assert all(isinstance(s, SessionInfo) for s in sessions)
        ids = {s.session_id for s in sessions}
        assert ids == {"s1", "s2"}

        s1_info = next(s for s in sessions if s.session_id == "s1")
        assert s1_info.message_count == 2
        assert "hello world" in s1_info.preview

        s2_info = next(s for s in sessions if s.session_id == "s2")
        assert s2_info.message_count == 1
        assert "second session" in s2_info.preview

    async def test_list_sessions_project_isolation(self, tmp_path: Path) -> None:
        conn = await connect(tmp_path / "test.db")
        try:
            s1 = SQLiteContextStore(conn, "s1", project="/proj_a")
            await s1.append(Message(role="user", content=[TextBlock(text="in A")]))

            s2 = SQLiteContextStore(conn, "s2", project="/proj_b")
            await s2.append(Message(role="user", content=[TextBlock(text="in B")]))

            sessions_a = await s1.list_sessions()
            assert len(sessions_a) == 1
            assert sessions_a[0].session_id == "s1"

            sessions_b = await s2.list_sessions()
            assert len(sessions_b) == 1
            assert sessions_b[0].session_id == "s2"
        finally:
            await conn.close()

    async def test_list_sessions_empty(self, tmp_path: Path) -> None:
        conn = await connect(tmp_path / "test.db")
        try:
            store = SQLiteContextStore(conn, "any", project="/empty")
            sessions = await store.list_sessions()
            assert sessions == []

            s = SQLiteContextStore(conn, "s1", project="/other")
            await s.append(Message(role="user", content=[TextBlock(text="msg")]))

            sessions = await store.list_sessions()
            assert sessions == []
        finally:
            await conn.close()


class TestContextTokens:
    async def test_context_tokens_default_zero(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        assert await store.get_context_tokens() == (0, 0)

    async def test_set_get_context_tokens(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.set_context_tokens(100, 200)
        assert await store.get_context_tokens() == (100, 200)

    async def test_context_tokens_persist(self, tmp_path: Path) -> None:
        conn1 = await connect(tmp_path / "test.db")
        try:
            store = SQLiteContextStore(conn1, "s1")
            await store.set_context_tokens(500, 300)
        finally:
            await conn1.close()

        conn2 = await connect(tmp_path / "test.db")
        try:
            store2 = SQLiteContextStore(conn2, "s1")
            assert await store2.get_context_tokens() == (500, 300)
        finally:
            await conn2.close()

    async def test_add_context_tokens(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.add_context_tokens(100, 200)
        assert await store.get_context_tokens() == (100, 200)

    async def test_add_context_tokens_accumulates(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.add_context_tokens(10, 20)
        await store.add_context_tokens(30, 40)
        assert await store.get_context_tokens() == (40, 60)

    async def test_clear_resets_context_tokens(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.set_context_tokens(100, 200)
        await store.clear()
        assert await store.get_context_tokens() == (0, 0)

    async def test_fork_copies_context_tokens(self, db_conn: aiosqlite.Connection) -> None:
        store = SQLiteContextStore(db_conn, "s1")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        await store.set_context_tokens(100, 200)
        child = await store.fork()
        assert await child.get_context_tokens() == (100, 200)

    async def test_list_sessions_includes_tokens(self, tmp_path: Path) -> None:
        proj = "/test/project"
        conn = await connect(tmp_path / "test.db")
        try:
            store = SQLiteContextStore(conn, "s1", project=proj)
            await store.append(Message(role="user", content=[TextBlock(text="hello")]))
            await store.set_context_tokens(1000, 500)
            sessions = await store.list_sessions()
        finally:
            await conn.close()

        assert len(sessions) == 1
        assert sessions[0].input_tokens == 1000
        assert sessions[0].output_tokens == 500
