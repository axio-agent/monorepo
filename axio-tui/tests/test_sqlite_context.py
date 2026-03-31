"""Tests for SQLiteContextStore and ProjectConfig."""

from __future__ import annotations

from pathlib import Path

from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import SessionInfo
from axio.messages import Message

from axio_tui.sqlite_context import ProjectConfig, SQLiteContextStore


def _db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


class TestSQLiteContextStore:
    async def test_append_and_get_history(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        msg = Message(role="user", content=[TextBlock(text="hi")])
        await store.append(msg)
        history = await store.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content[0] == TextBlock(text="hi")
        await store.close()

    async def test_ordering(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        await store.append(Message(role="assistant", content=[TextBlock(text="2")]))
        history = await store.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        await store.close()

    async def test_clear(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        await store.clear()
        assert await store.get_history() == []
        await store.close()

    async def test_fork_returns_copy(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        child = await store.fork()
        child_history = await child.get_history()
        assert len(child_history) == 1
        assert child_history[0].content[0] == TextBlock(text="hi")
        await child.close()
        await store.close()

    async def test_fork_isolation_child_to_parent(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        child = await store.fork()
        await child.append(Message(role="user", content=[TextBlock(text="2")]))
        assert len(await store.get_history()) == 1
        await child.close()
        await store.close()

    async def test_fork_isolation_parent_to_child(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        child = await store.fork()
        await store.append(Message(role="user", content=[TextBlock(text="2")]))
        assert len(await child.get_history()) == 1
        await child.close()
        await store.close()

    async def test_all_block_types(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
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
        await store.close()

    async def test_tool_result_nested_content(self, tmp_path: Path) -> None:
        nested = ToolResultBlock(
            tool_use_id="call_2",
            content=[TextBlock(text="inner"), ImageBlock(media_type="image/jpeg", data=b"\xff\xd8")],
            is_error=True,
        )
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.append(Message(role="user", content=[nested]))
        history = await store.get_history()
        restored = history[0].content[0]
        assert isinstance(restored, ToolResultBlock)
        assert restored.is_error is True
        assert restored.content == [TextBlock(text="inner"), ImageBlock(media_type="image/jpeg", data=b"\xff\xd8")]
        await store.close()

    async def test_session_isolation(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        s1 = SQLiteContextStore(db, "session_a")
        s2 = SQLiteContextStore(db, "session_b")
        await s1.append(Message(role="user", content=[TextBlock(text="a")]))
        await s2.append(Message(role="user", content=[TextBlock(text="b")]))
        assert len(await s1.get_history()) == 1
        assert (await s1.get_history())[0].content[0] == TextBlock(text="a")
        assert len(await s2.get_history()) == 1
        assert (await s2.get_history())[0].content[0] == TextBlock(text="b")
        await s1.close()
        await s2.close()

    async def test_persistence(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        store = SQLiteContextStore(db, "persist")
        await store.append(Message(role="user", content=[TextBlock(text="saved")]))
        await store.close()

        store2 = SQLiteContextStore(db, "persist")
        history = await store2.get_history()
        assert len(history) == 1
        assert history[0].content[0] == TextBlock(text="saved")
        await store2.close()


class TestProjectConfig:
    async def test_config_get_set(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(_db(tmp_path), project="/test")
        await cfg.set("model", "gpt-4")
        assert await cfg.get("model") == "gpt-4"
        await cfg.close()

    async def test_config_default(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(_db(tmp_path), project="/test")
        assert await cfg.get("missing") is None
        assert await cfg.get("missing", "fallback") == "fallback"
        await cfg.close()

    async def test_config_delete(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(_db(tmp_path), project="/test")
        await cfg.set("key", "val")
        await cfg.delete("key")
        assert await cfg.get("key") is None
        await cfg.close()

    async def test_config_all(self, tmp_path: Path) -> None:
        cfg = ProjectConfig(_db(tmp_path), project="/test")
        await cfg.set("a", "1")
        await cfg.set("b", "2")
        result = await cfg.all()
        assert result == {"a": "1", "b": "2"}
        await cfg.close()

    async def test_config_project_isolation(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
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
        db = _db(tmp_path)
        proj = "/test/project"
        s1 = SQLiteContextStore(db, "s1", project=proj)
        await s1.append(Message(role="user", content=[TextBlock(text="hello world")]))
        await s1.append(Message(role="assistant", content=[TextBlock(text="hi")]))
        await s1.close()

        s2 = SQLiteContextStore(db, "s2", project=proj)
        await s2.append(Message(role="user", content=[TextBlock(text="second session")]))
        await s2.close()

        query = SQLiteContextStore(db, "any", project=proj)
        sessions = await query.list_sessions()
        await query.close()
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
        db = _db(tmp_path)
        s1 = SQLiteContextStore(db, "s1", project="/proj_a")
        await s1.append(Message(role="user", content=[TextBlock(text="in A")]))
        await s1.close()

        s2 = SQLiteContextStore(db, "s2", project="/proj_b")
        await s2.append(Message(role="user", content=[TextBlock(text="in B")]))
        await s2.close()

        qa = SQLiteContextStore(db, "any", project="/proj_a")
        sessions_a = await qa.list_sessions()
        await qa.close()
        assert len(sessions_a) == 1
        assert sessions_a[0].session_id == "s1"

        qb = SQLiteContextStore(db, "any", project="/proj_b")
        sessions_b = await qb.list_sessions()
        await qb.close()
        assert len(sessions_b) == 1
        assert sessions_b[0].session_id == "s2"

    async def test_list_sessions_empty(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        # DB doesn't exist yet
        store = SQLiteContextStore(db, "any", project="/empty")
        sessions = await store.list_sessions()
        assert sessions == []

        # DB exists but no sessions for this project
        s = SQLiteContextStore(db, "s1", project="/other")
        await s.append(Message(role="user", content=[TextBlock(text="msg")]))
        await s.close()

        sessions = await store.list_sessions()
        await store.close()
        assert sessions == []


class TestContextTokens:
    async def test_context_tokens_default_zero(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        assert await store.get_context_tokens() == (0, 0)
        await store.close()

    async def test_set_get_context_tokens(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.set_context_tokens(100, 200)
        assert await store.get_context_tokens() == (100, 200)
        await store.close()

    async def test_context_tokens_persist(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        store = SQLiteContextStore(db, "s1")
        await store.set_context_tokens(500, 300)
        await store.close()

        store2 = SQLiteContextStore(db, "s1")
        assert await store2.get_context_tokens() == (500, 300)
        await store2.close()

    async def test_add_context_tokens(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.add_context_tokens(100, 200)
        assert await store.get_context_tokens() == (100, 200)
        await store.close()

    async def test_add_context_tokens_accumulates(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.add_context_tokens(10, 20)
        await store.add_context_tokens(30, 40)
        assert await store.get_context_tokens() == (40, 60)
        await store.close()

    async def test_clear_resets_context_tokens(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.set_context_tokens(100, 200)
        await store.clear()
        assert await store.get_context_tokens() == (0, 0)
        await store.close()

    async def test_fork_copies_context_tokens(self, tmp_path: Path) -> None:
        store = SQLiteContextStore(_db(tmp_path), "s1")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        await store.set_context_tokens(100, 200)
        child = await store.fork()
        assert await child.get_context_tokens() == (100, 200)
        await child.close()
        await store.close()

    async def test_list_sessions_includes_tokens(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        proj = "/test/project"
        store = SQLiteContextStore(db, "s1", project=proj)
        await store.append(Message(role="user", content=[TextBlock(text="hello")]))
        await store.set_context_tokens(1000, 500)
        await store.close()

        query = SQLiteContextStore(db, "any", project=proj)
        sessions = await query.list_sessions()
        await query.close()
        assert len(sessions) == 1
        assert sessions[0].input_tokens == 1000
        assert sessions[0].output_tokens == 500
