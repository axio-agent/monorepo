"""Tests for ContextStore — MemoryContextStore, ABC classmethods, compact_context."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from axio.blocks import TextBlock, ToolResultBlock
from axio.context import ContextStore, MemoryContextStore, _find_safe_boundary, compact_context
from axio.events import StreamEvent
from axio.messages import Message
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool


class TestMemoryContextStore:
    async def test_append_and_get_history(self) -> None:
        store = MemoryContextStore()
        msg = Message(role="user", content=[TextBlock(text="hi")])
        await store.append(msg)
        history = await store.get_history()
        assert len(history) == 1
        assert history[0].role == "user"

    async def test_ordering(self) -> None:
        store = MemoryContextStore()
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        await store.append(Message(role="assistant", content=[TextBlock(text="2")]))
        history = await store.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    async def test_clear(self) -> None:
        store = MemoryContextStore()
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        await store.clear()
        assert await store.get_history() == []

    async def test_fork_returns_copy(self) -> None:
        store = MemoryContextStore()
        msg = Message(role="user", content=[TextBlock(text="hi")])
        await store.append(msg)
        child = await store.fork()
        child_history = await child.get_history()
        assert len(child_history) == 1

    async def test_fork_isolation_child_to_parent(self) -> None:
        """C4: mutations to child don't affect parent."""
        store = MemoryContextStore()
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        child = await store.fork()
        await child.append(Message(role="user", content=[TextBlock(text="2")]))
        parent_history = await store.get_history()
        assert len(parent_history) == 1

    async def test_fork_isolation_parent_to_child(self) -> None:
        """C4: mutations to parent don't affect child."""
        store = MemoryContextStore()
        await store.append(Message(role="user", content=[TextBlock(text="1")]))
        child = await store.fork()
        await store.append(Message(role="user", content=[TextBlock(text="2")]))
        child_history = await child.get_history()
        assert len(child_history) == 1

    async def test_context_tokens_default_zero(self) -> None:
        store = MemoryContextStore()
        assert await store.get_context_tokens() == (0, 0)

    async def test_set_get_context_tokens(self) -> None:
        store = MemoryContextStore()
        await store.set_context_tokens(100, 200)
        assert await store.get_context_tokens() == (100, 200)

    async def test_add_context_tokens(self) -> None:
        store = MemoryContextStore()
        await store.add_context_tokens(100, 200)
        assert await store.get_context_tokens() == (100, 200)

    async def test_add_context_tokens_accumulates(self) -> None:
        store = MemoryContextStore()
        await store.add_context_tokens(10, 20)
        await store.add_context_tokens(30, 40)
        assert await store.get_context_tokens() == (40, 60)

    async def test_clear_resets_context_tokens(self) -> None:
        store = MemoryContextStore()
        await store.set_context_tokens(100, 200)
        await store.clear()
        assert await store.get_context_tokens() == (0, 0)

    async def test_fork_copies_context_tokens(self) -> None:
        store = MemoryContextStore()
        await store.set_context_tokens(100, 200)
        child = await store.fork()
        assert await child.get_context_tokens() == (100, 200)
        # Mutation isolation
        await child.set_context_tokens(300, 400)
        assert await store.get_context_tokens() == (100, 200)


class _ConcreteContextStore(ContextStore):
    """Minimal ABC subclass for testing classmethods."""

    def __init__(self) -> None:
        self._session_id = "test"
        self._history: list[Message] = []
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    async def append(self, message: Message) -> None:
        self._history.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._history)

    async def clear(self) -> None:
        self._history.clear()
        self._input_tokens = 0
        self._output_tokens = 0

    async def fork(self) -> _ConcreteContextStore:
        new = _ConcreteContextStore()
        new._history = list(self._history)
        new._input_tokens = self._input_tokens
        new._output_tokens = self._output_tokens
        return new

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    async def get_context_tokens(self) -> tuple[int, int]:
        return self._input_tokens, self._output_tokens

    async def close(self) -> None:
        pass


class TestContextStoreABC:
    async def test_from_history(self) -> None:
        msgs = [
            Message(role="user", content=[TextBlock(text="hello")]),
            Message(role="assistant", content=[TextBlock(text="hi")]),
        ]
        store = await _ConcreteContextStore.from_history(msgs)
        assert isinstance(store, _ConcreteContextStore)
        history = await store.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    async def test_from_history_empty(self) -> None:
        store = await _ConcreteContextStore.from_history([])
        assert await store.get_history() == []

    async def test_from_context(self) -> None:
        source = _ConcreteContextStore()
        await source.append(Message(role="user", content=[TextBlock(text="msg1")]))
        await source.append(Message(role="assistant", content=[TextBlock(text="msg2")]))
        copy = await _ConcreteContextStore.from_context(source)
        assert isinstance(copy, _ConcreteContextStore)
        history = await copy.get_history()
        assert len(history) == 2

    async def test_from_context_isolation(self) -> None:
        source = _ConcreteContextStore()
        await source.append(Message(role="user", content=[TextBlock(text="original")]))
        copy = await _ConcreteContextStore.from_context(source)
        await copy.append(Message(role="user", content=[TextBlock(text="extra")]))
        assert len(await source.get_history()) == 1
        assert len(await copy.get_history()) == 2


def _fill_store(store: MemoryContextStore, n: int) -> None:
    """Append *n* user/assistant message pairs to *store* synchronously."""
    for i in range(n):
        role: Literal["user", "assistant"] = "user" if i % 2 == 0 else "assistant"
        store._history.append(Message(role=role, content=[TextBlock(text=f"msg-{i}")]))


class TestCompactContext:
    async def test_no_compaction_below_threshold(self) -> None:
        store = MemoryContextStore()
        _fill_store(store, 10)
        transport = StubTransport([make_text_response("summary")])
        result = await compact_context(store, transport, max_messages=20)
        assert result is None
        assert len(await store.get_history()) == 10

    async def test_compaction_returns_messages(self) -> None:
        store = MemoryContextStore()
        _fill_store(store, 22)
        transport = StubTransport([make_text_response("summary")])
        messages = await compact_context(store, transport, max_messages=20, keep_recent=6)
        assert messages is not None
        # 2 (summary user + ack assistant) + 6 recent = 8
        assert len(messages) == 8
        assert messages[0].role == "user"
        assert isinstance(messages[0].content[0], TextBlock)
        assert "summary" in messages[0].content[0].text
        assert messages[1].role == "assistant"
        assert isinstance(messages[1].content[0], TextBlock)
        assert "Understood" in messages[1].content[0].text

    async def test_compaction_does_not_mutate_original(self) -> None:
        store = MemoryContextStore()
        _fill_store(store, 22)
        transport = StubTransport([make_text_response("summary")])
        await compact_context(store, transport, max_messages=20, keep_recent=6)
        assert len(await store.get_history()) == 22

    async def test_safe_boundary_skips_tool_result(self) -> None:
        store = MemoryContextStore()
        _fill_store(store, 18)
        # Place a tool_result message right at the default split point
        split_idx = 18 - 6  # = 12
        store._history[split_idx] = Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
        )
        result = _find_safe_boundary(store._history, keep_recent=6)
        assert result < split_idx

    async def test_no_recompaction_after_populating_new_store(self) -> None:
        store = MemoryContextStore()
        _fill_store(store, 22)
        transport = StubTransport([make_text_response("summary")])
        messages = await compact_context(store, transport, max_messages=20, keep_recent=6)
        assert messages is not None
        # Populate a fresh store with compacted messages
        new_store = MemoryContextStore(messages)
        # 8 messages — below threshold of 20, no second compaction
        result = await compact_context(new_store, transport, max_messages=20, keep_recent=6)
        assert result is None

    async def test_compaction_failure_returns_none(self) -> None:
        store = MemoryContextStore()
        _fill_store(store, 22)

        class FailTransport:
            def stream(self, messages: list[Message], tools: list[Tool], system: str) -> AsyncIterator[StreamEvent]:
                raise RuntimeError("boom")

        result = await compact_context(store, FailTransport(), max_messages=20, keep_recent=6)
        assert result is None
        assert len(await store.get_history()) == 22
