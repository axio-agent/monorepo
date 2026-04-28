"""Tests for AutoCompactStore."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from axio.blocks import TextBlock
from axio.compaction import AutoCompactStore
from axio.context import MemoryContextStore
from axio.events import StreamEvent
from axio.messages import Message
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str = "user", text: str = "x") -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


def _fill(store: MemoryContextStore, n: int) -> None:
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        store._history.append(_msg(role, f"msg-{i}"))


class FailTransport:
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        raise RuntimeError("boom")


@dataclass
class _FakeModel:
    context_window: int


@dataclass
class _FakeTransport:
    model: _FakeModel

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    async def test_append_get_history(self) -> None:
        inner = MemoryContextStore()
        store = AutoCompactStore(inner, StubTransport([]), max_tokens=999_999)
        msg = _msg()
        await store.append(msg)
        assert await store.get_history() == [msg]
        assert await inner.get_history() == [msg]

    async def test_session_id_delegates(self) -> None:
        inner = MemoryContextStore()
        store = AutoCompactStore(inner, StubTransport([]), max_tokens=999_999)
        assert store.session_id == inner.session_id

    async def test_clear_delegates(self) -> None:
        inner = MemoryContextStore()
        store = AutoCompactStore(inner, StubTransport([]), max_tokens=999_999)
        await store.append(_msg())
        await store.clear()
        assert await inner.get_history() == []

    async def test_close_delegates(self) -> None:
        closed: list[bool] = []

        class TrackClose(MemoryContextStore):
            async def close(self) -> None:
                closed.append(True)

        store = AutoCompactStore(TrackClose(), StubTransport([]), max_tokens=999_999)
        await store.close()
        assert closed == [True]

    async def test_get_set_context_tokens(self) -> None:
        inner = MemoryContextStore()
        store = AutoCompactStore(inner, StubTransport([]), max_tokens=999_999)
        await store.set_context_tokens(100, 50)
        assert await store.get_context_tokens() == (100, 50)
        assert await inner.get_context_tokens() == (100, 50)


# ---------------------------------------------------------------------------
# Threshold / trigger
# ---------------------------------------------------------------------------


class TestThreshold:
    async def test_no_compact_below_threshold(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 20)
        transport = StubTransport([make_text_response("summary")])
        store = AutoCompactStore(inner, transport, max_tokens=100_000, keep_recent=4)

        await store.add_context_tokens(50_000, 1_000)

        # history unchanged
        assert len(await inner.get_history()) == 20

    async def test_compact_triggers_above_threshold(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 20)
        transport = StubTransport([make_text_response("summary")])
        store = AutoCompactStore(inner, transport, max_tokens=100_000, keep_recent=4)

        await store.add_context_tokens(110_000, 1_000)

        # history compacted: 2 (summary pair) + 4 recent
        assert len(await inner.get_history()) == 6

    async def test_compact_rewrites_history(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 14)
        transport = StubTransport([make_text_response("Earlier work summary")])
        store = AutoCompactStore(inner, transport, max_tokens=1, keep_recent=4)

        await store.add_context_tokens(2, 0)

        history = await inner.get_history()
        assert history[0].role == "user"
        block0 = history[0].content[0]
        assert isinstance(block0, TextBlock)
        assert "Earlier work summary" in block0.text
        assert history[1].role == "assistant"
        block1 = history[1].content[0]
        assert isinstance(block1, TextBlock)
        assert "Understood" in block1.text
        # last 4 messages preserved verbatim
        assert history[2:] == (await MemoryContextStore(history[2:]).get_history())

    async def test_compact_reads_from_fork(self) -> None:
        """compact_context is called with a fork of the inner store, not the live store."""
        inner = MemoryContextStore()
        _fill(inner, 14)

        seen_ids: list[str] = []

        class RecordingTransport:
            # records the session_id of whatever context compact_context used
            async def run_compact(self, ctx: MemoryContextStore) -> str:
                seen_ids.append(ctx.session_id)
                return "summary"

            def stream(
                self, messages: list[Message], tools: list[Tool[Any]], system: str
            ) -> AsyncIterator[StreamEvent]:
                raise NotImplementedError

        # patch compact_context to capture the store it receives
        import axio.compaction as _mod

        original = _mod.compact_context

        async def _capturing(ctx, transport, **kw):  # type: ignore[no-untyped-def]
            seen_ids.append(ctx.session_id)
            return await original(ctx, StubTransport([make_text_response("summary")]), **kw)

        _mod.compact_context = _capturing  # type: ignore[assignment]  # noqa: SIM117
        try:
            store = AutoCompactStore(
                inner, StubTransport([make_text_response("summary")]), max_tokens=1, keep_recent=4
            )
            await store.add_context_tokens(2, 0)
        finally:
            _mod.compact_context = original

        assert len(seen_ids) == 1
        # the captured session_id must differ from inner - it was a fork
        assert seen_ids[0] != inner.session_id

    async def test_cumulative_tokens_preserved_after_compact(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 14)
        await inner.set_context_tokens(500_000, 20_000)
        transport = StubTransport([make_text_response("summary")])
        store = AutoCompactStore(inner, transport, max_tokens=1, keep_recent=4)

        await store.add_context_tokens(2, 0)

        # cumulative tokens must survive compaction (they are totals, not context size)
        in_tok, out_tok = await inner.get_context_tokens()
        assert in_tok == 500_002
        assert out_tok == 20_000

    async def test_compact_failure_preserves_history(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 14)
        store = AutoCompactStore(inner, FailTransport(), max_tokens=1, keep_recent=4)

        await store.add_context_tokens(2, 0)

        # history intact
        assert len(await inner.get_history()) == 14

    async def test_no_compact_when_history_too_short(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 3)  # less than keep_recent=6 → split <= 0
        transport = StubTransport([make_text_response("summary")])
        store = AutoCompactStore(inner, transport, max_tokens=1, keep_recent=6)

        await store.add_context_tokens(2, 0)

        assert len(await inner.get_history()) == 3


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


class TestFork:
    async def test_fork_returns_auto_compact_store(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 4)
        store = AutoCompactStore(inner, StubTransport([]), max_tokens=77_000, keep_recent=3)

        forked = await store.fork()

        assert isinstance(forked, AutoCompactStore)
        assert forked._max_tokens == 77_000
        assert forked._keep_recent == 3

    async def test_fork_isolation(self) -> None:
        inner = MemoryContextStore()
        _fill(inner, 4)
        store = AutoCompactStore(inner, StubTransport([]), max_tokens=999_999)

        forked = await store.fork()
        await forked.append(_msg(text="extra"))

        # parent unaffected
        assert len(await store.get_history()) == 4
        assert len(await forked.get_history()) == 5

    async def test_fork_session_id_differs(self) -> None:
        inner = MemoryContextStore()
        store = AutoCompactStore(inner, StubTransport([]), max_tokens=999_999)

        forked = await store.fork()

        assert forked.session_id != store.session_id


# ---------------------------------------------------------------------------
# Threshold derivation from model
# ---------------------------------------------------------------------------


class TestThresholdDerivation:
    def test_threshold_from_model_context_window(self) -> None:
        transport = _FakeTransport(model=_FakeModel(context_window=200_000))
        store = AutoCompactStore(MemoryContextStore(), transport)
        assert store._max_tokens == 150_000

    def test_explicit_max_tokens_overrides_model(self) -> None:
        transport = _FakeTransport(model=_FakeModel(context_window=200_000))
        store = AutoCompactStore(MemoryContextStore(), transport, max_tokens=50_000)
        assert store._max_tokens == 50_000

    def test_fallback_when_no_model_attr(self) -> None:
        transport = StubTransport([])  # no .model attribute
        store = AutoCompactStore(MemoryContextStore(), transport)
        assert store._max_tokens == int(128_000 * 0.75)

    def test_custom_threshold(self) -> None:
        transport = _FakeTransport(model=_FakeModel(context_window=100_000))
        store = AutoCompactStore(MemoryContextStore(), transport, threshold=0.5)
        assert store._max_tokens == 50_000

    async def test_fork_preserves_threshold(self) -> None:
        transport = _FakeTransport(model=_FakeModel(context_window=100_000))
        store = AutoCompactStore(MemoryContextStore(), transport, threshold=0.5)
        forked = await store.fork()
        assert forked._threshold == 0.5
        assert forked._max_tokens == 50_000
