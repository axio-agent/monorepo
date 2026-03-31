"""Tests for axio-tui — TUI-specific tool handlers."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import StreamEvent
from axio.messages import Message
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool

from axio_tui.tools import Confirm, StatusLine, SubAgent, VisionAnalyze


class TestStatusLine:
    async def test_returns_ok(self) -> None:
        handler = StatusLine(message="working")
        assert await handler() == "ok"


class TestConfirm:
    async def test_returns_verdict(self) -> None:
        handler = Confirm(verdict="SAFE", reason="harmless", category="read")
        assert await handler() == "SAFE"

    async def test_repr(self) -> None:
        handler = Confirm(verdict="DENY", reason="bad", category="exec")
        r = repr(handler)
        assert "DENY" in r
        assert "exec" in r


class TestSubAgent:
    def _make_agent(self, transport: StubTransport, tools: list[Tool] | None = None) -> Agent:
        return Agent(system="test", tools=tools or [], transport=transport, max_iterations=5)

    async def test_returns_subagent_result(self) -> None:
        transport = StubTransport([make_text_response("sub-result")])
        agent = self._make_agent(transport)

        async def factory() -> tuple[Agent, MemoryContextStore]:
            return agent, MemoryContextStore()

        SubAgent._factory = factory
        try:
            handler = SubAgent(task="do something")
            result = await handler()
            assert result == "sub-result"
        finally:
            SubAgent._factory = None

    async def test_no_factory_returns_error(self) -> None:
        SubAgent._factory = None
        handler = SubAgent(task="do something")
        result = await handler()
        assert result == "SubAgent is not configured"

    async def test_context_forking(self) -> None:
        """Factory receives a snapshot of parent context."""
        parent = MemoryContextStore()
        from axio.blocks import TextBlock
        from axio.messages import Message

        await parent.append(Message(role="user", content=[TextBlock(text="hello")]))

        transport = StubTransport([make_text_response("ok")])
        agent = self._make_agent(transport)
        received_context: list[MemoryContextStore] = []

        async def factory() -> tuple[Agent, MemoryContextStore]:
            ctx = await MemoryContextStore.from_context(parent)
            received_context.append(ctx)
            return agent, ctx

        SubAgent._factory = factory
        try:
            handler = SubAgent(task="check context")
            await handler()
            assert len(received_context) == 1
            history = await received_context[0].get_history()
            assert history[0].content[0].text == "hello"  # type: ignore[attr-defined]
            parent_history = await parent.get_history()
            assert len(parent_history) == 1
        finally:
            SubAgent._factory = None

    async def test_error_propagates(self) -> None:
        async def factory() -> tuple[Agent, MemoryContextStore]:
            raise RuntimeError("factory failed")

        SubAgent._factory = factory
        try:
            handler = SubAgent(task="boom")
            with pytest.raises(RuntimeError, match="factory failed"):
                await handler()
        finally:
            SubAgent._factory = None

    async def test_integration_full_loop(self) -> None:
        """Parent agent calls subagent tool via full agent loop."""
        sub_transport = StubTransport([make_text_response("sub-answer")])
        sub_agent = self._make_agent(sub_transport)

        async def factory() -> tuple[Agent, MemoryContextStore]:
            return sub_agent, MemoryContextStore()

        SubAgent._factory = factory
        try:
            parent_transport = StubTransport(
                [
                    make_tool_use_response(
                        tool_name="subagent",
                        tool_id="call_1",
                        tool_input={"task": "research X"},
                    ),
                    make_text_response("Final answer based on sub-agent"),
                ]
            )
            subagent_tool = Tool(
                name="subagent",
                description="Delegate a task",
                handler=SubAgent,
                concurrency=3,
            )
            parent_agent = self._make_agent(parent_transport, tools=[subagent_tool])
            result = await parent_agent.run("delegate something", MemoryContextStore())
            assert result == "Final answer based on sub-agent"
        finally:
            SubAgent._factory = None

    async def test_repr(self) -> None:
        handler = SubAgent(task="analyze the logs")
        r = repr(handler)
        assert "analyze the logs" in r


# Minimal valid 1x1 red PNG (67 bytes)
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"  # signature
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestVisionAnalyze:
    async def test_no_transport_returns_error(self) -> None:
        VisionAnalyze._transport = None
        handler = VisionAnalyze(path="img.png")
        result = await handler()
        assert "not configured" in result

    async def test_file_not_found(self, tmp_path: Path) -> None:
        VisionAnalyze._transport = StubTransport([make_text_response("ok")])
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            handler = VisionAnalyze(path="missing.png")
            result = await handler()
            assert "File not found" in result
        finally:
            os.chdir(old_cwd)
            VisionAnalyze._transport = None

    async def test_unsupported_format(self, tmp_path: Path) -> None:
        (tmp_path / "file.bmp").write_bytes(b"\x00")
        VisionAnalyze._transport = StubTransport([make_text_response("ok")])
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            handler = VisionAnalyze(path="file.bmp")
            result = await handler()
            assert "Unsupported image format" in result
        finally:
            os.chdir(old_cwd)
            VisionAnalyze._transport = None

    async def test_streams_vision_result(self, tmp_path: Path) -> None:
        (tmp_path / "photo.png").write_bytes(_TINY_PNG)
        transport = StubTransport([make_text_response("A red pixel")])
        VisionAnalyze._transport = transport
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            handler = VisionAnalyze(path="photo.png", prompt="What is this?")
            result = await handler()
            assert result == "A red pixel"
        finally:
            os.chdir(old_cwd)
            VisionAnalyze._transport = None

    async def test_constructs_image_message(self, tmp_path: Path) -> None:
        """Verify that the transport receives a message with TextBlock + ImageBlock."""
        from axio.blocks import ImageBlock, TextBlock

        (tmp_path / "test.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
        captured: list[list[Message]] = []

        class CapturingTransport(StubTransport):
            def stream(self, messages: list[Message], tools: list[Tool], system: str) -> AsyncIterator[StreamEvent]:
                captured.append(messages)
                return super().stream(messages, tools, system)

        transport = CapturingTransport([make_text_response("desc")])
        VisionAnalyze._transport = transport
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            handler = VisionAnalyze(path="test.jpg", prompt="Describe it")
            await handler()
            assert len(captured) == 1
            msg = captured[0][0]
            assert len(msg.content) == 2
            assert isinstance(msg.content[0], TextBlock)
            assert msg.content[0].text == "Describe it"
            assert isinstance(msg.content[1], ImageBlock)
            assert msg.content[1].media_type == "image/jpeg"
        finally:
            os.chdir(old_cwd)
            VisionAnalyze._transport = None

    async def test_repr(self) -> None:
        handler = VisionAnalyze(path="image.png")
        assert "image.png" in repr(handler)
