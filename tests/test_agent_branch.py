"""Tests for Agent copy via copy.copy: shallow clone, field overrides."""

from __future__ import annotations

import copy

from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool, ToolHandler


class HandlerA(ToolHandler):
    async def __call__(self) -> str:
        return "a"


class HandlerB(ToolHandler):
    async def __call__(self) -> str:
        return "b"


class NoopHandler(ToolHandler):
    async def __call__(self) -> str:
        return ""


class TestAgentCopy:
    def test_creates_new_agent(self) -> None:
        agent = Agent(system="test", tools=[], transport=StubTransport())
        clone = copy.copy(agent)
        assert clone is not agent

    def test_shares_tools(self) -> None:
        tool = Tool(name="t", description="t", handler=NoopHandler)
        agent = Agent(system="test", tools=[tool], transport=StubTransport())
        clone = copy.copy(agent)
        assert clone.tools is agent.tools

    def test_shares_transport(self) -> None:
        transport = StubTransport()
        agent = Agent(system="test", tools=[], transport=transport)
        clone = copy.copy(agent)
        assert clone.transport is transport

    def test_override_system(self) -> None:
        agent = Agent(system="original", tools=[], transport=StubTransport())
        clone = copy.copy(agent)
        clone.system = "overridden"
        assert clone.system == "overridden"
        assert agent.system == "original"

    def test_override_tools(self) -> None:
        tool_a = Tool(name="a", description="a", handler=HandlerA)
        tool_b = Tool(name="b", description="b", handler=HandlerB)
        agent = Agent(system="test", tools=[tool_a], transport=StubTransport())
        clone = copy.copy(agent)
        clone.tools = [tool_b]
        assert clone.tools[0].name == "b"
        assert agent.tools[0].name == "a"

    async def test_clone_runs_independently(self) -> None:
        transport = StubTransport([make_text_response("cloned")])
        agent = Agent(system="test", tools=[], transport=transport)
        clone = copy.copy(agent)
        result = await clone.run("go", MemoryContextStore())
        assert result == "cloned"
