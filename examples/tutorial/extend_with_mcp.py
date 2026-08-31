"""Load MCP tools with explicit connection scope and offline verification."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from axio import Agent, MemoryContextStore, Tool
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio_tools_mcp import MCPServerConfig, load_mcp_tools


# [docs:start-mcp-load-tools]
async def query_documentation(
    load_tools=load_mcp_tools,
    agent_factory=Agent,
) -> None:
    mcp_tools, sessions = await load_tools(
        [
            MCPServerConfig(
                name="docs",
                command="mcp-server-docs",
                args=["--root", "."],
            ),
        ]
    )
    async with AsyncExitStack() as resources:
        for session in sessions:
            resources.push_async_callback(session.close)

        transport = StubTransport(
            [
                make_tool_use_response(
                    "docs__search",
                    tool_input={"query": "context compaction"},
                ),
                make_text_response("The context guide describes AutoCompactStore."),
            ]
        )
        agent = agent_factory(
            system="Use the documentation tool before answering.",
            transport=transport,
            tools=mcp_tools,
        )

        reply = await agent.run(
            "How does context compaction work?",
            MemoryContextStore(),
        )
        assert reply.startswith("The context guide")
        assert [tool.name for tool in mcp_tools] == ["docs__search"]


# [docs:end-mcp-load-tools]


# [docs:start-mcp-shared-scope]
async def serve_with_shared_mcp(
    server_configs,
    base_agent,
    context_factory,
    sandbox_factory,
    serve,
    harness_type,
    *,
    load_tools=load_mcp_tools,
) -> None:
    mcp_tools, mcp_sessions = await load_tools(server_configs)
    async with AsyncExitStack() as resources:
        for session in mcp_sessions:
            resources.push_async_callback(session.close)

        prototype = base_agent.copy(tools=[*base_agent.tools, *mcp_tools])
        harness = harness_type(prototype, context_factory, sandbox_factory)
        resources.push_async_callback(harness.close)
        await serve(harness)


# [docs:end-mcp-shared-scope]


class _FakeSession:
    def __init__(self, events: list[str] | None = None) -> None:
        self.is_connected = True
        self.events = events

    async def close(self) -> None:
        self.is_connected = False
        if self.events is not None:
            self.events.append("mcp closed")


async def _search_docs(query: str) -> str:
    """Search the documentation service."""
    return f"result for {query}"


async def _fake_load_mcp_tools(servers):
    assert servers[0].name == "docs"
    return [Tool(name="docs__search", handler=_search_docs)], [_FakeSession()]


@dataclass(frozen=True, slots=True)
class _ScopeTool:
    name: str


class _ScopeAgent:
    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools

    def copy(self, *, tools: list[Any]) -> _ScopeAgent:
        return _ScopeAgent(tools)


class _ScopeHarness:
    instances: list[_ScopeHarness] = []
    events: list[str] = []

    def __init__(self, prototype, context_factory, sandbox_factory) -> None:
        self.prototype = prototype
        self.instances.append(self)

    async def close(self) -> None:
        self.events.append("harness closed")


async def _fake_scope_loader(servers):
    assert len(servers) == 1
    return [_ScopeTool("docs__search")], [_FakeSession(_ScopeHarness.events)]


async def _fake_serve(harness: _ScopeHarness) -> None:
    assert [tool.name for tool in harness.prototype.tools] == [
        "read_document",
        "docs__search",
    ]
    harness.events.append("served")


async def verify_failure_cleanup() -> None:
    agent_session = _FakeSession()

    async def load_for_agent_failure(servers):
        assert len(servers) == 1
        return [], [agent_session]

    def fail_agent_factory(**kwargs: Any) -> Agent:
        raise RuntimeError("agent construction failed")

    try:
        await query_documentation(
            load_tools=load_for_agent_failure,
            agent_factory=fail_agent_factory,
        )
    except RuntimeError as error:
        assert str(error) == "agent construction failed"
    else:
        raise AssertionError("Agent construction must fail")
    assert not agent_session.is_connected

    copy_session = _FakeSession()

    async def load_for_copy_failure(servers):
        assert len(servers) == 1
        return [], [copy_session]

    class CopyFailureAgent:
        tools: list[Any] = []

        def copy(self, *, tools: list[Any]) -> CopyFailureAgent:
            raise RuntimeError("agent copy failed")

    try:
        await serve_with_shared_mcp(
            server_configs=[object()],
            base_agent=CopyFailureAgent(),
            context_factory=object(),
            sandbox_factory=object(),
            serve=_fake_serve,
            harness_type=_ScopeHarness,
            load_tools=load_for_copy_failure,
        )
    except RuntimeError as error:
        assert str(error) == "agent copy failed"
    else:
        raise AssertionError("Agent copy must fail")
    assert not copy_session.is_connected

    construction_session = _FakeSession()

    async def load_for_construction_failure(servers):
        assert len(servers) == 1
        return [], [construction_session]

    class ConstructionFailureHarness:
        def __init__(self, prototype, context_factory, sandbox_factory) -> None:
            raise RuntimeError("harness construction failed")

    try:
        await serve_with_shared_mcp(
            server_configs=[object()],
            base_agent=_ScopeAgent([_ScopeTool("read_document")]),
            context_factory=object(),
            sandbox_factory=object(),
            serve=_fake_serve,
            harness_type=ConstructionFailureHarness,
            load_tools=load_for_construction_failure,
        )
    except RuntimeError as error:
        assert str(error) == "harness construction failed"
    else:
        raise AssertionError("Harness construction must fail")
    assert not construction_session.is_connected

    close_sessions = [_FakeSession(), _FakeSession()]

    async def load_for_close_failure(servers):
        assert len(servers) == 1
        return [_ScopeTool("docs__search")], close_sessions

    class CloseFailureHarness(_ScopeHarness):
        async def close(self) -> None:
            raise RuntimeError("harness close failed")

    try:
        await serve_with_shared_mcp(
            server_configs=[object()],
            base_agent=_ScopeAgent([_ScopeTool("read_document")]),
            context_factory=object(),
            sandbox_factory=object(),
            serve=_fake_serve,
            harness_type=CloseFailureHarness,
            load_tools=load_for_close_failure,
        )
    except RuntimeError as error:
        assert str(error) == "harness close failed"
    else:
        raise AssertionError("Harness close must fail")
    assert all(not session.is_connected for session in close_sessions)


async def verify_example() -> None:
    await query_documentation(load_tools=_fake_load_mcp_tools)

    _ScopeHarness.instances.clear()
    _ScopeHarness.events.clear()
    await serve_with_shared_mcp(
        server_configs=[object()],
        base_agent=_ScopeAgent([_ScopeTool("read_document")]),
        context_factory=object(),
        sandbox_factory=object(),
        serve=_fake_serve,
        harness_type=_ScopeHarness,
        load_tools=_fake_scope_loader,
    )
    assert len(_ScopeHarness.instances) == 1
    assert _ScopeHarness.events == ["served", "harness closed", "mcp closed"]
    await verify_failure_cleanup()


if __name__ == "__main__":
    asyncio.run(verify_example())
