"""Bind execution tools to one sandbox lifetime and verify it offline."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from axio import CONTEXT, Tool
from axio_tools_docker import DockerSandbox


TOOL_OUTPUT_MAX_CHARS = 4_000
TRUNCATION_MARKER = "\n[tool output truncated]"


def bound_tool_output(text: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    if max_chars < len(TRUNCATION_MARKER):
        raise ValueError("max_chars is too small for the truncation marker")
    if len(text) <= max_chars:
        return text
    prefix_length = max_chars - len(TRUNCATION_MARKER)
    return text[:prefix_length] + TRUNCATION_MARKER


def bound_text_tool(
    tool: Tool[Any],
    max_chars: int = TOOL_OUTPUT_MAX_CHARS,
) -> Tool[Any]:
    async def bounded_handler(**kwargs: Any) -> Any:
        result = await tool(**kwargs)
        if isinstance(result, str):
            return bound_tool_output(result, max_chars)
        return result

    return Tool(
        name=tool.name,
        handler=bounded_handler,
        description=tool.description,
        schema=tool.schema,
    )


# [docs:start-isolate-sandbox-factory]
def execution_sandbox() -> DockerSandbox:
    return DockerSandbox(
        image="python:3.12-slim",
        memory="256m",
        cpus="1.0",
        network=False,
        read_only=True,
        cap_drop=["ALL"],
        ulimits={"nofile": (256, 256), "nproc": 128},
        tmpfs={
            "/tmp": "size=64m,mode=1777",
            "/workspace": "size=256m",
        },
        workdir="/workspace",
    )


# [docs:end-isolate-sandbox-factory]


async def render_turn(agent, prompt, context) -> None:
    """Run one turn; a product can replace this with its event renderer."""
    await agent.run(prompt, context)


# [docs:start-isolate-run-turn]
async def run_isolated_turn(
    agent,
    prompt,
    context,
    *,
    sandbox_factory=execution_sandbox,
) -> None:
    async with sandbox_factory() as sandbox:
        execution_tools = [bound_text_tool(tool) for tool in sandbox.tools]
        isolated_agent = agent.copy(
            tools=[*agent.tools, *execution_tools],
        )
        await render_turn(isolated_agent, prompt, context)


# [docs:end-isolate-run-turn]


# [docs:start-isolate-conceptual-shell]
async def conceptual_shell(command: str) -> str:
    sandbox: DockerSandbox = CONTEXT.get()
    return await sandbox.exec(command)


# [docs:end-isolate-conceptual-shell]


# [docs:start-isolate-docker-check]
async def check_docker_boundary() -> None:
    docker_url = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    async with DockerSandbox(
        docker_url,
        image="python:3.12-slim",
        memory="256m",
        cpus="1.0",
        network=False,
    ) as sandbox:
        tools = {tool.name: tool for tool in sandbox.tools}
        assert tools["shell"].context is sandbox
        result = await tools["shell"](command="printf sandbox-ok")
        assert result == "sandbox-ok"


# [docs:end-isolate-docker-check]


async def _fake_shell(command: str) -> str:
    sandbox = CONTEXT.get()
    sandbox.commands.append(command)
    return "x" * 100


class _FakeSandbox:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False
        self.commands: list[str] = []
        self.tools = (Tool(name="shell", handler=_fake_shell, context=self),)

    async def __aenter__(self) -> _FakeSandbox:
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    async def exec(self, command: str) -> str:
        self.commands.append(command)
        return "sandbox output"


class _FakeAgent:
    def __init__(self, tools: list[Tool[Any]]) -> None:
        self.tools = tools
        self.copies: list[_FakeAgent] = []
        self.runs: list[tuple[str, object]] = []

    def copy(self, *, tools: list[Tool[Any]]) -> _FakeAgent:
        copied = _FakeAgent(tools)
        self.copies.append(copied)
        return copied

    async def run(self, prompt: str, context: object) -> str:
        self.runs.append((prompt, context))
        return "done"


async def verify_example() -> None:
    candidate = execution_sandbox()
    assert candidate.network is False
    assert candidate.memory == "256m"
    assert candidate.cpus == "1.0"

    async def read_document(path: str) -> str:
        return path

    document_tool = Tool(name="read_document", handler=read_document)
    agent = _FakeAgent([document_tool])
    sandbox = _FakeSandbox()
    await run_isolated_turn(
        agent,
        "Read README.md",
        "project-context",
        sandbox_factory=lambda: sandbox,
    )

    assert sandbox.entered and sandbox.closed
    assert [tool.name for tool in agent.tools] == ["read_document"]
    assert len(agent.copies) == 1
    copied = agent.copies[0]
    assert [tool.name for tool in copied.tools] == ["read_document", "shell"]
    assert copied.runs == [("Read README.md", "project-context")]
    assert len(await copied.tools[1](command="printf isolated")) == 100
    assert sandbox.commands == ["printf isolated"]

    token = CONTEXT.set(sandbox)
    try:
        result = await conceptual_shell("printf conceptual")
    finally:
        CONTEXT.reset(token)
    assert result == "sandbox output"
    assert sandbox.commands[-1] == "printf conceptual"


async def main() -> None:
    await verify_example()
    if "--docker" in sys.argv[1:]:
        await check_docker_boundary()


if __name__ == "__main__":
    asyncio.run(main())
