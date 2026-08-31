from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TextIO

from axio_context_sqlite import SQLiteContextStore, connect

from axio import (
    Agent,
    CompletionTransport,
    ContextStore,
    Field,
    GuardError,
    MemoryContextStore,
    Message,
    PermissionGuard,
    TextBlock,
    Tool,
    Usage,
)
from axio.compaction import AutoCompactStore
from axio.events import Error, SessionEndEvent, TextDelta, ToolResult, ToolUseStart
from axio.testing import StubTransport, make_text_response, make_tool_use_response

DOCUMENTS: dict[str, dict[str, str]] = {
    "README.md": {
        "summary": "Project overview",
        "visibility": "public",
    },
    ".env": {
        "summary": "Local credentials",
        "visibility": "private",
    },
    "docs/architecture.md": {
        "summary": "Agent harness architecture",
        "visibility": "public",
    },
}


# [docs:start-compact-output-bound]
TOOL_OUTPUT_MAX_CHARS = 4_000
TRUNCATION_MARKER = "\n[tool output truncated]"


def bound_tool_output(
    text: str,
    max_chars: int = TOOL_OUTPUT_MAX_CHARS,
) -> str:
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


# [docs:end-compact-output-bound]


@dataclass(frozen=True, slots=True)
class DocumentAccessGuard(PermissionGuard):
    allowed_paths: frozenset[str]

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        path = str(kwargs["path"])
        if path not in self.allowed_paths:
            raise GuardError(f"{tool.name} denied access to {path}")
        return {**kwargs, "path": path}


# [docs:start-compact-read-document]
async def read_document(
    path: Annotated[
        str, Field(description="Exact document path, for example README.md")
    ],
) -> str:
    """Return one document visible to the current user."""
    document = DOCUMENTS.get(path)
    if document is None:
        return f"Document {path} was not found."

    rendered = f"{path}: {document['summary']}"
    return bound_tool_output(rendered)


# [docs:end-compact-read-document]


async def search_documents(
    query: Annotated[str, Field(description="Words to match in public documents")],
    limit: Annotated[
        int, Field(description="Maximum summaries", default=5, ge=1, le=10)
    ] = 5,
) -> str:
    """Search public documents."""
    needle = query.casefold()
    matches = [
        f"{path}: {document['summary']}"
        for path, document in DOCUMENTS.items()
        if document["visibility"] == "public"
        and needle in f"{path} {document['summary']}".casefold()
    ]
    return bound_tool_output(
        "\n".join(matches[:limit]) or "No public documents matched."
    )


read_document_tool = Tool(
    name="read_document",
    handler=read_document,
    description="Read one document by exact path; do not use for discovery.",
    guards=(DocumentAccessGuard(frozenset({"README.md", "docs/architecture.md"})),),
)
search_documents_tool = Tool(
    name="search_documents",
    handler=search_documents,
    description="Search public documents when no exact path is available.",
)


def build_agent(transport: CompletionTransport) -> Agent:
    return Agent(
        system="Answer project questions. Use tools for document facts.",
        transport=transport,
        tools=[read_document_tool, search_documents_tool],
    )


async def render_turn(
    agent: Agent,
    prompt: str,
    context: ContextStore,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> SessionEndEvent:
    stream = agent.run_stream(prompt, context)
    session_end: SessionEndEvent | None = None
    try:
        async for event in stream:
            match event:
                case TextDelta(delta=text):
                    print(text, end="", file=stdout, flush=True)
                case ToolUseStart(name=name):
                    print(f"\n[tool] {name}", file=stderr)
                case ToolResult(name=name, content=content, is_error=is_error):
                    label = "tool error" if is_error else "tool result"
                    print(f"[{label}] {name}: {len(content)} characters", file=stderr)
                case Error(exception=exception):
                    print(f"[error] {type(exception).__name__}", file=stderr)
                case SessionEndEvent() as ending:
                    session_end = ending
                    usage = ending.total_usage
                    print(
                        f"[done] {ending.stop_reason.value}; input={usage.input_tokens}, output={usage.output_tokens}",
                        file=stderr,
                    )
    finally:
        await stream.aclose()

    if session_end is None:
        raise RuntimeError("Agent stream ended without SessionEndEvent")
    print(file=stdout)
    return session_end


transport = StubTransport(
    [
        make_tool_use_response(
            "read_document",
            tool_input={"path": "README.md"},
        ),
        make_text_response("README.md describes the agent harness."),
    ]
)
agent = build_agent(transport)


# [docs:start-compact-store-lifecycle]
DATA_DIRECTORY = Path(os.environ.get("AXIO_TUTORIAL_DATA_DIR", "data"))
DATABASE_PATH = DATA_DIRECTORY / "harness.db"
PROJECT_ID = "project-workbench"
SESSION_ID = os.environ.get("AXIO_SESSION_ID", "local-project-demo")


async def run_compacting_turn(prompt: str) -> SessionEndEvent:
    connection = await connect(DATABASE_PATH)
    base_context = SQLiteContextStore(
        connection,
        session_id=SESSION_ID,
        project=PROJECT_ID,
    )
    context = AutoCompactStore(
        base_context,
        transport,
        keep_recent=6,
        threshold=0.75,
    )
    try:
        return await render_turn(agent, prompt, context)
    finally:
        await context.close()
        await connection.close()


# [docs:end-compact-store-lifecycle]


async def noisy_result(query: str) -> str:
    """Return deliberately large text."""
    return query * 100


async def verify_output_bounds() -> None:
    assert bound_tool_output("short") == "short"
    assert len(bound_tool_output("x" * 5_000)) == TOOL_OUTPUT_MAX_CHARS
    assert bound_tool_output("x" * 5_000).endswith(TRUNCATION_MARKER)

    original = Tool(name="noisy_result", handler=noisy_result)
    bounded = bound_text_tool(original, max_chars=80)
    assert bounded.input_schema == original.input_schema
    assert len(await bounded(query="large")) == 80


# [docs:start-compact-trigger-demo]
async def demonstrate_compaction() -> None:
    base_context = MemoryContextStore()
    original = []
    for index in range(10):
        role = "user" if index % 2 == 0 else "assistant"
        message = Message(
            role=role,
            content=[TextBlock(text=f"project documentation message {index}")],
        )
        original.append(message)
        await base_context.append(message)

    summary_transport = StubTransport(
        [
            make_text_response(
                "Earlier discussion concerned README.md.",
                usage=Usage(input_tokens=20, output_tokens=4),
            )
        ]
    )
    context = AutoCompactStore(
        base_context,
        summary_transport,
        keep_recent=4,
        max_tokens=100,
    )

    await context.add_context_tokens(input_tokens=101, output_tokens=7)

    history = await context.get_history()
    assert len(history) == 6
    assert history[0].content == [
        TextBlock(text="Earlier discussion concerned README.md.")
    ]
    assert history[-4:] == original[-4:]
    assert await context.get_context_tokens() == (101, 7)


# [docs:end-compact-trigger-demo]


async def main() -> None:
    await verify_output_bounds()
    ending = await run_compacting_turn("What does README.md say about the harness?")
    assert ending.total_usage == Usage(input_tokens=20, output_tokens=10)
    await demonstrate_compaction()


if __name__ == "__main__":
    asyncio.run(main())
