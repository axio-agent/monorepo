from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from io import StringIO
from typing import Annotated, Any, TextIO

from axio import (
    Agent,
    CompletionTransport,
    ContextStore,
    Field,
    GuardError,
    MemoryContextStore,
    PermissionGuard,
    StopReason,
    StreamEvent,
    Tool,
    ToolResultBlock,
    Usage,
)
from axio.events import Error, SessionEndEvent, TextDelta, ToolResult, ToolUseStart
from axio.testing import StubTransport, make_text_response, make_tool_use_response


# [docs:start-stream-render-turn]
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
                    print(
                        f"[{label}] {name}: {len(content)} characters",
                        file=stderr,
                    )
                case Error(exception=exception):
                    print(
                        f"[error] {type(exception).__name__}",
                        file=stderr,
                    )
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


# [docs:end-stream-render-turn]


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


@dataclass(frozen=True, slots=True)
class DocumentAccessGuard(PermissionGuard):
    allowed_paths: frozenset[str]

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        path = str(kwargs["path"])
        if path not in self.allowed_paths:
            raise GuardError(f"{tool.name} denied access to {path}")
        return {**kwargs, "path": path}


async def read_document(
    path: Annotated[
        str, Field(description="Exact project document path, for example README.md")
    ],
) -> str:
    """Return one project document summary from its exact path."""
    document = DOCUMENTS.get(path)
    if document is None:
        return f"Document {path} was not found."
    return f"{path}: {document['summary']}"


async def search_documents(
    query: Annotated[
        str, Field(description="Words to match in public document summaries")
    ],
    limit: Annotated[
        int, Field(description="Maximum summaries", default=5, ge=1, le=10)
    ] = 5,
) -> str:
    """Search public project document summaries."""
    needle = query.casefold()
    matches = [
        f"{path}: {document['summary']}"
        for path, document in DOCUMENTS.items()
        if document["visibility"] == "public"
        and needle in f"{path} {document['summary']}".casefold()
    ]
    return "\n".join(matches[:limit]) or "No public documents matched."


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
        system="Answer questions about project documents. Use tools for facts.",
        transport=transport,
        tools=[read_document_tool, search_documents_tool],
    )


class FailingTransport:
    async def stream(
        self,
        messages: object,
        tools: object,
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        raise RuntimeError("private provider detail")
        yield TextDelta(index=0, delta="unreachable")


async def verify_error_rendering() -> None:
    stdout = StringIO()
    stderr = StringIO()
    ending = await render_turn(
        Agent(system="fail", transport=FailingTransport()),
        "Trigger a provider failure.",
        MemoryContextStore(),
        stdout=stdout,
        stderr=stderr,
    )
    assert ending.stop_reason is StopReason.error
    assert "[error] RuntimeError" in stderr.getvalue()
    assert "private provider detail" not in stderr.getvalue()


# [docs:start-stream-run-turn]
async def run_scripted_turn() -> None:
    transport = StubTransport(
        [
            make_tool_use_response(
                "read_document",
                tool_input={"path": "README.md"},
            ),
            make_text_response("README.md: Project overview."),
        ]
    )
    context = MemoryContextStore()
    stdout = StringIO()
    stderr = StringIO()

    ending = await render_turn(
        build_agent(transport),
        "Summarize README.md.",
        context,
        stdout=stdout,
        stderr=stderr,
    )

    history = await context.get_history()
    results = [
        block
        for message in history
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert ending.total_usage == Usage(input_tokens=20, output_tokens=10)
    assert results[0].content == "README.md: Project overview"
    assert stdout.getvalue() == "README.md: Project overview.\n"
    assert "[tool] read_document" in stderr.getvalue()
    assert "[tool result] read_document" in stderr.getvalue()

    sys.stdout.write(stdout.getvalue())
    sys.stderr.write(stderr.getvalue())


# [docs:end-stream-run-turn]


async def main() -> None:
    await run_scripted_turn()
    await verify_error_rendering()


if __name__ == "__main__":
    asyncio.run(main())
