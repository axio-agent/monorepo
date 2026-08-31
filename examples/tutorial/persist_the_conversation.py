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
    PermissionGuard,
    Tool,
    Usage,
)
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
        str, Field(description="Exact document path, for example README.md")
    ],
) -> str:
    """Return one document from its exact path."""
    document = DOCUMENTS.get(path)
    if document is None:
        return f"Document {path} was not found."
    return f"{path}: {document['summary']}"


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


# [docs:start-persist-store-lifecycle]
DATA_DIRECTORY = Path(os.environ.get("AXIO_TUTORIAL_DATA_DIR", "data"))
DATABASE_PATH = DATA_DIRECTORY / "harness.db"
PROJECT_ID = "project-workbench"
SESSION_ID = os.environ.get("AXIO_SESSION_ID", "local-project-demo")


async def run_persistent_turn(
    prompt: str,
) -> tuple[int, tuple[int, int], SessionEndEvent]:
    connection = await connect(DATABASE_PATH)
    context = SQLiteContextStore(
        connection,
        session_id=SESSION_ID,
        project=PROJECT_ID,
    )
    try:
        message_count = len(await context.get_history())
        previous_usage = await context.get_context_tokens()
        ending = await render_turn(agent, prompt, context)
        return message_count, previous_usage, ending
    finally:
        await context.close()
        await connection.close()


# [docs:end-persist-store-lifecycle]


# [docs:start-persist-resume-session]
async def demonstrate_restart() -> None:
    message_count, previous_usage, ending = await run_persistent_turn(
        "What does README.md say about the agent harness?",
    )

    connection = await connect(DATABASE_PATH)
    resumed = SQLiteContextStore(
        connection,
        session_id=SESSION_ID,
        project=PROJECT_ID,
    )
    try:
        history = await resumed.get_history()
        usage = await resumed.get_context_tokens()
        assert len(history) == message_count + 4
        assert usage == (
            previous_usage[0] + ending.total_usage.input_tokens,
            previous_usage[1] + ending.total_usage.output_tokens,
        )
        assert ending.total_usage == Usage(input_tokens=20, output_tokens=10)

        separate = SQLiteContextStore(
            connection,
            session_id=f"{SESSION_ID}-new",
            project=PROJECT_ID,
        )
        assert await separate.get_history() == []
    finally:
        await resumed.close()
        await connection.close()

    print(f"Resumed {len(history)} messages for {SESSION_ID}.")


# [docs:end-persist-resume-session]


if __name__ == "__main__":
    asyncio.run(demonstrate_restart())
