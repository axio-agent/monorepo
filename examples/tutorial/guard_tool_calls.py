from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from axio import (
    Agent,
    GuardError,
    MemoryContextStore,
    PermissionGuard,
    Tool,
    ToolResult,
)
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
handler_calls: list[str] = []


async def read_document(path: str) -> str:
    handler_calls.append(path)
    document = DOCUMENTS[path]
    return f"{path}: {document['summary']}"


async def search_documents(query: str, limit: int = 5) -> str:
    needle = query.casefold()
    matches = [
        f"{path}: {document['summary']}"
        for path, document in DOCUMENTS.items()
        if document["visibility"] == "public"
        and needle in f"{path} {document['summary']}".casefold()
    ]
    return "\n".join(matches[:limit]) or "No public documents matched."


# [docs:start-guard-document-access]
@dataclass(frozen=True, slots=True)
class DocumentAccessGuard(PermissionGuard):
    allowed_paths: frozenset[str]

    async def check(
        self,
        tool: Tool[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        path = str(kwargs["path"])
        if path not in self.allowed_paths:
            raise GuardError(f"{tool.name} denied access to {path}")
        return {**kwargs, "path": path}


document_access = DocumentAccessGuard(
    allowed_paths=frozenset({"README.md", "docs/architecture.md"}),
)
read_document_tool = Tool(
    name="read_document",
    handler=read_document,
    description="Read one document by exact path; do not use for discovery.",
    guards=(document_access,),
)
# [docs:end-guard-document-access]


search_documents_tool = Tool(
    name="search_documents",
    handler=search_documents,
    description="Search public documents when no exact path is available.",
)


# [docs:start-guard-try-calls]
async def try_guard() -> None:
    allowed = await read_document_tool(path="README.md")
    assert allowed == "README.md: Project overview"
    assert handler_calls == ["README.md"]

    try:
        await read_document_tool(path=".env")
    except GuardError as error:
        assert ".env" in str(error)
    else:
        raise AssertionError(".env must be denied")

    transport = StubTransport(
        [
            make_tool_use_response(
                tool_name="read_document",
                tool_input={"path": ".env"},
            ),
            make_text_response("I cannot access .env."),
        ]
    )
    agent = Agent(
        system="Use document tools for facts about project documents.",
        transport=transport,
        tools=[read_document_tool, search_documents_tool],
    )
    events = [
        event
        async for event in agent.run_stream(
            "Show .env",
            MemoryContextStore(),
        )
    ]
    denied_results = [
        event for event in events if isinstance(event, ToolResult) and event.is_error
    ]

    assert len(denied_results) == 1
    assert handler_calls == ["README.md"]
    public_results = await search_documents_tool(query="")
    assert "README.md" in public_results
    assert "docs/architecture.md" in public_results
    assert ".env" not in public_results
    print(denied_results[0].content)


async def main() -> None:
    logging.getLogger("axio.agent").setLevel(logging.CRITICAL)
    await try_guard()


if __name__ == "__main__":
    asyncio.run(main())
# [docs:end-guard-try-calls]
