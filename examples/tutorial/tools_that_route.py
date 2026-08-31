from __future__ import annotations

import asyncio
from typing import Annotated, Any

from axio import (
    Agent,
    CompletionTransport,
    Field,
    MemoryContextStore,
    Tool,
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


calls: list[tuple[str, str, int]] = []


# [docs:start-tools-routing-handlers]
async def read_document(
    path: Annotated[
        str,
        Field(description="Exact project document path, for example README.md"),
    ],
) -> str:
    """Return one project document summary from its exact path."""
    calls.append(("read_document", path, 0))
    document = DOCUMENTS.get(path)
    if document is None:
        return f"Document {path} was not found."
    return f"{path}: {document['summary']}"


async def search_documents(
    query: Annotated[
        str,
        Field(description="Words to match in a public document path or summary"),
    ],
    limit: Annotated[
        int,
        Field(description="Maximum summaries to return", default=5, ge=1, le=10),
    ] = 5,
) -> str:
    """Search public project document summaries."""
    calls.append(("search_documents", query, limit))
    needle = query.casefold()
    matches = [
        f"{path}: {document['summary']}"
        for path, document in DOCUMENTS.items()
        if document["visibility"] == "public"
        and needle in f"{path} {document['summary']}".casefold()
    ]
    return "\n".join(matches[:limit]) or "No public documents matched."


# [docs:end-tools-routing-handlers]


# [docs:start-tools-routing-definitions]
read_document_tool = Tool(
    name="read_document",
    handler=read_document,
    description=(
        "Read one project document by exact path. Choose this tool when the user "
        "supplies a path such as README.md. Do not use it to discover documents."
    ),
)
search_documents_tool = Tool(
    name="search_documents",
    handler=search_documents,
    description=(
        "Search public project documents by path or summary. Choose this tool "
        "when the user describes a document without supplying an exact path."
    ),
)


assert read_document_tool.input_schema["required"] == ["path"]
limit_schema = search_documents_tool.input_schema["properties"]["limit"]
assert limit_schema["default"] == 5
assert limit_schema["minimum"] == 1
assert limit_schema["maximum"] == 10
# [docs:end-tools-routing-definitions]


# [docs:start-tools-routing-agent]
def build_agent(transport: CompletionTransport) -> Agent:
    return Agent(
        system="Answer questions about project documents. Use tools for facts.",
        transport=transport,
        tools=[read_document_tool, search_documents_tool],
    )


agent = build_agent(StubTransport([make_text_response("Done.")]))
assert [tool.name for tool in agent.tools] == ["read_document", "search_documents"]
assert agent.system == "Answer questions about project documents. Use tools for facts."
# [docs:end-tools-routing-agent]


# [docs:start-tools-routing-dispatch]
async def dispatch(tool_name: str, tool_input: dict[str, Any]) -> None:
    transport = StubTransport(
        [
            make_tool_use_response(tool_name=tool_name, tool_input=tool_input),
            make_text_response("Done."),
        ]
    )
    agent = Agent(
        system="Use document tools for facts about project documents.",
        transport=transport,
        tools=[read_document_tool, search_documents_tool],
    )
    assert await agent.run("test request", MemoryContextStore()) == "Done."


async def try_routing() -> None:
    await dispatch("read_document", {"path": "README.md"})
    await dispatch("search_documents", {"query": "architecture"})

    assert calls == [
        ("read_document", "README.md", 0),
        ("search_documents", "architecture", 5),
    ]
    print(calls)


async def main() -> None:
    await try_routing()


if __name__ == "__main__":
    asyncio.run(main())
# [docs:end-tools-routing-dispatch]
