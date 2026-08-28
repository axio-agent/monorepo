from __future__ import annotations

# [docs:start-from-chat-chat-only]
import asyncio

from axio import (
    Agent,
    CompletionTransport,
    MemoryContextStore,
    Tool,
    ToolResultBlock,
)
from axio.testing import StubTransport, make_text_response, make_tool_use_response


async def try_chat() -> None:
    transport = StubTransport(
        [
            make_text_response("I cannot read README.md without a tool."),
        ]
    )
    agent = Agent(
        system="Answer questions about project documents. Use facts from tools.",
        transport=transport,
        tools=[],
    )

    answer = await agent.run("Summarize README.md.", MemoryContextStore())

    assert answer == "I cannot read README.md without a tool."
    assert agent.tools == []
    print(f"chat-only: {answer}")


# [docs:end-from-chat-chat-only]


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


# [docs:start-from-chat-first-tool]
async def read_document(path: str) -> str:
    """Return one project document summary from its exact path."""
    document = DOCUMENTS.get(path)
    if document is None:
        return f"Document {path} was not found."
    return f"{path}: {document['summary']}"


read_document_tool = Tool(name="read_document", handler=read_document)


def build_agent(transport: CompletionTransport) -> Agent:
    return Agent(
        system="Answer questions about project documents. Use tools for facts.",
        transport=transport,
        tools=[read_document_tool],
    )


# [docs:end-from-chat-first-tool]


async def try_agent() -> None:
    transport = StubTransport(
        [
            make_tool_use_response(
                tool_name="read_document",
                tool_id="document-call-1",
                tool_input={"path": "README.md"},
            ),
            make_text_response("README.md: Project overview."),
        ]
    )
    context = MemoryContextStore()

    answer = await build_agent(transport).run("Summarize README.md.", context)
    history = await context.get_history()
    results = [
        block
        for message in history
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]

    assert answer == "README.md: Project overview."
    assert len(results) == 1
    assert results[0].content == "README.md: Project overview"
    print(f"with-tool: {answer}")


async def main() -> None:
    await try_chat()
    await try_agent()


if __name__ == "__main__":
    asyncio.run(main())
