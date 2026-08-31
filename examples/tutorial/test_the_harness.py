"""Verify one complete Axio tool round trip with a scripted transport."""

from __future__ import annotations

import asyncio

from axio import (
    Agent,
    IterationEnd,
    MemoryContextStore,
    StopReason,
    TextDelta,
    Tool,
    ToolResult,
)
from axio.events import SessionEndEvent
from axio.testing import StubTransport, make_text_response, make_tool_use_response


# [docs:start-test-harness-complete-turn]
async def read_document(path: str) -> str:
    """Return one document from the public repository."""
    return f"{path}: Build an agent harness with Axio."


async def verify_complete_turn() -> None:
    transport = StubTransport(
        [
            make_tool_use_response(
                "read_document",
                tool_input={"path": "README.md"},
            ),
            make_text_response("README.md describes the agent harness."),
        ]
    )
    context = MemoryContextStore()
    agent = Agent(
        system="Use repository tools before reporting document content.",
        transport=transport,
        tools=[Tool(name="read_document", handler=read_document)],
    )

    events = []
    stream = agent.run_stream("Summarize README.md.", context)
    try:
        async for event in stream:
            events.append(event)
    finally:
        await stream.aclose()

    results = [event for event in events if isinstance(event, ToolResult)]
    iterations = [event for event in events if isinstance(event, IterationEnd)]
    endings = [event for event in events if isinstance(event, SessionEndEvent)]
    final_text = "".join(
        event.delta for event in events if isinstance(event, TextDelta)
    )

    assert len(results) == 1
    assert results[0].content == "README.md: Build an agent harness with Axio."
    assert results[0].is_error is False
    assert final_text == "README.md describes the agent harness."
    assert len(iterations) == 2
    assert len(endings) == 1
    assert endings[0].stop_reason is StopReason.end_turn

    history = await context.get_history()
    assert history[0].role == "user"
    assert history[-1].role == "assistant"


# [docs:end-test-harness-complete-turn]


if __name__ == "__main__":
    asyncio.run(verify_complete_turn())
