"""Minimal axio example — no API key, no external services.

Shows the core agent loop using StubTransport, which replays scripted
event sequences.  Replace StubTransport with a real transport (e.g.
``OpenAITransport``) and add your API key to go live.

Run:
    uv run python examples/minimal.py
"""

from __future__ import annotations

import asyncio

from axio import Agent, MemoryContextStore, TextDelta
from axio.testing import StubTransport, make_text_response, make_tool_use_response


# ── Tool ─────────────────────────────────────────────────────────────────────


async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22°C in {city}."


# ── Run ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Scripted: LLM calls get_weather, then sends a text reply.
    transport = StubTransport(
        [
            make_tool_use_response("get_weather", tool_input={"city": "Berlin"}),
            make_text_response("The weather in Berlin is sunny and 22°C."),
        ]
    )

    from axio import Tool

    agent = Agent(
        system="You are a helpful weather assistant.",
        transport=transport,
        tools=[Tool(name="get_weather", handler=get_weather)],
    )

    context = MemoryContextStore()
    async for event in agent.run_stream("What's the weather in Berlin?", context):
        if isinstance(event, TextDelta):
            print(event.delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
