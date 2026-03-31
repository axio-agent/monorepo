"""Tool selector: choose relevant tools per query via embedding similarity."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from axio.blocks import TextBlock, ToolResultBlock
from axio.events import StreamEvent
from axio.messages import Message
from axio.tool import Tool
from axio.transport import CompletionTransport, EmbeddingTransport


@runtime_checkable
class ToolSelector(Protocol):
    async def select(self, messages: list[Message], tools: list[Tool]) -> list[Tool]: ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_last_user_text(messages: list[Message]) -> str | None:
    """Return joined text from the last user message, or None if it's a tool-result iteration."""
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
        if texts:
            return " ".join(texts)
        # User message with only ToolResultBlocks → tool-result iteration
        if any(isinstance(b, ToolResultBlock) for b in msg.content):
            return None
        return None
    return None


class EmbeddingToolSelector:
    def __init__(
        self,
        transport: EmbeddingTransport,
        *,
        top_k: int = 5,
        pinned: frozenset[str] = frozenset(),
    ) -> None:
        self._transport = transport
        self._top_k = top_k
        self._pinned = pinned
        self._tool_embeddings: dict[str, list[float]] = {}
        self._tool_descriptions: dict[str, str] = {}

    async def _ensure_embeddings(self, tools: list[Tool]) -> None:
        to_embed: list[tuple[str, str]] = []
        for tool in tools:
            cached_desc = self._tool_descriptions.get(tool.name)
            if cached_desc is None or cached_desc != tool.description:
                to_embed.append((tool.name, tool.description))

        if not to_embed:
            return

        texts = [desc for _, desc in to_embed]
        vectors = await self._transport.embed(texts)
        for (name, desc), vec in zip(to_embed, vectors):
            self._tool_embeddings[name] = vec
            self._tool_descriptions[name] = desc

    async def select(self, messages: list[Message], tools: list[Tool]) -> list[Tool]:
        if len(tools) <= self._top_k:
            return tools

        query = _extract_last_user_text(messages)
        if query is None:
            return tools

        await self._ensure_embeddings(tools)

        query_vec = (await self._transport.embed([query]))[0]

        pinned_tools: list[Tool] = []
        scorable: list[tuple[float, Tool]] = []

        for tool in tools:
            if tool.name in self._pinned:
                pinned_tools.append(tool)
            elif tool.name in self._tool_embeddings:
                score = _cosine_similarity(query_vec, self._tool_embeddings[tool.name])
                scorable.append((score, tool))

        scorable.sort(key=lambda x: x[0], reverse=True)
        remaining_slots = max(0, self._top_k - len(pinned_tools))
        selected = pinned_tools + [t for _, t in scorable[:remaining_slots]]
        return selected


class ToolFilteringTransport:
    """CompletionTransport decorator that filters tools via a ToolSelector."""

    def __init__(self, transport: CompletionTransport, selector: ToolSelector) -> None:
        self._transport = transport
        self._selector = selector

    def stream(self, messages: list[Message], tools: list[Tool], system: str) -> AsyncIterator[StreamEvent]:
        return self._filtered_stream(messages, tools, system)

    async def _filtered_stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        selected = await self._selector.select(messages, tools)
        async for event in self._transport.stream(messages, selected, system):
            yield event
