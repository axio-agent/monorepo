"""ContextStore: protocol for conversation history storage."""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self
from uuid import uuid4

from axio.blocks import TextBlock, ToolResultBlock
from axio.messages import Message
from axio.transport import CompletionTransport

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    message_count: int
    preview: str
    created_at: str
    input_tokens: int = 0
    output_tokens: int = 0


class ContextStore(ABC):
    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @abstractmethod
    async def append(self, message: Message) -> None: ...

    @abstractmethod
    async def get_history(self) -> list[Message]: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def fork(self) -> ContextStore: ...

    @abstractmethod
    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None: ...

    @abstractmethod
    async def get_context_tokens(self) -> tuple[int, int]: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def list_sessions(self) -> list[SessionInfo]:
        """List available sessions. Default: returns a single entry for the current session."""
        history = await self.get_history()
        in_tok, out_tok = await self.get_context_tokens()
        preview = "(empty)"
        for msg in history:
            if msg.role == "user":
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        preview = text[:80] + ("..." if len(text) > 80 else "")
                        break
                break
        return [
            SessionInfo(
                session_id=self.session_id,
                message_count=len(history),
                preview=preview,
                created_at="",
                input_tokens=in_tok,
                output_tokens=out_tok,
            ),
        ]

    async def add_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        cur_in, cur_out = await self.get_context_tokens()
        await self.set_context_tokens(cur_in + input_tokens, cur_out + output_tokens)

    @classmethod
    async def from_history(cls, history: list[Message]) -> Self:
        """Create a new ContextStore pre-populated with *history*."""
        store = cls()
        for message in history:
            await store.append(message)
        return store

    @classmethod
    async def from_context(cls, context: ContextStore) -> Self:
        return await cls.from_history(await context.get_history())


class MemoryContextStore(ContextStore):
    """Simple in-memory context store. fork() returns a deep copy."""

    def __init__(self, history: list[Message] | None = None) -> None:
        self._session_id = uuid4().hex
        self._history: list[Message] = list(history or [])
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    async def append(self, message: Message) -> None:
        self._history.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._history)

    async def clear(self) -> None:
        self._history.clear()
        self._input_tokens = 0
        self._output_tokens = 0

    async def fork(self) -> MemoryContextStore:
        store = MemoryContextStore(copy.deepcopy(self._history))
        store._input_tokens = self._input_tokens
        store._output_tokens = self._output_tokens
        return store

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    async def get_context_tokens(self) -> tuple[int, int]:
        return self._input_tokens, self._output_tokens

    async def close(self) -> None:
        pass


_DEFAULT_COMPACTION_PROMPT = (
    "You are a conversation summarizer. You will see a conversation between"
    " a user and an AI assistant, including tool calls and their results."
    " Produce a concise summary preserving: user goals, decisions made,"
    " key facts, tool outcomes, and state changes. Write as narrative prose,"
    " not as a transcript."
)


async def compact_context(
    context: ContextStore,
    transport: CompletionTransport,
    *,
    max_messages: int = 20,
    keep_recent: int = 6,
    system_prompt: str | None = None,
) -> list[Message] | None:
    """Summarize old messages from *context*, keeping recent ones verbatim.

    Returns a compacted message list ready to populate a fresh store,
    or ``None`` when no compaction is needed.
    """
    history = await context.get_history()
    if len(history) <= max_messages:
        return None

    split = _find_safe_boundary(history, keep_recent)
    if split <= 0:
        return None

    old, recent = history[:split], history[split:]

    # Deferred import to avoid circular dependency (context ↔ agent)
    from axio.agent import Agent

    summary_ctx = MemoryContextStore(old)
    agent = Agent(
        system=system_prompt or _DEFAULT_COMPACTION_PROMPT,
        tools=[],
        transport=transport,
        max_iterations=1,
    )
    try:
        summary = await agent.run("Summarize the conversation above.", summary_ctx)
    except Exception:
        logger.warning("Context compaction failed, keeping original history", exc_info=True)
        return None

    return [
        Message(role="user", content=[TextBlock(text=summary)]),
        Message(role="assistant", content=[TextBlock(text="Understood, context restored.")]),
        *recent,
    ]


def _find_safe_boundary(history: list[Message], keep_recent: int) -> int:
    """Return a split index that never separates a tool_use from its tool_result."""
    split = len(history) - keep_recent
    while split > 0 and any(isinstance(b, ToolResultBlock) for b in history[split].content):
        split -= 1
    return split
