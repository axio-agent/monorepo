"""ContextStore: protocol for conversation history storage."""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self
from uuid import uuid4

from axio.blocks import TextBlock
from axio.messages import Message

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
    def session_id(self) -> str:
        """Lazy-init UUID hex; works without calling super().__init__()."""
        if "_session_id" not in self.__dict__:
            self.__dict__["_session_id"] = uuid4().hex
        return str(self.__dict__["_session_id"])

    @abstractmethod
    async def append(self, message: Message) -> None: ...

    @abstractmethod
    async def get_history(self) -> list[Message]: ...

    async def clear(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support clear()")

    async def fork(self) -> ContextStore:
        """Returns a MemoryContextStore deep copy by default."""
        messages = copy.deepcopy(await self.get_history())
        in_tok, out_tok = await self.get_context_tokens()
        store = MemoryContextStore(messages)
        await store.set_context_tokens(in_tok, out_tok)
        return store

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """No-op by default; tokens are silently dropped."""

    async def get_context_tokens(self) -> tuple[int, int]:
        """Returns (0, 0) by default."""
        return 0, 0

    async def close(self) -> None:
        """No-op by default."""

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
