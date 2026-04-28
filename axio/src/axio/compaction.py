"""Context compaction: summarise old history to stay within token limits."""

from __future__ import annotations

import logging

from .agent import Agent
from .blocks import TextBlock, ToolResultBlock
from .context import ContextStore, MemoryContextStore, SessionInfo
from .messages import Message
from .transport import CompletionTransport

logger = logging.getLogger(__name__)


DEFAULT_COMPACTION_PROMPT = (
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
    keep_recent: int = 6,
    system_prompt: str = DEFAULT_COMPACTION_PROMPT,
) -> list[Message] | None:
    """Summarize old messages from *context*, keeping recent ones verbatim.

    Returns a compacted message list ready to populate a fresh store,
    or ``None`` if the history is too short to compact (split <= 0).

    The caller is responsible for deciding *when* to compact (e.g. by tracking
    ``usage.input_tokens`` from ``IterationEnd`` events).
    """
    history = await context.get_history()
    split = _find_safe_boundary(history, keep_recent)
    if split <= 0:
        return None

    old, recent = history[:split], history[split:]

    summary_ctx = MemoryContextStore(old)
    agent = Agent(system=system_prompt, transport=transport, max_iterations=1)
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


class AutoCompactStore(ContextStore):
    """Delegating ``ContextStore`` wrapper that auto-compacts the inner store
    when token usage exceeds a threshold.  Works with any ``ContextStore``
    backend - ``MemoryContextStore``, ``SQLiteContextStore``, etc.

    The threshold defaults to 75 % of ``transport.model.context_window``
    (read via ``getattr``; falls back to 128 000 if the transport has no
    ``model`` attribute).  Pass ``max_tokens`` explicitly to override.

    Compaction fires from :meth:`add_context_tokens`, which the agent loop
    calls immediately after ``IterationEnd`` - ``input_tokens`` there equals
    the real context size sent to the model in that iteration.

    Internally, :meth:`_do_compact` forks the inner store before calling
    ``compact_context``.  The fork acts as a stable snapshot: the
    summarisation agent reads from it while the live store remains writable.
    The live store is only cleared and repopulated after the (async) summary
    call returns.

    Example::

        from axio.compaction import AutoCompactStore
        from axio.context import MemoryContextStore

        store = AutoCompactStore(MemoryContextStore(), transport, keep_recent=6)
        result = await agent.run(task, store)
    """

    def __init__(
        self,
        store: ContextStore,
        transport: CompletionTransport,
        *,
        keep_recent: int = 6,
        max_tokens: int | None = None,
        threshold: float = 0.75,
    ) -> None:
        self._store = store
        self._transport = transport
        self._keep_recent = keep_recent
        self._threshold = threshold
        if max_tokens is not None:
            self._max_tokens = max_tokens
        else:
            model = getattr(transport, "model", None)
            ctx_win: int = getattr(model, "context_window", 128_000) if model is not None else 128_000
            self._max_tokens = int(ctx_win * threshold)

    @property
    def session_id(self) -> str:
        return self._store.session_id

    async def append(self, message: Message) -> None:
        await self._store.append(message)

    async def get_history(self) -> list[Message]:
        return await self._store.get_history()

    async def clear(self) -> None:
        await self._store.clear()

    async def fork(self) -> AutoCompactStore:
        """Return an ``AutoCompactStore`` wrapping a fork of the inner store."""
        return AutoCompactStore(
            await self._store.fork(),
            self._transport,
            keep_recent=self._keep_recent,
            max_tokens=self._max_tokens,
            threshold=self._threshold,
        )

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        await self._store.set_context_tokens(input_tokens, output_tokens)

    async def get_context_tokens(self) -> tuple[int, int]:
        return await self._store.get_context_tokens()

    async def add_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        await self._store.add_context_tokens(input_tokens, output_tokens)
        if input_tokens > self._max_tokens:
            await self._do_compact()

    async def close(self) -> None:
        await self._store.close()

    async def list_sessions(self) -> list[SessionInfo]:
        return await self._store.list_sessions()

    async def _do_compact(self) -> None:
        snapshot = await self._store.fork()
        compacted = await compact_context(snapshot, self._transport, keep_recent=self._keep_recent)
        if compacted is None:
            return
        in_tok, out_tok = await self._store.get_context_tokens()
        await self._store.clear()
        for msg in compacted:
            await self._store.append(msg)
        await self._store.set_context_tokens(in_tok, out_tok)
