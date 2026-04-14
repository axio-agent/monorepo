"""Context compaction: summarise old history to stay within token limits."""

from __future__ import annotations

import logging

from axio.agent import Agent
from axio.blocks import TextBlock, ToolResultBlock
from axio.context import ContextStore, MemoryContextStore
from axio.messages import Message
from axio.transport import CompletionTransport

logger = logging.getLogger(__name__)

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
    system_prompt: str = _DEFAULT_COMPACTION_PROMPT,
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
