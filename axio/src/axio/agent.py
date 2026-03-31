"""Agent: the core agentic loop orchestrating transport, tools, and context."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import ContextStore
from axio.events import (
    Error,
    IterationEnd,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.messages import Message
from axio.stream import AgentStream
from axio.tool import Tool
from axio.transport import CompletionTransport
from axio.types import StopReason, Usage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Agent:
    system: str
    tools: list[Tool]
    transport: CompletionTransport
    max_iterations: int = field(default=50)

    def run_stream(self, user_message: str, context: ContextStore) -> AgentStream:
        return AgentStream(self._run_loop(user_message, context))

    async def run(self, user_message: str, context: ContextStore) -> str:
        return await self.run_stream(user_message, context).get_final_text()

    async def dispatch_tools(self, blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        tool_names = [b.name for b in blocks]
        logger.info("Dispatching %d tool(s): %s", len(blocks), tool_names)

        async def _run_one(block: ToolUseBlock) -> ToolResultBlock:
            tool = self._find_tool(block.name)
            if tool is None:
                logger.warning("Unknown tool requested: %s", block.name)
                return ToolResultBlock(tool_use_id=block.id, content=f"Unknown tool: {block.name}", is_error=True)
            logger.debug("Tool %s (id=%s) args=%s", block.name, block.id, json.dumps(block.input)[:200])
            try:
                result = await tool(**block.input)
                content = result if isinstance(result, str) else str(result)
            except Exception as exc:
                logger.error("Tool %s raised %s: %s", block.name, type(exc).__name__, exc, exc_info=True)
                return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
            return ToolResultBlock(tool_use_id=block.id, content=content)

        results = list(await asyncio.gather(*[_run_one(b) for b in blocks]))
        error_count = sum(1 for r in results if r.is_error)
        logger.info("Tools complete: %d total, %d errors", len(results), error_count)
        return results

    def _find_tool(self, name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    async def _append(self, context: ContextStore, message: Message) -> None:
        await context.append(message)

    @staticmethod
    def _accumulate_text(content: list[TextBlock | ToolUseBlock], delta: str) -> None:
        """Append text delta — merge into last TextBlock or start a new one."""
        if content and isinstance(content[-1], TextBlock):
            content[-1] = TextBlock(text=content[-1].text + delta)
        else:
            content.append(TextBlock(text=delta))

    @staticmethod
    def _finalize_pending_tools(
        pending: dict[str, dict[str, Any]],
        usage: Usage,
    ) -> tuple[list[ToolUseBlock], set[str]]:
        """Convert streamed tool-call fragments into ToolUseBlocks.

        Returns (blocks, malformed_ids).
        """
        blocks: list[ToolUseBlock] = []
        malformed: set[str] = set()
        for tid, info in pending.items():
            raw = "".join(info["json_parts"])
            if not raw:
                logger.warning(
                    "Tool %s (id=%s) received empty arguments (output may be truncated, output_tokens=%d)",
                    info["name"],
                    tid,
                    usage.output_tokens,
                )
                inp: dict[str, Any] = {}
            else:
                try:
                    inp = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Tool %s (id=%s) has malformed JSON arguments: %s\nRaw: %s",
                        info["name"],
                        tid,
                        exc,
                        raw,
                    )
                    malformed.add(tid)
                    inp = {}
            blocks.append(ToolUseBlock(id=tid, name=info["name"], input=inp))
        return blocks, malformed

    async def _run_loop(self, user_message: str, context: ContextStore) -> AsyncGenerator[StreamEvent, None]:
        total_usage = Usage(0, 0)
        session_end_emitted = False
        await self._append(context, Message(role="user", content=[TextBlock(text=user_message)]))

        try:
            for iteration in range(1, self.max_iterations + 1):
                history = await context.get_history()
                logger.info("Iteration %d, history length=%d", iteration, len(history))
                active_tools = self.tools

                content: list[TextBlock | ToolUseBlock] = []
                pending: dict[str, dict[str, Any]] = {}
                stop_reason = StopReason.end_turn
                malformed: set[str] = set()

                try:
                    async for event in self.transport.stream(history, active_tools, self.system):
                        yield event
                        match event:
                            case TextDelta(delta=delta):
                                self._accumulate_text(content, delta)
                            case ToolUseStart(tool_use_id=tid, name=name):
                                pending[tid] = {"name": name, "json_parts": []}
                            case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                                if tid in pending:
                                    pending[tid]["json_parts"].append(pj)
                            case IterationEnd(usage=usage, stop_reason=sr):
                                blocks, malformed = self._finalize_pending_tools(pending, usage)
                                content.extend(blocks)
                                pending.clear()
                                total_usage = total_usage + usage
                                await context.add_context_tokens(usage.input_tokens, usage.output_tokens)
                                stop_reason = sr
                except Exception as exc:
                    logger.error("Transport error: %s", exc, exc_info=True)
                    yield Error(exception=exc)
                    yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                    session_end_emitted = True
                    return

                tool_blocks = [b for b in content if isinstance(b, ToolUseBlock)]

                if tool_blocks:
                    if stop_reason != StopReason.tool_use:
                        logger.warning(
                            "Dispatching %d tool(s) despite stop_reason=%s",
                            len(tool_blocks),
                            stop_reason,
                        )

                    # Dispatch tools BEFORE appending to context — cancellation
                    # between here and the two appends below cannot leave orphan
                    # ToolUseBlocks in the persistent context store.
                    valid = [b for b in tool_blocks if b.id not in malformed]
                    error_results = [
                        ToolResultBlock(
                            tool_use_id=b.id,
                            content=(
                                f"Malformed JSON arguments for tool {b.name}."
                                f" Raw input could not be parsed. Please retry the tool call"
                                f" with valid JSON arguments."
                            ),
                            is_error=True,
                        )
                        for b in tool_blocks
                        if b.id in malformed
                    ]
                    dispatched = await self.dispatch_tools(valid, iteration) if valid else []
                    results = dispatched + error_results

                    # Append both messages atomically (assistant + tool results)
                    await self._append(context, Message(role="assistant", content=list(content)))
                    await self._append(context, Message(role="user", content=list(results)))

                    # Yield ToolResult events
                    by_id = {b.id: b for b in tool_blocks}
                    for r in results:
                        block = by_id.get(r.tool_use_id)
                        result_content = (
                            r.content
                            if isinstance(r.content, str)
                            else "\n".join(b.text for b in r.content if isinstance(b, TextBlock))
                        )
                        yield ToolResult(
                            tool_use_id=r.tool_use_id,
                            name=block.name if block else "",
                            is_error=r.is_error,
                            content=result_content,
                            input=block.input if block else {},
                        )
                    continue

                await self._append(context, Message(role="assistant", content=list(content)))

                match stop_reason:
                    case StopReason.end_turn:
                        logger.debug("End turn: total_usage=%s", total_usage)
                        yield SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case StopReason.max_tokens | StopReason.error:
                        yield Error(exception=RuntimeError(f"Transport stopped with: {stop_reason}"))
                        yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                        session_end_emitted = True
                        return

            logger.warning("Max iterations (%d) reached", self.max_iterations)
            yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
            session_end_emitted = True

        except GeneratorExit:
            return
        except BaseException:
            if not session_end_emitted:
                yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
            raise
