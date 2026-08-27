"""Agent: the core agentic loop orchestrating transport, tools, and context."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Self

from .blocks import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from .context import ContextStore
from .events import (
    AudioOutput,
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
    VideoOutput,
)
from .messages import Message
from .models import Capability
from .selector import ToolSelector
from .stream import AgentStream
from .tool import Tool
from .transport import CompletionTransport
from .types import StopReason, Usage

logger = logging.getLogger(__name__)


class _RepetitionDetector:
    """Detects when model output is stuck in a repetitive loop.

    Two complementary checks run periodically on accumulated text:

    1. **Short-period**: counts trailing consecutive repetitions of
       patterns from 1 to ``max_period`` chars.  Triggers when repetitions
       span >= ``min_repeat_span`` chars.  Catches single-token and
       short-phrase loops quickly.

    2. **Long-period**: checks whether the last ``long_window`` chars
       appear verbatim earlier in the output.  Catches paragraph-level
       repetition that the short-period check would miss.
    """

    __slots__ = (
        "_parts",
        "_total_len",
        "_last_check",
        "_interval",
        "_min_len",
        "_max_period",
        "_min_repeat_span",
        "_long_window",
    )

    def __init__(
        self,
        interval: int = 200,
        min_len: int = 800,
        max_period: int = 150,
        min_repeat_span: int = 200,
        long_window: int = 500,
    ) -> None:
        self._parts: list[str] = []
        self._total_len = 0
        self._last_check = 0
        self._interval = interval
        self._min_len = min_len
        self._max_period = max_period
        self._min_repeat_span = min_repeat_span
        self._long_window = long_window

    def feed(self, delta: str) -> bool:
        """Feed a text delta.  Returns ``True`` when a loop is detected."""
        self._parts.append(delta)
        self._total_len += len(delta)

        if self._total_len < self._min_len:
            return False
        if self._total_len - self._last_check < self._interval:
            return False
        self._last_check = self._total_len

        full = "".join(self._parts)
        self._parts = [full]
        n = len(full)

        # --- Short-period: trailing repetition of a small pattern ---
        max_p = min(self._max_period, n // 3)
        for p in range(1, max_p + 1):
            chunk = full[n - p : n]
            count = 1
            pos = n - 2 * p
            while pos >= 0 and full[pos : pos + p] == chunk:
                count += 1
                pos -= p
            if count >= 3 and count * p >= self._min_repeat_span:
                return True

        # --- Long-period: trailing window found earlier verbatim ---
        w = min(self._long_window, n // 2)
        if w >= self._min_repeat_span:
            window = full[-w:]
            if full.find(window, 0, n - w) >= 0:
                return True

        return False


#: Reasons that say the turn failed rather than finished. A call streamed inside one of them is
#: output nobody vouched for, so it is not run.
_NO_DISPATCH = frozenset({StopReason.error, StopReason.refusal, StopReason.cancelled})


@dataclass(slots=True)
class Agent:
    system: str
    transport: CompletionTransport
    tools: list[Tool[Any]] = field(default_factory=list)
    selector: ToolSelector | None = field(default=None)
    max_iterations: int = field(default=50)
    last_iteration_message: Message | None = field(default=None)

    def copy(self, **overrides: Any) -> Self:
        """Return a new Agent with *overrides* applied."""
        return dataclasses.replace(self, **overrides)

    def run_stream(self, user_message: str, context: ContextStore) -> AgentStream:
        return AgentStream(self._run_loop(user_message, context))

    async def run(self, user_message: str, context: ContextStore) -> str:
        return await self.run_stream(user_message, context).get_final_text()

    async def dispatch_tools(self, blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        logger.info("Dispatching %d tool(s): %r", len(blocks), blocks)

        async def _run_one(block: ToolUseBlock) -> ToolResultBlock:
            tool = self._find_tool(block.name)
            if tool is None:
                logger.warning("Unknown tool requested: %s", block.name)
                return ToolResultBlock(tool_use_id=block.id, content=f"Unknown tool: {block.name}", is_error=True)
            logger.debug("Tool %s (id=%s) args=%s", block.name, block.id, json.dumps(block.input)[:200])
            try:
                result = await tool(**block.input)
                if isinstance(result, str):
                    content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock] = result
                elif isinstance(result, list) and all(isinstance(b, ContentBlock) for b in result):
                    content = result
                else:
                    content = str(result)
            except Exception as exc:
                logger.error("Tool %s raised %s: %s", block.name, type(exc).__name__, exc, exc_info=True)
                return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
            return ToolResultBlock(tool_use_id=block.id, content=content)

        results = list(await asyncio.gather(*[_run_one(b) for b in blocks]))
        error_count = sum(1 for r in results if r.is_error)
        logger.info("Tools complete: %d total, %d errors", len(results), error_count)
        return results

    async def _dispatch_tools_streaming(
        self,
        blocks: list[ToolUseBlock],
        iteration: int,
        output_queue: asyncio.Queue[ToolOutputDelta | None],
    ) -> list[ToolResultBlock]:
        """Like dispatch_tools but pushes ToolOutputDelta events for streaming tools."""
        logger.info("Dispatching %d tool(s) with streaming: %r", len(blocks), blocks)

        async def _run_one(block: ToolUseBlock) -> ToolResultBlock:
            tool = self._find_tool(block.name)
            if tool is None:
                logger.warning("Unknown tool requested: %s", block.name)
                return ToolResultBlock(tool_use_id=block.id, content=f"Unknown tool: {block.name}", is_error=True)
            logger.debug("Tool %s (id=%s) args=%s", block.name, block.id, json.dumps(block.input)[:200])

            if tool.supports_streaming:
                chunks: list[tuple[float, str, str]] = []
                t0 = time.monotonic()
                try:
                    async for key, text in tool.call_streaming(**block.input):
                        chunks.append((time.monotonic() - t0, key, text))
                        await output_queue.put(
                            ToolOutputDelta(tool_use_id=block.id, name=block.name, key=key, delta=text)
                        )
                except Exception as exc:
                    logger.error("Tool %s raised %s: %s", block.name, type(exc).__name__, exc, exc_info=True)
                    return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
                return ToolResultBlock(tool_use_id=block.id, content=tool.format_stream_result(chunks))
            else:
                try:
                    result = await tool(**block.input)
                    if isinstance(result, str):
                        content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock] = result
                    elif isinstance(result, list) and all(isinstance(b, ContentBlock) for b in result):
                        content = result
                    else:
                        content = str(result)
                except Exception as exc:
                    logger.error("Tool %s raised %s: %s", block.name, type(exc).__name__, exc, exc_info=True)
                    return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
                return ToolResultBlock(tool_use_id=block.id, content=content)

        results = list(await asyncio.gather(*[_run_one(b) for b in blocks]))
        error_count = sum(1 for r in results if r.is_error)
        logger.info("Tools complete: %d total, %d errors", len(results), error_count)
        return results

    def _find_tool(self, name: str) -> Tool[Any] | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    async def _append(self, context: ContextStore, message: Message) -> None:
        await context.append(message)

    @staticmethod
    def _accumulate_text(
        content: list[TextBlock | ReasoningBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock],
        delta: str,
    ) -> None:
        """Append text delta — merge into last TextBlock or start a new one."""
        if content and isinstance(content[-1], TextBlock):
            content[-1] = TextBlock(text=content[-1].text + delta)
        else:
            content.append(TextBlock(text=delta))

    @staticmethod
    def _accumulate_reasoning(
        content: list[TextBlock | ReasoningBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock],
        delta: str,
    ) -> None:
        """Append a reasoning delta, merging into the block still being built.

        A signed block is finished. The provider computed the signature over the text it had, so a
        later delta starts a new block rather than making the stored signature disagree with it.
        """
        last = content[-1] if content else None
        if isinstance(last, ReasoningBlock) and not last.signature:
            content[-1] = replace(last, text=last.text + delta)
        else:
            content.append(ReasoningBlock(text=delta))

    @staticmethod
    def _sign_reasoning(
        content: list[TextBlock | ReasoningBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock],
        data: str,
        redacted: bool,
        id: str = "",
        joining: bool = False,
    ) -> None:
        """Attach the provider's proof to the reasoning it belongs to.

        The proof is kept because the turn has to be replayable. A provider refuses a thinking block
        whose signature is missing, and a redacted block carries a signature and no text at all.

        ``joining`` says this proof continues the one before it. The transports index a signature
        by the block it proves, so a repeated index is the rest of one proof rather than another.
        Stored apart, the block replays with half a signature and the turn after it is refused.
        """
        last = content[-1] if content else None
        if joining and isinstance(last, ReasoningBlock) and last.signature:
            content[-1] = replace(last, signature=last.signature + data)
        elif isinstance(last, ReasoningBlock) and not last.signature:
            content[-1] = replace(last, signature=data, redacted=redacted, id=id)
        else:
            content.append(ReasoningBlock(signature=data, redacted=redacted, id=id))

    @staticmethod
    def _finalize_pending_tools(
        pending: dict[str, dict[str, Any]],
        usage: Usage,
    ) -> tuple[list[ToolUseBlock], set[str]]:
        """Convert streamed tool-call fragments into ToolUseBlocks.

        Returns (blocks, malformed_ids).  Malformed IDs arise when
        max_tokens truncates the response mid-tool-call, producing
        incomplete JSON (expected with eager_input_streaming).  The
        caller is responsible for not executing malformed tools.
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

    async def _select_tools(self, history: list[Message], tools: list[Tool[Any]]) -> Iterable[Tool[Any]]:
        if not tools:
            return []
        if not self.selector:
            return tools
        return await self.selector.select(history, tools)

    async def _run_loop(self, user_message: str, context: ContextStore) -> AsyncGenerator[StreamEvent, None]:
        total_usage = Usage(0, 0)
        session_end_emitted = False
        ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        await self._append(context, Message(role="user", content=[TextBlock(text=f"[{ts}] {user_message}")]))

        try:
            for iteration in range(1, self.max_iterations + 1):
                history = await context.get_history()
                logger.info("Iteration %d, history length=%d", iteration, len(history))
                effective_history = (
                    [*history, self.last_iteration_message]
                    if self.last_iteration_message and iteration == self.max_iterations
                    else history
                )
                model = getattr(self.transport, "model", None)
                model_caps = getattr(model, "capabilities", None)
                if model_caps is not None and Capability.tool_use not in model_caps:
                    active_tools: list[Tool[Any]] = []
                else:
                    active_tools = list(await self._select_tools(effective_history, self.tools))

                content: list[TextBlock | ReasoningBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock] = []
                # Which block the last proof was for. A repeated index continues that proof.
                signed_at: int | None = None
                pending: dict[str, dict[str, Any]] = {}
                stop_reason = StopReason.end_turn
                malformed: set[str] = set()
                repetition_detected = False
                rep_detector = _RepetitionDetector()

                try:
                    async for event in self.transport.stream(effective_history, active_tools, self.system):
                        yield event
                        match event:
                            case TextDelta(delta=delta):
                                self._accumulate_text(content, delta)
                                if rep_detector.feed(delta):
                                    note = "\n\n[Output truncated: repetitive content detected]"
                                    self._accumulate_text(content, note)
                                    yield TextDelta(index=0, delta=note)
                                    repetition_detected = True
                                    break
                            case Refusal(text=text):
                                # Kept as the turn's text: a refusal is what the assistant said.
                                # Left out, the stored turn is empty and the next request carries a
                                # blank assistant message the provider then rejects.
                                if text:
                                    self._accumulate_text(content, text)
                            case ReasoningDelta(delta=delta):
                                self._accumulate_reasoning(content, delta)
                            case ReasoningSignature(index=at, data=signature, redacted=redacted, id=block_id):
                                self._sign_reasoning(content, signature, redacted, block_id, joining=at == signed_at)
                                signed_at = at
                            case ImageOutput(data=data, media_type=mt):
                                content.append(ImageBlock(media_type=mt, data=data))
                            case VideoOutput(data=data, media_type=mt):
                                content.append(VideoBlock(media_type=mt, data=data))
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

                if repetition_detected:
                    await self._append(context, Message(role="assistant", content=list(content)))
                    partial = getattr(self.transport, "last_usage", None)
                    if partial:
                        total_usage = total_usage + partial
                    yield SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=total_usage)
                    session_end_emitted = True
                    return

                tool_blocks = [b for b in content if isinstance(b, ToolUseBlock)]

                if tool_blocks and stop_reason in _NO_DISPATCH:
                    # The turn did not end in a way that vouches for what it produced. Running its
                    # calls acts on output the transport has just reported as failed, declined or
                    # cut short, and the run ends immediately afterwards regardless.
                    logger.warning(
                        "Not dispatching %d tool(s): the turn ended with stop_reason=%s",
                        len(tool_blocks),
                        stop_reason,
                    )
                    tool_blocks = []

                if tool_blocks:
                    if stop_reason != StopReason.tool_use:
                        logger.warning(
                            "Dispatching %d tool(s) despite stop_reason=%s",
                            len(tool_blocks),
                            stop_reason,
                        )

                    # Dispatch tools BEFORE appending to context. Cancellation
                    # between here and the two appends below then cannot leave
                    # orphan ToolUseBlocks in the persistent context store.
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

                    partial_output: dict[str, list[tuple[float, str, str]]] = {}
                    t0_map: dict[str, float] = {}
                    dispatch_task: asyncio.Task[list[ToolResultBlock]] | None = None
                    try:
                        if valid:
                            has_streaming = any(
                                (t := self._find_tool(b.name)) is not None and t.supports_streaming for b in valid
                            )
                            if has_streaming:
                                output_queue: asyncio.Queue[ToolOutputDelta | None] = asyncio.Queue()

                                async def _dispatch_and_signal() -> list[ToolResultBlock]:
                                    result = await self._dispatch_tools_streaming(valid, iteration, output_queue)
                                    await output_queue.put(None)
                                    return result

                                dispatch_task = asyncio.create_task(_dispatch_and_signal())
                                while True:
                                    ev = await output_queue.get()
                                    if ev is None:
                                        break
                                    if ev.tool_use_id not in t0_map:
                                        t0_map[ev.tool_use_id] = time.monotonic()
                                    partial_output.setdefault(ev.tool_use_id, []).append(
                                        (time.monotonic() - t0_map[ev.tool_use_id], ev.key, ev.delta)
                                    )
                                    yield ev
                                dispatched = await dispatch_task
                            else:
                                dispatched = await self.dispatch_tools(valid, iteration)
                        else:
                            dispatched = []
                        results = dispatched + error_results
                    except asyncio.CancelledError:
                        if dispatch_task is not None and not dispatch_task.done():
                            dispatch_task.cancel()
                            try:
                                await dispatch_task
                            except (asyncio.CancelledError, Exception):
                                pass
                        interrupted_results: list[ToolResultBlock] = []
                        for b in tool_blocks:
                            chunks = partial_output.get(b.id, [])
                            tool = self._find_tool(b.name)
                            if chunks and tool:
                                msg = tool.format_stream_result(chunks) + "\n[interrupted by user]"
                            elif chunks:
                                msg = "".join(text for _, _, text in chunks) + "\n[interrupted by user]"
                            else:
                                msg = "[interrupted by user]"
                            interrupted_results.append(ToolResultBlock(tool_use_id=b.id, content=msg, is_error=True))
                        await self._append(context, Message(role="assistant", content=list(content)))
                        await self._append(context, Message(role="user", content=list(interrupted_results)))
                        raise

                    # Append both messages atomically (assistant + tool results)
                    await self._append(context, Message(role="assistant", content=list(content)))
                    await self._append(context, Message(role="user", content=list(results)))

                    # Gemini stops generating (~20 tokens, end_turn) after receiving
                    # media as sibling inlineData parts alongside functionResponse.
                    # A "Proceed." user message nudges it to actually analyze the content.
                    if getattr(self.transport, "nudge_on_media_tool_result", False) and any(
                        not isinstance(r.content, str)
                        and any(isinstance(b, (AudioBlock, ImageBlock, VideoBlock)) for b in r.content)
                        for r in results
                    ):
                        await self._append(
                            context,
                            Message(
                                role="user",
                                content=[
                                    TextBlock(
                                        text="You now have the media file above in your context. Proceed.",
                                    )
                                ],
                            ),
                        )

                    # Yield ToolResult events + media output events.
                    # Non-streaming tools return full content (str or list of
                    # TextBlock/ImageBlock/VideoBlock) — no information is lost.
                    # Images/videos are yielded as separate ImageOutput/VideoOutput
                    # events so the REPL can save them to disk. The model sees the
                    # actual pixel data via ImageBlock/VideoBlock in the tool result.
                    by_id = {b.id: b for b in tool_blocks}
                    for r in results:
                        block = by_id.get(r.tool_use_id)
                        if isinstance(r.content, str):
                            result_content = r.content
                        else:
                            result_content = "\n".join(b.text for b in r.content if isinstance(b, TextBlock))
                            for media_block in r.content:
                                if isinstance(media_block, ImageBlock):
                                    yield ImageOutput(
                                        index=0, data=media_block.data, media_type=media_block.media_type
                                    )
                                elif isinstance(media_block, AudioBlock):
                                    yield AudioOutput(
                                        index=0, data=media_block.data, media_type=media_block.media_type
                                    )
                                elif isinstance(media_block, VideoBlock):
                                    yield VideoOutput(
                                        index=0, data=media_block.data, media_type=media_block.media_type
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
                    case StopReason.refusal:
                        # This is terminal, and not an error. The model declined, and the decline
                        # is stored as the turn's content. Sending the same prompt again cannot
                        # succeed. Reported as an error the caller cannot tell a decline from a
                        # broken connection, and retries something that will never work.
                        logger.info("Model declined: total_usage=%s", total_usage)
                        yield SessionEndEvent(stop_reason=StopReason.refusal, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case StopReason.pause_turn:
                        # The provider stopped its own tool loop and expects the assistant content
                        # back to finish the turn. The content was just appended, so going round
                        # again is the resume.
                        logger.debug("Paused turn, resuming: total_usage=%s", total_usage)
                        continue
                    case _:
                        # Wildcard on purpose. Named one by one, a reason added later matches
                        # nothing, falls out of the match, and the loop simply runs again. The
                        # model is then re-prompted with unchanged history until max_iterations,
                        # and every one of those turns is paid for.
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
