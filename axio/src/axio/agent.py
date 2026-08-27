"""Agent: the core agentic loop orchestrating transport, tools, and context."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Iterator
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


#: Reasons that say the turn failed or was cut off rather than finished. A call streamed inside
#: one is output nobody vouched for, and its arguments may be half-written.
_NO_DISPATCH = frozenset(
    {
        StopReason.error,
        StopReason.refusal,
        StopReason.cancelled,
        StopReason.max_tokens,
        StopReason.context_window_exceeded,
    }
)


#: Sent after a tool returns media. Gemini stops generating after receiving media as sibling
#: inlineData parts, so without it the turn ends in about twenty tokens having read nothing.
_MEDIA_NUDGE = "You now have the media file above in your context. Proceed."


def _malformed_results(blocks: list[ToolUseBlock], malformed: set[str]) -> list[ToolResultBlock]:
    """What a call whose arguments would not parse gets back instead of a run."""
    return [
        ToolResultBlock(
            tool_use_id=block.id,
            content=(
                f"Malformed JSON arguments for tool {block.name}. Raw input could not be parsed."
                f" Please retry the tool call with valid JSON arguments."
            ),
            is_error=True,
        )
        for block in blocks
        if block.id in malformed
    ]


def _carries_media(results: list[ToolResultBlock]) -> bool:
    """Whether any result holds a picture, a sound or a video rather than only text."""
    return any(
        not isinstance(r.content, str) and any(isinstance(b, (AudioBlock, ImageBlock, VideoBlock)) for b in r.content)
        for r in results
    )


def _result_events(results: list[ToolResultBlock], by_id: dict[str, ToolUseBlock]) -> Iterator[StreamEvent]:
    """What the caller sees for each finished call.

    Media travels twice: as its own output event, which is how a caller saves it to disk, and
    inside the result, which is how the model sees the pixels.
    """
    for result in results:
        block = by_id.get(result.tool_use_id)
        if isinstance(result.content, str):
            text = result.content
        else:
            text = "\n".join(b.text for b in result.content if isinstance(b, TextBlock))
            for part in result.content:
                if isinstance(part, ImageBlock):
                    yield ImageOutput(index=0, data=part.data, media_type=part.media_type)
                elif isinstance(part, AudioBlock):
                    yield AudioOutput(index=0, data=part.data, media_type=part.media_type)
                elif isinstance(part, VideoBlock):
                    yield VideoOutput(index=0, data=part.data, media_type=part.media_type)
        yield ToolResult(
            tool_use_id=result.tool_use_id,
            name=block.name if block else "",
            is_error=result.is_error,
            content=text,
            input=block.input if block else {},
        )


#: What one iteration accumulates while the transport streams it.
type TurnBlock = TextBlock | ReasoningBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock


@dataclass(slots=True)
class _Turn:
    """What one iteration accumulates from the transport's stream."""

    content: list[TurnBlock] = field(default_factory=list)
    #: Calls still arriving, by id. Emptied into ``content`` when the turn ends.
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Calls whose arguments would not parse, so nothing may be dispatched from them.
    malformed: set[str] = field(default_factory=set)
    usage: Usage = Usage(0, 0)
    stop_reason: StopReason = StopReason.end_turn
    #: The model repeated itself and the turn was cut short here rather than by the provider.
    repeated: bool = False
    #: Which block the last proof was for. A repeated index continues that proof.
    signed_at: int | None = None


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

        A signed block is finished. The provider computed the signature over the text it had. A
        later delta therefore starts a new block, rather than leaving a signature that disagrees
        with the text beside it.
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
            blocks.append(ToolUseBlock(id=tid, name=info["name"], input=inp, signature=info.get("signature", "")))
        return blocks, malformed

    async def _select_tools(self, history: list[Message], tools: list[Tool[Any]]) -> Iterable[Tool[Any]]:
        if not tools:
            return []
        if not self.selector:
            return tools
        return await self.selector.select(history, tools)

    def _interrupted_results(
        self, blocks: list[ToolUseBlock], partial: dict[str, list[tuple[float, str, str]]]
    ) -> list[ToolResultBlock]:
        """What each call reports when the user stops the run while it is still working.

        Whatever a streaming tool had already produced is kept, because a cancelled call still did
        part of its work and the model has to be told which part.
        """
        made: list[ToolResultBlock] = []
        for block in blocks:
            chunks = partial.get(block.id, [])
            tool = self._find_tool(block.name)
            if chunks and tool:
                text = tool.format_stream_result(chunks) + "\n[interrupted by user]"
            elif chunks:
                text = "".join(text for _, _, text in chunks) + "\n[interrupted by user]"
            else:
                text = "[interrupted by user]"
            made.append(ToolResultBlock(tool_use_id=block.id, content=text, is_error=True))
        return made

    async def _consume(
        self, stream: AsyncIterator[StreamEvent], turn: _Turn, context: ContextStore
    ) -> AsyncGenerator[StreamEvent, None]:
        """Pass every event through, building the turn's content as they go."""
        detector = _RepetitionDetector()
        async for event in stream:
            yield event
            match event:
                case TextDelta(delta=delta):
                    self._accumulate_text(turn.content, delta)
                    if detector.feed(delta):
                        note = "\n\n[Output truncated: repetitive content detected]"
                        self._accumulate_text(turn.content, note)
                        yield TextDelta(index=0, delta=note)
                        turn.repeated = True
                        return
                case Refusal(text=text) if text:
                    # Kept as the turn's text: a refusal is what the assistant said. Left out, the
                    # stored turn is empty and the next request carries a blank assistant message.
                    self._accumulate_text(turn.content, text)
                case ReasoningDelta(delta=delta):
                    self._accumulate_reasoning(turn.content, delta)
                case ReasoningSignature(index=at, data=proof, redacted=redacted, id=block_id):
                    self._sign_reasoning(turn.content, proof, redacted, block_id, joining=at == turn.signed_at)
                    turn.signed_at = at
                case ImageOutput(data=data, media_type=mt):
                    turn.content.append(ImageBlock(media_type=mt, data=data))
                case VideoOutput(data=data, media_type=mt):
                    turn.content.append(VideoBlock(media_type=mt, data=data))
                case ToolUseStart(tool_use_id=tid, name=name, signature=proof):
                    turn.pending[tid] = {"name": name, "json_parts": [], "signature": proof}
                case ToolInputDelta(tool_use_id=tid, partial_json=chunk) if tid in turn.pending:
                    turn.pending[tid]["json_parts"].append(chunk)
                case IterationEnd(usage=usage, stop_reason=reason):
                    blocks, turn.malformed = self._finalize_pending_tools(turn.pending, usage)
                    turn.content.extend(blocks)
                    turn.pending.clear()
                    turn.usage = turn.usage + usage
                    turn.stop_reason = reason
                    await context.add_context_tokens(usage.input_tokens, usage.output_tokens)

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

                turn = _Turn()
                stream = self.transport.stream(effective_history, active_tools, self.system)
                try:
                    async for event in self._consume(stream, turn, context):
                        yield event
                except Exception as exc:
                    logger.error("Transport error: %s", exc, exc_info=True)
                    yield Error(exception=exc)
                    yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                    session_end_emitted = True
                    return

                content = turn.content
                stop_reason = turn.stop_reason
                malformed = turn.malformed
                total_usage = total_usage + turn.usage

                if turn.repeated:
                    await self._append(context, Message(role="assistant", content=list(content)))
                    partial = getattr(self.transport, "last_usage", None)
                    if partial:
                        total_usage = total_usage + partial
                    yield SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=total_usage)
                    session_end_emitted = True
                    return

                tool_blocks = [b for b in content if isinstance(b, ToolUseBlock)]

                if tool_blocks and stop_reason in _NO_DISPATCH:
                    # The turn did not end in a way that vouches for what it produced, and the run ends
                    # immediately after this regardless.
                    logger.warning(
                        "Not dispatching %d tool(s): the turn ended with stop_reason=%s",
                        len(tool_blocks),
                        stop_reason,
                    )
                    # Removed from the turn as well, not only from this list. Left in, they are persisted
                    # with no result beside them, which Anthropic and Google refuse on the next request.
                    content[:] = [b for b in content if not isinstance(b, ToolUseBlock)]
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
                    error_results = _malformed_results(tool_blocks, malformed)

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
                        interrupted_results = self._interrupted_results(tool_blocks, partial_output)
                        await self._append(context, Message(role="assistant", content=list(content)))
                        await self._append(context, Message(role="user", content=list(interrupted_results)))
                        raise

                    # Append both messages atomically (assistant + tool results)
                    await self._append(context, Message(role="assistant", content=list(content)))
                    await self._append(context, Message(role="user", content=list(results)))

                    # Gemini stops generating (~20 tokens, end_turn) after receiving
                    # media as sibling inlineData parts alongside functionResponse.
                    # A "Proceed." user message nudges it to actually analyze the content.
                    if getattr(self.transport, "nudge_on_media_tool_result", False) and _carries_media(results):
                        await self._append(context, Message(role="user", content=[TextBlock(text=_MEDIA_NUDGE)]))

                    for event in _result_events(results, {b.id: b for b in tool_blocks}):
                        yield event
                    continue

                await self._append(context, Message(role="assistant", content=list(content)))

                match stop_reason:
                    case StopReason.end_turn:
                        logger.debug("End turn: total_usage=%s", total_usage)
                        yield SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case (
                        StopReason.refusal
                        | StopReason.cancelled
                        | StopReason.max_tokens
                        | StopReason.context_window_exceeded
                    ):
                        # Terminal, and none of them the transport breaking. Reported as an error a caller
                        # cannot tell them from a broken connection, and retries what can never work.
                        logger.info("Ending on %s: total_usage=%s", stop_reason, total_usage)
                        yield SessionEndEvent(stop_reason=stop_reason, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case StopReason.tool_use if content:
                        # No call survived, but the turn did add something. The next request differs from
                        # the last, so asking again is the recovery.
                        logger.warning("Asked for a tool and produced no call; re-prompting")
                        continue
                    case StopReason.tool_use:
                        # The turn added nothing at all, so the next request is byte-identical to the last.
                        # It ran to max_iterations: fifty paid requests for one malformed call.
                        yield Error(exception=RuntimeError("Transport asked for a tool but produced no call"))
                        yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case StopReason.pause_turn:
                        # The provider stopped its own tool loop and expects the assistant content back. It was
                        # just appended, so going round again is the resume.
                        logger.debug("Paused turn, resuming: total_usage=%s", total_usage)
                        continue
                    case _:
                        # Wildcard on purpose. Named one by one, a reason added later falls out of the
                        # match and the loop simply runs again until max_iterations.
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
