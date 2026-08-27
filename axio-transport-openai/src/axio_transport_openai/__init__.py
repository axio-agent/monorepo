"""OpenAI-compatible CompletionTransport via aiohttp."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Self

import aiohttp
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import (
    IterationEnd,
    IterationStart,
    ProviderEvent,
    ReasoningDelta,
    Refusal,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import CompletionTransport, EmbeddingTransport
from axio.types import StopReason, Usage
from axio_responses import Responses, convert_messages, convert_tools
from axio_sse import Payload, Wire, payloads

from .realtime import OpenAIRealtimeSession, OpenAIRealtimeTransport  # noqa: F401

logger = logging.getLogger(__name__)


_VT = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_VRT = frozenset({Capability.text, Capability.vision, Capability.reasoning, Capability.tool_use})
_RT = frozenset({Capability.text, Capability.reasoning, Capability.tool_use})
_TT = frozenset({Capability.text, Capability.tool_use})

OPENAI_MODELS: ModelRegistry = ModelRegistry(
    {
        # GPT-5.6 family (latest, 9 July 2026). Sizes and prices are the published ones. The tiers
        # differ only in price, except gpt-5.6-cyber, which has a smaller window.
        ModelSpec(
            id="gpt-5.6",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=4.0,
            output_cost=20.0,
        ),
        ModelSpec(
            id="gpt-5.6-sol",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=4.0,
            output_cost=20.0,
        ),
        ModelSpec(
            id="gpt-5.6-terra",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=2.0,
            output_cost=12.0,
        ),
        ModelSpec(
            id="gpt-5.6-luna",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=0.20,
            output_cost=1.20,
        ),
        ModelSpec(
            id="gpt-5.6-cyber",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=12.50,
            output_cost=75.0,
        ),
        # GPT-5.4 family (March 2026)
        ModelSpec(
            id="gpt-5.4",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VT,
            input_cost=10.0,
            output_cost=40.0,
        ),
        ModelSpec(
            id="gpt-5.4-mini",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=_VT,
            input_cost=1.5,
            output_cost=6.0,
        ),
        ModelSpec(
            id="gpt-5.4-nano",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=_TT,
            input_cost=0.30,
            output_cost=1.20,
        ),
        # GPT-5.x family
        ModelSpec(
            id="gpt-5.1",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=_VT,
            input_cost=5.0,
            output_cost=20.0,
        ),
        ModelSpec(
            id="gpt-5",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=_VT,
            input_cost=5.0,
            output_cost=20.0,
        ),
        ModelSpec(
            id="gpt-5-mini",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=_VT,
            input_cost=1.25,
            output_cost=5.0,
        ),
        ModelSpec(
            id="gpt-5-nano",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=_TT,
            input_cost=0.25,
            output_cost=1.0,
        ),
        # o-series reasoning models
        ModelSpec(
            id="o3",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=_RT,
            input_cost=10.0,
            output_cost=40.0,
        ),
        ModelSpec(
            id="o3-mini",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=_RT,
            input_cost=1.10,
            output_cost=4.40,
        ),
        ModelSpec(
            id="o4-mini",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=_RT,
            input_cost=1.10,
            output_cost=4.40,
        ),
        # GPT-4.1 family
        ModelSpec(
            id="gpt-4.1",
            context_window=1_047_576,
            max_output_tokens=32_768,
            capabilities=_VT,
            input_cost=2.0,
            output_cost=8.0,
        ),
        ModelSpec(
            id="gpt-4.1-mini",
            context_window=1_047_576,
            max_output_tokens=32_768,
            capabilities=_VT,
            input_cost=0.40,
            output_cost=1.60,
        ),
        ModelSpec(
            id="gpt-4.1-nano",
            context_window=1_047_576,
            max_output_tokens=32_768,
            capabilities=_TT,
            input_cost=0.10,
            output_cost=0.40,
        ),
        # GPT-4o family
        ModelSpec(
            id="gpt-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            capabilities=_VT,
            input_cost=2.50,
            output_cost=10.0,
        ),
        ModelSpec(
            id="gpt-4o-mini",
            context_window=128_000,
            max_output_tokens=16_384,
            capabilities=_VT,
            input_cost=0.15,
            output_cost=0.60,
        ),
    }
)

#: Every ``finish_reason`` the API publishes, plus the ones compatible servers add. A reason left
#: out of this map ends the run as an error rather than passing for a finished answer.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.end_turn,
    "tool_calls": StopReason.tool_use,
    "function_call": StopReason.tool_use,
    "length": StopReason.max_tokens,
    "content_filter": StopReason.refusal,
    "error": StopReason.error,
}


def _strip_title(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove pydantic 'title' keys from a JSON schema recursively."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if isinstance(value, dict):
            out[key] = _strip_title(value)
        elif isinstance(value, list):
            out[key] = [_strip_title(item) if isinstance(item, dict) else item for item in value]
        else:
            out[key] = value
    return out


def _extract_tool_result_text(tr: ToolResultBlock) -> str:
    """Extract text content from a ToolResultBlock (for APIs that don't support images in tool results)."""
    if isinstance(tr.content, str):
        return tr.content
    return "\n".join(b.text for b in tr.content if isinstance(b, TextBlock))


def _collect_tool_result_images(tool_results: list[ToolResultBlock]) -> list[dict[str, Any]]:
    """Collect image parts from tool results to inject as a follow-up user message."""
    parts: list[dict[str, Any]] = []
    for tr in tool_results:
        if isinstance(tr.content, list):
            images = [b for b in tr.content if isinstance(b, ImageBlock)]
            if images:
                parts.append({"type": "text", "text": f"[Image from tool call {tr.tool_use_id}]"})
                for img in images:
                    encoded = base64.b64encode(img.data).decode("ascii")
                    data_uri = f"data:{img.media_type};base64,{encoded}"
                    parts.append({"type": "image_url", "image_url": {"url": data_uri}})
    return parts


def _convert_messages(messages: list[Message], system: str) -> list[dict[str, Any]]:
    """Convert axio Message list to OpenAI message dicts."""
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        if msg.role == "user":
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results and len(tool_results) == len(msg.content):
                for tr in tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": _extract_tool_result_text(tr),
                        }
                    )
                # Chat Completions API doesn't support images in tool messages,
                # so inject them as a follow-up user message.
                image_parts = _collect_tool_result_images(tool_results)
                if image_parts:
                    result.append({"role": "user", "content": image_parts})
            else:
                has_images = any(isinstance(b, ImageBlock) for b in msg.content)
                if has_images:
                    content_parts: list[dict[str, Any]] = []
                    for b in msg.content:
                        if isinstance(b, TextBlock):
                            content_parts.append({"type": "text", "text": b.text})
                        elif isinstance(b, ImageBlock):
                            encoded = base64.b64encode(b.data).decode("ascii")
                            data_uri = f"data:{b.media_type};base64,{encoded}"
                            content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                    if content_parts:
                        result.append({"role": "user", "content": content_parts})
                else:
                    text_parts_u: list[str] = []
                    for b in msg.content:
                        if isinstance(b, TextBlock):
                            text_parts_u.append(b.text)
                    if text_parts_u:
                        result.append({"role": "user", "content": "".join(text_parts_u)})

        elif msg.role == "system":
            result.append(
                {
                    "role": "system",
                    "content": "".join(b.text for b in msg.content if isinstance(b, TextBlock)),
                }
            )

        elif msg.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for b in msg.content:
                if isinstance(b, TextBlock):
                    text_parts.append(b.text)
                elif isinstance(b, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": b.id,
                            "type": "function",
                            "function": {"name": b.name, "arguments": json.dumps(b.input)},
                        }
                    )

            entry: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                entry["content"] = "".join(text_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)

    return result


def _tool_key(tool: Any) -> str:
    """What makes two tool declarations the same one.

    The two endpoints put a function's name in different places. A hosted tool has no name at all
    and is identified by its type.
    """
    if not isinstance(tool, dict):
        return repr(tool)
    named = tool.get("name") or (tool.get("function") or {}).get("name")
    return f"function:{named}" if named else str(tool.get("type", ""))


def _convert_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio Tool list to OpenAI tool dicts."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


# ── The payload shapes a chat.completion.chunk carries ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PromptDetails(Wire):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class CompletionDetails(Wire):
    reasoning_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ChunkUsage(Wire):
    """Both slices arrive inside their totals here, so the reader adds nothing to either."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: PromptDetails = field(default_factory=PromptDetails)
    completion_tokens_details: CompletionDetails = field(default_factory=CompletionDetails)


@dataclass(frozen=True, slots=True)
class ToolFunction(Wire):
    name: str = ""
    #: None where the chunk carried no arguments at all, which is not the same as empty ones.
    arguments: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall(Wire):
    index: int = 0
    id: str = ""
    function: ToolFunction = field(default_factory=ToolFunction)


@dataclass(frozen=True, slots=True)
class ChunkDelta(Wire):
    #: None where the chunk carried no content key, which the API uses to mean "nothing this time".
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Vendor extensions, not in the OpenAI schema. OpenRouter and vLLM answer in ``reasoning``,
    #: DeepSeek in ``reasoning_content``. This transport asks for reasoning by sending
    #: ``enable_thinking``, so unread it was requested, billed and thrown away.
    reasoning: str | None = None
    reasoning_content: str | None = None
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class ChunkChoice(Wire):
    index: int = 0
    delta: ChunkDelta = field(default_factory=ChunkDelta)
    finish_reason: str | None = None
    logprobs: Payload = field(default_factory=Payload)
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class ChunkError(Wire):
    message: str = ""
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class CompletionChunk(Wire):
    """One SSE payload. The stream names no event, so every payload is this one shape."""

    id: str = ""
    model: str = ""
    choices: list[ChunkChoice] = field(default_factory=list)
    usage: ChunkUsage | None = None
    error: ChunkError | None = None


class ThinkTagParser:
    """Splits streaming content into reasoning (<think>...</think>) and text.

    Handles tags split across chunk boundaries via buffering.
    """

    __slots__ = ("_inside", "_buf")
    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Return list of (kind, text) where kind is 'reasoning' or 'text'."""
        self._buf += chunk
        result: list[tuple[str, str]] = []
        while True:
            tag = self._CLOSE if self._inside else self._OPEN
            pos = self._buf.find(tag)
            if pos != -1:
                before = self._buf[:pos]
                self._buf = self._buf[pos + len(tag) :]
                if before:
                    result.append(("reasoning" if self._inside else "text", before))
                self._inside = not self._inside
                continue
            # Check for partial tag prefix at end of buffer
            if self._could_be_partial(tag):
                break
            # No tag found and no partial - emit everything
            if self._buf:
                result.append(("reasoning" if self._inside else "text", self._buf))
                self._buf = ""
            break
        return result

    def flush(self) -> list[tuple[str, str]]:
        """Emit any remaining buffered content."""
        if self._buf:
            result = [("reasoning" if self._inside else "text", self._buf)]
            self._buf = ""
            return result
        return []

    def _could_be_partial(self, tag: str) -> bool:
        """Check if the end of buffer could be the start of *tag*."""
        for i in range(1, len(tag)):
            if self._buf.endswith(tag[:i]):
                return True
        return False


@dataclass(slots=True)
class OpenAITransport(CompletionTransport, EmbeddingTransport):
    name: str = "OpenAI"
    #: Which endpoint this server speaks. ``"responses"`` is the one that takes function tools and
    #: reasoning together. /v1/chat/completions refuses that pair for a model that reasons. OpenAI
    #: recommends ``"responses"`` for new work. Compatible servers rarely implement it, so the
    #: subclasses that point at them say ``"chat"``.
    api: Literal["responses", "chat"] = "responses"
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: ModelSpec = field(default_factory=lambda: OPENAI_MODELS["gpt-4.1-mini"])
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry(OPENAI_MODELS.values()))
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    max_retries: int = 10
    retry_base_delay: float = 5.0
    extra_params: Mapping[str, Any] = field(default=MappingProxyType({}), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.extra_params, MappingProxyType):
            self.extra_params = MappingProxyType(self.extra_params)

    def _get_retry_delay(self, resp: aiohttp.ClientResponse | None, attempt: int) -> float:
        """Return delay in seconds: prefer Retry-After header, fall back to exponential backoff."""
        if resp is not None:
            retry_after: str | None = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return max(0.0, float(retry_after))
                except (ValueError, TypeError):
                    pass
        return float(self.retry_base_delay * (2 ** (attempt - 1)))

    def build_chat_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.id,
            "messages": _convert_messages(messages, system),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self.model.max_output_tokens,
        }

        if tools:
            payload["tools"] = _convert_tools(tools)
            if Capability.reasoning in self.model.capabilities and "reasoning_effort" not in self.extra_params:
                # This endpoint refuses function tools beside any reasoning effort other than
                # "none". A reasoning model reasons by default. A request carrying tools therefore
                # fails with a 400 that names a parameter the caller never sent. The choice is made
                # plainly here rather than left to the provider, because the cost is real. The
                # model is paid for as a reasoning model and asked not to reason. /v1/responses
                # takes both.
                payload["reasoning_effort"] = "none"
                logger.warning(
                    "%s reasons and this request carries tools, which /v1/chat/completions refuses "
                    "together: reasoning_effort set to 'none'. Use /v1/responses, or pass "
                    "extra_params={'reasoning_effort': ...} to decide otherwise.",
                    self.model.id,
                )

        self._apply_extra(payload)
        return payload

    def _apply_extra(self, payload: dict[str, Any]) -> None:
        """Fold the caller's own parameters into the request.

        ``tools`` is merged rather than substituted. A caller adding a hosted tool — web search,
        code interpreter — would otherwise take away the function declarations the agent needs
        dispatched. The turn would then read as the model simply choosing to call nothing. A
        declaration whose name matches one already there wins, because the caller said it last.
        """
        if not self.extra_params:
            return
        extra = dict(self.extra_params)
        added = extra.pop("tools", None)
        payload.update(extra)
        if added is None:
            return
        replaced = {_tool_key(tool) for tool in added}
        kept = [tool for tool in payload.get("tools", []) if _tool_key(tool) not in replaced]
        payload["tools"] = [*kept, *added]

    def _path(self) -> str:
        return "responses" if self.api == "responses" else "chat/completions"

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        if self.api == "responses":
            return self.build_responses_payload(messages, tools, system)
        return self.build_chat_payload(messages, tools, system)

    def build_responses_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        """The request /v1/responses takes.

        The system prompt goes in ``instructions`` rather than in a message. Tool calls and their
        outputs are items beside the messages rather than blocks inside them. This endpoint takes
        tools and reasoning together, which is the reason to prefer it. /v1/chat/completions
        refuses that pair outright for a model that reasons.
        """
        instructions, items = convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model.id,
            "input": items,
            "stream": True,
            # Nothing is kept on the provider's side, because axio holds the conversation itself.
            "store": False,
            "max_output_tokens": self.model.max_output_tokens,
        }
        if instructions:
            payload["instructions"] = instructions
        if Capability.reasoning in self.model.capabilities:
            # Nothing is stored on the provider's side, so unless the reasoning comes back encrypted
            # there is nothing to send on the next round. The model then starts each round blind.
            payload["include"] = ["reasoning.encrypted_content"]
        if tools:
            payload["tools"] = convert_tools(tools)
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        self._apply_extra(payload)
        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        if self.api == "responses":
            async for event in self._parse_responses(resp):
                yield event
            return
        async for event in self._parse_chat(resp):
            yield event

    async def _parse_responses(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        """Read one Responses stream. The vocabulary lives in axio-responses."""
        turn = Responses()
        async for made in turn.over(resp.content.iter_any(), until="[DONE]"):
            yield made
        yield turn.finished()

    async def _parse_chat(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        tool_index_to_id: dict[int, str] = {}
        usage = Usage(0, 0)
        finish_reason: str | None = None
        error_message: str | None = None
        think_parser = ThinkTagParser()

        # payloads() dispatches what a stream that stopped without its last newline had collected.
        # The second copy of this loop that flushed the trailing buffer is therefore gone, and with
        # it the ways the two had drifted apart.
        served_by: str | None = None
        async for payload in payloads(resp.content.iter_any(), until="[DONE]"):
            data = CompletionChunk.read(payload)

            if served_by is None and data.model:
                # Which model actually answered, which need not be the one asked for. A compatible
                # gateway routes, falls back, and prices the substitute differently.
                served_by = data.model
                yield IterationStart(iteration=0, id=data.id or None, model=served_by)

            if data.error is not None:
                error_message = data.error.message or str(dict(data.error.raw))

            if data.usage is not None:
                # The slices are reported inside their totals here, so nothing is added.
                usage = Usage(
                    input_tokens=data.usage.prompt_tokens,
                    output_tokens=data.usage.completion_tokens,
                    cache_read_tokens=data.usage.prompt_tokens_details.cached_tokens,
                    cache_write_tokens=data.usage.prompt_tokens_details.cache_write_tokens,
                    reasoning_tokens=data.usage.completion_tokens_details.reasoning_tokens,
                )

            if not data.choices:
                continue

            choice = data.choices[0]
            delta = choice.delta

            thinking = delta.reasoning or delta.reasoning_content
            if thinking:
                yield ReasoningDelta(index=0, delta=thinking)

            if delta.refusal:
                # This is not a TextDelta, because as assistant text a refusal reads as an answer.
                finish_reason = finish_reason or "content_filter"
                yield Refusal(index=choice.index, text=delta.refusal, raw=dict(delta.raw))

            if delta.content is not None:
                for kind, text in think_parser.feed(delta.content):
                    if kind == "reasoning":
                        yield ReasoningDelta(index=0, delta=text)
                    else:
                        yield TextDelta(index=0, delta=text)

            for call in delta.tool_calls:
                if call.id:
                    tool_index_to_id[call.index] = call.id
                    yield ToolUseStart(index=call.index, tool_use_id=call.id, name=call.function.name)
                if call.function.arguments is not None:
                    yield ToolInputDelta(
                        index=call.index,
                        tool_use_id=tool_index_to_id.get(call.index, ""),
                        partial_json=call.function.arguments,
                    )

            if choice.logprobs:
                yield ProviderEvent(provider="openai", kind="logprobs", data=dict(choice.logprobs))

            # n>1 asks for several candidates. Only the first is read. The rest travel whole
            # rather than being discarded without a word.
            for other in data.choices[1:]:
                yield ProviderEvent(provider="openai", kind="choice", data=dict(other.raw), index=other.index)

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

        for kind, text in think_parser.flush():
            if kind == "reasoning":
                yield ReasoningDelta(index=0, delta=text)
            else:
                yield TextDelta(index=0, delta=text)

        stop = _STOP_REASON_MAP.get(finish_reason or "", StopReason.error)
        if finish_reason and finish_reason not in _STOP_REASON_MAP:
            logger.warning("Unknown finish_reason %r, mapped to %s", finish_reason, stop)
        logger.info(
            "Stream complete: stop_reason=%s, input_tokens=%d, output_tokens=%d",
            stop,
            usage.input_tokens,
            usage.output_tokens,
        )
        if stop == StopReason.error:
            msg = error_message or f"finish_reason={finish_reason!r}"
            raise StreamError(f"Provider error during streaming: {msg}")
        yield IterationEnd(iteration=0, stop_reason=stop, usage=usage)

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        return self._do_stream(messages, tools, system)

    async def _do_stream(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncIterator[StreamEvent]:
        assert self.session is not None, "session is required for streaming"
        url = f"{self.base_url.rstrip('/')}/{self._path()}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self.build_payload(messages, tools, system)

        logger.info(
            "POST %s model=%s messages=%d tools=%d",
            url,
            self.model.id,
            len(messages),
            len(tools),
        )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            dumped = json.dumps(payload, indent=2)
            if len(dumped) > 4000:
                dumped = dumped[:4000] + f"\n... truncated ({len(dumped)} chars total)"
            logger.debug("Request payload:\n%s", dumped)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            retry_resp: aiohttp.ClientResponse | None = None
            try:
                async with self.session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        async for event in self._parse_sse(resp):
                            yield event
                        return

                    body = await resp.text()
                    if resp.status == 429 or resp.status >= 500:
                        retry_resp = resp
                        last_exc = StreamError(f"OpenAI API error {resp.status}: {body}")
                        logger.warning(
                            "Retryable HTTP %d (attempt %d/%d): %s",
                            resp.status,
                            attempt,
                            self.max_retries,
                            body,
                        )
                    else:
                        logger.error("HTTP %d from %s: %s", resp.status, url, body)
                        raise StreamError(f"OpenAI API error {resp.status}: {body}")
            except aiohttp.ClientError as exc:
                last_exc = StreamError(str(exc))
                logger.warning("Connection error (attempt %d/%d): %s", attempt, self.max_retries, exc)

            if attempt < self.max_retries:
                delay = self._get_retry_delay(retry_resp, attempt)
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        raise last_exc or StreamError("Max retries exceeded")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI-compatible /v1/embeddings endpoint."""
        assert self.session is not None, "session is required for embedding"
        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {"model": self.model.id, "input": texts}

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            retry_resp: aiohttp.ClientResponse | None = None
            try:
                async with self.session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data: dict[str, Any] = await resp.json()
                        items = sorted(data["data"], key=lambda d: d["index"])
                        return [item["embedding"] for item in items]

                    body = await resp.text()
                    if resp.status == 429 or resp.status >= 500:
                        retry_resp = resp
                        last_exc = StreamError(f"Embedding API error {resp.status}: {body}")
                        logger.warning(
                            "Embedding retryable HTTP %d (attempt %d/%d): %s",
                            resp.status,
                            attempt,
                            self.max_retries,
                            body,
                        )
                    else:
                        raise StreamError(f"Embedding API error {resp.status}: {body}")
            except aiohttp.ClientError as exc:
                last_exc = StreamError(str(exc))
                logger.warning("Embedding connection error (attempt %d/%d): %s", attempt, self.max_retries, exc)

            if attempt < self.max_retries:
                delay = self._get_retry_delay(retry_resp, attempt)
                logger.info("Embedding retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        raise last_exc or StreamError("Embedding max retries exceeded")

    async def fetch_models(self) -> None:
        self.models = OPENAI_MODELS

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "models": [
                {
                    "id": m.id,
                    "context_window": m.context_window,
                    "max_output_tokens": m.max_output_tokens,
                    "capabilities": sorted(c.value for c in m.capabilities),
                    "input_cost": m.input_cost,
                    "output_cost": m.output_cost,
                }
                for m in self.models.values()
            ],
        }
        if self.extra_params:
            result["extra_params"] = dict(self.extra_params)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, session: aiohttp.ClientSession | None = None) -> Self:
        models = ModelRegistry(
            [
                ModelSpec(
                    id=str(m["id"]),
                    context_window=int(m.get("context_window", 128_000)),
                    max_output_tokens=int(m.get("max_output_tokens", 8_000)),
                    capabilities=frozenset(
                        Capability(c) for c in m.get("capabilities", []) if c in Capability.__members__
                    ),
                    input_cost=float(m.get("input_cost", 0.0)),
                    output_cost=float(m.get("output_cost", 0.0)),
                )
                for m in data.get("models", [])
            ]
        )
        return cls(
            name=str(data.get("name", "")),
            base_url=str(data.get("base_url", "")) or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=str(data.get("api_key", "")) or os.environ.get("OPENAI_API_KEY", ""),
            models=models,
            extra_params=dict(data.get("extra_params") or {}),
            session=session,
        )


class ThinkingMixin:
    """Mixin for OpenAI-compatible transports whose providers support enable_thinking.

    Providers like Nebius (Qwen models) and OpenRouter require an explicit
    ``enable_thinking: true`` request parameter to activate chain-of-thought
    reasoning.  Declare ``thinking: bool = False`` in the concrete dataclass,
    then mix this in.  That gives automatic payload injection and
    to_dict/from_dict round-trip support.
    """

    __slots__ = ()

    thinking: bool  # declared in the concrete dataclass subclass

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        payload: dict[str, Any] = super().build_payload(messages, tools, system)  # type: ignore[misc]
        if self.thinking and Capability.reasoning in self.model.capabilities and "enable_thinking" not in payload:  # type: ignore[attr-defined]
            payload["enable_thinking"] = True
        return payload

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = super().to_dict()  # type: ignore[misc]
        if self.thinking:
            d["thinking"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, session: aiohttp.ClientSession | None = None) -> Self:
        obj = super().from_dict(data, session=session)  # type: ignore[misc]
        return dataclasses.replace(obj, thinking=bool(data.get("thinking", False)))  # type: ignore[no-any-return]
