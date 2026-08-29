"""Anthropic Claude CompletionTransport via aiohttp (direct API and Vertex AI)."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, Self, cast

import aiohttp
from axio.blocks import (
    ImageBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
    proof,
    replayable,
)
from axio.events import (
    BlockEnd,
    Citation,
    IterationEnd,
    IterationStart,
    ProviderEvent,
    ProviderOutput,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.retry import is_retryable, retry_delay
from axio.schema import strip_title
from axio.tool import Tool
from axio.transport import CompletionTransport
from axio.types import StopReason, Usage, stop_reason_from
from axio_sse import EVENT_NAME, Payload, Reader, Wire, on

logger = logging.getLogger(__name__)

#: What this protocol is called wherever its name is written: on the proofs it issues and on the
#: events it forwards. Anthropic on Vertex AI speaks the same protocol, so it says this too.
PROVIDER: Final = "anthropic"

ANTHROPIC_API_VERSION = "2023-06-01"
VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"

_VT = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_RT = frozenset({Capability.text, Capability.vision, Capability.reasoning, Capability.tool_use})


class _RefreshableCredentials(Protocol):
    token: str | None

    def refresh(self, request: object) -> None: ...


ANTHROPIC_MODELS: ModelRegistry = ModelRegistry(
    {
        ModelSpec(
            id="claude-opus-4-6",
            context_window=1_000_000,
            max_output_tokens=128_000,
            capabilities=_RT,
            input_cost=5.0,
            output_cost=25.0,
        ),
        ModelSpec(
            id="claude-sonnet-4-6",
            context_window=1_000_000,
            max_output_tokens=64_000,
            capabilities=_RT,
            input_cost=3.0,
            output_cost=15.0,
        ),
        ModelSpec(
            id="claude-haiku-4-5",
            context_window=200_000,
            max_output_tokens=64_000,
            capabilities=_RT,
            input_cost=1.0,
            output_cost=5.0,
        ),
        ModelSpec(
            id="claude-haiku-4-5-20251001",
            context_window=200_000,
            max_output_tokens=64_000,
            capabilities=_RT,
            input_cost=1.0,
            output_cost=5.0,
        ),
        ModelSpec(
            id="claude-opus-4-5",
            context_window=200_000,
            max_output_tokens=64_000,
            capabilities=_RT,
            input_cost=5.0,
            output_cost=25.0,
        ),
        ModelSpec(
            id="claude-sonnet-4-5",
            context_window=200_000,
            max_output_tokens=64_000,
            capabilities=_RT,
            input_cost=3.0,
            output_cost=15.0,
        ),
    }
)

#: The smallest ``budget_tokens`` the API accepts. It also refuses a budget at or above
#: ``max_tokens``, which this transport always sets to the model's ceiling.
_MIN_THINKING = 1024

#: What stands in for an assistant turn whose blocks were all stripped. The API refuses two user
#: turns in a row, so the turn has to say something.
_EMPTY_TURN = "(no content)"

#: Every ``stop_reason`` the API publishes. One left out ends the run as an error.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.end_turn,
    "stop_sequence": StopReason.end_turn,
    "tool_use": StopReason.tool_use,
    "max_tokens": StopReason.max_tokens,
    "refusal": StopReason.refusal,
    # Resumable only if the server-tool content is stored, and it reaches the caller as
    # ProviderEvent. This transport declares no server tool, so the reason cannot arrive.
    "pause_turn": StopReason.pause_turn,
    "model_context_window_exceeded": StopReason.context_window_exceeded,
}

#: What each citation shape counts its span in. Only ``char_location`` counts characters.
_CITATION_UNITS: dict[str, Literal["char", "byte", "page", "block", "unknown"]] = {
    "char_location": "char",
    "page_location": "page",
    "content_block_location": "block",
    "search_result_location": "block",
    "web_search_result_location": "char",
}


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert axio Message list to Anthropic messages."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        content_parts: list[dict[str, Any]] = []

        if msg.role == "user":
            for b in msg.content:
                if isinstance(b, TextBlock):
                    content_parts.append({"type": "text", "text": b.text})
                elif isinstance(b, (ImageBlock, VideoBlock)):
                    encoded = base64.b64encode(b.data).decode("ascii")
                    content_parts.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": b.media_type, "data": encoded},
                        }
                    )
                elif isinstance(b, ToolResultBlock):
                    if isinstance(b.content, str):
                        tr_content: str | list[dict[str, Any]] = b.content
                    else:
                        tr_content = [
                            {"type": "text", "text": item.text}
                            if isinstance(item, TextBlock)
                            else {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.media_type,
                                    "data": base64.b64encode(item.data).decode("ascii"),
                                },
                            }
                            for item in b.content
                        ]
                    entry: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": b.tool_use_id,
                        "content": tr_content,
                    }
                    if b.is_error:
                        entry["is_error"] = True
                    content_parts.append(entry)

        elif msg.role == "assistant":
            for b in msg.content:
                if isinstance(b, TextBlock):
                    content_parts.append({"type": "text", "text": b.text})
                elif isinstance(b, ReasoningBlock):
                    # Unaltered, or not at all: the API checks the signature it issued. `proof`
                    # also leaves out one another provider issued, which this API never made and
                    # would read as its own.
                    signed = proof(b, PROVIDER)
                    if b.redacted and signed:
                        content_parts.append({"type": "redacted_thinking", "data": signed})
                    elif signed:
                        content_parts.append({"type": "thinking", "thinking": b.text, "signature": signed})
                    else:
                        # Unsigned, so the API would refuse it. Nothing proves the text is the model's.
                        logger.debug("Dropping an unsigned reasoning block from the replayed turn")
                elif isinstance(b, ToolUseBlock):
                    content_parts.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                elif isinstance(b, ProviderBlock):
                    # Back exactly as it arrived. This API keeps no copy of the turn, so a search
                    # it ran or code it executed is in the next request only if we put it there.
                    if replayable(b, PROVIDER):
                        content_parts.append(dict(b.data))

        if not content_parts and msg.role == "assistant":
            # Every block was stripped. Skipped, two user turns end up adjacent, which the API
            # refuses for the rest of the session.
            content_parts = [{"type": "text", "text": _EMPTY_TURN}]
        if content_parts:
            result.append({"role": msg.role, "content": content_parts})

    return result


def _convert_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio Tool list to Anthropic tool dicts."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": strip_title(tool.input_schema),
            # Stream tool input deltas as they're generated instead of buffering.
            # May produce truncated JSON if max_tokens is reached mid-call.
            "eager_input_streaming": True,
        }
        for tool in tools
    ]


def _google_auth_available() -> bool:
    """Return whether the Vertex AI credential dependencies are importable."""
    try:
        return all(importlib.util.find_spec(name) is not None for name in ("google.auth", "requests"))
    except Exception:
        # Partially installed namespace packages can make find_spec raise different exception types.
        return False


def _get_vertex_access_token() -> str:
    import google.auth
    import google.auth.transport.requests

    creds_obj, _project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds = cast(_RefreshableCredentials, creds_obj)
    creds.refresh(google.auth.transport.requests.Request())
    if not creds.token:
        raise RuntimeError("Google credentials did not return an access token")
    return creds.token


# ── The payload shapes the Messages API sends ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OutputDetails(Wire):
    thinking_tokens: int = 0


@dataclass(frozen=True, slots=True)
class MessageUsage(Wire):
    """The cache counts stand OUTSIDE ``input_tokens``, which holds only what follows the last
    cache breakpoint. The API states the arithmetic itself:
    ``total = cache_read + cache_creation + input_tokens``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens_details: OutputDetails = field(default_factory=OutputDetails)


@dataclass(frozen=True, slots=True)
class MessageObject(Wire):
    id: str = ""
    model: str = ""
    usage: MessageUsage = field(default_factory=MessageUsage)


@dataclass(frozen=True, slots=True)
class ContentBlock(Wire):
    type: str = ""
    id: str = ""
    name: str = ""
    #: The opaque reasoning of a ``redacted_thinking`` block, which carries no text at all.
    data: str = ""
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class CitationObject(Wire):
    """One attribution. It arrives under five location shapes that name their span differently, so
    the fields worth reading are declared. The whole object travels in ``raw``."""

    type: str = ""
    cited_text: str = ""
    document_title: str | None = None
    title: str | None = None
    url: str | None = None
    document_index: int | None = None
    start_char_index: int | None = None
    end_char_index: int | None = None
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class BlockDelta(Wire):
    """Every delta the format defines, in one shape. The ``type`` says which field was filled."""

    type: str = ""
    text: str = ""
    thinking: str = ""
    signature: str = ""
    partial_json: str = ""
    citation: CitationObject = field(default_factory=CitationObject)


@dataclass(frozen=True, slots=True)
class StopDetails(Wire):
    """Why the model declined. Null for every stop reason other than ``refusal``. Both fields are
    null where the decline maps to no named category."""

    type: str = ""
    category: str = ""
    #: Human-readable, and documented as unstable. Show it, never parse it.
    explanation: str = ""
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class MessageDeltaObject(Wire):
    stop_reason: str = ""
    stop_sequence: str | None = None
    stop_details: StopDetails = field(default_factory=StopDetails)


@dataclass(frozen=True, slots=True)
class ErrorObject(Wire):
    type: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class MessageStart(Wire, name="message_start"):
    message: MessageObject = field(default_factory=MessageObject)


@dataclass(frozen=True, slots=True)
class BlockStart(Wire, name="content_block_start"):
    index: int = 0
    content_block: ContentBlock = field(default_factory=ContentBlock)


@dataclass(frozen=True, slots=True)
class BlockDeltaEvent(Wire, name="content_block_delta"):
    index: int = 0
    delta: BlockDelta = field(default_factory=BlockDelta)


@dataclass(frozen=True, slots=True)
class BlockStop(Wire, name="content_block_stop"):
    index: int = 0


@dataclass(frozen=True, slots=True)
class MessageDeltaEvent(Wire, name="message_delta"):
    delta: MessageDeltaObject = field(default_factory=MessageDeltaObject)
    usage: MessageUsage = field(default_factory=MessageUsage)


@dataclass(frozen=True, slots=True)
class StreamFailure(Wire, name="error"):
    error: ErrorObject = field(default_factory=ErrorObject)


class Messages(Reader[StreamEvent], by=EVENT_NAME):
    """Every event the Messages API sends, and what each one becomes.

    The format names each event in its own ``event:`` field, so this reader dispatches on that
    rather than on anything inside the payload. The eight names below are the whole published
    vocabulary. A ninth would be news, which is what a test reading with ``strict=True`` holds
    against it.

    One instance reads one turn. The token counts and the index-to-id map are that turn's state.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_write = 0
        self.reasoning_tokens = 0
        self.stop_reason = ""
        # content_block_delta carries the index and never the tool id.
        self.tool_use_ids: dict[int, str] = {}
        # Blocks the API produced from its own tools, by index, until the block closes. This API
        # keeps no copy of the turn, so each one has to go back on the next request.
        self.hosted: dict[int, tuple[str, Payload]] = {}
        self.hosted_input: dict[int, list[str]] = {}

    # ── what reaches the caller ──────────────────────────────────────────────────────────────

    @on(MessageStart)
    def _started(self, wire: MessageStart) -> Iterator[StreamEvent]:
        usage = wire.message.usage
        self.cache_read = usage.cache_read_input_tokens
        self.cache_write = usage.cache_creation_input_tokens
        # input_tokens counts only what follows the last cache breakpoint, which this sets.
        self.input_tokens = usage.input_tokens + self.cache_read + self.cache_write
        yield IterationStart(iteration=0, id=wire.message.id or None, model=wire.message.model or None)

    @on(BlockStart)
    def _block_started(self, wire: BlockStart) -> Iterator[StreamEvent]:
        block = wire.content_block
        if block.type == "tool_use":
            self.tool_use_ids[wire.index] = block.id
            yield ToolUseStart(index=wire.index, tool_use_id=block.id, name=block.name)
        elif block.type == "redacted_thinking":
            # Only the proof travels. The API refuses a thinking block without one.
            yield ReasoningSignature(index=wire.index, signature=block.data, redacted=True, provider=PROVIDER)
        elif block.type not in ("text", "thinking"):
            # server_tool_use, web_search_tool_result, code execution, mcp: run on the API's side.
            # Kept until the block closes, because a server_tool_use streams its input in deltas
            # and is not whole yet. Forwarded as news too: a caller watching the turn wants to see
            # the search start, not only that it happened.
            self.hosted[wire.index] = (block.type, block.raw)
            yield ProviderEvent(provider=PROVIDER, kind=block.type, data=dict(block.raw), index=wire.index)

    @on(BlockDeltaEvent)
    def _block_delta(self, wire: BlockDeltaEvent) -> Iterator[StreamEvent]:
        delta, index = wire.delta, wire.index
        match delta.type:
            case "text_delta":
                yield TextDelta(index=index, delta=delta.text)
            case "thinking_delta":
                yield ReasoningDelta(index=index, delta=delta.thinking)
            case "signature_delta":
                yield ReasoningSignature(index=index, signature=delta.signature, provider=PROVIDER)
            case "input_json_delta" if index in self.hosted:
                # The arguments of a tool the API runs itself. They belong to the block being
                # assembled for replay, not to a call this side will ever dispatch.
                self.hosted_input.setdefault(index, []).append(delta.partial_json)
            case "input_json_delta":
                yield ToolInputDelta(
                    index=index,
                    tool_use_id=self.tool_use_ids.get(index, ""),
                    partial_json=delta.partial_json,
                )
            case "citations_delta":
                yield self._citation(index, delta.citation)
            case other:
                # The delta type is a second vocabulary inside one event, under one policy.
                self.unknown(other)

    @on(BlockStop)
    def _block_stopped(self, wire: BlockStop) -> Iterator[StreamEvent]:
        """The block is complete, so anything accumulated for it now parses."""
        if (found := self.hosted.pop(wire.index, None)) is not None:
            if (item := self._hosted_item(wire.index, *found)) is not None:
                yield ProviderOutput(
                    provider=PROVIDER,
                    kind=found[0],
                    data=item,
                    index=wire.index,
                    id=item.get("id", "") if isinstance(item.get("id"), str) else "",
                )
        yield BlockEnd(index=wire.index)

    def _hosted_item(self, index: int, kind: str, start: Payload) -> dict[str, Any] | None:
        """One finished block from a tool the API ran, as it will be sent back.

        The block opens with its shape and, for a ``server_tool_use``, an empty ``input`` that the
        deltas fill. Stored from the opening payload alone, the call goes back with no arguments
        and the API refuses the result that follows it.
        """
        item = dict(start)
        if (parts := self.hosted_input.pop(index, None)) is None:
            return item
        raw = "".join(parts)
        try:
            item["input"] = json.loads(raw)
        except json.JSONDecodeError:
            # Truncated arguments. Replayed, the API is handed a call the model never finished
            # writing; dropped, the turn simply does not carry that block.
            logger.warning("Dropping a %s block whose arguments did not parse (%d chars)", kind, len(raw))
            return None
        return item

    # ── what only moves this turn's state ────────────────────────────────────────────────────

    @on(MessageDeltaEvent)
    def _message_delta(self, wire: MessageDeltaEvent) -> Iterator[StreamEvent]:
        # An empty reason means "none yet". It must not erase the reason an earlier delta gave.
        self.stop_reason = wire.delta.stop_reason or self.stop_reason

        # Cumulative in every field, not only the output ones. Reading back the output alone left
        # the input frozen at what message_start had said.
        if wire.usage.input_tokens:
            self.cache_read = wire.usage.cache_read_input_tokens or self.cache_read
            self.cache_write = wire.usage.cache_creation_input_tokens or self.cache_write
            self.input_tokens = wire.usage.input_tokens + self.cache_read + self.cache_write
        # Thinking is already inside output_tokens. The API documents "output_tokens -
        # thinking_tokens" as the non-reasoning output.
        self.output_tokens = wire.usage.output_tokens or self.output_tokens
        self.reasoning_tokens = wire.usage.output_tokens_details.thinking_tokens or self.reasoning_tokens

        if wire.delta.stop_reason == "refusal":
            # A decline arrives as a successful response with no content at all.
            details = wire.delta.stop_details
            yield Refusal(
                index=0,
                text=details.explanation,
                category=details.category or None,
                raw=dict(wire.delta.stop_details.raw),
            )

    @on("message_stop", "ping")
    def _quiet(self, payload: Payload) -> None:
        """Arrive every turn and carry nothing. Named so strict fires only on something new."""

    def unmatched(self, name: str, payload: Payload) -> Iterator[StreamEvent]:
        """Anything this reader does not interpret, passed on rather than dropped.

        The eight names above are the whole published vocabulary today, so nothing reaches here
        yet. When the API adds a ninth it arrives under its own name instead of disappearing.
        """
        yield ProviderEvent(provider=PROVIDER, kind=name, data=dict(payload))

    # ── what ends the turn ───────────────────────────────────────────────────────────────────

    @on(StreamFailure)
    def _failed(self, wire: StreamFailure) -> None:
        raise StreamError(f"Anthropic error: {wire.error.type or 'unknown'}: {wire.error.message}")

    # ── the turn, once it is over ────────────────────────────────────────────────────────────

    @staticmethod
    def _citation(index: int, citation: CitationObject) -> Citation:
        """One attribution, whichever of the five location shapes it arrived in."""
        return Citation(
            index=index,
            cited_text=citation.cited_text,
            title=citation.title or citation.document_title,
            url=citation.url,
            source_id=str(citation.document_index) if citation.document_index is not None else None,
            start=citation.start_char_index,
            end=citation.end_char_index,
            # Only the char_location shape counts characters. The others count pages or blocks and say
            # so in their own type.
            unit=_CITATION_UNITS.get(citation.type, "unknown"),
            raw=dict(citation.raw),
        )

    def finished(self) -> IterationEnd:
        """What the turn added up to. The API sends no event that means this."""
        if not self.stop_reason:
            # Every turn ends on a message_delta carrying a stop_reason. Without one the connection
            # was cut.
            raise StreamError("Anthropic stream ended without a stop_reason")
        stop = stop_reason_from(self.stop_reason, _STOP_REASON_MAP, provider="Anthropic")
        usage = Usage.reported(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read,
            cache_write_tokens=self.cache_write,
            reasoning_tokens=self.reasoning_tokens,
        )
        logger.info(
            "Stream complete: stop_reason=%s, input_tokens=%d, output_tokens=%d",
            stop,
            self.input_tokens,
            self.output_tokens,
        )
        return IterationEnd(iteration=0, stop_reason=stop, usage=usage)


@dataclass(slots=True)
class AnthropicTransport(CompletionTransport):
    name: str = "Anthropic"
    base_url: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    vertexai: bool = False
    project: str = ""
    location: str = ""
    model: ModelSpec = field(default_factory=lambda: ANTHROPIC_MODELS["claude-sonnet-4-6"])
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry(ANTHROPIC_MODELS.values()))
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    thinking_budget: int | None = None
    max_retries: int = 10
    retry_base_delay: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.vertexai, str):
            self.vertexai = self.vertexai.lower() in ("true", "1")
        if self.vertexai and not _google_auth_available():
            raise ImportError(
                "vertexai=True requires google-auth[requests]. Install it, or use the direct Anthropic API."
            )

    def _build_url(self) -> str:
        if self.vertexai:
            project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            location = self.location or os.environ.get("GOOGLE_CLOUD_LOCATION", "") or "global"
            if not project:
                raise StreamError(
                    "Anthropic on Vertex AI requires a project. "
                    "Set GOOGLE_CLOUD_PROJECT or configure it in transport settings."
                )
            host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            bare = self.model.id.removeprefix("anthropic/")
            return (
                f"https://{host}/v1/"
                f"projects/{project}/locations/{location}/"
                f"publishers/anthropic/models/{bare}:streamRawPredict"
            )
        return f"{self.base_url.rstrip('/')}/messages"

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"content-type": "application/json"}
        if self.vertexai:
            token = _get_vertex_access_token()
            headers["Authorization"] = f"Bearer {token}"
        else:
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = ANTHROPIC_API_VERSION
        return headers

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        converted_messages = _convert_messages(messages)

        payload: dict[str, Any] = {
            "messages": converted_messages,
            "stream": True,
            "max_tokens": self.model.max_output_tokens,
        }

        if self.vertexai:
            payload["anthropic_version"] = VERTEX_ANTHROPIC_VERSION
        else:
            payload["model"] = self.model.id

        system_blocks: list[dict[str, Any]] = []
        if system:
            system_blocks.append({"type": "text", "text": system, "cache_control": {"type": "ephemeral"}})
        for msg in messages:
            if msg.role == "system":
                text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
                if text:
                    system_blocks.append({"type": "text", "text": text})
        if system_blocks:
            payload["system"] = system_blocks

        if tools:
            converted = _convert_tools(tools)
            converted[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = converted

        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.thinking_budget is not None:
            budget = max(_MIN_THINKING, min(self.thinking_budget, self.model.max_output_tokens - 1))
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        """Read one Messages stream into axio StreamEvents."""
        turn = Messages()
        async for made in turn.over(resp.content.iter_any()):
            yield made
        yield turn.finished()

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        return self._do_stream(messages, tools, system)

    async def _do_stream(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncIterator[StreamEvent]:
        assert self.session is not None, "session is required for streaming"
        url = self._build_url()
        headers = self._build_headers()
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
        sent = False
        for attempt in range(1, self.max_retries + 1):
            retry_resp: aiohttp.ClientResponse | None = None
            try:
                async with self.session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        async for event in self._parse_sse(resp):
                            sent = True
                            yield event
                        return

                    body = await resp.text()
                    if is_retryable(resp.status):
                        retry_resp = resp
                        last_exc = StreamError(f"Anthropic API error {resp.status}: {body}")
                        logger.warning(
                            "Retryable HTTP %d (attempt %d/%d): %s",
                            resp.status,
                            attempt,
                            self.max_retries,
                            body,
                        )
                    else:
                        logger.error("HTTP %d from %s: %s", resp.status, url, body)
                        raise StreamError(f"Anthropic API error {resp.status}: {body}")
            except aiohttp.ClientError as exc:
                last_exc = StreamError(str(exc))
                logger.warning("Connection error (attempt %d/%d): %s", attempt, self.max_retries, exc)

            if sent:
                # The caller has already seen events from this attempt. Going round again re-POSTs
                # and replays them: a tool ran twice, and its text was stored twice.
                raise last_exc or StreamError("Stream failed after events reached the caller")
            if attempt < self.max_retries:
                delay = retry_delay(retry_resp, attempt, base=self.retry_base_delay)
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        raise last_exc or StreamError("Max retries exceeded")

    async def fetch_models(self) -> None:
        self.models = ANTHROPIC_MODELS

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
        if self.vertexai:
            d["vertexai"] = True
            if self.project:
                d["project"] = self.project
            if self.location:
                d["location"] = self.location
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, session: aiohttp.ClientSession | None = None) -> Self:
        models = ModelRegistry(
            [
                ModelSpec(
                    id=str(m["id"]),
                    context_window=int(m.get("context_window", 200_000)),
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
            base_url=str(data.get("base_url", ""))
            or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            api_key=str(data.get("api_key", "")) or os.environ.get("ANTHROPIC_API_KEY", ""),
            vertexai=bool(data.get("vertexai", False)),
            project=str(data.get("project", "")),
            location=str(data.get("location", "")),
            temperature=float(data["temperature"]) if data.get("temperature") is not None else None,
            top_p=float(data["top_p"]) if data.get("top_p") is not None else None,
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            thinking_budget=int(data["thinking_budget"]) if data.get("thinking_budget") is not None else None,
            models=models,
            session=session,
        )
