"""Google GenAI (Gemini) transport — aiohttp streaming, SDK for media generation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import aiohttp
from axio.blocks import (
    AudioBlock,
    AudioMediaType,
    ImageBlock,
    ImageMediaType,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
    VideoMediaType,
)
from axio.events import (
    AudioOutput,
    ImageOutput,
    IterationEnd,
    IterationStart,
    ProviderEvent,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    StreamEvent,
    TextDelta,
    TextSignature,
    ToolInputDelta,
    ToolUseStart,
    VideoOutput,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.retry import is_retryable, retry_delay
from axio.schema import strip_title
from axio.tool import Tool
from axio.transport import CompletionTransport, ImageGenTransport, VideoGenTransport
from axio.types import StopReason, Usage, stop_reason_from
from axio_sse import Payload, Wire, payloads

from axio_transport_google._generated_types import (
    Content,
    FunctionDeclaration,
    GenerateContentRequest,
    GenerationConfig,
    Part,
    ThinkingConfig,
)
from axio_transport_google._generated_types import (
    SafetySetting as SafetySettingDict,
)
from axio_transport_google._generated_types import (
    Tool as ToolDict,
)

logger = logging.getLogger(__name__)


class _RefreshableCredentials(Protocol):
    valid: bool
    expired: bool
    token: str | None

    def refresh(self, request: object) -> None: ...


# ── Thinking level helpers ──────────────────────────────────────────


def valid_thinking_levels(model_id: str) -> tuple[str, ...] | None:
    """Return valid thinkingLevel values for a Gemini 3+ model, or None for budget-based (2.5) models."""
    if "gemini-3" not in model_id:
        return None
    if "-pro-image" in model_id:
        return ("HIGH",)
    if "-pro" in model_id:
        return ("LOW", "MEDIUM", "HIGH")
    if "-flash-image" in model_id:
        return ("MINIMAL", "HIGH")
    # Flash, Flash-Lite
    return ("MINIMAL", "LOW", "MEDIUM", "HIGH")


def _redact_body(obj: Any) -> Any:
    """Deep-copy a request/response dict, replacing large base64 blobs with a size summary."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, str) and len(v) > 200:
                out[k] = f"<{len(v)} chars base64>"
            else:
                out[k] = _redact_body(v)
        return out
    if isinstance(obj, list):
        return [_redact_body(x) for x in obj]
    return obj


# Capability sets for Gemini models
from .realtime import GeminiLiveSession, GeminiLiveTransport  # noqa: F401,E402

_VT = frozenset({Capability.text, Capability.vision, Capability.audio, Capability.video, Capability.tool_use})
_RT = frozenset(
    {Capability.text, Capability.reasoning, Capability.vision, Capability.audio, Capability.video, Capability.tool_use}
)
_IMG = frozenset({Capability.text, Capability.vision, Capability.image_generation})

GENAI_MODELS: ModelRegistry = ModelRegistry(
    {
        # --- Gemini chat/reasoning models ---
        ModelSpec(
            id="gemini-3.1-pro-preview",
            context_window=1_048_576,
            max_output_tokens=65_536,
            capabilities=_RT,
            input_cost=2.0,
            output_cost=12.0,
        ),
        ModelSpec(
            id="gemini-3-flash-preview",
            context_window=1_048_576,
            max_output_tokens=65_536,
            capabilities=_RT,
            input_cost=0.50,
            output_cost=3.0,
        ),
        ModelSpec(
            id="gemini-3.1-flash-lite-preview",
            context_window=1_048_576,
            max_output_tokens=65_536,
            capabilities=_RT,
            input_cost=0.25,
            output_cost=1.50,
        ),
        # --- Nano Banana (Gemini image generation via generateContent) ---
        ModelSpec(
            id="gemini-3.1-flash-image-preview",
            context_window=1_048_576,
            max_output_tokens=8_192,
            capabilities=_IMG,
        ),
        ModelSpec(
            id="gemini-3-pro-image-preview",
            context_window=1_048_576,
            max_output_tokens=8_192,
            capabilities=_IMG,
        ),
    }
)


def _get_anthropic_models() -> ModelRegistry:
    """Get Anthropic models with 'anthropic/' prefix for Vertex AI routing."""
    from axio_transport_anthropic import ANTHROPIC_MODELS

    return ModelRegistry(
        {
            ModelSpec(
                id=f"anthropic/{spec.id}",
                context_window=spec.context_window,
                max_output_tokens=spec.max_output_tokens,
                capabilities=spec.capabilities,
                input_cost=spec.input_cost,
                output_cost=spec.output_cost,
            )
            for spec in ANTHROPIC_MODELS.values()
        }
    )


#: Reasons that say the turn failed. A call streamed inside one is not a request to run it.
_BLOCKED = frozenset({StopReason.refusal, StopReason.error, StopReason.cancelled})

#: Every ``finishReason`` both surfaces publish: 21 on the developer API, 17 on Vertex.
#: One left out is read as an error.
_FINISH_REASON_MAP: dict[str, StopReason] = {
    "STOP": StopReason.end_turn,
    "MAX_TOKENS": StopReason.max_tokens,
    # Blocked, not broken. The same prompt sent again cannot succeed.
    "SAFETY": StopReason.refusal,
    "RECITATION": StopReason.refusal,
    "LANGUAGE": StopReason.refusal,
    "BLOCKLIST": StopReason.refusal,
    "PROHIBITED_CONTENT": StopReason.refusal,
    "SPII": StopReason.refusal,
    "MODEL_ARMOR": StopReason.refusal,
    "IMAGE_SAFETY": StopReason.refusal,
    "IMAGE_PROHIBITED_CONTENT": StopReason.refusal,
    "IMAGE_RECITATION": StopReason.refusal,
    # The API could not use the call. Read as tool_use so the agent prompts again.
    "MALFORMED_FUNCTION_CALL": StopReason.tool_use,
    "UNEXPECTED_TOOL_CALL": StopReason.tool_use,
    # Failures that prompting again does not fix.
    "TOO_MANY_TOOL_CALLS": StopReason.error,
    "MISSING_THOUGHT_SIGNATURE": StopReason.error,
    "MALFORMED_RESPONSE": StopReason.error,
    "IMAGE_OTHER": StopReason.error,
    "NO_IMAGE": StopReason.error,
    "OTHER": StopReason.error,
    "ESCALATION": StopReason.error,
    "FINISH_REASON_UNSPECIFIED": StopReason.error,
}

_DEVELOPER_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# API reference (discovery docs):
#   https://aiplatform.googleapis.com/$discovery/rest?version=v1
#   https://aiplatform.googleapis.com/$discovery/rest?version=v1beta1


# ── The payload shapes streamGenerateContent sends ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InlineData(Wire):
    mimeType: str = ""
    data: str = ""


@dataclass(frozen=True, slots=True)
class FunctionCall(Wire):
    id: str = ""
    name: str = ""
    args: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class ContentPart(Wire):
    """One piece of a candidate's content. Which field is filled says what it is."""

    text: str = ""
    #: True where ``text`` is the model thinking rather than answering.
    thought: bool = False
    #: Opaque proof that the reasoning is the model's own. Altered or missing, the next request
    #: fails with ``MISSING_THOUGHT_SIGNATURE``.
    thoughtSignature: str = ""
    inlineData: InlineData = field(default_factory=InlineData)
    functionCall: FunctionCall = field(default_factory=FunctionCall)
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class CandidateContent(Wire):
    role: str = ""
    parts: list[ContentPart] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Candidate(Wire):
    content: CandidateContent = field(default_factory=CandidateContent)
    finishReason: str = ""
    citationMetadata: Payload = field(default_factory=Payload)
    groundingMetadata: Payload = field(default_factory=Payload)
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class UsageMetadata(Wire):
    """Two of these stand outside the headline number. One is already inside it.

    ``cachedContentTokenCount`` is part of ``promptTokenCount``. ``toolUsePromptTokenCount`` and
    ``thoughtsTokenCount`` are not part of anything and have to be added.
    """

    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    toolUsePromptTokenCount: int = 0
    thoughtsTokenCount: int = 0
    cachedContentTokenCount: int = 0
    totalTokenCount: int | None = None


@dataclass(frozen=True, slots=True)
class GenerateContentChunk(Wire):
    """One SSE payload. The stream names no event, so every payload is this one shape."""

    candidates: list[Candidate] = field(default_factory=list)
    usageMetadata: UsageMetadata = field(default_factory=UsageMetadata)
    promptFeedback: Payload = field(default_factory=Payload)
    modelVersion: str = ""
    responseId: str = ""


def _usage(um: UsageMetadata, *, final: bool = False) -> Usage:
    """Gemini's token counts, converted to inclusive totals.

    Two of these stand outside the headline number and have to be added. Tool-use prompt tokens are
    not in ``promptTokenCount``. Thinking is not in ``candidatesTokenCount``. Cached content is the
    other way round and is already inside the prompt count. Read as reported, a thinking model
    billed its reasoning to nobody.
    """
    usage = Usage(
        input_tokens=um.promptTokenCount + um.toolUsePromptTokenCount,
        output_tokens=um.candidatesTokenCount + um.thoughtsTokenCount,
        cache_read_tokens=um.cachedContentTokenCount,
        # Gemini publishes no counter for what a cache write cost.
        cache_write_tokens=0,
        reasoning_tokens=um.thoughtsTokenCount,
    )
    # Only where the counts are final. Gemini attaches usageMetadata to every chunk, and a
    # mid-stream one totals parts that have not all arrived.
    if final and um.totalTokenCount is not None and um.totalTokenCount != usage.total_tokens:
        # The provider publishes the sum it expects, so this catches the day it changes the rule.
        logger.warning("usageMetadata total is %d, the parts add to %d", um.totalTokenCount, usage.total_tokens)
    return usage


# ── JSON payload builders (no SDK dependency) ───────────────────────


def _build_tools_json(tools: list[Tool[Any]]) -> list[ToolDict]:
    """Convert axio Tool list to Gemini REST API tool declarations."""
    declarations: list[FunctionDeclaration] = []
    for tool in tools:
        schema = strip_title(tool.input_schema)
        declarations.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            }
        )
    return [{"functionDeclarations": declarations}]


def _inline_data_part(block: ImageBlock | AudioBlock | VideoBlock) -> Part:
    return {
        "inlineData": {
            "mimeType": block.media_type,
            "data": base64.b64encode(block.data).decode(),
        }
    }


def _tool_result_parts(results: list[ToolResultBlock], messages: list[Message]) -> list[Part]:
    """One functionResponse per result, with any media it carried beside it."""
    parts: list[Part] = []
    for result in results:
        if isinstance(result.content, str):
            answer: dict[str, Any] = {"result": result.content}
        else:
            text = "\n".join(b.text for b in result.content if isinstance(b, TextBlock))
            answer = {"result": text}
        if result.is_error:
            answer = {"error": answer["result"]}
        parts.append(
            {
                "functionResponse": {
                    "name": _tool_name_from_id(result.tool_use_id, messages),
                    "response": answer,
                    "id": result.tool_use_id,
                }
            }
        )
        if not isinstance(result.content, str):
            # Media travels as a sibling inlineData part: functionResponse takes only JSON.
            parts.extend(
                _inline_data_part(b) for b in result.content if isinstance(b, (ImageBlock, AudioBlock, VideoBlock))
            )
    return parts


def _user_parts(msg: Message, messages: list[Message]) -> list[Part]:
    """What one user turn sends, which is either its tool results or its own content."""
    results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
    parts: list[Part] = _tool_result_parts(results, messages) if results else []
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append({"text": block.text})
        elif isinstance(block, (ImageBlock, AudioBlock, VideoBlock)):
            parts.append(_inline_data_part(block))
    return parts


def _assistant_parts(msg: Message, thought_signatures: dict[str, str] | None) -> list[Part]:
    """What one assistant turn sends back, each proof on the part Gemini issued it for."""
    parts: list[Part] = []
    # Proofs from parts that carried no text of their own. They belong to the calls that follow,
    # in arrival order.
    unplaced: deque[str] = deque()
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append(_text_part(block))
        elif isinstance(block, ReasoningBlock):
            if block.text:
                thought: Part = {"text": block.text, "thought": True}
                if block.signature:
                    thought["thoughtSignature"] = block.signature
                parts.append(thought)
            elif block.signature:
                # Gemini puts the proof on the part it signed, and a thought with no text is not it.
                unplaced.append(block.signature)
        elif isinstance(block, (ImageBlock, AudioBlock, VideoBlock)):
            parts.append(_inline_data_part(block))
        elif isinstance(block, ToolUseBlock):
            parts.append(_call_part(block, unplaced, thought_signatures))
    return parts


def _text_part(block: TextBlock) -> Part:
    """One answer-text part, carrying the proof Gemini issued for that text.

    Sent without it, a turn whose text Gemini signed fails with ``MISSING_THOUGHT_SIGNATURE``.
    """
    part: Part = {"text": block.text}
    if block.signature:
        part["thoughtSignature"] = block.signature
    return part


def _call_part(block: ToolUseBlock, unplaced: deque[str], from_transport: dict[str, str] | None) -> Part:
    """One functionCall, carrying whichever proof is most likely to be its own.

    The call's own signature comes first because it survives a restart, which the map held on the
    transport does not.
    """
    part: Part = {"functionCall": {"name": block.name, "args": block.input, "id": block.id}}
    proof = block.signature or (unplaced.popleft() if unplaced else "") or (from_transport or {}).get(block.id, "")
    if proof:
        part["thoughtSignature"] = proof
    return part


def _build_contents_json(
    messages: list[Message],
    thought_signatures: dict[str, str] | None = None,
) -> list[Content]:
    """Convert axio Message list to Gemini REST API contents array.

    thought_signatures values are base64-encoded strings ready for JSON.
    """
    contents: list[Content] = []
    for msg in messages:
        if msg.role == "user":
            if parts := _user_parts(msg, messages):
                contents.append({"role": "user", "parts": parts})
        elif msg.role == "assistant":
            if parts := _assistant_parts(msg, thought_signatures):
                contents.append({"role": "model", "parts": parts})

    # Gemini requires alternating user/model roles, so consecutive same-role turns merge. A tool
    # result followed by a "Proceed." nudge is one such pair.
    merged: list[Content] = []
    for content in contents:
        if merged and merged[-1]["role"] == content["role"]:
            merged[-1]["parts"].extend(content["parts"])
        else:
            merged.append(content)
    return merged


def _tool_name_from_id(tool_use_id: str, messages: list[Message]) -> str:
    """Find the tool name for a given tool_use_id by scanning assistant messages."""
    for msg in messages:
        if msg.role == "assistant":
            for b in msg.content:
                if isinstance(b, ToolUseBlock) and b.id == tool_use_id:
                    return b.name or "unknown"
    return "unknown"


@dataclass
class _Turn:
    """What one streaming attempt accumulates as its chunks arrive."""

    #: The part counter, which runs across the whole turn including its retries.
    at: int = -1
    #: Which stream this is, so a synthesized call id is unique for the life of the transport.
    seq: int = 0
    usage: Usage = Usage(0, 0)
    #: The counts the turn ended with. Gemini attaches usageMetadata to every chunk, and a
    #: mid-stream one totals parts that have not all arrived.
    counts: UsageMetadata | None = None
    stop_reason: StopReason = StopReason.end_turn
    finished: bool = False
    has_tool_calls: bool = False
    served_by: str | None = None
    #: The provider's own word for why it stopped, kept for the message when it means an error.
    reason: str = ""

    def restart(self) -> None:
        """Forget the attempt that failed, but not the part counter it advanced."""
        self.usage = Usage(0, 0)
        self.counts = None
        self.stop_reason = StopReason.end_turn
        self.finished = False
        self.has_tool_calls = False
        self.served_by = None


def _media_event(part: ContentPart, at: int) -> StreamEvent:
    """One inlineData part as the event its media type calls for.

    The prefix is all the wire guarantees, so each cast says the narrower type is unproven.
    """
    mime = part.inlineData.mimeType
    raw = base64.b64decode(part.inlineData.data)
    if mime.startswith("image/"):
        return ImageOutput(index=at, data=raw, media_type=cast(ImageMediaType, mime))
    if mime.startswith("audio/"):
        return AudioOutput(index=at, data=raw, media_type=cast(AudioMediaType, mime))
    if mime.startswith("video/"):
        return VideoOutput(index=at, data=raw, media_type=cast(VideoMediaType, mime))
    return ProviderEvent(provider="google", kind="inlineData", data=dict(part.raw), index=at)


# ── Transport ───────────────────────────────────────────────────────


@dataclass(slots=True)
class GoogleTransport(CompletionTransport, ImageGenTransport, VideoGenTransport):
    name: str = "Google GenAI"
    api_key: str = ""
    vertexai: bool | None = None
    project: str = ""
    location: str = ""
    model: ModelSpec = field(default_factory=lambda: GENAI_MODELS["gemini-3.1-flash-lite-preview"])
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry(GENAI_MODELS.values()))
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    max_retries: int = 5
    #: Seconds before the first retry, doubling after that. The other transports name it too.
    retry_base_delay: float = 5.0
    temperature: float | None = field(default=None, repr=False)
    top_p: float | None = field(default=None, repr=False)
    top_k: float | None = field(default=None, repr=False)
    seed: int | None = field(default=None, repr=False)
    safety_settings: list[SafetySettingDict] | None = field(default=None, repr=False)
    debug: bool = False
    nudge_on_media_tool_result: bool = True
    max_output_tokens: int | None = field(default=None, repr=False)
    thinking_budget: int | None = field(default=None, repr=False)
    thinking_level: str | None = field(default=None, repr=False)
    service_tier: str | None = field(default=None, repr=False)
    media_resolution: str | None = field(default=None, repr=False)
    # thought_signature values stored as base64 strings for direct JSON embedding
    _thought_signatures: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _streams: int = field(default=0, repr=False, compare=False)
    last_usage: Usage | None = field(default=None, repr=False, compare=False)
    # Vertex AI credentials (lazily initialised)
    _credentials: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.vertexai, str):
            self.vertexai = self.vertexai.lower() in ("true", "1")
        if self.vertexai is None:
            self.vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")
        if isinstance(self.temperature, str):
            self.temperature = float(self.temperature) if self.temperature else None
        if isinstance(self.top_p, str):
            self.top_p = float(self.top_p) if self.top_p else None
        if isinstance(self.top_k, str):
            self.top_k = float(self.top_k) if self.top_k else None
        if isinstance(self.seed, str):
            self.seed = int(self.seed) if self.seed else None
        if isinstance(self.thinking_budget, str):
            self.thinking_budget = int(self.thinking_budget) if self.thinking_budget else None
        if isinstance(self.thinking_level, str) and self.thinking_level:
            self.thinking_level = self.thinking_level.upper()
        elif not self.thinking_level:
            self.thinking_level = None

    # ── Auth & URL helpers ──

    def _get_api_key(self) -> str:
        return self.api_key or os.environ.get("GEMINI_API_KEY", "")

    async def _get_vertex_token(self) -> str:
        import google.auth
        import google.auth.transport.urllib3
        import urllib3

        if self._credentials is None:
            credentials, _ = await asyncio.to_thread(
                google.auth.default,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._credentials = credentials
        creds = cast(_RefreshableCredentials, self._credentials)
        if creds.valid and not creds.expired:
            if not creds.token:
                raise RuntimeError("Google credentials did not return an access token")
            return creds.token
        # creds.refresh() handles all credential types: user OAuth2, service
        # accounts, compute engine metadata, workload identity federation, etc.
        request_factory = cast(Any, google.auth.transport.urllib3.Request)
        await asyncio.to_thread(creds.refresh, request_factory(urllib3.PoolManager()))
        if not creds.token:
            raise RuntimeError("Google credentials did not return an access token")
        return creds.token

    def _build_url(self, path: str, qs: str = "") -> str:
        """Build a full API URL for the given path.

        For Developer API:  {base}/models/{model}:{method}?key=...&{qs}
        For Vertex AI:      {base}/projects/.../models/{model}:{method}?{qs}
        """
        if self.vertexai:
            project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            location = self.location or os.environ.get("GOOGLE_CLOUD_LOCATION", "")
            if location and location != "global":
                base = f"https://{location}-aiplatform.googleapis.com/v1beta1"
            else:
                base = "https://aiplatform.googleapis.com/v1beta1"
            url = f"{base}/projects/{project}/locations/{location}/{path}"
        else:
            api_key = self._get_api_key()
            qs = f"key={api_key}&{qs}" if qs else f"key={api_key}"
            url = f"{_DEVELOPER_API_BASE}/{path}"
        if qs:
            url = f"{url}?{qs}" if "?" not in url else f"{url}&{qs}"
        return url

    async def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.vertexai:
            token = await self._get_vertex_token()
            headers["Authorization"] = f"Bearer {token}"
            project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            if project:
                headers["x-goog-user-project"] = project
        return headers

    def get_thinking_options(self) -> tuple[str, ...] | None:
        """Valid thinkingLevel values for the current model, or None if budget-based (2.5)."""
        return valid_thinking_levels(self.model.id)

    # ── Generation config ──

    def _build_generation_config_json(self) -> GenerationConfig:
        config: GenerationConfig = {
            "maxOutputTokens": self.max_output_tokens or self.model.max_output_tokens,
            "audioTimestamp": True,
        }
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.top_p is not None:
            config["topP"] = self.top_p
        if self.top_k is not None:
            config["topK"] = self.top_k
        if self.seed is not None:
            config["seed"] = self.seed
        if self.media_resolution:
            config["mediaResolution"] = self.media_resolution.upper()  # type: ignore[typeddict-item]
        if self.thinking_level or self.thinking_budget is not None or Capability.reasoning in self.model.capabilities:
            thinking: ThinkingConfig = {"includeThoughts": True}
            levels = valid_thinking_levels(self.model.id)
            if levels is not None:
                # Gemini 3+: use thinkingLevel (thinkingBudget is not supported)
                level = (self.thinking_level or "HIGH").upper()
                if level not in levels:
                    level = levels[-1]  # fall back to highest supported
                thinking["thinkingLevel"] = level  # type: ignore[typeddict-item]
            elif self.thinking_budget is not None:
                # Gemini 2.5: use thinkingBudget (thinkingLevel is not supported)
                thinking["thinkingBudget"] = self.thinking_budget
            config["thinkingConfig"] = thinking
        if self.service_tier:
            config["serviceTier"] = self.service_tier  # type: ignore[typeddict-unknown-key]
        if Capability.image_generation in self.model.capabilities:
            config["responseModalities"] = ["TEXT", "IMAGE"]
        return config

    # ── Streaming ──

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        if self.model.id.startswith("anthropic/"):
            return self._stream_anthropic(messages, tools, system)
        return self._do_stream(messages, tools, system)

    async def _stream_anthropic(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncIterator[StreamEvent]:
        from axio_transport_anthropic import ANTHROPIC_MODELS, AnthropicTransport

        bare_id = self.model.id.removeprefix("anthropic/")
        model_spec = ANTHROPIC_MODELS.get(bare_id) or self.model
        proxy = AnthropicTransport(
            vertexai=True,
            project=self.project,
            location=self.location,
            model=model_spec,
            max_retries=self.max_retries,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=int(self.top_k) if self.top_k is not None else None,
            thinking_budget=self.thinking_budget,
            session=self.session,
        )
        async for event in proxy.stream(messages, tools, system):
            yield event

    def _build_request_body(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> GenerateContentRequest:
        """The whole streamGenerateContent request for this turn."""
        body: GenerateContentRequest = {"contents": _build_contents_json(messages, self._thought_signatures)}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools and Capability.image_generation not in self.model.capabilities:
            # An image model takes no tools, and sending them is a 400.
            body["tools"] = _build_tools_json(tools)
        body["generationConfig"] = self._build_generation_config_json()
        if self.safety_settings:
            body["safetySettings"] = self.safety_settings
        return body

    async def _do_stream(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncIterator[StreamEvent]:
        assert self.session is not None, "aiohttp session required"

        body = self._build_request_body(messages, tools, system)
        model_path = f"publishers/google/models/{self.model.id}" if self.vertexai else f"models/{self.model.id}"
        url = self._build_url(f"{model_path}:streamGenerateContent", "alt=sse")
        headers = await self._get_headers()

        logger.info("Gemini stream: model=%s, contents=%d, tools=%d", self.model.id, len(body["contents"]), len(tools))
        if self.debug:
            logger.warning("DEBUG request body:\n%s", json.dumps(_redact_body(body), indent=2, ensure_ascii=False))

        self._streams += 1
        turn = _Turn(seq=self._streams)

        sent = False
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            turn.restart()
            try:
                async with self.session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        if is_retryable(resp.status) and attempt < self.max_retries and not sent:
                            logger.warning(
                                "Gemini HTTP %d (attempt %d/%d): %.200s",
                                resp.status,
                                attempt,
                                self.max_retries,
                                error_text,
                            )
                            await asyncio.sleep(retry_delay(resp, attempt, base=self.retry_base_delay))
                            continue
                        raise StreamError(f"{resp.status} {resp.reason}: {error_text[:1000]}")

                    async for payload in payloads(resp.content.iter_any()):
                        if self.debug:
                            logger.warning(
                                "DEBUG response chunk:\n%s",
                                json.dumps(_redact_body(dict(payload)), indent=2, ensure_ascii=False),
                            )
                        for event in self._chunk_events(GenerateContentChunk.read(payload), turn):
                            sent = True
                            yield event

                usage = _usage(turn.counts, final=True) if turn.counts is not None else turn.usage
                stop_reason, candidate_reason = turn.stop_reason, turn.reason

                if not turn.finished:
                    # Every Gemini stream ends on a finishReason. Without one the connection was cut.
                    raise StreamError("Gemini stream ended without a finishReason")

                if turn.has_tool_calls and stop_reason not in _BLOCKED:
                    # Never over a blocked or failed turn.
                    stop_reason = StopReason.tool_use

                logger.info(
                    "Gemini stream complete: stop=%s, in=%d, out=%d",
                    stop_reason,
                    usage.input_tokens,
                    usage.output_tokens,
                )
                if stop_reason is StopReason.error:
                    # The caller is told only `Transport stopped with: error` if this is yielded,
                    # and MISSING_THOUGHT_SIGNATURE is a reason they can act on.
                    raise StreamError(f"Gemini stopped with {candidate_reason or 'an error'}")
                yield IterationEnd(iteration=0, stop_reason=stop_reason, usage=usage)
                return

            except StreamError:
                raise
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status", getattr(exc, "status_code", None))
                # A connection error carries no status, and every other transport retries one.
                # Tested on status alone, a disconnect before the first byte failed the turn here.
                worth_retrying = isinstance(exc, aiohttp.ClientError) or (
                    isinstance(status, int) and is_retryable(status)
                )
                if not sent and (worth_retrying or "ResourceExhausted" in str(exc)):
                    # Not once the caller has seen events: going round again re-POSTs and replays
                    # them, so a tool runs twice and its text is stored twice.
                    logger.warning("Gemini retryable error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                    if attempt < self.max_retries:
                        await asyncio.sleep(retry_delay(None, attempt, base=self.retry_base_delay))
                        continue
                logger.error("Gemini stream error: %s", exc, exc_info=True)
                raise StreamError(str(exc)) from exc

        # Chained, not flattened: the transports that kept the original exception let a caller see
        # what actually failed after the last attempt.
        raise StreamError(str(last_exc)) from last_exc

    def _chunk_events(self, chunk: GenerateContentChunk, turn: _Turn) -> Iterator[StreamEvent]:
        """Every event one streamGenerateContent chunk produces, advancing the turn's state."""
        if chunk.usageMetadata.promptTokenCount:
            turn.counts = chunk.usageMetadata
            turn.usage = _usage(chunk.usageMetadata)
            self.last_usage = turn.usage

        if turn.served_by is None and chunk.modelVersion:
            # The model that answered, which need not be the one asked for.
            turn.served_by = chunk.modelVersion
            yield IterationStart(iteration=0, id=chunk.responseId or None, model=turn.served_by)

        if block_reason := chunk.promptFeedback.string("blockReason"):
            # blockReason, not presence: promptFeedback rides along with healthy answers too.
            # A blocked prompt is a finished turn, so no candidate and no finishReason follow.
            turn.finished = True
            turn.stop_reason = StopReason.refusal
            yield Refusal(index=0, category=block_reason, blocked_input=True, raw=dict(chunk.promptFeedback))

        if not chunk.candidates:
            return
        candidate = chunk.candidates[0]

        if candidate.finishReason:
            turn.finished = True
            turn.reason = candidate.finishReason
            turn.stop_reason = stop_reason_from(candidate.finishReason, _FINISH_REASON_MAP, provider="Gemini")

        # Grounding travels whole. Four providers shape it four incompatible ways.
        for kind, metadata in (
            ("citationMetadata", candidate.citationMetadata),
            ("groundingMetadata", candidate.groundingMetadata),
        ):
            if metadata:
                yield ProviderEvent(provider="google", kind=kind, data=dict(metadata), index=0)

        for part in candidate.content.parts:
            turn.at += 1
            if part.functionCall.name:
                turn.has_tool_calls = True
            yield from self._part_events(part, turn)

    def _part_events(self, part: ContentPart, turn: _Turn) -> Iterator[StreamEvent]:
        """Every event one part of a candidate produces."""
        at = turn.at
        if part.functionCall.name:
            yield from self._call_events(part, turn)
            return

        carried = True
        if part.text and part.thought:
            yield ReasoningDelta(index=at, delta=part.text)
        elif part.text:
            yield TextDelta(index=at, delta=part.text)
        elif part.inlineData.data:
            yield _media_event(part, at)
        elif set(part.raw) - {"thought", "thoughtSignature"}:
            # executableCode, codeExecutionResult, fileData and whatever the API adds next: content
            # of the turn axio has no type for. Emitted even when signed, because a proof is not a
            # reason to drop what it proves.
            yield ProviderEvent(provider="google", kind="part", data=dict(part.raw), index=at)
        else:
            carried = False

        if not part.thoughtSignature:
            return
        if part.thought or not carried:
            # It signs reasoning, or it is the bare proof of a call that follows. Emitted after
            # the reasoning, never before: the agent signs the block it has just built.
            yield ReasoningSignature(index=at, signature=part.thoughtSignature)
        elif part.text:
            # The proof signs answer text, so it rides on that text block. Emitted after the text,
            # never before, for the same reason as reasoning.
            yield TextSignature(index=at, signature=part.thoughtSignature)
        else:
            # Media, executableCode and the rest: axio has no block that can hold this proof, so it
            # travels raw rather than attaching to a block the provider did not sign.
            yield ProviderEvent(provider="google", kind="thoughtSignature", data=dict(part.raw), index=at)

    def _call_events(self, part: ContentPart, turn: _Turn) -> Iterator[StreamEvent]:
        """The start and the arguments of one function call."""
        at = turn.at
        call = part.functionCall
        # By position and by stream, never by id(): the part is a temporary whose address CPython
        # reuses, and a position alone repeats every turn while _thought_signatures does not.
        call_id = call.id or f"genai_{call.name}_{turn.seq}_{at}"
        if part.thoughtSignature:
            self._thought_signatures[call_id] = part.thoughtSignature
        # The signature goes on the call, not beside it. Sent bare it attaches to whatever block
        # is still unsigned.
        yield ToolUseStart(index=at, tool_use_id=call_id, name=call.name, signature=part.thoughtSignature)
        yield ToolInputDelta(
            index=at, tool_use_id=call_id, partial_json=json.dumps(dict(call.args)) if call.args else "{}"
        )

    # ── Image / Veo generation ──

    async def generate_images(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        """Generate images via Gemini Nano Banana (generateContent with IMAGE response modality)."""
        assert self.session is not None, "aiohttp session required"
        model_id = model or "gemini-3-pro-image-preview"
        return await self._generate_images_gemini(prompt, model_id=model_id, n=n)

    async def _generate_images_gemini(self, prompt: str, *, model_id: str, n: int) -> list[bytes]:
        assert self.session is not None
        model_path = f"publishers/google/models/{model_id}" if self.vertexai else f"models/{model_id}"
        url = self._build_url(f"{model_path}:generateContent")
        headers = await self._get_headers()
        results: list[bytes] = []
        for _ in range(n):
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"]},
            }
            async with self.session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise StreamError(f"Gemini image {resp.status}: {error_text[:1000]}")
                data = await resp.json()
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    idata = part.get("inlineData")
                    if idata and idata.get("mimeType", "").startswith("image/"):
                        results.append(base64.b64decode(idata["data"]))
        return results

    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        image: bytes | None = None,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]:
        """Generate videos using Veo models. Polls until the operation completes."""
        assert self.session is not None, "aiohttp session required"
        model_id = model or "veo-3.1-fast-generate-001"
        model_path = f"publishers/google/models/{model_id}" if self.vertexai else f"models/{model_id}"
        url = self._build_url(f"{model_path}:predictLongRunning")
        headers = await self._get_headers()

        instance: dict[str, Any] = {"prompt": prompt}
        if image:
            instance["image"] = {
                "bytesBase64Encoded": base64.b64encode(image).decode(),
                "mimeType": "image/jpeg",
            }
        params: dict[str, Any] = {"sampleCount": n}
        if duration_seconds:
            params["durationSeconds"] = duration_seconds
        if aspect_ratio:
            params["aspectRatio"] = aspect_ratio
        body = {"instances": [instance], "parameters": params}

        async with self.session.post(url, json=body, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise StreamError(f"Veo {resp.status}: {error_text[:1000]}")
            operation = await resp.json()

        # Poll until done
        op_name = operation.get("name", "")
        while not operation.get("done"):
            await asyncio.sleep(5)
            headers = await self._get_headers()
            if self.vertexai:
                poll_url = self._build_url(f"{model_path}:fetchPredictOperation")
                async with self.session.post(
                    poll_url,
                    json={"operationName": op_name},
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise StreamError(f"Veo poll {resp.status}: {error_text[:1000]}")
                    operation = await resp.json()
            else:
                op_id = op_name.rsplit("/", 1)[-1]
                poll_url = self._build_url(f"models/{model_id}/operations/{op_id}")
                async with self.session.get(poll_url, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise StreamError(f"Veo poll {resp.status}: {error_text[:1000]}")
                    operation = await resp.json()

        response = operation.get("response", {})
        results: list[bytes] = []
        # Vertex AI: response.videos[].bytesBase64Encoded (inline) or .gcsUri
        for vid in response.get("videos", []):
            b64 = vid.get("bytesBase64Encoded")
            if b64:
                results.append(base64.b64decode(b64))
        # Vertex AI fallback / Developer API nested structure
        generated = response.get("generatedSamples") or response.get("generateVideoResponse", {}).get(
            "generatedSamples", []
        )
        for sample in generated:
            video = sample.get("video", {})
            b64 = video.get("encodedVideo") or video.get("bytesBase64Encoded")
            if b64:
                results.append(base64.b64decode(b64))
            elif not results and video.get("uri"):
                # Developer API returns a temporary download URL
                headers = await self._get_headers()
                async with self.session.get(video["uri"], headers=headers) as resp:
                    if resp.status == 200:
                        results.append(await resp.read())
                    else:
                        logger.warning("Veo video download failed: %d", resp.status)
        return results

    # ── Model listing ──

    async def fetch_models(self) -> None:
        """Fetch available Gemini models.

        Developer API: GET /v1beta/models?key=...
        Vertex AI:     GET /v1beta1/publishers/google/models (no project prefix)
        """
        assert self.session is not None, "aiohttp session required"
        try:
            headers = await self._get_headers()
            if self.vertexai:
                # Vertex AI model catalog — no project/location prefix
                base_url = "https://aiplatform.googleapis.com/v1beta1/publishers/google/models"
            else:
                api_key = self._get_api_key()
                base_url = f"{_DEVELOPER_API_BASE}/models?key={api_key}"

            fetched: list[ModelSpec] = []
            page_token: str | None = None
            while True:
                sep = "&" if "?" in base_url else "?"
                url = f"{base_url}{sep}pageToken={page_token}" if page_token else base_url
                async with self.session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning("fetch_models HTTP %d", resp.status)
                        break
                    data = await resp.json()
                # Developer API: {"models": [...]}, Vertex AI: {"publisherModels": [...]}
                raw_models = data.get("models") or data.get("publisherModels") or []
                for model in raw_models:
                    name: str = model.get("name", "")
                    if "models/" in name:
                        model_id = name.split("models/", 1)[1]
                    else:
                        model_id = name
                    if not model_id:
                        continue

                    # Developer API populates supportedGenerationMethods;
                    # Vertex AI does not — filter by name instead.
                    gen_methods: list[str] = model.get("supportedGenerationMethods", [])
                    if gen_methods and "generateContent" not in gen_methods:
                        continue
                    if any(s in model_id for s in ("-tts", "native-audio", "gemini-live-")):
                        continue

                    if model_id in GENAI_MODELS:
                        fetched.append(GENAI_MODELS[model_id])
                    else:
                        caps = _RT if model.get("thinking") else _VT
                        fetched.append(
                            ModelSpec(
                                id=model_id,
                                context_window=model.get("inputTokenLimit", 1_048_576),
                                max_output_tokens=model.get("outputTokenLimit", 8_192),
                                capabilities=caps,
                            )
                        )
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
            if fetched:
                self.models = ModelRegistry(fetched)
            else:
                self.models = GENAI_MODELS
        except Exception:
            logger.warning("fetch_models failed, using defaults", exc_info=True)
            self.models = GENAI_MODELS

        if self.vertexai:
            for spec in _get_anthropic_models().values():
                self.models[spec.id] = spec

    # ── Serialization ──

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
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
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "max_output_tokens",
            "thinking_budget",
            "thinking_level",
            "service_tier",
            "media_resolution",
        ):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.safety_settings:
            d["safety_settings"] = self.safety_settings
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoogleTransport:
        models = ModelRegistry(
            [
                ModelSpec(
                    id=str(m["id"]),
                    context_window=int(m.get("context_window", 1_048_576)),
                    max_output_tokens=int(m.get("max_output_tokens", 8_192)),
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
            api_key=str(data.get("api_key", "")),
            vertexai=bool(data.get("vertexai", False)),
            project=str(data.get("project", "")),
            location=str(data.get("location", "")),
            models=models,
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            seed=data.get("seed"),
            safety_settings=data.get("safety_settings"),
            max_output_tokens=data.get("max_output_tokens"),
            thinking_budget=data.get("thinking_budget"),
            thinking_level=data.get("thinking_level"),
            service_tier=data.get("service_tier"),
            media_resolution=data.get("media_resolution"),
        )


@dataclass(slots=True)
class VertexAITransport(GoogleTransport):
    """GoogleTransport pre-configured for Vertex AI (includes Anthropic models)."""

    name: str = "Google Vertex AI"
    vertexai: bool | None = True
