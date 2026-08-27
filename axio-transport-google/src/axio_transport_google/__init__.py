"""Google GenAI (Gemini) transport — aiohttp streaming, SDK for media generation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections import deque
from collections.abc import AsyncIterator
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
    ToolInputDelta,
    ToolUseStart,
    VideoOutput,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import CompletionTransport, ImageGenTransport, VideoGenTransport
from axio.types import StopReason, Usage
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


#: The union of every ``finishReason`` either surface publishes. The developer API publishes 21 and Vertex
#: publishes 17. One left out is read as an error. That tells the caller the transport broke when the model
#: was in fact blocked, or asked to be prompted again.
_FINISH_REASON_MAP: dict[str, StopReason] = {
    "STOP": StopReason.end_turn,
    "MAX_TOKENS": StopReason.max_tokens,
    # The turn was blocked. This is not the transport failing. Sending the same prompt cannot
    # succeed.
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
    # The model produced a call the API could not use. Read as tool_use, the agent prompts again.
    # That is the recovery, because the next answer may parse.
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
    #: Opaque proof that the reasoning is the model's own. Returned altered or missing, the next
    #: request comes back with ``MISSING_THOUGHT_SIGNATURE``.
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


def _usage(um: UsageMetadata) -> Usage:
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
    if um.totalTokenCount is not None and um.totalTokenCount != usage.total_tokens:
        # The provider publishes the sum it expects, so this catches the day it changes the rule.
        logger.warning("usageMetadata total is %d, the parts add to %d", um.totalTokenCount, usage.total_tokens)
    return usage


# ── JSON payload builders (no SDK dependency) ───────────────────────


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


def _build_tools_json(tools: list[Tool[Any]]) -> list[ToolDict]:
    """Convert axio Tool list to Gemini REST API tool declarations."""
    declarations: list[FunctionDeclaration] = []
    for tool in tools:
        schema = _strip_title(tool.input_schema)
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
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results and len(tool_results) == len(msg.content):
                tool_result_parts: list[Part] = []
                for tr in tool_results:
                    if isinstance(tr.content, str):
                        response_dict: dict[str, Any] = {"result": tr.content}
                    else:
                        text_parts = [b.text for b in tr.content if isinstance(b, TextBlock)]
                        response_dict = {"result": "\n".join(text_parts)} if text_parts else {"result": ""}
                    if tr.is_error:
                        response_dict = {"error": response_dict.get("result", "")}
                    tool_result_parts.append(
                        {
                            "functionResponse": {
                                "name": _tool_name_from_id(tr.tool_use_id, messages),
                                "response": response_dict,
                                "id": tr.tool_use_id,
                            }
                        }
                    )
                    # Media from tool results as sibling inlineData parts
                    if not isinstance(tr.content, str):
                        for content_block in tr.content:
                            if isinstance(content_block, (ImageBlock, AudioBlock, VideoBlock)):
                                tool_result_parts.append(_inline_data_part(content_block))
                contents.append({"role": "user", "parts": tool_result_parts})
            else:
                user_parts: list[Part] = []
                for message_block in msg.content:
                    if isinstance(message_block, TextBlock):
                        user_parts.append({"text": message_block.text})
                    elif isinstance(message_block, (ImageBlock, AudioBlock, VideoBlock)):
                        user_parts.append(_inline_data_part(message_block))
                if user_parts:
                    contents.append({"role": "user", "parts": user_parts})

        elif msg.role == "assistant":
            assistant_parts: list[Part] = []
            # Signatures the model issued for parts that carried no text of their own. They
            # belong to the calls that follow, in the order they arrived. A turn may hold several
            # parallel calls. The agent appends every signature before any of the calls. A single
            # slot therefore gave the first call the last signature and left the rest unsigned.
            unplaced_signatures: deque[str] = deque()
            for assistant_block in msg.content:
                if isinstance(assistant_block, TextBlock):
                    assistant_parts.append({"text": assistant_block.text})
                elif isinstance(assistant_block, ReasoningBlock):
                    # The signature is documented as "an opaque signature for the thought so it can
                    # be reused in subsequent requests". Read from the stored block rather than from
                    # the map on this instance. That map is per-transport and empty after a restart,
                    # so a resumed session used to come back MISSING_THOUGHT_SIGNATURE.
                    if assistant_block.text:
                        thought: Part = {"text": assistant_block.text, "thought": True}
                        if assistant_block.signature:
                            thought["thoughtSignature"] = assistant_block.signature
                        assistant_parts.append(thought)
                    elif assistant_block.signature:
                        # Gemini puts the proof on the part it signed. A thought part with no text
                        # is not that part. Sent as one, the call it belongs to comes back
                        # MISSING_THOUGHT_SIGNATURE.
                        unplaced_signatures.append(assistant_block.signature)
                elif isinstance(assistant_block, (ImageBlock, AudioBlock, VideoBlock)):
                    assistant_parts.append(_inline_data_part(assistant_block))
                elif isinstance(assistant_block, ToolUseBlock):
                    part: Part = {
                        "functionCall": {
                            "name": assistant_block.name,
                            "args": assistant_block.input,
                            "id": assistant_block.id,
                        }
                    }
                    # The stored signature comes first, taken in arrival order. The map is this
                    # instance's, so it is empty in a session restored into a new transport, which
                    # is the case that fails.
                    stored_signature = unplaced_signatures.popleft() if unplaced_signatures else ""
                    signature = stored_signature or (thought_signatures or {}).get(assistant_block.id, "")
                    if signature:
                        part["thoughtSignature"] = signature
                    assistant_parts.append(part)
            if assistant_parts:
                contents.append({"role": "model", "parts": assistant_parts})

    # Gemini requires alternating user/model roles.  Merge consecutive
    # same-role contents (e.g. tool-result message + "Proceed." nudge).
    merged: list[Content] = []
    for c in contents:
        if merged and merged[-1]["role"] == c["role"]:
            merged[-1]["parts"].extend(c["parts"])
        else:
            merged.append(c)
    return merged


def _tool_name_from_id(tool_use_id: str, messages: list[Message]) -> str:
    """Find the tool name for a given tool_use_id by scanning assistant messages."""
    for msg in messages:
        if msg.role == "assistant":
            for b in msg.content:
                if isinstance(b, ToolUseBlock) and b.id == tool_use_id:
                    return b.name or "unknown"
    return "unknown"


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

    async def _do_stream(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncIterator[StreamEvent]:
        assert self.session is not None, "aiohttp session required"

        contents = _build_contents_json(messages, self._thought_signatures)

        body: GenerateContentRequest = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        is_image_model = Capability.image_generation in self.model.capabilities
        if tools and not is_image_model:
            body["tools"] = _build_tools_json(tools)

        body["generationConfig"] = self._build_generation_config_json()

        if self.safety_settings:
            body["safetySettings"] = self.safety_settings

        model_path = f"publishers/google/models/{self.model.id}" if self.vertexai else f"models/{self.model.id}"
        url = self._build_url(f"{model_path}:streamGenerateContent", "alt=sse")
        headers = await self._get_headers()

        logger.info(
            "Gemini stream: model=%s, contents=%d, tools=%d",
            self.model.id,
            len(contents),
            len(tools),
        )
        if self.debug:
            logger.warning("DEBUG request body:\n%s", json.dumps(_redact_body(body), indent=2, ensure_ascii=False))
        elif logger.getEffectiveLevel() <= logging.DEBUG:
            for i, c in enumerate(contents):
                logger.debug("  content[%d] role=%s parts=%d", i, c.get("role"), len(c.get("parts", [])))

        usage = Usage(0, 0)
        stop_reason = StopReason.end_turn
        has_tool_calls = False
        served_by: str | None = None

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        if resp.status in (429, 500, 503) and attempt < self.max_retries:
                            logger.warning(
                                "Gemini HTTP %d (attempt %d/%d): %.200s",
                                resp.status,
                                attempt,
                                self.max_retries,
                                error_text,
                            )
                            await asyncio.sleep(2.0 * (2 ** (attempt - 1)))
                            continue
                        raise StreamError(f"{resp.status} {resp.reason}: {error_text[:1000]}")

                    async for payload in payloads(resp.content.iter_any()):
                        if self.debug:
                            logger.warning(
                                "DEBUG response chunk:\n%s",
                                json.dumps(_redact_body(dict(payload)), indent=2, ensure_ascii=False),
                            )
                        chunk = GenerateContentChunk.read(payload)

                        if chunk.usageMetadata.promptTokenCount:
                            usage = _usage(chunk.usageMetadata)
                            self.last_usage = usage

                        if served_by is None and chunk.modelVersion:
                            # The model that answered, which need not be the one asked for.
                            served_by = chunk.modelVersion
                            yield IterationStart(iteration=0, id=chunk.responseId or None, model=served_by)

                        if block_reason := chunk.promptFeedback.string("blockReason"):
                            # Test blockReason and not presence. The developer API attaches
                            # promptFeedback with safetyRatings to healthy answers too, and a
                            # Payload is a dict. Gating on truthiness therefore called every one of
                            # those healthy answers a blocked prompt. Only blockReason means
                            # nothing was generated.
                            stop_reason = StopReason.refusal
                            yield Refusal(
                                index=0,
                                category=block_reason,
                                blocked_input=True,
                                raw=dict(chunk.promptFeedback),
                            )

                        if not chunk.candidates:
                            continue
                        candidate = chunk.candidates[0]

                        if candidate.finishReason:
                            stop_reason = _FINISH_REASON_MAP.get(candidate.finishReason, StopReason.error)
                            if candidate.finishReason not in _FINISH_REASON_MAP:
                                logger.warning("Unknown finishReason %r", candidate.finishReason)

                        # What the answer was attributed to. Four providers shape this four
                        # incompatible ways, so it travels whole rather than as an invented common
                        # form.
                        if candidate.citationMetadata:
                            yield ProviderEvent(
                                provider="google",
                                kind="citationMetadata",
                                data=dict(candidate.citationMetadata),
                                index=0,
                            )
                        if candidate.groundingMetadata:
                            yield ProviderEvent(
                                provider="google",
                                kind="groundingMetadata",
                                data=dict(candidate.groundingMetadata),
                                index=0,
                            )

                        for at, part in enumerate(candidate.content.parts):
                            if part.text and part.thought:
                                yield ReasoningDelta(index=0, delta=part.text)
                            elif part.text:
                                yield TextDelta(index=0, delta=part.text)
                            elif part.inlineData.data:
                                mt = part.inlineData.mimeType
                                raw = base64.b64decode(part.inlineData.data)
                                # The prefix is all the wire guarantees. The event types name the
                                # exact media types they accept. The provider is free to send one
                                # outside that list. So the check tests the prefix. The cast says
                                # so rather than pretending the narrower type was proven.
                                if mt.startswith("image/"):
                                    yield ImageOutput(index=0, data=raw, media_type=cast(ImageMediaType, mt))
                                elif mt.startswith("audio/"):
                                    yield AudioOutput(index=0, data=raw, media_type=cast(AudioMediaType, mt))
                                elif mt.startswith("video/"):
                                    yield VideoOutput(index=0, data=raw, media_type=cast(VideoMediaType, mt))
                                else:
                                    yield ProviderEvent(
                                        provider="google", kind="inlineData", data=dict(part.raw), index=0
                                    )
                            elif part.functionCall.name:
                                call = part.functionCall
                                call_id = call.id or f"genai_{call.name}_{id(part)}"
                                if part.thoughtSignature:
                                    self._thought_signatures[call_id] = part.thoughtSignature
                                yield ToolUseStart(index=0, tool_use_id=call_id, name=call.name)
                                args_json = json.dumps(dict(call.args)) if call.args else "{}"
                                yield ToolInputDelta(index=0, tool_use_id=call_id, partial_json=args_json)
                                has_tool_calls = True
                            elif not part.thoughtSignature:
                                # executableCode, codeExecutionResult, fileData and whatever the API
                                # adds next: content of the turn that axio has no type for.
                                yield ProviderEvent(provider="google", kind="part", data=dict(part.raw), index=0)

                            if part.thoughtSignature:
                                # Emit after the reasoning it signs, never before. The agent
                                # attaches a signature to the block it just built, and refuses to
                                # extend a block already signed. Emitted first, one part became two
                                # blocks — a signature on nothing, and reasoning left unsigned.
                                #
                                # The index is the part's own position, because the index on a
                                # signature means which block it proves. Fixed at zero, two
                                # parallel signed calls looked like two halves of one proof.
                                yield ReasoningSignature(index=at, data=part.thoughtSignature)

                if has_tool_calls:
                    stop_reason = StopReason.tool_use

                logger.info(
                    "Gemini stream complete: stop=%s, in=%d, out=%d",
                    stop_reason,
                    usage.input_tokens,
                    usage.output_tokens,
                )
                yield IterationEnd(iteration=0, stop_reason=stop_reason, usage=usage)
                return

            except StreamError:
                raise
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status", getattr(exc, "status_code", None))
                if status in (429, 500, 503) or "ResourceExhausted" in str(exc):
                    logger.warning("Gemini retryable error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                    if attempt < self.max_retries:
                        await asyncio.sleep(2.0 * (2 ** (attempt - 1)))
                        continue
                logger.error("Gemini stream error: %s", exc, exc_info=True)
                raise StreamError(str(exc)) from exc

        raise StreamError(str(last_exc)) from last_exc

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
