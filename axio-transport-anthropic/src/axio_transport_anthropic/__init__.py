"""Anthropic Claude CompletionTransport via aiohttp (direct API and Vertex AI)."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, cast

import aiohttp
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock, VideoBlock
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import CompletionTransport
from axio.types import StopReason, Usage

logger = logging.getLogger(__name__)

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

_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.end_turn,
    "stop_sequence": StopReason.end_turn,
    "tool_use": StopReason.tool_use,
    "max_tokens": StopReason.max_tokens,
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
                elif isinstance(b, ToolUseBlock):
                    content_parts.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})

        if content_parts:
            result.append({"role": msg.role, "content": content_parts})

    return result


def _convert_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio Tool list to Anthropic tool dicts."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": _strip_title(tool.input_schema),
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
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}

        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None
        index_to_tool_use_id: dict[int, str] = {}

        buffer = b""
        event_type: str = ""

        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    event_type = line[7:]
                    continue
                if not line.startswith("data: "):
                    continue

                data: dict[str, Any] = json.loads(line[6:])

                if event_type == "message_start":
                    usage = data.get("message", {}).get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)

                elif event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        idx: int = data.get("index", 0)
                        tool_use_id: str = block.get("id", "")
                        tool_name: str = block.get("name", "")
                        index_to_tool_use_id[idx] = tool_use_id
                        yield ToolUseStart(index=idx, tool_use_id=tool_use_id, name=tool_name)

                elif event_type == "content_block_delta":
                    idx = data.get("index", 0)
                    delta: dict[str, Any] = data.get("delta", {})
                    delta_type: str = delta.get("type", "")

                    if delta_type == "text_delta":
                        yield TextDelta(index=idx, delta=delta.get("text", ""))
                    elif delta_type == "thinking_delta":
                        yield ReasoningDelta(index=idx, delta=delta.get("thinking", ""))
                    elif delta_type == "input_json_delta":
                        tid = index_to_tool_use_id.get(idx, "")
                        yield ToolInputDelta(index=idx, tool_use_id=tid, partial_json=delta.get("partial_json", ""))

                elif event_type == "message_delta":
                    delta = data.get("delta", {})
                    if "stop_reason" in delta and delta["stop_reason"] is not None:
                        stop_reason = delta["stop_reason"]
                    usage = data.get("usage", {})
                    output_tokens = usage.get("output_tokens", output_tokens)

                elif event_type == "error":
                    err = data.get("error", {})
                    raise StreamError(f"Anthropic error: {err.get('type', 'unknown')}: {err.get('message', '')}")

        usage_obj = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        stop = _STOP_REASON_MAP.get(stop_reason or "", StopReason.error)
        if stop_reason and stop_reason not in _STOP_REASON_MAP:
            logger.warning("Unknown stop_reason %r, mapped to %s", stop_reason, stop)
        logger.info(
            "Stream complete: stop_reason=%s, input_tokens=%d, output_tokens=%d",
            stop,
            input_tokens,
            output_tokens,
        )
        yield IterationEnd(iteration=0, stop_reason=stop, usage=usage_obj)

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

            if attempt < self.max_retries:
                delay = self._get_retry_delay(retry_resp, attempt)
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
