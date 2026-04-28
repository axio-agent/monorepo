"""Anthropic Claude CompletionTransport via aiohttp."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Self

import aiohttp
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import CompletionTransport
from axio.types import StopReason, Usage

logger = logging.getLogger(__name__)


_VT = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_RT = frozenset({Capability.text, Capability.vision, Capability.reasoning, Capability.tool_use})

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
    "tool_use": StopReason.tool_use,
    "max_tokens": StopReason.max_tokens,
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
                elif isinstance(b, ImageBlock):
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
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


@dataclass(slots=True)
class AnthropicTransport(CompletionTransport):
    name: str = "Anthropic"
    base_url: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    model: ModelSpec = field(default_factory=lambda: ANTHROPIC_MODELS["claude-sonnet-4-6"])
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry(ANTHROPIC_MODELS.values()))
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    max_retries: int = 10
    retry_base_delay: float = 5.0

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

        system_blocks: list[dict[str, Any]] = []
        if system:
            system_blocks.append({"type": "text", "text": system, "cache_control": {"type": "ephemeral"}})
        for msg in messages:
            if msg.role == "system":
                text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
                if text:
                    system_blocks.append({"type": "text", "text": text})

        payload: dict[str, Any] = {
            "model": self.model.id,
            "messages": converted_messages,
            "stream": True,
            "max_tokens": self.model.max_output_tokens,
        }

        if system_blocks:
            payload["system"] = system_blocks

        if tools:
            converted = _convert_tools(tools)
            converted[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = converted

        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None
        # Track tool_use_id per content block index for ToolInputDelta
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
        url = f"{self.base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
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
        return {
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
            models=models,
            session=session,
        )
