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
from typing import Any, Self

import aiohttp
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import CompletionTransport, EmbeddingTransport
from axio.types import StopReason, Usage

from .realtime import OpenAIRealtimeSession, OpenAIRealtimeTransport  # noqa: F401

logger = logging.getLogger(__name__)


_VT = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_RT = frozenset({Capability.text, Capability.reasoning, Capability.tool_use})
_TT = frozenset({Capability.text, Capability.tool_use})
_VRT = frozenset({Capability.text, Capability.vision, Capability.reasoning, Capability.tool_use})

OPENAI_MODELS: ModelRegistry = ModelRegistry(
    {
        # GPT-5.6 family (latest)
        ModelSpec(
            id="gpt-5.6",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=5,
            output_cost=30,
        ),
        ModelSpec(
            id="gpt-5.6-sol",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=5,
            output_cost=30,
        ),
        ModelSpec(
            id="gpt-5.6-terra",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=2,
            output_cost=12,
        ),
        ModelSpec(
            id="gpt-5.6-luna",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=0.2,
            output_cost=1.2,
        ),
        # GPT-5.5 family
        ModelSpec(
            id="gpt-5.5",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=5,
            output_cost=30,
        ),
        ModelSpec(
            id="gpt-5.5-pro",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=30,
            output_cost=180,
        ),
        # GPT-5.4 family
        ModelSpec(
            id="gpt-5.4",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=2.5,
            output_cost=15,
        ),
        ModelSpec(
            id="gpt-5.4-mini",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=0.75,
            output_cost=4.5,
        ),
        ModelSpec(
            id="gpt-5.4-nano",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=0.2,
            output_cost=1.25,
        ),
        ModelSpec(
            id="gpt-5.4-pro",
            context_window=1_050_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=30,
            output_cost=180,
        ),
        # GPT-5.x family
        ModelSpec(
            id="gpt-5.2",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=1.75,
            output_cost=14,
        ),
        ModelSpec(
            id="gpt-5.2-pro",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=21,
            output_cost=168,
        ),
        ModelSpec(
            id="gpt-5.1",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=1.25,
            output_cost=10,
        ),
        ModelSpec(
            id="gpt-5",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=1.25,
            output_cost=10,
        ),
        ModelSpec(
            id="gpt-5-mini",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=0.25,
            output_cost=2,
        ),
        ModelSpec(
            id="gpt-5-nano",
            context_window=272_000,
            max_output_tokens=128_000,
            capabilities=_VRT,
            input_cost=0.05,
            output_cost=0.4,
        ),
        # o-series reasoning models
        ModelSpec(
            id="o3",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=_VRT,
            input_cost=2,
            output_cost=8,
        ),
        ModelSpec(
            id="o3-mini",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=_RT,
            input_cost=1.1,
            output_cost=4.4,
        ),
        ModelSpec(
            id="o4-mini",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=_VRT,
            input_cost=1.1,
            output_cost=4.4,
        ),
        # GPT-4.1 family
        ModelSpec(
            id="gpt-4.1",
            context_window=1_047_576,
            max_output_tokens=32_768,
            capabilities=_VT,
            input_cost=2,
            output_cost=8,
        ),
        ModelSpec(
            id="gpt-4.1-mini",
            context_window=1_047_576,
            max_output_tokens=32_768,
            capabilities=_VT,
            input_cost=0.4,
            output_cost=1.6,
        ),
        ModelSpec(
            id="gpt-4.1-nano",
            context_window=1_047_576,
            max_output_tokens=32_768,
            capabilities=_VT,
            input_cost=0.1,
            output_cost=0.4,
        ),
        # GPT-4o family
        ModelSpec(
            id="gpt-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            capabilities=_VT,
            input_cost=2.5,
            output_cost=10,
        ),
        ModelSpec(
            id="gpt-4o-mini",
            context_window=128_000,
            max_output_tokens=16_384,
            capabilities=_VT,
            input_cost=0.15,
            output_cost=0.6,
        ),
    }
)

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.end_turn,
    "tool_calls": StopReason.tool_use,
    "length": StopReason.max_tokens,
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

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.id,
            "messages": _convert_messages(messages, system),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self.model.max_output_tokens,
        }

        if tools:
            payload["tools"] = _convert_tools(tools)

        if self.extra_params:
            payload.update(self.extra_params)

        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        tool_index_to_id: dict[int, str] = {}
        usage = Usage(0, 0)
        finish_reason: str | None = None
        error_message: str | None = None
        think_parser = ThinkTagParser()

        buffer = b""
        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8").strip()
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue

                data: dict[str, Any] = json.loads(line[6:])

                if "error" in data and data["error"] is not None:
                    err = data["error"]
                    error_message = err.get("message") or str(err) if isinstance(err, dict) else str(err)

                if "usage" in data and data["usage"] is not None:
                    u: dict[str, int] = data["usage"]
                    usage = Usage(
                        input_tokens=u.get("prompt_tokens", 0),
                        output_tokens=u.get("completion_tokens", 0),
                    )

                choices: list[dict[str, Any]] = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta: dict[str, Any] = choice.get("delta") or {}

                # llama.cpp and DeepSeek-style servers hand thoughts over in
                # their own field instead of wrapping them in <think> tags, so
                # ThinkTagParser never sees them.
                if delta.get("reasoning_content"):
                    yield ReasoningDelta(index=0, delta=delta["reasoning_content"])

                if "content" in delta and delta["content"] is not None:
                    for kind, text in think_parser.feed(delta["content"]):
                        if kind == "reasoning":
                            yield ReasoningDelta(index=0, delta=text)
                        else:
                            yield TextDelta(index=0, delta=text)

                if "tool_calls" in delta and delta["tool_calls"] is not None:
                    for tc in delta["tool_calls"]:
                        idx: int = tc["index"]
                        if "id" in tc and tc["id"]:
                            tool_id: str = tc["id"]
                            fn = tc.get("function") or {}
                            tool_name: str = fn.get("name", "")
                            tool_index_to_id[idx] = tool_id
                            yield ToolUseStart(index=idx, tool_use_id=tool_id, name=tool_name)
                        fn = tc.get("function") or {}
                        if "arguments" in fn:
                            tid = tool_index_to_id.get(idx, "")
                            yield ToolInputDelta(index=idx, tool_use_id=tid, partial_json=fn["arguments"])

                if "finish_reason" in choice and choice["finish_reason"] is not None:
                    finish_reason = choice["finish_reason"]

        # Flush remaining SSE buffer (streams that don't end with \n)
        if buffer:
            line = buffer.decode("utf-8").strip()
            if line and line != "data: [DONE]" and line.startswith("data: "):
                data = json.loads(line[6:])

                if "usage" in data and data["usage"] is not None:
                    u = data["usage"]
                    usage = Usage(
                        input_tokens=u.get("prompt_tokens", 0),
                        output_tokens=u.get("completion_tokens", 0),
                    )

                choices = data.get("choices", [])
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    if delta.get("reasoning_content"):
                        yield ReasoningDelta(index=0, delta=delta["reasoning_content"])

                    if "content" in delta and delta["content"] is not None:
                        for kind, text in think_parser.feed(delta["content"]):
                            if kind == "reasoning":
                                yield ReasoningDelta(index=0, delta=text)
                            else:
                                yield TextDelta(index=0, delta=text)

                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc["index"]
                            if "id" in tc and tc["id"]:
                                tool_id = tc["id"]
                                tool_name = tc["function"]["name"]
                                tool_index_to_id[idx] = tool_id
                                yield ToolUseStart(index=idx, tool_use_id=tool_id, name=tool_name)
                            if "function" in tc and "arguments" in tc["function"]:
                                tid = tool_index_to_id.get(idx, "")
                                yield ToolInputDelta(
                                    index=idx,
                                    tool_use_id=tid,
                                    partial_json=tc["function"]["arguments"],
                                )

                    if "finish_reason" in choice and choice["finish_reason"] is not None:
                        finish_reason = choice["finish_reason"]

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
        url = f"{self.base_url.rstrip('/')}/chat/completions"
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
    reasoning.  Declare ``thinking: bool = False`` in the concrete dataclass
    and mix this in to get automatic payload injection and to_dict/from_dict
    round-trip support.
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
