"""OpenAI transport speaking /v1/responses instead of /v1/chat/completions.

Not a preference: on chat/completions the newer models refuse to combine
function tools with a reasoning effort — "use /v1/responses or set
reasoning_effort to 'none'" — which for an agent means giving up either its
tools or the reasoning that makes the strong models worth using.

The wire format differs enough to be worth naming: messages become a flat
`input` array where tool calls and their outputs are items in their own right
rather than fields on a message, the system prompt moves to `instructions`, and
the stream is a sequence of named events instead of choice deltas.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar

import aiohttp
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability
from axio.tool import Tool
from axio.types import StopReason, Usage

from axio_transport_openai import OpenAITransport, _strip_title

logger = logging.getLogger(__name__)

# Values accepted by the reasoning.effort field.
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

_ORPHAN_OUTPUT = "[Tool was not executed - context was interrupted or compacted]"


def _convert_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Function tools are flat here — no nested "function" object."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": _strip_title(tool.input_schema),
        }
        for tool in tools
    ]


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results and len(tool_results) == len(msg.content):
                for tr in tool_results:
                    content = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
                    items.append({"type": "function_call_output", "call_id": tr.tool_use_id, "output": content})
                continue
            parts: list[dict[str, Any]] = []
            for b in msg.content:
                if isinstance(b, TextBlock):
                    parts.append({"type": "input_text", "text": b.text})
                elif isinstance(b, ImageBlock):
                    encoded = base64.b64encode(b.data).decode("ascii")
                    parts.append({"type": "input_image", "image_url": f"data:{b.media_type};base64,{encoded}"})
            if parts:
                items.append({"role": "user", "content": parts})

        elif msg.role == "assistant":
            calls = [b for b in msg.content if isinstance(b, ToolUseBlock)]
            parts_out = [{"type": "output_text", "text": b.text} for b in msg.content if isinstance(b, TextBlock)]
            if parts_out:
                items.append({"role": "assistant", "content": parts_out})
            for b in calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": b.id,
                        "name": b.name,
                        "arguments": json.dumps(b.input),
                        "status": "completed",
                    }
                )

    # A call without its output makes the API reject the whole request, which
    # happens whenever a turn was interrupted or the context was compacted.
    answered = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    for item in list(items):
        if item.get("type") == "function_call" and item.get("call_id") not in answered:
            logger.warning("Synthesising placeholder output for unanswered call %s", item.get("call_id"))
            items.append({"type": "function_call_output", "call_id": item["call_id"], "output": _ORPHAN_OUTPUT})

    return items


@dataclass
class OpenAIResponsesTransport(OpenAITransport):
    name: str = "OpenAI Responses"
    # none|low|medium|high|xhigh|max. Left unset the API picks its own default;
    # set it to "none" to turn reasoning off entirely.
    reasoning_effort: str | None = None

    stream_path: ClassVar[str] = "responses"

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.id,
            "input": _convert_messages(messages),
            "stream": True,
            # Server-side conversation state is not used: axio keeps the history.
            "store": False,
        }
        if system:
            payload["instructions"] = system
        if tools:
            payload["tools"] = _convert_tools(tools)
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if self.reasoning_effort and Capability.reasoning in self.model.capabilities:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        payload["max_output_tokens"] = self.model.max_output_tokens
        if self.extra_params:
            payload.update(self.extra_params)
        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        usage = Usage(0, 0)
        stop_reason: StopReason | None = None
        # Argument deltas arrive keyed by item id, while the call was announced
        # under its call_id; without this map the two halves never join up.
        item_to_call: dict[str, str] = {}

        buffer = b""
        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                await asyncio.sleep(0)
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    continue
                try:
                    data: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Unparseable SSE payload: %s", raw[:200])
                    continue

                event_type = data.get("type", "")

                if event_type == "response.output_text.delta":
                    yield TextDelta(index=0, delta=data.get("delta", ""))

                elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                    yield ReasoningDelta(index=0, delta=data.get("delta", ""))

                elif event_type == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        if item.get("id"):
                            item_to_call[item["id"]] = call_id
                        yield ToolUseStart(
                            index=data.get("output_index", 0),
                            tool_use_id=call_id,
                            name=item.get("name", ""),
                        )

                elif event_type == "response.function_call_arguments.delta":
                    item_id = data.get("item_id", "")
                    yield ToolInputDelta(
                        index=data.get("output_index", 0),
                        tool_use_id=item_to_call.get(item_id, item_id),
                        partial_json=data.get("delta", ""),
                    )

                elif event_type == "response.completed":
                    response = data.get("response", {})
                    resp_usage = response.get("usage", {})
                    usage = Usage(
                        input_tokens=resp_usage.get("input_tokens", 0),
                        output_tokens=resp_usage.get("output_tokens", 0),
                    )
                    output = response.get("output", [])
                    if any(i.get("type") == "function_call" for i in output):
                        stop_reason = StopReason.tool_use
                    elif response.get("status") == "incomplete":
                        stop_reason = StopReason.max_tokens
                    else:
                        stop_reason = StopReason.end_turn

                elif event_type in ("response.failed", "error"):
                    response = data.get("response", data)
                    message = (response.get("error") or {}).get("message", "unknown error")
                    raise StreamError(f"OpenAI Responses API error: {message}")

        yield IterationEnd(iteration=0, stop_reason=stop_reason or StopReason.end_turn, usage=usage)
