"""Building a Responses request: instructions, input items, and tool declarations."""

import base64
import json
import logging
from typing import Any

from axio.blocks import ImageBlock, ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.messages import Message
from axio.tool import Tool
from axio.types import StopReason

logger = logging.getLogger(__name__)

STOP_REASONS: dict[str, StopReason] = {
    "completed": StopReason.end_turn,
    "end_turn": StopReason.end_turn,
    "stop": StopReason.end_turn,
    "max_output_tokens": StopReason.max_tokens,
    "length": StopReason.max_tokens,
    "cancelled": StopReason.cancelled,
    "content_filter": StopReason.refusal,
}


def strip_title(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove pydantic 'title' keys from a JSON schema recursively."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if isinstance(value, dict):
            out[key] = strip_title(value)
        elif isinstance(value, list):
            out[key] = [strip_title(item) if isinstance(item, dict) else item for item in value]
        else:
            out[key] = value
    return out


def convert_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio Tool list to Responses API function tool dicts."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": strip_title(tool.input_schema),
        }
        for tool in tools
    ]


def convert_messages(messages: list[Message], system: str) -> tuple[str, list[dict[str, Any]]]:
    """Convert axio Message list to Responses API input array.

    Returns (instructions, input_items).
    """
    items: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            # Check if this is purely tool results
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results and len(tool_results) == len(msg.content):
                for tr in tool_results:
                    content = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tr.tool_use_id,
                            "output": content,
                        }
                    )
            else:
                content_parts: list[dict[str, Any]] = []
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        content_parts.append({"type": "input_text", "text": b.text})
                    elif isinstance(b, ImageBlock):
                        encoded = base64.b64encode(b.data).decode("ascii")
                        data_uri = f"data:{b.media_type};base64,{encoded}"
                        content_parts.append({"type": "input_image", "image_url": data_uri})
                if content_parts:
                    items.append({"role": "user", "content": content_parts})

        elif msg.role == "assistant":
            # Collect text and tool uses
            content_parts_a: list[dict[str, Any]] = []
            for b in msg.content:
                if isinstance(b, ReasoningBlock):
                    # `id` and `summary` are required beside the proof. Sent without the proof the
                    # item says nothing, so a block that never got one is left out rather than
                    # replayed empty.
                    if b.id and b.signature:
                        items.append(
                            {
                                "type": "reasoning",
                                "id": b.id,
                                "encrypted_content": b.signature,
                                "summary": [],
                            }
                        )
                    else:
                        logger.debug("Dropping a reasoning block with no encrypted content to replay")
                elif isinstance(b, TextBlock):
                    content_parts_a.append({"type": "output_text", "text": b.text})
                elif isinstance(b, ToolUseBlock):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": b.id,
                            "name": b.name,
                            "arguments": json.dumps(b.input),
                            "status": "completed",
                        }
                    )
            if content_parts_a:
                items.insert(
                    len(items) - sum(1 for b in msg.content if isinstance(b, ToolUseBlock)),
                    {
                        "role": "assistant",
                        "content": content_parts_a,
                    },
                )

    # Synthesize placeholder outputs for orphan function_calls (no corresponding output)
    output_ids = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    for item in list(items):
        if item.get("type") == "function_call" and item.get("call_id") not in output_ids:
            call_id = item.get("call_id", "")
            logger.warning("Synthesizing placeholder output for orphan function_call: call_id=%s", call_id)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "[Tool was not executed - context was interrupted or compacted]",
                }
            )

    return system, items
