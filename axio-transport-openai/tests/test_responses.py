import json
from typing import Any

import pytest
from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.tool import Tool
from axio.types import StopReason

from axio_transport_openai.responses import OpenAIResponsesTransport, _convert_messages

_REASONING = ModelSpec(
    id="gpt-5.6",
    capabilities=frozenset({Capability.text, Capability.reasoning, Capability.tool_use}),
    max_output_tokens=128_000,
    context_window=1_050_000,
)


async def _echo(text: str) -> str:
    return text


def _transport(**kwargs: Any) -> OpenAIResponsesTransport:
    return OpenAIResponsesTransport(model=_REASONING, **kwargs)


def test_streams_from_the_responses_endpoint() -> None:
    assert OpenAIResponsesTransport.stream_path == "responses"


def test_system_prompt_moves_to_instructions() -> None:
    payload = _transport().build_payload([Message(role="user", content=[TextBlock(text="hi")])], [], "be terse")
    assert payload["instructions"] == "be terse"
    assert payload["store"] is False
    assert payload["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]


def test_tools_are_flat_not_nested() -> None:
    payload = _transport().build_payload([], [Tool[Any](name="echo", handler=_echo)], "")
    [tool] = payload["tools"]
    # chat/completions nests these under "function"; responses does not.
    assert tool["type"] == "function"
    assert tool["name"] == "echo"
    assert "parameters" in tool


def test_reasoning_effort_is_sent_only_when_supported() -> None:
    assert "reasoning" not in _transport().build_payload([], [], "")
    payload = _transport(reasoning_effort="high").build_payload([], [], "")
    assert payload["reasoning"] == {"effort": "high"}

    plain = ModelSpec(id="gpt-4o", capabilities=frozenset({Capability.text, Capability.tool_use}))
    payload = OpenAIResponsesTransport(model=plain, reasoning_effort="high").build_payload([], [], "")
    assert "reasoning" not in payload


def test_tool_calls_and_results_become_top_level_items() -> None:
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="call-1", name="echo", input={"text": "x"})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call-1", content="x")]),
    ]
    items = _convert_messages(messages)
    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == "call-1"
    assert json.loads(items[0]["arguments"]) == {"text": "x"}
    assert items[1] == {"type": "function_call_output", "call_id": "call-1", "output": "x"}


def test_unanswered_call_gets_a_placeholder_output() -> None:
    # The API rejects the whole request when a call has no output, which is what
    # an interrupted turn leaves behind.
    items = _convert_messages([Message(role="assistant", content=[ToolUseBlock(id="orphan", name="echo", input={})])])
    assert [i["type"] for i in items] == ["function_call", "function_call_output"]
    assert "not executed" in items[1]["output"]


class _FakeContent:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._body = b"".join(f"data: {json.dumps(e)}\n".encode() for e in events)

    async def iter_any(self):  # type: ignore[no-untyped-def]
        yield self._body


class _FakeResponse:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.content = _FakeContent(events)


async def _collect(events: list[dict[str, Any]]) -> list[Any]:
    return [e async for e in _transport()._parse_sse(_FakeResponse(events))]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_text_and_reasoning_deltas_are_separated() -> None:
    events = await _collect(
        [
            {"type": "response.reasoning_text.delta", "delta": "thinking"},
            {"type": "response.output_text.delta", "delta": "hello"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 3, "output_tokens": 4}}},
        ]
    )
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ReasoningDelta", "TextDelta", "IterationEnd"]
    assert events[-1].usage.input_tokens == 3
    assert events[-1].stop_reason is StopReason.end_turn


@pytest.mark.asyncio
async def test_argument_deltas_are_joined_to_the_call_id() -> None:
    events = await _collect(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "id": "item-1", "call_id": "call-1", "name": "echo"},
            },
            {"type": "response.function_call_arguments.delta", "item_id": "item-1", "delta": '{"text"'},
            {"type": "response.completed", "response": {"output": [{"type": "function_call"}]}},
        ]
    )
    start, delta, end = events
    assert start.tool_use_id == "call-1"
    # Deltas arrive keyed by item id and must be reported under the call id.
    assert delta.tool_use_id == "call-1"
    assert end.stop_reason is StopReason.tool_use
