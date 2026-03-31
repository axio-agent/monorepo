"""Tests for OpenAI-compatible CompletionTransport."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool, ToolHandler
from axio.types import StopReason, Usage

from axio_transport_openai import OPENAI_MODELS, OpenAITransport, ThinkTagParser, _strip_title

# ---------------------------------------------------------------------------
# Test tool handler
# ---------------------------------------------------------------------------


class GetWeather(ToolHandler):
    location: str
    units: str = "celsius"

    async def __call__(self) -> str:
        return f"Weather in {self.location}: 22{self.units[0]}"


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _text_chunks(text: str, finish_reason: str = "stop", usage: dict[str, int] | None = None) -> str:
    """Build SSE for a simple text response."""
    parts = list(text)
    lines = ""
    for i, ch in enumerate(parts):
        lines += _sse_chunk(
            {
                "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}],
            }
        )
    lines += _sse_chunk(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
    )
    lines += _sse_chunk(
        {
            "choices": [],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    lines += _sse_done()
    return lines


def _tool_call_chunks(
    tool_id: str,
    name: str,
    arguments: str,
    *,
    finish_reason: str = "tool_calls",
    usage: dict[str, int] | None = None,
) -> str:
    """Build SSE for a single tool call response."""
    lines = ""
    # First chunk: id + name
    lines += _sse_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": tool_id, "type": "function", "function": {"name": name}}],
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    # Argument chunks (split into halves)
    mid = len(arguments) // 2
    for part in [arguments[:mid], arguments[mid:]]:
        if part:
            lines += _sse_chunk(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": part}}]},
                            "finish_reason": None,
                        }
                    ],
                }
            )
    # Finish
    lines += _sse_chunk(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
    )
    lines += _sse_chunk(
        {
            "choices": [],
            "usage": usage or {"prompt_tokens": 15, "completion_tokens": 8},
        }
    )
    lines += _sse_done()
    return lines


# ---------------------------------------------------------------------------
# Fake OpenAI server
# ---------------------------------------------------------------------------


class FakeOpenAIServer:
    """aiohttp app that serves /v1/chat/completions and /v1/embeddings with configurable responses."""

    def __init__(self) -> None:
        self.responses: list[str] = []
        self.received_payloads: list[dict[str, Any]] = []
        self.status_code: int = 200
        self.error_body: str = ""
        self.error_headers: dict[str, str] = {}
        self._status_sequence: list[int] = []
        self._error_headers_sequence: list[dict[str, str]] = []
        self.embedding_responses: list[dict[str, Any]] = []
        self.embedding_status: int = 200
        self.embedding_error: str = ""
        self.embedding_error_headers: dict[str, str] = {}
        self._embedding_status_sequence: list[int] = []
        self._embedding_error_headers_sequence: list[dict[str, str]] = []
        self.received_embedding_payloads: list[dict[str, Any]] = []
        self._call_count = 0
        self._embedding_call_count = 0

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        app.router.add_post("/v1/embeddings", self._handle_embeddings)
        return app

    def _get_status(self) -> tuple[int, dict[str, str]]:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._status_sequence):
            hdrs = self._error_headers_sequence[idx] if idx < len(self._error_headers_sequence) else {}
            return self._status_sequence[idx], hdrs
        return self.status_code, self.error_headers

    def _get_embedding_status(self) -> tuple[int, dict[str, str]]:
        idx = self._embedding_call_count
        self._embedding_call_count += 1
        if idx < len(self._embedding_status_sequence):
            hdrs = (
                self._embedding_error_headers_sequence[idx]
                if idx < len(self._embedding_error_headers_sequence)
                else {}
            )
            return self._embedding_status_sequence[idx], hdrs
        return self.embedding_status, self.embedding_error_headers

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.received_payloads.append(payload)

        status, hdrs = self._get_status()
        if status != 200:
            return web.Response(status=status, text=self.error_body, headers=hdrs)

        sse_body = self.responses.pop(0) if self.responses else _sse_done()

        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(sse_body.encode("utf-8"))
        await resp.write_eof()
        return resp

    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.received_embedding_payloads.append(payload)

        status, hdrs = self._get_embedding_status()
        if status != 200:
            return web.Response(status=status, text=self.embedding_error, headers=hdrs)

        if self.embedding_responses:
            body = self.embedding_responses.pop(0)
        else:
            texts = payload.get("input", [])
            body = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1 * (i + 1)] * 3} for i in range(len(texts))
                ],
                "model": payload.get("model", "test"),
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        return web.json_response(body)


@pytest.fixture
async def fake_server() -> AsyncIterator[tuple[FakeOpenAIServer, str]]:
    server = FakeOpenAIServer()
    app = server.make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Get the bound address
    sock = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = sock.getsockname()[:2]
    base_url = f"http://{host}:{port}/v1"

    yield server, base_url

    await runner.cleanup()


@pytest.fixture
async def transport(fake_server: tuple[FakeOpenAIServer, str]) -> AsyncIterator[OpenAITransport]:
    _, base_url = fake_server
    async with aiohttp.ClientSession() as session:
        yield OpenAITransport(
            base_url=base_url,
            api_key="test-key",
            model=OPENAI_MODELS["gpt-4.1-mini"],
            session=session,
            retry_base_delay=0.0,
        )


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# Text streaming
# ---------------------------------------------------------------------------


async def test_text_streaming(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, _ = fake_server
    server.responses.append(_text_chunks("Hello"))

    with caplog.at_level(logging.DEBUG, logger="axio_transport_openai"):
        events = await _collect(transport.stream([], [], ""))

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_deltas) == 5  # H, e, l, l, o
    assert "".join(e.delta for e in text_deltas) == "Hello"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert len(ends) == 1
    assert ends[0].stop_reason == StopReason.end_turn
    assert ends[0].usage == Usage(10, 5)

    assert any("Stream complete" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tool call streaming
# ---------------------------------------------------------------------------


async def test_tool_call_streaming(fake_server: tuple[FakeOpenAIServer, str], transport: OpenAITransport) -> None:
    server, _ = fake_server
    args = json.dumps({"location": "Paris"})
    server.responses.append(_tool_call_chunks("call_abc", "get_weather", args))

    events = await _collect(transport.stream([], [], ""))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    assert len(starts) == 1
    assert starts[0].tool_use_id == "call_abc"
    assert starts[0].name == "get_weather"

    deltas = [e for e in events if isinstance(e, ToolInputDelta)]
    assert len(deltas) == 2
    assert all(d.tool_use_id == "call_abc" for d in deltas)
    assert "".join(d.partial_json for d in deltas) == args

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == StopReason.tool_use


# ---------------------------------------------------------------------------
# Multiple tool calls in one response
# ---------------------------------------------------------------------------


async def test_multiple_tool_calls(fake_server: tuple[FakeOpenAIServer, str], transport: OpenAITransport) -> None:
    server, _ = fake_server
    args_a = json.dumps({"location": "Paris"})
    args_b = json.dumps({"location": "London"})
    sse = ""
    # Tool A start
    sse += _sse_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather"}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    # Tool B start
    sse += _sse_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_weather"}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    # Tool A args
    sse += _sse_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_a}}]},
                    "finish_reason": None,
                }
            ],
        }
    )
    # Tool B args
    sse += _sse_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 1, "function": {"arguments": args_b}}]},
                    "finish_reason": None,
                }
            ],
        }
    )
    # Finish
    sse += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    sse += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 10}})
    sse += _sse_done()

    server.responses.append(sse)
    events = await _collect(transport.stream([], [], ""))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    assert len(starts) == 2
    assert starts[0].tool_use_id == "call_a"
    assert starts[0].index == 0
    assert starts[1].tool_use_id == "call_b"
    assert starts[1].index == 1

    deltas = [e for e in events if isinstance(e, ToolInputDelta)]
    assert len(deltas) == 2
    assert deltas[0].tool_use_id == "call_a"
    assert deltas[0].partial_json == args_a
    assert deltas[1].tool_use_id == "call_b"
    assert deltas[1].partial_json == args_b


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("openai_reason", "expected"),
    [
        ("stop", StopReason.end_turn),
        ("tool_calls", StopReason.tool_use),
        ("length", StopReason.max_tokens),
        ("content_filter", StopReason.error),
    ],
)
async def test_stop_reason_mapping(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
    openai_reason: str,
    expected: StopReason,
) -> None:
    server, _ = fake_server
    server.responses.append(_text_chunks("x", finish_reason=openai_reason))

    events = await _collect(transport.stream([], [], ""))

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == expected


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def test_build_payload_system_prompt() -> None:
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    payload = t.build_payload([], [], "You are helpful.")
    msgs = payload["messages"]
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 32_768


def test_build_payload_user_text() -> None:
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    messages = [Message(role="user", content=[TextBlock(text="Hello")])]
    payload = t.build_payload(messages, [], "")
    # No system message when empty
    assert payload["messages"][0] == {"role": "user", "content": "Hello"}


def test_build_payload_assistant_with_tool_calls() -> None:
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    messages = [
        Message(
            role="assistant",
            content=[
                TextBlock(text="Let me check."),
                ToolUseBlock(id="call_1", name="get_weather", input={"location": "Paris"}),
            ],
        ),
    ]
    payload = t.build_payload(messages, [], "")
    msg = payload["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me check."
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"location": "Paris"}


def test_build_payload_tool_results() -> None:
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    messages = [
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="call_1", content="22C"),
            ],
        ),
    ]
    payload = t.build_payload(messages, [], "")
    msg = payload["messages"][0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "22C"


def test_build_payload_tool_schema() -> None:
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    tool = Tool(name="get_weather", description="Get weather", handler=GetWeather)
    payload = t.build_payload([], [tool], "")
    assert len(payload["tools"]) == 1
    fn = payload["tools"][0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "get_weather"
    assert fn["function"]["description"] == "Get weather"
    params = fn["function"]["parameters"]
    assert "title" not in params
    assert "location" in params["properties"]


def test_build_payload_uses_model_spec_max_tokens() -> None:
    t = OpenAITransport(model=ModelSpec(id="custom-model", max_output_tokens=4096))
    payload = t.build_payload([], [], "")
    assert payload["max_tokens"] == 4096


def test_build_payload_image_block() -> None:
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    img_data = b"\x89PNG\r\n\x1a\nfake"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="What is in this image?"),
                ImageBlock(media_type="image/png", data=img_data),
            ],
        ),
    ]
    payload = t.build_payload(messages, [], "")
    msg = payload["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 2
    assert msg["content"][0] == {"type": "text", "text": "What is in this image?"}
    img_part = msg["content"][1]
    assert img_part["type"] == "image_url"
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_payload_image_only() -> None:
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    img_data = b"\xff\xd8\xff\xe0fake-jpeg"
    messages = [
        Message(role="user", content=[ImageBlock(media_type="image/jpeg", data=img_data)]),
    ]
    payload = t.build_payload(messages, [], "")
    msg = payload["messages"][0]
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "image_url"
    assert msg["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_build_payload_text_only_stays_string() -> None:
    """Text-only user messages should remain plain strings, not arrays."""
    t = OpenAITransport(model=OPENAI_MODELS["gpt-4.1-mini"])
    messages = [Message(role="user", content=[TextBlock(text="Hello")])]
    payload = t.build_payload(messages, [], "")
    assert payload["messages"][0]["content"] == "Hello"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_http_401(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, _ = fake_server
    server.status_code = 401
    server.error_body = '{"error": "invalid api key"}'

    with (
        caplog.at_level(logging.ERROR, logger="axio_transport_openai"),
        pytest.raises(StreamError, match="401"),
    ):
        await _collect(transport.stream([], [], ""))

    assert any(r.levelno == logging.ERROR and "401" in r.message for r in caplog.records)


async def test_http_500(fake_server: tuple[FakeOpenAIServer, str], transport: OpenAITransport) -> None:
    server, _ = fake_server
    server.status_code = 500
    server.error_body = "Internal Server Error"

    with pytest.raises(StreamError, match="500"):
        await _collect(transport.stream([], [], ""))


# ---------------------------------------------------------------------------
# _strip_title
# ---------------------------------------------------------------------------


def test_strip_title_recursive() -> None:
    schema = {
        "title": "Root",
        "type": "object",
        "properties": {
            "name": {"title": "Name", "type": "string"},
            "items": {"title": "Items", "type": "array", "items": [{"title": "Item", "type": "string"}]},
        },
    }
    result = _strip_title(schema)
    assert "title" not in result
    assert "title" not in result["properties"]["name"]
    assert "title" not in result["properties"]["items"]
    assert "title" not in result["properties"]["items"]["items"][0]


# ---------------------------------------------------------------------------
# Authorization header
# ---------------------------------------------------------------------------


async def test_auth_header_sent(fake_server: tuple[FakeOpenAIServer, str]) -> None:
    server, base_url = fake_server
    server.responses.append(_text_chunks("ok"))
    async with aiohttp.ClientSession() as session:
        t = OpenAITransport(
            base_url=base_url, api_key="sk-secret", model=OPENAI_MODELS["gpt-4.1-mini"], session=session
        )
        await _collect(t.stream([], [], ""))
    assert len(server.received_payloads) == 1
    assert server.received_payloads[0]["model"] == "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

_SPEC_A = ModelSpec(
    id="alpha-1",
    context_window=100,
    max_output_tokens=50,
    capabilities=frozenset({Capability.text, Capability.vision}),
    input_cost=5.0,
    output_cost=20.0,
)
_SPEC_B = ModelSpec(
    id="alpha-2",
    context_window=200,
    max_output_tokens=100,
    capabilities=frozenset({Capability.text, Capability.tool_use}),
    input_cost=1.0,
    output_cost=4.0,
)
_SPEC_C = ModelSpec(
    id="beta-1",
    context_window=300,
    max_output_tokens=150,
    capabilities=frozenset({Capability.text, Capability.vision, Capability.tool_use}),
    input_cost=10.0,
    output_cost=2.0,
)


def _sample_registry() -> ModelRegistry:
    return ModelRegistry([_SPEC_A, _SPEC_B, _SPEC_C])


def test_registry_get() -> None:
    r = _sample_registry()
    assert r.get("alpha-1") == _SPEC_A
    assert r.get("missing") is None


def test_registry_getitem() -> None:
    r = _sample_registry()
    assert r["beta-1"] == _SPEC_C


def test_registry_getitem_missing() -> None:
    r = _sample_registry()
    with pytest.raises(KeyError):
        r["missing"]


def test_registry_contains() -> None:
    r = _sample_registry()
    assert "alpha-1" in r
    assert "missing" not in r


def test_registry_len() -> None:
    r = _sample_registry()
    assert len(r) == 3


def test_registry_iter() -> None:
    r = _sample_registry()
    assert list(r) == [_SPEC_A, _SPEC_B, _SPEC_C]  # type: ignore[comparison-overlap]


def test_registry_setitem() -> None:
    r = ModelRegistry()
    r["new"] = _SPEC_A
    assert r["new"] == _SPEC_A
    assert len(r) == 1


def test_registry_by_prefix() -> None:
    r = _sample_registry()
    sub = r.by_prefix("alpha")
    assert isinstance(sub, ModelRegistry)
    assert sub.ids() == ["alpha-1", "alpha-2"]


def test_registry_by_prefix_empty() -> None:
    r = _sample_registry()
    sub = r.by_prefix("gamma")
    assert len(sub) == 0


def test_registry_by_capability_single() -> None:
    r = _sample_registry()
    sub = r.by_capability(Capability.vision)
    assert sorted(sub.ids()) == ["alpha-1", "beta-1"]


def test_registry_by_capability_multiple() -> None:
    r = _sample_registry()
    sub = r.by_capability(Capability.vision, Capability.tool_use)
    assert sub.ids() == ["beta-1"]


def test_registry_chaining() -> None:
    r = _sample_registry()
    sub = r.by_prefix("alpha").by_capability(Capability.vision)
    assert sub.ids() == ["alpha-1"]


def test_registry_ids() -> None:
    r = _sample_registry()
    assert r.ids() == ["alpha-1", "alpha-2", "beta-1"]


def test_registry_eq() -> None:
    a = ModelRegistry([_SPEC_A])
    b = ModelRegistry([_SPEC_A])
    assert a == b
    assert a != ModelRegistry([_SPEC_B])


def test_registry_eq_dict() -> None:
    r = ModelRegistry([_SPEC_A])
    assert r == {"alpha-1": _SPEC_A}


def test_registry_repr() -> None:
    r = ModelRegistry()
    assert repr(r) == "ModelRegistry({})"


def test_registry_delitem() -> None:
    r = _sample_registry()
    del r["alpha-1"]
    assert "alpha-1" not in r
    assert len(r) == 2


def test_registry_delitem_missing() -> None:
    r = _sample_registry()
    with pytest.raises(KeyError):
        del r["missing"]


def test_registry_setitem_rejects_non_spec() -> None:
    r = ModelRegistry()
    with pytest.raises(ValueError, match="ModelSpec"):
        r["bad"] = "not a spec"  # type: ignore[assignment]


def test_registry_search_single_term() -> None:
    r = _sample_registry()
    sub = r.search("alpha")
    assert isinstance(sub, ModelRegistry)
    assert len(sub) == 2
    assert "alpha-1" in sub
    assert "alpha-2" in sub


def test_registry_search_multiple_terms() -> None:
    r = _sample_registry()
    sub = r.search("alpha", "1")
    assert list(sub) == [_SPEC_A]  # type: ignore[comparison-overlap]


def test_registry_search_no_match() -> None:
    r = _sample_registry()
    sub = r.search("gamma")
    assert len(sub) == 0


def test_registry_search_chaining() -> None:
    r = _sample_registry()
    sub = r.search("alpha").by_capability(Capability.vision)
    assert list(sub) == [_SPEC_A]  # type: ignore[comparison-overlap]


def test_registry_mutable_mapping_protocol() -> None:
    r = _sample_registry()
    assert set(r.keys()) == {"alpha-1", "alpha-2", "beta-1"}
    assert _SPEC_A in r.values()
    assert ("alpha-1", _SPEC_A) in r.items()


def test_registry_by_cost_ascending() -> None:
    r = _sample_registry()
    ordered = r.by_cost()
    assert ordered.ids() == ["alpha-2", "alpha-1", "beta-1"]


def test_registry_by_cost_descending() -> None:
    r = _sample_registry()
    ordered = r.by_cost(desc=True)
    assert ordered.ids() == ["beta-1", "alpha-1", "alpha-2"]


def test_registry_by_cost_output() -> None:
    r = _sample_registry()
    ordered = r.by_cost(output=True)
    # output_cost: beta-1=2.0, alpha-2=4.0, alpha-1=20.0
    assert ordered.ids() == ["beta-1", "alpha-2", "alpha-1"]


def test_registry_by_cost_output_desc() -> None:
    r = _sample_registry()
    ordered = r.by_cost(output=True, desc=True)
    assert ordered.ids() == ["alpha-1", "alpha-2", "beta-1"]


def test_registry_by_cost_chaining() -> None:
    r = _sample_registry()
    cheap_alpha = r.by_prefix("alpha").by_cost()
    assert cheap_alpha.ids() == ["alpha-2", "alpha-1"]


def test_model_spec_cost_defaults() -> None:
    spec = ModelSpec(id="test")
    assert spec.input_cost == 0.0
    assert spec.output_cost == 0.0


def test_openai_models_is_registry() -> None:
    assert isinstance(OPENAI_MODELS, ModelRegistry)
    assert len(OPENAI_MODELS) > 0
    assert "gpt-4.1-mini" in OPENAI_MODELS


def test_openai_models_have_costs() -> None:
    for model_id, spec in OPENAI_MODELS.items():
        assert spec.input_cost > 0, f"{model_id} has no input_cost"
        assert spec.output_cost > 0, f"{model_id} has no output_cost"


def test_openai_models_cheapest_first() -> None:
    ordered = OPENAI_MODELS.by_cost()
    costs = [spec.input_cost for spec in ordered.values()]
    assert costs == sorted(costs)


# ---------------------------------------------------------------------------
# ThinkTagParser
# ---------------------------------------------------------------------------


class TestThinkTagParser:
    def test_plain_text_no_tags(self) -> None:
        p = ThinkTagParser()
        assert p.feed("hello world") == [("text", "hello world")]

    def test_full_think_then_answer(self) -> None:
        p = ThinkTagParser()
        result = p.feed("<think>reasoning</think>answer")
        assert result == [("reasoning", "reasoning"), ("text", "answer")]

    def test_open_tag_split_across_chunks(self) -> None:
        p = ThinkTagParser()
        assert p.feed("<thi") == []
        assert p.feed("nk>hello") == [("reasoning", "hello")]

    def test_close_tag_split_across_chunks(self) -> None:
        p = ThinkTagParser()
        result1 = p.feed("<think>reasoning")
        assert result1 == [("reasoning", "reasoning")]
        result2 = p.feed("</thi")
        assert result2 == []
        result3 = p.feed("nk>answer")
        assert result3 == [("text", "answer")]

    def test_empty_think_block(self) -> None:
        p = ThinkTagParser()
        result = p.feed("<think></think>answer")
        assert result == [("text", "answer")]

    def test_think_block_no_answer(self) -> None:
        p = ThinkTagParser()
        result = p.feed("<think>just reasoning</think>")
        assert result == [("reasoning", "just reasoning")]

    def test_flush_emits_buffered_partial(self) -> None:
        p = ThinkTagParser()
        assert p.feed("<thi") == []
        result = p.flush()
        assert result == [("text", "<thi")]

    def test_flush_emits_reasoning_in_progress(self) -> None:
        p = ThinkTagParser()
        # "partial" is emitted eagerly since it can't be a tag prefix
        result = p.feed("<think>partial")
        assert result == [("reasoning", "partial")]
        # Nothing left to flush
        assert p.flush() == []

    def test_flush_emits_buffered_reasoning_partial_tag(self) -> None:
        """When buffer ends with a partial close tag, flush emits it as reasoning."""
        p = ThinkTagParser()
        result1 = p.feed("<think>data")
        assert result1 == [("reasoning", "data")]
        result2 = p.feed("</thi")
        # "</thi" is buffered as potential close tag
        assert result2 == []
        flushed = p.flush()
        assert flushed == [("reasoning", "</thi")]

    def test_flush_empty(self) -> None:
        p = ThinkTagParser()
        assert p.flush() == []

    def test_incremental_streaming(self) -> None:
        """Simulate real streaming: tags arrive char-by-char."""
        p = ThinkTagParser()
        all_results: list[tuple[str, str]] = []
        for ch in "<think>Hmm let me think</think>42":
            all_results.extend(p.feed(ch))
        all_results.extend(p.flush())
        reasoning = "".join(t for k, t in all_results if k == "reasoning")
        text = "".join(t for k, t in all_results if k == "text")
        assert reasoning == "Hmm let me think"
        assert text == "42"

    def test_newlines_around_tags(self) -> None:
        p = ThinkTagParser()
        result = p.feed("<think>\nreasoning\n</think>\nanswer")
        assert result == [("reasoning", "\nreasoning\n"), ("text", "\nanswer")]


# ---------------------------------------------------------------------------
# SSE integration: reasoning deltas
# ---------------------------------------------------------------------------


def _think_text_chunks(think: str, answer: str) -> str:
    """Build SSE with <think>...</think> wrapped content followed by answer."""
    content = f"<think>{think}</think>{answer}"
    lines = ""
    for ch in content:
        lines += _sse_chunk({"choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]})
    lines += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    lines += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    lines += _sse_done()
    return lines


async def test_sse_reasoning_then_text(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    server, _ = fake_server
    server.responses.append(_think_text_chunks("let me think", "42"))

    events = await _collect(transport.stream([], [], ""))

    reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
    text = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in reasoning) == "let me think"
    assert "".join(e.delta for e in text) == "42"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == StopReason.end_turn


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------


async def test_embed_returns_vectors(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    server, _ = fake_server
    server.embedding_responses.append(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.2, 0.3]},
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.4]},
            ],
            "model": "gpt-4.1-mini",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
    )

    result = await transport.embed(["hello", "world"])
    assert len(result) == 2
    # Sorted by index: index 0 first, index 1 second
    assert result[0] == [0.1, 0.4]
    assert result[1] == [0.2, 0.3]


async def test_embed_sends_correct_payload(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    server, _ = fake_server
    await transport.embed(["test input"])
    assert len(server.received_embedding_payloads) == 1
    payload = server.received_embedding_payloads[0]
    assert payload["model"] == "gpt-4.1-mini"
    assert payload["input"] == ["test input"]


async def test_embed_error_raises(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    server, _ = fake_server
    server.embedding_status = 400
    server.embedding_error = "Bad request"

    with pytest.raises(StreamError, match="400"):
        await transport.embed(["test"])


# ---------------------------------------------------------------------------
# Retry with Retry-After header
# ---------------------------------------------------------------------------


async def test_stream_429_uses_retry_after_header(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """429 with Retry-After header → retries and succeeds."""
    server, _ = fake_server
    server._status_sequence = [429, 200]
    server._error_headers_sequence = [{"Retry-After": "0"}]
    server.error_body = "Rate limited"
    server.responses.append(_text_chunks("ok"))

    events = await _collect(transport.stream([], [], ""))
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in text_deltas) == "ok"
    assert len(server.received_payloads) == 2


async def test_stream_429_exhausts_retries(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """429 on all attempts → raises StreamError."""
    server, _ = fake_server
    server.status_code = 429
    server.error_body = "Rate limited"
    server.error_headers = {"Retry-After": "0"}

    with pytest.raises(StreamError, match="429"):
        await _collect(transport.stream([], [], ""))
    assert len(server.received_payloads) == 10


async def test_stream_500_retries_without_retry_after(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """500 without Retry-After header → falls back to exponential backoff and succeeds."""
    server, _ = fake_server
    server._status_sequence = [500, 200]
    server.error_body = "Internal Server Error"
    server.responses.append(_text_chunks("recovered"))

    events = await _collect(transport.stream([], [], ""))
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in text_deltas) == "recovered"


async def test_embed_429_retries_with_retry_after(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """embed() retries on 429 using Retry-After header."""
    server, _ = fake_server
    server._embedding_status_sequence = [429, 200]
    server._embedding_error_headers_sequence = [{"Retry-After": "0"}]
    server.embedding_error = "Rate limited"

    result = await transport.embed(["hello"])
    assert len(result) == 1
    assert len(server.received_embedding_payloads) == 2


async def test_embed_429_exhausts_retries(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """embed() 429 on all attempts → raises StreamError."""
    server, _ = fake_server
    server.embedding_status = 429
    server.embedding_error = "Rate limited"
    server.embedding_error_headers = {"Retry-After": "0"}

    with pytest.raises(StreamError, match="429"):
        await transport.embed(["test"])
    assert len(server.received_embedding_payloads) == 10


async def test_embed_500_retries_and_recovers(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """embed() retries on 500 and succeeds on second attempt."""
    server, _ = fake_server
    server._embedding_status_sequence = [500, 200]
    server.embedding_error = "Internal Server Error"

    result = await transport.embed(["hello"])
    assert len(result) == 1
    assert len(server.received_embedding_payloads) == 2


def test_get_retry_delay_uses_header() -> None:
    """_get_retry_delay prefers Retry-After header value."""
    t = OpenAITransport(retry_base_delay=1.0)
    # Create a mock response with Retry-After header
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.headers = {"Retry-After": "5"}
    assert t._get_retry_delay(resp, 1) == 5.0


def test_get_retry_delay_falls_back_to_exponential() -> None:
    """_get_retry_delay uses exponential backoff when no header."""
    t = OpenAITransport(retry_base_delay=2.0)
    assert t._get_retry_delay(None, 1) == 2.0  # 2 * 2^0
    assert t._get_retry_delay(None, 2) == 4.0  # 2 * 2^1
    assert t._get_retry_delay(None, 3) == 8.0  # 2 * 2^2


def test_get_retry_delay_ignores_invalid_header() -> None:
    """_get_retry_delay falls back when Retry-After is not a number."""
    t = OpenAITransport(retry_base_delay=1.0)
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.headers = {"Retry-After": "invalid"}
    assert t._get_retry_delay(resp, 1) == 1.0


# ---------------------------------------------------------------------------
# Tool call: id + arguments in the same delta (vLLM/Nebius pattern)
# ---------------------------------------------------------------------------


async def test_tool_call_args_in_first_chunk(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """When id, name, AND arguments arrive in the same SSE delta, arguments must not be dropped."""
    server, _ = fake_server
    args = json.dumps({"directory": "."})

    # Single delta with id + name + arguments (vLLM/Nebius style)
    sse = _sse_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_vllm_1",
                                "type": "function",
                                "function": {"name": "list_files", "arguments": args},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    sse += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    sse += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    sse += _sse_done()

    server.responses.append(sse)
    events = await _collect(transport.stream([], [], ""))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    assert len(starts) == 1
    assert starts[0].tool_use_id == "call_vllm_1"
    assert starts[0].name == "list_files"

    deltas = [e for e in events if isinstance(e, ToolInputDelta)]
    assert len(deltas) == 1
    assert deltas[0].tool_use_id == "call_vllm_1"
    assert deltas[0].partial_json == args


# ---------------------------------------------------------------------------
# SSE buffer flush: last data line lacks trailing \n
# ---------------------------------------------------------------------------


async def test_sse_buffer_flush(
    fake_server: tuple[FakeOpenAIServer, str],
    transport: OpenAITransport,
) -> None:
    """When the last SSE data line has no trailing newline, data must not be lost."""
    server, _ = fake_server

    # Build SSE where the final usage chunk has no trailing \n
    sse = _sse_chunk({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]})
    sse += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    # Usage chunk without trailing \n\n — just "data: {...}" with no newline
    usage_data = json.dumps({"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 7}})
    sse += f"data: {usage_data}"  # no trailing \n

    server.responses.append(sse)
    events = await _collect(transport.stream([], [], ""))

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in text_deltas) == "hi"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert len(ends) == 1
    assert ends[0].usage == Usage(42, 7)
    assert ends[0].stop_reason == StopReason.end_turn
