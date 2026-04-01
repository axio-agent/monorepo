"""Tests for OpenRouter CompletionTransport."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from axio.events import IterationEnd, StreamEvent, TextDelta
from axio.exceptions import StreamError
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.types import StopReason, Usage

from axio_transport_openai.openrouter import OpenRouterTransport

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _text_chunks(text: str) -> str:
    lines = ""
    for ch in text:
        lines += _sse_chunk({"choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]})
    lines += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    lines += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    lines += _sse_done()
    return lines


# ---------------------------------------------------------------------------
# Fake server with /api/v1/models and /api/v1/chat/completions
# ---------------------------------------------------------------------------


class FakeOpenRouterServer:
    def __init__(self) -> None:
        self.sse_responses: list[str] = []
        self.received_payloads: list[dict[str, Any]] = []
        self.models_response: dict[str, Any] = {"data": []}
        self.models_status: int = 200
        self.completions_status: int = 200
        self.error_body: str = ""

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/api/v1/models", self._handle_models)
        app.router.add_post("/api/v1/chat/completions", self._handle_completions)
        return app

    async def _handle_models(self, request: web.Request) -> web.Response:
        if self.models_status != 200:
            return web.Response(status=self.models_status, text=self.error_body)
        return web.json_response(self.models_response)

    async def _handle_completions(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.received_payloads.append(payload)

        if self.completions_status != 200:
            return web.Response(status=self.completions_status, text=self.error_body)

        sse_body = self.sse_responses.pop(0) if self.sse_responses else _sse_done()
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(sse_body.encode("utf-8"))
        await resp.write_eof()
        return resp


@pytest.fixture
async def fake_server() -> AsyncIterator[tuple[FakeOpenRouterServer, str]]:
    server = FakeOpenRouterServer()
    app = server.make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    sock = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = sock.getsockname()[:2]
    base_url = f"http://{host}:{port}/api/v1"

    yield server, base_url

    await runner.cleanup()


@pytest.fixture
async def transport(fake_server: tuple[FakeOpenRouterServer, str]) -> AsyncIterator[OpenRouterTransport]:
    _, base_url = fake_server
    async with aiohttp.ClientSession() as session:
        yield OpenRouterTransport(base_url=base_url, api_key="test-key", session=session)


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_base_url() -> None:
    t = OpenRouterTransport()
    assert t.base_url == "https://openrouter.ai/api/v1"


def test_default_model() -> None:
    t = OpenRouterTransport()
    assert t.model.id == "google/gemini-2.5-pro-preview"


def test_default_models_inherited() -> None:
    t = OpenRouterTransport()
    assert len(t.models) > 0
    assert "gpt-4.1-mini" in t.models


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


async def test_fetch_models_populates_registry(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {"id": "openai/gpt-4", "context_length": 8192, "top_provider": {"max_completion_tokens": 4096}},
            {
                "id": "anthropic/claude-3-opus",
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 4096},
            },  # noqa: E501
        ]
    }

    with caplog.at_level(logging.INFO, logger="axio_transport_openai.openrouter"):
        await transport.fetch_models()

    assert isinstance(transport.models, ModelRegistry)
    assert "openai/gpt-4" in transport.models
    assert "anthropic/claude-3-opus" in transport.models
    assert any("Loaded 2 models" in r.message for r in caplog.records)


async def test_fetch_models_populates_specs(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "openai/gpt-4o",
                "context_length": 128000,
                "top_provider": {"max_completion_tokens": 16384, "context_length": 128000, "is_moderated": False},
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "tokenizer": "GPT",
                    "instruct_type": "chatml",
                },
                "supported_parameters": ["temperature", "tools", "max_tokens"],
                "pricing": {"prompt": "0.0000025", "completion": "0.00001", "request": "0", "image": "0"},
            },
            {
                "id": "openai/text-embedding-3-small",
                "context_length": 8192,
                "top_provider": {"max_completion_tokens": 0},
                "architecture": {
                    "modality": "text->embedding",
                    "input_modalities": ["text"],
                    "output_modalities": ["embedding"],
                },
                "supported_parameters": [],
                "pricing": {"prompt": "0.00000002", "completion": "0"},
            },
        ]
    }

    await transport.fetch_models()

    gpt4o = transport.models["openai/gpt-4o"]
    assert gpt4o.context_window == 128000
    assert gpt4o.max_output_tokens == 16384
    assert Capability.tool_use in gpt4o.capabilities
    assert Capability.vision in gpt4o.capabilities
    assert gpt4o.input_cost == pytest.approx(2.5)
    assert gpt4o.output_cost == pytest.approx(10.0)

    emb = transport.models["openai/text-embedding-3-small"]
    assert emb.context_window == 8192
    assert Capability.embedding in emb.capabilities
    assert Capability.vision not in emb.capabilities
    assert Capability.tool_use not in emb.capabilities


async def test_fetch_models_clears_and_repopulates(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    transport.models["custom/model"] = ModelSpec(id="custom/model", context_window=4096, max_output_tokens=1024)

    server.models_response = {
        "data": [
            {"id": "openai/gpt-4", "context_length": 8192, "top_provider": {"max_completion_tokens": 4096}},
        ]
    }

    await transport.fetch_models()

    assert "custom/model" not in transport.models
    assert "openai/gpt-4" in transport.models


async def test_fetch_models_empty(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {"data": []}

    await transport.fetch_models()

    assert len(transport.models) == 0


async def test_fetch_models_error(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_status = 401
    server.error_body = "Unauthorized"

    with pytest.raises(StreamError, match="401"):
        await transport.fetch_models()


async def test_fetch_models_context_from_top_provider_fallback(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    """context_length from top_provider used when top-level field is missing/null."""
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "some/model",
                "context_length": None,
                "top_provider": {"context_length": 32768, "max_completion_tokens": 2048},
            },
        ]
    }

    await transport.fetch_models()

    assert transport.models["some/model"].context_window == 32768


async def test_fetch_models_defaults_when_missing(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    """Minimal model entry with no context/pricing fields gets safe defaults."""
    server, _ = fake_server
    server.models_response = {
        "data": [
            {"id": "some/minimal"},
        ]
    }

    await transport.fetch_models()

    m = transport.models["some/minimal"]
    assert m.context_window == 128_000
    assert m.max_output_tokens == 8_000
    assert m.input_cost == 0.0
    assert m.output_cost == 0.0


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


async def test_no_tools_no_capability(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "text-only/model",
                "context_length": 4096,
                "top_provider": {"max_completion_tokens": 512},
                "supported_parameters": ["temperature", "max_tokens"],
            }
        ]
    }

    await transport.fetch_models()

    assert Capability.tool_use not in transport.models["text-only/model"].capabilities


# ---------------------------------------------------------------------------
# Streaming (inherited from OpenAITransport)
# ---------------------------------------------------------------------------


async def test_text_streaming(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.sse_responses.append(_text_chunks("Hi"))

    events = await _collect(transport.stream([], [], ""))

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in text_deltas) == "Hi"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == StopReason.end_turn
    assert ends[0].usage == Usage(10, 5)
