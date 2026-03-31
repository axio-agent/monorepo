"""Tests for Nebius AI Studio CompletionTransport."""

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

from axio_transport_nebius import NebiusTransport

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
# Fake server with /v1/models and /v1/chat/completions
# ---------------------------------------------------------------------------


class FakeNebiusServer:
    def __init__(self) -> None:
        self.sse_responses: list[str] = []
        self.received_payloads: list[dict[str, Any]] = []
        self.models_response: dict[str, Any] = {"object": "list", "data": []}
        self.models_status: int = 200
        self.completions_status: int = 200
        self.error_body: str = ""
        self.models_params: dict[str, str] = {}

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/models", self._handle_models)
        app.router.add_post("/v1/chat/completions", self._handle_completions)
        return app

    async def _handle_models(self, request: web.Request) -> web.Response:
        self.models_params = dict(request.query)
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
async def fake_server() -> AsyncIterator[tuple[FakeNebiusServer, str]]:
    server = FakeNebiusServer()
    app = server.make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    sock = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = sock.getsockname()[:2]
    base_url = f"http://{host}:{port}/v1"

    yield server, base_url

    await runner.cleanup()


@pytest.fixture
async def transport(fake_server: tuple[FakeNebiusServer, str]) -> AsyncIterator[NebiusTransport]:
    _, base_url = fake_server
    async with aiohttp.ClientSession() as session:
        yield NebiusTransport(base_url=base_url, api_key="test-key", session=session)


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_base_url() -> None:
    t = NebiusTransport()
    assert t.base_url == "https://api.tokenfactory.nebius.com/v1"


def test_default_model() -> None:
    t = NebiusTransport()
    assert t.model.id == "deepseek-ai/DeepSeek-V3-0324"


def test_default_models_inherited() -> None:
    t = NebiusTransport()
    assert len(t.models) > 0
    assert "gpt-4.1-mini" in t.models


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


async def test_fetch_models_sends_verbose(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {"object": "list", "data": []}

    await transport.fetch_models()

    assert server.models_params.get("verbose") == "true"


async def test_fetch_models_populates_registry(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {"id": "meta-llama/Llama-3.3-70B-Instruct", "object": "model"},
            {"id": "deepseek-ai/DeepSeek-V3-0324", "object": "model"},
        ],
    }

    with caplog.at_level(logging.INFO, logger="axio_transport_nebius"):
        await transport.fetch_models()

    assert isinstance(transport.models, ModelRegistry)
    assert "meta-llama/Llama-3.3-70B-Instruct" in transport.models
    assert "deepseek-ai/DeepSeek-V3-0324" in transport.models

    assert any("Loaded 2 models" in r.message for r in caplog.records)


async def test_fetch_models_populates_specs(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-3.3-70B-Instruct",
                "object": "model",
                "context_length": 131072,
                "max_output_tokens": 4096,
                "supported_features": ["tools", "json_mode", "structured_outputs"],
                "pricing": {"prompt": "0.0000004", "completion": "0.0000018"},
            },
            {
                "id": "deepseek-ai/DeepSeek-V3-0324",
                "object": "model",
                "context_length": 65536,
                "max_output_tokens": 8192,
            },
            {
                "id": "some/basic-model",
                "object": "model",
                # no context_length / max_output_tokens / pricing → defaults
            },
        ],
    }

    await transport.fetch_models()

    llama = transport.models["meta-llama/Llama-3.3-70B-Instruct"]
    assert llama.id == "meta-llama/Llama-3.3-70B-Instruct"
    assert llama.context_window == 131072
    assert llama.max_output_tokens == 4096
    assert llama.capabilities == frozenset({"tool_use", "json_mode", "structured_outputs"})
    assert llama.input_cost == pytest.approx(0.4)
    assert llama.output_cost == pytest.approx(1.8)

    ds = transport.models["deepseek-ai/DeepSeek-V3-0324"]
    assert ds.context_window == 65536
    assert ds.max_output_tokens == 8192
    assert ds.capabilities == frozenset()
    assert ds.input_cost == 0.0
    assert ds.output_cost == 0.0

    basic = transport.models["some/basic-model"]
    assert basic.context_window == 128_000
    assert basic.max_output_tokens == 25_000
    assert basic.input_cost == 0.0


async def test_fetch_models_clears_and_repopulates(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    server, _ = fake_server
    # Pre-populate with a custom spec
    transport.models["custom/model"] = ModelSpec(id="custom/model", context_window=128_000, max_output_tokens=8_192)

    server.models_response = {
        "object": "list",
        "data": [
            {"id": "deepseek-ai/DeepSeek-V3-0324", "object": "model", "context_length": 65536},
        ],
    }

    await transport.fetch_models()

    # Old entries are cleared, only API results remain
    assert "custom/model" not in transport.models
    assert "deepseek-ai/DeepSeek-V3-0324" in transport.models


async def test_fetch_models_empty(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {"object": "list", "data": []}

    await transport.fetch_models()

    assert len(transport.models) == 0


async def test_fetch_models_error(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    server, _ = fake_server
    server.models_status = 401
    server.error_body = "Unauthorized"

    with pytest.raises(StreamError, match="401"):
        await transport.fetch_models()


async def test_fetch_models_capability_aliases(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    """'tools' from the API maps to Capability.tool_use, unknown caps are dropped."""
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {
                "id": "test/model",
                "object": "model",
                "supported_features": ["tools", "json_mode", "structured_outputs", "unknown_cap"],
            },
        ],
    }

    await transport.fetch_models()

    spec = transport.models["test/model"]
    assert spec.capabilities == frozenset({"tool_use", "json_mode", "structured_outputs"})


async def test_fetch_models_vision_from_modality(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    """Vision capability is inferred from architecture.modality containing 'image' on input side."""
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen2.5-VL-72B-Instruct",
                "object": "model",
                "architecture": {"modality": "text+image->text", "tokenizer": "Other"},
                "supported_features": ["tools"],
            },
            {
                "id": "meta-llama/Llama-3.3-70B-Instruct",
                "object": "model",
                "architecture": {"modality": "text->text", "tokenizer": "Other"},
                "supported_features": ["tools"],
            },
        ],
    }

    await transport.fetch_models()

    vl = transport.models["Qwen/Qwen2.5-VL-72B-Instruct"]
    assert "vision" in vl.capabilities
    assert "tool_use" in vl.capabilities

    llama = transport.models["meta-llama/Llama-3.3-70B-Instruct"]
    assert "vision" not in llama.capabilities
    assert "tool_use" in llama.capabilities


async def test_text_to_image_no_vision(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    """text->image models (e.g. Flux) should NOT get vision capability."""
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {
                "id": "black-forest-labs/FLUX.1-schnell",
                "object": "model",
                "architecture": {"modality": "text->image", "tokenizer": "Other"},
            },
        ],
    }

    await transport.fetch_models()

    flux = transport.models["black-forest-labs/FLUX.1-schnell"]
    assert "vision" not in flux.capabilities


# ---------------------------------------------------------------------------
# Streaming (inherited from OpenAITransport)
# ---------------------------------------------------------------------------


async def test_text_streaming(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    server, _ = fake_server
    server.sse_responses.append(_text_chunks("Hi"))

    events = await _collect(transport.stream([], [], ""))

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in text_deltas) == "Hi"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == StopReason.end_turn
    assert ends[0].usage == Usage(10, 5)


# ---------------------------------------------------------------------------
# build_payload: max_tokens behaviour
# ---------------------------------------------------------------------------


def test_default_max_tokens() -> None:
    t = NebiusTransport()
    payload = t.build_payload([], [], "You are helpful.")
    assert payload["max_tokens"] == t.model.max_output_tokens
    assert payload["model"] == "deepseek-ai/DeepSeek-V3-0324"


def test_max_tokens_when_spec_provided() -> None:
    spec = ModelSpec(id="deepseek-ai/DeepSeek-V3-0324", context_window=128_000, max_output_tokens=8_192)
    t = NebiusTransport(
        model=spec,
        models=ModelRegistry([spec]),
    )
    payload = t.build_payload([], [], "")
    assert payload["max_tokens"] == 8_192


# ---------------------------------------------------------------------------
# Embedding capability detection
# ---------------------------------------------------------------------------


async def test_embedding_from_output_modality(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    """Embedding capability detected from 'text->embedding' output modality."""
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {
                "id": "BAAI/bge-en-icl",
                "object": "model",
                "architecture": {"modality": "text->embedding", "tokenizer": "Other"},
            },
        ],
    }

    await transport.fetch_models()

    spec = transport.models["BAAI/bge-en-icl"]
    assert Capability.embedding in spec.capabilities
    assert Capability.vision not in spec.capabilities


async def test_embedding_from_id_heuristic(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    """Embedding capability detected from known model ID prefixes."""
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {"id": "BAAI/bge-large-en", "object": "model"},
            {"id": "intfloat/e5-large", "object": "model"},
            {"id": "intfloat/multilingual-e5-large", "object": "model"},
            {"id": "some/Embedding-Model", "object": "model"},
            {"id": "meta-llama/Llama-3.3-70B-Instruct", "object": "model"},
        ],
    }

    await transport.fetch_models()

    assert Capability.embedding in transport.models["BAAI/bge-large-en"].capabilities
    assert Capability.embedding in transport.models["intfloat/e5-large"].capabilities
    assert Capability.embedding in transport.models["intfloat/multilingual-e5-large"].capabilities
    assert Capability.embedding in transport.models["some/Embedding-Model"].capabilities
    assert Capability.embedding not in transport.models["meta-llama/Llama-3.3-70B-Instruct"].capabilities


async def test_embedding_not_from_input_modality(
    fake_server: tuple[FakeNebiusServer, str],
    transport: NebiusTransport,
) -> None:
    """Input modality 'text->text' should NOT get embedding capability."""
    server, _ = fake_server
    server.models_response = {
        "object": "list",
        "data": [
            {
                "id": "some/chat-model",
                "object": "model",
                "architecture": {"modality": "text->text", "tokenizer": "Other"},
            },
        ],
    }

    await transport.fetch_models()

    assert Capability.embedding not in transport.models["some/chat-model"].capabilities
