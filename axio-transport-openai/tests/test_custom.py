"""Tests for OpenAICompatibleTransport (custom OpenAI-compatible providers)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio.events import IterationEnd, StreamEvent, TextDelta
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.types import StopReason, Usage

from axio_transport_openai.custom import OpenAICompatibleTransport
from axio_transport_openai.tui.custom import CustomHubScreen

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(model_id: str = "mymodel", **kwargs: Any) -> ModelSpec:
    defaults: dict[str, Any] = {
        "context_window": 32_000,
        "max_output_tokens": 4_096,
        "capabilities": frozenset({Capability.text, Capability.tool_use}),
    }
    defaults.update(kwargs)
    return ModelSpec(id=model_id, **defaults)


def _make_provider(
    name: str = "testprovider",
    base_url: str = "http://localhost:9000/v1",
    api_key: str = "sk-test",
    models: list[ModelSpec] | None = None,
) -> OpenAICompatibleTransport:
    return OpenAICompatibleTransport(
        name=name,
        base_url=base_url,
        api_key=api_key,
        models=ModelRegistry(models or [_make_spec()]),
    )


def _sse_chunk(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _text_chunks(text: str) -> str:
    lines = ""
    for ch in text:
        lines += _sse_chunk({"choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]})
    lines += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    lines += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
    lines += _sse_done()
    return lines


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def test_save_and_load_config(tmp_path: Path) -> None:
    providers = [
        _make_provider("p1", "http://a/v1", "key1"),
        _make_provider("p2", "http://b/v1", ""),
    ]
    cfg_path = tmp_path / "openai-custom.json"
    with patch.object(CustomHubScreen, "CONFIG_PATH", cfg_path):
        CustomHubScreen.save_config(providers)
        loaded = CustomHubScreen.load_config()

    assert len(loaded) == 2
    assert loaded[0].name == "p1"
    assert loaded[0].api_key == "key1"
    assert loaded[1].name == "p2"
    assert loaded[1].api_key == ""


def test_load_config_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.json"
    with patch.object(CustomHubScreen, "CONFIG_PATH", missing):
        result = CustomHubScreen.load_config()
    assert result == []


def test_load_config_corrupt_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json", "utf-8")
    with patch.object(CustomHubScreen, "CONFIG_PATH", bad):
        result = CustomHubScreen.load_config()
    assert result == []


def test_save_config_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "openai-custom.json"
    with patch.object(CustomHubScreen, "CONFIG_PATH", nested):
        CustomHubScreen.save_config([_make_provider()])
    assert nested.exists()


def test_roundtrip_to_dict_from_dict() -> None:
    t = OpenAICompatibleTransport(
        name="local",
        base_url="http://x/v1",
        api_key="k",
        models=ModelRegistry(
            [
                ModelSpec(
                    id="llama3",
                    context_window=131072,
                    max_output_tokens=4096,
                    capabilities=frozenset({Capability.text, Capability.vision, Capability.tool_use}),
                    input_cost=1.5,
                    output_cost=3.0,
                )
            ]
        ),
    )
    d = t.to_dict()
    sentinel = MagicMock(spec=aiohttp.ClientSession)
    t2 = OpenAICompatibleTransport.from_dict(d, session=sentinel)
    assert t2.name == "local"
    assert t2.base_url == "http://x/v1"
    assert t2.api_key == "k"
    assert t2.session is sentinel
    assert len(t2.models) == 1
    m = t2.models["llama3"]
    assert m.id == "llama3"
    assert m.context_window == 131072
    assert Capability.vision in m.capabilities
    assert m.input_cost == pytest.approx(1.5)
    assert m.output_cost == pytest.approx(3.0)

    t3 = OpenAICompatibleTransport.from_dict(d)
    assert t3.session is None


def test_roundtrip_unknown_capability_dropped(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    raw = [
        {
            "name": "p",
            "base_url": "http://x/v1",
            "models": [{"id": "m", "capabilities": ["text", "not_a_cap"]}],
        }
    ]
    cfg_path.write_text(json.dumps(raw), "utf-8")
    with patch.object(CustomHubScreen, "CONFIG_PATH", cfg_path):
        loaded = CustomHubScreen.load_config()
    assert Capability.text in loaded[0].models["m"].capabilities
    assert len(loaded[0].models["m"].capabilities) == 1


# ---------------------------------------------------------------------------
# fetch_models - must be a no-op (does NOT reset models to OPENAI_MODELS)
# ---------------------------------------------------------------------------


async def test_fetch_models_noop_preserves_models() -> None:
    reg = ModelRegistry([_make_spec()])
    t = OpenAICompatibleTransport(base_url="http://x/v1", api_key="k", models=reg)
    assert "mymodel" in t.models
    await t.fetch_models()
    assert "mymodel" in t.models  # unchanged


async def test_fetch_models_noop_on_empty() -> None:
    t = OpenAICompatibleTransport()
    await t.fetch_models()
    assert len(t.models) == 0  # stays empty, NOT populated with OPENAI_MODELS


# ---------------------------------------------------------------------------
# Streaming - direct (no routing; transport IS the provider)
# ---------------------------------------------------------------------------


class FakeServer:
    def __init__(self) -> None:
        self.sse_responses: list[str] = []
        self.received_payloads: list[dict[str, Any]] = []

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        return app

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.received_payloads.append(await request.json())
        sse_body = self.sse_responses.pop(0) if self.sse_responses else _sse_done()
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(sse_body.encode())
        await resp.write_eof()
        return resp


@pytest.fixture
async def fake_server() -> AsyncIterator[tuple[FakeServer, str]]:
    server = FakeServer()
    runner = web.AppRunner(server.make_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sock = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = sock.getsockname()[:2]
    yield server, f"http://{host}:{port}/v1"
    await runner.cleanup()


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [e async for e in stream]


async def test_stream_direct(fake_server: tuple[FakeServer, str]) -> None:
    server, base_url = fake_server
    server.sse_responses.append(_text_chunks("hello"))

    reg = ModelRegistry([_make_spec()])
    async with aiohttp.ClientSession() as session:
        t = OpenAICompatibleTransport(base_url=base_url, api_key="k", models=reg, session=session)
        t.model = t.models["mymodel"]
        events = await _collect(t.stream([], [], ""))

    text = "".join(e.delta for e in events if isinstance(e, TextDelta))
    assert text == "hello"
    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == StopReason.end_turn
    assert ends[0].usage == Usage(5, 3)


async def test_stream_sends_bare_model_id(fake_server: tuple[FakeServer, str]) -> None:
    server, base_url = fake_server
    server.sse_responses.append(_text_chunks("x"))

    reg = ModelRegistry([_make_spec()])
    async with aiohttp.ClientSession() as session:
        t = OpenAICompatibleTransport(base_url=base_url, api_key="k", models=reg, session=session)
        t.model = t.models["mymodel"]
        await _collect(t.stream([], [], ""))

    assert server.received_payloads[0]["model"] == "mymodel"


# ---------------------------------------------------------------------------
