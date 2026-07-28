"""Tests for Anthropic transport — Vertex AI, endpoint/header building, config."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from axio.blocks import TextBlock
from axio.messages import Message
from axio.models import ModelRegistry

import axio_transport_anthropic
from axio_transport_anthropic import (
    ANTHROPIC_MODELS,
    AnthropicTransport,
    _convert_messages,
)

# ---------------------------------------------------------------------------
# _convert_messages
# ---------------------------------------------------------------------------


def test_convert_messages_basic() -> None:
    messages = [
        Message(role="user", content=[TextBlock(text="Hi")]),
        Message(role="assistant", content=[TextBlock(text="Hello")]),
    ]
    result = _convert_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Transport config
# ---------------------------------------------------------------------------


def test_transport_defaults() -> None:
    t = AnthropicTransport()
    assert t.model.id == "claude-sonnet-4-6"
    assert t.name == "Anthropic"


def test_transport_to_from_dict() -> None:
    t = AnthropicTransport(api_key="sk-test")
    d = t.to_dict()
    assert d["api_key"] == "sk-test"
    t2 = AnthropicTransport.from_dict(d)
    assert t2.api_key == "sk-test"


def test_transport_vertexai_to_from_dict() -> None:
    t = AnthropicTransport(vertexai=True, project="my-project", location="us-east5")
    d = t.to_dict()
    assert d["vertexai"] is True
    assert d["project"] == "my-project"
    assert d["location"] == "us-east5"
    t2 = AnthropicTransport.from_dict(d)
    assert t2.vertexai is True
    assert t2.project == "my-project"


def test_string_settings_coercion() -> None:
    t = AnthropicTransport(
        vertexai="true",  # type: ignore[arg-type]
    )
    assert t.vertexai is True


# ---------------------------------------------------------------------------
# Vertex AI env-var auto-detection requires google-auth
# ---------------------------------------------------------------------------


def test_env_vertexai_without_google_auth_falls_back_to_direct_api(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="axio_transport_anthropic"):
        t = AnthropicTransport(api_key="sk-test")
    assert t.vertexai is False
    assert any("google-auth" in r.message for r in caplog.records)


def test_env_vertexai_with_google_auth_available(monkeypatch: Any) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: True)
    t = AnthropicTransport(api_key="sk-test")
    assert t.vertexai is True


def test_env_vertexai_unset_is_false(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = AnthropicTransport(api_key="sk-test")
    assert t.vertexai is False


def test_explicit_vertexai_true_downgraded_when_google_auth_missing(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="axio_transport_anthropic"):
        t = AnthropicTransport(vertexai=True, project="proj")
    assert t.vertexai is False
    assert any("google-auth" in r.message for r in caplog.records)


def test_models_registry() -> None:
    assert isinstance(ANTHROPIC_MODELS, ModelRegistry)
    assert "claude-sonnet-4-6" in ANTHROPIC_MODELS
    assert "claude-opus-4-6" in ANTHROPIC_MODELS
    assert "claude-haiku-4-5" in ANTHROPIC_MODELS
    assert "claude-haiku-4-5-20251001" in ANTHROPIC_MODELS


# ---------------------------------------------------------------------------
# Endpoint building
# ---------------------------------------------------------------------------


def test_direct_api_endpoint(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = AnthropicTransport(api_key="sk-test")
    assert t._build_url() == "https://api.anthropic.com/v1/messages"


def test_vertex_endpoint_regional() -> None:
    t = AnthropicTransport(vertexai=True, project="proj", location="us-east5")
    endpoint = t._build_url()
    assert "us-east5-aiplatform.googleapis.com" in endpoint
    assert "publishers/anthropic/models/claude-sonnet-4-6" in endpoint
    assert ":streamRawPredict" in endpoint


def test_vertex_endpoint_global() -> None:
    t = AnthropicTransport(vertexai=True, project="proj", location="global")
    endpoint = t._build_url()
    assert "aiplatform.googleapis.com/v1/" in endpoint
    assert "locations/global/" in endpoint
    assert "global-aiplatform" not in endpoint


# ---------------------------------------------------------------------------
# Header building
# ---------------------------------------------------------------------------


def test_direct_api_headers(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = AnthropicTransport(api_key="sk-test")
    headers = t._build_headers()
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_direct_api_body_includes_model(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = AnthropicTransport(api_key="sk-test")
    body = t.build_payload(
        [Message(role="user", content=[TextBlock(text="Hi")])],
        [],
        "system prompt",
    )
    assert body["model"] == "claude-sonnet-4-6"
    assert "anthropic_version" not in body


def test_vertex_body_includes_version() -> None:
    t = AnthropicTransport(vertexai=True, project="proj", location="us-east5")
    body = t.build_payload(
        [Message(role="user", content=[TextBlock(text="Hi")])],
        [],
        "",
    )
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in body
