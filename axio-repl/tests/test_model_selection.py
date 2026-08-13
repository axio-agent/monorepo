"""Tests for REPL model selection helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from axio.models import Capability, ModelRegistry, ModelSpec

from axio_repl import _adopt_catalogue_metadata, _apply_model, _resolve_model_arg

_BASE_MODEL = ModelSpec(
    id="z-ai/glm-4.7",
    capabilities=frozenset({Capability.text, Capability.tool_use}),
    context_window=256_000,
    max_output_tokens=16_384,
)


class _VariantTransport:
    def __init__(self) -> None:
        self.model = _BASE_MODEL
        self.models = ModelRegistry([_BASE_MODEL])

    def resolve_model(self, model_id: str) -> ModelSpec:
        base_id, sep, _variant = model_id.rpartition(":")
        if sep and base_id in self.models:
            return replace(self.models[base_id], id=model_id)
        return self.models[model_id]


def test_resolve_model_arg_uses_transport_resolver() -> None:
    transport = _VariantTransport()

    spec = _resolve_model_arg(transport, "z-ai/glm-4.7:nitro")

    assert spec.id == "z-ai/glm-4.7:nitro"
    assert spec.context_window == _BASE_MODEL.context_window


def test_apply_model_accepts_resolved_variant(capsys: pytest.CaptureFixture[str]) -> None:
    transport = _VariantTransport()
    agent: Any = SimpleNamespace(system="old")

    _apply_model(transport, agent, [], Path("/tmp/test-workspace"), "", "z-ai/glm-4.7:nitro")

    assert transport.model.id == "z-ai/glm-4.7:nitro"
    assert "z-ai/glm-4.7:nitro" in agent.system
    captured = capsys.readouterr()
    assert "Switched to" in captured.out


def test_default_placeholder_adopts_catalogue_metadata() -> None:
    # What a transport ships as its default: an id, and nothing behind it.
    transport = _VariantTransport()
    transport.model = ModelSpec(id="z-ai/glm-4.7")
    assert Capability.tool_use not in transport.model.capabilities

    _adopt_catalogue_metadata(transport)

    assert Capability.tool_use in transport.model.capabilities
    assert transport.model.context_window == 256_000


def test_a_model_missing_from_the_catalogue_is_left_alone() -> None:
    transport = _VariantTransport()
    transport.model = ModelSpec(id="nobody/knows-this")

    _adopt_catalogue_metadata(transport)

    assert transport.model.id == "nobody/knows-this"
