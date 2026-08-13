"""Tests for REPL model selection helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from axio.models import Capability, ModelRegistry, ModelSpec

from axio_repl import (
    _adopt_catalogue_metadata,
    _apply_iterations,
    _apply_model,
    _choose_model,
    _resolve_model_arg,
    _select_transport,
)

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


def test_naming_a_transport_without_its_key_says_which_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Auto-detection asks this question for you; naming a transport used to skip
    # it, and the answer arrived as a traceback from the first API call.
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        _select_transport("nebius")

    assert exc_info.value.code == 1
    assert "NEBIUS_API_KEY" in capsys.readouterr().err


def test_naming_a_transport_with_its_key_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    cls, _ = _select_transport("nebius")

    assert cls is not None


def test_zero_iterations_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    # It reads as "no limit" and does the opposite: an agent that never calls
    # the model and then reports that it ran out of iterations.
    agent = cast(Any, SimpleNamespace(max_iterations=50))

    _apply_iterations(agent, "0")

    assert agent.max_iterations == 50
    assert "at least 1" in capsys.readouterr().out


def test_a_real_iteration_count_is_applied() -> None:
    agent = cast(Any, SimpleNamespace(max_iterations=50))

    _apply_iterations(agent, "200")

    assert agent.max_iterations == 200


def test_a_fragment_of_an_id_names_the_model(capsys: pytest.CaptureFixture[str]) -> None:
    # --model used to demand the full id, capitals and vendor prefix included.
    minimax = ModelSpec(id="MiniMaxAI/MiniMax-M3")
    transport = cast(Any, SimpleNamespace(models=ModelRegistry([minimax, ModelSpec(id="zai-org/GLM-5.2")])))

    assert _choose_model(transport, "minimax-m3") is minimax
    assert _choose_model(transport, "glm-5.2") is not None


def test_an_ambiguous_fragment_names_the_candidates(capsys: pytest.CaptureFixture[str]) -> None:
    transport = cast(
        Any,
        SimpleNamespace(
            models=ModelRegistry([ModelSpec(id="MiniMaxAI/MiniMax-M3"), ModelSpec(id="MiniMaxAI/MiniMax-M2.5")])
        ),
    )

    assert _choose_model(transport, "minimax") is None
    assert "Ambiguous" in capsys.readouterr().out
