"""Tests for axio.testing: StubTransport and response builders."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from axio.events import IterationEnd, TextDelta, ToolInputDelta, ToolUseStart
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.testing import (
    StubTransport,
    assert_static_model_transport_contract,
    make_echo_tool,
    make_ephemeral_context,
    make_stub_transport,
    make_text_response,
    make_tool_use_response,
)
from axio.types import StopReason, Usage


@dataclass(slots=True)
class _ModelTransport:
    model: ModelSpec
    models: ModelRegistry


def _model_transport(spec: ModelSpec) -> _ModelTransport:
    return _ModelTransport(model=spec, models=ModelRegistry([spec]))


class TestStaticModelTransportContract:
    def test_accepts_a_well_formed_catalog(self) -> None:
        spec = ModelSpec(
            id="model",
            capabilities=frozenset({Capability.text, Capability.tool_use}),
            context_window=16_384,
            max_output_tokens=4_096,
            input_cost=1.0,
            output_cost=2.0,
        )

        assert_static_model_transport_contract(_model_transport(spec))

    def test_rejects_an_empty_catalog(self) -> None:
        spec = ModelSpec(id="model", capabilities=frozenset({Capability.text}))

        with pytest.raises(AssertionError, match="must not be empty"):
            assert_static_model_transport_contract(_ModelTransport(model=spec, models=ModelRegistry()))

    def test_rejects_a_selected_model_outside_the_catalog(self) -> None:
        selected = ModelSpec(id="selected", capabilities=frozenset({Capability.text}))
        registered = ModelSpec(id="registered", capabilities=frozenset({Capability.text}))

        with pytest.raises(AssertionError, match="absent from its registry"):
            assert_static_model_transport_contract(_ModelTransport(model=selected, models=ModelRegistry([registered])))

    def test_rejects_stale_selected_model_metadata(self) -> None:
        selected = ModelSpec(id="model", capabilities=frozenset({Capability.text}), context_window=8_192)
        registered = ModelSpec(id="model", capabilities=frozenset({Capability.text}), context_window=16_384)

        with pytest.raises(AssertionError, match="stale registry metadata"):
            assert_static_model_transport_contract(_ModelTransport(model=selected, models=ModelRegistry([registered])))

    @pytest.mark.parametrize(
        "spec",
        [
            ModelSpec(id="", capabilities=frozenset({Capability.text})),
            ModelSpec(id="model", capabilities=frozenset()),
            ModelSpec(id="model", capabilities=frozenset({Capability.vision})),
            ModelSpec(id="model", capabilities=frozenset({Capability.text}), context_window=0),
            ModelSpec(
                id="model",
                capabilities=frozenset({Capability.text}),
                context_window=1_000,
                max_output_tokens=0,
            ),
            ModelSpec(id="model", capabilities=frozenset({Capability.text}), input_cost=-1.0),
            ModelSpec(id="model", capabilities=frozenset({Capability.text}), output_cost=float("inf")),
        ],
    )
    def test_rejects_malformed_model_metadata(self, spec: ModelSpec) -> None:
        with pytest.raises(AssertionError):
            assert_static_model_transport_contract(_model_transport(spec))


class TestMakeTextResponse:
    def test_default_text(self) -> None:
        events = make_text_response()
        assert any(isinstance(e, TextDelta) and e.delta == "Done" for e in events)

    def test_custom_text(self) -> None:
        events = make_text_response("hello")
        assert any(isinstance(e, TextDelta) and e.delta == "hello" for e in events)

    def test_ends_with_iteration_end(self) -> None:
        events = make_text_response()
        assert isinstance(events[-1], IterationEnd)
        assert events[-1].stop_reason == StopReason.end_turn

    def test_custom_iteration(self) -> None:
        events = make_text_response(iteration=5)
        end = events[-1]
        assert isinstance(end, IterationEnd)
        assert end.iteration == 5

    def test_custom_usage(self) -> None:
        u = Usage(1, 2)
        events = make_text_response(usage=u)
        end = events[-1]
        assert isinstance(end, IterationEnd)
        assert end.usage == u


class TestMakeToolUseResponse:
    def test_default_tool_name(self) -> None:
        events = make_tool_use_response()
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert len(starts) == 1
        assert starts[0].name == "echo"

    def test_custom_tool_name(self) -> None:
        events = make_tool_use_response("my_tool")
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert starts[0].name == "my_tool"

    def test_tool_input_delta_contains_json(self) -> None:
        events = make_tool_use_response(tool_input={"x": 1})
        deltas = [e for e in events if isinstance(e, ToolInputDelta)]
        assert len(deltas) == 1
        assert json.loads(deltas[0].partial_json) == {"x": 1}

    def test_ends_with_tool_use_stop_reason(self) -> None:
        events = make_tool_use_response()
        assert isinstance(events[-1], IterationEnd)
        assert events[-1].stop_reason == StopReason.tool_use

    def test_custom_tool_id(self) -> None:
        events = make_tool_use_response(tool_id="call_abc")
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert starts[0].tool_use_id == "call_abc"


class TestStubTransport:
    async def test_yields_configured_events(self) -> None:
        events = make_text_response("hi")
        transport = StubTransport([events])
        received = [e async for e in transport.stream([], [], "")]
        assert received == events

    async def test_pops_next_sequence_per_call(self) -> None:
        r1 = make_text_response("first")
        r2 = make_text_response("second")
        transport = StubTransport([r1, r2])

        first = [e async for e in transport.stream([], [], "")]
        second = [e async for e in transport.stream([], [], "")]

        assert any(isinstance(e, TextDelta) and e.delta == "first" for e in first)
        assert any(isinstance(e, TextDelta) and e.delta == "second" for e in second)

    async def test_repeats_last_sequence_when_exhausted(self) -> None:
        events = make_text_response("only")
        transport = StubTransport([events])

        first = [e async for e in transport.stream([], [], "")]
        second = [e async for e in transport.stream([], [], "")]

        assert first == second

    def test_call_count_increments(self) -> None:
        transport = StubTransport([make_text_response()])
        assert transport._call_count == 0
        _ = transport.stream([], [], "")
        assert transport._call_count == 1

    async def test_empty_responses_list(self) -> None:
        transport = StubTransport([])
        with pytest.raises((IndexError, Exception)):
            _ = [e async for e in transport.stream([], [], "")]


class TestMakeStubTransport:
    async def test_yields_hello_world(self) -> None:
        transport = make_stub_transport()
        events = [e async for e in transport.stream([], [], "")]
        text = "".join(e.delta for e in events if isinstance(e, TextDelta))
        assert text == "Hello world"

    async def test_repeats_on_second_call(self) -> None:
        transport = make_stub_transport()
        first = [e async for e in transport.stream([], [], "")]
        second = [e async for e in transport.stream([], [], "")]
        assert first == second


class TestMakeEphemeralContext:
    async def test_returns_empty_context(self) -> None:
        ctx = make_ephemeral_context()
        assert await ctx.get_history() == []

    def test_each_call_returns_new_instance(self) -> None:
        a = make_ephemeral_context()
        b = make_ephemeral_context()
        assert a is not b


class TestMakeEchoTool:
    def test_name(self) -> None:
        tool = make_echo_tool()
        assert tool.name == "echo"

    async def test_returns_json_with_msg(self) -> None:
        tool = make_echo_tool()
        result = await tool(msg="hello")
        assert json.loads(result) == {"msg": "hello"}
