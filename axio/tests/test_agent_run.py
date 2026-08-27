"""Tests for Agent.run_stream() and run(): core loop, stop reasons, usage."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from axio.agent import Agent
from axio.blocks import ReasoningBlock, TextBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import (
    Error,
    IterationEnd,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.messages import Message
from axio.testing import StubTransport, make_echo_tool, make_text_response, make_tool_use_response
from axio.tool import Tool
from axio.types import StopReason, Usage


class CapturingTransport:
    """Records messages passed to each stream() call."""

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self._responses = responses
        self._call_count = 0
        self.calls: list[list[Message]] = []

    async def _generate(self, events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._generate(self._responses[idx])


async def _ok(msg: str) -> str:
    return "ok"


class TestRunStream:
    async def test_end_turn_yields_text_and_session_end(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "Hello"),
                    TextDelta(0, " world"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", MemoryContextStore()):
            events.append(e)

        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 2
        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.end_turn

    async def test_session_end_total_usage(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        end = await agent.run_stream("hi", MemoryContextStore()).get_session_end()
        assert end.total_usage == Usage(10, 5)


class TestRun:
    async def test_returns_concatenated_text(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "Hello"),
                    TextDelta(0, " world"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        result = await agent.run("hi", MemoryContextStore())
        assert result == "Hello world"


class TestMultiIteration:
    async def test_tool_use_then_end_turn(self) -> None:
        tool = make_echo_tool()
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("do it", MemoryContextStore()):
            events.append(e)

        iteration_ends = [e for e in events if isinstance(e, IterationEnd)]
        assert len(iteration_ends) == 2
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.end_turn

    async def test_total_usage_across_iterations(self) -> None:
        tool = make_echo_tool()
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1, Usage(10, 5)),
                make_text_response("Done", 2, Usage(3, 7)),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        end = await agent.run_stream("go", MemoryContextStore()).get_session_end()
        assert end.total_usage == Usage(13, 12)


class TestContextTokenTracking:
    async def test_agent_updates_context_tokens(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        await agent.run("go", context)
        assert await context.get_context_tokens() == (10, 5)

    async def test_agent_accumulates_context_tokens_across_iterations(self) -> None:
        tool = make_echo_tool()
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1, Usage(10, 5)),
                make_text_response("Done", 2, Usage(3, 7)),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        context = MemoryContextStore()
        await agent.run("go", context)
        assert await context.get_context_tokens() == (13, 12)


class TestReasoningPassthrough:
    async def test_reasoning_delta_yielded_but_not_stored(self) -> None:
        """ReasoningDelta events pass through the stream but are NOT stored in context."""
        transport = StubTransport(
            [
                [
                    ReasoningDelta(0, "thinking..."),
                    TextDelta(0, "answer"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", context):
            events.append(e)

        # ReasoningDelta is yielded
        reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
        assert len(reasoning) == 1
        assert reasoning[0].delta == "thinking..."

        # TextDelta is yielded
        text = [e for e in events if isinstance(e, TextDelta)]
        assert len(text) == 1
        assert text[0].delta == "answer"

        # Only text is stored in assistant message, not reasoning
        history = await context.get_history()
        assistant_msgs = [m for m in history if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        from axio.blocks import TextBlock

        text_blocks = [b for b in assistant_msgs[0].content if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "answer"


class TestMaxIterations:
    async def test_max_iterations_reached(self) -> None:
        """C7: max_iterations emits SessionEndEvent(stop_reason=error)."""
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_ok)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_tool_use_response("echo", "c2", {"msg": "hi"}, 2),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport, max_iterations=1)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.error


class TestLastIterationMessage:
    async def test_injected_only_on_last_iteration(self) -> None:
        """last_iteration_message is appended to history only on the final iteration."""
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_ok)
        hint = Message(role="system", content=[TextBlock(text="wrap up now")])
        transport = CapturingTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(
            system="test",
            tools=[tool],
            transport=transport,
            max_iterations=2,
            last_iteration_message=hint,
        )
        await agent.run("go", MemoryContextStore())

        # iteration 1: hint NOT in history
        assert hint not in transport.calls[0]
        # iteration 2 (last): hint IS the final message
        assert transport.calls[1][-1] is hint

    async def test_not_injected_when_none(self) -> None:
        """No injection when last_iteration_message is None (default)."""
        transport = CapturingTransport([make_text_response("hi", 1)])
        agent = Agent(system="test", tools=[], transport=transport)
        await agent.run("go", MemoryContextStore())

        history = transport.calls[0]
        assert all(m.role != "system" for m in history)

    async def test_not_stored_in_context(self) -> None:
        """last_iteration_message is injected into the stream but not persisted."""
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_ok)
        hint = Message(role="system", content=[TextBlock(text="wrap up")])
        transport = CapturingTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(
            system="test",
            tools=[tool],
            transport=transport,
            max_iterations=2,
            last_iteration_message=hint,
        )
        context = MemoryContextStore()
        await agent.run("go", context)

        history = await context.get_history()
        assert hint not in history


class TestStopReasonGuard:
    """A stop reason the agent does not name must end the run, not start another one."""

    async def test_an_unnamed_stop_reason_ends_the_run(self) -> None:
        # Without the wildcard this fell out of the match and the loop ran again, re-prompting the
        # model with unchanged history until max_iterations — every one of those turns paid for.
        transport = StubTransport([[TextDelta(0, "no"), IterationEnd(1, StopReason.cancelled, Usage(10, 5))]])
        agent = Agent(system="", tools=[], transport=transport, max_iterations=5)

        events = [event async for event in agent.run_stream("hi", MemoryContextStore())]

        assert transport._call_count == 1, "the agent asked again after a reason it does not act on"
        assert any(isinstance(e, Error) for e in events)
        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert [e.stop_reason for e in ends] == [StopReason.error]

    async def test_a_refusal_ends_the_session_as_a_refusal_and_not_as_an_error(self) -> None:
        # The model declined. Reported as an error the caller cannot tell a decline from a broken
        # connection, and retries something that can never work.
        transport = StubTransport(
            [
                [
                    Refusal(index=0, text="I cannot help with that"),
                    IterationEnd(1, StopReason.refusal, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="", tools=[], transport=transport, max_iterations=5)

        events = [event async for event in agent.run_stream("hi", MemoryContextStore())]

        assert not [e for e in events if isinstance(e, Error)], "a decline was reported as a failure"
        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert [e.stop_reason for e in ends] == [StopReason.refusal]
        assert transport._call_count == 1

    async def test_a_paused_turn_is_resumed_rather_than_ended(self) -> None:
        # The one reason that does not end the run: the provider stopped its own tool loop and
        # expects the assistant content back so it can finish.
        transport = StubTransport(
            [
                [TextDelta(0, "half"), IterationEnd(1, StopReason.pause_turn, Usage(10, 5))],
                [TextDelta(0, " and half"), IterationEnd(2, StopReason.end_turn, Usage(3, 2))],
            ]
        )
        agent = Agent(system="", tools=[], transport=transport, max_iterations=5)

        events = [event async for event in agent.run_stream("hi", MemoryContextStore())]

        assert transport._call_count == 2, "the paused turn was never resumed"
        assert not [e for e in events if isinstance(e, Error)]
        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert [e.stop_reason for e in ends] == [StopReason.end_turn]


class TestReasoningIsKept:
    """Reasoning has to survive into the stored turn, or the turn cannot be replayed."""

    async def test_reasoning_and_its_signature_reach_the_stored_message(self) -> None:
        transport = StubTransport(
            [
                [
                    ReasoningDelta(0, "weighing "),
                    ReasoningDelta(0, "the options"),
                    ReasoningSignature(0, "ErUBCkYIBRgC"),
                    TextDelta(0, "the answer"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="", tools=[], transport=transport)

        async for _ in agent.run_stream("hi", context):
            pass

        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        assert assistant.content[0] == ReasoningBlock(text="weighing the options", signature="ErUBCkYIBRgC")
        assert assistant.content[1] == TextBlock(text="the answer")

    async def test_a_delta_after_a_signature_starts_a_new_block(self) -> None:
        # The provider signed the text it had. Extending that block would leave a stored signature
        # that disagrees with the text beside it, and the replay is refused.
        transport = StubTransport(
            [
                [
                    ReasoningDelta(0, "first"),
                    ReasoningSignature(0, "sig-one"),
                    ReasoningDelta(0, "second"),
                    ReasoningSignature(0, "sig-two"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="", tools=[], transport=transport)

        async for _ in agent.run_stream("hi", context):
            pass

        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        assert assistant.content == [
            ReasoningBlock(text="first", signature="sig-one"),
            ReasoningBlock(text="second", signature="sig-two"),
        ]


class TestRefusalIsKept:
    async def test_a_refusal_reaches_the_stored_turn(self) -> None:
        # A refusal is what the assistant said. Left out, the stored turn is empty and the next
        # request carries a blank assistant message the provider then rejects.
        transport = StubTransport(
            [
                [
                    Refusal(index=0, text="I cannot help with that", category="cyber"),
                    IterationEnd(1, StopReason.refusal, Usage(10, 5)),
                ]
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="", tools=[], transport=transport)

        events = [event async for event in agent.run_stream("hi", context)]

        assert any(isinstance(e, Refusal) for e in events), "the refusal never reached the caller"
        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        assert assistant.content == [TextBlock(text="I cannot help with that")]
        assert transport._call_count == 1, "a refusal was retried, which cannot succeed"


class TestSignaturesAreNotConcatenated:
    """Two signatures in a row are two proofs, not one proof in two pieces.

    Anthropic documents one per block — "The thinking block opens, receives a single
    signature_delta, and closes" — and Gemini puts one on each parallel function-call part. So a
    reader that appended a second signature to the first would corrupt both of Gemini's, which is
    the case these two tests hold the line on.
    """

    async def test_two_reasoning_blocks_each_keep_their_own_proof(self) -> None:
        transport = StubTransport(
            [
                [
                    ReasoningDelta(0, "first"),
                    ReasoningSignature(0, "SIG-1"),
                    ReasoningDelta(1, "second"),
                    ReasoningSignature(1, "SIG-2"),
                    IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
                ]
            ]
        )
        context = MemoryContextStore()
        async for _ in Agent(system="", tools=[], transport=transport).run_stream("hi", context):
            pass

        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        assert assistant.content == [
            ReasoningBlock(text="first", signature="SIG-1"),
            ReasoningBlock(text="second", signature="SIG-2"),
        ]

    async def test_two_signatures_for_different_blocks_stay_apart(self) -> None:
        # What Gemini sends for two signed parallel function calls: one proof per part, and the
        # index says which part. Merged, the first call would replay with both proofs joined and
        # the second with none.
        transport = StubTransport(
            [
                [
                    ReasoningSignature(0, "SIG-A"),
                    ReasoningSignature(1, "SIG-B"),
                    IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
                ]
            ]
        )
        context = MemoryContextStore()
        async for _ in Agent(system="", tools=[], transport=transport).run_stream("hi", context):
            pass

        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        assert [b.signature for b in assistant.content if isinstance(b, ReasoningBlock)] == ["SIG-A", "SIG-B"]


class TestReasoningIdIsKept:
    async def test_the_id_reaches_the_stored_block(self) -> None:
        # Carried by the event and dropped by the agent, the proof would be stored without the name
        # the provider requires beside it, and the replay refused.
        transport = StubTransport(
            [
                [
                    ReasoningSignature(0, "gAAAAAB...", id="rs_1"),
                    IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
                ]
            ]
        )
        context = MemoryContextStore()
        async for _ in Agent(system="", tools=[], transport=transport).run_stream("hi", context):
            pass

        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        assert assistant.content == [ReasoningBlock(signature="gAAAAAB...", id="rs_1")]


class TestFragmentedSignatures:
    """A proof split across events is one proof; two proofs are two. The index says which."""

    @staticmethod
    async def _stored(*events: object) -> list[object]:
        transport = StubTransport([[*events, IterationEnd(1, StopReason.end_turn, Usage(1, 1))]])  # type: ignore[list-item]
        context = MemoryContextStore()
        async for _ in Agent(system="", tools=[], transport=transport).run_stream("hi", context):
            pass
        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        return list(assistant.content)

    async def test_a_proof_split_across_events_is_stored_whole(self) -> None:
        # Stored apart, the block replays with half a signature and the turn after it is refused.
        stored = await self._stored(
            ReasoningDelta(0, "thinking"),
            ReasoningSignature(0, "EqQBCg"),
            ReasoningSignature(0, "IYAhIM"),
        )
        assert stored == [ReasoningBlock(text="thinking", signature="EqQBCgIYAhIM")]

    async def test_two_thinking_blocks_keep_their_own_proofs(self) -> None:
        stored = await self._stored(
            ReasoningDelta(0, "first"),
            ReasoningSignature(0, "SIG-1"),
            ReasoningDelta(1, "second"),
            ReasoningSignature(1, "SIG-2"),
        )
        assert stored == [
            ReasoningBlock(text="first", signature="SIG-1"),
            ReasoningBlock(text="second", signature="SIG-2"),
        ]

    async def test_two_reasoning_items_keep_their_own_proofs(self) -> None:
        # What the Responses API sends for two reasoning items: one proof each, indexed by output.
        stored = await self._stored(
            ReasoningSignature(0, "gAAAAAB-one", id="rs_1"),
            ReasoningSignature(1, "gAAAAAB-two", id="rs_2"),
        )
        assert stored == [
            ReasoningBlock(signature="gAAAAAB-one", id="rs_1"),
            ReasoningBlock(signature="gAAAAAB-two", id="rs_2"),
        ]


class TestToolsAreNotRunForAFailedTurn:
    """A call streamed inside a failed turn is output nobody vouched for."""

    @staticmethod
    def _turn(stop: StopReason) -> StubTransport:
        # A second response ends the run, so the tool is offered once rather than every iteration
        # until max_iterations.
        return StubTransport(
            [
                [
                    ToolUseStart(index=0, tool_use_id="call_1", name="echo"),
                    ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"msg":"x"}'),
                    IterationEnd(1, stop, Usage(1, 1)),
                ],
                [TextDelta(0, "done"), IterationEnd(2, StopReason.end_turn, Usage(1, 1))],
            ]
        )

    async def test_a_failed_turn_does_not_run_its_calls(self) -> None:
        # The run ends immediately afterwards regardless, so running them buys nothing and acts on
        # output the transport has just reported as failed.
        agent = Agent(system="", tools=[make_echo_tool()], transport=self._turn(StopReason.error))
        events = [e async for e in agent.run_stream("hi", MemoryContextStore())]
        assert not [e for e in events if isinstance(e, ToolResult)]

    async def test_a_declined_turn_does_not_run_its_calls(self) -> None:
        agent = Agent(system="", tools=[make_echo_tool()], transport=self._turn(StopReason.refusal))
        events = [e async for e in agent.run_stream("hi", MemoryContextStore())]
        assert not [e for e in events if isinstance(e, ToolResult)]

    async def test_a_turn_that_asked_for_them_still_runs_them(self) -> None:
        agent = Agent(system="", tools=[make_echo_tool()], transport=self._turn(StopReason.tool_use))
        events = [e async for e in agent.run_stream("hi", MemoryContextStore())]
        assert [e.tool_use_id for e in events if isinstance(e, ToolResult)] == ["call_1"]

    async def test_a_finished_turn_still_runs_a_call_it_left_behind(self) -> None:
        # Deliberate and long-standing: a provider that reports end_turn while streaming a call is
        # a mismatch the agent works around rather than a failure it must not act on.
        agent = Agent(system="", tools=[make_echo_tool()], transport=self._turn(StopReason.end_turn))
        events = [e async for e in agent.run_stream("hi", MemoryContextStore())]
        assert [e.tool_use_id for e in events if isinstance(e, ToolResult)] == ["call_1"]


class TestAnUnrunCallIsNotPersisted:
    async def test_a_failed_turn_leaves_no_orphan_call_in_the_history(self) -> None:
        """Cleared from the dispatch list only, the call was still stored with no result beside it.

        The next request then carried a call nothing answered, which Anthropic and Google refuse.
        """
        transport = StubTransport(
            [
                [
                    ToolUseStart(index=0, tool_use_id="call_1", name="echo"),
                    ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"msg":"x"}'),
                    IterationEnd(1, StopReason.error, Usage(1, 1)),
                ]
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="", tools=[make_echo_tool()], transport=transport)

        async for _ in agent.run_stream("hi", context):
            pass

        stored = [b for m in await context.get_history() for b in m.content]
        assert not [b for b in stored if isinstance(b, ToolUseBlock)]

    async def test_a_turn_that_asked_for_a_tool_still_stores_the_call(self) -> None:
        transport = StubTransport(
            [
                [
                    ToolUseStart(index=0, tool_use_id="call_1", name="echo"),
                    ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"msg":"x"}'),
                    IterationEnd(1, StopReason.tool_use, Usage(1, 1)),
                ],
                [TextDelta(0, "done"), IterationEnd(2, StopReason.end_turn, Usage(1, 1))],
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="", tools=[make_echo_tool()], transport=transport)

        async for _ in agent.run_stream("hi", context):
            pass

        stored = [b for m in await context.get_history() for b in m.content]
        assert [b.id for b in stored if isinstance(b, ToolUseBlock)] == ["call_1"]
