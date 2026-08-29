"""Tests for Agent.run_stream() and run(): core loop, stop reasons, usage."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from axio.agent import Agent
from axio.blocks import ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import (
    AudioOutput,
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    TextSignature,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.exceptions import StreamError
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
        # A value the enum does not have stands for a reason added to the API later. Without the
        # wildcard it fell out of the match and the loop ran again until max_iterations.
        later = cast(StopReason, "invented_later")
        transport = StubTransport([[TextDelta(0, "no"), IterationEnd(1, later, Usage(10, 5))]])
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


class TestAnswerTextKeepsItsProof:
    """Gemini signs the answer part too, and that proof has to reach the block it signs."""

    @staticmethod
    async def _stored(*events: StreamEvent) -> list[Any]:
        transport = StubTransport([[*events, IterationEnd(1, StopReason.end_turn, Usage(1, 1))]])
        context = MemoryContextStore()
        async for _ in Agent(system="", tools=[], transport=transport).run_stream("hi", context):
            pass
        assistant = [m for m in await context.get_history() if m.role == "assistant"][0]
        return list(assistant.content)

    async def test_a_proof_on_answer_text_reaches_the_stored_text_block(self) -> None:
        # Dropped here, the turn replays without the proof Gemini issued for that part and the
        # next request fails with MISSING_THOUGHT_SIGNATURE.
        stored = await self._stored(TextDelta(0, "42"), TextSignature(0, "SIG"))

        assert stored == [TextBlock(text="42", signature="SIG")]

    async def test_a_delta_after_a_signed_text_block_starts_a_new_block(self) -> None:
        # The provider signed the text it had. Extending that block would store a proof that
        # disagrees with the text beside it.
        stored = await self._stored(
            TextDelta(0, "first"),
            TextSignature(0, "SIG-1"),
            TextDelta(1, "second"),
            TextSignature(1, "SIG-2"),
        )

        assert stored == [
            TextBlock(text="first", signature="SIG-1"),
            TextBlock(text="second", signature="SIG-2"),
        ]

    async def test_a_proof_for_text_never_lands_on_the_call_that_follows(self) -> None:
        # Held as a reasoning proof it made a textless block, and the next unsigned call replayed
        # with a proof that was never its own.
        stored = await self._stored(
            TextDelta(0, "42"),
            TextSignature(0, "SIG"),
            ToolUseStart(1, "c1", "echo"),
            ToolInputDelta(1, "c1", "{}"),
        )

        assert stored == [TextBlock(text="42", signature="SIG"), ToolUseBlock(id="c1", name="echo", input={})]
        assert not [b for b in stored if isinstance(b, ReasoningBlock)]


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
        # index says which part.
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

    @staticmethod
    def _echo() -> tuple[Tool[Any], list[str]]:
        """An echo tool that records every run, so a test can say nothing ran at all."""
        ran: list[str] = []

        async def echo(msg: str) -> str:
            ran.append(msg)
            return msg

        return Tool(name="echo", description="Returns input", handler=echo), ran

    async def test_a_failed_turn_does_not_run_its_calls(self) -> None:
        # The run ends immediately afterwards regardless, so running them buys nothing and acts on
        # output the transport has just reported as failed.
        tool, ran = self._echo()
        agent = Agent(system="", tools=[tool], transport=self._turn(StopReason.refusal))
        events = [e async for e in agent.run_stream("hi", MemoryContextStore())]
        assert ran == []
        # Reported all the same, and as an error: the transport announced the call as started, so
        # a caller left with no result for it shows a call that never resolves.
        assert [(e.tool_use_id, e.is_error) for e in events if isinstance(e, ToolResult)] == [("call_1", True)]

    async def test_a_declined_turn_does_not_run_its_calls(self) -> None:
        tool, ran = self._echo()
        agent = Agent(system="", tools=[tool], transport=self._turn(StopReason.refusal))
        events = [e async for e in agent.run_stream("hi", MemoryContextStore())]
        assert ran == []
        assert all(e.is_error for e in events if isinstance(e, ToolResult))

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


class TestAnUnrunCallIsPersistedWithItsRefusal:
    async def test_a_failed_turn_leaves_no_orphan_call_in_the_history(self) -> None:
        """The call is stored, and a result saying it did not run is stored beside it.

        A call nothing answered is refused by Anthropic and Google on the next request. Dropping
        the call closed that, and opened another: the next request then held no trace of the
        attempt, so the model could not tell it from a turn that called nothing, and made the same
        call again.
        """
        transport = StubTransport(
            [
                [
                    ToolUseStart(index=0, tool_use_id="call_1", name="echo"),
                    ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"msg":"x"}'),
                    IterationEnd(1, StopReason.refusal, Usage(1, 1)),
                ]
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="", tools=[make_echo_tool()], transport=transport)

        async for _ in agent.run_stream("hi", context):
            pass

        stored = [b for m in await context.get_history() for b in m.content]
        calls = [b for b in stored if isinstance(b, ToolUseBlock)]
        results = [b for b in stored if isinstance(b, ToolResultBlock)]
        assert [b.id for b in calls] == ["call_1"]
        assert [b.tool_use_id for b in results] == ["call_1"], "no call may be left without a result"
        assert results[0].is_error and "refusal" in str(results[0].content)

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


async def test_a_tool_turn_with_no_usable_call_stops_instead_of_asking_again() -> None:
    """Going round again would send byte-identical input.

    It did so until max_iterations: fifty paid requests for one malformed call. Google maps
    MALFORMED_FUNCTION_CALL to tool_use, which is exactly how a turn reaches here with no call.
    """
    transport = StubTransport([[IterationEnd(1, StopReason.tool_use, Usage(1, 1))]])
    context = MemoryContextStore()
    agent = Agent(system="", tools=[], transport=transport, max_iterations=5)

    events = [event async for event in agent.run_stream("hi", context)]

    assert transport._call_count == 1, "the same request was sent again"
    assert [e for e in events if isinstance(e, Error)]
    assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.error]


class TestTerminalReasonsThatAreNotFailures:
    """A decline, a cancellation and an overflow each end the run for a reason a caller can act on."""

    @staticmethod
    async def _end(stop: StopReason) -> list[object]:
        transport = StubTransport([[TextDelta(0, "x"), IterationEnd(1, stop, Usage(1, 1))]])
        agent = Agent(system="", tools=[], transport=transport, max_iterations=3)
        return [event async for event in agent.run_stream("hi", MemoryContextStore())]

    async def test_a_cancellation_is_not_reported_as_a_broken_transport(self) -> None:
        events = await self._end(StopReason.cancelled)
        assert not [e for e in events if isinstance(e, Error)]
        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.cancelled]

    async def test_an_overflow_is_not_reported_as_a_broken_transport(self) -> None:
        # A caller that retries on Error would retry a conversation that cannot fit, for ever.
        events = await self._end(StopReason.context_window_exceeded)
        assert not [e for e in events if isinstance(e, Error)]
        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [
            StopReason.context_window_exceeded
        ]

    async def test_a_genuine_failure_still_reports_one(self) -> None:
        # A transport reports a failure by raising. IterationEnd cannot carry StopReason.error,
        # because the agent can only pass that on as a bare RuntimeError naming nothing.
        transport = StubTransport([[TextDelta(0, "half"), StreamError("the provider failed")]])
        agent = Agent(system="", tools=[], transport=transport, max_iterations=3)

        events = [event async for event in agent.run_stream("hi", MemoryContextStore())]

        assert [str(e.exception) for e in events if isinstance(e, Error)] == ["the provider failed"]
        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.error]


class TestATruncatedTurnDoesNotAct:
    """A turn cut off mid-generation may hold a half-written call, so nothing is run from it."""

    @staticmethod
    async def _dispatched(stop: StopReason) -> list[str]:
        ran: list[str] = []

        async def wire(amount: str = "") -> str:
            ran.append(amount)
            return "sent"

        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "wire"),
                    ToolInputDelta(0, "c1", '{"amount": "1000"}'),
                    IterationEnd(1, stop, Usage(1, 1)),
                ],
                make_text_response("done", 2),
            ]
        )
        agent = Agent(
            system="",
            tools=[Tool(name="wire", description="send money", handler=wire)],
            transport=transport,
            max_iterations=3,
        )
        async for _ in agent.run_stream("go", MemoryContextStore()):
            pass
        return ran

    async def test_an_overflowed_turn_runs_nothing(self) -> None:
        # It dispatched, then looped without ever reading the stop reason, so the tool ran twice.
        assert await self._dispatched(StopReason.context_window_exceeded) == []

    async def test_a_turn_that_hit_the_output_cap_runs_nothing(self) -> None:
        assert await self._dispatched(StopReason.max_tokens) == []

    async def test_a_finished_turn_still_runs_its_call(self) -> None:
        assert await self._dispatched(StopReason.tool_use) == ["1000"]


class TestWhereOneTextBlockEnds:
    """Text merges into the block being built. A proof finishes that block."""

    @staticmethod
    async def _stored(events: list[StreamEvent]) -> list[tuple[str, str]]:
        context = MemoryContextStore()
        await Agent(system="", tools=[], transport=StubTransport([events])).run("hi", context)
        turn = (await context.get_history())[-1]
        return [(b.text, b.signature) for b in turn.content if isinstance(b, TextBlock)]

    async def test_an_answer_split_across_chunks_is_one_block(self) -> None:
        # Gemini numbers the parts it has seen, not the parts of the answer, so a chunk boundary
        # carries a fresh index. Read as a part boundary it broke every answer into one block per
        # chunk, and the proof then covered only the last of them.
        stored = await self._stored(
            [
                TextDelta(0, "Hello "),
                TextDelta(1, "there "),
                TextDelta(2, "world"),
                IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
            ]
        )

        assert stored == [("Hello there world", "")]

    async def test_a_proof_finishes_the_block_it_signs(self) -> None:
        # The provider computed the signature over the text it had, so a later delta starts a new
        # block rather than leaving a proof that disagrees with the text beside it.
        stored = await self._stored(
            [
                TextDelta(0, "signed"),
                TextSignature(index=0, signature="S1"),
                TextDelta(0, "after"),
                IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
            ]
        )

        assert stored == [("signed", "S1"), ("after", "")]


class TestOnlyVouchedForCallsRun:
    """The turn must say it finished before anything it produced is acted on."""

    @staticmethod
    async def _dispatched(stop: StopReason) -> list[str]:
        ran: list[str] = []

        async def wire(amount: str = "") -> str:
            ran.append(amount)
            return "sent"

        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "wire"),
                    ToolInputDelta(0, "c1", '{"amount": "1000"}'),
                    IterationEnd(1, stop, Usage(1, 1)),
                ],
                make_text_response("done", 2),
            ]
        )
        agent = Agent(
            system="",
            tools=[Tool(name="wire", description="send money", handler=wire)],
            transport=transport,
            max_iterations=3,
        )
        async for _ in agent.run_stream("go", MemoryContextStore()):
            pass
        return ran

    async def test_a_reason_the_agent_does_not_know_runs_nothing(self) -> None:
        # Named as the reasons to refuse instead, a reason added to StopReason later was absent
        # from that list, so a turn nobody vouched for had its calls dispatched.
        assert await self._dispatched(cast(StopReason, "invented_later")) == []

    @pytest.mark.parametrize("stop", [StopReason.tool_use, StopReason.end_turn, StopReason.pause_turn])
    async def test_a_turn_that_finished_still_runs_its_call(self, stop: StopReason) -> None:
        # A gateway answers "stop" beside a tool call, and a paused server-side loop did not fail.
        assert await self._dispatched(stop) == ["1000"]


async def test_closing_a_stream_stops_a_tool_that_is_still_producing() -> None:
    # Closing an async generator throws GeneratorExit, which is not a CancelledError, so the
    # dispatch task went on running and on filling a queue nobody would read again.
    produced: list[int] = []

    async def handler(**kwargs: object) -> str:
        return "done"

    async def stream(**kwargs: object) -> AsyncIterator[tuple[str, str]]:
        for at in range(400):
            produced.append(at)
            await asyncio.sleep(0.002)
            yield "chunk", str(at)

    handler.stream = stream  # type: ignore[attr-defined]
    tool: Tool[Any] = Tool(name="slow", description="streams for a while", handler=handler)
    transport = StubTransport(
        [
            [
                ToolUseStart(0, "c1", "slow"),
                ToolInputDelta(0, "c1", "{}"),
                IterationEnd(1, StopReason.tool_use, Usage(1, 1)),
            ]
        ]
    )
    agent = Agent(system="", tools=[tool], transport=transport, max_iterations=2)

    events = agent.run_stream("go", MemoryContextStore())
    async for event in events:
        if isinstance(event, ToolOutputDelta):
            break
    await events.aclose()

    at_close = len(produced)
    await asyncio.sleep(0.2)
    assert len(produced) == at_close, "the tool kept running after the consumer closed the stream"


async def test_audio_the_turn_produced_is_stored_with_it() -> None:
    # Gemini returns audio as an inlineData part, so the transport emits AudioOutput. Yielded but
    # never accumulated, the sound reached the caller and was missing from the replayed turn.
    transport = StubTransport(
        [
            [
                ImageOutput(index=0, data=b"png", media_type="image/png"),
                AudioOutput(index=1, data=b"wav", media_type="audio/wav"),
                IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
            ]
        ]
    )
    context = MemoryContextStore()

    await Agent(system="", tools=[], transport=transport).run("hi", context)

    stored = (await context.get_history())[-1].content
    assert [type(block).__name__ for block in stored] == ["ImageBlock", "AudioBlock"]


class TestATransportThatSaysNothing:
    """The turn is over only when the transport says why it stopped."""

    async def test_a_stream_that_just_ends_is_not_a_finished_answer(self) -> None:
        # Defaulted to end_turn, half an answer was stored and returned as a whole one. Three
        # transports guard this themselves; this is the backstop for the one that forgets.
        transport = StubTransport([[TextDelta(0, "half an ans")]])
        context = MemoryContextStore()

        events = [e async for e in Agent(system="", tools=[], transport=transport).run_stream("hi", context)]

        assert [str(e.exception) for e in events if isinstance(e, Error)] == [
            "Transport ended without an IterationEnd"
        ]
        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.error]

    async def test_an_iteration_end_cannot_carry_the_reason_it_cannot_explain(self) -> None:
        # The agent can only pass StopReason.error on as a bare RuntimeError naming nothing, so a
        # transport must raise with the provider's own message instead.
        with pytest.raises(StreamError, match="cannot carry"):
            IterationEnd(1, StopReason.error, Usage(1, 1))


class TestAProofNeverInventsABlock:
    """A signature attaches to content the provider produced, or it is dropped."""

    @staticmethod
    async def _stored(events: list[StreamEvent]) -> list[tuple[str, str, str]]:
        context = MemoryContextStore()
        await Agent(system="", tools=[], transport=StubTransport([events])).run("hi", context)
        turn = (await context.get_history())[-1]
        return [(type(b).__name__, getattr(b, "text", ""), getattr(b, "signature", "")) for b in turn.content]

    async def test_a_second_proof_does_not_make_an_empty_block(self) -> None:
        # The empty block it used to append is replayed to the provider as a signed empty part.
        stored = await self._stored(
            [
                TextDelta(0, "42"),
                TextSignature(index=0, signature="A"),
                TextSignature(index=0, signature="B"),
                IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
            ]
        )

        assert stored == [("TextBlock", "42", "A")]

    async def test_a_proof_reaches_its_text_past_a_block_of_another_kind(self) -> None:
        stored = await self._stored(
            [
                TextDelta(0, "answer"),
                ReasoningDelta(1, "after"),
                TextSignature(index=0, signature="SIG"),
                IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
            ]
        )

        assert stored == [("TextBlock", "answer", "SIG"), ("ReasoningBlock", "after", "")]


async def test_a_turn_that_broke_after_reporting_its_cost_still_bills_for_it() -> None:
    # Counted after the consume loop only, a failure between IterationEnd and the end of the stream
    # told the caller the turn was free. The provider had already charged for it.
    transport = StubTransport(
        [
            [
                TextDelta(0, "hi"),
                IterationEnd(1, StopReason.end_turn, Usage(1000, 500)),
                StreamError("died after reporting usage"),
            ]
        ]
    )

    events = [e async for e in Agent(system="", tools=[], transport=transport).run_stream("hi", MemoryContextStore())]

    ends = [e for e in events if isinstance(e, SessionEndEvent)]
    assert ends[0].total_usage == Usage(1000, 500)


class TestARepetitionCutIsReportedAsOne:
    """`_RepetitionDetector` stops the turn mid-stream, and no IterationEnd follows it."""

    #: Long enough to reach the detector's minimum, and a two-character period it fires on.
    LOOP = "ab" * 500

    @staticmethod
    def _transport(usage: Usage | None = None) -> StubTransport:
        transport = StubTransport([[TextDelta(0, TestARepetitionCutIsReportedAsOne.LOOP)]])
        if usage is not None:
            # What a transport that reports running totals has by the time the cut lands.
            transport.last_usage = usage  # type: ignore[attr-defined]
        return transport

    async def test_the_caller_is_told_axio_cut_the_turn_and_not_that_it_finished(self) -> None:
        agent = Agent(system="", transport=self._transport())
        events = [e async for e in agent.run_stream("hi", MemoryContextStore())]

        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert [e.stop_reason for e in ends] == [StopReason.repetition]

    async def test_the_tokens_of_the_turn_it_cut_are_counted_in_the_store(self) -> None:
        # The message is appended either way. A store that never counted it drifts further from
        # the real context size on every cut, and autocompaction fires late by that much.
        context = MemoryContextStore()
        agent = Agent(system="", transport=self._transport(Usage(120, 40)))

        events = [e async for e in agent.run_stream("hi", context)]

        assert await context.get_context_tokens() == (120, 40)
        ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert ends[0].total_usage == Usage(120, 40)

    async def test_a_transport_with_nothing_to_report_says_so(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = Agent(system="", transport=self._transport())

        with caplog.at_level(logging.WARNING, logger="axio.agent"):
            async for _ in agent.run_stream("hi", MemoryContextStore()):
                pass

        assert any("missing from the total" in record.getMessage() for record in caplog.records)


async def test_the_transport_stream_is_closed_when_the_turn_ends() -> None:
    """Left suspended, the HTTP response under it is released by the collector and not by the turn.

    The repetition cut returns from the middle of the loop, which is where this went unnoticed.
    """
    closed = False

    class _Watched:
        async def _generate(self) -> AsyncIterator[StreamEvent]:
            nonlocal closed
            try:
                yield TextDelta(0, TestARepetitionCutIsReportedAsOne.LOOP)
                yield IterationEnd(1, StopReason.end_turn, Usage(1, 1))
            finally:
                closed = True

        def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
            return self._generate()

    agent = Agent(system="", transport=cast(Any, _Watched()))
    async for _ in agent.run_stream("hi", MemoryContextStore()):
        pass

    assert closed, "the turn ended without closing the stream it stopped reading"
