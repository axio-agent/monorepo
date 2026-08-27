"""The Responses API: the vocabulary it publishes, and the request it takes."""

import json
import logging
from typing import Any

import pytest
from axio.blocks import AudioBlock, ImageBlock, ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import (
    BlockEnd,
    Citation,
    IterationEnd,
    IterationStart,
    ProviderEvent,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    StreamEvent,
    TextDelta,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.tool import Tool
from axio.types import StopReason, Usage
from axio_sse import Event, UnknownEvent

from axio_responses import Responses, convert_messages, convert_tools


async def get_weather(location: str, units: str = "celsius") -> str:
    return f"Weather in {location}: 22{units[0]}"


PUBLISHED_EVENTS = {
    "error",
    "response.audio.delta",
    "response.audio.done",
    "response.audio.transcript.delta",
    "response.audio.transcript.done",
    "response.code_interpreter_call.completed",
    "response.code_interpreter_call.in_progress",
    "response.code_interpreter_call.interpreting",
    "response.code_interpreter_call_code.delta",
    "response.code_interpreter_call_code.done",
    "response.completed",
    "response.content_part.added",
    "response.content_part.done",
    "response.created",
    "response.custom_tool_call_input.delta",
    "response.custom_tool_call_input.done",
    "response.failed",
    "response.file_search_call.completed",
    "response.file_search_call.in_progress",
    "response.file_search_call.searching",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.image_generation_call.completed",
    "response.image_generation_call.generating",
    "response.image_generation_call.in_progress",
    "response.image_generation_call.partial_image",
    "response.in_progress",
    "response.incomplete",
    "response.mcp_call.completed",
    "response.mcp_call.failed",
    "response.mcp_call.in_progress",
    "response.mcp_call_arguments.delta",
    "response.mcp_call_arguments.done",
    "response.mcp_list_tools.completed",
    "response.mcp_list_tools.failed",
    "response.mcp_list_tools.in_progress",
    "response.output_item.added",
    "response.output_item.done",
    "response.output_text.annotation.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.queued",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
    "response.refusal.delta",
    "response.refusal.done",
    "response.shell_call_command.added",
    "response.shell_call_command.delta",
    "response.shell_call_command.done",
    "response.shell_call_output_content.delta",
    "response.shell_call_output_content.done",
    "response.web_search_call.completed",
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
}


def _reads(reader: Responses, **payload: Any) -> list[StreamEvent]:
    return reader.read(Event(data=json.dumps(payload)))


def test_the_reader_names_only_what_it_interprets() -> None:
    """The names are the reader's own vocabulary, not the API's.

    The API names one event family per tool it can run, so the published set grows whenever a tool
    is added — a list of it here would be stale by the next release, and would report a new tool as
    news about the protocol. What the reader claims is what it interprets; the rest is forwarded.
    """
    interpreted = Responses.names()
    assert interpreted <= PUBLISHED_EVENTS, "a name here is not one the schema publishes"
    assert "response.output_text.delta" in interpreted
    assert "response.web_search_call.searching" not in interpreted


def test_every_name_it_claims_is_one_the_schema_publishes() -> None:
    # The check worth keeping: a typo in a claimed name is a handler that never runs.
    assert Responses.names() - PUBLISHED_EVENTS == set()


def test_an_event_outside_the_published_list_is_refused_under_strict() -> None:
    # What a CI run holds against the schema on the day OpenAI adds an event.
    with pytest.raises(UnknownEvent, match="response.something-new"):
        Responses().read(Event(data='{"type":"response.something-new"}'), strict=True)


def test_an_incomplete_response_reports_max_tokens() -> None:
    # Left unread this ended the turn as end_turn, telling the agent a truncated answer was whole.
    reader = Responses()
    _reads(
        reader,
        type="response.incomplete",
        response={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 3, "output_tokens": 4},
        },
    )
    assert reader.finished() == IterationEnd(
        iteration=0, stop_reason=StopReason.max_tokens, usage=Usage(input_tokens=3, output_tokens=4)
    )


def test_a_stream_error_event_raises() -> None:
    # Left unread the stream simply stopped and the turn reported a normal finish.
    with pytest.raises(StreamError, match="rate limit reached"):
        _reads(Responses(), type="error", code="rate_limit_exceeded", message="rate limit reached")


def test_a_refusal_reaches_the_caller_as_a_refusal() -> None:
    # A refusal arrives instead of the text, never beside it, so dropping it answered nothing. It
    # is not a TextDelta either: as ordinary assistant text no consumer can tell it from an answer.
    reader = Responses()
    made = _reads(reader, type="response.refusal.delta", delta="I cannot help with that", output_index=0)

    refusal = made[0]
    assert isinstance(refusal, Refusal)
    assert (refusal.index, refusal.text) == (0, "I cannot help with that")
    assert refusal.raw["delta"] == "I cannot help with that"
    assert reader.finished().stop_reason == StopReason.refusal


def test_an_annotation_becomes_a_citation() -> None:
    made = _reads(
        Responses(),
        type="response.output_text.annotation.added",
        output_index=1,
        content_index=3,
        annotation={
            "type": "url",
            "text": "as reported",
            "title": "The Report",
            "source": {"type": "url", "url": "https://example.invalid/r"},
            "start_index": 10,
            "end_index": 21,
        },
    )
    citation = made[0]
    assert isinstance(citation, Citation)
    # Indexed by the output item, as the deltas and BlockEnd are. content_index numbers the content
    # part inside that item, which axio has no index of its own for.
    assert (citation.index, citation.url, citation.title) == (1, "https://example.invalid/r", "The Report")
    assert (citation.start, citation.end, citation.unit) == (10, 21, "char")


def test_a_hosted_tool_event_is_forwarded_without_being_listed() -> None:
    # axio has no type for the shell the API runs on its own side, and no list naming it either:
    # one event family per tool means the set is a function of the tools, not of the protocol.
    assert "response.shell_call_command.done" not in Responses.names()
    made = _reads(
        Responses(),
        type="response.shell_call_command.done",
        output_index=2,
        command=["bash", "-lc", "ls"],
    )
    forwarded = made[0]
    assert isinstance(forwarded, ProviderEvent)
    assert (forwarded.provider, forwarded.kind, forwarded.index) == ("openai", "response.shell_call_command.done", 2)
    assert forwarded.data["command"] == ["bash", "-lc", "ls"]


def test_an_event_from_a_tool_nobody_has_heard_of_is_forwarded_too() -> None:
    # The point of having no list: a tool added tomorrow needs no change here.
    made = _reads(Responses(), type="response.brand_new_call.in_progress", output_index=0)
    assert isinstance(made[0], ProviderEvent)
    assert made[0].kind == "response.brand_new_call.in_progress"


def test_the_model_that_served_the_turn_is_reported() -> None:
    # Server-side fallback and sticky routing substitute a different model at a different price.
    assert _reads(Responses(), type="response.created", response={"id": "resp_1", "model": "gpt-5-codex"}) == [
        IterationStart(iteration=0, id="resp_1", model="gpt-5-codex")
    ]


def test_reasoning_text_reads_like_the_reasoning_summary() -> None:
    reader = Responses()
    assert _reads(reader, type="response.reasoning_text.delta", delta="thinking") == [
        ReasoningDelta(index=0, delta="thinking")
    ]
    assert _reads(reader, type="response.reasoning_summary_text.delta", delta="summary") == [
        ReasoningDelta(index=0, delta="summary")
    ]


def test_the_token_slices_are_read_and_not_added() -> None:
    """The Responses API reports its slices inside their totals, so nothing is added here."""
    reader = Responses()
    _reads(
        reader,
        type="response.completed",
        response={
            "status": "completed",
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 50},
                "output_tokens": 500,
                "output_tokens_details": {"reasoning_tokens": 320},
                "total_tokens": 1500,
            },
        },
    )
    usage = reader.finished().usage
    assert (usage.input_tokens, usage.output_tokens) == (1000, 500)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (800, 50)
    assert usage.reasoning_tokens == 320
    assert usage.uncached_input_tokens == 150
    assert usage.answer_tokens == 180


def test_a_refusal_survives_the_response_that_carries_it() -> None:
    """A refusal is ordinary output content, so `response.completed` always follows it.

    The status enum has no refusal member, so reading the status over the top reported a declined
    turn as a finished answer, and the session ended as if the model had answered.
    """
    reader = Responses()
    _reads(reader, type="response.refusal.delta", delta="I cannot help with that", output_index=0)
    _reads(
        reader,
        type="response.completed",
        response={"status": "completed", "usage": {"input_tokens": 4, "output_tokens": 0}},
    )
    assert reader.finished().stop_reason == StopReason.refusal


def test_an_ordinary_turn_still_reads_its_status() -> None:
    reader = Responses()
    _reads(reader, type="response.completed", response={"status": "completed", "usage": {}})
    assert reader.finished().stop_reason == StopReason.end_turn


# ---------------------------------------------------------------------------
# The request this API takes
# ---------------------------------------------------------------------------


def test_the_system_prompt_is_instructions_and_not_a_message() -> None:
    instructions, items = convert_messages([Message(role="user", content=[TextBlock(text="hi")])], "be brief")
    assert instructions == "be brief"
    assert items == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]


def test_a_tool_call_and_its_output_are_items_beside_the_messages() -> None:
    # Not blocks inside them: this API keeps the call and the result at the top level.
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="call_1", name="get_weather", input={"city": "Paris"})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="22C")]),
    ]
    _, items = convert_messages(messages, "")
    assert [i["type"] for i in items] == ["function_call", "function_call_output"]
    assert items[0]["call_id"] == "call_1" and json.loads(items[0]["arguments"]) == {"city": "Paris"}
    assert items[1]["output"] == "22C"


def test_an_image_travels_as_a_data_uri() -> None:
    block = ImageBlock(media_type="image/png", data=b"\x89PNG\r\n\x1a\nfake")
    _, items = convert_messages([Message(role="user", content=[block])], "")
    part = items[0]["content"][0]
    assert part["type"] == "input_image"
    assert part["image_url"].startswith("data:image/png;base64,")


def test_a_tool_declares_itself_without_the_titles_pydantic_adds() -> None:
    tool: Tool[Any] = Tool(name="get_weather", description="Look it up", handler=get_weather)
    declared = convert_tools([tool])[0]
    assert declared["name"] == "get_weather"
    assert "title" not in json.dumps(declared["parameters"])


# ---------------------------------------------------------------------------
# Reasoning that survives into the next request
# ---------------------------------------------------------------------------


def test_a_finished_reasoning_item_yields_its_proof() -> None:
    """response.output_item.done is the only event carrying the encrypted reasoning.

    Nothing is stored on the provider's side, so unread it is gone and the next round starts blind.
    """
    made = _reads(
        Responses(),
        type="response.output_item.done",
        output_index=0,
        item={"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAAAB..."},
    )
    assert made[0] == ReasoningSignature(index=0, data="gAAAAAB...", id="rs_1")
    assert made[1] == BlockEnd(index=0)


def test_a_finished_item_that_is_not_reasoning_only_closes_the_block() -> None:
    made = _reads(
        Responses(),
        type="response.output_item.done",
        output_index=2,
        item={"type": "function_call", "id": "item_1", "call_id": "call_1"},
    )
    assert made == [BlockEnd(index=2)]


def test_a_reasoning_item_with_nothing_to_replay_only_closes_the_block() -> None:
    made = _reads(Responses(), type="response.output_item.done", output_index=0, item={"type": "reasoning"})
    assert made == [BlockEnd(index=0)]


def test_the_reasoning_goes_back_as_the_item_this_api_takes() -> None:
    messages = [
        Message(
            role="assistant",
            content=[
                ReasoningBlock(text="", signature="gAAAAAB...", id="rs_1"),
                ToolUseBlock(id="call_1", name="get_weather", input={"city": "Paris"}),
            ],
        )
    ]
    _, items = convert_messages(messages, "")
    assert items[0] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAAAB...", "summary": []}
    assert items[1]["type"] == "function_call"


def test_a_reasoning_block_with_no_proof_is_left_out_rather_than_sent_empty() -> None:
    # id and summary are required beside the proof; without the proof the item says nothing.
    messages = [Message(role="assistant", content=[ReasoningBlock(text="thought about it"), TextBlock(text="answer")])]
    _, items = convert_messages(messages, "")
    assert [i.get("type") or i["role"] for i in items] == ["assistant"]


def test_splitting_the_done_events_kept_both_names() -> None:
    # response.output_item.done and response.content_part.done are read by different methods now;
    # neither name may have gone missing in the split.
    assert {"response.output_item.done", "response.content_part.done"} <= Responses.names()


def test_an_incomplete_reason_nobody_knows_is_raised_and_not_returned() -> None:
    """The event says the response did not complete, and the reason names why.

    Returned as `IterationEnd(error)` it reaches the agent's wildcard, and the caller is told only
    `Transport stopped with: error` with the API's own reason gone.
    """
    with pytest.raises(StreamError, match="some_new_limit"):
        _reads(
            Responses(),
            type="response.incomplete",
            response={"status": "incomplete", "incomplete_details": {"reason": "some_new_limit"}, "usage": {}},
        )


def test_the_incomplete_reasons_the_schema_publishes_still_map() -> None:
    for reason, expected in (("max_output_tokens", StopReason.max_tokens), ("content_filter", StopReason.refusal)):
        reader = Responses()
        _reads(
            reader,
            type="response.incomplete",
            response={"status": "incomplete", "incomplete_details": {"reason": reason}, "usage": {}},
        )
        assert reader.finished().stop_reason == expected, reason


# ---------------------------------------------------------------------------
# Which output item a delta belongs to
# ---------------------------------------------------------------------------


def test_a_text_delta_carries_the_item_it_belongs_to() -> None:
    made = _reads(Responses(), type="response.output_text.delta", delta="hi", output_index=2)
    assert made == [TextDelta(index=2, delta="hi")]


def test_a_reasoning_delta_carries_the_item_it_belongs_to() -> None:
    made = _reads(Responses(), type="response.reasoning_summary_text.delta", delta="weighing", output_index=1)
    assert made == [ReasoningDelta(index=1, delta="weighing")]


def test_a_delta_and_the_event_that_closes_its_block_agree() -> None:
    """Fixed at zero, every delta claimed block 0 while BlockEnd kept the real index.

    Nothing downstream could then tell which block a delta was part of, and the agent merged the
    reasoning of two items into one block that replayed under the wrong id.
    """
    reader = Responses()
    delta = _reads(reader, type="response.reasoning_summary_text.delta", delta="second", output_index=1)
    closed = _reads(
        reader,
        type="response.output_item.done",
        output_index=1,
        item={"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAAAB"},
    )
    reasoning = delta[0]
    assert isinstance(reasoning, ReasoningDelta)
    assert reasoning.index == 1
    assert [e.index for e in closed if isinstance(e, (ReasoningSignature, BlockEnd))] == [1, 1]


# ---------------------------------------------------------------------------
# A tool that returns more than a string
# ---------------------------------------------------------------------------


def test_a_structured_tool_result_travels_as_the_parts_this_api_takes() -> None:
    """json.dumps on the blocks raises: they are slotted dataclasses and not JSON.

    A tool returning anything but a string crashed the request before it was sent.
    """
    result = ToolResultBlock(
        tool_use_id="call_1",
        content=[TextBlock(text="here is the chart"), ImageBlock(media_type="image/png", data=b"\x89PNG")],
    )
    _, items = convert_messages([Message(role="user", content=[result])], "")
    output = items[0]["output"]
    assert output[0] == {"type": "input_text", "text": "here is the chart"}
    assert output[1]["type"] == "input_image"
    assert output[1]["image_url"].startswith("data:image/png;base64,")


def test_a_plain_string_result_is_left_a_string() -> None:
    result = ToolResultBlock(tool_use_id="call_1", content="22C")
    _, items = convert_messages([Message(role="user", content=[result])], "")
    assert items[0]["output"] == "22C"


def test_media_this_api_has_no_part_for_is_named_rather_than_dropped() -> None:
    # The model is told what the tool produced instead of being handed a turn with a gap in it.
    result = ToolResultBlock(tool_use_id="call_1", content=[AudioBlock(media_type="audio/wav", data=b"RIFF")])
    _, items = convert_messages([Message(role="user", content=[result])], "")
    assert items[0]["output"] == [{"type": "input_text", "text": "[audio/wav, which this API takes no part for]"}]


def test_a_result_with_no_readable_part_is_not_reported_as_nothing() -> None:
    result = ToolResultBlock(tool_use_id="call_1", content=[])
    _, items = convert_messages([Message(role="user", content=[result])], "")
    assert items[0]["output"] == ""


def test_a_stream_that_ended_without_a_terminal_event_raises() -> None:
    """A cut connection is not a finished answer.

    Reported as end_turn a truncated response was stored and returned to the caller as the model's
    whole answer.
    """
    reader = Responses()
    _reads(reader, type="response.output_text.delta", delta="half an ans", output_index=0)
    with pytest.raises(StreamError, match="without response.completed"):
        reader.finished()


def test_a_completed_stream_still_finishes() -> None:
    reader = Responses()
    _reads(reader, type="response.completed", response={"status": "completed", "usage": {}})
    assert reader.finished().stop_reason == StopReason.end_turn


def test_an_incomplete_stream_still_finishes() -> None:
    # It reached a terminal event; it just did not complete.
    reader = Responses()
    _reads(
        reader,
        type="response.incomplete",
        response={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {}},
    )
    assert reader.finished().stop_reason == StopReason.max_tokens


def test_a_completed_status_nobody_knows_is_not_a_finished_answer() -> None:
    # Read as end_turn, a response the API failed or cut short is stored and returned as the
    # model's whole answer.
    reader = Responses()
    _reads(reader, type="response.completed", response={"status": "some_new_state", "usage": {}})
    assert reader.finished().stop_reason == StopReason.error


def test_a_failed_status_on_a_completed_event_is_an_error() -> None:
    reader = Responses()
    _reads(reader, type="response.completed", response={"status": "failed", "usage": {}})
    assert reader.finished().stop_reason == StopReason.error


def test_an_unknown_status_is_not_overridden_by_a_pending_tool_call() -> None:
    # The calls are real output, but an unrecognised status says the turn is not known to have
    # succeeded, so running the tool on it would act on a response nobody vouched for.
    reader = Responses()
    _reads(
        reader,
        type="response.completed",
        response={"status": "some_new_state", "usage": {}, "output": [{"type": "function_call"}]},
    )
    assert reader.finished().stop_reason == StopReason.error


def test_a_completed_turn_holding_a_call_still_asks_for_the_tool() -> None:
    reader = Responses()
    _reads(
        reader,
        type="response.completed",
        response={"status": "completed", "usage": {}, "output": [{"type": "function_call"}]},
    )
    assert reader.finished().stop_reason == StopReason.tool_use


def test_a_refusal_still_survives_the_status() -> None:
    reader = Responses()
    _reads(reader, type="response.refusal.delta", delta="no", output_index=0)
    _reads(reader, type="response.completed", response={"status": "completed", "usage": {}})
    assert reader.finished().stop_reason == StopReason.refusal


def test_a_citation_names_the_same_block_the_text_did() -> None:
    """Indexed by content_index a citation named a different block from its own text.

    The deltas and the event that closes a block are both indexed by output item.
    """
    reader = Responses()
    delta = _reads(reader, type="response.output_text.delta", delta="as reported", output_index=2)
    cited = _reads(
        reader,
        type="response.output_text.annotation.added",
        output_index=2,
        content_index=0,
        annotation={"type": "url", "source": {"url": "https://example.invalid/r"}},
    )
    text, citation = delta[0], cited[0]
    assert isinstance(text, TextDelta) and isinstance(citation, Citation)
    assert text.index == citation.index == 2


def test_a_system_message_in_the_history_is_not_dropped() -> None:
    """A system message inside the history is not the system prompt of this request.

    Skipped, an instruction the caller put in the conversation disappeared from every request
    after it. The chat-completions converter has always kept it.
    """
    messages = [
        Message(role="system", content=[TextBlock(text="wrap up now")]),
        Message(role="user", content=[TextBlock(text="hi")]),
    ]
    instructions, items = convert_messages(messages, "be brief")
    assert instructions == "be brief"
    assert items[0] == {"role": "system", "content": [{"type": "input_text", "text": "wrap up now"}]}
    assert items[1]["role"] == "user"


def test_an_empty_system_message_adds_nothing() -> None:
    _, items = convert_messages([Message(role="system", content=[])], "")
    assert items == []


def test_a_turn_carrying_a_result_and_a_question_keeps_both() -> None:
    """Read only where the turn was nothing but results, the result was dropped.

    The orphan sweep then fabricated "[Tool was not executed]" and told the model its tool had not
    run when it had.
    """
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="t", input={})]),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="c1", content="22C"), TextBlock(text="and tomorrow?")],
        ),
    ]
    _, items = convert_messages(messages, "")
    assert [i.get("type") or i["role"] for i in items] == ["function_call", "function_call_output", "user"]
    assert items[1]["output"] == "22C"
    assert items[2]["content"] == [{"type": "input_text", "text": "and tomorrow?"}]


class TestAssistantTurnOrder:
    """The request carries the turn's blocks in the order the turn stored them."""

    def test_text_stays_in_front_of_the_call_it_introduces(self) -> None:
        # Positioned by counting ToolUseBlocks back from the tail, the text moved behind the call
        # as soon as reasoning was stored after that call.
        turn = Message(
            role="assistant",
            content=[
                TextBlock(text="Let me look."),
                ToolUseBlock(id="c1", name="search", input={"q": "x"}),
                ReasoningBlock(text="", signature="enc", id="rs_1"),
            ],
        )
        _, items = convert_messages([turn], "")

        kinds = [item.get("type") or item.get("role") for item in items]
        # The trailing reasoning item is gone: the API refuses one with no item after it.
        assert kinds[:2] == ["assistant", "function_call"]
        assert "reasoning" not in kinds

    def test_reasoning_still_precedes_the_call_it_belongs_to(self) -> None:
        turn = Message(
            role="assistant",
            content=[
                ReasoningBlock(text="", signature="enc", id="rs_1"),
                ToolUseBlock(id="c1", name="search", input={"q": "x"}),
                TextBlock(text="Here it is."),
            ],
        )
        _, items = convert_messages([turn], "")

        kinds = [item.get("type") or item.get("role") for item in items]
        assert kinds[:3] == ["reasoning", "function_call", "assistant"]

    def test_consecutive_text_blocks_stay_one_message(self) -> None:
        turn = Message(role="assistant", content=[TextBlock(text="a"), TextBlock(text="b")])
        _, items = convert_messages([turn], "")

        assert items == [{"role": "assistant", "content": "ab"}]

    def test_the_text_goes_back_as_a_shape_the_api_defines(self) -> None:
        # An `output_text` part belongs to an output message, which also requires `id`, `type` and
        # `status`. Sent without them it matched no input item the API defines.
        turn = Message(role="assistant", content=[TextBlock(text="hello")])

        _, items = convert_messages([turn], "")

        assert items == [{"role": "assistant", "content": "hello"}]
        assert "output_text" not in json.dumps(items)


def test_a_turn_that_ends_on_reasoning_does_not_replay_it() -> None:
    # /v1/responses answers "Item 'rs_...' of type 'reasoning' was provided without its required
    # following item", so one trailing item would 400 every request after it.
    turn = Message(role="assistant", content=[TextBlock(text="done"), ReasoningBlock(text="", signature="e", id="r")])

    _, items = convert_messages([turn], "")

    assert [item.get("type") or item.get("role") for item in items] == ["assistant"]


def test_tool_arguments_are_not_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    # Arguments are the model's input to a tool, so they carry whatever the user typed. INFO is on
    # in most deployments, which put an API key in a shipped log.
    secret = '{"token": "sk-live-do-not-log-me"}'
    reader = Responses()
    with caplog.at_level(logging.INFO, logger="axio.responses"):
        _reads(reader, type="response.function_call_arguments.done", item_id="fc_1", name="pay", arguments=secret)

    assert "Tool args complete" in caplog.text, "the operational line still has to be logged"
    assert "sk-live-do-not-log-me" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="axio.responses"):
        _reads(reader, type="response.function_call_arguments.done", item_id="fc_1", name="pay", arguments=secret)

    assert "sk-live-do-not-log-me" in caplog.text, "debug still has to be able to show them"
