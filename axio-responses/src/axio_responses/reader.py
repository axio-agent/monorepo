"""Reading a Responses stream: the payload shapes, and what each event becomes.

The vocabulary is the published ``ResponseStreamEvent`` union. An event missing from ``Responses``
is one the API added after this was written, not one nobody named. A test reading with
``strict=True`` holds that against the schema.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

from axio.events import (
    BlockEnd,
    Citation,
    IterationEnd,
    IterationStart,
    ProviderEvent,
    ProviderOutput,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
)
from axio.exceptions import StreamError
from axio.types import StopReason, Usage, stop_reason_from
from axio_sse import Payload, Reader, Wire, on

from .request import PROVIDER, STOP_REASONS

logger = logging.getLogger("axio.responses")

#: Item types this reader turns into content of its own. Everything else is the API's own tooling
#: — a search it ran, a file it read, code it executed — and has to travel back unread.
_INTERPRETED: Final = frozenset({"message", "reasoning", "function_call"})


@dataclass(frozen=True, slots=True)
class InputDetails(Wire):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class OutputDetails(Wire):
    reasoning_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ResponseUsage(Wire):
    """Both slices arrive inside their totals here, so the reader adds nothing to either."""

    input_tokens: int = 0
    output_tokens: int = 0
    input_tokens_details: InputDetails = field(default_factory=InputDetails)
    output_tokens_details: OutputDetails = field(default_factory=OutputDetails)


@dataclass(frozen=True, slots=True)
class OutputItem(Wire):
    type: str = ""
    id: str = ""
    call_id: str = ""
    name: str = ""
    #: Present on a ``reasoning`` item when the request asked for it. Opaque, and sent back on
    #: the next request.
    encrypted_content: str = ""
    #: The item exactly as it arrived. An item this reader does not interpret has to go back whole
    #: on the next request, and no shape declared here would hold a type nobody has published yet.
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class IncompleteDetails(Wire):
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ResponseError(Wire):
    message: str = ""
    code: str = ""


@dataclass(frozen=True, slots=True)
class ResponseObject(Wire):
    id: str = ""
    model: str = ""
    status: str = ""
    usage: ResponseUsage = field(default_factory=ResponseUsage)
    output: list[OutputItem] = field(default_factory=list)
    error: ResponseError = field(default_factory=ResponseError)
    incomplete_details: IncompleteDetails = field(default_factory=IncompleteDetails)


@dataclass(frozen=True, slots=True)
class AnnotationSource(Wire):
    url: str = ""
    filename: str = ""


@dataclass(frozen=True, slots=True)
class Annotation(Wire):
    """One attribution. It arrives under a url shape, a file shape and the older flat citation, so
    each field is declared wherever a shape put it. The whole object travels in ``raw``."""

    text: str = ""
    title: str | None = None
    url: str | None = None
    file_id: str | None = None
    source: AnnotationSource = field(default_factory=AnnotationSource)
    start_index: int | None = None
    end_index: int | None = None
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class Created(Wire, name="response.created"):
    response: ResponseObject = field(default_factory=ResponseObject)


@dataclass(frozen=True, slots=True)
class Completed(Wire, name="response.completed"):
    response: ResponseObject = field(default_factory=ResponseObject)


@dataclass(frozen=True, slots=True)
class Incomplete(Wire, name="response.incomplete"):
    response: ResponseObject = field(default_factory=ResponseObject)


@dataclass(frozen=True, slots=True)
class Failed(Wire, name="response.failed"):
    response: ResponseObject = field(default_factory=ResponseObject)


@dataclass(frozen=True, slots=True)
class StreamFailure(Wire, name="error"):
    """The stream's own error, which carries no response object."""

    message: str = ""
    code: str = ""


@dataclass(frozen=True, slots=True)
class TextDeltaEvent(Wire, name="response.output_text.delta"):
    delta: str = ""
    #: Which output item this belongs to. Fixed at zero, every delta of a multi-item response
    #: shared one index while the events that close a block kept the real one.
    output_index: int = 0


@dataclass(frozen=True, slots=True)
class ReasoningDeltaEvent(Wire, name="response.reasoning_summary_text.delta", also="response.reasoning_text.delta"):
    """Both reasoning channels read the same. A model that sends the text rather than the summary
    would otherwise think in silence."""

    delta: str = ""
    output_index: int = 0


@dataclass(frozen=True, slots=True)
class RefusalDeltaEvent(Wire, name="response.refusal.delta"):
    delta: str = ""
    output_index: int = 0
    raw: Payload = field(default_factory=Payload)


@dataclass(frozen=True, slots=True)
class AnnotationAdded(Wire, name="response.output_text.annotation.added"):
    #: Which output item the cited text belongs to. The deltas and the closing event are indexed
    #: by this, not by content_index.
    output_index: int = 0
    #: Which content part inside that item, which axio has no index of its own for.
    content_index: int = 0
    annotation: Annotation = field(default_factory=Annotation)


@dataclass(frozen=True, slots=True)
class ContentPartDone(Wire, name="response.content_part.done"):
    output_index: int = 0
    content_index: int = 0


@dataclass(frozen=True, slots=True)
class ItemDone(Wire, name="response.output_item.done"):
    output_index: int = 0
    item: OutputItem = field(default_factory=OutputItem)


@dataclass(frozen=True, slots=True)
class ItemAdded(Wire, name="response.output_item.added"):
    output_index: int = 0
    item: OutputItem = field(default_factory=OutputItem)


@dataclass(frozen=True, slots=True)
class ArgumentsDelta(Wire, name="response.function_call_arguments.delta"):
    item_id: str = ""
    output_index: int = 0
    delta: str = ""


@dataclass(frozen=True, slots=True)
class ArgumentsDone(Wire, name="response.function_call_arguments.done"):
    item_id: str = ""
    name: str = ""
    arguments: str = ""


class Responses(Reader[StreamEvent]):
    """Every event the Responses API sends, and what each one becomes.

    The vocabulary is the published ``ResponseStreamEvent`` union. An event missing from this body
    is one the API added after it was written, not one nobody named. A test reading
    with ``strict=True`` holds that against the schema.

    One instance reads one response. Its state is the usage, the stop reason and the id map.
    """

    def __init__(self) -> None:
        self.usage = Usage(0, 0)
        self.stop_reason: StopReason | None = None
        # item_id -> call_id, so a ToolInputDelta carries the id ToolUseStart announced.
        self.call_ids: dict[str, str] = {}

    @on(TextDeltaEvent)
    def _text(self, wire: TextDeltaEvent) -> Iterator[StreamEvent]:
        yield TextDelta(index=wire.output_index, delta=wire.delta)

    @on(ReasoningDeltaEvent)
    def _reasoning(self, wire: ReasoningDeltaEvent) -> Iterator[StreamEvent]:
        yield ReasoningDelta(index=wire.output_index, delta=wire.delta)

    @on(RefusalDeltaEvent)
    def _refusal(self, wire: RefusalDeltaEvent) -> Iterator[StreamEvent]:
        """A refusal arrives instead of the text, never beside it, so dropping it answers nothing."""
        self.stop_reason = StopReason.refusal
        yield Refusal(index=wire.output_index, text=wire.delta, raw=dict(wire.raw))

    @on(AnnotationAdded)
    def _annotation(self, wire: AnnotationAdded) -> Iterator[StreamEvent]:
        """What the text just sent was attributed to."""
        note = wire.annotation
        yield Citation(
            index=wire.output_index,
            cited_text=note.text,
            title=note.title,
            url=note.url or note.source.url or None,
            source_id=note.file_id or note.source.filename or None,
            start=note.start_index,
            end=note.end_index,
            # This API counts characters. Google counts bytes, so the unit travels with them.
            unit="char",
            raw=dict(note.raw),
        )

    @on(Created)
    def _created(self, wire: Created) -> Iterator[StreamEvent]:
        """Which model actually served the turn, which need not be the one asked for."""
        yield IterationStart(iteration=0, id=wire.response.id or None, model=wire.response.model or None)

    @on(ContentPartDone)
    def _content_part_done(self, wire: ContentPartDone) -> None:
        """One content part inside an item is complete.

        Not a BlockEnd: axio indexes a block by output item, and the item's own done event closes
        it. Emitting both closed every block twice, so a consumer that finalises on BlockEnd
        finalised a block it had already finished.
        """

    @on(ItemDone)
    def _item_done(self, wire: ItemDone) -> Iterator[StreamEvent]:
        """The finished item, which for reasoning is the only place its proof arrives.

        Sent back on the next request, it lets the model see the reasoning it had already done.
        Without it a turn that reasoned and then called a tool starts the next round blind.
        """
        if wire.item.type == "reasoning" and wire.item.encrypted_content:
            yield ReasoningSignature(
                index=wire.output_index,
                signature=wire.item.encrypted_content,
                id=wire.item.id,
                provider=PROVIDER,
            )
        elif wire.item.type not in _INTERPRETED:
            # The request says store=False, so this API keeps nothing: every item it produced is
            # expected back on the next one. Watched as a ProviderEvent and never stored, the next
            # request was missing the search the model had just answered from.
            yield ProviderOutput(
                index=wire.output_index,
                provider=PROVIDER,
                kind=wire.item.type,
                data=dict(wire.item.raw),
                id=wire.item.id,
            )
        yield BlockEnd(index=wire.output_index)

    @on(ItemAdded)
    def _item_added(self, wire: ItemAdded) -> Iterator[StreamEvent]:
        item = wire.item
        if item.type != "function_call":
            logger.info("Output item added: type=%s", item.type)
            return
        if item.id:
            self.call_ids[item.id] = item.call_id
        logger.info("Tool call started: %s (call_id=%s, item_id=%s)", item.name, item.call_id, item.id)
        yield ToolUseStart(index=wire.output_index, tool_use_id=item.call_id, name=item.name, provider=PROVIDER)

    @on(ArgumentsDelta)
    def _arguments(self, wire: ArgumentsDelta) -> Iterator[StreamEvent]:
        call_id = self._call_id(wire.item_id)
        logger.debug("Tool args delta: call_id=%s, +%d chars", call_id, len(wire.delta))
        yield ToolInputDelta(index=wire.output_index, tool_use_id=call_id, partial_json=wire.delta)

    # ── what only moves this turn's state ────────────────────────────────────────────────────

    @on(Completed)
    def _completed(self, wire: Completed) -> None:
        response = wire.response
        self._count(response.usage)
        # Not `or "completed"`: a missing or wrongly typed status reads as the empty string here,
        # and coercing it to success made an envelope that never said it finished into a whole
        # answer. Unnamed, it falls to the raise below with the empty string as its name.
        status = response.status
        if self.stop_reason == StopReason.refusal:
            # The status enum has no refusal member, so a declined response still completes.
            pass
        elif status not in STOP_REASONS:
            # Not a finished answer, whatever else it is. Returned as IterationEnd(error) the
            # caller is told only `Transport stopped with: error`, and the status is what they can
            # act on.
            raise StreamError(f"Responses completed with an unknown status: {status!r}")
        else:
            self.stop_reason = STOP_REASONS[status]
            # A finished response still holding a call is a turn that wants the tool run first.
            # Only a finished one: `cancelled`, `content_filter` and `max_output_tokens` also
            # arrive here carrying the calls the turn had streamed before it stopped. Rewritten to
            # tool_use they pass the agent's dispatch gate, and half a turn's calls run.
            if self.stop_reason is StopReason.end_turn and any(
                item.type == "function_call" for item in response.output
            ):
                self.stop_reason = StopReason.tool_use
        logger.info(
            "Response completed: status=%s, stop=%s, in=%d, out=%d",
            status,
            self.stop_reason,
            self.usage.input_tokens,
            self.usage.output_tokens,
        )

    @on(Incomplete)
    def _incomplete(self, wire: Incomplete) -> None:
        """A response the API cut short. Left unread it ends the turn as ``end_turn``, which tells
        the agent a truncated answer is a whole one."""
        self._count(wire.response.usage)
        reason = wire.response.incomplete_details.reason
        self.stop_reason = stop_reason_from(reason, STOP_REASONS, provider="Responses")
        logger.warning("Response incomplete: reason=%s, stop=%s", reason or "unstated", self.stop_reason)

    @on(ArgumentsDone)
    def _arguments_done(self, wire: ArgumentsDone) -> None:
        # Arguments carry whatever the user typed, and INFO is on in most deployments.
        call_id = self._call_id(wire.item_id)
        logger.info("Tool args complete: %s call_id=%s, %d chars", wire.name or "?", call_id, len(wire.arguments))
        logger.debug("Tool args for call_id=%s: %.200s", call_id, wire.arguments)

    # ── what ends the turn ───────────────────────────────────────────────────────────────────

    @on(Failed)
    def _failed(self, wire: Failed) -> None:
        message = wire.response.error.message or "Unknown error"
        logger.error("Response failed: %s", message)
        raise StreamError(f"Responses API error: {message}")

    @on(StreamFailure)
    def _errored(self, wire: StreamFailure) -> None:
        """Left unread the turn simply stopped and reported a normal finish."""
        message = wire.message or "Unknown error"
        logger.error("Stream error: %s (code=%s)", message, wire.code or "none")
        raise StreamError(f"Responses API error: {message}")

    # ── named, and deliberately not read ─────────────────────────────────────────────────────

    @on(
        "response.queued",
        "response.in_progress",
        "response.content_part.added",
        "response.output_text.done",
        "response.refusal.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.done",
    )
    def _expected(self, payload: Payload) -> None:
        """The envelope opening, and the whole of things already sent delta by delta.

        Named rather than forwarded because reading them would send the same content twice, and
        because this list is closed. It is the protocol's own bookkeeping, so it does not grow when
        a tool is added.
        """

    # ── everything else, forwarded rather than dropped ───────────────────────────────────────

    def unmatched(self, name: str, payload: Payload) -> Iterator[StreamEvent]:
        """Anything this reader does not interpret, passed on under the provider's own name.

        Almost all of it is the API running a tool on its own side. Each such tool has its own
        event family. That set depends on which tools exist and which the caller declared, not on
        the protocol. Named one by one, the list goes stale the day a tool is added. It also
        reports a new tool as news about the protocol when it is news about the tools.

        So nothing is listed and nothing is dropped. A consumer that wants the shell commands the
        model ran, or the searches it made, matches on ``kind``. Any other consumer ignores it.
        """
        yield ProviderEvent(
            provider=PROVIDER,
            kind=name,
            data=dict(payload),
            index=payload.number("output_index", default=-1) if "output_index" in payload else None,
        )

    # ── the turn, once it is over ────────────────────────────────────────────────────────────

    def _call_id(self, item_id: str) -> str:
        """The call id for this item, or the item id while no mapping for it has arrived."""
        return self.call_ids.get(item_id, item_id)

    def _count(self, usage: ResponseUsage) -> None:
        """The token counts, whose slices this API reports inside their totals. Nothing is added."""
        self.usage = Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.input_tokens_details.cached_tokens,
            cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        )

    def finished(self) -> IterationEnd:
        """What the turn added up to. The API sends no event that means this.

        A stream that ended without one of its terminal events did not finish. The connection was
        cut. Reported as ``end_turn``, a truncated answer is stored and returned as a whole one.
        So it is raised, the way a transport reports any other broken stream.
        """
        if self.stop_reason is None:
            raise StreamError("Responses stream ended without response.completed, response.incomplete or an error")
        stop = self.stop_reason
        logger.debug(
            "Stream complete: stop_reason=%s, input_tokens=%d, output_tokens=%d",
            stop,
            self.usage.input_tokens,
            self.usage.output_tokens,
        )
        if stop is StopReason.error:
            raise StreamError("Responses stopped with an error the reader did not name")
        return IterationEnd(iteration=0, stop_reason=stop, usage=self.usage)
