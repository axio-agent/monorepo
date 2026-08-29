# Writing Transports

A transport turns one provider's stream into axio's events. It is the only place in the stack that
knows a provider exists. Everything a provider does differently has to be resolved here: the wire
format, the token counts, the names for stopping, and what the provider needs sent back.

Four transports ship. Each does the same five things:

1. reads `text/event-stream` through [`axio-sse`](../api/sse.md), never by hand,
2. declares the payload shapes it reads as `Wire` classes,
3. converts the provider's token counts into axio's inclusive-total rule,
4. maps every stop reason the provider publishes,
5. replays reasoning where the provider requires it.

Each has its own section below.

## The protocol

`CompletionTransport` is a `@runtime_checkable` Protocol with one method. Import it rather than
restating it. The real signature is generic in `Tool`, and a copy drifts:

<!-- name: test_transport_protocol -->
```python
from axio import CompletionTransport
from axio.testing import StubTransport

assert isinstance(StubTransport(), CompletionTransport)
```

~~~python
def stream(
    self,
    messages: list[Message],
    tools: list[Tool[Any]],
    system: str,
) -> AsyncIterator[StreamEvent]: ...
~~~

`messages` is the whole conversation, including the user turn that started this run. The agent
appends that turn before reading the history. `tools` is what the selector left. It is empty when
the model has no `Capability.tool_use`. `system` is the system prompt.

## Minimal implementation

<!-- name: test_echo_transport -->
```python
import asyncio
from collections.abc import AsyncIterator
from typing import Any
from axio import IterationStart, StopReason, StreamEvent, TextDelta, Tool, Usage
from axio.events import IterationEnd
from axio.blocks import TextBlock
from axio.messages import Message


class EchoTransport:
    """Echoes the last user message. A transport with no provider behind it."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        last_text = ""
        for msg in reversed(messages):
            if msg.role == "user":
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        last_text = block.text
                        break
                break

        # First, so a consumer knows which model answered before any content arrives.
        yield IterationStart(iteration=0, id="echo-1", model="echo")
        yield TextDelta(index=0, delta=f"Echo: {last_text}")
        # Exactly once, and last: the agent reads the stop reason and the usage off it.
        yield IterationEnd(
            iteration=0,
            stop_reason=StopReason.end_turn,
            usage=Usage(input_tokens=0, output_tokens=0),
        )


async def main() -> None:
    transport = EchoTransport()
    msgs = [Message(role="user", content=[TextBlock(text="ping")])]
    events = [e async for e in transport.stream(msgs, [], "")]
    assert isinstance(events[1], TextDelta)
    assert events[1].delta == "Echo: ping"
    assert isinstance(events[2], IterationEnd)
    assert events[2].stop_reason == StopReason.end_turn

asyncio.run(main())
```

Here `stream` is an `async def` with `yield`, which makes it an async generator. The production
transports instead declare `stream` as a plain `def` returning a call to a separate
`async def _do_stream(...)`. Both satisfy the protocol: both return an `AsyncIterator[StreamEvent]`.

## What you do not write yourself

Four helpers in `axio` already say what every transport needs, and a fifth checks the result:

| | |
|---|---|
| `axio.retry.is_retryable(status)` | Which HTTP statuses are worth another attempt |
| `axio.retry.retry_delay(resp, attempt, base=...)` | How long to wait, honouring `Retry-After` as seconds or as a date |
| `axio.types.stop_reason_from(raw, table, provider=...)` | A provider's stop value, raising `StreamError` on one the table does not name |
| `axio.schema.strip_title(schema)` | A tool schema without its `title` keywords |
| `axio.testing.assert_stream_contract(events)` | What every `stream()` must produce, for your tests |

Written by hand instead, these drifted. One transport retried three statuses where the others
retried any server fault, and ignored `Retry-After` entirely.

## The event contract

One `IterationStart` first, content in the order the provider sent it, exactly one `IterationEnd`
last. Everything between is optional and provider-dependent:

| Event | Meaning | Emitted by |
|---|---|---|
| `IterationStart` | The request began. `model` is the model that *served* the turn. | all four |
| `TextDelta` | A chunk of the answer. | all four |
| `ReasoningDelta` | A chunk of reasoning the caller is not being answered with. | all four |
| `ReasoningSignature` | Opaque proof for the reasoning just sent. | Anthropic, Google, Responses |
| `TextSignature` | Opaque proof for the answer text just sent. | Google |
| `ToolUseStart` / `ToolInputDelta` | A tool call and its JSON, fragment by fragment. | all four |
| `BlockEnd` | The block at `index` is complete, so its fragments now parse. | Anthropic, Responses |
| `Refusal` | The model declined, or the provider blocked the turn. | all four |
| `Citation` | A span of text attributed to a source. | Anthropic, Responses |
| `ImageOutput` / `AudioOutput` / `VideoOutput` | Generated media inside the turn. | Google |
| `ProviderEvent` | Anything the provider sent that axio does not model, forwarded for a caller to watch. | all four |
| `ProviderOutput` | Output of a tool the provider ran itself, which the next request has to carry back. | Anthropic, Google, Responses |
| `IterationEnd` | The turn is over: stop reason and token counts. | all four |

Do not emit an event a provider does not send. A consumer that needs `BlockEnd` and matches on it
for Anthropic gets nothing on Google. The docs say so, rather than letting a transport invent one.

### Tool calls

<!-- name: test_tool_call_events -->
```python
import asyncio
from axio import StopReason, ToolInputDelta, ToolUseStart, Usage
from axio.events import IterationEnd


async def example_tool_call_stream():
    yield ToolUseStart(index=0, tool_use_id="call_abc", name="my_tool")
    yield ToolInputDelta(index=0, tool_use_id="call_abc", partial_json='{"arg": "value"}')
    yield IterationEnd(iteration=0, stop_reason=StopReason.tool_use, usage=Usage(10, 5))


async def main():
    events = [e async for e in example_tool_call_stream()]
    assert len(events) == 3

asyncio.run(main())
```

The agent assembles the `ToolInputDelta` fragments into complete JSON at `IterationEnd`. It answers
a call whose JSON never parsed with a retry message, rather than executing it. Send as many
fragments as the API streams.

For parallel calls give each one its own `index`. The id map from index to `tool_use_id` is the
transport's job, because most APIs put the id only in the event that opens the block:

<!-- name: test_multiple_tool_calls -->
```python
import asyncio
from axio import StopReason, ToolInputDelta, ToolUseStart, Usage
from axio.events import IterationEnd


async def example_parallel_stream():
    yield ToolUseStart(index=0, tool_use_id="call_1", name="tool_a")
    yield ToolUseStart(index=1, tool_use_id="call_2", name="tool_b")
    yield ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"x": 1}')
    yield ToolInputDelta(index=1, tool_use_id="call_2", partial_json='{"y": 2}')
    yield IterationEnd(iteration=0, stop_reason=StopReason.tool_use, usage=Usage(10, 5))


async def main():
    events = [e async for e in example_parallel_stream()]
    assert len(events) == 5

asyncio.run(main())
```

## Reading the stream

Do not parse SSE. `axio-sse` reads the format, and no transport in this repository contains a line
splitter any more.

`Decoder` is the format as a state machine. `decode(chunk, final=False)` takes bytes or text cut
anywhere, and returns the events those bytes completed. It takes chunks and never lines, and that is
not a preference. `aiohttp`'s `readuntil` raises `LineTooLong` past 131072 bytes. `LineTooLong` is
not a `ClientError`, so it escapes the retry loop. One large reasoning event ended a turn with no
answer at all. Feed it `resp.content.iter_any()`.

### A stream with one payload shape

Where every event carries the same JSON object — chat completions, Gemini's
`streamGenerateContent` — `payloads()` is the whole reading layer. `until` names the non-JSON
sentinel that closes the stream, so it never reaches a caller:

<!-- name: test_sse_payloads -->
```python
import asyncio
from collections.abc import AsyncIterator
from axio_sse import payloads


async def chunks() -> AsyncIterator[bytes]:
    # Cut mid-object on purpose: the decoder holds the half and completes it with the next chunk.
    yield b'data: {"delta": "Hel"}\n\ndata: {"del'
    yield b'ta": "lo"}\n\ndata: [DONE]\n\n'


async def main() -> None:
    read = [p.string("delta") async for p in payloads(chunks(), until="[DONE]")]
    assert read == ["Hel", "lo"]

asyncio.run(main())
```

`Payload` is a `dict` with four readers that take a path — `string()`, `number()`, `obj()`,
`objs()`. Each returns its default where any step is missing, null, or the wrong type, which is
exactly what an optional provider field is.

### Declaring what a payload holds

Read fields through a `Wire` shape rather than out of the dict. Every field is read by its declared
name and type. A key you misspelled is therefore a type error where you use it, instead of a default
quietly standing in for a value:

<!-- name: test_sse_wire -->
```python
from dataclasses import dataclass
from axio_sse import Payload, Wire


@dataclass(frozen=True, slots=True)
class Chunk(Wire):
    model: str = ""
    delta: str = ""
    finish_reason: str = ""


read = Chunk.read(Payload({"model": "m-1", "delta": "hi", "finish_reason": None, "extra": 1}))
# A null field and an unknown key are both normal on the wire: the null takes the declared
# default, the extra key is ignored, and neither loses the fields beside it.
assert read == Chunk(model="m-1", delta="hi", finish_reason="")
```

A nested object is another `Wire`; a list of them is `list[ThatWire]`. A shape declared with no
`name=` is only ever nested. Declare `raw: Payload` on a shape that varies too much to write out. A
citation arrives under five shapes that each name their span differently. The whole payload then
travels beside the fields worth declaring.

### A stream that names its events

Where the stream says what each event is, subclass `Reader` and write one `@on(...)` method per
event. `by=` on the class line names the payload key holding the name, or `EVENT_NAME` for the
format's own `event:` field. Anthropic names its events in the `event:` field, so its reader says
`by=EVENT_NAME`. The Responses API puts the name in the payload's `type` key, which is the default:

<!-- name: test_sse_reader -->
```python
import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from axio.events import ProviderEvent, StreamEvent, TextDelta
from axio_sse import EVENT_NAME, Payload, Reader, Wire, on


@dataclass(frozen=True, slots=True)
class TextChunk(Wire, name="text.delta"):
    text: str = ""


class Reply(Reader[StreamEvent], by=EVENT_NAME):
    @on(TextChunk)
    def _text(self, wire: TextChunk) -> Iterator[StreamEvent]:
        yield TextDelta(index=0, delta=wire.text)

    @on("ping")
    def _quiet(self, payload: Payload) -> None:
        """Arrives every turn and carries nothing. Named so strict fires only on something new."""

    def unmatched(self, name: str, payload: Payload) -> Iterator[StreamEvent]:
        yield ProviderEvent(provider="demo", kind=name, data=dict(payload))


async def stream() -> AsyncIterator[bytes]:
    yield b'event: text.delta\ndata: {"text": "Hi"}\n\n'
    yield b'event: ping\ndata: {}\n\n'
    yield b'event: web_search.started\ndata: {"query": "axio"}\n\n'


async def main() -> None:
    made = [e async for e in Reply().over(stream())]
    # The ping carried nothing, and the event nobody named was forwarded rather than dropped.
    assert made == [
        TextDelta(index=0, delta="Hi"),
        ProviderEvent(provider="demo", kind="web_search.started", data={"query": "axio"}),
    ]

asyncio.run(main())

# What this reader claims, for a test to hold against the provider's own list.
assert Reply.names() == frozenset({"text.delta", "ping"})
```

Give `@on` a `Wire` shape and the method is handed that shape. Give it names and the method is
handed the `Payload` itself. That is what a method that only forwards or only drops an event wants.
A handler returns what the event became, or `None` where it only moved the reader's state. That pair
is `Handled[T]`, `Iterable[T] | None` — one rule for none, one and many. One instance reads one
response, so the turn's running totals and id maps live on `self`.

### Name only what you interpret

This is the part that is not obvious, and getting it wrong is the standard mistake.

A `Reader` names the events it interprets and nothing else. Everything else goes through
`unmatched()`, which returns nothing by default. The OpenAI and Anthropic readers override it to
forward as `ProviderEvent` under the provider's own name.

The instinct is to add a handler for every event in the provider's documentation. That list cannot
be kept true. An endpoint that runs tools on its own side publishes one event family per tool. That
set therefore depends on which tools exist and which the caller declared, not on the protocol. A
list of it goes stale the day a tool is added. A new tool then reads as news about the protocol when
it is news about the tools. Forwarding costs nothing. A consumer that wants the searches the model
ran matches on `kind`, and one that does not ignores it.

`strict=True` is the counterweight. It refuses anything unnamed even when the reader forwards, so a
test can hold the interpreted set against the schema the provider publishes:

<!-- name: test_sse_strict -->
```python
import pytest
from collections.abc import Iterator
from axio.events import ProviderEvent, StreamEvent
from axio_sse import EVENT_NAME, Decoder, Payload, Reader, UnknownEvent


class Quiet(Reader[StreamEvent], by=EVENT_NAME):
    """Interprets nothing and forwards everything — the policy above, in miniature."""

    def unmatched(self, name: str, payload: Payload) -> Iterator[StreamEvent]:
        yield ProviderEvent(provider="demo", kind=name, data=dict(payload))


[event] = Decoder().decode(b'event: web_search.started\ndata: {}\n\n', final=True)
assert len(Quiet().read(event)) == 1      # forwarded through unmatched()
with pytest.raises(UnknownEvent):
    Quiet().read(event, strict=True)      # and still refused under strict
```

The real test to write is `Responses.names() <= PUBLISHED_EVENTS`: every name the reader claims is
one the schema publishes. The other direction is deliberately not asserted, because the reader is
meant to be a subset.

`unknown()` is the same policy for a second discriminator nested inside one event — the delta type
inside a content block. Call it from a handler's `case _` so a nested name nobody read fails the
same strict replay that a new event fails, instead of disappearing.

## Token counts

`Usage.input_tokens` and `Usage.output_tokens` are always inclusive grand totals. Every other field
is a disjoint slice of one of them:

```
cache_read_tokens + cache_write_tokens  <=  input_tokens
reasoning_tokens                        <=  output_tokens
```

Providers disagree about whether their own headline number already contains the slices. They
disagree in opposite directions. Converting into the rule is the transport's job, so that nothing
downstream has to know which provider answered. Pass the numbers through unchanged and the error
lands in the context-window accounting, because the agent feeds `input_tokens` and `output_tokens`
straight into `context.add_context_tokens(...)`.

| Provider | What it reports | What the transport does |
|---|---|---|
| Anthropic | `input_tokens` counts only what follows the last cache breakpoint; thinking is already inside `output_tokens`. | Adds `cache_read_input_tokens` and `cache_creation_input_tokens` back into the input total. |
| Google | `cachedContentTokenCount` is inside `promptTokenCount`; `toolUsePromptTokenCount` and `thoughtsTokenCount` are outside their totals. | Adds tool-use prompt tokens to the input and thinking to the output. |
| Responses | Both slices arrive inside their totals. | Adds nothing. |
| Chat completions | Both slices arrive inside their totals. | Adds nothing. |

<!-- name: test_usage_conversion -->
```python
from axio import Usage

# What Anthropic reported for a turn served almost entirely from cache. Thinking is nested,
# which is why it is read through a shape rather than off the top level.
reported = {
    "input_tokens": 12,
    "cache_read_input_tokens": 4000,
    "cache_creation_input_tokens": 100,
    "output_tokens": 300,
    "output_tokens_details": {"thinking_tokens": 120},
}

usage = Usage(
    # Added back: read as reported, a cached 100k prompt is a handful of tokens.
    input_tokens=(
        reported["input_tokens"]
        + reported["cache_read_input_tokens"]
        + reported["cache_creation_input_tokens"]
    ),
    # Thinking is already inside this one, so nothing is added.
    output_tokens=reported["output_tokens"],
    cache_read_tokens=reported["cache_read_input_tokens"],
    cache_write_tokens=reported["cache_creation_input_tokens"],
    reasoning_tokens=reported["output_tokens_details"]["thinking_tokens"],
)

assert usage.uncached_input_tokens == 12
assert usage.answer_tokens == 180
assert usage.total_tokens == 4412
```

`cache_read_tokens`, `cache_write_tokens` and `reasoning_tokens` are keyword-only and default to
`0`, so a transport reports only the slices its provider breaks out. `Usage.__add__` sums all five,
which is how `SessionEndEvent.total_usage` stays a whole.

Where the provider publishes its own total, check it and warn on a mismatch rather than trusting
either number silently. That check is what catches the day the provider changes the rule.

These are counts, never money. A cached token and a written one bill at different multipliers, so a
cost line multiplies the slices by its own per-model rates. A zero slice means the provider billed
none of it, or reported no breakdown at all. axio cannot tell those apart.

## Stop reasons

Map every reason the provider publishes, in one table. Fall back to `StopReason.error` with a
warning. A reason left out of the map is not a finished answer:

<!-- name: test_stop_reason_map -->
```python
from axio import StopReason

STOP_REASONS: dict[str, StopReason] = {
    "stop": StopReason.end_turn,
    "tool_calls": StopReason.tool_use,
    "length": StopReason.max_tokens,
    "content_filter": StopReason.refusal,
}

assert STOP_REASONS.get("stop") is StopReason.end_turn
# A reason nobody mapped ends the run, rather than passing for a finished answer.
assert STOP_REASONS.get("something_new", StopReason.error) is StopReason.error
```

`StopReason` has ten members. Four are older: `end_turn`, `tool_use`, `max_tokens`, `error`. Six
are not. A transport written against the old four maps a decline onto `error` or, worse, onto
`end_turn`:

- `refusal` — the model declined, or the provider blocked the turn. It is terminal and
  deliberately not an error. Reported as an error, it leaves the caller unable to tell a decline
  from a broken connection. The caller then retries something that can never work.
- `pause_turn` — the provider stopped its own server-side tool loop. It expects the assistant
  content back so it can finish. This is the one reason that does not end the run. The agent appends
  the turn and goes round again, which *is* the resume. Anthropic publishes it.
- `context_window_exceeded` — the conversation outgrew the window. Truncated, like `max_tokens`.
  Only Anthropic publishes a reason that maps here.
- `cancelled` — the caller or the provider stopped the turn early. Only the Responses map has one.
- `unknown` — the provider said something the map does not name. Terminal, and it vouches for
  nothing. Use `stop_reason_from()`, which returns it and keeps the provider's own word in
  `IterationEnd.raw`. Every other answer claims something the provider did not say.
- `repetition` — Axio stopped the turn because the model was repeating itself. The agent emits it;
  no transport maps a provider reason onto it.

The agent loop matches on the stop reason with a `case _` wildcard, so a reason nothing handled ends
the run with an `Error`. Without the wildcard the loop would simply run again, re-prompting with
unchanged history until `max_iterations`. Every one of those turns is paid for.

Two judgement calls from the Google map are worth copying, because they are what the map is for.
`MALFORMED_FUNCTION_CALL` is read as `tool_use`, because prompting again is the recovery and the
next answer may parse. `MISSING_THOUGHT_SIGNATURE` is read as `error`, because prompting again is
not.

## What the provider ran itself

An endpoint that runs its own tools answers with items this vocabulary has no shape for: a web
search, a file search, code it executed, and whatever it adds next. All four APIs here are
stateless. The application holds the conversation and sends it whole on every request, so an item
the transport does not send back is one the model answered from and can no longer see.

Emit `ProviderOutput` for each such item, carrying the provider's own object unaltered. The agent
stores it as a `ProviderBlock`. Do not model the item: a shape declared here would not hold a type
the provider publishes next month, and the point is that the item travels unread.

`ProviderEvent` is the other half and is not a substitute. It is what a caller watches; it is never
stored, so an item forwarded only that way is gone by the next request.

On the way back, replay `data` verbatim, guarded by `axio.blocks.replayable(block, PROVIDER)`. An
item another protocol produced means nothing here and must be left out.

Wait until the item is complete. Anthropic streams the arguments of a `server_tool_use` in
`input_json_delta` after the block opens, so the object at `content_block_start` still has an empty
`input`. The Responses API sends the finished item in `response.output_item.done`, and Gemini sends
each part whole.

## Reasoning replay

A provider that reasons will not accept its own reasoning back altered. This is the defect most
easily reproduced in a new transport. It does not fail on the turn that drops the signature — it
fails on the next one.

Send `ReasoningSignature` for the proof and `ReasoningDelta` for the text. Emit the signature
*after* the reasoning it signs, never before. The agent attaches a proof to the block it has just
built. It refuses to extend a block that is already signed. A signature sent first therefore turns
one part into two blocks: a proof of nothing, and reasoning left unsigned.

What the three providers need sent back:

| Provider | Replayed as |
|---|---|
| Anthropic | A `thinking` block with its `signature`, or `redacted_thinking` carrying the signature and no text. An unsigned block is dropped rather than sent, because the API would refuse the request. |
| Google | A part with `thought: true` and its `thoughtSignature`. A signature that arrived on a function-call part goes back on *that* part — sent as a thought part with no text, the call comes back `MISSING_THOUGHT_SIGNATURE`. Parallel calls take their signatures in arrival order. |
| Responses | A `reasoning` item with `id` and `encrypted_content`, which only arrives at all because the request asked for it with `include: ["reasoning.encrypted_content"]` and stored nothing provider-side. |

Never inspect, decode, re-encode or truncate a signature. Anthropic refuses a changed one. Google
answers `MISSING_THOUGHT_SIGNATURE`, which its transport maps to `StopReason.error`.

Name the protocol on every proof you emit. `ReasoningSignature`, `TextSignature` and `ToolUseStart`
all take `provider=`, and the agent stores it beside the signature. The three protocols read the
same stored field by their own rules — a thinking signature, a `thoughtSignature`, an
`encrypted_content` — so a session that changes transport would otherwise hand one provider's
opaque data to another. Use the name your `ProviderEvent`s already use.

Read it back through `axio.blocks.proof(block, PROVIDER)` rather than off `block.signature`. It
returns an empty string where the block was signed by someone else, and the block's own signature
where the provider matches or where none was recorded.

Index a signature by the block it proves. That is what the agent joins on. Two signatures under one
index are the two halves of one proof, and are concatenated. Two under different indices are two
proofs, and stay apart. Anthropic uses the content-block index, the Responses reader uses the output
index, and Gemini uses the part's own position. Fixed at zero, two parallel signed calls looked like
one proof cut in half. They were stored as a signature matching neither.

The agent does the accumulating. `ReasoningDelta` builds a `ReasoningBlock`, `ReasoningSignature`
signs it, and the block is stored in the turn. A transport tests the round trip by recording what it
was handed on the second call:

<!-- name: test_reasoning_replay -->
```python
import asyncio
from collections.abc import AsyncIterator
from typing import Any
from axio import Agent, MemoryContextStore, StopReason, StreamEvent, TextDelta, Tool, Usage
from axio.blocks import ReasoningBlock
from axio.events import IterationEnd, ReasoningDelta, ReasoningSignature, ToolInputDelta, ToolUseStart
from axio.messages import Message


class Recorder:
    """A transport that keeps every history it was given, so a replay can be asserted on."""

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self.responses = responses
        self.sent: list[list[Message]] = []

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.sent.append(list(messages))
        return self._replay(self.responses[min(len(self.sent) - 1, len(self.responses) - 1)])

    async def _replay(self, events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event


async def clock(msg: str) -> str:
    """Answer with the message it was given."""
    return msg


async def main() -> None:
    transport = Recorder([
        [
            ReasoningDelta(index=0, delta="The user asked for the time."),
            # After the reasoning it signs, never before.
            ReasoningSignature(index=0, signature="opaque-proof"),
            ToolUseStart(index=0, tool_use_id="call_1", name="clock"),
            ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"msg": "now"}'),
            IterationEnd(iteration=0, stop_reason=StopReason.tool_use, usage=Usage(10, 8, reasoning_tokens=6)),
        ],
        [
            TextDelta(index=0, delta="now"),
            IterationEnd(iteration=0, stop_reason=StopReason.end_turn, usage=Usage(20, 2)),
        ],
    ])
    agent = Agent(system="", transport=transport, tools=[Tool(name="clock", handler=clock)])
    await agent.run("what time is it", MemoryContextStore())

    replayed = [b for m in transport.sent[1] for b in m.content if isinstance(b, ReasoningBlock)]
    assert replayed == [ReasoningBlock(text="The user asked for the time.", signature="opaque-proof")]

asyncio.run(main())
```

## Refusals

A refusal is not a `TextDelta`. As ordinary assistant text a decline is indistinguishable from an
answer. As an empty successful turn it is indistinguishable from nothing at all. Either way no
consumer can act on it.

Emit `Refusal` and finish with `StopReason.refusal`. `AgentStream.get_final_text()` collects the
refusal text, so `run()` returns the decline rather than the empty string it used to.

Set `spoken=False` where the text is your own account of the decline rather than the model's
words. OpenAI and the Responses API stream a refusal as output content, so theirs is spoken.
Anthropic sends `stop_details.explanation` beside a response with no content at all, and Gemini
rejects the prompt and generates nothing. The agent stores either kind as the turn's text — a
stored turn with no content is refused by the next request — and a consumer that renders the two
differently reads this flag:

<!-- name: test_refusal_reaches_caller -->
```python
import asyncio
from axio import Agent, MemoryContextStore, Refusal, StopReason, Usage
from axio.events import IterationEnd, SessionEndEvent
from axio.testing import StubTransport


async def main() -> None:
    transport = StubTransport([[
        Refusal(index=0, text="I can't help with that.", category="policy"),
        IterationEnd(iteration=0, stop_reason=StopReason.refusal, usage=Usage(12, 5)),
    ]])
    agent = Agent(system="", transport=transport)

    assert await agent.run("...", MemoryContextStore()) == "I can't help with that."

    ends = [
        e async for e in agent.run_stream("...", MemoryContextStore())
        if isinstance(e, SessionEndEvent)
    ]
    # Terminal, and not an error: the same prompt sent again is declined again.
    assert ends[0].stop_reason is StopReason.refusal

asyncio.run(main())
```

The shapes differ and the docs should not flatten them. Anthropic sends one whole refusal with an
explanation and a category. Chat completions and the Responses API stream it fragment by fragment,
one event per delta. Google's is prompt-level. It carries `category` and `blocked_input=True` and no
text at all, so `run()` still returns `""` there. Set `blocked_input` so a consumer can tell a
rejected prompt from a declined answer. Google also maps a candidate that finished on `SAFETY` or
`RECITATION` to `StopReason.refusal`, while emitting no `Refusal` event. That is why the stop reason
and the event are read separately.

## Errors

Raise `StreamError`. Every reference transport does. The agent loop catches it, yields `Error` and
ends the session.

Do not finish with `IterationEnd(stop_reason=StopReason.error)` instead. That reason falls into the
agent's `case _` wildcard, which produces `Error(RuntimeError("Transport stopped with: error"))`.
The run is terminated just the same, and reported with a message that names nothing.
`StopReason.error` is what an *unmapped* reason becomes, and the wildcard is what keeps that
terminal. It is not how a transport reports a failure it can describe.

For retryable statuses (429, 5xx) retry inside the transport. Honour `Retry-After` when the response
carries one, and fall back to exponential backoff when it does not.

## What the agent reads off a transport

The protocol is one method, but the agent looks for three optional attributes with `getattr`. A
transport that does not have them behaves exactly as before:

| Attribute | Read for |
|---|---|
| `model` | Its `capabilities`. Without `Capability.tool_use` the agent sends no tools at all. |
| `last_usage` | The partial turn's counts, added to the total when repetition truncation cuts a stream short. |
| `nudge_on_media_tool_result` | Whether to append a short user message after a tool returns media. Only `GoogleTransport` sets it, because Gemini stops generating after media arrives beside a `functionResponse`. |

Two things a transport is deliberately not asked for:

- **The iteration number.** `stream()` is not given one and cannot know it. All four transports pass
  `iteration=0` on `IterationStart` and `IterationEnd`. The agent forwards the event unchanged. Pass
  `0` and be consistent with the rest.
- **A retry policy in the protocol.** Retries, backoff and connection reuse are the transport's own.
  The agent has no opinion and no hook.

## One transport, two endpoints

`OpenAITransport` carries `api: Literal["responses", "chat"] = "responses"` and routes on it.
This is worth knowing before writing a fifth transport, because the reason is a provider constraint
rather than a preference. `/v1/chat/completions` refuses function tools beside any reasoning effort
other than `"none"`. A reasoning model carrying tools then fails with a 400 naming a parameter the
caller never sent. The chat path therefore sets `reasoning_effort: "none"` itself and warns. The
cost is real: the model is paid for as a reasoning model and asked not to reason. `/v1/responses`
takes both. `OpenAICompatibleTransport`, `NebiusTransport` and `OpenRouterTransport`
all say `"chat"`, because compatible servers rarely implement `/v1/responses`.

`extra_params` merges `tools` rather than replacing them. A caller adding a hosted tool — web
search, code interpreter — would otherwise take away the function declarations the agent needs
dispatched. The turn would then read as the model simply choosing to call nothing. A declaration
whose name matches one already there wins, because the caller said it last.

## Reference implementations

| Package | Endpoint | Reads with |
|---|---|---|
| `axio-transport-anthropic` | Messages API, direct and Vertex | `axio_sse.Reader`, `by=EVENT_NAME` |
| `axio-transport-openai` | `/v1/responses` and `/v1/chat/completions` | `axio_responses.Responses`, and `payloads()` with `Wire` shapes |
| `axio-transport-google` | `streamGenerateContent`, Developer API and Vertex | `payloads()` with `Wire` shapes |
| `axio-transport-codex` | The ChatGPT backend, Responses API | `axio_responses.Responses` |

Only two of them subclass `Reader`. The chat-completions and Gemini streams name no event, because
every payload is one shape. Those two read `payloads()` and emit `ProviderEvent` inline for what
they do not interpret.

Both halves of the Responses API live in `axio-responses` rather than in a transport, because two
transports speak it. `convert_messages` and `convert_tools` build the request, and `Responses` reads
the stream.
