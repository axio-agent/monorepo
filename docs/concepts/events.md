# Stream Events

All agent I/O flows through typed **stream events**. The transport produces
events, the agent processes them, and your harness renders or forwards them.

## Event pipeline

```{mermaid}
flowchart TD
    T[Transport] -->|StreamEvent| A[Agent]
    A -->|StreamEvent| S[AgentStream]
    S -->|StreamEvent| C[Consumer]
```

The transport yields events as they arrive from the LLM. The agent forwards
every one of them verbatim, adds `ToolResult` and media events after
dispatching tool calls, then passes the stream through `AgentStream` to the
consumer.

Not every transport emits every event. Where an event depends on the provider,
the entry below says which ones produce it. A consumer that treats an optional
event as guaranteed breaks against the first provider that never sends it.

## Text and reasoning

All events are frozen dataclasses with `slots=True`:

`TextDelta`
: A chunk of text output from the model.
  ```python
  @dataclass(frozen=True, slots=True)
  class TextDelta:
      index: int
      delta: str
  ```

`ReasoningDelta`
: A chunk of reasoning/thinking output (for models that support it).
  Same shape as `TextDelta`. All four transports emit it.

`ReasoningSignature`
: The provider's opaque proof that the reasoning block at `index` is its own.
  ```python
  @dataclass(frozen=True, slots=True)
  class ReasoningSignature:
      index: int
      data: str
      redacted: bool = False
      id: str = ""
  ```

`TextSignature`
: The provider's opaque proof that the answer text at `index` is its own. Gemini signs the part it
  produced, which may be plain text rather than reasoning. Stored on the `TextBlock`, because a
  proof held as reasoning would be replayed on a part the provider never signed.
  ```python
  @dataclass(frozen=True, slots=True)
  class TextSignature:
      index: int
      data: str
  ```
  The agent attaches it to the `ReasoningBlock` it belongs to so the turn can be
  sent back unaltered. Never inspect, decode, re-encode or truncate `data`.
  Anthropic refuses a returned thinking block whose signature is missing or
  changed. Google reports a `MISSING_THOUGHT_SIGNATURE` finish reason for the
  same failure. `redacted=True` means the provider withheld the reasoning text.
  Only the proof travels. `id` is how the provider names the block, where it
  names them at all.

`Refusal`
: The model declined, or the provider blocked the turn.
  ```python
  @dataclass(frozen=True, slots=True)
  class Refusal:
      index: int
      text: str = ""
      category: str | None = None
      blocked_input: bool = False
      raw: dict[str, Any] = field(default_factory=dict)
  ```
  It is deliberately not a `TextDelta`. As ordinary assistant text, or as an
  empty turn that succeeded, a decline is indistinguishable from an answer.
  No consumer can act on it. `category` is the provider's own label,
  verbatim. The taxonomies do not overlap, so axio does not map between
  them. `blocked_input=True` means the prompt was rejected before anything
  was generated, so sending it again cannot succeed.

  The shape differs by provider. `text` is not always populated. Anthropic
  sends one event with the explanation and the category. The OpenAI chat and
  Responses paths stream the refusal fragment by fragment, one event per delta.
  Google's refusal is prompt-level. It carries `category` and
  `blocked_input=True` and **no text at all**. A Google candidate blocked
  mid-answer ends the turn with `StopReason.refusal` while emitting no
  `Refusal` event.

`Citation`
: A span of generated text attributed to a source.
  ```python
  @dataclass(frozen=True, slots=True)
  class Citation:
      index: int
      cited_text: str = ""
      title: str | None = None
      url: str | None = None
      source_id: str | None = None
      start: int | None = None
      end: int | None = None
      unit: Literal["char", "byte", "page", "block", "unknown"] = "unknown"
      raw: dict[str, Any] = field(default_factory=dict)
  ```
  `unit` exists because the providers count spans differently: OpenAI counts
  characters, Google counts bytes. Offsets carrying different units must
  therefore never be compared. `raw` keeps the provider's whole object for
  the fields that differ. Emitted by Anthropic and the OpenAI Responses
  path. Google forwards its grounding metadata as `ProviderEvent` instead.
  The agent does not store citations. They reach the consumer and nothing
  else.

## Generated media

`ImageOutput`
: An image the model generated inline, or one carried by a tool result.
  ```python
  @dataclass(frozen=True, slots=True)
  class ImageOutput:
      index: int
      data: bytes
      media_type: ImageMediaType
  ```

`AudioOutput` / `VideoOutput`
: The same shape with `AudioMediaType` / `VideoMediaType`. The agent re-emits
  these out of tool results so a harness can save the bytes to disk. The model
  sees the same data as `ImageBlock` / `AudioBlock` / `VideoBlock` inside the
  tool result itself.

## Tool calls

`ToolUseStart`
: Signals the beginning of a tool call.
  ```python
  @dataclass(frozen=True, slots=True)
  class ToolUseStart:
      index: int
      tool_use_id: ToolCallID
      name: ToolName
  ```

`ToolInputDelta`
: A partial JSON fragment of tool input, streamed incrementally.
  ```python
  @dataclass(frozen=True, slots=True)
  class ToolInputDelta:
      index: int
      tool_use_id: ToolCallID
      partial_json: str
  ```

`ToolFieldStart`
: Emitted when a new top-level field of a tool's JSON input has been identified.
  ```python
  @dataclass(frozen=True, slots=True)
  class ToolFieldStart:
      index: int
      tool_use_id: ToolCallID
      key: str
  ```

`ToolFieldDelta`
: A decoded chunk of the current field's value. String values have escape
  sequences resolved and surrounding quotes stripped. Other types are raw JSON.
  ```python
  @dataclass(frozen=True, slots=True)
  class ToolFieldDelta:
      index: int
      tool_use_id: ToolCallID
      key: str
      text: str
  ```

`ToolFieldEnd`
: Emitted when the current top-level field is fully received.
  ```python
  @dataclass(frozen=True, slots=True)
  class ToolFieldEnd:
      index: int
      tool_use_id: ToolCallID
      key: str
  ```

`ToolOutputDelta`
: Output from a tool while it is still running, for tools that stream.
  ```python
  @dataclass(frozen=True, slots=True)
  class ToolOutputDelta:
      tool_use_id: ToolCallID
      name: ToolName
      key: str
      delta: str
  ```
  The agent emits these only for a tool whose handler exposes a `.stream` async
  generator, which is what `Tool.supports_streaming` reports. A long shell
  command then reaches the harness line by line instead of arriving whole at the
  end. `key` is the tool's own channel name (`stdout`, `stderr`). The model
  still sees one finished result, built by `Tool.format_stream_result`. See
  {doc}`tools`.

`ToolResult`
: The result of executing a tool, added by the agent after dispatch.
  ```python
  @dataclass(frozen=True, slots=True)
  class ToolResult:
      tool_use_id: ToolCallID
      name: ToolName
      is_error: bool
      content: str = ""
      input: dict[str, Any] = field(default_factory=dict)
  ```

## Iteration lifecycle

`IterationStart`
: One provider request has begun.
  ```python
  @dataclass(frozen=True, slots=True)
  class IterationStart:
      iteration: int
      id: str | None = None
      model: str | None = None
  ```
  `model` is the model that actually served the turn, which need not be the one
  asked for. Server-side fallback, sticky routing and dated-snapshot resolution
  all substitute a different model at a different price. A cost lookup
  therefore keys off this field rather than off the request.

  `iteration` is always `0` today, from every transport. A transport cannot know
  the agent's iteration number, because `stream()` is not told it. The agent
  does not renumber the event on the way through. `IterationEnd.iteration` is
  `0` for the same reason. Count iterations yourself if you need them.

`BlockEnd`
: The content block at `index` is complete, which is the point at which the
  `ToolInputDelta` fragments accumulated for it are guaranteed to parse.
  ```python
  @dataclass(frozen=True, slots=True)
  class BlockEnd:
      index: int
  ```
  Emitted by Anthropic and by the OpenAI Responses path. The Google and chat
  completions paths emit nothing for it, so a consumer must not wait on it. The
  agent does not act on it either. It finalizes tool JSON at `IterationEnd`,
  which every transport sends.

`IterationEnd`
: Marks the end of one transport call. Carries the stop reason and token usage.
  ```python
  @dataclass(frozen=True, slots=True)
  class IterationEnd:
      iteration: int
      stop_reason: StopReason
      usage: Usage
  ```

`Error`
: Wraps an exception that occurred during streaming.
  ```python
  @dataclass(frozen=True, slots=True)
  class Error:
      exception: BaseException
  ```
  The annotation is `BaseException`, not `Exception`. A handler written against
  the narrower type does not type-check under the `mypy --strict` this repo
  requires.

`SessionEndEvent`
: Final event of the session. Carries the stop reason and cumulative token usage.
  ```python
  @dataclass(frozen=True, slots=True)
  class SessionEndEvent:
      stop_reason: StopReason
      total_usage: Usage
  ```
  See {doc}`agent` for which stop reasons end a run, and which one resumes it.

## Provider passthrough

`ProviderEvent`
: A provider payload axio does not model, forwarded verbatim.
  ```python
  @dataclass(frozen=True, slots=True)
  class ProviderEvent:
      provider: str
      kind: str
      data: dict[str, Any]
      index: int | None = None
  ```
  `data` is the provider's own JSON object exactly as it was parsed: no
  renaming, no coercion, no filtering. `kind` is the provider's own
  discriminator, verbatim, never a name axio invented. Matching on it is
  matching the vocabulary the provider publishes. `provider` is `"anthropic"`,
  `"openai"` or `"google"`. The Codex transport reads the stream through the
  shared Responses reader, so its events say `"openai"` too.

  A consumer that does not recognise `(provider, kind)` ignores it.

### Why the transports forward instead of naming

The readers built on `axio-sse` name only the events they interpret. Everything
else goes through `Reader.unmatched()`. The OpenAI and Anthropic readers
override it to forward those events as `ProviderEvent` rather than drop them.

The instinct - write a handler for every event in the provider's docs - is the
wrong one here. An endpoint that runs tools on its own side publishes one event
family per tool. That set therefore depends on which tools exist and which the
caller declared, not on the protocol. Listing it goes stale the day a tool is
added, and reports a new tool as news about the protocol. Nothing is listed and
nothing is dropped instead.

`strict=True` still raises `UnknownEvent` for any name no method claims, so a
test can hold a reader's interpreted set against the schema the provider
publishes without the reader carrying a list it cannot keep true.

Google and the chat completions path have no per-event discriminator,
because their streams are one shape. Both emit `ProviderEvent` inline for
the payload fields they do not model (grounding metadata, citation metadata,
logprobs, extra choices).

## Realtime events

These come from a duplex {ref}`realtime session <protocols>`, not from
`CompletionTransport.stream()`:

`AudioOutputDelta`
: A chunk of assistant audio. `data: bytes`, `media_type: str` defaulting to
  `"audio/pcm;rate=24000"`.

`TranscriptDelta`
: A live transcript chunk. `role: Literal["user", "assistant"]` says whether it
  is server-side STT of the microphone or the assistant's own speech.

`SpeechStarted` / `SpeechStopped`
: Server voice-activity detection saw the user start or stop speaking. Neither
  carries a field.

`TurnComplete`
: The assistant turn finished. `stop_reason: StopReason` and
  `usage: Usage | None = None`. A `tool_use` stop reason means pending tool
  calls should run before the next turn starts.

<!-- The blocks above mirror the source; the ones below are executed by markdown-pytest. -->

## Token usage

`IterationEnd.usage` and `SessionEndEvent.total_usage` are `Usage` values, which
support `+` to accumulate:

<!-- name: test_usage_add -->
```python
from axio import Usage

u1 = Usage(input_tokens=100, output_tokens=50)
u2 = Usage(input_tokens=200, output_tokens=80, reasoning_tokens=40)
total = u1 + u2

assert (total.input_tokens, total.output_tokens) == (300, 130)
assert total.reasoning_tokens == 40  # every slice adds, not just the two totals
```

`Usage` carries three more fields than the two headline numbers. They are
slices of them rather than extras. The full shape, the rule the transports
convert into, and what the registry can and cannot cost are under
{ref}`Token accounting <token-accounting>`.

## StreamEvent union

All events are combined into a single type alias:

<!-- not executed -->
```python
type StreamEvent = (
    ReasoningDelta
    | ReasoningSignature
    | TextDelta
    | TextSignature
    | Refusal
    | Citation
    | ImageOutput
    | AudioOutput
    | VideoOutput
    | ToolUseStart
    | ToolInputDelta
    | ToolFieldStart
    | ToolFieldDelta
    | ToolFieldEnd
    | ToolOutputDelta
    | ToolResult
    | BlockEnd
    | IterationStart
    | IterationEnd
    | Error
    | ProviderEvent
    | SessionEndEvent
    | AudioOutputDelta
    | TranscriptDelta
    | SpeechStarted
    | SpeechStopped
    | TurnComplete
)
```

`StreamEvent` is a union alias, not a base class. No event inherits from it.
`isinstance(event, StreamEvent)` raises. Dispatch with `match` or with
`isinstance` against a concrete event type:

<!-- not executed -->
```python
async for event in agent.run_stream("Hello", context):
    match event:
        case TextDelta(delta=text):
            print(text, end="", flush=True)
        case Refusal(text=text):
            print(f"\n[declined] {text}")
        case ToolResult(name=name, content=content):
            print(f"\n[Tool: {name}] {content}")
        case SessionEndEvent():
            print("\n--- Done ---")
```

Match the events you act on. Ignore the rest. A `case _` that raises is a
liability. The union grows. A consumer that treats an unknown event as a
failure breaks on the release that adds one.

Sixteen of the twenty-six are importable from the `axio` top level. The other
ten come from `axio.events`: `ReasoningDelta`, `Error`, `SessionEndEvent`,
`ToolOutputDelta`, the `ToolField*` trio, and `ImageOutput` / `AudioOutput` /
`VideoOutput`.

## AgentStream

`AgentStream` is a thin async-iterator wrapper around the event generator:

<!-- not executed -->
```python
class AgentStream:
    def __aiter__(self) -> AgentStream: ...
    async def __anext__(self) -> StreamEvent: ...
    async def aclose(self) -> None: ...
```

It also provides convenience methods:

`get_final_text() -> str`
: Consume the stream and return the concatenated `TextDelta` and `Refusal`
  text. Raises `StreamError` (from `axio.exceptions`) on `Error` events.

`get_session_end() -> SessionEndEvent`
: Consume the stream and return the final `SessionEndEvent`. Raises
  `StreamError` on an `Error` event, and also when the stream ended without a
  `SessionEndEvent` at all.

A refusal arrives instead of the answer, never beside it, so it is what the turn
said. `get_final_text()` collects it for that reason. `Agent.run()`, which is
`get_final_text()`, returns the decline rather than the empty string it used to:

<!-- name: test_refusal_is_the_answer -->
```python
import asyncio

from axio import Agent, IterationEnd, MemoryContextStore, Refusal, StopReason, Usage
from axio.testing import StubTransport

transport = StubTransport([[
    Refusal(index=0, text="I can't help with that.", category="policy"),
    IterationEnd(0, StopReason.refusal, Usage(12, 4)),
]])
agent = Agent(system="You are helpful.", transport=transport)

answer = asyncio.run(agent.run("...", MemoryContextStore()))
assert answer == "I can't help with that."
```

The stop reason still says which it was, so a caller that must tell a decline
from an answer reads `SessionEndEvent.stop_reason` rather than the text.

<!-- Examples from this section onwards are covered by doc tests. -->

## Streaming tool call arguments

`ToolInputDelta` events carry partial JSON fragments of tool arguments as the
LLM generates them. This enables real-time display of tool inputs - for
example, rendering file content character-by-character as it streams in,
similar to how Claude Code shows Edit tool diffs live.

```{image} /_static/stream_tool_args.svg
:alt: Streaming tool arguments demo
```

### ToolArgStream

`axio` ships a zero-dependency, O(1)-per-character streaming JSON parser that
converts `ToolInputDelta` chunks into structured `ToolField*` events:

<!-- name: test_tool_arg_stream_basic -->
```python
from axio.tool_args import ToolArgStream

stream = ToolArgStream("call_1", index=0)  # index defaults to 0
stream.feed('{"path":"/tmp/f')
# → [ToolFieldStart(0, "call_1", "path"),
#    ToolFieldDelta(0, "call_1", "path", "/tmp/f")]

stream.feed('oo.py"}')
# → [ToolFieldDelta(0, "call_1", "path", "oo.py"),
#    ToolFieldEnd(0, "call_1", "path")]
```

Top-level **string** fields are decoded (escape sequences resolved, quotes
stripped). All other top-level values (numbers, booleans, objects, arrays) are
emitted as raw JSON fragments via `ToolFieldDelta.text`.

Typical usage - create one `ToolArgStream` per tool call and forward its
output events downstream:

<!-- not executed -->
```python
from axio.tool_args import ToolArgStream
from axio.events import ToolFieldStart, ToolFieldDelta, ToolFieldEnd

parsers: dict[str, ToolArgStream] = {}

async for event in agent.run_stream(prompt, ctx):
    match event:
        case ToolUseStart(tool_use_id=tid, name=name, index=idx):
            parsers[tid] = ToolArgStream(tid, idx)
            print(f"▶ {name}")

        case ToolInputDelta(tool_use_id=tid, partial_json=pj):
            for field_event in parsers[tid].feed(pj):
                match field_event:
                    case ToolFieldStart(key=key):
                        print(f"\n  {key}: ", end="", flush=True)
                    case ToolFieldDelta(text=text):
                        print(text, end="", flush=True)
                    case ToolFieldEnd():
                        pass

        case ToolResult(tool_use_id=tid, content=content):
            print(f"\n  → {content}")
            parsers.pop(tid, None)
```

Nothing produces `ToolField*` events on its own. No transport emits them. The
agent does not run a `ToolArgStream` for you. The parser is opt-in, because it
costs a per-call parse that a harness which only renders finished tool inputs
does not need. The events are in the `StreamEvent` union so that the ones you
produce travel the same pipe as the rest.

See the full working example in
[examples/stream_tool_args.py](https://github.com/mosquito/axio-agent/blob/master/examples/stream_tool_args.py)
and {doc}`tool-args-streaming` for the parser's full API.
