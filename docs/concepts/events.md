# Stream Events

All agent I/O flows through typed **stream events**. The transport produces
events, the agent processes them, and consumers (like the TUI) render them.

## Event pipeline

```{mermaid}
flowchart LR
    T[Transport] -->|StreamEvent| A[Agent]
    A -->|StreamEvent| S[AgentStream]
    S -->|StreamEvent| C[Consumer]
```

The transport yields events as they arrive from the LLM. The agent enriches
the stream with `ToolResult` events after dispatching tool calls, then
forwards everything through `AgentStream` to the consumer.

## Event types

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
  Same shape as `TextDelta`.

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
  sequences resolved and surrounding quotes stripped; other types are raw JSON.
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

`SessionEndEvent`
: Final event of the session. Carries the stop reason and cumulative token usage.
  ```python
  @dataclass(frozen=True, slots=True)
  class SessionEndEvent:
      stop_reason: StopReason
      total_usage: Usage
  ```

`Usage`
: Token counts for one iteration or an entire session. Supports `+` to
  accumulate totals across multiple iterations:
  ```python
  from axio.types import Usage
  u1 = Usage(input_tokens=100, output_tokens=50)
  u2 = Usage(input_tokens=200, output_tokens=80)
  total = u1 + u2  # Usage(input_tokens=300, output_tokens=130)
  ```

## StreamEvent union

All events are combined into a single type alias:

```python
type StreamEvent = (
    ReasoningDelta | TextDelta
    | ToolUseStart | ToolInputDelta
    | ToolFieldStart | ToolFieldDelta | ToolFieldEnd
    | ToolResult | IterationEnd | Error | SessionEndEvent
)
```

Use `match` or `isinstance` to dispatch on event types:

```python
async for event in agent.run_stream("Hello", context):
    match event:
        case TextDelta(delta=text):
            print(text, end="", flush=True)
        case ToolResult(name=name, content=content):
            print(f"\n[Tool: {name}] {content}")
        case SessionEndEvent():
            print("\n--- Done ---")
```

## AgentStream

`AgentStream` is a thin async-iterator wrapper around the event generator:

```python
class AgentStream:
    def __aiter__(self) -> AgentStream: ...
    async def __anext__(self) -> StreamEvent: ...
    async def aclose(self) -> None: ...
```

It also provides convenience methods:

`get_final_text() -> str`
: Consume the stream and return only the concatenated text deltas.
  Raises `StreamError` (from `axio.exceptions`) on `Error` events.

`get_session_end() -> SessionEndEvent`
: Consume the stream and return the final `SessionEndEvent`.

<!-- Examples from this section onwards are covered by doc tests. -->

## Streaming tool call arguments

`ToolInputDelta` events carry partial JSON fragments of tool arguments as the
LLM generates them. This enables real-time display of tool inputs — for
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

Typical usage — create one `ToolArgStream` per tool call and forward its
output events downstream:

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

The `ToolField*` events are also emitted directly by the agent stream when a
transport produces `ToolInputDelta` events — you can match them without
instantiating `ToolArgStream` yourself if you prefer to rely on the agent-level
integration (see below).

See the full working example in
[examples/stream_tool_args.py](https://github.com/mosquito/axio-agent/blob/master/examples/stream_tool_args.py).
