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

`SessionEndEvent`
: Final event of the session. Carries the stop reason and cumulative token usage.
  ```python
  @dataclass(frozen=True, slots=True)
  class SessionEndEvent:
      stop_reason: StopReason
      total_usage: Usage
  ```

## StreamEvent union

All events are combined into a single type alias:

```python
type StreamEvent = (
    ReasoningDelta | TextDelta | ToolUseStart | ToolInputDelta
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
  Raises `StreamError` on `Error` events.

`get_session_end() -> SessionEndEvent`
: Consume the stream and return the final `SessionEndEvent`.
