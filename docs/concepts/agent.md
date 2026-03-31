# Agent & the Agentic Loop

The {class}`Agent` is the central orchestrator. It connects a transport, a set of
tools, and a context store into a single loop that streams LLM responses and
dispatches tool calls until the model signals it is done.

## The Agent dataclass

```python
@dataclass(slots=True)
class Agent:
    system: str
    tools: list[Tool]
    transport: CompletionTransport
    max_iterations: int = 50
```

`system`
: The system prompt sent with every request.

`tools`
: Available tools. The agent searches this list by name when the model
  issues a tool call.

`transport`
: Any object satisfying the {ref}`CompletionTransport <protocols>` protocol.

`max_iterations`
: Safety limit preventing runaway loops. The agent emits a
  `SessionEndEvent` with an error if this limit is reached.

## How the loop works

```{mermaid}
flowchart TD
    A[User message] --> B[Append to context]
    B --> C[Get history from context]
    C --> D[Stream from transport]
    D --> E{Tool calls?}
    E -- Yes --> F[Dispatch tools concurrently]
    F --> G[Append results to context]
    G --> C
    E -- No --> H{Stop reason?}
    H -- end_turn --> I[SessionEndEvent]
    H -- max_tokens / error --> J[Error event]
```

1. The user message is appended to the context store.
2. The agent retrieves the full conversation history and streams it to the
   transport along with the tool definitions and system prompt.
3. As `StreamEvent` values arrive, the agent accumulates text deltas and
   buffers pending tool calls.
4. When the transport yields an `IterationEnd` event:
   - If tool-use blocks were collected, the agent dispatches **all tool calls
     concurrently** via `asyncio.gather`, appends the assistant message and
     tool results to context, and loops back to step 2.
   - If only text was produced and the stop reason is `end_turn`, the agent
     emits a `SessionEndEvent` and returns.
5. If `max_iterations` is exceeded, the loop terminates with an error.

## Streaming API

`Agent` exposes two methods:

`run_stream(user_message, context) -> AgentStream`
: Returns an `AgentStream` — an async iterator over `StreamEvent` values.
  Use this when you need per-token streaming or want to observe tool calls
  as they happen.

`run(user_message, context) -> str`
: Convenience wrapper that consumes the stream and returns the final text.

## Concurrent tool dispatch

When the model requests multiple tool calls in a single response, the agent
runs them all concurrently:

```python
async def dispatch_tools(
    self,
    blocks: list[ToolUseBlock],
    iteration: int,
) -> list[ToolResultBlock]:
    tasks = [self._call_tool(block) for block in blocks]
    return list(await asyncio.gather(*tasks))
```

Each tool call goes through the full guard chain before execution. If a tool
raises an exception, the agent catches it and wraps it in a `ToolResultBlock`
with `is_error=True` — the model sees the error and can react accordingly.
