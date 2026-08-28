# Stream Every Action

The guarded document assistant works, but `await agent.run(...)` reveals only
text. During a tool call, the terminal appears idle. It hides the tool name,
the result, failures, and token usage.

`run()` is useful when another function needs only the final text. A harness
needs the typed event stream.

## Outcome

`harness.py` uses `run_stream()` and renders text, tool activity, errors, and
the final session status. It also closes an interrupted stream explicitly.

## Fast Track

1. Replace `await agent.run(...)` with `agent.run_stream(...)`.
2. Match concrete event dataclasses instead of inspecting untyped dictionaries.
3. Send assistant text to stdout and operational details to stderr.
4. Close `AgentStream` in `finally`.

{download}`Download the complete example <../../examples/tutorial/stream_every_action.py>`.

## Hands-on delta

### 1. Preserve both levels of progress

A **user turn** starts with one `run_stream(prompt, context)` call. It ends when
Axio emits one `SessionEndEvent`.

An **agent iteration** is one request to the transport. Each request ends with
`IterationEnd`. A `tool_use` stop runs the requested tools and starts another
iteration inside the same user turn. An `end_turn` stop finishes the user turn.

For example, one document lookup normally uses two iterations:

1. The model requests `read_document`, then the tool returns its result.
2. The model reads that result and writes the answer.

`SessionEndEvent.total_usage` adds the usage from both iterations.

### 2. Add a terminal renderer

Keep `DOCUMENTS`, both tools, and `DocumentAccessGuard` unchanged. Add this
function to `harness.py`:

```{literalinclude} ../../examples/tutorial/stream_every_action.py
:language: python
:caption: examples/tutorial/stream_every_action.py
:start-after: "# [docs:start-stream-render-turn]"
:end-before: "# [docs:end-stream-render-turn]"
```

`ToolUseStart` identifies the tool before execution. `ToolResult` contains the
completed result and its `is_error` flag. A failed tool is therefore different
from `Error`, which reports a stream or transport failure.

The renderer reports result size and exception type without printing raw tool
content or exception text. Those values can contain credentials, paths, or
private records. Send full details only to access-controlled logs after the
application applies its redaction policy.

Axio still adds the complete tool result to conversation history.
{doc}`Compact Long Context <compact-long-context>` bounds that stored content
at its source.

`AgentStream` is an async iterator, but it is not an async context manager. The
`finally` block closes its underlying async generator when iteration finishes,
the user cancels the task, or rendering raises an exception.

See {doc}`../concepts/events` for all event variants and
{doc}`../concepts/agent` for the complete loop contract.

## Try It

The offline example scripts one tool request and one final answer. It captures
both terminal channels, checks the typed result and usage, then forwards the
rendered output:

```{literalinclude} ../../examples/tutorial/stream_every_action.py
:language: python
:caption: examples/tutorial/stream_every_action.py
:start-after: "# [docs:start-stream-run-turn]"
:end-before: "# [docs:end-stream-run-turn]"
```

Run `uv run python examples/tutorial/stream_every_action.py` from the repository
root. The tool activity appears before the final answer. No API key is required.

## Done when

- [ ] Assistant text streams without waiting for the complete turn.
- [ ] `ToolUseStart` and `ToolResult` show tool activity.
- [ ] `Error` and `SessionEndEvent` show failure and completion state.
- [ ] One tool call can create multiple iterations within one user turn.
- [ ] Every stream closes in a `finally` block.

## Next failure

The harness is now observable, but its `MemoryContextStore` still disappears
with the process. The next lesson gives each conversation durable identity.

Continue with {doc}`persist-the-conversation`.
