# Compact Long Context

SQLite makes the project conversation durable. It also preserves every result
forever. One large document can dominate the next request, and a long session
eventually reaches the model's context limit.

Persistence controls lifetime. It does not control size.

## Outcome

Tool handlers return bounded text, and `AutoCompactStore` summarizes older
messages when actual provider usage crosses a configured input threshold.
Recent messages remain verbatim.

## Fast Track

1. Bound tool output before the handler returns it.
2. Keep the `SQLiteContextStore` as the durable inner store.
3. Wrap it with `AutoCompactStore` and pass the agent transport.
4. Choose `keep_recent` as a message count.
5. Observe stored usage instead of estimating tokens from characters.

{download}`Download the complete example <../../examples/tutorial/compact_long_context.py>`.

## Hands-on delta

### 1. Bound tool results first

The renderer controls terminal output only. Axio stores the handler's complete
return value and sends it on later model requests.

Add one application-level bound. The adapter also covers third-party text tools
whose handlers you do not control:

```{literalinclude} ../../examples/tutorial/compact_long_context.py
:language: python
:caption: examples/tutorial/compact_long_context.py
:start-after: "# [docs:start-compact-output-bound]"
:end-before: "# [docs:end-compact-output-bound]"
```

Keep the existing access guard and change only the return path in
`read_document`:

```{literalinclude} ../../examples/tutorial/compact_long_context.py
:language: python
:caption: examples/tutorial/compact_long_context.py
:start-after: "# [docs:start-compact-read-document]"
:end-before: "# [docs:end-compact-read-document]"
```

Apply `bound_tool_output` inside handlers you control. Use `bound_text_tool`
for an existing text tool. The wrapper keeps the original tool contract and
bounds its result before Axio stores it.

Bound structured fields before encoding JSON. Use separate size and dimension
limits for images or other binary results.

### 2. Compact accumulated history

Keep connection ownership from the previous lesson. Wrap the session store
before passing it to `render_turn`:

```{literalinclude} ../../examples/tutorial/compact_long_context.py
:language: python
:caption: examples/tutorial/compact_long_context.py
:start-after: "# [docs:start-compact-store-lifecycle]"
:end-before: "# [docs:end-compact-store-lifecycle]"
```

The wrapper keeps SQLite as the durable store. It adds a compaction decision
when Axio records each transport iteration.

Use two rules:

1. Trigger from the current iteration's real `input_tokens`, not cumulative
   session usage.
2. Treat `keep_recent=6` as six `Message` objects, not six turns or tokens.

Axio keeps tool calls paired with their results. It also leaves the original
history intact when summarization cannot complete.

See {doc}`../concepts/context` for the full compaction algorithm and
{doc}`../concepts/events` for provider usage events.

## Try It

Use an explicit low threshold to trigger compaction without a live provider.
The scripted response is the summary produced by the compaction agent:

```{literalinclude} ../../examples/tutorial/compact_long_context.py
:language: python
:caption: examples/tutorial/compact_long_context.py
:start-after: "# [docs:start-compact-trigger-demo]"
:end-before: "# [docs:end-compact-trigger-demo]"
```

Run `uv run python examples/tutorial/compact_long_context.py`. The complete
example also checks both output adapters before it runs the persistent agent
turn.

## Done when

- [ ] Every potentially large tool has an explicit output bound.
- [ ] `AutoCompactStore` wraps the durable SQLite store.
- [ ] Compaction uses real `IterationEnd` input usage.
- [ ] `keep_recent` is chosen as a message count.
- [ ] Cumulative input and output totals survive compaction.
- [ ] A failed summary leaves the original history available.

## Next failure

The project assistant can now preserve a long documentation conversation. A
developer next asks it to inspect repository files and run commands. Local
handlers would inherit the host process's permissions.

The next lesson adds those capabilities through bounded
`DockerSandbox.tools` wrappers and keeps the agent loop unchanged.

Continue with {doc}`isolate-execution`.
