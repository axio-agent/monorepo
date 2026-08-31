# Test the Harness, Not the Model

A live demonstration can succeed twice and fail on the third run. It proves
that a model answered once. It does not prove that the loop executes a tool,
records the result, emits terminal events, or cleans up after failure.

Axio's scripted transport makes those contracts deterministic.

## Outcome

One fast agent-loop check proves the complete request sequence: model request,
tool call, tool result, second model request, final text, and session end.

## Fast Track

1. Replace the provider transport with `StubTransport`.
2. Script one tool-use response and one final text response.
3. Collect the event stream and assert behavior, not unstable model prose.
4. Keep live-provider and Docker checks as a smaller integration suite.

{download}`Download the complete example <../../examples/tutorial/test_the_harness.py>`.

## Hands-on delta

### 1. Prove one complete agent turn

The handler is real. Only the model boundary is scripted. This focused example
does not reconstruct the final deployment harness:

```{literalinclude} ../../examples/tutorial/test_the_harness.py
:language: python
:caption: examples/tutorial/test_the_harness.py
:start-after: "# [docs:start-test-harness-complete-turn]"
:end-before: "# [docs:end-test-harness-complete-turn]"
```

The check does not assert whether a provider chooses the right tool from
natural language. That behavior needs a small model evaluation. This example
owns the harness contract after a tool call has been emitted.

Earlier lessons verify the guard, SQLite, compaction, Docker binding, session
registry, MCP lifecycle, and event adapter as separate contracts. Keep each
failure close to the boundary that must handle it.

### 2. Build the failure matrix

Add one focused check for each boundary that can fail:

| Failure | Required evidence |
|---|---|
| malformed tool JSON | error `ToolResult`; loop remains valid |
| handler exception | `ToolResult.is_error` is true; model receives the failure |
| guard denial | handler does not run; denial is observable |
| provider exception | `Error` and one terminal `SessionEndEvent` |
| client cancellation | stream closes; no orphaned persistent tool request |
| concurrent sessions | different contexts progress independently |
| same-session overlap | the turn lock prevents interleaved history |
| sandbox startup failure | partial resources close through `AsyncExitStack` |
| wire encoding | every public event produces the documented envelope |

Use {doc}`../guides/testing` for the complete helper API and more failure
examples.

### 3. Separate deterministic checks from evaluations

These checks answer different questions:

**Unit and harness checks**
: Does the system validate, dispatch, persist, stream, and clean up correctly?

**Model evaluations**
: Does a selected model choose the intended tool and complete realistic tasks?

**Integration checks**
: Do the provider, SQLite database, MCP server, Docker daemon, and delivery
  framework work together in the target environment?

Do not make the fast suite depend on API credentials or a Docker daemon. Run a
smaller integration suite where those services are available.

## Try It

Run `uv run python examples/tutorial/test_the_harness.py` from the repository
root. Then run your project's lint and type checks.

## Done when

- [ ] The scripted turn proves one tool result and exactly two iterations.
- [ ] The stream ends with one `SessionEndEvent`.
- [ ] Each important failure boundary has one focused check.
- [ ] Model evaluations do not replace deterministic harness checks.
- [ ] External-service checks are isolated from the fast suite.

## Capstone

Run the finished harness against a real project task such as:

> Read the repository documentation, find one failing test, make the smallest
> correct change, run the relevant verification, and report the evidence.

Observe the event stream, token growth, guard decisions, session ownership, and
sandbox lifetime. When behavior fails, add the smallest deterministic check
that reproduces the harness failure before changing the implementation.

You now have the important boundary Axio is designed to provide: a provider-
independent loop inside an application-owned, observable, testable harness.

Continue with {doc}`../concepts/index`, {doc}`../guides/index`, or the
{doc}`../api/index`.
