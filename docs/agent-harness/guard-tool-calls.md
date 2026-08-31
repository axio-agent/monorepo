# Guard Tool Calls

The model can now request either document tool. It can also request private
file `.env`. A system-prompt sentence is guidance, not an authorization check.

The policy must run after Axio validates arguments and before the handler reads
data.

## Outcome

You will attach a `DocumentAccessGuard` to `read_document`. It accepts allowed
paths, rejects disallowed paths, and prevents the handler from running on
denial.

## Fast Track

1. Subclass `PermissionGuard` and implement `check`.
2. Return keyword arguments to allow a call.
3. Raise `GuardError` to deny a call.
4. Attach the guard through `Tool(..., guards=(guard,))`.
5. Verify direct denial and the Agent's error event.

{download}`Download the complete example <../../examples/tutorial/guard_tool_calls.py>`.

## Hands-on delta

### 1. Put access data in executable policy

Add this guard beside the handlers in `harness.py`:

```{literalinclude} ../../examples/tutorial/guard_tool_calls.py
:language: python
:caption: examples/tutorial/guard_tool_calls.py
:start-after: "# [docs:start-guard-document-access]"
:end-before: "# [docs:end-guard-document-access]"
```

`check` receives validated keyword arguments. Returning a dictionary allows the
call and can replace arguments. Here, the guard checks the exact path before
the handler reads the document.

Raising `GuardError` denies the call. Axio does not invoke the handler after a
guard denial.

Guards run in tuple order. Each guard receives the keyword arguments returned
by the previous guard. Read {doc}`../concepts/tools` for the full execution
order.

### 2. Keep search from becoming a side door

The previous lesson's `search_documents` returns only records whose visibility
is `public`. Keep that filter.

A guard protects one tool invocation. It does not replace authorization in the
document service or database. The backing service must still apply the current
user's access policy for every data path.

Use {doc}`../guides/writing-guards` for reusable guards, audit guards, and
concurrency guidance.

## Try It

First call the `Tool` directly. Then send the same denied input through the
Agent to verify its stream behavior.

Run `uv run python examples/tutorial/guard_tool_calls.py` from the repository
root.

```{literalinclude} ../../examples/tutorial/guard_tool_calls.py
:language: python
:caption: examples/tutorial/guard_tool_calls.py
:start-after: "# [docs:start-guard-try-calls]"
:end-before: "# [docs:end-guard-try-calls]"
```

A direct Tool call raises `GuardError`. During an Agent run, Axio converts the
denial into an error tool result. The transport can then produce a safe final
response or request a different action.

This distinction keeps policy errors inside the agent loop without hiding them
from your application event stream.

## Done when

- [ ] `DocumentAccessGuard` returns arguments for an allowed exact path.
- [ ] It raises `GuardError` for a disallowed path.
- [ ] The guarded handler does not run after denial.
- [ ] `read_document_tool` receives the guard as a one-item tuple.
- [ ] An Agent run emits one error `ToolResult` for the denied call.
- [ ] The backing search path still limits results independently.

## Next failure

`Agent.run` returns the final text, but the application cannot render the tool
request, denial, retry, or token usage as they happen. The next lesson switches
the harness boundary to `run_stream` and handles every action explicitly.

Continue with {doc}`stream-every-action`.
