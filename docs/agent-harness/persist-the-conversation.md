# Persist the Conversation

The renderer makes every action visible until the process exits. Restart
`harness.py`, ask a follow-up question, and the assistant has no earlier
project context. `MemoryContextStore` stores only Python objects in that process.

## Outcome

The harness stores messages and usage in SQLite. Reusing one stable session ID
after a restart resumes the same conversation.

## Fast Track

1. Install `axio-context-sqlite`.
2. Open one connection with `connect()` at application startup.
3. Bind `SQLiteContextStore` to a stable session ID and project name.
4. Keep the connection open while the store is in use.
5. Close the connection in the application shutdown path.

{download}`Download the complete example <../../examples/tutorial/persist_the_conversation.py>`.

## Hands-on delta

### 1. Replace the context construction

Add the context package if the application does not already depend on it. Run
`uv add axio-context-sqlite` from the application root.

Replace the process-local memory store with a connection and a session-bound
store:

```{literalinclude} ../../examples/tutorial/persist_the_conversation.py
:language: python
:caption: examples/tutorial/persist_the_conversation.py
:start-after: "# [docs:start-persist-store-lifecycle]"
:end-before: "# [docs:end-persist-store-lifecycle]"
```

`AXIO_TUTORIAL_DATA_DIR` selects the data directory for this runnable example.
It defaults to `data`. The application-level `AXIO_SESSION_ID` selects the
conversation to resume.

Open the connection once around the application lifetime. A REPL should keep it
open around the complete input loop, not reconnect for every user turn.

### 2. Resume by stable ID

`session_id` selects a conversation. `local-project-demo` is useful for one
local user because it is stable across process starts. A production harness
should receive a globally unique conversation ID from its authenticated
application layer.

Do not generate a fresh UUID at startup when the goal is to resume. Also do not
share one fixed ID between users. Use a new stable ID when the user explicitly
starts another conversation.

`project` groups sessions for listing and usage reporting. Keep it stable too,
but do not treat it as an authorization boundary.

### 3. Own the connection lifetime

`connect()` creates and initializes the `aiosqlite.Connection`. The code that
calls `connect()` owns that connection and must close it.

`SQLiteContextStore` can share the connection with other session stores.
Consequently, `SQLiteContextStore.close()` is intentionally a no-op. Calling it
still honors the `ContextStore` lifecycle, but only `connection.close()` releases
the database resource.

See {doc}`../concepts/context` for session and storage semantics. See
{doc}`../guides/writing-context-stores` for the ownership contract.

## Try It

The example opens the same database through a new connection after the agent
turn. It confirms that the stable ID restores messages and cumulative usage.
A different ID starts empty:

```{literalinclude} ../../examples/tutorial/persist_the_conversation.py
:language: python
:caption: examples/tutorial/persist_the_conversation.py
:start-after: "# [docs:start-persist-resume-session]"
:end-before: "# [docs:end-persist-resume-session]"
```

Run `uv run python examples/tutorial/persist_the_conversation.py` twice. The
message count increases because both processes use the same default session and
data path. Set a different `AXIO_SESSION_ID` to start another conversation.

## Done when

- [ ] Restarting the process preserves the conversation.
- [ ] The same session ID resumes the same message history.
- [ ] A different session ID starts a separate conversation.
- [ ] The connection stays open for every store operation.
- [ ] Application shutdown closes the connection explicitly.

## Next failure

Persistence fixes forgetting, but it does not limit growth. The database now
keeps every old message and every large document result. The next lesson bounds
tool output and compacts older context.

Continue with {doc}`compact-long-context`.
