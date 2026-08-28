# Serve Many Sessions

The isolated workbench is correct for one user. A server that shares its
`context` and sandbox across users is not.

Alice's next model request can contain Bob's messages. Their commands can edit
the same workspace. If two turns finish concurrently, their messages can also
enter one history in the wrong order.

## Outcome

Build a small in-process session registry. Each stable session ID selects one
context, sandbox, agent copy, cleanup stack, and turn lock.

## Fast Track

1. Resolve each authenticated request to a stable server-issued session ID.
2. Bundle one context, sandbox, agent copy, and turn lock per session.
3. Protect registry creation with one lock and each turn with another.
4. Own session resources through an `AsyncExitStack`.
5. Close every session before closing shared infrastructure.

{download}`Download the complete example <../../examples/tutorial/serve_many_sessions.py>`.

## Hands-on delta

### 1. Bundle session-owned state

One object keeps the resources that must never cross a session boundary:

```{literalinclude} ../../examples/tutorial/serve_many_sessions.py
:language: python
:caption: examples/tutorial/serve_many_sessions.py
:start-after: "# [docs:start-serve-session-state]"
:end-before: "# [docs:end-serve-session-state]"
```

### 2. Use one lock for each scope

The registry lock prevents two cold requests from creating duplicate resources
for one ID. A per-session lock serializes history mutation without blocking
other sessions.

```{literalinclude} ../../examples/tutorial/serve_many_sessions.py
:language: python
:caption: examples/tutorial/serve_many_sessions.py
:start-after: "# [docs:start-serve-session-create]"
:end-before: "# [docs:end-serve-session-create]"
:dedent: 4
```

Creation and turn execution use different locks. Closing the stream releases
the turn lock after completion, failure, or client cancellation:

```{literalinclude} ../../examples/tutorial/serve_many_sessions.py
:language: python
:caption: examples/tutorial/serve_many_sessions.py
:start-after: "# [docs:start-serve-turn-lifecycle]"
:end-before: "# [docs:end-serve-turn-lifecycle]"
:dedent: 4
```

The prototype carries shared configuration and non-execution tools. It must not
carry local execution tools or another session's Docker tools.

`Agent.copy()` creates a distinct `Agent` and replaces its tool list. The
prototype and transport remain shared, while Docker tools point only to that
session's sandbox through `Tool.context`.

`AsyncExitStack` owns resources created during a cold start. If sandbox startup
fails or the task is cancelled, the partial context and Docker client still
close. `stream_turn()` also closes `AgentStream` when a client disconnects.

:::{admonition} Production note
The example holds the registry lock during cold startup. Use keyed single-flight
creation when serialized container startup becomes a measured bottleneck.

Stop accepting requests and drain active turns before closing the harness.
:::

### 3. Reuse one stable ID

Issue IDs on the server, for example with `uuid4().hex`. Do not place an
arbitrary client string directly in a Docker name or database key.

Constructing this factory is daemon-free. Docker is contacted when the harness
enters the returned sandbox.

```{literalinclude} ../../examples/tutorial/serve_many_sessions.py
:language: python
:caption: examples/tutorial/serve_many_sessions.py
:start-after: "# [docs:start-serve-sandbox-factory]"
:end-before: "# [docs:end-serve-sandbox-factory]"
```

`name=session_id` reattaches to an existing named container and starts it when
necessary. The named volume preserves `/workspace` if that container was
deleted and must be created again.

Existing containers keep their original sandbox policy. Version that policy,
and replace stale containers before reuse. Keep fixed dependencies in the image
or under the persistent workspace.

### 4. Recover the conversation

Open one SQLite connection during application startup. Replace the fixed
`SESSION_ID` from {doc}`Persist the Conversation <persist-the-conversation>`
with a factory. Preserve compaction by wrapping each session store with its own
`AutoCompactStore`:

```{literalinclude} ../../examples/tutorial/serve_many_sessions.py
:language: python
:caption: examples/tutorial/serve_many_sessions.py
:start-after: "# [docs:start-serve-application-lifecycle]"
:end-before: "# [docs:end-serve-application-lifecycle]"
```

The application owns the shared connection. Closing one session closes its
context wrapper, not the connection.

See {doc}`Context and Messages <../concepts/context>` for store behavior,
compaction, and custom backends.

## Try It

This check does not require Docker or a model API. It proves that same-session
turns serialize, different sessions overlap, resources stay separate, and
SQLite recovers one stable ID.

Run `uv run python examples/tutorial/serve_many_sessions.py` from the repository
root.

```{literalinclude} ../../examples/tutorial/serve_many_sessions.py
:language: python
:caption: examples/tutorial/serve_many_sessions.py
:start-after: "# [docs:start-serve-sqlite-recovery]"
:end-before: "# [docs:end-serve-sqlite-recovery]"
```

After a service restart, the registry starts empty. The first request recreates
the in-memory `CloudSession`; SQLite reloads its messages, and Docker reattaches
or remounts its workspace using the same ID.

If the database survives but both the container and volume are missing,
history recovers into an empty workspace. Report that condition instead of
claiming full recovery.

## Done when

- [ ] Every request resolves to an authenticated, stable server-issued ID.
- [ ] Same-session turns never overlap.
- [ ] Different sessions can stream concurrently.
- [ ] Contexts, agent copies, and sandboxes stay session-local.
- [ ] `AsyncExitStack` closes partial and completed resources.
- [ ] SQLite recovers history by the same stable ID.

## Next failure

The next integration request creates another boundary. External tool
connections have their own names, permissions, and lifetimes.

Continue with {doc}`extend-with-mcp`.
