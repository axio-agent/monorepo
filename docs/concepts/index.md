# Core Concepts

Axio is assembled from a small set of well-defined building blocks. Each one
has a single responsibility and a stable interface — swap any of them without
touching the rest.

## Building blocks

### Agent

The {doc}`Agent <agent>` is the central orchestrator. It is a frozen dataclass
that wires a transport, a context store, and a set of tools into a loop:

1. Send conversation history to the transport.
2. Collect `StreamEvent` values as they arrive.
3. When the model issues tool calls — dispatch all of them **concurrently** via
   `asyncio.gather`, append the results, and loop.
4. When the model produces a final text response — emit `SessionEndEvent` and
   return.

The agent is deliberately thin. It has no opinions about retries, rate limits,
or logging — those belong to the transport or the application layer.

### Transport

A {doc}`transport <protocols>` is any object that implements the
`CompletionTransport` protocol — a single `stream()` method that takes the
conversation history and yields `StreamEvent` values. The core package ships no
transport of its own; install the one that matches your model provider:

| Package | Provider |
|---|---|
| `axio-transport-anthropic` | Anthropic Claude |
| `axio-transport-openai` | OpenAI and any OpenAI-compatible API |
| `axio-transport-codex` | ChatGPT via OAuth |

Because a transport is just a protocol, you can implement your own with nothing
more than `aiohttp` — no SDK required.

### Tools

A {doc}`tool <tools>` is a `ToolHandler` subclass — a Pydantic model whose
fields define the JSON schema exposed to the LLM and whose `__call__` method
implements the actual logic. Wrap it in a `Tool` dataclass to attach a name,
description, guards, and an optional concurrency limit.

The `context` field on `Tool` lets you pass arbitrary state — a database
connection, a sandbox object, a file path — to the handler at call time
without any global state or class-level variables.

### Context store

The {doc}`context store <context>` manages conversation history. It is an
abstract base class with two required methods: `append(message)` and
`get_history()`. Built-in options:

- `MemoryContextStore` — in-memory, lives for the duration of a session.
- `SQLiteContextStore` (`axio-context-sqlite`) — persistent across process
  restarts, supports multiple named sessions.

Implement your own to back conversations with Redis, a relational database,
or any other storage layer.

### Stream events

All agent I/O flows through typed, frozen {doc}`stream events <events>`.
The transport produces them; the agent enriches the stream with tool results;
consumers (TUI, your application code) react to them. There are no callbacks,
no side channels — everything observable happens as a named event in the stream.

### Permission guards

{doc}`Guards <guards>` gate tool execution. They sit between parameter
validation and handler invocation and form a composable middleware chain. A
guard receives the validated handler instance and either returns it (allow),
raises `GuardError` (deny), or mutates it (modify). Guards are reusable across
tools and composable — each tool carries its own `guards` tuple.

### Plugin system

The {doc}`plugin system <plugins>` uses Python's standard entry-point
mechanism. Packages register transports, tools, guards, and selectors in
`pyproject.toml` — `axio-tui` discovers them at startup with no import-time
coupling and no centralized registry. Adding a new integration is a matter of
publishing a package and declaring an entry point.

---

```{toctree}
:maxdepth: 1

agent
protocols
tools
events
context
guards
plugins
models
```
