# Core Concepts

Axio is assembled from a small set of well-defined building blocks. Each one
has a single responsibility and a stable interface - swap any of them without
touching the rest.

## Building blocks

### Agent

The {doc}`Agent <agent>` is the central orchestrator. It is a frozen dataclass
that wires a transport, a context store, and a set of tools into a loop:

1. Send conversation history to the transport.
2. Collect `StreamEvent` values as they arrive.
3. When the model issues tool calls - dispatch all of them **concurrently** via
   `asyncio.gather`, append the results, and loop.
4. When the model produces a final text response - emit `SessionEndEvent` and
   return.

The agent is deliberately thin. It has no opinions about retries, rate limits,
or logging - those belong to the transport or the application layer.

### Transport

A {doc}`transport <protocols>` is any object that implements the
`CompletionTransport` protocol - a single `stream()` method that takes the
conversation history and yields `StreamEvent` values. The core package ships no
transport of its own; install the one that matches your model provider:

| Package | Provider |
|---|---|
| `axio-transport-anthropic` | Anthropic Claude |
| `axio-transport-openai` | OpenAI and any OpenAI-compatible API |
| `axio-transport-codex` | ChatGPT via OAuth |

Because a transport is just a protocol, you can implement your own with nothing
more than `aiohttp` - no SDK required.

### Tools

A {doc}`tool <tools>` is a plain `async def` function whose parameters define
the JSON schema exposed to the LLM and whose body implements the actual logic.
Wrap it in a `Tool` dataclass to attach a name, description, guards, and an
optional concurrency limit.

The `context` field on `Tool` lets you pass arbitrary state - a database
connection, a sandbox object, a file path - to the handler via `CONTEXT.get()`
at call time without any global state or class-level variables.

### Field Metadata

The {doc}`field metadata <field>` system uses `Field()` to add descriptions,
defaults, and constraints to tool parameters. These annotations are converted
to JSON Schema for LLM consumption.

### JSON Schema Generation

The {doc}`schema builder <schema>` converts Python type annotations into JSON
Schema objects sent to LLM providers in tool definitions.

### Context store

The {doc}`context store <context>` manages conversation history. It is an
abstract base class with two required methods: `append(message)` and
`get_history()`. Built-in options:

- `MemoryContextStore` - in-memory, lives for the duration of a session.
- `SQLiteContextStore` (`axio-context-sqlite`) - persistent across process
  restarts, supports multiple named sessions.

Implement your own to back conversations with Redis, a relational database,
or any other storage layer.

```{toctree}
:maxdepth: 1

agent
protocols
tools
selector
field
schema
events
tool-args-streaming
context
guards
plugins
models
```
