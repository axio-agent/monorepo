# Build an Agent Harness with Axio

A chatbot can produce useful text, but it cannot inspect application state or
perform work. Giving it tools starts a second problem: your application must
control sessions, permissions, persistence, execution, and output.

That application layer is the **agent harness**.

This course builds one small Axio harness through eleven causal lessons. Each
lesson starts with a concrete failure, adds one capability, and verifies the
result without a live model.

Axio already owns the model → tool → model loop. You will build the application
around that loop.

## What you will build

The course project starts as an in-memory project-document assistant. It grows
into a multi-session service with these capabilities:

- typed tools that route distinct requests;
- guards that check model-produced arguments before execution;
- a complete stream of text, tool, iteration, and session events;
- persistent history with bounded long-context behavior;
- isolated execution resources with explicit lifetimes;
- independent contexts for concurrent sessions;
- tools discovered through MCP;
- an adapter between Axio events and an application protocol;
- deterministic tests for the complete harness.

The repository domain keeps each code delta small. The same boundaries apply
to a coding agent, research worker, or automation service.

## Module 1: Give the loop controlled actions

- {doc}`agent-harness/from-chat-to-agent`: No tools means no application state;
  `read_document` adds one controlled read.
- {doc}`agent-harness/tools-that-route`: Exact lookup cannot search documents;
  distinct schemas and descriptions enable routing.
- {doc}`agent-harness/guard-tool-calls`: A schema cannot express user access;
  `PermissionGuard` enforces policy before execution.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Module 1 — Actions and policy

agent-harness/from-chat-to-agent
agent-harness/tools-that-route
agent-harness/guard-tool-calls
```

## Module 2: Observe and remember the work

- {doc}`agent-harness/stream-every-action`: Final text hides intermediate work;
  `run_stream()` exposes every typed event.
- {doc}`agent-harness/persist-the-conversation`: Process memory loses prior
  turns; SQLite gives each conversation durable identity.
- {doc}`agent-harness/compact-long-context`: Durable history grows without
  limit; bounded results and compaction control context size.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Module 2 — Events and context

agent-harness/stream-every-action
agent-harness/persist-the-conversation
agent-harness/compact-long-context
```

## Module 3: Isolate and scale the harness

- {doc}`agent-harness/isolate-execution`: Local tools inherit host access;
  `DockerSandbox` contains file and shell execution.
- {doc}`agent-harness/serve-many-sessions`: Shared state mixes users and turns;
  session bundles isolate context, resources, and locks.
- {doc}`agent-harness/extend-with-mcp`: Custom handlers couple each integration
  to the harness; MCP loads external tools without changing the loop.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Module 3 — Resources and sessions

agent-harness/isolate-execution
agent-harness/serve-many-sessions
agent-harness/extend-with-mcp
```

## Module 4: Adapt and prove the boundary

- {doc}`agent-harness/adapt-the-event-stream`: Python events cannot cross a
  product boundary; a JSON adapter creates a versioned protocol.
- {doc}`agent-harness/test-the-harness`: Live demonstrations cannot prove
  harness invariants; scripted transports make complete turns deterministic.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Module 4 — Interfaces and verification

agent-harness/adapt-the-event-stream
agent-harness/test-the-harness
```

## The boundary that stays stable

Axio and your harness have different responsibilities.

| Axio owns | Your harness owns |
|---|---|
| Repeated transport calls | Transport construction and credentials |
| Tool-call collection and dispatch | Which tools and policies a session receives |
| Argument validation and guard execution | User identity and authorization data |
| Conversation updates through a context store | Context selection, lifetime, and persistence |
| Axio stream events | Rendering, API messages, logs, and metrics |
| Iteration and session completion | Cancellation, resource cleanup, and deployment |

You will keep this boundary while the delivery surface changes from one call
to many concurrent sessions.

For exact loop behavior, read {doc}`concepts/agent`. For the component
interfaces, read {doc}`concepts/protocols`.

## Prerequisites

You need:

- Python 3.12 or later;
- basic `async` and `await` knowledge;
- an environment with `axio` installed;
- a provider transport only when you want to run against a live model.

Inside the Axio repository, run `uv sync --all-packages` once.

The course examples use {class}`axio.testing.StubTransport`. They do not need
network access, credentials, or a specific model.

## How the course works

### Follow the failure

Each lesson changes the harness because its current behavior is insufficient.
The bridge at the end identifies the next failure before the next lesson fixes
it.

### Keep one runnable project

Apply each lesson's delta to your own `harness.py`. The repository also provides
one complete snapshot for each lesson under `examples/tutorial/`.

Read the focused fragment in the lesson. You can also download its snapshot and
run it directly.

### Use the Fast Track when you know the concept

Every lesson includes a short implementation path. Read the full explanation
when a result differs from the stated outcome.

### Verify without model variance

The **Try It** section uses stubbed transport events or direct tool calls. A
test can therefore prove dispatch, guard, context, and event behavior exactly.

A live model still decides which offered tool to request. Tool descriptions and
field schemas provide that model-facing selection contract.

## Course project shape

The exact structure evolves, but the ownership stays clear:

```text
project-harness/
├── harness.py          # Agent, tools, guards, and context policy
├── app.py              # Session and output surface
└── tests/
    └── test_harness.py # Deterministic contract tests
```

Keep provider-specific construction at the application edge. The harness
depends on Axio's `CompletionTransport` interface, not one vendor client.

## When to use the reference docs

This course explains why the next capability is necessary. The reference docs
describe every available option.

- {doc}`concepts/tools` covers schemas, validation, guards, and concurrency.
- {doc}`concepts/events` defines the complete event model.
- {doc}`concepts/context` defines context-store behavior.
- {doc}`guides/writing-guards` shows reusable guard patterns.
- {doc}`guides/mcp-tools` covers MCP configuration and lifecycle.
- {doc}`guides/testing` covers Axio's deterministic test helpers.

## Capstone

Connect the completed harness to one real workflow. Use a task that needs more
than one tool call and more than one user turn.

Observe these boundaries:

1. Did the model receive only the tools that this session needs?
2. Did every sensitive call pass through an application policy?
3. Can a client reconstruct progress from your event adapter?
4. Does each session retain only its own conversation and resources?
5. Do deterministic tests cover failure paths as well as success paths?

Start with {doc}`agent-harness/from-chat-to-agent`.
