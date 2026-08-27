# Quick Start: Build an Agent Harness

This tutorial builds two complete applications:

1. a small local REPL that makes the agent loop visible;
2. a multi-session cloud harness in which every session owns its conversation
   context and its own Docker sandbox.

The second program grows out of the first. Agent, transport, tool call, context,
and event keep the same meaning when input moves from a terminal to HTTP or a
queue.

## The five pieces

**Agent**
: Runs the model → tool → model loop. It is an orchestrator, not an LLM
  client, a UI, or a session manager.

**Transport**
: Adapts one provider to Axio. It serializes messages and tool schemas,
  performs one model request, and converts the response stream into Axio
  events.

**Tool**
: The glue that declares an async Python function as an agent tool. It gives
  the function a stable name, exposes its signature to the model, and connects
  a model request to the Python handler.

**Context store**
: Holds message history for one conversation. The agent reads it before every
  model request and appends user, assistant, and tool-result messages.

**Harness**
: Your application layer. It owns input and output, session lookup, resource
  lifetimes, cancellation, error policy, and observability.

Axio owns the agent loop. Your harness owns the application.

# Part I: a simple REPL

The first program has one process, one user, one agent, and one in-memory
conversation.

The REPL is the harness. It accepts a prompt, asks one Agent to run it, and
keeps using the same ContextStore so the next prompt can refer to earlier work:

~~~{mermaid}
flowchart TD
    C[Terminal input] -->|prompt| H[REPL harness]
    H -->|run prompt| A[Agent]
    A <-->|conversation history| S[Context store]
~~~

## 1. Install a transport

The core `axio` package defines the agent loop and the transport protocol. It
does not contain an LLM client. Provider packages distribute concrete
transport classes. Install the package for the API you use. The first REPL
uses OpenAI:

~~~bash
pip install axio axio-transport-openai
export OPENAI_API_KEY="..."
~~~

The available packages and their public transports are:

| Install | Import | Provider-specific configuration |
|---|---|---|
| `axio-transport-openai` | {class}`~axio_transport_openai.OpenAITransport`, {class}`~axio_transport_openai.nebius.NebiusTransport`, {class}`~axio_transport_openai.openrouter.OpenRouterTransport`, {class}`~axio_transport_openai.custom.OpenAICompatibleTransport` | API key, base URL, OpenAI-compatible model |
| `axio-transport-anthropic` | {class}`~axio_transport_anthropic.AnthropicTransport` | Anthropic API key, or Google project and location for Vertex AI |
| `axio-transport-google` | {class}`~axio_transport_google.GoogleTransport`, {class}`~axio_transport_google.VertexAITransport` | Gemini API key, or Google Application Default Credentials |
| `axio-transport-codex` | {class}`~axio_transport_codex.CodexTransport` | ChatGPT OAuth access, refresh, expiry, and account tokens |

The four classes on the first row differ in one way that outweighs the rest.
`OpenAITransport` speaks `/v1/responses`. `NebiusTransport`, `OpenRouterTransport`
and `OpenAICompatibleTransport` speak `/v1/chat/completions`. The field is
`api: Literal["responses", "chat"]`. It matters because `/v1/chat/completions` refuses
function tools beside any reasoning effort other than `"none"`, so a reasoning model with
tools cannot reason there. `/v1/responses` takes both. See
{ref}`the troubleshooting entry <tools-and-reasoning-400>`.

For example, installing `axio-transport-anthropic` makes this import available:

~~~python
from axio_transport_anthropic import AnthropicTransport

transport = AnthropicTransport()
~~~

Providers disagree about authentication, endpoints, message formats, tool
encoding, streaming protocols, stop reasons, usage, retries, and model
capabilities. Their constructors therefore have different fields. Follow the
class link in the table for the exact API. Once constructed, every completion
transport presents the same runtime interface to Agent:

~~~{mermaid}
flowchart TD
    A[Agent] -->|messages + tool definitions| T[Transport]
    T -->|provider request| P[LLM provider]
    P -->|provider stream| T
    T -->|events| A
~~~

~~~python
from collections.abc import AsyncIterator
from typing import Protocol

from axio import StreamEvent, Tool
from axio.messages import Message


class CompletionTransport(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]: ...
~~~

A transport performs **one model request**. It does not execute tools or
decide when to make another request. Those are agent-loop responsibilities.

## 2. Turn Python functions into tools

Install the local coding-tool handlers:

~~~bash
pip install axio-tools-local
~~~

An async function is only a Python handler. Wrapping it in `Tool` declares that
function as something the agent may offer to the model:

~~~text
async function + Tool(name=..., handler=...) = agent tool
model chooses that tool + arguments          = tool call
~~~

`Tool` is the glue between these sides. It gives the handler a stable name,
derives the input contract from its signature, uses its docstring as the
model-facing description, validates arguments, and invokes the function when
the agent receives a matching tool call.

~~~python
from axio import Tool
from axio_tools_local.list_files import list_files
from axio_tools_local.patch_file import patch_file
from axio_tools_local.read_file import read_file
from axio_tools_local.run_python import run_python
from axio_tools_local.shell import shell
from axio_tools_local.write_file import write_file


coding_tools = [
    Tool(name="list_files", handler=list_files),
    Tool(name="read_file", handler=read_file),
    Tool(name="write_file", handler=write_file),
    Tool(name="patch_file", handler=patch_file),
    Tool(name="shell", handler=shell, concurrency=2),
    Tool(name="run_python", handler=run_python, concurrency=2),
]
~~~

When the model requests one of these names, Tool connects the structured call
to whichever handler was registered for this Agent:

~~~{mermaid}
flowchart TD
    A[Agent] -->|tool name + arguments| T[Tool]
    T -->|local binding| L[Harness host]
    T -->|sandbox binding| D[Session Docker sandbox]
    L -->|result| A
    D -->|result| A
~~~

These handlers operate in the REPL process with its filesystem permissions.
Use them only in a trusted local workspace. They are not a requirement of the
harness. `axio-tools-docker` provides drop-in replacements with the same tool
names and field schemas. The switch is only the list passed to Agent:

~~~python
# Trusted local workspace
agent = Agent(system=SYSTEM, transport=transport, tools=coding_tools)

# Isolated execution; create the agent inside the sandbox lifetime
async with DockerSandbox(image="python:3.12-slim") as sandbox:
    agent = Agent(system=SYSTEM, transport=transport, tools=sandbox.tools)
~~~

The transport, agent loop, prompts, and event handling do not change. Part II
shows the imports and lifecycle code needed to bind each cloud session to its
own Docker sandbox.

This choice determines where generated code executes. Local tools run shell
commands and file operations in the harness process. Docker tools expose the
same names and input schemas, but run inside a container. The LLM does not know
which implementation is registered. With Docker tools, the harness only moves
structured calls and results between the model and the session container.
Generated commands never execute on the harness host.

You can see the generated schema in an ordinary Python REPL:

~~~console
$ python3
>>> from axio_tools_local.list_files import list_files
>>> from axio import Tool
>>> tool = Tool(name="list_files", handler=list_files)
>>> tool.input_schema
{'type': 'object', 'properties': {'directory': {'type': 'string', 'default': '.'}}}
>>> tool.schema
mappingproxy({'type': 'object', 'properties': {'directory': {'type': 'string', 'default': '.'}}})
~~~

`input_schema` is the generated JSON-compatible dictionary. `schema` is its
read-only view used by transports. Axio builds both from the Python signature
and docstring, so the harness does not need to construct provider-specific tool
descriptions.

For your own tool, use Annotated and Field when parameter names are not
self-explanatory or need bounds:

<!-- name: test_tutorial_custom_tool -->
```python
from typing import Annotated

from axio import Field, Tool


async def search_code(
    query: Annotated[str, Field(description="Literal text to find")],
    max_results: Annotated[
        int,
        Field(description="Maximum matches to return", default=20, ge=1, le=100),
    ] = 20,
) -> str:
    """Search the current project without modifying it."""
    return f"search {query!r}, limit={max_results}"


search_tool = Tool(name="search_code", handler=search_code)
```

Inspect `search_tool.input_schema` in the same REPL to see the descriptions,
default, and bounds generated from `Field`.

The important Tool fields are:

| Field | Meaning |
|---|---|
| name | Stable name emitted by the model and resolved by the agent |
| handler | Async function that performs the work |
| description | Model-facing description; defaults to the docstring |
| guards | Checks that may allow, reject, or rewrite arguments |
| concurrency | Maximum concurrent executions of this tool instance |
| context | Runtime dependency made available to the handler |

Do not put authorization only in the system prompt. Model-produced arguments
are untrusted input. Enforce access in guards and in the service called by the
handler.

## 3. Create the coding agent and context

~~~python
from axio import Agent, MemoryContextStore
from axio_transport_openai import OpenAITransport

transport = OpenAITransport()
agent = Agent(
    system=(
        "You are a coding agent working in the current directory. "
        "Inspect files before editing them, use tools to perform the work, "
        "run relevant tests, and keep working until the request is complete."
    ),
    transport=transport,
    tools=coding_tools,
)
context = MemoryContextStore()
~~~

The Agent is reusable configuration: system prompt, transport, tools, and loop
limits. The context is mutable session state. Keeping it outside the agent is
what later lets one definition serve many users.

For a one-shot coding task:

~~~python
reply = await agent.run(
    "Find the failing test, fix the bug, and rerun that test.",
    context,
)
print(reply)
~~~

The run method collects text. A REPL uses run_stream so it can also show every
tool call, incremental output, errors, and usage.

## 4. Make the REPL persistent with SQLite

`MemoryContextStore` forgets the conversation when the process exits. The
agent and the REPL loop do not depend on that implementation, so persistence
requires changing only construction and cleanup.

Install the SQLite context package:

~~~bash
pip install axio-context-sqlite
~~~

Open the database, then create a store with a stable session ID:

~~~python
from axio_context_sqlite import SQLiteContextStore, connect


connection = await connect("data/repl.db")
context = SQLiteContextStore(
    connection,
    session_id="local-repl",
    project="coding-repl",
)
~~~

Pass this `context` to the same `agent.run_stream()` call. On the next process
start, using the same database, `project`, and `session_id` resumes the existing
conversation. Use a new session ID to start a separate conversation.

The object that opens the connection owns it. A complete REPL lifetime looks
like this:

~~~python
async def persistent_repl() -> None:
    connection = await connect("data/repl.db")
    context = SQLiteContextStore(
        connection,
        session_id="local-repl",
        project="coding-repl",
    )
    try:
        await repl_loop(agent, context)
    finally:
        await context.close()
        await connection.close()
~~~

`SQLiteContextStore.close()` follows the `ContextStore` lifecycle contract.
The shared database connection is closed separately. That distinction
becomes important in the cloud harness, where many session stores share one
connection.

## 5. Follow the agent loop

Assume the user asks, "Inspect this repository and run the tests."

~~~{mermaid}
sequenceDiagram
    participant R as REPL
    participant A as Agent
    participant C as Context
    participant T as Transport
    participant M as Model
    participant X as coding tools

    R->>A: run_stream(prompt, context)
    A->>C: append user message
    A->>C: get_history()
    A->>T: stream(history, tools, system)
    T->>M: provider request 1
    M-->>T: list_files(directory=".")
    T-->>A: tool events + IterationEnd(tool_use)
    A->>X: await list_files(directory=".")
    X-->>A: project files
    A->>C: append tool request and result
    A->>T: stream(updated history, tools, system)
    T->>M: provider request 2
    M-->>T: final answer
    T-->>A: text events + IterationEnd(end_turn)
    A->>C: append assistant answer
    A-->>R: SessionEndEvent
~~~

One user turn may contain multiple **iterations**. An iteration is one model
request. The agent continues after tool calls. It ends when the model returns
end_turn.

The first transport stream might be:

~~~text
ToolUseStart(id="call_1", name="list_files")
ToolInputDelta(id="call_1", partial_json='{"directory":"."}')
IterationEnd(stop_reason=tool_use, usage=...)
~~~

Arguments are partial JSON because providers may split them anywhere. The
agent buffers fragments, parses them at IterationEnd, validates them against
the tool contract, and invokes the handler.

After the handler completes, the agent emits ToolResult and makes the next
model request:

~~~text
TextDelta("I inspected the project and ran the relevant tests.")
IterationEnd(stop_reason=end_turn, usage=...)
SessionEndEvent(stop_reason=end_turn, total_usage=...)
~~~

If one model response requests several tools, Axio dispatches them
concurrently. Tool handlers and dependencies must be safe for concurrent use.

## 6. Render typed events

The renderer is the boundary between the agent and terminal:

<!-- name: test_tutorial_render_event -->
```python
from axio import Refusal, StreamEvent, TextDelta, ToolResult, ToolUseStart
from axio.events import Error, SessionEndEvent


def render_event(event: StreamEvent) -> None:
    match event:
        case TextDelta(delta=delta):
            print(delta, end="", flush=True)
        case Refusal(text=text, category=category):
            print(f"\ndeclined ({category or 'unstated'}): {text}", flush=True)
        case ToolUseStart(name=name):
            print(f"\n→ {name}", flush=True)
        case ToolResult(name=name, is_error=is_error, input=tool_input):
            status = "failed" if is_error else "done"
            print(f"\n← {name} {status}: {tool_input}", flush=True)
        case Error(exception=exception):
            print(f"\nerror: {exception}", flush=True)
        case SessionEndEvent(total_usage=usage):
            print(
                f"\n[{usage.input_tokens} input, "
                f"{usage.output_tokens} output tokens]"
            )
```

ToolUseStart says a call began. ToolInputDelta is useful for live argument
rendering. Completed ToolResult contains parsed input, content, and is_error,
so logs normally use it.

Refusal is a separate event and not a TextDelta on purpose. As assistant text, a
decline is indistinguishable from an answer. A renderer that matches only TextDelta
prints nothing at all for a declined turn. Its text may be empty, because a provider
that blocks the prompt generates nothing. The category carries the reason.

The renderer matches six event types out of the twenty-six in `StreamEvent`. Everything
else falls through the match. That is the intended shape. Add a case when you need one:
`Citation` for attribution, `ProviderEvent` for what a provider ran on its own side,
`ReasoningDelta` for visible thinking. See {doc}`concepts/events` for the whole set.

Transport failures appear as Error followed by SessionEndEvent with an error
stop reason. End of iteration without a Python exception does not imply
success.

## 7. Assemble the REPL

The turn runner owns rendering, failure policy, and stream cleanup:

~~~python
from axio import Agent, ContextStore, StopReason
from axio.events import Error, SessionEndEvent

#: Stop reasons the runner treats as a finished turn. Everything else is a failure.
COMPLETED = frozenset(
    {
        StopReason.end_turn,
        # The model declined, or the provider blocked the turn. Terminal, and not a
        # failure: the same prompt sent again will be declined again. Raised here, the
        # caller cannot tell a decline from a broken connection and retries something
        # that can never work.
        StopReason.refusal,
        # The caller or the provider stopped the turn. Nothing broke.
        StopReason.cancelled,
    }
)


class TurnFailed(RuntimeError):
    """The model turn failed."""


async def run_repl_turn(
    agent: Agent,
    context: ContextStore,
    prompt: str,
) -> None:
    failure: BaseException | None = None
    session_end: SessionEndEvent | None = None
    stream = agent.run_stream(prompt, context)

    try:
        async for event in stream:
            render_event(event)
            match event:
                case Error(exception=exception):
                    failure = exception
                case SessionEndEvent() as end:
                    session_end = end
    finally:
        await stream.aclose()

    if failure is not None:
        raise TurnFailed("agent turn failed") from failure
    if session_end is None:
        raise TurnFailed("stream ended without SessionEndEvent")
    if session_end.stop_reason not in COMPLETED:
        raise TurnFailed(f"agent stopped with {session_end.stop_reason}")
~~~

`StopReason` publishes eight members. The accepted set decides which of them the harness
treats as a bug. `max_tokens` and `context_window_exceeded` are left out because both
mean a truncated answer, which a harness does want to hear about. `pause_turn` never
reaches here. The agent resumes on it rather than ending the run.

The complete input loop is deliberately boring:

~~~python
import asyncio

from axio import Agent, ContextStore, MemoryContextStore
from axio_transport_openai import OpenAITransport


async def repl_loop(agent: Agent, context: ContextStore) -> None:
    while True:
        try:
            prompt = await asyncio.to_thread(input, "\nyou> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() in {"/exit", "/quit"}:
            break
        if not prompt.strip():
            continue

        print("agent> ", end="", flush=True)
        try:
            await run_repl_turn(agent, context, prompt)
        except TurnFailed as exc:
            print(f"turn failed: {exc}")


async def repl() -> None:
    agent = Agent(
        system=(
            "You are a coding agent working in the current directory. "
            "Inspect before editing, use tools, and verify the result."
        ),
        transport=OpenAITransport(),
        tools=coding_tools,
    )
    await repl_loop(agent, MemoryContextStore())


asyncio.run(repl())
~~~

The `while True` loop is intentional. The REPL session has no turn limit. It
keeps accepting requests against the same context until the user explicitly
enters `/exit` or `/quit`, sends EOF, or the process receives a shutdown
signal. Completing one request returns to the prompt. It does not end the
session.

There are therefore two nested loops:

1. the **REPL loop** is unbounded and owns the long-lived conversation;
2. inside each prompt, the **agent loop** automatically repeats
   model → tool → model until the model finishes that task.

`Agent.max_iterations` is only a per-prompt circuit breaker for a broken model
or a tool-call loop. The default does not cap the number of REPL prompts or the
conversation lifetime.

This is already a harness:

- one Agent definition;
- one long-lived ContextStore, so follow-up requests keep history;
- one event renderer;
- explicit error and cancellation policy.

## Optional: add tools from an MCP server

`axio-tools-mcp` discovers a server's tool definitions and returns ordinary
Axio `Tool` objects. Add them to the same list as local tools. The agent loop
does not need a special MCP mode:

~~~bash
pip install axio-tools-mcp
~~~

~~~python
import asyncio

from axio import Agent, MemoryContextStore
from axio_tools_mcp import MCPServerConfig, load_mcp_tools
from axio_transport_openai import OpenAITransport


async def mcp_repl() -> None:
    mcp_tools, sessions = await load_mcp_tools([
        MCPServerConfig(
            name="fs",
            command="mcp-server-filesystem",
            args=["--root", "."],
        ),
    ])
    agent = Agent(
        system="Use the available tools to inspect and modify the project.",
        transport=OpenAITransport(),
        tools=[*coding_tools, *mcp_tools],
    )
    try:
        await repl_loop(agent, MemoryContextStore())
    finally:
        await asyncio.gather(*(session.close() for session in sessions))


asyncio.run(mcp_repl())
~~~

The server name prefixes every discovered tool, so a server tool named
`read_file` becomes `fs__read_file`. The returned sessions own live stdio or
HTTP connections. They must remain open for as long as their tools are in
use. `mcp-server-filesystem` is an example MCP server executable. Install it
separately. Replace `command` and `args` with your server, or use
`url="https://.../mcp"` for a remote server.

An MCP tool executes wherever its MCP server runs. Starting a stdio server as
above runs it beside the harness. It does not automatically move execution into
the Docker sandbox. For isolation, run the MCP server inside the session
container or connect to an isolated remote MCP service.

# Part II: a multi-session cloud harness

A cloud service must additionally decide:

- which conversation belongs to a request;
- whether two turns may mutate one context concurrently;
- which resources belong to a session;
- how a coding agent gets its own isolated filesystem;
- when idle sessions and containers are removed.

Each session will own:

~~~text
CloudSession
├── ContextStore       conversation history
├── DockerSandbox      container and filesystem
├── Agent              prototype + this sandbox's tools
├── asyncio.Lock       one active turn for this context
└── AsyncExitStack     deterministic cleanup
~~~

Different sessions can run concurrently. Turns sharing one context are
serialized because interleaved messages create invalid history.

## 8. Bind Docker tools to one session

~~~bash
pip install axio-tools-docker axio-context-sqlite
~~~

DockerSandbox is an async context manager. Its tools exist only after entry:

~~~python
from axio_tools_docker import DockerSandbox


async with DockerSandbox(
    image="python:3.12-slim",
    memory="512m",
    cpus="1.0",
    network=False,
) as sandbox:
    session_agent = prototype_agent.copy(
        tools=[*prototype_agent.tools, *sandbox.tools],
    )
~~~

The binding is the session-specific tool list. Every Tool returned by
sandbox.tools has context=sandbox. Before the handler runs, Tool places that
object in Axio's CONTEXT context variable. The shell handler is conceptually:

~~~python
from axio import CONTEXT


async def shell(command: str) -> str:
    sandbox: DockerSandbox = CONTEXT.get()
    return await sandbox.exec(command)
~~~

Therefore never create Docker tools before entering the sandbox. Never cache
one sandbox's tools globally. Never attach one sandbox tool list to every
user. Two agents may both expose shell while each Tool.context points to a
different container.

## 9. Implement the small session registry

The cloud harness needs only a session record, a dictionary, and two locks.
One lock protects creation in the dictionary. One lock per session prevents
two turns from interleaving in the same context.

<!-- name: test_cloud_harness_definition -->
```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from axio import Agent, ContextStore, StreamEvent
from axio_tools_docker import DockerSandbox


type ContextFactory = Callable[[str], ContextStore]
type SandboxFactory = Callable[[str], DockerSandbox]


@dataclass(slots=True)
class CloudSession:
    agent: Agent
    context: ContextStore
    resources: AsyncExitStack
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CloudHarness:
    def __init__(
        self,
        prototype_agent: Agent,
        context_factory: ContextFactory,
        sandbox_factory: SandboxFactory,
    ) -> None:
        self._prototype_agent = prototype_agent
        self._context_factory = context_factory
        self._sandbox_factory = sandbox_factory
        self._sessions: dict[str, CloudSession] = {}
        self._registry_lock = asyncio.Lock()
        self._closed = False

    async def _session(self, session_id: str) -> CloudSession:
        async with self._registry_lock:
            if self._closed:
                raise RuntimeError("harness is closed")
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing

            resources = AsyncExitStack()
            try:
                context = self._context_factory(session_id)
                resources.push_async_callback(context.close)
                sandbox = await resources.enter_async_context(
                    self._sandbox_factory(session_id)
                )
                session = CloudSession(
                    agent=self._prototype_agent.copy(
                        tools=[
                            *self._prototype_agent.tools,
                            *sandbox.tools,
                        ],
                    ),
                    context=context,
                    resources=resources,
                )
            except BaseException:
                await resources.aclose()
                raise

            self._sessions[session_id] = session
            return session

    async def stream_turn(
        self,
        session_id: str,
        prompt: str,
    ) -> AsyncIterator[StreamEvent]:
        session = await self._session(session_id)
        async with session.turn_lock:
            stream = session.agent.run_stream(prompt, session.context)
            try:
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()

    async def close(self) -> None:
        async with self._registry_lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(
            *(session.resources.aclose() for session in sessions)
        )
```

That is the complete baseline:

- the first request for an ID creates or restores its context and sandbox;
- later requests reuse that same pair;
- different sessions can run concurrently;
- the per-session lock serializes one conversation;
- closing the harness releases every context and sandbox handle according to
  the configured persistence policy;
- disconnecting a stream reaches the finally block and closes AgentStream.

The registry lock is held during a cold session start to keep the example small
and correct. Active turns do not hold it. If container startup throughput
becomes measurable, replace only _session() with keyed single-flight creation.
The agent, tools, event stream, and public harness API stay unchanged.

This baseline keeps active sessions until service shutdown. Idle eviction,
distributed session routing, and quotas are deployment policies, not Axio
concepts. Add them around this registry when the product requires them instead
of hiding them in the introductory example.

## 10. Restore the session sandbox

The registry is only an in-process cache. After a service restart,
`_sessions` is empty, even though the SQLite conversation and an earlier Docker
container may still exist. Recovery happens when the first new request calls
the two factories again:

1. `context_factory(session_id)` passes that ID directly to
   `SQLiteContextStore`, which opens the same conversation;
2. `sandbox_factory(session_id)` passes the same ID as the Docker name;
3. entering `DockerSandbox` attaches to that container and starts it if it was
   stopped;
4. the harness creates fresh `sandbox.tools`, whose contexts point to the
   reattached container.

Do not serialize `DockerSandbox`, `Tool`, or an in-memory `CloudSession`. If
the server issues Docker-safe session IDs such as UUIDs or ULIDs, use the same
ID as the container name and keep it on exit with `remove=False`:

~~~python
from axio_tools_docker import DockerSandbox


def docker_for_session(session_id: str) -> DockerSandbox:
    return DockerSandbox(
        image="python:3.12-slim",
        name=session_id,
        remove=False,
        memory="512m",
        cpus="1.0",
        network=False,
        cap_drop=["ALL"],
        ulimits={"nofile": (256, 256), "nproc": 128},
        read_only=True,
        tmpfs={
            "/tmp": "size=64m,mode=1777",
        },
        named_volumes={"/workspace": f"{session_id}-workspace"},
        volumes_remove=False,
        workdir="/workspace",
    )
~~~

The name and volume solve different recovery cases. If the named container still
exists, the sandbox reattaches to that exact container. If it was removed, the
sandbox creates a new container and mounts the existing workspace volume. Files
under `/workspace` therefore survive either case. Packages installed elsewhere
in the old container do not survive container replacement. Bake them into the
image, or install them into an environment under `/workspace`.

If neither the container nor its volume exists, the conversation still opens
from SQLite. Execution then starts with an empty workspace. The harness
should report that state instead of pretending the filesystem was restored.

An explicit "delete session" operation must remove all three resources: the
SQLite conversation, the named container, and the named volume. Normal process
shutdown only closes the Docker client because `remove=False` keeps the runtime
available for the next process. The `session_id` in this example is an internal,
authenticated server-issued ID. If clients choose arbitrary IDs, map them to a
validated internal ID before using one as a Docker name.

Disabled networking, resource limits, and dropped capabilities are safe
defaults for model-generated code. Enable network or host mounts only for a
specific product requirement.

## 11. Add persistent conversation contexts

Open one shared database connection at service startup. `SQLiteContextStore`
already accepts `session_id`. Pass the authenticated server-side ID directly
to it. The factory is only how `CloudHarness` constructs a store on demand.
It is not another session registry:

~~~python
from axio import Agent
from axio_context_sqlite import SQLiteContextStore, connect
from axio_transport_openai import OpenAITransport


connection = await connect("data/agent.db")
prototype_agent = Agent(
    system=(
        "You are a coding assistant. Work only through the provided "
        "sandbox tools. Inspect files before changing them."
    ),
    transport=OpenAITransport(),
    tools=[],
    max_iterations=30,
)


def context_for_session(session_id: str) -> SQLiteContextStore:
    return SQLiteContextStore(
        connection,
        session_id=session_id,
        project="coding-service",
    )


harness = CloudHarness(
    prototype_agent=prototype_agent,
    context_factory=context_for_session,
    sandbox_factory=docker_for_session,
)
~~~

On restart, constructing another store with the same `session_id` is enough.
`get_history()` reads the existing messages. The same ID selects the context,
the in-process registry entry, and the Docker container. The prototype and
transport are shared. `agent.copy()` creates the per-session agent with tools
bound to that session's sandbox.

The shared SQLite connection is owned by application startup, not by an
individual context:

~~~python
try:
    await serve(harness)
finally:
    await harness.close()
    await connection.close()
~~~

## 12. Adapt events to HTTP or WebSocket

The cloud harness yields events. The delivery adapter decides their wire
format:

<!-- name: test_event_to_dict -->
```python
import base64
from dataclasses import asdict
from enum import Enum
from typing import Any

from axio import StopReason, StreamEvent, Usage
from axio.events import SessionEndEvent


def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
        }
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode(),
        }
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def event_to_dict(event: StreamEvent) -> dict[str, Any]:
    return {
        "type": type(event).__name__,
        **{
            key: json_value(value)
            for key, value in asdict(event).items()
        },
    }


encoded = event_to_dict(
    SessionEndEvent(
        stop_reason=StopReason.end_turn,
        total_usage=Usage(input_tokens=10, output_tokens=4),
    )
)
assert encoded["stop_reason"] == "end_turn"
assert encoded["total_usage"] == {
    "input_tokens": 10,
    "output_tokens": 4,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "reasoning_tokens": 0,
}
```

`Usage` carries the two totals and, beside them, the slices that bill at a
different rate: tokens served from cache, tokens written to cache, and tokens
spent on reasoning. The slices are always inside their totals, whatever the
provider's own arithmetic was, so `input_tokens` and `output_tokens` mean the
same thing for every transport.

`dataclasses.asdict()` handles the event and nested dataclasses such as
`Usage`. The small `json_value()` pass handles the three values JSON cannot
encode directly: enums, exceptions, and binary media. `Message.to_dict()` is
an Axio method for persisted conversation messages. Stream-event dataclasses
do not define that method.

The endpoint itself is only an adapter:

~~~python
async def handle_prompt(
    websocket: WebSocket,
    session_id: str,
    prompt: str,
) -> None:
    async for event in harness.stream_turn(session_id, prompt):
        await websocket.send_json(event_to_dict(event))
~~~

Authentication must determine session_id. Never trust a client to select
another user's context or volume. On disconnect, close the async generator or
cancel the request task so stream_turn reaches its finally block.

## 13. Keep the boundaries explicit

| Concern | Owner |
|---|---|
| Provider request format and SSE parsing | transport |
| Repeating after a tool result | agent |
| Tool argument schema and validation | Tool and handler |
| Tool authorization | guards and downstream service |
| Conversation messages and token counters | context store |
| External user → session mapping | cloud application |
| Per-session Docker sandbox | cloud harness |
| Streaming JSON or WebSocket protocol | delivery adapter |
| Timeouts, quotas, metrics, and eviction | harness/application |

The harness should not parse provider-specific SSE. The transport should not
know HTTP users or Docker session IDs. A tool should not decide which
conversation it belongs to. The per-session Tool.context binding answers that.

## 14. Production checklist

**Session isolation**
: Derive session IDs from authenticated state. Serialize turns sharing one
  context. Hash IDs before using them as infrastructure identifiers.

**Tool safety**
: Treat model arguments as untrusted. Use narrow schemas, guards, timeouts,
  least-privilege credentials, disabled networking, and bounded resources.

**Resource ownership**
: Open shared transports and database pools at startup. Create sandboxes per
  session. Close the harness before shared resources.

**Cancellation**
: Propagate disconnects. Always close AgentStream. Decide whether long tools
  should be cancelled or allowed to finish.

**Limits**
: Set max_iterations, a whole-turn deadline, provider timeouts, tool
  concurrency, prompt-size limits, and per-user quotas.

**Observability**
: Record stop reasons, tool duration, ToolResult.is_error, token usage, active
  sessions, sandbox startup time, and eviction without logging secrets.

**Durability**
: Decide independently how long conversation history, containers, and volumes
  live. Implement explicit delete semantics for all three.

**Testing**
: Use StubTransport for text, tool request → result → final answer, provider
  errors, malformed arguments, cancellation, concurrent sessions,
  same-session serialization, and eviction.

## Reference material

- {doc}`concepts/agent` — exact agent-loop behavior;
- {doc}`concepts/events` — every stream event;
- {doc}`concepts/tools` — schemas, validation, guards, and concurrency;
- {doc}`concepts/context` — context semantics and persistence;
- {doc}`guides/docker-sandbox` — every sandbox option;
- {doc}`guides/writing-transports` — implement a provider adapter;
- {doc}`guides/testing` — deterministic agent and harness tests.
