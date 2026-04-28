# AGENTS.md

Guidance for AI coding agents working in this repository.

---

## Repository layout

This is a **uv workspace** monorepo. Each subdirectory is an independent Python package with its own `pyproject.toml`, `src/` layout, and `tests/`. They share a single `uv.lock` and a single `.venv`.

- `axio/` - core library; start here when unsure
- `axio-tui/` - TUI application and plugin discovery
- `axio-transport-*/` - transport implementations
- `axio-tools-*/` - tool providers
- `axio-context-sqlite/` - SQLite context store
- `axio-tui-guards/` - permission guard plugins
- `docs/` - Sphinx sources and markdown-pytest doc tests
- `examples/` - runnable example scripts

| Package | Purpose |
|---|---|
| `axio` | Agent loop, Tool, Transport protocol, ContextStore, PermissionGuard, events, blocks, types |
| `axio-transport-openai` | OpenAI-compatible transport (OpenAI, Nebius, OpenRouter, custom) |
| `axio-transport-anthropic` | Anthropic Claude transport with prompt caching |
| `axio-transport-codex` | ChatGPT via OAuth Responses API |
| `axio-tools-local` | File, shell, Python execution tools |
| `axio-tools-mcp` | MCP server bridge |
| `axio-tools-docker` | Docker sandbox tool provider |
| `axio-context-sqlite` | SQLite-backed persistent context store |
| `axio-tui` | Textual TUI, SQLite context store, plugin discovery |
| `axio-tui-guards` | PathGuard + LLMGuard plugins |

---

## Commands

Always use `make`. Never invoke `uv run pytest`, `ruff`, or `mypy` directly at the repo root.

```bash
make              # lint + type-check + tests for all packages + doc tests
make linter       # ruff check + ruff format --check on all packages
make typing       # mypy --strict on all packages
make pytest       # pytest on all packages
make test-docs    # markdown-pytest on docs/
```

Run a single package's tests or checks:

```bash
make PACKAGES=axio-transport-anthropic
```

Run a specific test file inside a package:

```bash
uv run --directory axio pytest tests/test_agent_run.py -v
```

Run doc tests for a single file:

```bash
uv run --directory docs pytest -v guides/best-practices.md
```

---

## Development setup

```bash
git clone https://github.com/mosquito/axio-agent.git
cd axio-agent
uv sync --all-packages   # installs all workspace members + dev deps into .venv
```

After sync, all local packages resolve to their workspace sources via `[tool.uv.sources]` - no `pip install -e` or PYTHONPATH hacks needed.

---

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/), `line-length = 119`, `target-version = "py312"`
- **Type checker**: [mypy](https://mypy.readthedocs.io/) strict mode (`--strict`), `python_version = "3.12"`
- Enabled ruff rules: `E`, `F`, `I`, `UP`
- All new code must pass `mypy --strict` with zero errors
- Use `from __future__ import annotations` at the top of every module
- Prefer `dataclass(frozen=True, slots=True)` for value types

Always run `make linter` and `make typing` before considering a task done.

---

## Architecture

### Public API (`axio/__init__.py`)

Common symbols are importable directly from `axio`:

```python
from axio import Agent, Tool, Field, PermissionGuard, IterationEnd
from axio import StopReason, Usage, GuardError, HandlerError
from axio import ContextStore, MemoryContextStore, CompletionTransport
```

Submodule-only (not re-exported at top level):

```python
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.schema import build_tool_schema
from axio.agent_loader import AgentSpec, load_agents
from axio.compaction import AutoCompactStore, CompactionStrategy
```

### Types (`axio/types.py`)

Primitive type aliases and enums:

```python
type ToolName = str          # Unique tool identifier
type ToolCallID = str         # Opaque ID for a tool invocation

class StopReason(StrEnum):
    end_turn = "end_turn"     # Assistant finished responding
    tool_use = "tool_use"     # Assistant wants to call a tool
    max_tokens = "max_tokens" # Output truncated
    error = "error"           # Something went wrong

@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    # Supports arithmetic: usage1 + usage2
```

### Messages (`axio/messages.py`)

The fundamental unit of conversation history:

```python
@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock]  # List of Text/Image/ToolUse/ToolResult blocks

    # Serialization
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message: ...
```

Messages are serialized for transport to LLM providers and for persistence in context stores.

### Content Blocks (`axio/blocks.py`)

Four block types represent all content in messages:

| Block | Fields | Purpose |
|---|---|---|
| `TextBlock` | `text: str` | Plain text content |
| `ImageBlock` | `media_type: Literal[image/*]`, `data: bytes` | Image attachments (base64 encoded in serialization) |
| `ToolUseBlock` | `id: ToolCallID`, `name: ToolName`, `input: dict[str, Any]` | A tool call request |
| `ToolResultBlock` | `tool_use_id: ToolCallID`, `content: str | list[Block]`, `is_error: bool` | Result of tool execution |

```python
# Serialization helpers
def to_dict(block: ContentBlock) -> dict[str, Any]: ...
def from_dict(data: dict[str, Any]) -> ContentBlock: ...
```

Both functions handle the full round-trip including nested blocks in `ToolResultBlock`.

### Events (`axio/events.py`)

All events are `dataclass(frozen=True, slots=True)`. `StreamEvent` is a type union of all event types:

| Event | Fields | When |
|---|---|---|
| `TextDelta(index, delta)` | `index: int`, `delta: str` | Streamed text chunk |
| `ReasoningDelta(index, delta)` | `index: int`, `delta: str` | Streamed reasoning/thinking chunk |
| `ToolUseStart(index, tool_use_id, name)` | `index: int`, `tool_use_id: ToolCallID`, `name: ToolName` | Tool call begins |
| `ToolInputDelta(index, tool_use_id, partial_json)` | `index: int`, `tool_use_id: ToolCallID`, `partial_json: str` | Streaming tool arguments (JSON string) |
| `ToolResult(tool_use_id, name, is_error, content, input)` | Various | Tool execution result (emitted by agent, not transport) |
| `IterationEnd(iteration, stop_reason, usage)` | `iteration: int`, `stop_reason: StopReason`, `usage: Usage` | One LLM call complete |
| `SessionEndEvent(stop_reason, total_usage)` | `stop_reason: StopReason`, `total_usage: Usage` | Full agent run complete |
| `Error(exception)` | `exception: Exception` | Unhandled exception in the stream |

**Important**: The stream **must** end with exactly one `IterationEnd` event per LLM call.

### Transport protocol (`axio/transport.py`)

`CompletionTransport` is a `@runtime_checkable` Protocol. Implement one method:

```python
@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]: ...
```

Contract:
- The stream **must** end with exactly one `IterationEnd` event
- Do not suppress exceptions - let them propagate as `Error` events or raise naturally
- `messages` contains the full conversation history except the current user input (already appended)
- `tools` is the Tool definitions from the Agent's registry

### Agent loop (`axio/agent.py`)

`Agent` is a `dataclass(slots=True)` with fields:

```python
@dataclass(slots=True)
class Agent:
    system: str                              # System prompt
    transport: CompletionTransport          # LLM backend
    tools: list[Tool]                        # Available tools
    selector: ToolSelector | None = None     # Optional tool selection logic
    max_iterations: int = 50                # Iteration limit
    last_iteration_message: Message | None = None  # Set after run completes
```

Public API:
- `agent.run_stream(user_message, context) -> AgentStream` - streaming entry point, returns an async iterator that yields `StreamEvent` plus `SessionEndEvent`
- `await agent.run(user_message, context) -> str` - convenience wrapper, returns final text

**Loop per iteration**:
1. Call `transport.stream(history, tools, system)` → `AsyncIterator[StreamEvent]`
2. Accumulate `TextDelta` and `ToolUseStart`/`ToolInputDelta` from events
3. On `IterationEnd(stop_reason=tool_use)`:
   - Parse fully accumulated tool calls
   - Dispatch all pending tool calls **concurrently** via `asyncio.gather()`
   - Append results as `ToolResultBlock` messages to context
   - Loop
4. On `IterationEnd(stop_reason=end_turn)`:
   - Append assistant message with text content
   - Emit `SessionEndEvent`
   - Return

Tool dispatch happens **before** appending to context. This prevents orphaned `ToolUseBlock`s if a task is cancelled mid-loop.

### Tool system (`axio/tool.py`)

A tool handler is a plain `async def` function. Parameters become the input JSON schema; the docstring becomes the description. Use `Annotated` + `Field` from `axio.field` to add per-parameter descriptions, defaults, or numeric bounds.

```python
async def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    Path(path).write_text(content)
    return f"wrote {len(content)} bytes"
```

`Tool` is a `dataclass(frozen=True, slots=True)`:

```python
@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: ToolName                         # Unique identifier
    handler: Callable[..., Awaitable[str]] # Plain async function
    description: str = ""                  # Defaults to handler.__doc__
    guards: tuple[PermissionGuard, ...] = ()  # Run sequentially; any GuardError denies
    concurrency: int | None = None         # Optional per-tool semaphore limit
    context: T = ...                       # Runtime state for CONTEXT.get()
```

`Tool.__call__(**kwargs)` pipeline:
1. Acquire semaphore (if `concurrency` is set)
2. Field validation from type hints / `FieldInfo`
3. Guards run **sequentially**: each receives `(tool, **kwargs)` and returns modified kwargs or raises `GuardError`
4. `await handler(**kwargs)` - execute handler

### ContextStore (`axio/context.py`)

`ContextStore` is an ABC. Implement all methods:

```python
class ContextStore(ABC):
    @abstractmethod
    async def append(self, message: Message) -> None: ...

    @abstractmethod
    async def get_history(self) -> list[Message]: ...

    @abstractmethod
    async def compact(self, summary: str) -> None: ...
        """Replace history with a summary message."""
```

**Implementations**:
- `MemoryContextStore` - in-process, ephemeral storage
- `axio-context-sqlite` - `SQLiteContextStore` for persistence across sessions

### PermissionGuard (`axio/permission.py`)

`PermissionGuard` is an ABC. Implement:

```python
class PermissionGuard(ABC):
    @abstractmethod
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        """Return (possibly modified) kwargs to allow, raise GuardError to deny."""
        ...
```

Guards are not limited to access control. Because `check()` receives the `Tool` object
and the raw kwargs before the handler executes, guards are also the right place for
**logging, auditing, and display**:

```python
class AuditGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        logger.info("tool=%s args=%s", tool.name, kwargs)
        return kwargs  # always allow; raise GuardError to deny
```

See `examples/agent_swarm/agent_swarm/__main__.py` (`RoleGuard`) for a production example.

**ConcurrentGuard**: Use as base when the guard itself must be rate-limited (e.g. LLM-based approval). Provides internal semaphore management.

### Exceptions (`axio/exceptions.py`)

Full exception hierarchy:

```python
class AxioError(Exception):
    """Base exception for all axio errors."""

class ToolError(AxioError):
    """Base for tool-related errors."""

class GuardError(ToolError):
    """Guard denied or crashed during permission check."""

class HandlerError(ToolError):
    """Handler raised during execution."""

class StreamError(AxioError):
    """Error during stream collection."""
```

### Testing (`axio/testing.py`)

Helper classes and functions for testing:

| Function | Returns | Purpose |
|---|---|---|
| `StubTransport(responses)` | `StubTransport` | Yields pre-configured event sequences per `stream()` call |
| `make_text_response(text, iteration, usage)` | `list[StreamEvent]` | Build a simple end_turn response |
| `make_tool_use_response(tool_name, tool_id, tool_input, iteration, usage)` | `list[StreamEvent]` | Build a tool_use response sequence |
| `make_stub_transport()` | `StubTransport` | Pre-configured with "Hello world" text response |
| `make_ephemeral_context()` | `MemoryContextStore` | Fresh empty context |
| `make_echo_tool()` | `Tool` | Tool with a plain async handler that returns its `msg` arg as JSON |

```python
# Example: StubTransport with multiple responses
transport = StubTransport([
    make_tool_use_response("my_tool", tool_input={"x": 1}),
    make_text_response("Done"),
])
agent = Agent(system="...", transport=transport, tools=[my_tool])
result = await agent.run("go", make_ephemeral_context())
```

`StubTransport` pops the next event sequence on each `stream()` call. If there are fewer sequences than calls, it repeats the last one.

### Plugin system (entry points)

Plugins register via `pyproject.toml` entry points and are discovered by `axio-tui` at startup:

| Group | Registers |
|---|---|
| `axio.tools` | plain async handler functions |
| `axio.tools.settings` | `ToolsPlugin` (dynamic tool sets, e.g. MCP) |
| `axio.transport` | `CompletionTransport` implementations |
| `axio.transport.settings` | TUI settings screens (Textual `Screen` subclasses) |
| `axio.guards` | `PermissionGuard` subclasses |

---

## Testing

### Unit tests

Each package has `tests/` with pytest. Test files follow the pattern `test_<module>.py`.

Use helpers from `axio.testing`:

```python
from axio.testing import (
    StubTransport,          # pre-configured event sequences
    make_text_response,     # build an end_turn event list
    make_tool_use_response, # build a tool_use event list
    make_stub_transport,    # StubTransport with a single "Hello world" response
    make_ephemeral_context, # fresh MemoryContextStore
    make_echo_tool,         # Tool(name="echo", handler=MsgInput)
)
```

`StubTransport` pops the next event sequence on each `stream()` call. If there are fewer sequences than calls, it repeats the last one.

```python
transport = StubTransport([
    make_tool_use_response("my_tool", tool_input={"x": 1}),
    make_text_response("Done"),
])
agent = Agent(system="...", transport=transport, tools=[my_tool])
result = await agent.run("go", make_ephemeral_context())
```

### Doc tests

Documentation in `docs/` is tested with [markdown-pytest](https://github.com/mosquito/markdown-pytest). Annotate code blocks with HTML comments:

```markdown
<!-- name: test_my_example -->
```python
import asyncio
from axio.agent import Agent
# ... asyncio.run() for async code
```
```

Hidden setup (stubs that must not appear in rendered docs):

```markdown
<!--
name: test_my_example
```python
# This block is invisible in docs but runs before the named block
from axio.testing import StubTransport, make_text_response
```
-->
```

Run doc tests:

```bash
make test-docs
# or for a single file:
uv run --directory docs pytest -v guides/writing-transports.md
```

---

## Adding a new package

1. Create `axio-<name>/` with a `src/axio_<name>/` layout and a `pyproject.toml` matching the style of existing packages (hatchling build backend, ruff + mypy + pytest dev deps, `asyncio_mode = "auto"`).
2. Add to `[tool.uv.workspace] members` and `[tool.uv.sources]` in the root `pyproject.toml`.
3. Add to `PACKAGES` in `Makefile`.
4. Run `uv sync --all-packages` to update `uv.lock`.

---

## What not to do

- **Do not** run `uv run pytest` or `ruff` from the repo root - use `make`.
- **Do not** add dependencies to the root `pyproject.toml` - it is a workspace manifest only.
- **Do not** edit `uv.lock` manually - it is generated by `uv sync`.
- **Do not** use `asyncio_mode = "auto"` in ad-hoc scripts - it is only for pytest.
- **Do not** add a guard's blocking I/O in the hot path without subclassing `ConcurrentGuard`.