# Agent Swarm

This guide walks through building an **agent swarm**: a team of role-specialized AI
agents coordinated by an orchestrator, each tackling a different part of a task —
just like a real engineering team.

The full example lives in `examples/agent_swarm/` in the repository.

## What you'll build

```{mermaid}
flowchart TD
    User([User]) -->|task| O[Orchestrator]

    O -->|delegate| AR[Architect]
    O -->|delegate| BE[Backend Dev]
    O -->|delegate| QA[QA Engineer]
    O -->|delegate| SEC[Security Engineer]
    O -->|delegate| CH[Challenger]
    O -->|delegate| AN[Analyst]

    AR  -->|design.md|         WS[(workspace)]
    BE  -->|solution.py|       WS
    QA  -->|test_solution.py|  WS
    SEC -->|security_review.md| WS

    WS -->|reads artifacts| BE
    WS -->|reads artifacts| QA
    WS -->|reads artifacts| SEC
```

The **orchestrator** receives a task, decides which specialists to involve, and calls
each one in sequence (or in parallel when independent). Specialists communicate through
a shared **workspace directory**: the architect writes `design.md`, the backend dev reads
it before implementing, the QA engineer reads the implementation before writing tests.

## Project structure

The example package contains four Python modules — the CLI entry point (`__main__.py`),
the core swarm logic (`swarm.py`), the `ask_user` tool, and the `todo` tool — plus a
`roles/` subdirectory. `roles/__init__.py` declares `ROLE_NAMES` and `make_orchestrator()`;
every specialist role is a TOML file in that directory.

## 1. Defining roles

Roles are TOML files. Each file describes one specialist: its name, description,
`max_iterations`, tool list, and system prompt.

```toml
# roles/architect.toml
name = "architect"
description = "Designs system architecture, component interfaces, and documents technical decisions"
max_iterations = 100
tools = ["read_file", "write_file", "patch_file", "list_files", "shell", "run_python", "analyze"]

[system]
text = """
You are a senior software architect. When given a task you:
1. Read AGENTS.md if it exists for project history.
2. Analyse requirements and identify key components.
3. Write design.md with component map, interfaces, and contracts.
...
"""
```

The `tools` list contains names that are resolved against the shared toolbox at
runtime — the TOML file does not know about `Tool` objects, only names.

`roles/__init__.py` derives the role list from TOML filenames and builds the
orchestrator with a dynamic roster:

```python
from pathlib import Path
from axio.agent import Agent
from axio.transport import DummyCompletionTransport

ROLES_DIR = Path(__file__).parent
ROLE_NAMES = [p.stem for p in sorted(ROLES_DIR.glob("*.toml"))]

def make_orchestrator(roster: str) -> Agent:
    return Agent(
        max_iterations=200,
        system=f"""You are a tech lead managing a team of specialist agents.
...
Available team members
----------------------
{roster}
...""",
        transport=DummyCompletionTransport(),
    )
```

`make_orchestrator()` is called from `run_swarm()` after `load_agents()` runs,
so the roster is always derived from actual loaded agents.

**Adding a new role** is a single-file change: create `roles/new_role.toml`. No
changes to Python files needed — `ROLE_NAMES` is derived from filenames automatically.

## 2. Loading roles at runtime

`load_agents()` from `axio.agent_loader` scans a directory for TOML/JSON/INI files,
resolves tool names against a toolbox dict, and returns
`dict[str, tuple[str, Agent]]`.

`build_toolbox()` in `swarm.py` constructs the shared tool registry before `load_agents()` is called:

```python
from axio.agent_loader import load_agents

toolbox = build_toolbox(workspace, on_event, transport, role_models, guard_factory)
roles = load_agents(ROLES_DIR, toolbox=toolbox)
# roles == {"architect": ("Designs system...", Agent(...)), "backend_dev": (...), ...}
```

The toolbox contains all tools any specialist might need:
`read_file`, `write_file`, `patch_file`, `list_files`, `shell`, `run_python`, and `analyze`.
Each tool is created once and shared across roles — guard tuples are attached per tool.

## 3. Activating a role with `copy()`

`Agent` is a frozen dataclass. `copy()` applies field overrides without mutation.
The Delegate tool activates a specialist by copying its prototype:

```python
description, proto = roles[self.role]
role_transport = transport_for(self.role, context["transport"], context["role_models"])
specialist = proto.copy(transport=role_transport)
specialist_ctx = AutoCompactStore(MemoryContextStore(), role_transport, keep_recent=6)
stream = specialist.run_stream(
    f"Workspace: {context['workspace']}\n\n{self.task}",
    specialist_ctx,
)
```

`transport_for()` returns a shallow copy of the base transport with the role's model applied:

```python
def transport_for(role, base, role_models):
    model = role_models.get(role) or role_models["default"]
    new_transport = copy.copy(base)
    new_transport.model = model
    return new_transport
```

This preserves the shared HTTP session while giving each agent a different model.

## 4. Tool contexts via TypedDict

Tools that spawn sub-agents carry their runtime dependencies in a typed context dict.
`Delegate` is a top-level `ToolHandler[DelegateContext]` — not a closure:

```python
class DelegateContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    roles: dict[str, tuple[str, Agent]]
    guard_factory: GuardFactory | None
    counters: dict[str, int]


class Delegate(ToolHandler[DelegateContext]):
    """Delegate a task to a specialist team member."""

    role: Annotated[
        str,
        Field(
            description=f"Which specialist to delegate to. One of: {', '.join(ROLE_NAMES)}",
            json_schema_extra={"enum": ROLE_NAMES},
        ),
    ]
    topic: Annotated[str, Field(description="Short label, e.g. 'auth middleware'")]
    task: Annotated[str, Field(description="Instructions for the specialist")]

    async def __call__(self, context: DelegateContext) -> str:
        ...
```

`json_schema_extra={"enum": ROLE_NAMES}` makes the JSON schema expose a strict enum
so the LLM cannot hallucinate a role name.

## 5. The Analyze tool

The swarm includes a read-only `analyze` tool that spawns ephemeral analyst subagents
for investigation tasks. Analysts can only read files — no write tools.

```python
class AnalyzeContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    guard_factory: GuardFactory | None
    counter: list[int]  # [0] holds mutable call count


ANALYST = Agent(
    system="You are a read-only analyst. Read files and produce a concise report. "
           "You must not create, modify, or delete any files.",
    transport=DummyCompletionTransport(),
)


class Analyze(ToolHandler[AnalyzeContext]):
    """Spawn a read-only analyst subagent to investigate a question and return a report."""
    task: Annotated[str, Field(description="Question or analysis task")]

    async def __call__(self, context: AnalyzeContext) -> str:
        context["counter"][0] += 1
        n = context["counter"][0]
        agent_id = f"analyst#{n}:{self.task[:40]}"

        analyst = ANALYST.copy(
            transport=transport_for("analyst", context["transport"], context["role_models"]),
            tools=[list_files_tool, read_file_tool],
            max_iterations=10,
        )
        stream = analyst.run_stream(
            f"Workspace: {context['workspace']}\n\n{self.task}",
            MemoryContextStore(),
        )
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)
```

Both the orchestrator and specialists get an `analyze` tool. Multiple analyst
instances can run concurrently — Axio dispatches all tool calls in one response via
`asyncio.gather()`.

## 6. Streaming sub-agent output

`run_stream()` returns an async iterator of `StreamEvent`. We iterate manually to
both forward events to the display and collect the final text:

```python
stream = specialist.run_stream(task, specialist_ctx)
parts: list[str] = []
async for event in stream:
    await context["on_event"](agent_id, event)
    if isinstance(event, TextDelta):
        parts.append(event.delta)
return "".join(parts)
```

`on_event` is async — the renderer holds an `asyncio.Lock` so concurrent delegates
don't interleave their output.

## 7. Guards for logging and auditing

`PermissionGuard.check()` receives the fully-parsed `ToolHandler` instance (all fields
already validated) and runs **before** the tool executes. This is the right place
for logging, auditing, or display — not a separate event stream:

```python
class RoleGuard(PermissionGuard):
    def __init__(self, role: str, tool_name: str, renderer: SwarmRenderer) -> None:
        self._role = role
        self._tool_name = tool_name
        self._renderer = renderer

    async def check(self, handler: ToolHandler) -> ToolHandler:
        async with self._renderer._lock:   # same lock as on_event
            self._renderer._print_tool_call(self._role, self._tool_name, handler)
        return handler                     # return to allow; raise GuardError to deny
```

`SwarmRenderer` exposes a factory method passed into `run_swarm()`:

```python
await run_swarm(
    task=args.task,
    workspace=workspace,
    on_event=renderer.on_event,
    transport=transport,
    role_models=role_models,
    guard_factory=renderer.make_guard,
    prompt_fn=renderer.make_prompt_fn(),
)
```

## 8. Orchestrator tools

The orchestrator gets five tools:

| Tool | Purpose |
|---|---|
| `delegate` | Spawn a specialist for a task |
| `ask_user` | Ask the user a question before starting work |
| `todo` | SQLite-backed task list (list/add/update) |
| `analyze` | Spawn read-only analyst subagents |
| `list_files`, `read_file` | Browse the workspace |

The `todo` tool persists to `workspace/.axio-swarm/todos.db` and survives restarts.
The `ask_user` tool pauses the Rich `Live` display during input.

## 9. Rich output

`SwarmRenderer` accumulates `TextDelta` events in a per-role buffer and renders the
complete text as Markdown when the role finishes speaking (on `ToolUseStart` or
`SessionEndEvent`). `_handle()` uses `match/case` on `StreamEvent` subtypes:

```python
match event:
    case ReasoningDelta():
        self._print(f"[dim italic]{event.delta}[/dim italic]", end="")
    case TextDelta():
        self._text_buf.setdefault(role, []).append(event.delta)
    case ToolUseStart():
        self._flush_text(role)
        self._agent_status[role] = f"▶ {event.name}"
    case ToolResult():
        ...
    case IterationEnd():
        u = event.usage
        self._print(f"[dim]  iter {event.iteration} · {event.stop_reason} "
                    f"· ↑{u.input_tokens} ↓{u.output_tokens}[/dim]")
    case SessionEndEvent():
        ...
```

A `StatusBar` renderable passed once to `Live()` at construction reads renderer state
at Rich's refresh rate (4 fps) — shows active agents with spinners and a running
event/token counter.

## 10. Running it

```bash
cd examples/agent_swarm
uv sync
export NEBIUS_API_KEY=...   # Nebius AI Studio

uv run python -m agent_swarm --workspace /tmp/my_project \
    "Build a Python rate limiter with token-bucket and sliding-window strategies"
```

Model assignment in `__main__.py` — edit `role_models` to change per-role models:

```python
role_models: dict[str, ModelSpec] = {
    "default":          transport.models["MiniMaxAI/MiniMax-M2.5"],
    "architect":        transport.models["Qwen/Qwen3-235B-A22B-Instruct-2507"],
    "security_engineer": transport.models["openai/gpt-oss-120b"],
    "project_manager":  transport.models["openai/gpt-oss-120b"],
    "challenger":       transport.models["zai-org/GLM-5"],
    # analyst runs many instances in parallel — use a fast model
    "analyst":          transport.models["deepseek-ai/DeepSeek-V3.2"],
}
```

`transport.models` is a `ModelRegistry` populated from the Nebius API after
`fetch_models()`. Index by exact model ID or use `.search("substring").first()`.

After the run the workspace contains all produced artifacts: `AGENTS.md` (living project
memory), `design.md` (architect), implementation files (backend/frontend developers),
tests (qa), and review reports (security_engineer, challenger). The `.axio-swarm/`
subdirectory holds internal orchestration data — the todo SQLite database and per-role
analysis reports — and should not be treated as project output.

## 11. Extending the team

Add a role by creating one new TOML file:

```toml
# roles/data_scientist.toml
name = "data_scientist"
description = "Explores data, builds models, and produces analytical reports"
max_iterations = 50
tools = ["read_file", "write_file", "list_files", "shell", "run_python", "analyze"]

[system]
text = """
You are a senior data scientist...
"""
```

`ROLE_NAMES` is derived from TOML filenames — the orchestrator's roster and the
`Delegate` enum update automatically. No Python changes needed.

## When to use this pattern

Use agent swarms when:

- The task genuinely benefits from different expertise applied in parallel
  (design → implement → test → review).
- You want **isolation**: each agent starts with a clean context and cannot
  accidentally carry state from a previous subtask.
- You need **parallel work**: the orchestrator can delegate to all independent
  agents simultaneously — frontend + backend + security all run at once if the
  LLM issues multiple tool calls in one response.

For simpler cases — a single agent that can call tools to break down its own work —
the built-in `subagent` tool from `axio-tui` is sufficient. The swarm pattern adds
structure at the cost of more orchestration prompting and more LLM calls.

```{seealso}
- {doc}`gas-town` — the bead-based convoy pattern with explicit work tracking
- {doc}`writing-tools` — how to build custom ToolHandlers
- {doc}`testing` — how to unit-test agents and tools with StubTransport
- {ref}`concepts/agent` — the agent loop in detail
```
