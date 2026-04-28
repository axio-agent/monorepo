# Agent Swarm

This guide walks through building an **agent swarm**: a team of role-specialized AI
agents coordinated by an orchestrator, each tackling a different part of a task -
just like a real engineering team.

The full example lives in `examples/agent_swarm/` in the repository.

## Prerequisites

- **Docker** must be installed and running. The swarm executes all file and shell
  operations inside a Docker container - agents never touch the host filesystem
  directly. Install Docker from <https://docs.docker.com/get-docker/>.
- A Nebius AI Studio API key (`NEBIUS_API_KEY`).

## What you'll build

```{mermaid}
flowchart TD
    User --> Orchestrator
    Orchestrator -->|delegate| Architect
    Orchestrator -->|delegate| B[Backend Dev]
    Orchestrator -->|delegate| Q[QA Engineer]
    Orchestrator -->|delegate| S[Security Engineer]
    Orchestrator -->|delegate| Challenger
    Orchestrator -->|delegate| Analyst
    Architect -->|design| W[Workspace]
    B -->|code| W
    Q -->|tests| W
    S -->|review| W
    W -->|reads| B
    W -->|reads| Q
    W -->|reads| S
```

The **orchestrator** receives a task, decides which specialists to involve, and calls
each one in sequence (or in parallel when independent). Specialists communicate through
a shared **workspace directory**: the architect writes `design.md`, the backend dev reads
it before implementing, the QA engineer reads the implementation before writing tests.

## Project structure

The example package contains four Python modules - the CLI entry point (`__main__.py`),
the core swarm logic (`swarm.py`), the `ask_user` tool, and the `todo` tool - plus a
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
tools = ["read_file", "write_file", "patch_file", "list_files", "shell", "run_python", "analyze", "notes"]

[system]
text = """
You are a senior software architect. When given a task you:
1. Read AGENTS.md if it exists for project history.
2. Analyse requirements and identify key components.
3. Write design.md with component map, interfaces, and contracts.
...
Notes
-----
Use `notes` to persist findings, decisions, or summaries that matter beyond this task.
If you wrote or updated a note, say so explicitly in your response.
"""
```

The `tools` list contains names that are resolved against the shared toolbox at
runtime - the TOML file does not know about `Tool` objects, only names.
`analyze` and `notes` are added to the toolbox by `run_swarm()` after the sandbox
tools are injected.

`roles/__init__.py` derives the role list from TOML filenames and builds the
orchestrator with a dynamic roster:

```python
from pathlib import Path
from axio import Agent
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
changes to Python files needed - `ROLE_NAMES` is derived from filenames automatically.

## 2. Loading roles at runtime

`load_agents()` from `axio.agent_loader` scans a directory for TOML/JSON/INI files,
resolves tool names against a toolbox dict, and returns
`dict[str, tuple[str, Agent]]`.

The toolbox is built from a **`DockerSandbox`** - every file and shell operation
runs inside an isolated container mounted on the workspace directory:

```python
from axio_tools_docker.sandbox import DockerSandbox
from axio.agent_loader import load_agents

async with DockerSandbox(
    image="python:3.12-slim",
    volumes={"/workspace": str(workspace)},
    workdir="/workspace",
    name=sandbox_name,   # reattach to the same container on resume
    remove=False,        # keep container alive between sessions
) as sandbox:
    toolbox = {t.name: t for t in sandbox.tools}
    # toolbox == {"read_file": Tool(...), "write_file": Tool(...),
    #             "patch_file": Tool(...), "list_files": Tool(...),
    #             "shell": Tool(...), "run_python": Tool(...)}
    ...
```

`run_swarm()` then extends the toolbox in-place with runtime-only tools before
calling `load_agents()`:

```python
toolbox["analyze"] = make_analyze_tool(toolbox, ...)
toolbox["notes"]   = make_notes_tool(workspace)
roles = load_agents(ROLES_DIR, toolbox=toolbox)
# roles == {"architect": ("Designs system...", Agent(...)), "backend_dev": (...), ...}
```

Sandbox tools bind to the running container - all file paths are resolved relative
to `/workspace` inside the container, which is mounted from the host workspace directory.

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
`delegate` is a top-level plain async function - not a closure:

```python
class DelegateContext(TypedDict):
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    roles: dict[str, tuple[str, Agent]]
    guard_factory: GuardFactory | None
    counters: dict[str, int]


async def delegate(
    role: Annotated[
        str,
        Field(
            description=f"Which specialist to delegate to. One of: {', '.join(ROLE_NAMES)}",
            json_schema_extra={"enum": ROLE_NAMES},
        ),
    ],
    topic: Annotated[str, Field(description="Short label, e.g. 'auth middleware'")],
    task: Annotated[str, Field(description="Instructions for the specialist")],
) -> str:
    """Delegate a task to a specialist team member."""
    context: DelegateContext = CONTEXT.get()
    ...
    stream = specialist.run_stream(
        f"Workspace: {WORKDIR}\n\n{task}",   # WORKDIR = "/workspace"
        specialist_ctx,
    )
```

`workspace` is gone from `DelegateContext` - the container path `/workspace` is a
module-level constant (`WORKDIR`). The workspace directory is always the same inside
the container regardless of where it is on the host.

`json_schema_extra={"enum": ROLE_NAMES}` makes the JSON schema expose a strict enum
so the LLM cannot hallucinate a role name.

## 5. The Analyze tool

The swarm includes a read-only `analyze` tool that spawns ephemeral analyst subagents
for investigation tasks. Analysts can only read files - no write tools.

```python
class AnalyzeContext(TypedDict):
    toolbox: dict[str, Tool[Any]]  # shared toolbox; analyst uses list_files + read_file
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    guard_factory: GuardFactory | None
    counter: list[int]  # [0] holds mutable call count


async def analyze(
    task: Annotated[str, Field(description="Question or analysis task")],
) -> str:
    """Spawn a read-only analyst subagent to investigate a question and return a report."""
    context: AnalyzeContext = CONTEXT.get()
    context["counter"][0] += 1
    n = context["counter"][0]
    agent_id = f"analyst#{n}:{task[:40]}"

    tb = context["toolbox"]
    read_tools = [tb[k] for k in ("list_files", "read_file") if k in tb]
    analyst = ANALYST.copy(
        transport=transport_for("analyst", context["transport"], context["role_models"]),
        tools=read_tools,   # docker-bound tools from the shared sandbox
        max_iterations=10,
    )
    stream = analyst.run_stream(
        f"Workspace: {WORKDIR}\n\n{task}",
        MemoryContextStore(),
    )
    parts: list[str] = []
    async for event in stream:
        await context["on_event"](agent_id, event)
        if isinstance(event, TextDelta):
            parts.append(event.delta)
    return "".join(parts)
```

`workspace: Path` is replaced by `toolbox` - analyst read tools come directly from
the sandbox toolbox and are already bound to the running container. Both the
orchestrator and specialists get an `analyze` tool. Multiple analyst instances can
run concurrently - Axio dispatches all tool calls in one response via
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

`on_event` is async - the renderer holds an `asyncio.Lock` so concurrent delegates
don't interleave their output.

## 7. Guards for logging and auditing

`PermissionGuard.check()` receives the `Tool` object and raw kwargs (already
validated) and runs **before** the tool executes. This is the right place
for logging, auditing, or display - not a separate event stream:

```python
class RoleGuard(PermissionGuard):
    def __init__(self, role: str, tool_name: str, renderer: SwarmRenderer) -> None:
        self._role = role
        self._tool_name = tool_name
        self._renderer = renderer

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        async with self._renderer._lock:   # same lock as on_event
            self._renderer._print_tool_call(self._role, self._tool_name, kwargs)
        return kwargs                      # return to allow; raise GuardError to deny
```

`SwarmRenderer` exposes a factory method passed into `run_swarm()`:

```python
async with DockerSandbox(..., name=sandbox_name, remove=False) as sandbox:
    toolbox = {t.name: t for t in sandbox.tools}
    await run_swarm(
        task=args.task,
        workspace=workspace,
        on_event=renderer.on_event,
        transport=transport,
        role_models=role_models,
        toolbox=toolbox,
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
| `notes` | Persist findings across iterations and sessions |

The orchestrator does **not** have `read_file` or `list_files` - it uses `analyze` for
all file investigation. Specialists receive those tools from the sandbox toolbox.

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
at Rich's refresh rate (4 fps) - shows active agents with spinners and a running
event/token counter.

## 10. Running it

```bash
cd examples/agent_swarm
uv sync
export NEBIUS_API_KEY=...   # Nebius AI Studio

uv run python -m agent_swarm --workspace /tmp/my_project \
    "Build a Python rate limiter with token-bucket and sliding-window strategies"
```

Docker is required. The swarm creates a container on first run and saves its name to
`workspace/.axio-swarm/sandbox`. On the next run you are asked whether to resume the
same container (useful to keep installed packages and build artifacts):

```
Existing sandbox: axio-swarm-ce744057
Resume this container? [Y/n]
```

Press Enter or `y` to reattach, `n` to start a fresh container.

**Docker options:**

| Flag | Default | Description |
|---|---|---|
| `--image` | `python:3.12-slim` | Container image |
| `--memory` | `512m` | Memory limit (e.g. `1g`) |
| `--cpus` | `2.0` | CPU limit |
| `--network` | off | Enable network access inside the container |

```bash
axio-swarm --workspace /tmp/my_project \
    --image python:3.12-slim --memory 1g --cpus 4 \
    "Build a Python rate limiter with token-bucket and sliding-window strategies"
```

Model assignment in `__main__.py` - edit `role_models` to change per-role models:

```python
role_models: dict[str, ModelSpec] = {
    "default":          transport.models["MiniMaxAI/MiniMax-M2.5"],
    "architect":        transport.models["Qwen/Qwen3-235B-A22B-Instruct-2507"],
    "security_engineer": transport.models["openai/gpt-oss-120b"],
    "project_manager":  transport.models["openai/gpt-oss-120b"],
    "challenger":       transport.models["zai-org/GLM-5"],
    # analyst runs many instances in parallel - use a fast model
    "analyst":          transport.models["deepseek-ai/DeepSeek-V3.2"],
}
```

`transport.models` is a `ModelRegistry` populated from the Nebius API after
`fetch_models()`. Index by exact model ID or use `.search("substring").first()`.

After the run the workspace directory (host-side) contains all produced artifacts:
`AGENTS.md` (living project memory), `design.md` (architect), implementation files
(backend/frontend developers), tests (qa), and review reports (security_engineer,
challenger). The `.axio-swarm/` subdirectory holds internal orchestration data - the
todo SQLite database, per-role analysis reports, notes, and the sandbox container
name - and should not be treated as project output.

## 11. Extending the team

Add a role by creating one new TOML file:

```toml
# roles/data_scientist.toml
name = "data_scientist"
description = "Explores data, builds models, and produces analytical reports"
max_iterations = 50
tools = ["read_file", "write_file", "list_files", "shell", "run_python", "analyze", "notes"]

[system]
text = """
You are a senior data scientist...
"""
```

`ROLE_NAMES` is derived from TOML filenames - the orchestrator's roster and the
`Delegate` enum update automatically. No Python changes needed.

## When to use this pattern

Use agent swarms when:

- The task genuinely benefits from different expertise applied in parallel
  (design → implement → test → review).
- You want **isolation**: each agent starts with a clean context and cannot
  accidentally carry state from a previous subtask.
- You need **parallel work**: the orchestrator can delegate to all independent
  agents simultaneously - frontend + backend + security all run at once if the
  LLM issues multiple tool calls in one response.

For simpler cases - a single agent that can call tools to break down its own work -
the built-in `subagent` tool from `axio-tui` is sufficient. The swarm pattern adds
structure at the cost of more orchestration prompting and more LLM calls.

```{seealso}
- {doc}`gas-town` - the bead-based convoy pattern with explicit work tracking
- {doc}`writing-tools` - how to build custom tools
- {doc}`testing` - how to unit-test agents and tools with StubTransport
- {doc}`../concepts/agent` - the agent loop in detail
```
