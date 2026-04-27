# Gas Town

This guide walks through the **Gas Town** example: a multi-agent convoy system
modelled on Steve Yegge's [Gas Town](https://github.com/gastownhall/gastown) methodology.

The full example lives in `examples/gas_town/` in the repository.
For a deep dive into the methodology itself, see {doc}`../gas-town-rules`.

## Prerequisites

- **Docker** must be installed and running. All file and shell operations run inside
  a Docker container — agents never touch the host filesystem directly.
  Install Docker from <https://docs.docker.com/get-docker/>.
- A Nebius AI Studio API key (`NEBIUS_API_KEY`).

## What is Gas Town?

Gas Town is an opinionated orchestration model with a few core ideas:

- **Beads** are the atomic unit of work — a SQLite-backed issue with id, title,
  status, assignee, and notes.
- **Convoys** are work orders: a named collection of beads representing one feature
  or task.
- **GUPP** (Gastown Universal Propulsion Principle): *if you find work assigned to
  you, you run it immediately* — no announcement, no waiting for approval.
- **Roles are fixed**: Mayor, Polecat, Witness, Refinery, Crew. Each has a strict
  contract. Polecats are ephemeral (one bead → done means gone); Crew are long-lived.

## Architecture

```{mermaid}
flowchart TD
    User([Overseer]) -->|task| M[Mayor]

    M -->|spawn polecat| PC1[Polecat 1]
    M -->|spawn polecat| PC2[Polecat 2]
    M -->|spawn polecat| PCn[Polecat N]
    M -->|spawn witness| W[Witness]
    M -->|spawn refinery| R[Refinery]
    M -->|spawn crew| C[Crew]

    PC1 -->|closes bead 1| DB[(bead store)]
    PC2 -->|closes bead 2| DB
    PCn -->|closes bead N| DB
    W   -->|reads beads| DB
    R   -->|notes results| DB
    M   -->|tracks convoy| DB
```

The **Mayor** is the only agent the user interacts with. It decomposes the task
into beads, spawns polecats in parallel to work them, optionally spawns Witness for
health checks, and finally spawns Refinery to integrate the result.

## Project structure

The package has three Python modules: `__main__.py` (CLI and Rich renderer),
`beads.py` (SQLite bead store and `BeadTool`), and `swarm.py` (spawn tools,
`Analyze` tool, `build_toolbox()`, and `run_gastown()`). The `roles/` subdirectory
contains `__init__.py` (Mayor agent and role metadata) plus one TOML file per worker
role: polecat, witness, refinery, and crew.

## 1. Beads — the data plane

Beads are stored in `workspace/.gas-town/beads.db` (SQLite).
`run_gastown()` opens a single `aiosqlite.Connection` and passes it as tool context —
the same lifetime pattern as an `aiohttp.ClientSession`.

```python
async with aiosqlite.connect(db_path) as db:
    await db.execute(BEAD_DDL)
    await db.commit()
    toolbox = build_toolbox(workspace, on_event, transport, role_models, db)
    roles = load_agents(ROLES_DIR, toolbox=toolbox)
    ...
```

`BeadTool` is a `ToolHandler[aiosqlite.Connection]` — the open connection is its
typed context. It exposes five actions:

```python
class BeadTool(ToolHandler[aiosqlite.Connection]):
    """Manage the shared bead store (convoy issue tracker)."""

    action: Literal["list", "create", "update", "close", "note"]
    id: int = 0
    title: str = ""
    status: BStatus | None = None   # None preserves the current status
    assignee: str = ""
    notes: str = ""

    async def __call__(self, context: aiosqlite.Connection) -> str:
        db = context
        if self.action == "list":
            return await bead_summary(db)
        if self.action == "create":
            cur = await db.execute("INSERT INTO beads (title) VALUES (?)", (self.title,))
            await db.commit()
            return f"Created bead [{cur.lastrowid}]: {self.title}"
        ...
```

All worker agents get the same bead tool instance. Because the open connection is
a direct Python object (not looked up by path), concurrent polecats all write
through the same connection without any per-task lookup.

The `.gas-town/` directory is reserved for internal orchestration data.
All role prompts explicitly instruct agents **not** to read or write anything inside it.

## 2. Roles

Worker roles (polecat, witness, refinery, crew) are TOML files. Each declares its
name, description, `max_iterations`, tool list, and system prompt:

```toml
# roles/polecat.toml
name = "polecat"
description = "Autonomous worker. Completes exactly one assigned bead, then closes it."
max_iterations = 25
tools = ["read_file", "write_file", "patch_file", "list_files",
         "shell", "run_python", "bead", "analyze"]

[system]
text = """
You are a Polecat — an autonomous worker in a Gas Town rig.
You have ONE job: complete your assigned bead and close it.
...
"""
```

`roles/__init__.py` declares only the **Mayor** in Python (because its tools include
dynamically-built spawn tools), and derives `ROLE_NAMES` from TOML filenames:

```python
from pathlib import Path
from axio.agent import Agent
from axio.transport import DummyCompletionTransport

ROLES_DIR = Path(__file__).parent
ROLE_NAMES = [p.stem for p in sorted(ROLES_DIR.glob("*.toml"))]

MAYOR = Agent(system="...", transport=DummyCompletionTransport())
```

Worker roles are loaded at runtime via `load_agents()` once the shared toolbox
(including the open DB connection) is available.

### Mayor — chief-of-staff

The Mayor is the agent you talk to. It translates a task into a convoy:

1. Creates a `[CONVOY]` bead as the work-order unit.
2. Reads or analyses the domain (may use `analyze` or `list_files`).
3. Decomposes the task into small child beads — one per component.
4. Calls `sling(bead_id=X)` for each bead — all in one response, fire-and-forget.
5. Calls `await_beads()` to block until all polecats finish.
6. Optionally spawns Witness for a mid-convoy health check (before or during the convoy).
7. After all polecats finish, spawns Refinery to integrate the work.
8. Closes the convoy bead and reports to the user.

The Mayor never writes code itself. It only spawns workers.

### Polecat — ephemeral worker

A polecat has exactly one job: complete its assigned bead, then close it.

```
GUPP: if you find work, YOU RUN IT.
No announcement. No waiting. No idle state.
```

Lifecycle:

1. Receives bead assignment via the worker pool (bead ID pulled from the channel).
2. Bead is already marked `in_progress` by `sling` before the polecat starts.
3. Does the work (file tools, shell, run_python, analyze).
4. Closes the bead: `bead(action='close', id=<id>)`.
5. Session ends — done means gone.

If a polecat discovers unrelated work, it creates a new bead and continues.
It never fixes things outside its assigned bead.

### Witness — per-rig monitor

The Witness is a read-only monitor. It checks the bead store and workspace quality
via `bead(action='list')` and `analyze`, and produces a health report. It **does
not write code** and has no write tools.

Useful for long convoys (5+ polecats) as a mid-convoy checkpoint.

### Refinery — merge queue processor

The Refinery is the integration engineer. It is **not** a passive reviewer —
it actively integrates polecat work:

- Verifies that all pieces fit together (imports resolve, interfaces match).
- Runs tests and the linter.
- **Fixes integration issues** — broken imports, mismatched signatures, conflicts.
- Escalates to Mayor only when a fix requires re-doing a full bead.

The Refinery has full write tools. "No work can be lost" is its core rule.

### Crew — long-lived agents

Crew members are the agents you interact with for sustained, back-and-forth work —
design sessions, complex investigations, exploratory coding. Unlike polecats, they
are not ephemeral and not managed by the Witness. Each crew member gets an
`AutoCompactStore` context so long sessions survive context limits.

## 3. Dispatch tools and TypedDict contexts

`swarm.py` defines dispatch tools as top-level classes with typed `TypedDict` contexts —
not closures. The Mayor uses two tools to manage polecats asynchronously:

**`Sling`** — fire-and-forget polecat dispatch. Marks the bead `in_progress` and puts
the bead ID into a shared async channel; returns immediately so the Mayor can sling
multiple polecats in one response:

```python
class SlingContext(TypedDict):
    db: aiosqlite.Connection
    queue: Channel[int]   # aiochannel.Channel


class Sling(ToolHandler[SlingContext]):
    """Sling a polecat at a bead — fire-and-forget, returns immediately.
    Sling multiple polecats in one response for parallel execution.
    After slinging all beads for a phase, call await_beads()."""
    bead_id: Annotated[int, Field(description="ID of the bead to work on")]
    topic: Annotated[str, Field(description="Short label, e.g. 'auth middleware'")]

    async def __call__(self, context: SlingContext) -> str:
        db = context["db"]
        row = await get_bead(db, self.bead_id)
        if row is None:
            return f"Bead {self.bead_id} not found"
        bead_id, title, *_ = row
        await mark_in_progress(db, bead_id, assignee=f"polecat:{self.topic}")
        await context["queue"].put(bead_id)
        return f"[{bead_id}] {title} → slung to polecat pool"
```

**`AwaitBeads`** — synchronisation point. Polls the bead store until all active beads
are closed (or a timeout expires). The Mayor calls this after slinging a phase:

```python
class AwaitBeads(ToolHandler[AwaitBeadsContext]):
    """Wait until all active (open/in_progress) beads are closed."""
    timeout: int = 3600

    async def __call__(self, context: AwaitBeadsContext) -> str:
        while True:
            if not await has_active_beads(context["db"]):
                return f"All beads complete.\n\n{await bead_summary(context['db'])}"
            await asyncio.sleep(5)
```

**Worker pool** — pre-spawned coroutines pull bead IDs from the channel and run
polecats concurrently. The pool is started before the Mayor runs and shut down
after `await_beads()` returns:

```python
polecat_queue: Channel[int] = Channel()
worker_tasks = [
    asyncio.create_task(polecat_worker(i + 1, roles["polecat"][1], polecat_queue, ...))
    for i in range(num_polecats)
]
# ... run Mayor ...
polecat_queue.close()           # workers exit their async-for loop cleanly
await asyncio.gather(*worker_tasks, return_exceptions=True)
```

Key points:

- `sling` is called multiple times in one Mayor response — all beads are queued
  immediately and workers pick them up in parallel.
- `topic` populates the status bar: **Polecat [auth middleware]**, **Polecat [data models]**,
  making parallel polecats identifiable at a glance.
- `mark_in_progress()` updates the SQLite row before the polecat starts, so Witness
  sees accurate status if it runs mid-convoy.
- `channel.close()` provides clean shutdown — no sentinel values, no `task.cancel()`.

## 4. The Analyze tool

Both Mayor and workers get an `analyze` tool that spawns ephemeral read-only analyst
subagents. The `AnalyzeContext` TypedDict carries the shared toolbox and callbacks:

```python
class AnalyzeContext(TypedDict):
    toolbox: dict[str, Tool[Any]]  # analyst uses list_files + read_file from this
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    guard_factory: GuardFactory | None
    counter: list[int]  # [0] holds mutable call count
```

`workspace: Path` is gone — the analyst's read tools come directly from the sandbox
toolbox, already bound to the running container. The analyst is told to look in
`/workspace` (the constant `WORKDIR`), which maps to the host workspace directory
via the Docker volume mount.

Analysts are fast (`max_iterations=10`, fast model) and safe to call in parallel
from multiple polecats simultaneously.

## 5. Toolbox and role loading

The toolbox starts from a **`DockerSandbox`** and is extended in-place with runtime
tools before `load_agents()` is called:

```python
async with DockerSandbox(
    image=args.image,
    volumes={"/workspace": str(workspace)},
    workdir="/workspace",
    name=sandbox_name,
    remove=False,
) as sandbox:
    toolbox = {t.name: t for t in sandbox.tools}
    # toolbox == {"read_file": Tool(...), "write_file": Tool(...), ...}

    toolbox["bead"]    = make_bead_tool(db, ...)
    toolbox["analyze"] = make_analyze_tool(toolbox=toolbox, ...)

    roles = load_agents(ROLES_DIR, toolbox=toolbox)
    # roles == {"polecat": ("Autonomous worker...", Agent(...)), "witness": ..., ...}
```

All file and shell tools in the toolbox are bound to the running container — agents
read and write files inside `/workspace`, which is mounted from the host workspace
directory. The Mayor gets its own separate tool set (sling + await_beads + bead +
analyze + read tools) and does not use the worker toolbox directly.

## 6. GUPP in practice

GUPP is enforced through the system prompt. Every role includes a variant of:

```
The Propulsion Principle (GUPP)
--------------------------------
If you find assigned work, YOU RUN IT. No announcement, no confirmation, no waiting.
The assignment IS the authorisation. Gas Town is a steam engine — you are a piston.

Failure mode to avoid:
  Agent receives assignment → announces itself → waits for "ok go"
  Human is AFK → work sits idle → the whole convoy stalls.
```

The polecat prompt reinforces this with:

```
There is no step 5. There is no "wait for approval". There is no idle state.

If your assigned bead has nothing to implement:
- Note the reason: bead(action='note', id=<id>, notes='no-changes: <reason>')
- Close the bead:  bead(action='close', id=<id>)
Never leave without closing your bead.
```

This prevents the most common failure mode: a polecat that finishes work, says
"done!", and then sits idle waiting for acknowledgement — blocking the convoy.

## 7. Running it

```bash
cd examples/gas_town
uv sync
export NEBIUS_API_KEY=...

axio-gastown --workspace /tmp/my_project \
    "Build a Python rate limiter with token-bucket and sliding-window"
```

Docker is required. A fresh container is created on each run and removed on exit.

**Docker options:**

| Flag | Default | Description |
|---|---|---|
| `--image` | `python:3.12-slim` | Container image |
| `--memory` | `512m` | Memory limit (e.g. `1g`) |
| `--cpus` | `2.0` | CPU limit |
| `--network` | off | Enable network access inside the container |
| `--polecats` | `5` | Number of pre-spawned polecat workers |

```bash
axio-gastown --workspace /tmp/my_project \
    --image python:3.12-slim --memory 2g --polecats 8 \
    "Write an async task queue with priority levels"
```

Default model assignments in `__main__.py`:

```python
role_models: dict[str, ModelSpec] = {
    "default":  transport.models["MiniMaxAI/MiniMax-M2.5"],
    "mayor":    transport.models["Qwen/Qwen3-235B-A22B-Instruct-2507"],
    "polecat":  transport.models["Qwen/Qwen3.5-397B-A17B"],
    "witness":  transport.models["openai/gpt-oss-120b"],
    "refinery": transport.models["openai/gpt-oss-120b"],
    # analyst runs many instances in parallel — use a fast model
    "analyst":  transport.models["deepseek-ai/DeepSeek-V3.2"],
}
```

After a run the workspace directory (host-side) contains all produced artifacts
alongside `AGENTS.md` (the living project memory written by agents). The `.gas-town/`
subdirectory holds internal orchestration data including the bead SQLite database —
agents are instructed never to touch it.

Inspect the bead history with any SQLite tool:

```bash
sqlite3 /tmp/my_project/.gas-town/beads.db \
    "SELECT id, title, status, assignee FROM beads"
```

## Gas Town vs Agent Swarm

Both examples implement multi-agent coordination on top of Axio.
The key differences:

| | Agent Swarm | Gas Town |
|---|---|---|
| **Task tracking** | SQLite todo list (orchestrator) | SQLite bead store (all agents) |
| **Delegation** | `delegate(role, task)` — any role | `spawn_polecat(bead_id)` — per-bead |
| **Worker lifecycle** | Specialist returns result to orchestrator | Polecat closes its bead and disappears |
| **Oversight** | None | Witness monitors; Refinery integrates |
| **Crew** | No equivalent | Long-lived human-facing agents |
| **Work granularity** | One big task per specialist | One small bead per polecat |
| **Propulsion** | Orchestrator drives everything | GUPP: agent drives itself |
| **Workspace state** | Implicit (files only) | Explicit (bead store + files) |

Use **agent_swarm** when you want a straightforward team of specialists with minimal
overhead. Use **gas_town** when you want explicit work tracking, parallel polecat
swarms, integration review, and a workflow model that survives restarts.

```{seealso}
- {doc}`../gas-town-rules` — full Gas Town methodology reference
- {doc}`agent-swarm` — the simpler team-of-specialists pattern
- {doc}`writing-tools` — how to build custom ToolHandlers
```

```{toctree}
:hidden:

../gas-town-rules
```
