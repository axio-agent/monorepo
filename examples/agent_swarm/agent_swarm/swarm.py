"""Agent swarm: orchestrator + specialist team powered by Axio.

Specialist roles are loaded from TOML files at runtime once the shared toolbox
is ready - each role's ``tools`` list is resolved against the toolbox.  The
Orchestrator is the only agent declared in Python (in roles/__init__.py).

Transport setup lives in main.py - swarm.py only uses what it is given.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

import aiosqlite
from axio import (
    CONTEXT,
    Agent,
    CompletionTransport,
    Field,
    MemoryContextStore,
    PermissionGuard,
    StreamEvent,
    TextDelta,
    Tool,
)
from axio.agent_loader import load_agents
from axio.compaction import AutoCompactStore
from axio.models import ModelSpec
from axio.transport import DummyCompletionTransport

from .ask_user import make_ask_user_tool
from .notes import make_notes_tool
from .roles import ROLE_NAMES, ROLES_DIR, make_orchestrator
from .todo import DDL as TODO_DDL
from .todo import make_todo_tool

OnEventCallback = Callable[[str, StreamEvent], Awaitable[None]]
GuardFactory = Callable[[str], PermissionGuard]

WORKDIR = "/workspace"

SANDBOX_CONTEXT = """\
Sandbox environment
-------------------
You are running inside an isolated Docker container. The only path shared with
the host is /workspace - all project reads and writes happen there.
Paths inside the container differ from host paths; do not assume they match.
You have full root access inside this sandbox: install any packages, compilers,
or CLI tools you need via shell (apt, pip, npm, cargo, …). Modify system files
freely. This container is yours - treat it that way.

Tool discipline
---------------
Only call tools that are declared in your tool definitions.
Never guess or invent a tool name. If a tool is not in your list, it does not exist."""

# ---------------------------------------------------------------------------
# Read-only analyst prototype
# ---------------------------------------------------------------------------

ANALYST = Agent(
    system="""\
You are a read-only analyst. Your job is to read files in the workspace and produce
a thorough, precise report answering the question you are given. Other specialists
will make decisions based solely on your output - omitting details is worse than
being verbose.

Precision rules:
- Every claim must cite its source: filename and line number (e.g. `auth.py:42`).
- Quote the exact lines when they matter; do not paraphrase code or config.
- If something is absent or you could not find it, say so explicitly: which files you
  checked and what you searched for. Never invent or guess - "not found" is a valid
  and valuable answer.
- Cover edge cases, surprises, and anything that contradicts the obvious interpretation.

You must not create, modify, or delete any files. Never use write_file or patch_file.""",
    transport=DummyCompletionTransport(),
)


# ---------------------------------------------------------------------------
# Transport helper
# ---------------------------------------------------------------------------


def transport_for(
    role: str,
    base: CompletionTransport,
    role_models: dict[str, ModelSpec],
) -> CompletionTransport:
    """Return a copy of *base* with the model for *role* (falls back to "default")."""
    model = role_models.get(role) or role_models["default"]
    new_transport = copy.copy(base)
    new_transport.model = model  # type: ignore[attr-defined]
    return new_transport


# ---------------------------------------------------------------------------
# Analyze tool
# ---------------------------------------------------------------------------


class AnalyzeContext(TypedDict):
    toolbox: dict[str, Tool[Any]]
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    guard_factory: GuardFactory | None
    counter: list[int]


async def analyze(
    task: Annotated[str, Field(description="Question or analysis task for the analyst")],
) -> str:
    """Spawn a read-only analyst subagent to investigate a question and return a report.
    The analyst can only read files - it cannot modify anything.
    Safe to call many times in parallel; use one per file or question."""
    context: AnalyzeContext = CONTEXT.get()
    context["counter"][0] += 1
    n = context["counter"][0]
    agent_id = f"analyst#{n}:{task[:40]}"

    analyst_transport = transport_for("analyst", context["transport"], context["role_models"])
    tb = context["toolbox"]
    read_tools = [tb[k] for k in ("list_files", "read_file") if k in tb]
    analyst_system = f"Workspace root: {WORKDIR}\n\n---\n\n{ANALYST.system}"
    analyst = ANALYST.copy(transport=analyst_transport, tools=read_tools, max_iterations=10, system=analyst_system)
    stream = analyst.run_stream(task, MemoryContextStore())
    parts: list[str] = []
    async for event in stream:
        await context["on_event"](agent_id, event)
        if isinstance(event, TextDelta):
            parts.append(event.delta)
    return "".join(parts)


def make_analyze_tool(
    toolbox: dict[str, Tool[Any]],
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    caller_role: str = "specialist",
    guard_factory: GuardFactory | None = None,
) -> Tool[AnalyzeContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory(caller_role),) if guard_factory else ()
    return Tool(
        name="analyze",
        handler=analyze,
        context=AnalyzeContext(
            toolbox=toolbox,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            guard_factory=guard_factory,
            counter=[0],
        ),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Delegate tool
# ---------------------------------------------------------------------------


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
        Field(description=f"Which specialist to delegate to. One of: {', '.join(ROLE_NAMES)}"),
    ],
    topic: Annotated[
        str,
        Field(description="Short label for this task, e.g. 'auth middleware'. Shown in the status bar."),
    ],
    task: Annotated[str, Field(description="Specific instructions for the specialist")],
) -> str:
    """Delegate a task to a specialist team member.
    The specialist reads the workspace, does the work, and writes output back.
    Returns the specialist's final response text."""
    context: DelegateContext = CONTEXT.get()
    context["counters"][role] = context["counters"].get(role, 0) + 1
    n = context["counters"][role]
    base_id = role if n == 1 else f"{role}#{n}"
    agent_id = f"{base_id}:{topic}" if topic else base_id

    description, proto = context["roles"][role]
    role_transport = transport_for(role, context["transport"], context["role_models"])
    specialist = proto.copy(transport=role_transport)
    specialist_ctx = AutoCompactStore(MemoryContextStore(), role_transport, keep_recent=6)
    stream = specialist.run_stream(
        f"Workspace: {WORKDIR}\n\n{task}",
        specialist_ctx,
    )
    parts: list[str] = []
    async for event in stream:
        await context["on_event"](agent_id, event)
        if isinstance(event, TextDelta):
            parts.append(event.delta)
    return "".join(parts)


def make_delegate_tool(
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    roles: dict[str, tuple[str, Agent]],
    guard_factory: GuardFactory | None = None,
) -> Tool[DelegateContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("orchestrator"),) if guard_factory else ()
    return Tool(
        name="delegate",
        handler=delegate,
        context=DelegateContext(
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            roles=roles,
            guard_factory=guard_factory,
            counters={},
        ),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_swarm(
    task: str,
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    toolbox: dict[str, Tool[Any]],
    guard_factory: GuardFactory | None = None,
    prompt_fn: Callable[[], str] | None = None,
) -> str:
    """Run the agent swarm on *task*.

    Args:
        task:          What to build.
        workspace:     Directory where agents read and write files (host path).
        on_event:      Callback for every StreamEvent emitted by any agent.
        transport:     Base transport - shared session, per-role model applied via copy().
        role_models:   Maps role names (and "default") to ModelSpec.
                       Every role not listed falls back to role_models["default"].
        toolbox:       Mapping of tool name → Tool, typically from a DockerSandbox.
        guard_factory: Optional ``(role, tool_name) -> PermissionGuard`` factory.
        prompt_fn:     Optional callable ``() -> str`` used by ``ask_user`` to read input.
    """
    assert "default" in role_models, "role_models must contain a 'default' key"

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    def og(name: str) -> tuple[PermissionGuard, ...]:
        return (guard_factory("orchestrator"),) if guard_factory else ()

    todo_path = workspace / ".axio-swarm" / "todos.db"
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    if todo_path.is_symlink():
        todo_path.unlink()
    async with aiosqlite.connect(todo_path) as todo_db:
        await todo_db.execute(TODO_DDL)
        await todo_db.commit()

        toolbox = dict(toolbox)
        toolbox["analyze"] = make_analyze_tool(
            toolbox,
            on_event,
            transport,
            role_models,
            caller_role="specialist",
            guard_factory=guard_factory,
        )
        toolbox["notes"] = make_notes_tool(workspace)

        raw_roles = load_agents(ROLES_DIR, toolbox=toolbox)
        roles = {
            name: (desc, agent.copy(system=f"{SANDBOX_CONTEXT}\n\n---\n\n{agent.system}"))
            for name, (desc, agent) in raw_roles.items()
        }

        orch_analyze = make_analyze_tool(
            toolbox,
            on_event,
            transport,
            role_models,
            caller_role="orchestrator",
            guard_factory=guard_factory,
        )
        delegate_tool = make_delegate_tool(
            on_event,
            transport,
            role_models,
            roles=roles,
            guard_factory=guard_factory,
        )
        todo_tool = make_todo_tool(todo_db, guards=og("todo"))
        ask_user_tool = make_ask_user_tool(prompt_fn=prompt_fn, guards=og("ask_user"))
        orch_notes_tool = make_notes_tool(workspace, guards=og("notes"))

        orch_transport = transport_for("orchestrator", transport, role_models)
        roster = "\n".join(f"  {name:20s} - {desc}" for name, (desc, _) in roles.items())
        orchestrator = make_orchestrator(roster, sandbox_context=SANDBOX_CONTEXT).copy(
            transport=orch_transport,
            tools=[delegate_tool, ask_user_tool, todo_tool, orch_analyze, orch_notes_tool],
        )
        orch_ctx = AutoCompactStore(MemoryContextStore(), orch_transport, keep_recent=10)

        agents_md = workspace / "AGENTS.md"
        full_task = task
        if agents_md.is_file() and not agents_md.is_symlink():
            content = agents_md.read_text().strip()
            if content:
                full_task = f"## Project context (AGENTS.md)\n\n{content}\n\n---\n\n## Task\n\n{task}"

        stream = orchestrator.run_stream(full_task, orch_ctx)
        parts: list[str] = []
        async for event in stream:
            await on_event("orchestrator", event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
    return "".join(parts)
