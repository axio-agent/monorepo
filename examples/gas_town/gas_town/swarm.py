"""Gas Town swarm: Mayor + Polecats + Witness + Refinery powered by Axio.

Architecture:
  - Mayor decomposes work into Beads and spawns workers.
  - Each Polecat is ephemeral: one bead, done means gone.
  - Witness monitors Polecat health via the shared Bead store.
  - Refinery reviews completed work and runs quality gates.
  - Crew are long-lived human-facing agents, not managed by Witness.

Worker roles (polecat, witness, refinery, crew) are loaded from TOML files
inside ``run_gastown()`` once the shared toolbox is available.  Each role
declares its required tools in its TOML ``tools`` list; the toolbox injects
the actual ``Tool`` instances.

Transport setup lives in __main__.py — swarm.py only uses what it is given.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

import aiosqlite
from axio.agent import Agent
from axio.agent_loader import load_agents
from axio.compaction import AutoCompactStore
from axio.context import MemoryContextStore
from axio.events import StreamEvent, TextDelta
from axio.models import ModelSpec
from axio.permission import PermissionGuard
from axio.tool import Tool, ToolHandler
from axio.transport import CompletionTransport, DummyCompletionTransport
from axio_tools_local.list_files import ListFiles  # type: ignore[import-untyped]
from axio_tools_local.patch_file import PatchFile  # type: ignore[import-untyped]
from axio_tools_local.read_file import ReadFile  # type: ignore[import-untyped]
from axio_tools_local.run_python import RunPython  # type: ignore[import-untyped]
from axio_tools_local.shell import Shell  # type: ignore[import-untyped]
from axio_tools_local.write_file import WriteFile  # type: ignore[import-untyped]
from pydantic import Field

from .beads import DDL as BEAD_DDL
from .beads import bead_summary, get_bead, make_bead_tool, mark_in_progress
from .roles import MAYOR

OnEventCallback = Callable[[str, StreamEvent], Awaitable[None]]
GuardFactory = Callable[[str, str], PermissionGuard]

ROLES_DIR = Path(__file__).parent / "roles"

# ---------------------------------------------------------------------------
# Read-only analyst prototype (no tools — injected via copy())
# ---------------------------------------------------------------------------

ANALYST = Agent(
    system="""\
You are a read-only analyst. Your only job is to read files in the workspace and
produce a concise, well-structured report answering the question you are given.
You must not create, modify, or delete any files.
Return your findings as plain text — the caller will use them directly.""",
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
    """Return a copy of *base* with the model for *role* (falls back to 'default')."""
    model = role_models.get(role) or role_models["default"]
    new_transport = copy.copy(base)
    new_transport.model = model  # type: ignore[attr-defined]
    return new_transport


# ---------------------------------------------------------------------------
# Analyze tool — top-level handler, context carries all runtime deps
# ---------------------------------------------------------------------------


class AnalyzeContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    guard_factory: GuardFactory | None
    counter: list[int]  # [0] holds the mutable call count


class Analyze(ToolHandler[AnalyzeContext]):
    """Spawn a read-only analyst subagent to investigate a question and return a report.
    The analyst can only read files — it cannot modify anything.
    Safe to call many times in parallel; use one per file or question."""

    task: Annotated[str, Field(description="Question or analysis task for the analyst")]

    async def __call__(self, context: AnalyzeContext) -> str:
        context["counter"][0] += 1
        n = context["counter"][0]
        agent_id = f"analyst#{n}:{self.task[:40]}"

        analyst_transport = transport_for("analyst", context["transport"], context["role_models"])
        analyst = ANALYST.copy(
            transport=analyst_transport,
            tools=[
                Tool(name="list_files", description=ListFiles.__doc__ or "", handler=ListFiles),
                Tool(name="read_file", description=ReadFile.__doc__ or "", handler=ReadFile),
            ],
            max_iterations=10,
        )
        stream = analyst.run_stream(f"Workspace: {context['workspace']}\n\n{self.task}", MemoryContextStore())
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)


def make_analyze_tool(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    caller_role: str = "specialist",
    guard_factory: GuardFactory | None = None,
) -> Tool:
    """Create an analyze tool.  *caller_role* is used only for guard_factory calls."""
    guards: tuple[PermissionGuard, ...] = (guard_factory(caller_role, "analyze"),) if guard_factory else ()
    return Tool(
        name="analyze",
        description=Analyze.__doc__ or "",
        handler=Analyze,
        context=AnalyzeContext(
            workspace=workspace,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            guard_factory=guard_factory,
            counter=[0],
        ),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Toolbox — shared tools for worker agents, built once per run
# ---------------------------------------------------------------------------


def build_toolbox(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    db: aiosqlite.Connection,
    guard_factory: GuardFactory | None = None,
) -> dict[str, Tool[Any]]:
    """Build the shared toolbox injected into worker agents via load_agents().

    The toolbox contains all tools listed in the worker TOML files.  The Mayor
    is given its own separate tool set (spawn tools + bead + read tools).
    """

    def guards(role: str, name: str) -> tuple[PermissionGuard, ...]:
        return (guard_factory(role, name),) if guard_factory else ()

    return {
        "read_file": Tool(
            name="read_file",
            description=ReadFile.__doc__ or "",
            handler=ReadFile,
            guards=guards("worker", "read_file"),
        ),
        "write_file": Tool(
            name="write_file",
            description=WriteFile.__doc__ or "",
            handler=WriteFile,
            guards=guards("worker", "write_file"),
        ),
        "patch_file": Tool(
            name="patch_file",
            description=PatchFile.__doc__ or "",
            handler=PatchFile,
            guards=guards("worker", "patch_file"),
        ),
        "list_files": Tool(
            name="list_files",
            description=ListFiles.__doc__ or "",
            handler=ListFiles,
            guards=guards("worker", "list_files"),
        ),
        "shell": Tool(
            name="shell",
            description=Shell.__doc__ or "",
            handler=Shell,
            guards=guards("worker", "shell"),
        ),
        "run_python": Tool(
            name="run_python",
            description=RunPython.__doc__ or "",
            handler=RunPython,
            guards=guards("worker", "run_python"),
        ),
        "bead": make_bead_tool(db, guards=guards("worker", "bead")),
        "analyze": make_analyze_tool(
            workspace, on_event, transport, role_models, caller_role="specialist", guard_factory=guard_factory
        ),
    }


# ---------------------------------------------------------------------------
# Spawn tools — top-level handlers, context carries all runtime deps
# ---------------------------------------------------------------------------


class SpawnPolecatContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    proto: Agent
    db: aiosqlite.Connection
    counters: dict[int, int]


class SpawnPolecat(ToolHandler[SpawnPolecatContext]):
    """Spawn a polecat worker to complete a specific bead.
    The polecat has ONE job: complete the assigned bead and close it.
    Multiple polecats can run in parallel — spawn them all in one response.
    Returns the polecat's final summary when it finishes."""

    bead_id: Annotated[int, Field(description="ID of the bead to work on")]
    topic: Annotated[str, Field(description="Short label for the status bar, e.g. 'auth middleware'")]

    async def __call__(self, context: SpawnPolecatContext) -> str:
        db = context["db"]
        row = await get_bead(db, self.bead_id)
        if row is None:
            return f"Bead {self.bead_id} not found"
        bead_id, title, *_ = row
        await mark_in_progress(db, bead_id)

        context["counters"][self.bead_id] = context["counters"].get(self.bead_id, 0) + 1
        n = context["counters"][self.bead_id]
        agent_id = f"polecat#{n}:{self.topic}"

        polecat_transport = transport_for("polecat", context["transport"], context["role_models"])
        polecat = context["proto"].copy(transport=polecat_transport)
        polecat_ctx = AutoCompactStore(MemoryContextStore(), polecat_transport, keep_recent=6)
        task_msg = (
            f"Workspace: {context['workspace']}\n\n"
            f"Your assigned bead: [{bead_id}] {title}\n\n"
            f"Work this bead to completion, then close it with "
            f"`bead(action='close', id={bead_id})`."
        )
        stream = polecat.run_stream(task_msg, polecat_ctx)
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)


class SpawnWitnessContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    proto: Agent
    db: aiosqlite.Connection
    counter: list[int]


class SpawnWitness(ToolHandler[SpawnWitnessContext]):
    """Spawn the Witness to check polecat health via the bead store.
    The Witness reviews bead status and workspace quality, then returns a report.
    Useful for long convoys (5+ polecats) as a mid-convoy health check."""

    notes: Annotated[str, Field(default="", description="Optional context to give the Witness")]

    async def __call__(self, context: SpawnWitnessContext) -> str:
        context["counter"][0] += 1
        n = context["counter"][0]
        agent_id = f"witness#{n}" if n > 1 else "witness"

        bead_sum = await bead_summary(context["db"])
        witness_transport = transport_for("witness", context["transport"], context["role_models"])
        witness = context["proto"].copy(transport=witness_transport)
        task_msg = (
            f"Workspace: {context['workspace']}\n\n"
            f"Current bead status:\n{bead_sum}\n\n"
            f"{self.notes or 'Check polecat health and produce a status report.'}"
        )
        stream = witness.run_stream(task_msg, MemoryContextStore())
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)


class SpawnRefineryContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    proto: Agent
    db: aiosqlite.Connection
    counter: list[int]


class SpawnRefinery(ToolHandler[SpawnRefineryContext]):
    """Spawn the Refinery to integrate and fix completed polecat work.
    The Refinery is the merge queue processor: it resolves conflicts between polecats,
    runs quality gates, fixes integration issues, and ensures no work is lost.
    Call this after all polecats have finished (all beads closed)."""

    focus: Annotated[str, Field(default="", description="Optional specific areas to focus on during review")]

    async def __call__(self, context: SpawnRefineryContext) -> str:
        context["counter"][0] += 1
        n = context["counter"][0]
        agent_id = f"refinery#{n}" if n > 1 else "refinery"

        bead_sum = await bead_summary(context["db"])
        refinery_transport = transport_for("refinery", context["transport"], context["role_models"])
        refinery = context["proto"].copy(transport=refinery_transport)
        task_msg = (
            f"Workspace: {context['workspace']}\n\n"
            f"Completed convoy — bead summary:\n{bead_sum}\n\n"
            f"{'Focus: ' + self.focus if self.focus else 'Review all completed work.'}"
        )
        stream = refinery.run_stream(task_msg, MemoryContextStore())
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)


class SpawnCrewContext(TypedDict):
    workspace: Path
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    proto: Agent
    db: aiosqlite.Connection
    crew_names: dict[str, int]


class SpawnCrew(ToolHandler[SpawnCrewContext]):
    """Spawn a Crew member — a long-lived coding agent for interactive or complex work.
    Unlike polecats, Crew are not bound to a single bead and are not managed by Witness.
    Use for design work, explorations, or tasks requiring sustained back-and-forth.
    Returns the Crew member's output when the session ends."""

    name: Annotated[str, Field(description="Name for this Crew member, e.g. 'alice' or 'dom'")]
    task: Annotated[str, Field(description="Task or context for the Crew member")]

    async def __call__(self, context: SpawnCrewContext) -> str:
        context["crew_names"][self.name] = context["crew_names"].get(self.name, 0) + 1
        n = context["crew_names"][self.name]
        agent_id = f"crew#{n}:{self.name}" if n > 1 else f"crew:{self.name}"

        crew_transport = transport_for("crew", context["transport"], context["role_models"])
        member = context["proto"].copy(transport=crew_transport)
        crew_ctx = AutoCompactStore(MemoryContextStore(), crew_transport, keep_recent=10)
        task_msg = f"Workspace: {context['workspace']}\n\nYou are {self.name}, a Crew member.\n\n{self.task}"
        stream = member.run_stream(task_msg, crew_ctx)
        parts: list[str] = []
        async for event in stream:
            await context["on_event"](agent_id, event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Spawn tool factories
# ---------------------------------------------------------------------------


def make_spawn_polecat_tool(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    proto: Agent,
    db: aiosqlite.Connection,
    guard_factory: GuardFactory | None = None,
) -> Tool[SpawnPolecatContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("mayor", "spawn_polecat"),) if guard_factory else ()
    return Tool(
        name="spawn_polecat",
        description=SpawnPolecat.__doc__ or "",
        handler=SpawnPolecat,
        context=SpawnPolecatContext(
            workspace=workspace,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            proto=proto,
            db=db,
            counters={},
        ),
        guards=guards,
    )


def make_spawn_witness_tool(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    proto: Agent,
    db: aiosqlite.Connection,
    guard_factory: GuardFactory | None = None,
) -> Tool[SpawnWitnessContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("mayor", "spawn_witness"),) if guard_factory else ()
    return Tool(
        name="spawn_witness",
        description=SpawnWitness.__doc__ or "",
        handler=SpawnWitness,
        context=SpawnWitnessContext(
            workspace=workspace,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            proto=proto,
            db=db,
            counter=[0],
        ),
        guards=guards,
    )


def make_spawn_refinery_tool(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    proto: Agent,
    db: aiosqlite.Connection,
    guard_factory: GuardFactory | None = None,
) -> Tool[SpawnRefineryContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("mayor", "spawn_refinery"),) if guard_factory else ()
    return Tool(
        name="spawn_refinery",
        description=SpawnRefinery.__doc__ or "",
        handler=SpawnRefinery,
        context=SpawnRefineryContext(
            workspace=workspace,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            proto=proto,
            db=db,
            counter=[0],
        ),
        guards=guards,
    )


def make_spawn_crew_tool(
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    proto: Agent,
    db: aiosqlite.Connection,
    guard_factory: GuardFactory | None = None,
) -> Tool[SpawnCrewContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("mayor", "spawn_crew"),) if guard_factory else ()
    return Tool(
        name="spawn_crew",
        description=SpawnCrew.__doc__ or "",
        handler=SpawnCrew,
        context=SpawnCrewContext(
            workspace=workspace,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            proto=proto,
            db=db,
            crew_names={},
        ),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_gastown(
    task: str,
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    guard_factory: GuardFactory | None = None,
    prompt_fn: Callable[[], str] | None = None,  # noqa: F841  (reserved for future ask_user)
) -> str:
    """Run a Gas Town convoy on *task*.

    Args:
        task:          What to build or accomplish.
        workspace:     Directory where agents read and write files.
        on_event:      Callback for every StreamEvent emitted by any agent.
        transport:     Base transport — shared session, per-role model applied via copy().
        role_models:   Maps role names (and "default") to ModelSpec.
                       Recognised roles: "mayor", "polecat", "witness", "refinery", "crew", "analyst".
                       Roles not listed fall back to role_models["default"].
        guard_factory: Optional ``(role, tool_name) -> PermissionGuard`` factory.
        prompt_fn:     Reserved for future ask_user integration (not used yet).
    """
    assert "default" in role_models, "role_models must contain a 'default' key"

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)

    def og(name: str) -> tuple[PermissionGuard, ...]:
        return (guard_factory("mayor", name),) if guard_factory else ()

    db_path = workspace / ".gas-town" / "beads.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(BEAD_DDL)
        await db.commit()

        # Build the shared toolbox and load worker roles with tools injected.
        toolbox = build_toolbox(workspace, on_event, transport, role_models, db, guard_factory)
        roles = load_agents(ROLES_DIR, toolbox=toolbox)

        # Build Mayor's spawn tools — each wraps a role prototype.
        spawn_polecat = make_spawn_polecat_tool(
            workspace,
            on_event,
            transport,
            role_models,
            proto=roles["polecat"][1],
            db=db,
            guard_factory=guard_factory,
        )
        spawn_witness = make_spawn_witness_tool(
            workspace,
            on_event,
            transport,
            role_models,
            proto=roles["witness"][1],
            db=db,
            guard_factory=guard_factory,
        )
        spawn_refinery = make_spawn_refinery_tool(
            workspace,
            on_event,
            transport,
            role_models,
            proto=roles["refinery"][1],
            db=db,
            guard_factory=guard_factory,
        )
        spawn_crew = make_spawn_crew_tool(
            workspace,
            on_event,
            transport,
            role_models,
            proto=roles["crew"][1],
            db=db,
            guard_factory=guard_factory,
        )

        mayor_analyze = make_analyze_tool(
            workspace, on_event, transport, role_models, caller_role="mayor", guard_factory=guard_factory
        )
        mayor_bead = make_bead_tool(db, guards=og("bead"))
        mayor_read_tools = [
            Tool(name="list_files", description=ListFiles.__doc__ or "", handler=ListFiles, guards=og("list_files")),
            Tool(name="read_file", description=ReadFile.__doc__ or "", handler=ReadFile, guards=og("read_file")),
        ]

        mayor_transport = transport_for("mayor", transport, role_models)
        mayor = MAYOR.copy(
            transport=mayor_transport,
            max_iterations=200,
            tools=[
                mayor_bead,
                spawn_polecat,
                spawn_witness,
                spawn_refinery,
                spawn_crew,
                mayor_analyze,
                *mayor_read_tools,
            ],
        )
        mayor_ctx = AutoCompactStore(MemoryContextStore(), mayor_transport, keep_recent=10)

        stream = mayor.run_stream(task, mayor_ctx)
        parts: list[str] = []
        async for event in stream:
            await on_event("mayor", event)
            if isinstance(event, TextDelta):
                parts.append(event.delta)
    return "".join(parts)
