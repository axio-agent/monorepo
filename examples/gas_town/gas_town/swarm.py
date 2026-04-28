"""Gas Town swarm: Mayor + Polecats + Witness + Refinery powered by Axio.

Architecture:
  - Mayor decomposes work into Beads, slings polecats at them (fire-and-forget),
    then calls await_beads() to block until all polecats finish.
  - Polecats are a pre-spawned worker pool: N coroutines pulling bead IDs from
    a Channel, each working one bead at a time and looping for more.
  - Witness runs as a background patrol: wakes periodically, checks polecat health
    via the bead store, reports status with exponential backoff when idle.
  - Refinery runs as a background patrol: wakes when closed beads appear, integrates
    completed work, and marks each bead reviewed.
  - Crew are long-lived human-facing agents, not managed by Witness.

``run_gastown()`` receives a pre-built *toolbox* from the caller (see ``__main__.py``
which creates a ``DockerSandbox`` and passes ``{t.name: t for t in sandbox.tools}``).
The function adds runtime tools (``bead``, ``analyze``) to the toolbox, then calls
``load_agents()`` to wire tools into each role from its TOML declaration.

Transport setup lives in __main__.py - swarm.py only uses what it is given.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

import aiosqlite
from aiochannel import Channel
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

from .beads import DDL as BEAD_DDL
from .beads import (
    bead_summary,
    get_bead,
    get_unreviewed_closed_beads,
    has_active_beads,
    make_bead_tool,
    mark_in_progress,
)
from .roles import MAYOR

OnEventCallback = Callable[[str, StreamEvent], Awaitable[None]]
GuardFactory = Callable[[str], PermissionGuard]

ROLES_DIR = Path(__file__).parent / "roles"

WORKDIR = "/workspace"

SANDBOX_CONTEXT = """\
Sandbox environment
-------------------
You are running inside an isolated Docker container. The only path shared with
the host is /workspace - all project reads and writes happen there.
Paths inside the container differ from host paths; do not assume they match.
You have full root access inside this sandbox: install any packages, compilers,
or CLI tools you need via shell (apt, pip, npm, cargo, …). Modify system files
freely. This container is yours - treat it that way."""

# ---------------------------------------------------------------------------
# Read-only analyst prototype
# ---------------------------------------------------------------------------

ANALYST = Agent(
    system="""\
You are a read-only analyst. Your only job is to read files in the workspace and
produce a concise, well-structured report answering the question you are given.
You must not create, modify, or delete any files.
Return your findings as plain text - the caller will use them directly.""",
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
# Analyze tool
# ---------------------------------------------------------------------------


class AnalyzeContext(TypedDict):
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    guard_factory: GuardFactory | None
    counter: list[int]
    toolbox: dict[str, Tool[Any]]


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
    analyst_system = f"{SANDBOX_CONTEXT}\n\n---\n\n{ANALYST.system}"
    analyst = ANALYST.copy(transport=analyst_transport, tools=read_tools, max_iterations=10, system=analyst_system)
    stream = analyst.run_stream(f"Workspace: {WORKDIR}\n\n{task}", MemoryContextStore())
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
) -> Tool:
    """Create an analyze tool backed by *toolbox* read tools."""
    guards: tuple[PermissionGuard, ...] = (guard_factory(caller_role),) if guard_factory else ()
    return Tool(
        name="analyze",
        handler=analyze,
        context=AnalyzeContext(
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            guard_factory=guard_factory,
            counter=[0],
            toolbox=toolbox,
        ),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Sling - fire-and-forget polecat dispatch (Mayor's tool)
# ---------------------------------------------------------------------------


class SlingContext(TypedDict):
    db: aiosqlite.Connection
    queue: Channel[int]


async def sling(
    bead_id: Annotated[int, Field(description="ID of the bead to work on")],
    topic: Annotated[str, Field(description="Short label for the status bar, e.g. 'auth middleware'")],
) -> str:
    """Sling a polecat at a bead - fire-and-forget, returns immediately.
    The polecat picks it up from the pool and works it in the background.
    Sling multiple polecats in one response for parallel execution.
    After slinging all beads for a phase, call await_beads() to wait for completion."""
    context: SlingContext = CONTEXT.get()
    db = context["db"]
    row = await get_bead(db, bead_id)
    if row is None:
        return f"Bead {bead_id} not found"
    bid, title, *_ = row
    await mark_in_progress(db, bid, assignee=f"polecat:{topic}")
    await context["queue"].put(bid)
    return f"[{bid}] {title} → slung to polecat pool"


def make_sling_tool(
    db: aiosqlite.Connection,
    queue: Channel[int],
    guard_factory: GuardFactory | None = None,
) -> Tool[SlingContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("mayor"),) if guard_factory else ()
    return Tool(
        name="sling",
        handler=sling,
        context=SlingContext(db=db, queue=queue),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# AwaitBeads - block Mayor until all active beads are done
# ---------------------------------------------------------------------------


class AwaitBeadsContext(TypedDict):
    db: aiosqlite.Connection


async def await_beads(
    timeout: Annotated[int, Field(default=3600, description="Max seconds to wait (default 3600)")] = 3600,
) -> str:
    """Wait until all active (open/in_progress) beads are closed or timeout expires.
    Call this after sling()ing all polecats to wait for the convoy phase to complete."""
    context: AwaitBeadsContext = CONTEXT.get()
    db = context["db"]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if not await has_active_beads(db):
            summary = await bead_summary(db)
            return f"All beads complete.\n\n{summary}"
        if loop.time() >= deadline:
            summary = await bead_summary(db)
            return f"Timeout ({timeout}s). Some beads still active.\n\n{summary}"
        await asyncio.sleep(5.0)


def make_await_beads_tool(
    db: aiosqlite.Connection,
    guard_factory: GuardFactory | None = None,
) -> Tool[AwaitBeadsContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("mayor"),) if guard_factory else ()
    return Tool(
        name="await_beads",
        handler=await_beads,
        context=AwaitBeadsContext(db=db),
        guards=guards,
    )


# ---------------------------------------------------------------------------
# Polecat worker - pre-spawned coroutine, pulls bead IDs from the channel
# ---------------------------------------------------------------------------


async def polecat_worker(
    worker_id: int,
    proto: Agent,
    queue: Channel[int],
    db: aiosqlite.Connection,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    on_event: OnEventCallback,
) -> None:
    """Continuously pull bead IDs from *queue* and run a polecat agent for each.

    Exits cleanly when *queue* is closed (run_gastown calls channel.close()).
    A failed bead is left in_progress so the Witness can flag it on patrol.
    """
    n = 0
    async for bead_id in queue:
        try:
            row = await get_bead(db, bead_id)
            if row is None:
                continue
            _, title, *_ = row
            n += 1
            agent_id = f"polecat#{worker_id}:{title[:30]}"
            polecat_transport = transport_for("polecat", transport, role_models)
            polecat = proto.copy(transport=polecat_transport)
            polecat_ctx = AutoCompactStore(MemoryContextStore(), polecat_transport, keep_recent=6)
            task_msg = (
                f"Workspace: {WORKDIR}\n\n"
                f"Your assigned bead: [{bead_id}] {title}\n\n"
                f"Work this bead to completion, then close it with "
                f"`bead(action='close', id={bead_id})`."
            )
            async for event in polecat.run_stream(task_msg, polecat_ctx):
                await on_event(agent_id, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SpawnCrew - long-lived human-facing workers, Mayor's tool
# ---------------------------------------------------------------------------


class SpawnCrewContext(TypedDict):
    on_event: OnEventCallback
    transport: CompletionTransport
    role_models: dict[str, ModelSpec]
    proto: Agent
    db: aiosqlite.Connection
    crew_names: dict[str, int]


async def spawn_crew(
    name: Annotated[str, Field(description="Name for this Crew member, e.g. 'alice' or 'dom'")],
    task: Annotated[str, Field(description="Task or context for the Crew member")],
) -> str:
    """Spawn a Crew member - a long-lived coding agent for interactive or complex work.
    Unlike polecats, Crew are not bound to a single bead and are not managed by Witness.
    Use for design work, explorations, or tasks requiring sustained back-and-forth.
    Returns the Crew member's output when the session ends."""
    context: SpawnCrewContext = CONTEXT.get()
    context["crew_names"][name] = context["crew_names"].get(name, 0) + 1
    n = context["crew_names"][name]
    agent_id = f"crew#{n}:{name}" if n > 1 else f"crew:{name}"

    crew_transport = transport_for("crew", context["transport"], context["role_models"])
    member = context["proto"].copy(transport=crew_transport)
    crew_ctx = AutoCompactStore(MemoryContextStore(), crew_transport, keep_recent=10)
    task_msg = f"Workspace: {WORKDIR}\n\nYou are {name}, a Crew member.\n\n{task}"
    stream = member.run_stream(task_msg, crew_ctx)
    parts: list[str] = []
    async for event in stream:
        await context["on_event"](agent_id, event)
        if isinstance(event, TextDelta):
            parts.append(event.delta)
    return "".join(parts)


def make_spawn_crew_tool(
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    proto: Agent,
    db: aiosqlite.Connection,
    guard_factory: GuardFactory | None = None,
) -> Tool[SpawnCrewContext]:
    guards: tuple[PermissionGuard, ...] = (guard_factory("mayor"),) if guard_factory else ()
    return Tool(
        name="spawn_crew",
        handler=spawn_crew,
        context=SpawnCrewContext(
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
# Patrol loops - background asyncio tasks, not tool handlers
# ---------------------------------------------------------------------------


async def _patrol_sleep(delay: float, stop: asyncio.Event) -> bool:
    """Sleep for *delay* seconds or until *stop* is set. Returns True if stopped."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
        return True
    except TimeoutError:
        return False


async def witness_patrol(
    proto: Agent,
    db: aiosqlite.Connection,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    on_event: OnEventCallback,
    stop: asyncio.Event,
    start_delay: float = 30.0,
    max_delay: float = 300.0,
) -> None:
    """Periodically spawn a Witness to monitor polecat health."""
    delay = start_delay
    n = 0
    while True:
        if await _patrol_sleep(delay, stop):
            return
        if not await has_active_beads(db):
            delay = min(delay * 2, max_delay)
            continue
        n += 1
        summary = await bead_summary(db)
        w_transport = transport_for("witness", transport, role_models)
        witness = proto.copy(transport=w_transport)
        task_msg = (
            f"Workspace: {WORKDIR}\n\n"
            f"Witness patrol #{n}.\n\n"
            f"Current bead status:\n{summary}\n\n"
            f"Check the health of in-progress beads. Identify any stuck or stalled. "
            f"Report status concisely."
        )
        async for event in witness.run_stream(task_msg, MemoryContextStore()):
            await on_event(f"witness#{n}", event)
        delay = start_delay


async def refinery_patrol(
    proto: Agent,
    db: aiosqlite.Connection,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    on_event: OnEventCallback,
    stop: asyncio.Event,
    start_delay: float = 45.0,
    max_delay: float = 300.0,
) -> None:
    """Periodically spawn a Refinery to integrate closed polecat work."""
    delay = start_delay
    n = 0
    while True:
        if await _patrol_sleep(delay, stop):
            return
        unreviewed = await get_unreviewed_closed_beads(db)
        if not unreviewed:
            delay = min(delay * 2, max_delay)
            continue
        n += 1
        summary = await bead_summary(db)
        bead_list = "\n".join(f"  [{bid}] {title}" for bid, title in unreviewed)
        r_transport = transport_for("refinery", transport, role_models)
        refinery = proto.copy(transport=r_transport)
        task_msg = (
            f"Workspace: {WORKDIR}\n\n"
            f"Refinery patrol #{n}.\n\n"
            f"Bead status:\n{summary}\n\n"
            f"The following closed beads need integration review:\n{bead_list}\n\n"
            f"Integrate and verify each. When done with a bead add a note:\n"
            f"  bead(action='note', id=<id>, notes='refinery:reviewed')"
        )
        async for event in refinery.run_stream(task_msg, MemoryContextStore()):
            await on_event(f"refinery#{n}", event)
        delay = start_delay


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_gastown(
    task: str,
    workspace: Path,
    on_event: OnEventCallback,
    transport: CompletionTransport,
    role_models: dict[str, ModelSpec],
    toolbox: dict[str, Tool[Any]],
    guard_factory: GuardFactory | None = None,
    prompt_fn: Callable[[], str] | None = None,  # noqa: F841
    num_polecats: int = 5,
) -> str:
    """Run a Gas Town convoy on *task*."""
    assert "default" in role_models, "role_models must contain a 'default' key"

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    def og(name: str) -> tuple[PermissionGuard, ...]:
        return (guard_factory("mayor"),) if guard_factory else ()

    db_path = workspace / ".gas-town" / "beads.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.is_symlink():
        db_path.unlink()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(BEAD_DDL)
        await db.commit()

        toolbox = dict(toolbox)
        toolbox["bead"] = make_bead_tool(db)
        toolbox["analyze"] = make_analyze_tool(
            toolbox=toolbox,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            caller_role="specialist",
            guard_factory=None,
        )
        raw_roles = load_agents(ROLES_DIR, toolbox=toolbox)
        roles = {
            name: (desc, agent.copy(system=f"{SANDBOX_CONTEXT}\n\n---\n\n{agent.system}"))
            for name, (desc, agent) in raw_roles.items()
        }

        polecat_queue: Channel[int] = Channel()
        worker_tasks = [
            asyncio.create_task(
                polecat_worker(
                    worker_id=i + 1,
                    proto=roles["polecat"][1],
                    queue=polecat_queue,
                    db=db,
                    transport=transport,
                    role_models=role_models,
                    on_event=on_event,
                )
            )
            for i in range(num_polecats)
        ]

        stop = asyncio.Event()
        patrol_tasks = [
            asyncio.create_task(
                witness_patrol(
                    proto=roles["witness"][1],
                    db=db,
                    transport=transport,
                    role_models=role_models,
                    on_event=on_event,
                    stop=stop,
                )
            ),
            asyncio.create_task(
                refinery_patrol(
                    proto=roles["refinery"][1],
                    db=db,
                    transport=transport,
                    role_models=role_models,
                    on_event=on_event,
                    stop=stop,
                )
            ),
        ]

        mayor_bead = make_bead_tool(db, guards=og("bead"))
        sling_tool = make_sling_tool(db, polecat_queue, guard_factory)
        await_beads_tool = make_await_beads_tool(db, guard_factory)
        spawn_crew_tool = make_spawn_crew_tool(
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            proto=roles["crew"][1],
            db=db,
            guard_factory=guard_factory,
        )
        mayor_analyze = make_analyze_tool(
            toolbox=toolbox,
            on_event=on_event,
            transport=transport,
            role_models=role_models,
            caller_role="mayor",
            guard_factory=guard_factory,
        )
        mayor_read_tools = [
            Tool(name=t.name, description=t.description, handler=t.handler, context=t.context, guards=og(t.name))
            for t in toolbox.values()
            if t.name in {"list_files", "read_file"}
        ]

        mayor_transport = transport_for("mayor", transport, role_models)
        mayor = MAYOR.copy(
            system=f"{SANDBOX_CONTEXT}\n\n---\n\n{MAYOR.system}",
            transport=mayor_transport,
            max_iterations=200,
            tools=[
                mayor_bead,
                sling_tool,
                await_beads_tool,
                spawn_crew_tool,
                mayor_analyze,
                *mayor_read_tools,
            ],
        )
        mayor_ctx = AutoCompactStore(MemoryContextStore(), mayor_transport, keep_recent=10)

        stream = mayor.run_stream(task, mayor_ctx)
        parts: list[str] = []
        try:
            async for event in stream:
                await on_event("mayor", event)
                if isinstance(event, TextDelta):
                    parts.append(event.delta)
        finally:
            polecat_queue.close()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            stop.set()
            await asyncio.gather(*patrol_tasks, return_exceptions=True)

    return "".join(parts)
