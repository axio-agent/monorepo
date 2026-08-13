"""Block until something happens, instead of asking again and again.

Polling from the model is the expensive way to wait: every check is a whole turn
— the system prompt re-prefilled, tokens generated to say "still waiting" — and
on a local model that costs more than the work being waited on. Worse, an agent
polling `list_peers` never learns that what it waits for has died, so it can
loop forever.

`monitor` collapses that into one call: it blocks inside the tool, where waiting
is free, and returns as soon as any watched condition fires or the timeout runs
out. Polling still happens for files and processes, which expose no event, but
it happens here rather than through the model.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from axio import background
from axio.field import StrictStr

from axio_tools_agents.peers import (
    background_agent_state,
    next_peer_message,
    pending_message_count,
    wait_local_background_agents_idle,
)

DEFAULT_TIMEOUT = 300.0
POLL_INTERVAL = 1.0


def _stat_signature(path: str) -> tuple[float, int] | None:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return st.st_mtime, st.st_size


async def _watch_paths(paths: list[str]) -> str:
    """Resolve when any listed path is created, removed or modified."""
    baseline = {path: _stat_signature(path) for path in paths}
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        for path in paths:
            current = _stat_signature(path)
            if current != baseline[path]:
                if baseline[path] is None:
                    return f"path created: {path}"
                if current is None:
                    return f"path removed: {path}"
                return f"path changed: {path}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _watch_pids(pids: list[int]) -> str:
    while True:
        for pid in pids:
            if not _pid_alive(pid):
                return f"process exited: {pid}"
        await asyncio.sleep(POLL_INTERVAL)


async def _watch_tasks(handles: list[str]) -> str:
    while True:
        for handle in handles:
            call = background.get(handle)
            if call is None:
                return f"unknown task handle: {handle}"
            if call.state != "running":
                return f"task finished: {handle}"
        await background.next_completion()


async def _watch_message() -> str:
    # Messages already delivered are the common case right after spawned agents
    # report back: waiting for the *next* one would block behind them, and they
    # cannot be consumed until this turn ends.
    waiting = pending_message_count()
    if waiting:
        # Saying only that they exist invites the caller to ask again, and again:
        # they are delivered as the next prompt, so the turn has to end first.
        return (
            f"{waiting} message(s) have arrived. They are delivered as your next prompt, so stop calling "
            f"tools and finish this turn to read them — calling monitor again will only repeat this."
        )
    message = await next_peer_message()
    # Same as above: knowing who wrote is not reading what they wrote, and only
    # ending the turn delivers it.
    return (
        f"a message from {message.from_name} ({message.from_id}) has arrived. It is delivered as your "
        f"next prompt, so stop calling tools and finish this turn to read it."
    )


async def _watch_agents(agent_ids: list[str], wait_all: bool) -> str:
    if wait_all:
        await wait_local_background_agents_idle(agent_ids)
        return "all watched agents are idle"
    while True:
        for agent_id in agent_ids:
            state, _ = background_agent_state(agent_id)
            if state == "idle":
                return f"agent finished: {agent_id}"
            if state == "unknown":
                return f"agent gone: {agent_id}"
        await asyncio.sleep(POLL_INTERVAL)


def _route_agent_ids(agent_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Sort ids into agents, task handles passed by mistake, and strangers.

    Waiting on an id nobody knows is indistinguishable from waiting on one that
    has finished: there is nothing to wait for either way, so the wait returns
    at once and reports success. That answer is a lie whenever the id is simply
    wrong - which it is every time a background task handle lands here, since
    spawn_agent hands one out and it is not an agent id.
    """
    agents, handles, strangers = [], [], []
    for agent_id in agent_ids:
        if background.get(agent_id) is not None:
            handles.append(agent_id)
        elif background_agent_state(agent_id)[0] == "unknown":
            strangers.append(agent_id)
        else:
            agents.append(agent_id)
    return agents, handles, strangers


def _agent_report(agent_ids: list[str]) -> list[str]:
    lines = []
    for agent_id in agent_ids:
        state, error = background_agent_state(agent_id)
        line = f"  {agent_id}: {state}"
        if error:
            line += f" (last error — {error})"
        lines.append(line)
    return lines


async def monitor(
    agents: list[StrictStr] | None = None,
    tasks: list[StrictStr] | None = None,
    paths: list[StrictStr] | None = None,
    pids: list[int] | None = None,
    messages: bool | None = None,
    wait_all: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Wait for something to happen instead of polling for it.

    Blocks until one of the watched conditions fires, then reports what happened
    and the state of everything watched. Watch spawned agents by id (`agents`),
    detached tool calls by handle (`tasks`), files or directories (`paths`),
    processes (`pids`), or incoming messages from other agents (`messages`).
    Finished tasks are reported with their full output — this is how a call made
    with background=true delivers its result. With `wait_all=true` and `agents`
    given, it returns only once every one of them has finished.

    Messages are watched only when nothing else is named, or when you ask for
    them with `messages=true`. Naming what to wait for and then returning on an
    unrelated message would make `wait_all` mean nothing: a child that reports
    in early ends the wait for every other child.

    Returns on timeout as well, reporting what is still outstanding, so a
    result is never an error — decide from the report whether to wait again.
    Prefer this over calling list_peers repeatedly: a crashed agent counts as
    finished here and its failure is reported, which polling cannot tell you.
    """
    if messages is None:
        messages = not (agents or tasks or paths or pids)
    agent_ids, misrouted, strangers = _route_agent_ids([str(a) for a in agents or []])
    task_handles = [str(t) for t in tasks or []] + misrouted
    watched: dict[str, asyncio.Task[str]] = {}
    notes = []
    if misrouted:
        notes.append(f"watched as tasks, not agents (spawn_agent returns task handles): {', '.join(misrouted)}")
    if strangers:
        notes.append(f"no such agent, not watched (see list_peers for live ids): {', '.join(strangers)}")

    if agent_ids:
        watched["agents"] = asyncio.create_task(_watch_agents(agent_ids, wait_all))
    if task_handles:
        watched["tasks"] = asyncio.create_task(_watch_tasks(task_handles))
    if paths:
        watched["paths"] = asyncio.create_task(_watch_paths([str(p) for p in paths]))
    if pids:
        watched["pids"] = asyncio.create_task(_watch_pids(list(pids)))
    if messages:
        watched["messages"] = asyncio.create_task(_watch_message())

    if not watched:
        nothing = "monitor: nothing to watch — pass agents, tasks, paths, pids, or messages=true"
        return "\n".join([nothing, *notes])

    done, pending = await asyncio.wait(watched.values(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    triggers = []
    for task in done:
        exc = task.exception()
        triggers.append(f"watch failed: {exc}" if exc is not None else task.result())

    lines = [f"Triggered: {t}" for t in triggers] or [f"Timed out after {timeout:g}s with nothing triggered"]
    lines += [f"Note: {n}" for n in notes]
    if agent_ids:
        lines.append("Agents:")
        lines += _agent_report(agent_ids)
    if task_handles:
        lines.append("Tasks:")
        lines += [f"  {background.describe(h)}" for h in task_handles]
    return "\n".join(lines)


__all__ = ["monitor"]
