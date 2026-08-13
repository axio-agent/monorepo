"""A status line pinned to the bottom of an otherwise ordinary terminal.

Deliberately not a full-screen UI. Output keeps going to real stdout, so it
scrolls into the terminal's own history exactly as before and nothing here owns
the screen buffer. prompt_toolkit reserves only the lines it draws, and
patch_stdout redraws the input when a background agent prints underneath it —
which is what made typing unusable while a swarm was reporting back.

The summary is a pure function so the state it reports can be tested without a
terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axio import background
from axio_tools_agents.peers import background_agent_state, local_background_agent_records

HISTORY_PATH = Path.home() / ".axio_repl_history"


def agent_summary(now: float | None = None) -> str:
    """One line describing background agents and detached tool calls.

    Empty when there is nothing running — the line should not cost space until
    it has something to say.
    """
    parts: list[str] = []

    states: dict[str, int] = {}
    failed: list[str] = []
    for record in local_background_agent_records():
        state, error = background_agent_state(record.id)
        states[state] = states.get(state, 0) + 1
        if error:
            failed.append(record.name)
    if states:
        counts = ", ".join(f"{count} {state}" for state, count in sorted(states.items()))
        parts.append(f"agents: {counts}")
    if failed:
        parts.append(f"failed: {', '.join(sorted(set(failed))[:3])}")

    calls = background.snapshot()
    running = [c for c in calls if c.state == "running"]
    done = [c for c in calls if c.state != "running" and not c.collected]
    if running:
        parts.append(f"tasks: {len(running)} running")
    if done:
        parts.append(f"{len(done)} ready to collect")

    return " │ ".join(parts)


def make_session() -> Any:
    """A prompt session with history and the status line attached."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    return PromptSession(
        history=FileHistory(str(HISTORY_PATH)),
        bottom_toolbar=lambda: agent_summary() or None,
        # Redraw while idle so finished agents show up without a keypress.
        refresh_interval=0.5,
    )
