"""A status line pinned to the bottom of an otherwise ordinary terminal.

Deliberately not a full-screen UI. Output keeps going to real stdout, so it
scrolls into the terminal's own history exactly as before and nothing here owns
the screen buffer. prompt_toolkit reserves only the lines it draws, and
patch_stdout redraws the input when a background agent prints underneath it —
which is what made typing unusable while a swarm was reporting back.

Everything the line reports is a pure function of state passed in, so what it
says can be tested without a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axio import background
from axio.models import ModelSpec
from axio.types import Usage
from axio_tools_agents.peers import background_agent_state, local_background_agent_records

HISTORY_PATH = Path.home() / ".axio_repl_history"

SEPARATOR = " │ "

MAIN_AGENT = "main"


@dataclass
class SessionStats:
    """What the session has spent, and how full its context is.

    Usage is recorded per iteration rather than per turn so the line moves while
    an answer is still being written; a turn's total is the sum of its
    iterations, so counting both would double everything.

    Cost is priced at the model in use when the tokens were spent, which is the
    right answer for a spawned agent sharing the parent's model and an
    approximation for one that does not.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    context_tokens: int = 0
    per_model: dict[str, Usage] = field(default_factory=dict)

    def record(self, agent_id: str, usage: Usage, model: ModelSpec | None) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if agent_id == MAIN_AGENT:
            # The prompt of the latest iteration is what occupies the window
            # now: it grows with the conversation and drops when it is compacted.
            self.context_tokens = usage.input_tokens
        if model is None:
            return
        self.per_model[model.id] = self.per_model.get(model.id, Usage(0, 0)) + usage
        self.cost += (usage.input_tokens * model.input_cost + usage.output_tokens * model.output_cost) / 1_000_000


def compact(value: int) -> str:
    """A token count short enough to sit in a status line."""
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return f"{value / 1_000_000:.2f}M".replace(".00M", "M")


def format_cost(cost: float) -> str:
    if cost == 0:
        return "$0"
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 10:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


def agent_summary(now: float | None = None) -> str:
    """Background agents and detached tool calls, when there are any.

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

    return SEPARATOR.join(parts)


def status_line(model: ModelSpec | None, stats: SessionStats) -> str:
    """The whole line: which model, how full, how much spent, what is running."""
    parts: list[str] = []
    if model is not None:
        parts.append(model.id)
        if model.context_window:
            parts.append(f"ctx {compact(stats.context_tokens)}/{compact(model.context_window)}")
    parts.append(f"{compact(stats.input_tokens)} in / {compact(stats.output_tokens)} out")
    parts.append(format_cost(stats.cost))
    agents = agent_summary()
    if agents:
        parts.append(agents)
    return SEPARATOR.join(parts)


def submit_bindings() -> Any:
    """Escape sends, so that Enter is free to be a newline.

    A prompt worth typing into holds more than one line - a paragraph of task,
    a pasted traceback - and Enter cannot both end a line and end the message.
    Escape is bound without ``eager``: a lone press only becomes Escape once the
    parser has waited long enough to rule out an arrow key, which begins with
    the same byte.
    """
    from prompt_toolkit.key_binding import KeyBindings

    bindings = KeyBindings()

    @bindings.add("escape")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    return bindings


def make_session(status: Any = None) -> Any:
    """A prompt session with history, the status line, and Escape to send.

    The default toolbar style is reverse video, which paints a solid white band
    across the bottom of the terminal. Dim text on the terminal's own background
    keeps the line readable without turning it into a wall.

    Up recalls the previous message for editing once the cursor is on the first
    line, and moves within the text before that - prompt_toolkit's own
    behaviour for a multiline buffer, which is the one wanted here.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    return PromptSession(
        history=FileHistory(str(HISTORY_PATH)),
        bottom_toolbar=status or (lambda: agent_summary() or None),
        style=Style.from_dict({"bottom-toolbar": "noreverse bg:default fg:#808080"}),
        multiline=True,
        key_bindings=submit_bindings(),
        # Redraw while idle so finished agents show up without a keypress.
        refresh_interval=0.5,
    )
