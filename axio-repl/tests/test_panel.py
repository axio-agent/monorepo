import asyncio
from collections.abc import AsyncGenerator

import pytest
from axio import background

from axio_repl import _panel


class _Record:
    def __init__(self, agent_id: str, name: str) -> None:
        self.id = agent_id
        self.name = name


@pytest.fixture(autouse=True)
async def clean_registry() -> AsyncGenerator[None, None]:
    yield
    await background.cancel_all()


def test_silent_when_nothing_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)
    # A status line that always says something wastes a row of terminal.
    assert _panel.agent_summary() == ""


def test_counts_agents_by_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _panel,
        "local_background_agent_records",
        lambda: [_Record("a", "one"), _Record("b", "two"), _Record("c", "three")],
    )
    states = {"a": ("running", None), "b": ("running", None), "c": ("idle", None)}
    monkeypatch.setattr(_panel, "background_agent_state", lambda i: states[i])
    assert _panel.agent_summary() == "agents: 1 idle, 2 running"


def test_names_the_failed_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", lambda: [_Record("a", "analyst")])
    monkeypatch.setattr(_panel, "background_agent_state", lambda i: ("idle", "StreamError: boom"))
    summary = _panel.agent_summary()
    assert "failed: analyst" in summary


@pytest.mark.asyncio
async def test_reports_detached_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)

    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(60)
        return "never"

    background.start("shell", slow())
    await started.wait()
    assert "tasks: 1 running" in _panel.agent_summary()


@pytest.mark.asyncio
async def test_finished_calls_are_flagged_until_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)

    async def quick() -> str:
        return "done"

    handle = background.start("shell", quick())
    await background.get(handle).task  # type: ignore[union-attr]
    assert "1 ready to collect" in _panel.agent_summary()

    background.describe(handle)  # reading it marks it collected
    assert "ready to collect" not in _panel.agent_summary()
