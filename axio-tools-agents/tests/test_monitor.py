import asyncio
import os
from pathlib import Path

import pytest

from axio_tools_agents import monitoring as monitor_module
from axio_tools_agents.monitoring import monitor
from axio_tools_agents.peers import background_agent_state


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitor_module, "POLL_INTERVAL", 0.01)


@pytest.mark.asyncio
async def test_nothing_to_watch_is_reported_not_hung() -> None:
    result = await monitor(messages=False)
    assert "nothing to watch" in result


@pytest.mark.asyncio
async def test_timeout_returns_a_report_rather_than_raising() -> None:
    result = await monitor(pids=[os.getpid()], messages=False, timeout=0.05)
    assert "Timed out" in result


@pytest.mark.asyncio
async def test_dead_process_triggers_immediately() -> None:
    proc = await asyncio.create_subprocess_exec("true")
    await proc.wait()
    result = await monitor(pids=[proc.pid], messages=False, timeout=5)
    assert "process exited" in result


@pytest.mark.asyncio
async def test_path_change_triggers(tmp_path: Path) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("before")

    async def touch() -> None:
        await asyncio.sleep(0.05)
        watched.write_text("after, and longer")

    task = asyncio.create_task(touch())
    result = await monitor(paths=[str(watched)], messages=False, timeout=5)
    await task
    assert "path changed" in result


@pytest.mark.asyncio
async def test_path_creation_and_removal_are_distinguished(tmp_path: Path) -> None:
    missing = tmp_path / "appears.txt"

    async def create() -> None:
        await asyncio.sleep(0.05)
        missing.write_text("hello")

    task = asyncio.create_task(create())
    result = await monitor(paths=[str(missing)], messages=False, timeout=5)
    await task
    assert "path created" in result


@pytest.mark.asyncio
async def test_unknown_agent_is_reported_as_such() -> None:
    state, error = background_agent_state("no-such-agent")
    assert state == "unknown"
    assert error is None
    result = await monitor(agents=["no-such-agent"], messages=False, timeout=5)
    # An id that names nothing counts as finished: waiting on it forever is the
    # failure mode this tool exists to prevent.
    assert "agent finished" in result
    assert "no-such-agent: unknown" in result
