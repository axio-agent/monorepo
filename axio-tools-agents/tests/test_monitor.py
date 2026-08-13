import asyncio
import os
from pathlib import Path

import pytest
from axio import background

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
    # It must not hang, and it must not claim the thing finished either: the id
    # names nothing, and "finished" would be read as "the work is done".
    assert "no such agent" in result
    assert "agent finished" not in result


@pytest.mark.asyncio
async def test_already_delivered_messages_do_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    # The deadlock this prevents: spawned agents report back, their messages sit
    # unread until the turn ends, and the turn is a monitor() call waiting for a
    # message that has already arrived.
    from axio_tools_agents import peers

    monkeypatch.setattr(peers, "_pending_probe", lambda: 3)
    result = await monitor(timeout=5)
    assert "3 message(s) have arrived" in result
    assert "finish this turn" in result


@pytest.mark.asyncio
async def test_no_pending_messages_still_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    from axio_tools_agents import peers

    monkeypatch.setattr(peers, "_pending_probe", lambda: 0)
    result = await monitor(timeout=0.05)
    assert "Timed out" in result


@pytest.mark.asyncio
async def test_a_task_handle_passed_as_an_agent_is_watched_as_a_task() -> None:
    # spawn_agent hands out a background task handle and tells the caller to
    # collect it with tasks=[...]; passing it to agents=[...] used to return
    # "all watched agents are idle" instantly, before the agent had done a thing.
    async def slow() -> str:
        await asyncio.sleep(30)
        return "done"

    handle = background.start("spawn_agent", slow())
    try:
        result = await monitor(agents=[handle], messages=False, timeout=0.5)
        assert "idle" not in result
        assert "watched as tasks" in result
        assert handle in result
    finally:
        await background.cancel_all()


@pytest.mark.asyncio
async def test_an_incoming_message_says_the_turn_must_end() -> None:
    # Naming the sender is not reading the message: it is delivered as the next
    # prompt, so a caller told only "message from X" calls monitor again.
    from axio_tools_agents import peers

    async def deliver() -> None:
        await asyncio.sleep(0.05)
        peers._notify_message(
            peers.PeerMessage(
                id="m-1",
                from_id="child-1",
                from_name="health-check",
                to_id="parent",
                body="done",
                sent_at=0.0,
            )
        )

    task = asyncio.create_task(deliver())
    try:
        result = await monitor(timeout=5)
    finally:
        await task
    assert "health-check" in result
    assert "finish this turn" in result
