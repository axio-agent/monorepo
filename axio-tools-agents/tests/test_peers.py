from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import IterationEnd, StreamEvent, TextDelta
from axio.messages import Message
from axio.tool import Tool
from axio.types import StopReason, Usage

from axio_tools_agents.peers import (
    PeerMessage,
    PeerServer,
    format_message_for_dialog,
    list_peers,
    send_message,
    set_spawn_agent_factory,
    spawn_agent,
)


@pytest.fixture(autouse=True)
def peer_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))


async def _noop_handler(message: PeerMessage) -> None:
    return None


async def test_list_peers_filters_to_current_project_by_default(tmp_path: Path) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    current = await PeerServer("current", kind="test", handler=_noop_handler, project=str(project_a)).start()
    same_project = await PeerServer(
        "same",
        kind="test",
        handler=_noop_handler,
        project=str(project_a),
    ).start(set_current=False)
    other_project = await PeerServer(
        "other",
        kind="test",
        handler=_noop_handler,
        project=str(project_b),
    ).start(set_current=False)

    try:
        scoped = await list_peers()
        assert same_project.id in scoped
        assert current.id not in scoped
        assert other_project.id not in scoped

        all_projects = await list_peers(all_projects=True)
        assert same_project.id in all_projects
        assert other_project.id in all_projects
    finally:
        await current.close()
        await same_project.close()
        await other_project.close()


async def test_send_message_delivers_by_global_agent_id(tmp_path: Path) -> None:
    received: list[PeerMessage] = []

    async def handler(message: PeerMessage) -> None:
        received.append(message)

    sender = await PeerServer("sender", kind="test", handler=_noop_handler, project=str(tmp_path)).start()
    recipient = await PeerServer(
        "recipient",
        kind="test",
        handler=handler,
        project=str(tmp_path),
    ).start(set_current=False)

    try:
        result = await send_message(agent_id=recipient.id, message="hello")
        assert result.startswith("Delivered message")
        assert len(received) == 1
        assert received[0].from_id == sender.id
        assert received[0].body == "hello"
    finally:
        await sender.close()
        await recipient.close()


class _MessagingTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sender: PeerServer | None = None

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        prompt = messages[-1].content[0].text  # type: ignore[attr-defined]
        self.calls.append(prompt)
        if len(self.calls) == 1:
            from axio_tools_agents.peers import list_peer_records

            agent_id = next(record.id for record in await list_peer_records() if record.kind == "spawned-agent")
            assert self.sender is not None
            from axio_tools_agents.peers import peer_context

            with peer_context(self.sender):
                delivered = await send_message(agent_id=agent_id, message="follow-up")
            assert delivered.startswith("Delivered message")
            yield TextDelta(index=0, delta="first")
        else:
            assert "follow-up" in prompt
            yield TextDelta(index=0, delta="second")
        yield IterationEnd(iteration=len(self.calls), stop_reason=StopReason.end_turn, usage=Usage(0, 0))


async def test_spawn_agent_registers_peer_and_processes_inbound_after_turn(tmp_path: Path) -> None:
    transport = _MessagingTransport()
    transport.sender = await PeerServer(
        "sender",
        kind="test",
        handler=_noop_handler,
        project=str(tmp_path),
    ).start(set_current=False)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        assert not inherit_context
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        result = await spawn_agent(task="initial")
        assert result == "first\n\nsecond"
        assert transport.calls[0].endswith("initial")
        assert transport.calls[1].endswith(
            format_message_for_dialog(
                PeerMessage(
                    id="unused",
                    from_id=transport.sender.id,
                    from_name=transport.sender.name,
                    to_id="unused",
                    body="follow-up",
                    sent_at=0,
                )
            )
        )
    finally:
        set_spawn_agent_factory(None)
        await transport.sender.close()
