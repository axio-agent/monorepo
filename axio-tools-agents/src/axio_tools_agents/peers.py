from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from axio.agent import Agent
from axio.context import ContextStore
from axio.events import StreamEvent, TextDelta
from axio.field import StrictStr

MAX_MESSAGE_CHARS = 200_000
MAX_WIRE_BYTES = MAX_MESSAGE_CHARS * 4 + 4096


@dataclass(frozen=True, slots=True)
class PeerRecord:
    id: str
    name: str
    kind: str
    project: str
    pid: int
    cwd: str
    socket_path: str
    started_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "project": self.project,
            "pid": self.pid,
            "cwd": self.cwd,
            "socket_path": self.socket_path,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeerRecord:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            kind=str(data["kind"]),
            project=str(data.get("project") or data["cwd"]),
            pid=int(data["pid"]),
            cwd=str(data["cwd"]),
            socket_path=str(data["socket_path"]),
            started_at=float(data["started_at"]),
        )


@dataclass(frozen=True, slots=True)
class PeerMessage:
    id: str
    from_id: str
    from_name: str
    to_id: str
    body: str
    sent_at: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeerMessage:
        return cls(
            id=str(data["id"]),
            from_id=str(data["from_id"]),
            from_name=str(data["from_name"]),
            to_id=str(data["to_id"]),
            body=str(data["body"]),
            sent_at=float(data["sent_at"]),
        )


MessageHandler = Callable[[PeerMessage], Awaitable[None]]
SpawnAgentFactory = Callable[[bool], Awaitable[tuple[Agent, ContextStore]]]
AgentEventHandler = Callable[[str, StreamEvent], Awaitable[None]]

_current_peer: contextvars.ContextVar[PeerServer | None] = contextvars.ContextVar(
    "axio_tools_agents_current_peer",
    default=None,
)
_spawn_agent_factory: SpawnAgentFactory | None = None
_agent_event_handler: AgentEventHandler | None = None


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug[:48] or "peer"


def _runtime_dir() -> Path:
    override = os.environ.get("AXIO_PEER_DIR")
    if override:
        path = Path(override)
    elif xdg := os.environ.get("XDG_RUNTIME_DIR"):
        path = Path(xdg) / "axio-agent" / "peers"
    else:
        path = Path(tempfile.gettempdir()) / f"axio-agent-{os.getuid()}" / "peers"
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.chmod(0o700)
    return path


def _safe_unlink(path: str | Path) -> None:
    with contextlib.suppress(FileNotFoundError, OSError):
        Path(path).unlink()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _registry_path(peer_id: str) -> Path:
    return _runtime_dir() / f"{peer_id}.json"


def _read_records_sync() -> list[PeerRecord]:
    records: list[PeerRecord] = []
    for path in _runtime_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            record = PeerRecord.from_dict(data)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            _safe_unlink(path)
            continue
        if not _pid_alive(record.pid) or not Path(record.socket_path).exists():
            _safe_unlink(path)
            _safe_unlink(record.socket_path)
            continue
        records.append(record)
    return sorted(records, key=lambda r: (r.name, r.id))


async def list_peer_records() -> list[PeerRecord]:
    return await asyncio.to_thread(_read_records_sync)


def set_current_peer(peer: PeerServer | None) -> contextvars.Token[PeerServer | None]:
    return _current_peer.set(peer)


def current_peer() -> PeerServer | None:
    return _current_peer.get()


@contextlib.contextmanager
def peer_context(peer: PeerServer) -> Iterator[None]:
    token = set_current_peer(peer)
    try:
        yield
    finally:
        _current_peer.reset(token)


def set_spawn_agent_factory(factory: SpawnAgentFactory | None) -> None:
    global _spawn_agent_factory
    _spawn_agent_factory = factory


def set_agent_event_handler(handler: AgentEventHandler | None) -> None:
    global _agent_event_handler
    _agent_event_handler = handler


def format_message_for_dialog(message: PeerMessage) -> str:
    return f"Peer message from {message.from_name} ({message.from_id}):\n\n{message.body}"


def _normalize_project(value: str | Path | None) -> str:
    return str(Path(value or Path.cwd()).resolve())


def _current_project() -> str:
    current = current_peer()
    return current.project if current is not None else _normalize_project(None)


def _visible_records(
    records: list[PeerRecord],
    *,
    include_self: bool = False,
    all_projects: bool = False,
) -> list[PeerRecord]:
    current = current_peer()
    project = _current_project()
    return [
        record
        for record in records
        if (all_projects or record.project == project) and (include_self or current is None or record.id != current.id)
    ]


def _resolve_peer_by_id(records: list[PeerRecord], agent_id: str) -> PeerRecord | None:
    return next((record for record in records if record.id == agent_id), None)


async def _write_response(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    writer.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
    await writer.drain()


class PeerServer:
    def __init__(
        self,
        name: str,
        *,
        kind: str,
        handler: MessageHandler,
        cwd: str | None = None,
        project: str | None = None,
        peer_id: str | None = None,
    ) -> None:
        self.id = peer_id or f"{_safe_slug(name)}-{os.getpid()}-{uuid4().hex[:8]}"
        self.name = name
        self.kind = kind
        self.cwd = _normalize_project(cwd)
        self.project = _normalize_project(project or self.cwd)
        self._handler = handler
        self._server: asyncio.AbstractServer | None = None
        self._socket_path: Path | None = None
        self._socket_token = uuid4().hex[:16]
        self._started_at = time.time()

    @property
    def record(self) -> PeerRecord:
        if self._socket_path is None:
            raise RuntimeError("PeerServer is not started")
        return PeerRecord(
            id=self.id,
            name=self.name,
            kind=self.kind,
            project=self.project,
            pid=os.getpid(),
            cwd=self.cwd,
            socket_path=str(self._socket_path),
            started_at=self._started_at,
        )

    async def start(self, *, set_current: bool = True) -> Self:
        if self._server is not None:
            return self
        socket_path = _runtime_dir() / f"{self._socket_token}.sock"
        _safe_unlink(socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=socket_path,
            limit=MAX_WIRE_BYTES + 1024,
        )
        self._socket_path = socket_path
        with contextlib.suppress(OSError):
            socket_path.chmod(0o600)
        await asyncio.to_thread(self._write_registry)
        if set_current:
            set_current_peer(self)
        return self

    async def close(self) -> None:
        if current_peer() is self:
            set_current_peer(None)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        _safe_unlink(_registry_path(self.id))
        if self._socket_path is not None:
            _safe_unlink(self._socket_path)
            self._socket_path = None

    def _write_registry(self) -> None:
        record_path = _registry_path(self.id)
        temp_path = record_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.record.to_dict(), sort_keys=True), encoding="utf-8")
        temp_path.replace(record_path)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readline()
            if not raw:
                return
            if len(raw) > MAX_WIRE_BYTES:
                await _write_response(writer, {"ok": False, "error": "message too large"})
                return
            data = json.loads(raw.decode("utf-8"))
            if data.get("type") != "message":
                await _write_response(writer, {"ok": False, "error": "unsupported message type"})
                return
            message = PeerMessage.from_dict(data)
            if message.to_id != self.id:
                await _write_response(writer, {"ok": False, "error": "wrong recipient"})
                return
            if len(message.body) > MAX_MESSAGE_CHARS:
                await _write_response(writer, {"ok": False, "error": "message too large"})
                return
            await self._handler(message)
            await _write_response(writer, {"ok": True})
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
            await _write_response(writer, {"ok": False, "error": f"bad request: {exc}"})
        # Listener callbacks are supplied by applications, so protocol errors
        # are the only useful isolation boundary for unexpected callback failures.
        except Exception as exc:
            await _write_response(writer, {"ok": False, "error": f"handler failed: {exc}"})
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()


async def list_peers(all_projects: bool = False) -> str:
    """List other running axio agents. By default only peers in the current
    project are returned. Pass all_projects=true to inspect peers from every
    project on this host."""
    records = _visible_records(await list_peer_records(), all_projects=all_projects)
    if not records:
        if all_projects:
            return "No peers registered."
        return f"No peers registered for project: {_current_project()}"
    lines = ["Available peers:" if all_projects else f"Available peers for project {_current_project()}:"]
    for record in records:
        lines.append(
            f"- id={record.id} name={record.name!r} kind={record.kind} "
            f"project={record.project} pid={record.pid} cwd={record.cwd}"
        )
    return "\n".join(lines)


async def send_message(agent_id: StrictStr, message: StrictStr) -> str:
    """Send a message to another running axio peer by global agent id. Use
    list_peers first to find the id. Incoming peer messages appear automatically
    in the recipient's dialog; there is no receive tool."""
    if len(message) > MAX_MESSAGE_CHARS:
        return f"Message is too large; limit is {MAX_MESSAGE_CHARS} characters."

    records = _visible_records(await list_peer_records(), all_projects=True)
    peer = _resolve_peer_by_id(records, agent_id)
    if peer is None:
        return f"No peer found for agent_id={agent_id!r}. Call list_peers first."

    sender = current_peer()
    if sender is not None and peer.id == sender.id:
        return "Cannot send a peer message to the current agent."

    from_id = sender.id if sender is not None else f"unregistered-{os.getpid()}"
    from_name = sender.name if sender is not None else f"unregistered-{os.getpid()}"
    payload = {
        "type": "message",
        "id": uuid4().hex,
        "from_id": from_id,
        "from_name": from_name,
        "to_id": peer.id,
        "body": message,
        "sent_at": time.time(),
    }

    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(peer.socket_path), timeout=3)
    except (OSError, TimeoutError) as exc:
        _safe_unlink(_registry_path(peer.id))
        _safe_unlink(peer.socket_path)
        return f"Failed to connect to peer {peer.id}: {exc}"

    try:
        writer.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
        await asyncio.wait_for(writer.drain(), timeout=3)
        raw = await asyncio.wait_for(reader.readline(), timeout=3)
    except (ConnectionError, OSError, TimeoutError) as exc:
        return f"Failed to send message to peer {peer.id}: {exc}"
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await writer.wait_closed()

    try:
        response = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"Peer {peer.id} returned an invalid response: {exc}"
    if response.get("ok") is not True:
        return f"Peer {peer.id} rejected the message: {response.get('error', 'unknown error')}"
    return f"Delivered message to {peer.name} ({peer.id})."


async def _run_agent_turns_with_peer(
    *,
    agent: Agent,
    context: ContextStore,
    initial_task: str,
    name: str,
    kind: str,
    project: str,
    cwd: str,
) -> str:
    inbox: asyncio.Queue[str] = asyncio.Queue()
    accept_lock = asyncio.Lock()
    closing = False

    async def _on_message(message: PeerMessage) -> None:
        async with accept_lock:
            if closing:
                raise RuntimeError("agent is no longer accepting messages")
            inbox.put_nowait(format_message_for_dialog(message))

    peer = await PeerServer(name, kind=kind, handler=_on_message, project=project, cwd=cwd).start(set_current=False)
    results: list[str] = []
    try:
        with peer_context(peer):
            task = initial_task
            while True:
                parts: list[str] = []
                async for event in agent.run_stream(task, context):
                    if _agent_event_handler is not None:
                        await _agent_event_handler(peer.id, event)
                    if isinstance(event, TextDelta):
                        parts.append(event.delta)
                results.append("".join(parts))
                try:
                    task = inbox.get_nowait()
                    continue
                except asyncio.QueueEmpty:
                    pass
                async with accept_lock:
                    try:
                        task = inbox.get_nowait()
                    except asyncio.QueueEmpty:
                        closing = True
                        break
    finally:
        await peer.close()
    return "\n\n".join(part for part in results if part)


async def spawn_agent(
    task: StrictStr,
    inherit_context: bool = False,
    name: str | None = None,
) -> str:
    """Spawn an independent agent and return its final response. By default the
    spawned agent starts with an empty context. Set inherit_context=true only
    when the spawned agent must see the current conversation. The spawned agent
    is registered as an IPC peer while it runs, so other agents can send messages
    to it by global agent id."""
    if _spawn_agent_factory is None:
        return "spawn_agent is not configured"

    agent, context = await _spawn_agent_factory(inherit_context)
    parent = current_peer()
    project = parent.project if parent is not None else _current_project()
    cwd = parent.cwd if parent is not None else _normalize_project(None)
    base_name = name or f"spawn_agent:{task[:40]}"
    return await _run_agent_turns_with_peer(
        agent=agent,
        context=context,
        initial_task=task,
        name=base_name,
        kind="spawned-agent",
        project=project,
        cwd=cwd,
    )


spawn_agent._tool_concurrency = 3  # type: ignore[attr-defined]
