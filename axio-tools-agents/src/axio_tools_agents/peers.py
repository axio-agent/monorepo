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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from axio.agent import Agent
from axio.context import ContextStore
from axio.events import Error, StreamEvent
from axio.field import StrictStr

from axio_tools_agents.names import generate_name

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
StopHandler = Callable[[str, str], Awaitable[None]]
SpawnAgentFactory = Callable[[bool], Awaitable[tuple[Agent, ContextStore]]]
AgentEventHandler = Callable[[str, StreamEvent], Awaitable[None]]

_current_peer: contextvars.ContextVar[PeerServer | None] = contextvars.ContextVar(
    "axio_tools_agents_current_peer",
    default=None,
)
_spawn_agent_factory: SpawnAgentFactory | None = None
_agent_event_handler: AgentEventHandler | None = None


@dataclass(slots=True)
class _QueuedPrompt:
    prompt: str | None
    done: asyncio.Future[None] | None = None


@dataclass(slots=True)
class _BackgroundAgent:
    agent: Agent
    context: ContextStore
    peer: PeerServer
    inbox: asyncio.Queue[_QueuedPrompt] = field(default_factory=asyncio.Queue)
    idle_waiters: list[asyncio.Future[None]] = field(default_factory=list)
    runner: asyncio.Task[None] | None = None
    current_turn: asyncio.Task[None] | None = None
    stopping: bool = False
    # A failed turn leaves the agent alive and waiting, so without this the
    # parent cannot tell a crashed agent from one that finished its work.
    last_error: str | None = None


_background_agents: dict[str, _BackgroundAgent] = {}


def _is_background_idle(background: _BackgroundAgent) -> bool:
    current_turn = background.current_turn
    return (current_turn is None or current_turn.done()) and background.inbox.empty()


_message_waiters: list[asyncio.Future[PeerMessage]] = []


def _notify_message(message: PeerMessage) -> None:
    waiters, _message_waiters[:] = list(_message_waiters), []
    for waiter in waiters:
        if not waiter.done():
            waiter.set_result(message)


_pending_probe: Callable[[], int] | None = None


def set_pending_message_probe(probe: Callable[[], int] | None) -> None:
    """Register how to count messages this process has received but not read.

    A foreground REPL queues them outside this module, and without a way to ask
    it, waiting for "a message" would block behind messages already delivered.
    """
    global _pending_probe
    _pending_probe = probe


def pending_message_count() -> int:
    peer = current_peer()
    if peer is not None:
        background = _background_agents.get(peer.id)
        if background is not None:
            return background.inbox.qsize()
    return _pending_probe() if _pending_probe is not None else 0


async def next_peer_message() -> PeerMessage:
    """Resolve when this process next receives a peer message.

    Only messages arriving after the call are seen — a caller that needs to
    avoid missing one already in flight should check its own inbox as well.
    """
    waiter: asyncio.Future[PeerMessage] = asyncio.get_running_loop().create_future()
    _message_waiters.append(waiter)
    return await waiter


def background_agent_state(agent_id: str) -> tuple[str, str | None]:
    """Return ``(state, error)`` for a background agent living in this process.

    ``state`` is ``running``, ``idle`` or ``unknown``; ``error`` carries the last
    failure, which survives the turn because the agent stays alive after one.
    """
    background = _background_agents.get(agent_id)
    if background is None:
        return "unknown", None
    if background.stopping:
        return "stopping", background.last_error
    if not _is_background_idle(background):
        return "running", background.last_error
    return "idle", background.last_error


def _notify_idle(background: _BackgroundAgent) -> None:
    if not _is_background_idle(background):
        return
    waiters = background.idle_waiters
    background.idle_waiters = []
    for waiter in waiters:
        if not waiter.done():
            waiter.set_result(None)


def _finish_pending_prompts(background: _BackgroundAgent) -> None:
    while True:
        try:
            queued = background.inbox.get_nowait()
        except asyncio.QueueEmpty:
            break
        if queued.done is not None and not queued.done.done():
            queued.done.set_result(None)
    _notify_idle(background)


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
        stop_handler: StopHandler | None = None,
        interrupt_handler: StopHandler | None = None,
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
        self._stop_handler = stop_handler
        self._interrupt_handler = interrupt_handler
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
            request_type = data.get("type")
            if request_type == "stop":
                await self._handle_control_request(data, writer, handler=self._stop_handler, unsupported="stop")
                return
            if request_type == "interrupt":
                await self._handle_control_request(
                    data,
                    writer,
                    handler=self._interrupt_handler,
                    unsupported="interrupt",
                )
                return
            if request_type != "message":
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
            # Single delivery point for every incoming message in this process,
            # so monitor() can block on one instead of polling for it.
            _notify_message(message)
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

    async def _handle_control_request(
        self,
        data: dict[str, Any],
        writer: asyncio.StreamWriter,
        *,
        handler: StopHandler | None,
        unsupported: str,
    ) -> None:
        if handler is None:
            await _write_response(writer, {"ok": False, "error": f"peer does not support {unsupported}"})
            return
        if str(data.get("to_id", "")) != self.id:
            await _write_response(writer, {"ok": False, "error": "wrong recipient"})
            return
        await handler(str(data.get("from_id", "")), str(data.get("reason", "")))
        await _write_response(writer, {"ok": True})


async def list_peers(all_projects: bool = False) -> str:
    """List other running axio agents. By default only peers in the current
    project are returned. Pass all_projects=true to inspect peers from every
    project on this host. This is a snapshot for discovery — to wait for an
    agent, call monitor() rather than calling this in a loop."""
    records = _visible_records(await list_peer_records(), all_projects=all_projects)
    if not records:
        if all_projects:
            return "No peers registered."
        return f"No peers registered for project: {_current_project()}"
    lines = ["Available peers:" if all_projects else f"Available peers for project {_current_project()}:"]
    for record in records:
        line = (
            f"- id={record.id} name={record.name!r} kind={record.kind} "
            f"project={record.project} pid={record.pid} cwd={record.cwd}"
        )
        # Spawned agents share the parent's pid, so liveness cannot be read from
        # it: a crashed one looks alive for as long as the parent runs. Their
        # real state is only known in-process.
        if record.id in _background_agents:
            state, error = background_agent_state(record.id)
            line += f" state={state}"
            if error:
                line += f" last_error={error!r}"
        lines.append(line)
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


async def _send_control_request(agent_id: str, request_type: str, reason: str) -> str:
    records = _visible_records(await list_peer_records(), all_projects=True)
    peer = _resolve_peer_by_id(records, agent_id)
    if peer is None:
        return f"No peer found for agent_id={agent_id!r}. Call list_peers first."

    sender = current_peer()
    if sender is not None and peer.id == sender.id:
        return f"Cannot {request_type} the current agent."

    payload = {
        "type": request_type,
        "id": uuid4().hex,
        "from_id": sender.id if sender is not None else f"unregistered-{os.getpid()}",
        "from_name": sender.name if sender is not None else f"unregistered-{os.getpid()}",
        "to_id": peer.id,
        "reason": reason,
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
        return f"Failed to send {request_type} to peer {peer.id}: {exc}"
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await writer.wait_closed()

    try:
        response = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"Peer {peer.id} returned an invalid response: {exc}"
    if response.get("ok") is not True:
        return f"Peer {peer.id} rejected {request_type}: {response.get('error', 'unknown error')}"
    return f"Sent {request_type} to {peer.name} ({peer.id})."


async def stop_agent(agent_id: StrictStr, reason: str = "") -> str:
    """Stop a running axio peer by global agent id. This permanently ends a
    spawned background agent; use interrupt_agent to cancel only the current
    response and keep the agent available."""
    return await _send_control_request(agent_id, "stop", reason)


async def interrupt_agent(agent_id: StrictStr, reason: str = "") -> str:
    """Interrupt the current response of a running axio peer by global agent id
    without stopping the agent."""
    return await _send_control_request(agent_id, "interrupt", reason)


async def _run_agent_turn(
    *,
    background: _BackgroundAgent,
    prompt: str,
) -> None:
    with peer_context(background.peer):
        async for event in background.agent.run_stream(prompt, background.context):
            if isinstance(event, Error):
                # A failed turn that did not raise - running out of iterations is
                # the common one. Unrecorded, it leaves the agent reporting the
                # same idle as one that answered, and the reason lives only in a
                # log line in whichever process happened to host it.
                background.last_error = str(event.exception)
            if _agent_event_handler is not None:
                await _agent_event_handler(background.peer.id, event)


async def _run_background_agent(background: _BackgroundAgent) -> None:
    try:
        while True:
            queued = await background.inbox.get()
            if queued.prompt is None or background.stopping:
                if queued.done is not None and not queued.done.done():
                    queued.done.set_result(None)
                break
            turn = asyncio.create_task(_run_agent_turn(background=background, prompt=queued.prompt))
            background.current_turn = turn
            background.last_error = None
            try:
                await turn
            except asyncio.CancelledError:
                if background.stopping:
                    break
            # Background agents must survive a failed turn so that the parent can
            # inspect, interrupt, stop, or send a recovery prompt.
            except Exception as exc:
                background.last_error = f"{type(exc).__name__}: {exc}"
                if _agent_event_handler is not None:
                    await _agent_event_handler(background.peer.id, Error(exc))
            finally:
                background.current_turn = None
                if queued.done is not None and not queued.done.done():
                    queued.done.set_result(None)
                _notify_idle(background)
            if background.stopping:
                break
    finally:
        _finish_pending_prompts(background)
        _background_agents.pop(background.peer.id, None)
        await background.peer.close()


async def _start_background_agent(
    *,
    agent: Agent,
    context: ContextStore,
    initial_task: str,
    name: str,
    project: str,
    cwd: str,
) -> _BackgroundAgent:
    accept_lock = asyncio.Lock()
    background: _BackgroundAgent

    async def _on_message(message: PeerMessage) -> None:
        async with accept_lock:
            if background.stopping:
                raise RuntimeError("agent is no longer accepting messages")
            background.inbox.put_nowait(_QueuedPrompt(format_message_for_dialog(message)))

    async def _on_stop(_from_id: str, _reason: str) -> None:
        async with accept_lock:
            background.stopping = True
            if background.current_turn is not None and not background.current_turn.done():
                background.current_turn.cancel()
            background.inbox.put_nowait(_QueuedPrompt(None))

    async def _on_interrupt(_from_id: str, _reason: str) -> None:
        async with accept_lock:
            if background.current_turn is not None and not background.current_turn.done():
                background.current_turn.cancel()

    peer = await PeerServer(
        name,
        kind="spawned-agent",
        handler=_on_message,
        stop_handler=_on_stop,
        interrupt_handler=_on_interrupt,
        project=project,
        cwd=cwd,
    ).start(set_current=False)
    background = _BackgroundAgent(agent=agent, context=context, peer=peer)
    _background_agents[peer.id] = background
    background.inbox.put_nowait(_QueuedPrompt(initial_task))
    background.runner = asyncio.create_task(_run_background_agent(background))
    return background


def local_background_agent_records() -> list[PeerRecord]:
    return [background.peer.record for background in _background_agents.values()]


def is_local_background_agent(agent_id: str) -> bool:
    return agent_id in _background_agents


async def wait_local_background_agents_idle(agent_ids: list[str] | None = None) -> None:
    selected = set(agent_ids) if agent_ids is not None else None
    while True:
        waiters: list[asyncio.Future[None]] = []
        for agent_id, background in list(_background_agents.items()):
            if selected is not None and agent_id not in selected:
                continue
            if _is_background_idle(background):
                continue
            waiter = asyncio.get_running_loop().create_future()
            background.idle_waiters.append(waiter)
            waiters.append(waiter)
        if not waiters:
            return
        await asyncio.gather(*waiters)


async def enqueue_local_agent_prompt(agent_id: str, prompt: str, *, wait: bool = False) -> bool:
    background = _background_agents.get(agent_id)
    if background is None or background.stopping:
        return False
    done: asyncio.Future[None] | None = None
    if wait:
        done = asyncio.get_running_loop().create_future()
    background.inbox.put_nowait(_QueuedPrompt(prompt, done))
    if done is not None:
        await done
    return True


async def stop_local_background_agents() -> None:
    backgrounds = list(_background_agents.values())
    for background in backgrounds:
        background.stopping = True
        if background.current_turn is not None and not background.current_turn.done():
            background.current_turn.cancel()
        background.inbox.put_nowait(_QueuedPrompt(None))
    for background in backgrounds:
        if background.runner is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await background.runner


async def spawn_agent(
    task: StrictStr,
    inherit_context: bool = False,
    name: str | None = None,
) -> str:
    """Spawn an independent background agent and return its global agent id. By
    default the spawned agent starts with an empty context. Set
    inherit_context=true only when the spawned agent must see the current
    conversation. The spawned agent remains available for IPC until it is
    explicitly stopped. This returns immediately and never carries the child's
    answer — wait for it with monitor(agents=[id]), which also tells you if the
    child died instead of finishing. Pass a short `name` describing the child's
    job — "docs-audit", "transport-review" — since you will be reading it back
    in list_peers and monitor; without one the child is named at random, which
    identifies it but tells you nothing."""
    if _spawn_agent_factory is None:
        return "spawn_agent is not configured"

    agent, context = await _spawn_agent_factory(inherit_context)
    parent = current_peer()
    project = parent.project if parent is not None else _current_project()
    cwd = parent.cwd if parent is not None else _normalize_project(None)
    base_name = name or generate_name()
    background = await _start_background_agent(
        agent=agent,
        context=context,
        initial_task=task,
        name=base_name,
        project=project,
        cwd=cwd,
    )
    return (
        f"Spawned background agent_id={background.peer.id} name={background.peer.name!r}. "
        "Use send_message, interrupt_agent, or stop_agent with this agent_id."
    )


spawn_agent._tool_concurrency = 3  # type: ignore[attr-defined]
