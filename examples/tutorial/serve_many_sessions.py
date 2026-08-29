"""Serve independent Axio sessions and verify the registry offline."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axio import (
    Agent,
    CONTEXT,
    ContextStore,
    MemoryContextStore,
    Message,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    Tool,
    Usage,
)
from axio.compaction import AutoCompactStore
from axio.events import SessionEndEvent
from axio.testing import StubTransport, make_text_response
from axio_context_sqlite import SQLiteContextStore, connect
from axio_tools_docker import DockerSandbox


TOOL_OUTPUT_MAX_CHARS = 4_000
TRUNCATION_MARKER = "\n[tool output truncated]"


def bound_tool_output(text: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    if max_chars < len(TRUNCATION_MARKER):
        raise ValueError("max_chars is too small for the truncation marker")
    if len(text) <= max_chars:
        return text
    prefix_length = max_chars - len(TRUNCATION_MARKER)
    return text[:prefix_length] + TRUNCATION_MARKER


def bound_text_tool(
    tool: Tool[Any],
    max_chars: int = TOOL_OUTPUT_MAX_CHARS,
) -> Tool[Any]:
    async def bounded_handler(**kwargs: Any) -> Any:
        result = await tool(**kwargs)
        if isinstance(result, str):
            return bound_tool_output(result, max_chars)
        return result

    return Tool(
        name=tool.name,
        handler=bounded_handler,
        description=tool.description,
        schema=tool.schema,
    )


# [docs:start-serve-session-state]
type ContextFactory = Callable[[str], ContextStore]
type SandboxFactory = Callable[[str], DockerSandbox]


@dataclass(slots=True)
class CloudSession:
    agent: Agent
    context: ContextStore
    resources: AsyncExitStack
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# [docs:end-serve-session-state]


class CloudHarness:
    def __init__(
        self,
        prototype_agent: Agent,
        context_factory: ContextFactory,
        sandbox_factory: SandboxFactory,
    ) -> None:
        self._prototype_agent = prototype_agent
        self._context_factory = context_factory
        self._sandbox_factory = sandbox_factory
        self._sessions: dict[str, asyncio.Task[CloudSession]] = {}
        self._registry_lock = asyncio.Lock()
        self._closed = False

    # [docs:start-serve-session-create]
    @staticmethod
    def _failed(opening: asyncio.Task[CloudSession]) -> bool:
        """Whether this open finished without producing a session, which is when to forget it.

        Cancelled counts. Kept, a cancelled open hands `CancelledError` to every later turn for
        that ID for the life of the harness.
        """
        if not opening.done():
            return False
        return opening.cancelled() or opening.exception() is not None

    async def _session(self, session_id: str) -> CloudSession:
        async with self._registry_lock:
            if self._closed:
                raise RuntimeError("harness is closed")

            opening = self._sessions.get(session_id)
            if opening is not None and self._failed(opening):
                # A session that would not open is not a session. Kept, its failure is handed to
                # every later turn for that ID; dropped here, the next request tries again.
                opening = None
            if opening is None:
                opening = asyncio.create_task(self._open(session_id))
                self._sessions[session_id] = opening

        # Started under the lock and awaited outside it. Held while the container starts, one cold
        # session made every other session's first turn wait behind its image pull. `shield` keeps
        # the open running when this caller gives up: it is already creating a container, and a
        # caller walking away is not a reason to lose track of it.
        return await asyncio.shield(opening)

    async def _open(self, session_id: str) -> CloudSession:
        resources = AsyncExitStack()
        try:
            context = self._context_factory(session_id)
            resources.push_async_callback(context.close)
            sandbox = await resources.enter_async_context(
                self._sandbox_factory(session_id)
            )
            return CloudSession(
                agent=self._prototype_agent.copy(
                    tools=[
                        *self._prototype_agent.tools,
                        *(bound_text_tool(tool) for tool in sandbox.tools),
                    ],
                ),
                context=context,
                resources=resources,
            )
        except BaseException:
            await resources.aclose()
            raise

    # [docs:end-serve-session-create]

    # [docs:start-serve-turn-lifecycle]
    async def stream_turn(
        self,
        session_id: str,
        prompt: str,
    ) -> AsyncIterator[StreamEvent]:
        session = await self._session(session_id)
        async with session.turn_lock:
            stream = session.agent.run_stream(prompt, session.context)
            try:
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()

    async def close(self) -> None:
        async with self._registry_lock:
            self._closed = True
            opening = list(self._sessions.values())
            self._sessions.clear()

        # A session still opening owns a container already. Wait for it before closing, or the
        # resources it is in the middle of taking outlive the harness that asked for them.
        opened = await asyncio.gather(*opening, return_exceptions=True)
        await asyncio.gather(
            *(
                session.resources.aclose()
                for session in opened
                if isinstance(session, CloudSession)
            )
        )

    # [docs:end-serve-turn-lifecycle]


# [docs:start-serve-sandbox-factory]
def docker_for_session(session_id: str) -> DockerSandbox:
    return DockerSandbox(
        image="python:3.12-slim",
        name=session_id,
        remove=False,
        memory="512m",
        cpus="1.0",
        network=False,
        read_only=True,
        cap_drop=["ALL"],
        ulimits={"nofile": (256, 256), "nproc": 128},
        tmpfs={"/tmp": "size=64m,mode=1777"},
        named_volumes={
            "/workspace": f"{session_id}-workspace",
        },
        volumes_remove=False,
        workdir="/workspace",
    )


# [docs:end-serve-sandbox-factory]


# [docs:start-serve-application-lifecycle]
async def serve_application(
    database_path: Path,
    prototype_agent: Agent,
    transport,
    sandbox_factory: SandboxFactory,
    serve,
) -> None:
    async with AsyncExitStack() as resources:
        connection = await connect(database_path)
        resources.push_async_callback(connection.close)

        def context_for_session(session_id: str) -> ContextStore:
            base_context = SQLiteContextStore(
                connection,
                session_id=session_id,
                project="public-repository",
            )
            return AutoCompactStore(
                base_context,
                transport,
                keep_recent=6,
                threshold=0.75,
            )

        harness = CloudHarness(
            prototype_agent=prototype_agent,
            context_factory=context_for_session,
            sandbox_factory=sandbox_factory,
        )
        resources.push_async_callback(harness.close)
        await serve(harness)


# [docs:end-serve-application-lifecycle]


# [docs:start-serve-sqlite-recovery]
async def verify_sqlite_recovery(database: Path) -> None:
    session_id = "4f0f29b8a5d94f688306c231d86aa531"

    first_connection = await connect(database)
    try:
        first = SQLiteContextStore(
            first_connection,
            session_id=session_id,
            project="public-repository",
        )
        await first.append(
            Message(
                role="user",
                content=[TextBlock(text="Review README.md")],
            )
        )
    finally:
        await first_connection.close()

    second_connection = await connect(database)
    try:
        recovered = SQLiteContextStore(
            second_connection,
            session_id=session_id,
            project="public-repository",
        )
        history = await recovered.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == [TextBlock(text="Review README.md")]
    finally:
        await second_connection.close()


# [docs:end-serve-sqlite-recovery]


class _TrackingContext(MemoryContextStore):
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._session_id = session_id
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def _sandbox_shell() -> str:
    sandbox = CONTEXT.get()
    sandbox.tool_calls += 1
    return sandbox.session_id


class _FakeSandbox:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.tools = (
            Tool(
                name=f"shell-{session_id}",
                handler=_sandbox_shell,
                context=self,
            ),
        )
        self.closed = False
        self.tool_calls = 0

    async def __aenter__(self) -> _FakeSandbox:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True


@dataclass(slots=True)
class _ProbeState:
    active_by_session: dict[str, int] = field(default_factory=dict)
    max_active_by_session: dict[str, int] = field(default_factory=dict)
    active_total: int = 0
    max_active_total: int = 0
    hold_started: asyncio.Event = field(default_factory=asyncio.Event)
    hold_release: asyncio.Event = field(default_factory=asyncio.Event)
    parallel_started: asyncio.Event = field(default_factory=asyncio.Event)
    parallel_release: asyncio.Event = field(default_factory=asyncio.Event)


class _ProbeAgent:
    def __init__(
        self,
        state: _ProbeState,
        tools: list[Tool[Any]] | None = None,
    ) -> None:
        self.state = state
        self.tools = tools or []

    def copy(self, *, tools: list[Tool[Any]]) -> _ProbeAgent:
        return _ProbeAgent(self.state, tools)

    def run_stream(
        self,
        prompt: str,
        context: ContextStore,
    ) -> AsyncIterator[StreamEvent]:
        async def events() -> AsyncIterator[StreamEvent]:
            session_id = context.session_id
            active = self.state.active_by_session.get(session_id, 0) + 1
            self.state.active_by_session[session_id] = active
            self.state.max_active_by_session[session_id] = max(
                active,
                self.state.max_active_by_session.get(session_id, 0),
            )
            self.state.active_total += 1
            self.state.max_active_total = max(
                self.state.max_active_total,
                self.state.active_total,
            )

            try:
                if prompt == "hold":
                    self.state.hold_started.set()
                    await self.state.hold_release.wait()
                elif prompt.startswith("parallel"):
                    if self.state.active_total >= 2:
                        self.state.parallel_started.set()
                    await self.state.parallel_release.wait()

                yield TextDelta(index=0, delta=prompt)
                yield SessionEndEvent(
                    stop_reason=StopReason.end_turn,
                    total_usage=Usage(input_tokens=0, output_tokens=0),
                )
            finally:
                self.state.active_by_session[session_id] -= 1
                self.state.active_total -= 1

        return events()


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


async def verify_registry() -> None:
    contexts: dict[str, _TrackingContext] = {}
    sandboxes: dict[str, _FakeSandbox] = {}

    def context_for_test(session_id: str) -> _TrackingContext:
        context = _TrackingContext(session_id)
        contexts[session_id] = context
        return context

    def sandbox_for_test(session_id: str) -> _FakeSandbox:
        sandbox = _FakeSandbox(session_id)
        sandboxes[session_id] = sandbox
        return sandbox

    async def read_document(path: str) -> str:
        return path

    state = _ProbeState()
    harness = CloudHarness(
        prototype_agent=_ProbeAgent(
            state,
            [Tool(name="read_document", handler=read_document)],
        ),  # type: ignore[arg-type]
        context_factory=context_for_test,
        sandbox_factory=sandbox_for_test,  # type: ignore[arg-type]
    )

    first = asyncio.create_task(_collect(harness.stream_turn("same", "hold")))
    await asyncio.wait_for(state.hold_started.wait(), timeout=1)
    second = asyncio.create_task(_collect(harness.stream_turn("same", "queued")))
    await asyncio.sleep(0)
    assert not second.done()
    state.hold_release.set()
    await asyncio.gather(first, second)
    assert state.max_active_by_session["same"] == 1

    alice = asyncio.create_task(_collect(harness.stream_turn("alice", "parallel-a")))
    bob = asyncio.create_task(_collect(harness.stream_turn("bob", "parallel-b")))
    await asyncio.wait_for(state.parallel_started.wait(), timeout=1)
    state.parallel_release.set()
    await asyncio.gather(alice, bob)

    assert state.max_active_total == 2
    assert contexts["alice"] is not contexts["bob"]
    assert sandboxes["alice"] is not sandboxes["bob"]

    # The registry holds the task that opens a session, so a cold turn does not wait behind
    # another session's container start.
    alice_session = await harness._sessions["alice"]
    bob_session = await harness._sessions["bob"]
    assert alice_session is not bob_session
    assert alice_session.agent is not bob_session.agent
    assert [tool.name for tool in alice_session.agent.tools] == [
        "read_document",
        "shell-alice",
    ]
    assert [tool.name for tool in bob_session.agent.tools] == [
        "read_document",
        "shell-bob",
    ]
    assert await alice_session.agent.tools[-1]() == "alice"
    assert await bob_session.agent.tools[-1]() == "bob"
    assert sandboxes["alice"].tool_calls == 1
    assert sandboxes["bob"].tool_calls == 1

    await harness.close()
    assert harness._sessions == {}
    assert all(context.closed for context in contexts.values())
    assert all(sandbox.closed for sandbox in sandboxes.values())


async def verify_service_lifecycle(database: Path) -> None:
    state = _ProbeState()
    sandboxes: dict[str, _FakeSandbox] = {}

    def sandbox_factory(session_id: str) -> _FakeSandbox:
        sandbox = _FakeSandbox(session_id)
        sandboxes[session_id] = sandbox
        return sandbox

    async def serve_once(harness: CloudHarness) -> None:
        events = await _collect(harness.stream_turn("service-session", "ready"))
        assert any(isinstance(event, SessionEndEvent) for event in events)

    await serve_application(
        database_path=database,
        prototype_agent=_ProbeAgent(state),  # type: ignore[arg-type]
        transport=StubTransport([make_text_response("summary")]),
        sandbox_factory=sandbox_factory,  # type: ignore[arg-type]
        serve=serve_once,
    )
    assert sandboxes["service-session"].closed


async def verify_example() -> None:
    candidate = docker_for_session("4f0f29b8a5d94f688306c231d86aa531")
    assert candidate.name == "4f0f29b8a5d94f688306c231d86aa531"
    assert candidate.named_volumes["/workspace"].endswith("-workspace")

    data_directory = Path(os.environ.get("AXIO_TUTORIAL_DATA_DIR", "."))
    await verify_registry()
    await verify_service_lifecycle(data_directory / "service.db")
    await verify_sqlite_recovery(data_directory / "recovery.db")


if __name__ == "__main__":
    asyncio.run(verify_example())
