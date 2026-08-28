"""Adapt Axio events to a small JSON protocol without a web server."""

from __future__ import annotations

import asyncio
import base64
import re
from contextlib import aclosing
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from axio import StopReason, StreamEvent, Usage
from axio.events import Error, ImageOutput, SessionEndEvent


# [docs:start-adapt-event-adapter]
def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseException):
        return {
            "code": "internal_error",
            "message": "The agent turn failed.",
        }
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: json_value(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def event_to_dict(event: StreamEvent) -> dict[str, Any]:
    return {
        "version": 1,
        "type": type(event).__name__,
        "data": json_value(event),
    }


# [docs:end-adapt-event-adapter]


# [docs:start-adapt-stream-endpoint]
@dataclass(frozen=True, slots=True)
class AuthenticatedIdentity:
    subject: str
    session_id: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", self.session_id) is None:
            raise ValueError("session_id must be a server-issued UUID hex value")


async def handle_prompt(websocket, harness, identity, prompt):
    stream = harness.stream_turn(identity.session_id, prompt)
    async with aclosing(stream):
        async for event in stream:
            await websocket.send_json(event_to_dict(event))


# [docs:end-adapt-stream-endpoint]


class _FakeHarness:
    def __init__(self) -> None:
        self.selected_sessions: list[str] = []
        self.closed_streams = 0

    def stream_turn(self, session_id: str, prompt: str):
        self.selected_sessions.append(session_id)

        async def events():
            try:
                yield SessionEndEvent(
                    stop_reason=StopReason.end_turn,
                    total_usage=Usage(input_tokens=len(prompt), output_tokens=0),
                )
            finally:
                self.closed_streams += 1

        return events()


class _FakeWebSocket:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[dict[str, Any]] = []

    async def send_json(self, message: dict[str, Any]) -> None:
        if self.fail:
            raise ConnectionError("client disconnected")
        self.messages.append(message)


async def verify_example() -> None:
    end = event_to_dict(
        SessionEndEvent(
            stop_reason=StopReason.end_turn,
            total_usage=Usage(input_tokens=10, output_tokens=4),
        )
    )
    assert end == {
        "version": 1,
        "type": "SessionEndEvent",
        "data": {
            "stop_reason": "end_turn",
            "total_usage": {"input_tokens": 10, "output_tokens": 4},
        },
    }

    failure = event_to_dict(Error(ValueError("provider secret")))
    assert failure["data"]["exception"] == {
        "code": "internal_error",
        "message": "The agent turn failed.",
    }

    image = event_to_dict(ImageOutput(0, b"PNG", "image/png"))
    assert image["data"]["data"] == {
        "encoding": "base64",
        "data": "UE5H",
    }

    harness = _FakeHarness()
    websocket = _FakeWebSocket()
    identity = AuthenticatedIdentity(
        subject="alice",
        session_id="4f0f29b8a5d94f688306c231d86aa531",
    )
    await handle_prompt(websocket, harness, identity, "Read README.md")
    assert harness.selected_sessions == [identity.session_id]
    assert websocket.messages[0]["type"] == "SessionEndEvent"
    assert harness.closed_streams == 1

    try:
        await handle_prompt(
            _FakeWebSocket(fail=True),
            harness,
            identity,
            "Disconnect",
        )
    except ConnectionError:
        pass
    else:
        raise AssertionError("disconnect must propagate")

    assert harness.closed_streams == 2

    try:
        AuthenticatedIdentity(subject="alice", session_id="session:alice")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe session IDs must be rejected")


if __name__ == "__main__":
    asyncio.run(verify_example())
