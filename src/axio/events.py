"""Stream events: all variants emitted by AgentStream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axio.types import StopReason, ToolCallID, ToolName, Usage


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class ToolUseStart:
    index: int
    tool_use_id: ToolCallID
    name: ToolName


@dataclass(frozen=True, slots=True)
class ToolInputDelta:
    index: int
    tool_use_id: ToolCallID
    partial_json: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_use_id: ToolCallID
    name: ToolName
    is_error: bool
    content: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IterationEnd:
    iteration: int
    stop_reason: StopReason
    usage: Usage


@dataclass(frozen=True, slots=True)
class Error:
    exception: BaseException


@dataclass(frozen=True, slots=True)
class SessionEndEvent:
    stop_reason: StopReason
    total_usage: Usage


type StreamEvent = (
    ReasoningDelta | TextDelta | ToolUseStart | ToolInputDelta | ToolResult | IterationEnd | Error | SessionEndEvent
)
