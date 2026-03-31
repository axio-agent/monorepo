"""Primitive types: ToolName, ToolCallID, StopReason, Usage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

type ToolName = str
type ToolCallID = str


class StopReason(StrEnum):
    end_turn = "end_turn"
    tool_use = "tool_use"
    max_tokens = "max_tokens"
    error = "error"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )
