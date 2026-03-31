"""Message: the fundamental unit of conversation history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from axio.blocks import ContentBlock, from_dict, to_dict


@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": [to_dict(b) for b in self.content]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(role=data["role"], content=[from_dict(b) for b in data["content"]])
