"""MCPServerConfig: frozen dataclass for MCP server connection parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Configuration for connecting to an MCP server.

    Exactly one of ``command`` (stdio transport) or ``url`` (HTTP transport) must be set.
    """

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    scope: str = "global"

    def __post_init__(self) -> None:
        has_command = self.command is not None
        has_url = self.url is not None
        if has_command == has_url:
            raise ValueError("Exactly one of 'command' or 'url' must be set")

    def to_dict(self) -> dict[str, str]:
        """Serialize to flat string dict for config DB persistence."""
        result: dict[str, str] = {}
        if self.command is not None:
            result["command"] = self.command
        if self.args:
            result["args"] = json.dumps(self.args)
        if self.env:
            result["env"] = json.dumps(self.env)
        if self.url is not None:
            result["url"] = self.url
        if self.headers:
            result["headers"] = json.dumps(self.headers)
        if self.timeout != 30.0:
            result["timeout"] = str(self.timeout)
        return result

    @classmethod
    def from_dict(cls, name: str, data: dict[str, str]) -> MCPServerConfig:
        """Deserialize from flat string dict."""
        args: list[str] = json.loads(data["args"]) if "args" in data else []
        env: dict[str, str] | None = json.loads(data["env"]) if "env" in data else None
        headers: dict[str, str] = json.loads(data["headers"]) if "headers" in data else {}
        timeout = float(data["timeout"]) if "timeout" in data else 30.0
        return cls(
            name=name,
            command=data.get("command"),
            args=args,
            env=env,
            url=data.get("url"),
            headers=headers,
            timeout=timeout,
        )
