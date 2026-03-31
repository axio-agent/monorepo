"""SandboxConfig: frozen dataclass for Docker sandbox parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Configuration for a Docker sandbox container."""

    image: str = "python:latest"
    memory: str = "256m"
    cpus: str = "1.0"
    network: bool = False
    workdir: str = "/workspace"

    def to_dict(self) -> dict[str, str]:
        """Serialize to flat string dict for config DB persistence.

        Only non-default values are included.
        """
        defaults = SandboxConfig()
        result: dict[str, str] = {}
        if self.image != defaults.image:
            result["image"] = self.image
        if self.memory != defaults.memory:
            result["memory"] = self.memory
        if self.cpus != defaults.cpus:
            result["cpus"] = self.cpus
        if self.network != defaults.network:
            result["network"] = str(self.network)
        if self.workdir != defaults.workdir:
            result["workdir"] = self.workdir
        return result

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SandboxConfig:
        """Deserialize from flat string dict."""
        return cls(
            image=data.get("image", "python:latest"),
            memory=data.get("memory", "256m"),
            cpus=data.get("cpus", "1.0"),
            network=data["network"].lower() in ("true", "1", "yes") if "network" in data else False,
            workdir=data.get("workdir", "/workspace"),
        )
