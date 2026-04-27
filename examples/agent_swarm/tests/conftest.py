"""Test configuration for agent_swarm example tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from axio.tool import Tool, ToolHandler


class _NoOp(ToolHandler[None]):
    """No-op stub handler used to build a test toolbox."""

    async def __call__(self, context: None) -> str:  # type: ignore[override]
        return ""


_STUB_NAMES = ["read_file", "write_file", "patch_file", "list_files", "shell", "run_python"]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace directory for tests."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


@pytest.fixture
def stub_toolbox() -> dict[str, Tool[Any]]:
    """Stub toolbox matching the tool names used by specialist TOML roles."""
    return {n: Tool(name=n, description=f"stub {n}", handler=_NoOp) for n in _STUB_NAMES}


@pytest.fixture
def swarm_config() -> dict[str, Any]:
    """Provide a default swarm configuration for testing."""
    return {
        "roles": ["architect", "developer", "reviewer"],
        "max_iterations": 10,
    }
