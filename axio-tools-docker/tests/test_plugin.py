"""Tests for DockerPlugin."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from axio_tools_docker.config import SandboxConfig
from axio_tools_docker.plugin import DockerPlugin


async def test_init_without_config() -> None:
    plugin = DockerPlugin()
    await plugin.init()
    assert len(plugin.all_tools) == 3


async def test_init_with_config_db() -> None:
    mock_db = MagicMock()
    mock_db.get_prefix = AsyncMock(
        return_value={
            "docker.image": "ubuntu:22.04",
            "docker.memory": "512m",
        },
    )

    plugin = DockerPlugin()
    await plugin.init(config=mock_db)

    assert plugin._manager.config.image == "ubuntu:22.04"
    assert plugin._manager.config.memory == "512m"
    mock_db.get_prefix.assert_awaited_once_with("docker.")


async def test_init_project_takes_priority() -> None:
    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(return_value={"docker.image": "node:20"})

    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(return_value={"docker.image": "python:3.11"})

    plugin = DockerPlugin()
    await plugin.init(config=project_db, global_config=global_db)

    assert plugin._manager.config.image == "node:20"


async def test_init_falls_back_to_global() -> None:
    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(return_value={})

    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(return_value={"docker.image": "python:3.11"})

    plugin = DockerPlugin()
    await plugin.init(config=project_db, global_config=global_db)

    assert plugin._manager.config.image == "python:3.11"


async def test_all_tools_returns_three() -> None:
    plugin = DockerPlugin()
    await plugin.init()

    tools = plugin.all_tools
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"sandbox_exec", "sandbox_write", "sandbox_read"}


async def test_close_delegates_to_manager() -> None:
    plugin = DockerPlugin()
    plugin._manager.close = AsyncMock()  # type: ignore[method-assign]
    await plugin.close()
    plugin._manager.close.assert_awaited_once()


def test_label() -> None:
    plugin = DockerPlugin()
    assert plugin.label == "Docker Sandbox"


async def test_default_config_when_no_db() -> None:
    plugin = DockerPlugin()
    await plugin.init()
    assert plugin._manager.config == SandboxConfig()
