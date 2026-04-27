"""Pytest configuration for docs tests."""

from __future__ import annotations

import asyncio

import pytest


def _check_docker() -> bool:
    """Return True if a Docker daemon is reachable."""
    try:
        import aiodocker
    except ImportError:
        return False

    async def _probe() -> bool:
        try:
            async with aiodocker.Docker() as client:
                await client.system.info()
            return True
        except Exception:
            return False

    return asyncio.run(_probe())


DOCKER_AVAILABLE: bool = _check_docker()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "docker: mark test as requiring a running Docker daemon",
    )


def pytest_collection_modifyitems(
    items: list[pytest.Item],
    config: pytest.Config,
) -> None:
    if DOCKER_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="Docker daemon not available")
    for item in items:
        if item.get_closest_marker("docker"):
            item.add_marker(skip)
