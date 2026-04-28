"""Pytest configuration for axio-tools-docker."""

from __future__ import annotations

import asyncio
import os

import aiodocker
import pytest


@pytest.fixture(scope="session")
def docker() -> str:
    docker_url = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")

    async def _probe() -> bool:
        # noinspection PyBroadException
        try:
            async with aiodocker.Docker(url=docker_url) as client:
                await client.system.info()
            return True
        except Exception:
            return False

    if not asyncio.run(_probe()):
        pytest.skip("Skipping due to missing Docker daemon")

    return docker_url
