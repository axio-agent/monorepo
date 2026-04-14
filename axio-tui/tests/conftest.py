"""Shared test fixtures for axio-tui."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import aiosqlite
import pytest
from axio_context_sqlite import connect


@pytest.fixture
async def db_conn(tmp_path: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Open a fresh context database connection for the test."""
    conn = await connect(tmp_path / "test.db")
    yield conn
    await conn.close()
