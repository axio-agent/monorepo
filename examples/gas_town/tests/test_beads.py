"""Tests for gas_town.beads module."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
import pytest
from axio import CONTEXT

from gas_town.beads import (
    DDL,
    bead,
    bead_summary,
    get_bead,
    mark_in_progress,
)


@pytest.fixture
async def db_connection(tmp_path) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Create an in-memory SQLite connection with schema."""
    db_path = tmp_path / "test_beads.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute(DDL)
        await db.commit()
        yield db


@pytest.fixture
async def db_path(tmp_path) -> str:
    """Return the path to the test database for multi-connection tests."""
    db_path = tmp_path / "test_beads2.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute(DDL)
        await db.commit()
    return str(db_path)


async def call_bead(db: aiosqlite.Connection, **kwargs: Any) -> str:
    """Call bead() with db set as CONTEXT."""
    token = CONTEXT.set(db)
    try:
        return await bead(**kwargs)
    finally:
        CONTEXT.reset(token)


class TestBeadSummary:
    """Tests for the bead_summary function."""

    async def test_empty_database_returns_no_beads(self, db_connection) -> None:
        result = await bead_summary(db_connection)
        assert result == "(no beads)"

    async def test_single_bead_formatted_correctly(self, db_connection) -> None:
        await db_connection.execute(
            "INSERT INTO beads (title, status) VALUES (?, ?)",
            ("Test bead", "open"),
        )
        await db_connection.commit()

        result = await bead_summary(db_connection)

        assert "○" in result  # open icon
        assert "Test bead" in result
        assert "(open)" in result

    async def test_multiple_beads_sorted_by_id(self, db_connection) -> None:
        await db_connection.execute(
            "INSERT INTO beads (title, status) VALUES (?, ?)",
            ("Bead A", "closed"),
        )
        await db_connection.execute(
            "INSERT INTO beads (title, status) VALUES (?, ?)",
            ("Bead B", "in_progress"),
        )
        await db_connection.commit()

        result = await bead_summary(db_connection)

        lines = result.split("\n")
        assert len(lines) == 2
        assert "Bead A" in lines[0]
        assert "Bead B" in lines[1]

    async def test_bead_with_assignee_shown(self, db_connection) -> None:
        await db_connection.execute(
            "INSERT INTO beads (title, status, assignee) VALUES (?, ?, ?)",
            ("Assigned bead", "in_progress", "polecat#1"),
        )
        await db_connection.commit()

        result = await bead_summary(db_connection)

        assert "→ polecat#1" in result


class TestGetBead:
    async def test_returns_none_for_missing_bead(self, db_connection) -> None:
        result = await get_bead(db_connection, 999)
        assert result is None

    async def test_returns_all_fields(self, db_connection) -> None:
        await db_connection.execute(
            "INSERT INTO beads (title, status, assignee, notes) VALUES (?, ?, ?, ?)",
            ("Test", "open", "worker", "Some notes"),
        )
        await db_connection.commit()
        # SQLite uses 1-based IDs
        result = await get_bead(db_connection, 1)

        assert result is not None
        bead_id, title, status, assignee, notes = result
        assert bead_id == 1
        assert title == "Test"
        assert status == "open"
        assert assignee == "worker"
        assert notes == "Some notes"


class TestMarkInProgress:
    async def test_updates_status_and_assignee(self, db_connection) -> None:
        await db_connection.execute(
            "INSERT INTO beads (title, status) VALUES (?, ?)",
            ("Test", "open"),
        )
        await db_connection.commit()
        bead_id = 1

        await mark_in_progress(db_connection, bead_id, "test_worker")

        row = await get_bead(db_connection, bead_id)
        assert row is not None
        _, _, status, assignee, _ = row
        assert status == "in_progress"
        assert assignee == "test_worker"

    async def test_commits_transaction(self, db_path) -> None:
        """Verify that mark_in_progress commits changes so they are visible to other connections."""
        # First connection: set up initial data
        async with aiosqlite.connect(db_path) as db1:
            await db1.execute(
                "INSERT INTO beads (title, status) VALUES (?, ?)",
                ("Test", "open"),
            )
            await db1.commit()

        # Second connection: mark in progress with another connection
        async with aiosqlite.connect(db_path) as db2:
            await mark_in_progress(db2, 1, "worker")

        # Third connection: verify commit was done (another connection can read)
        async with aiosqlite.connect(db_path) as db3:
            row = await get_bead(db3, 1)
            assert row is not None
            _, _, status, assignee, _ = row
            assert status == "in_progress"
            assert assignee == "worker"


class TestBeadToolCreate:
    async def test_create_returns_bead_id(self, db_connection) -> None:
        result = await call_bead(db_connection, action="create", title="My New Bead")
        assert "Created bead [1]" in result
        assert "My New Bead" in result

    async def test_create_with_empty_title(self, db_connection) -> None:
        result = await call_bead(db_connection, action="create", title="")
        assert "Created bead" in result


class TestBeadToolList:
    async def test_list_empty(self, db_connection) -> None:
        result = await call_bead(db_connection, action="list")
        assert "(no beads)" in result

    async def test_list_with_beads(self, db_connection) -> None:
        await db_connection.execute(
            "INSERT INTO beads (title, status) VALUES (?, ?)",
            ("Test Bead", "open"),
        )
        await db_connection.commit()

        result = await call_bead(db_connection, action="list")
        assert "Test Bead" in result


class TestBeadToolUpdate:
    async def test_update_modifies_status(self, db_connection) -> None:
        await call_bead(db_connection, action="create", title="To Update")
        result = await call_bead(db_connection, action="update", id=1, status="closed")
        assert "[1] updated" in result
        assert "status=closed" in result

    async def test_update_modifies_assignee(self, db_connection) -> None:
        await call_bead(db_connection, action="create", title="Assign Test")
        result = await call_bead(db_connection, action="update", id=1, assignee="worker1")
        assert "assignee=worker1" in result

    async def test_update_preserves_existing_values(self, db_connection) -> None:
        await call_bead(db_connection, action="create", title="Preserve Test")
        result = await call_bead(db_connection, action="update", id=1, assignee="new_worker")
        assert "status=open" in result

    async def test_update_nonexistent_bead_returns_not_found(self, db_connection) -> None:
        result = await call_bead(db_connection, action="update", id=999, status="closed")
        assert "not found" in result


class TestBeadToolClose:
    async def test_close_sets_status_closed(self, db_connection) -> None:
        await call_bead(db_connection, action="create", title="To Close")
        result = await call_bead(db_connection, action="close", id=1)
        assert "[1] closed" in result

    async def test_close_nonexistent_bead_returns_not_found(self, db_connection) -> None:
        result = await call_bead(db_connection, action="close", id=999)
        assert "not found" in result


class TestBeadToolNote:
    async def test_note_appends_to_existing_notes(self, db_connection) -> None:
        await db_connection.execute(
            "INSERT INTO beads (title, notes) VALUES (?, ?)",
            ("Has notes", "Initial note\n"),
        )
        await db_connection.commit()

        result = await call_bead(db_connection, action="note", id=1, notes="Appended note")
        assert "[1] note appended" in result

        row = await get_bead(db_connection, 1)
        assert row is not None
        _, _, _, _, notes = row
        assert "Initial note" in notes
        assert "Appended note" in notes

    async def test_note_with_empty_text(self, db_connection) -> None:
        await call_bead(db_connection, action="create", title="Test")
        result = await call_bead(db_connection, action="note", id=1, notes="")
        assert "[1] note appended" in result

    async def test_note_nonexistent_bead_returns_not_found(self, db_connection) -> None:
        result = await call_bead(db_connection, action="note", id=999, notes="test")
        assert "not found" in result
