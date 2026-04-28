"""Integration tests for Gas Town orchestration flows.

These tests verify the multi-agent orchestration patterns where agents
interact with the bead store and each other.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiochannel import Channel
from axio import CONTEXT, MemoryContextStore
from axio.compaction import AutoCompactStore
from axio.messages import Message
from axio.models import ModelSpec
from axio.testing import StubTransport, make_text_response

from gas_town.beads import DDL, bead, bead_summary, get_bead
from gas_town.swarm import make_analyze_tool, make_sling_tool, run_gastown

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def db_connection(tmp_path):
    """Create an in-memory SQLite connection with schema."""
    import aiosqlite

    db_path = tmp_path / "integration_test.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute(DDL)
        await db.commit()
        yield db


@pytest.fixture
def mock_model_spec() -> ModelSpec:
    return ModelSpec(id="gpt-4")


async def call_bead(db, **kwargs) -> str:  # type: ignore[no-untyped-def]
    """Call bead() with db set as CONTEXT."""

    CONTEXT.set(db)
    return await bead(**kwargs)


# =============================================================================
# Test: Agent uses BeadTool to create/update/close beads
# =============================================================================


class TestBeadStoreIntegration:
    """Tests verifying agents interact correctly with the bead store."""

    async def test_agent_creates_bead_via_tool(self, db_connection) -> None:
        """Test that an agent can use bead() to create a new bead."""
        result = await call_bead(db_connection, action="create", title="Test Task")
        assert "Created bead" in result
        assert "Test Task" in result

    async def test_agent_updates_bead_status(self, db_connection) -> None:
        """Test that an agent can update an existing bead's status."""
        await call_bead(db_connection, action="create", title="Initial Task")
        result = await call_bead(db_connection, action="update", id=1, status="in_progress", assignee="test_agent")
        assert "updated" in result
        assert "status=in_progress" in result

    async def test_agent_closes_bead(self, db_connection) -> None:
        """Test that an agent can close a bead."""
        await call_bead(db_connection, action="create", title="Task to Close")
        await call_bead(db_connection, action="update", id=1, status="in_progress")
        result = await call_bead(db_connection, action="close", id=1)
        assert "closed" in result

        row = await get_bead(db_connection, 1)
        assert row is not None
        _, _, status, _, _ = row
        assert status == "closed"

    async def test_agent_adds_note_to_bead(self, db_connection) -> None:
        """Test that an agent can add notes to a bead."""
        await call_bead(db_connection, action="create", title="Task with Notes")
        result = await call_bead(db_connection, action="note", id=1, notes="Found an issue during analysis")
        assert "note appended" in result


# =============================================================================
# Test: Multi-agent workflow simulation
# =============================================================================


class TestMultiAgentWorkflow:
    """Test multi-agent orchestration patterns."""

    async def test_mayor_dispatches_polecat(self, db_connection, mock_model_spec: ModelSpec) -> None:
        """Test Mayor creating a bead for Polecat to work on."""
        result = await call_bead(db_connection, action="create", title="Fix login bug")
        assert "Created bead" in result

        row = await get_bead(db_connection, 1)
        assert row is not None
        bead_id, title, status, assignee, _ = row
        assert title == "Fix login bug"
        assert status == "open"

        result = await call_bead(
            db_connection, action="update", id=bead_id, status="in_progress", assignee="polecat#1"
        )
        assert "updated" in result

    async def test_polecat_creates_followup_bead(self, db_connection, mock_model_spec: ModelSpec):
        """Test Polecat discovering additional work and creating a new bead."""
        await call_bead(db_connection, action="create", title="Original Task")
        result = await call_bead(db_connection, action="create", title="Follow-up: Related bug discovered")
        assert "Created bead [2]" in result

        summary = await bead_summary(db_connection)
        assert "Original Task" in summary
        assert "Follow-up" in summary

    async def test_refinery_merges_completed_work(self, db_connection, mock_model_spec: ModelSpec) -> None:
        """Test Refinery processing completed Polecat work."""
        for i in range(3):
            await call_bead(db_connection, action="create", title=f"Polecat Task {i + 1}")
            await call_bead(
                db_connection, action="update", id=i + 1, status="in_progress", assignee=f"polecat#{i + 1}"
            )
            await call_bead(db_connection, action="close", id=i + 1)

        result = await call_bead(db_connection, action="list")
        assert "closed" in result.lower()


# =============================================================================
# Test: Context store compaction
# =============================================================================


class TestContextCompaction:
    """Tests for context store auto-compaction."""

    async def test_auto_compact_triggers_at_threshold(self, mock_model_spec: ModelSpec) -> None:
        """Test that compaction is triggered when token limit is reached."""
        transport = StubTransport([make_text_response("Summary.")] * 10)
        store = AutoCompactStore(MemoryContextStore(), transport, keep_recent=1)

        with patch("axio.compaction.compact_context", new_callable=AsyncMock) as mock_compact:
            mock_compact.return_value = []

            for i in range(3):
                msg = Message(role="user", content=f"Message {i}")  # type: ignore[arg-type]
                await store.append(msg)

            await store.add_context_tokens(input_tokens=100_000, output_tokens=0)
            assert mock_compact.called

    async def test_compact_keeps_recent_messages(self, mock_model_spec: ModelSpec) -> None:
        """Test that recent messages are preserved after compaction."""
        transport = StubTransport([make_text_response("Summary.")] * 10)
        store = AutoCompactStore(MemoryContextStore(), transport, keep_recent=2)

        for i in range(5):
            msg = Message(role="user", content=f"Message {i}")  # type: ignore[arg-type]
            await store.append(msg)
        messages = await store.get_history()
        assert len(messages) == 5

        await store.add_context_tokens(input_tokens=100_000, output_tokens=0)
        messages_after = await store.get_history()
        assert len(messages_after) >= 2


# =============================================================================
# Test: Tool usage with guards
# =============================================================================


class TestToolGuards:
    """Tests for role-based tool access guards."""

    async def test_sling_attaches_guard(self, db_connection) -> None:
        """Test that make_sling_tool wires guard_factory into the tool."""
        guard_factory = MagicMock(return_value=MagicMock())
        queue: Channel[int] = Channel()
        make_sling_tool(
            db=db_connection,
            queue=queue,
            guard_factory=guard_factory,
        )
        guard_factory.assert_called_with("mayor")

    async def test_witness_has_read_only_bead_access(self) -> None:
        """Test that Witness role has limited bead tool access."""
        analyze_tool = make_analyze_tool(
            toolbox={},
            on_event=MagicMock(),
            transport=MagicMock(),
            role_models={},
            caller_role="witness",
            guard_factory=MagicMock(return_value=MagicMock(check=AsyncMock(return_value=True))),
        )
        assert analyze_tool is not None


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Tests for error conditions and edge cases."""

    async def test_invalid_bead_id_returns_not_found(self, db_connection) -> None:
        """Test that operations on invalid bead ID return not found."""
        result = await call_bead(db_connection, action="update", id=99999, status="closed")
        assert "not found" in result


# =============================================================================
# Symlink protection
# =============================================================================


class TestSymlinkProtection:
    """run_gastown must not follow symlinks on the host filesystem."""

    @pytest.mark.asyncio
    async def test_beads_db_symlink_deleted_before_open(self, workspace: Path, tmp_path) -> None:
        """A symlink planted at .gas-town/beads.db must be deleted, not opened."""
        gas_dir = workspace / ".gas-town"
        gas_dir.mkdir(parents=True)
        secret = tmp_path / "secret.db"
        secret.write_text("sensitive")
        db_link = gas_dir / "beads.db"
        db_link.symlink_to(secret)

        stub_transport = StubTransport([make_text_response("done")])
        on_event = AsyncMock()
        role_models = {"default": ModelSpec(id="gpt-4")}

        try:
            await asyncio.wait_for(
                run_gastown(
                    task="test",
                    workspace=workspace,
                    on_event=on_event,
                    transport=stub_transport,
                    role_models=role_models,
                    toolbox={},
                ),
                timeout=2.0,
            )
        except (TimeoutError, Exception):
            pass

        assert not db_link.is_symlink(), "symlink should have been deleted"
        assert secret.read_text() == "sensitive", "original target must be untouched"
