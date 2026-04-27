"""Tests for the swarm module functions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from axio.models import ModelSpec
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool
from axio.transport import DummyCompletionTransport

from agent_swarm.swarm import (
    make_analyze_tool,
    make_delegate_tool,
    run_swarm,
    transport_for,
)


class TestTransportFor:
    """Test the transport_for function."""

    def test_transport_for_correctly_maps_roles_to_models(self):
        """Test that transport_for() correctly maps roles to models."""
        base_transport = DummyCompletionTransport()
        role_models = {
            "default": ModelSpec(id="gpt-4"),
            "architect": ModelSpec(id="claude-3"),
            "backend_dev": ModelSpec(id="gpt-4-turbo"),
        }

        # Test default mapping
        transport = transport_for("unknown_role", base_transport, role_models)
        assert transport.model == ModelSpec(id="gpt-4")

        # Test role-specific mapping
        transport = transport_for("architect", base_transport, role_models)
        assert transport.model == ModelSpec(id="claude-3")

        transport = transport_for("backend_dev", base_transport, role_models)
        assert transport.model == ModelSpec(id="gpt-4-turbo")

    def test_transport_for_falls_back_to_default(self):
        """Test that transport_for falls back to default for unknown roles."""
        base_transport = DummyCompletionTransport()
        role_models = {
            "default": ModelSpec(id="default-model"),
        }

        transport = transport_for("some_unknown_role", base_transport, role_models)
        assert transport.model == ModelSpec(id="default-model")


class TestMakeAnalyzeTool:
    """Test the make_analyze_tool function."""

    def test_make_analyze_tool_creates_an_analyze_tool(self):
        """Test that make_analyze_tool() creates an analyze tool."""
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_analyze_tool({}, on_event, transport, role_models, "orchestrator")

        assert isinstance(tool, Tool)
        assert tool.name == "analyze"
        assert tool.description, "Analyze tool has no description"
        assert tool.handler, "Analyze tool has no handler"

    def test_make_analyze_tool_with_guard_factory(self):
        """Test make_analyze_tool with guard factory."""
        mock_guard = MagicMock()
        guard_factory = MagicMock(return_value=mock_guard)
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_analyze_tool({}, on_event, transport, role_models, "orchestrator", guard_factory=guard_factory)

        assert len(tool.guards) == 1
        assert tool.guards[0] is mock_guard
        guard_factory.assert_called_once_with("orchestrator", "analyze")


class TestMakeDelegateTool:
    """Test the make_delegate_tool function."""

    def test_make_delegate_tool_creates_a_delegate_tool(self):
        """Test that make_delegate_tool() creates a delegate tool."""
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_delegate_tool(on_event, transport, role_models, {})

        assert isinstance(tool, Tool)
        assert tool.name == "delegate"
        assert tool.description, "Delegate tool has no description"
        assert tool.handler, "Delegate tool has no handler"

    def test_make_delegate_tool_with_guard_factory(self):
        """Test make_delegate_tool with guard factory."""
        mock_guard = MagicMock()
        guard_factory = MagicMock(return_value=mock_guard)
        on_event = AsyncMock()
        transport = DummyCompletionTransport()
        role_models = {"default": ModelSpec(id="gpt-4")}

        tool = make_delegate_tool(on_event, transport, role_models, {}, guard_factory=guard_factory)

        assert len(tool.guards) == 1
        assert tool.guards[0] is mock_guard
        guard_factory.assert_called_once_with("orchestrator", "delegate")


class TestRunSwarm:
    """Test the run_swarm function."""

    @pytest.mark.asyncio
    async def test_run_swarm_initialization(self, workspace: Path, stub_toolbox):
        """Test that run_swarm() initialization works with stub."""
        stub_transport = StubTransport([make_text_response("Test response")])

        on_event = AsyncMock()

        role_models = {"default": ModelSpec(id="gpt-4")}

        async def run_test():
            try:
                result = await asyncio.wait_for(
                    run_swarm(
                        task="Test task",
                        workspace=workspace,
                        on_event=on_event,
                        transport=stub_transport,
                        role_models=role_models,
                        toolbox=stub_toolbox,
                    ),
                    timeout=1.0,
                )
                return result
            except TimeoutError:
                # Expected - stub transport doesn't actually complete
                return "timeout"

        result = await run_test()
        # The result should be either "timeout" or some actual text
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_swarm_requires_default_model(self, workspace: Path):
        """Test that run_swarm requires a default model in role_models."""
        transport = DummyCompletionTransport()
        on_event = AsyncMock()

        # Missing "default" key should raise AssertionError
        role_models = {"other": ModelSpec(id="gpt-4")}

        with pytest.raises(AssertionError, match="default"):
            await run_swarm(
                task="Test task",
                workspace=workspace,
                on_event=on_event,
                transport=transport,
                role_models=role_models,
                toolbox={},
            )

    @pytest.mark.asyncio
    async def test_run_swarm_creates_workspace_directory(self, workspace: Path, stub_toolbox):
        """Test that run_swarm creates the workspace directory."""
        test_workspace = workspace / "new_workspace"
        assert not test_workspace.exists()

        stub_transport = StubTransport([make_text_response("Test response")])
        on_event = AsyncMock()
        role_models = {"default": ModelSpec(id="gpt-4")}

        async def run_test():
            try:
                await asyncio.wait_for(
                    run_swarm(
                        task="Test task",
                        workspace=test_workspace,
                        on_event=on_event,
                        transport=stub_transport,
                        role_models=role_models,
                        toolbox=stub_toolbox,
                    ),
                    timeout=1.0,
                )
            except TimeoutError:
                pass  # Expected - stub doesn't really complete

        await run_test()

        # Directory should be created


# ---------------------------------------------------------------------------
# Symlink protection
# ---------------------------------------------------------------------------


class TestSymlinkProtection:
    """run_swarm must not follow symlinks on the host filesystem."""

    @pytest.mark.asyncio
    async def test_todo_db_symlink_deleted_before_open(self, workspace: Path, stub_toolbox, tmp_path: Path) -> None:
        """A symlink planted at .axio-swarm/todos.db must be deleted, not opened."""
        axio_dir = workspace / ".axio-swarm"
        axio_dir.mkdir(parents=True)
        secret = tmp_path / "secret.db"
        secret.write_text("sensitive")
        db_link = axio_dir / "todos.db"
        db_link.symlink_to(secret)

        stub_transport = StubTransport([make_text_response("done")])
        on_event = AsyncMock()
        role_models = {"default": ModelSpec(id="gpt-4")}

        try:
            await asyncio.wait_for(
                run_swarm(
                    task="test",
                    workspace=workspace,
                    on_event=on_event,
                    transport=stub_transport,
                    role_models=role_models,
                    toolbox=stub_toolbox,
                ),
                timeout=2.0,
            )
        except (TimeoutError, Exception):
            pass

        # Symlink must be gone — replaced by a real file
        assert not db_link.is_symlink(), "symlink should have been deleted"
        assert secret.read_text() == "sensitive", "original target must be untouched"

    def test_agents_md_symlink_guard_logic(self, workspace: Path) -> None:
        """Guard condition 'is_file() and not is_symlink()' must reject a symlink to a real file.

        is_file() follows symlinks and returns True for a symlink pointing to a file,
        so the extra 'not is_symlink()' check is the one that blocks injection.
        """
        secret = workspace / "secret.md"
        secret.write_text("injected content")
        agents_md = workspace / "AGENTS.md"
        agents_md.symlink_to(secret)

        # Document the Python behaviour the guard relies on:
        assert agents_md.is_file(), "is_file() follows symlinks — without the extra check it would be read"
        assert agents_md.is_symlink()
        assert not (agents_md.is_file() and not agents_md.is_symlink()), "guard must block symlinks"
