"""Tests for role definitions in the agent_swarm package."""

from __future__ import annotations

import pytest
from axio import Agent
from axio.agent_loader import TomlAgentLoader
from axio.transport import DummyCompletionTransport

from agent_swarm.roles import ROLE_NAMES, ROLES_DIR, make_orchestrator


class TestRoles:
    """Test the role registry."""

    def test_role_names_contains_expected_roles(self):
        """Test that ROLE_NAMES contains all expected roles."""
        expected_roles = {
            "architect",
            "backend_dev",
            "frontend_dev",
            "project_manager",
            "qa",
            "designer",
            "ux_engineer",
            "etl_engineer",
            "security_engineer",
            "challenger",
        }
        assert expected_roles.issubset(set(ROLE_NAMES)), f"Missing roles: {expected_roles - set(ROLE_NAMES)}"
        assert set(ROLE_NAMES) == expected_roles, f"Unexpected roles: {set(ROLE_NAMES) - expected_roles}"

    def test_each_role_toml_has_description_and_system(self):
        """Test that each role TOML has a description and system prompt."""
        loader = TomlAgentLoader()
        for role_name in ROLE_NAMES:
            spec = loader.load_file(ROLES_DIR / f"{role_name}.toml")
            assert isinstance(spec.description, str), f"Role {role_name} description is not a string"
            assert len(spec.description) > 0, f"Role {role_name} description is empty"
            assert isinstance(spec.system, str), f"Role {role_name} system is not a string"
            assert len(spec.system) > 0, f"Role {role_name} system is empty"

    def test_each_role_toml_builds_agent(self):
        """Test that each role TOML can be used to build an Agent."""
        loader = TomlAgentLoader()
        for role_name in ROLE_NAMES:
            spec = loader.load_file(ROLES_DIR / f"{role_name}.toml")
            agent = Agent(system=spec.system, transport=DummyCompletionTransport(), max_iterations=spec.max_iterations)
            assert isinstance(agent, Agent), f"Role {role_name} failed to build Agent"
            assert isinstance(agent.transport, DummyCompletionTransport)

    def test_make_orchestrator_returns_agent(self):
        """Test that make_orchestrator() returns an Agent."""
        orchestrator = make_orchestrator("test roster")
        assert orchestrator is not None
        assert isinstance(orchestrator, Agent)

    def test_role_descriptions_are_non_empty_strings(self):
        """Test that role descriptions are non-empty strings."""
        loader = TomlAgentLoader()
        for role_name in ROLE_NAMES:
            spec = loader.load_file(ROLES_DIR / f"{role_name}.toml")
            assert isinstance(spec.description, str), f"Role {role_name} description is not a string"
            assert spec.description.strip(), f"Role {role_name} description is empty or whitespace only"

    @pytest.mark.parametrize(
        "role_name",
        [
            "architect",
            "backend_dev",
            "frontend_dev",
            "project_manager",
            "qa",
            "designer",
            "ux_engineer",
            "etl_engineer",
            "security_engineer",
            "challenger",
        ],
    )
    def test_role_description_format(self, role_name):
        """Test that role descriptions follow expected format."""
        loader = TomlAgentLoader()
        spec = loader.load_file(ROLES_DIR / f"{role_name}.toml")
        assert len(spec.description) < 200, f"Role {role_name} description too long: {len(spec.description)} chars"

    def test_make_orchestrator_system_contains_delegate(self):
        """Test that orchestrator system prompt mentions delegation."""
        orchestrator = make_orchestrator("test roster")
        assert "delegate" in orchestrator.system.lower(), "Orchestrator system prompt should mention delegate tool"

    def test_make_orchestrator_system_contains_roster(self):
        """Test that make_orchestrator embeds the roster into the system prompt."""
        roster = "\n".join(f"  {name:20s} - test description" for name in ROLE_NAMES)
        orchestrator = make_orchestrator(roster)
        for role_name in ROLE_NAMES:
            assert role_name in orchestrator.system, f"Role {role_name} not found in orchestrator system prompt"
