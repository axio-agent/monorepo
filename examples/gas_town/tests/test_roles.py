"""Tests for gas_town.roles module."""

from __future__ import annotations

import pytest
from axio.agent import Agent
from axio.agent_loader import TomlAgentLoader
from axio.transport import DummyCompletionTransport

from gas_town.roles import MAYOR, ROLE_NAMES, ROLES_DIR


@pytest.fixture
def loader() -> TomlAgentLoader:
    return TomlAgentLoader()


class TestMayor:
    """Tests for the Mayor agent."""

    def test_mayor_is_agent(self) -> None:
        assert isinstance(MAYOR, Agent)

    def test_mayor_has_dummy_transport(self) -> None:
        assert isinstance(MAYOR.transport, DummyCompletionTransport)

    def test_mayor_is_chief_of_staff(self) -> None:
        assert "convoy" in MAYOR.system.lower() or "dispatch" in MAYOR.system.lower()

    def test_mayor_dispatches_polecats_directly(self) -> None:
        assert "sling" in MAYOR.system
        assert "await_beads" in MAYOR.system
        assert "spawn_witness" not in MAYOR.system


class TestWorkerRoles:
    """Tests for worker role TOML definitions."""

    def test_all_expected_roles_present(self) -> None:
        expected = {"polecat", "witness", "refinery", "crew"}
        assert expected == set(ROLE_NAMES)

    def test_each_role_toml_loads(self, loader) -> None:
        for name in ROLE_NAMES:
            spec = loader.load_file(ROLES_DIR / f"{name}.toml")
            assert spec.name == name
            assert spec.system
            assert spec.description

    def test_descriptions_are_concise(self, loader) -> None:
        for name in ROLE_NAMES:
            spec = loader.load_file(ROLES_DIR / f"{name}.toml")
            assert len(spec.description) < 200, f"{name} description too long"

    def test_descriptions_are_unique(self, loader) -> None:
        descriptions = [loader.load_file(ROLES_DIR / f"{n}.toml").description for n in ROLE_NAMES]
        assert len(descriptions) == len(set(descriptions)), "Role descriptions are not unique"

    def test_polecat_has_single_task_focus(self, loader) -> None:
        spec = loader.load_file(ROLES_DIR / "polecat.toml")
        assert "one job" in spec.system.lower() or "ONE job" in spec.system
        assert "close" in spec.system.lower()

    def test_witness_has_monitoring_focus(self, loader) -> None:
        spec = loader.load_file(ROLES_DIR / "witness.toml")
        assert "monitor" in spec.system.lower() or "patrol" in spec.system.lower()

    def test_refinery_has_merge_focus(self, loader) -> None:
        spec = loader.load_file(ROLES_DIR / "refinery.toml")
        assert "merge" in spec.system.lower() or "integrate" in spec.system.lower()

    def test_crew_has_human_facing_focus(self, loader) -> None:
        spec = loader.load_file(ROLES_DIR / "crew.toml")
        assert "long" in spec.system.lower() or "human" in spec.system.lower()

    def test_roles_build_agents(self, loader) -> None:
        for name in ROLE_NAMES:
            spec = loader.load_file(ROLES_DIR / f"{name}.toml")
            agent = Agent(system=spec.system, transport=DummyCompletionTransport(), max_iterations=spec.max_iterations)
            assert isinstance(agent, Agent)
