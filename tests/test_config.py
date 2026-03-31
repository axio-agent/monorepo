"""Tests for SandboxConfig."""

from __future__ import annotations

import pytest

from axio_tools_docker.config import SandboxConfig


def test_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.image == "python:latest"
    assert cfg.memory == "256m"
    assert cfg.cpus == "1.0"
    assert cfg.network is False
    assert cfg.workdir == "/workspace"


def test_frozen() -> None:
    cfg = SandboxConfig()
    with pytest.raises(AttributeError):
        cfg.image = "ubuntu"  # type: ignore[misc]


def test_to_dict_omits_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.to_dict() == {}


def test_to_dict_non_defaults() -> None:
    cfg = SandboxConfig(image="ubuntu:22.04", memory="512m", network=True)
    d = cfg.to_dict()
    assert d == {"image": "ubuntu:22.04", "memory": "512m", "network": "True"}


def test_roundtrip() -> None:
    original = SandboxConfig(image="node:20", cpus="2.0", workdir="/app", network=True)
    restored = SandboxConfig.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_empty() -> None:
    cfg = SandboxConfig.from_dict({})
    assert cfg == SandboxConfig()


def test_network_bool_serialization() -> None:
    for truthy in ("True", "true", "1", "yes"):
        cfg = SandboxConfig.from_dict({"network": truthy})
        assert cfg.network is True

    for falsy in ("False", "false", "0", "no"):
        cfg = SandboxConfig.from_dict({"network": falsy})
        assert cfg.network is False
