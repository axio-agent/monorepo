"""Tests for MCPServerConfig."""

import pytest

from axio_tools_mcp.config import MCPServerConfig


def test_stdio_config() -> None:
    cfg = MCPServerConfig(name="test", command="python", args=["-m", "server"])
    assert cfg.command == "python"
    assert cfg.args == ["-m", "server"]
    assert cfg.url is None


def test_http_config() -> None:
    cfg = MCPServerConfig(name="test", url="http://localhost:8000/mcp")
    assert cfg.url == "http://localhost:8000/mcp"
    assert cfg.command is None


def test_both_set_raises() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        MCPServerConfig(name="test", command="python", url="http://localhost")


def test_neither_set_raises() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        MCPServerConfig(name="test")


def test_frozen() -> None:
    cfg = MCPServerConfig(name="test", command="python")
    with pytest.raises(AttributeError):
        cfg.name = "other"  # type: ignore[misc]


def test_to_dict_stdio() -> None:
    cfg = MCPServerConfig(
        name="fs",
        command="npx",
        args=["-y", "@mcp/filesystem"],
        env={"HOME": "/tmp"},
    )
    d = cfg.to_dict()
    assert d["command"] == "npx"
    assert '"HOME"' in d["args"] or "-y" in d["args"]
    assert "env" in d
    assert "url" not in d


def test_to_dict_http() -> None:
    cfg = MCPServerConfig(
        name="remote",
        url="http://example.com/mcp",
        headers={"Authorization": "Bearer token"},
        timeout=60.0,
    )
    d = cfg.to_dict()
    assert d["url"] == "http://example.com/mcp"
    assert "headers" in d
    assert d["timeout"] == "60.0"
    assert "command" not in d


def test_roundtrip_stdio() -> None:
    original = MCPServerConfig(
        name="fs",
        command="npx",
        args=["-y", "@mcp/filesystem"],
        env={"HOME": "/tmp"},
        timeout=45.0,
    )
    restored = MCPServerConfig.from_dict("fs", original.to_dict())
    assert restored == original


def test_roundtrip_http() -> None:
    original = MCPServerConfig(
        name="remote",
        url="http://example.com/mcp",
        headers={"Authorization": "Bearer xxx"},
    )
    restored = MCPServerConfig.from_dict("remote", original.to_dict())
    assert restored == original


def test_default_timeout() -> None:
    cfg = MCPServerConfig(name="test", command="python")
    assert cfg.timeout == 30.0


def test_to_dict_omits_defaults() -> None:
    cfg = MCPServerConfig(name="test", command="python")
    d = cfg.to_dict()
    assert "timeout" not in d
    assert "args" not in d
    assert "env" not in d
    assert "headers" not in d
