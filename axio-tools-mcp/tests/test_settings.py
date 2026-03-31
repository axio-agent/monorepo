"""Tests for settings screens (basic import and sentinel logic)."""

from __future__ import annotations

from axio_tools_mcp.settings import _is_delete_sentinel, _make_delete_sentinel, _parse_headers


def test_delete_sentinel() -> None:
    sentinel = _make_delete_sentinel()
    assert _is_delete_sentinel(sentinel)
    assert not _is_delete_sentinel(None)
    assert not _is_delete_sentinel("delete")


def test_parse_headers_empty() -> None:
    assert _parse_headers("") == {}


def test_parse_headers_single() -> None:
    result = _parse_headers("Authorization:Bearer token")
    assert result == {"Authorization": "Bearer token"}


def test_parse_headers_multiple() -> None:
    result = _parse_headers("X-Key:abc, X-Secret:xyz")
    assert result == {"X-Key": "abc", "X-Secret": "xyz"}


def test_import_hub_screen() -> None:
    """MCPHubScreen can be imported."""
    from axio_tools_mcp.settings import MCPHubScreen

    assert MCPHubScreen is not None


def test_import_edit_screen() -> None:
    """MCPServerEditScreen can be imported."""
    from axio_tools_mcp.settings import MCPServerEditScreen

    assert MCPServerEditScreen is not None
