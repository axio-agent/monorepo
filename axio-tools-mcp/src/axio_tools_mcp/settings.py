"""Non-TUI helpers and sentinel for MCP settings."""

from __future__ import annotations


def _headers_to_str(headers: dict[str, str]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in headers.items())


def _parse_headers(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, value = pair.split(":", 1)
            result[key.strip()] = value.strip()
    return result


class _DeleteSentinel:
    """Sentinel object to signal a delete action from the edit screen."""


def _make_delete_sentinel() -> _DeleteSentinel:
    return _DeleteSentinel()


def _is_delete_sentinel(obj: object) -> bool:
    return isinstance(obj, _DeleteSentinel)
