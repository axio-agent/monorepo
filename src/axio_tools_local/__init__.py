from typing import Any


def _short(value: Any, limit: int = 60) -> str:
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "..."
