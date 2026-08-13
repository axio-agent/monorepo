"""Text search over a directory tree, in one file so it can run in two places.

On the host :func:`search` is called directly. Inside a Docker sandbox the tools
execute in a container that shares nothing but the workspace, so the same logic
is shipped there as source and run through the sandbox's ``run_python``. Keeping
it importable *and* self-contained is what lets both paths behave identically —
same walk order, same skip list, same Python regex dialect, same output format.

Nothing here may import from the rest of axio-repl: in the container only the
standard library exists.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


def iter_files(base: Path, skip: set[str]) -> Iterator[Path]:
    for current_dir, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in sorted(files):
            if not name.startswith("."):
                yield Path(current_dir) / name


def search(query: str, path: str = ".", regex: bool = False, max_results: int = 100) -> str:
    base = Path(path).resolve()
    if not base.exists():
        return f"error: path not found: {path}"

    try:
        pattern = re.compile(query) if regex else None
    except re.error as exc:
        return f"error: invalid regex: {exc}"

    matches: list[str] = []
    files = [base] if base.is_file() else list(iter_files(base, SKIP_DIRS))
    for file_path in files:
        if len(matches) >= max_results:
            break
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            found = pattern.search(line) if pattern else (query in line)
            if found:
                matches.append(f"{file_path}:{idx}: {line}")
                if len(matches) >= max_results:
                    break

    if not matches:
        return f"No matches for {query!r}"
    return "\n".join(matches)


if __name__ == "__main__":
    # Entry point for the sandboxed copy. The sandbox runs the script with no
    # arguments, so parameters arrive as a JSON object on stdin — which also
    # keeps regexes clear of shell quoting.
    print(search(**json.loads(sys.stdin.read())), end="")
