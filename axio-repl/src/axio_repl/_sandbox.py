"""Optional Docker isolation for the tools handed to the model.

``axio-tools-docker`` already implements every filesystem and execution tool
against a container; this module only decides when to use it and swaps the tools
by name. Coordination tools (spawn/stop/interrupt/peers) are left alone — they
touch no files. Spawned subagents inherit the parent's tools, so substituting
once covers the whole tree.

The container runs with networking disabled, which costs nothing here: the model
is contacted by axio-repl on the host, never from inside the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from axio import Tool

from axio_repl import _search

# Replaced by the container-backed implementations of the same name.
SANDBOXED_TOOL_NAMES = frozenset({"read_file", "write_file", "patch_file", "list_files", "shell"})

# Offered only inside the sandbox: running arbitrary Python on the host is
# exactly what this is meant to avoid.
SANDBOX_ONLY_TOOL_NAMES = frozenset({"run_python"})

WORKDIR = "/workspace"

# ast-grep installs its binary as `ast-grep`. `sg` is shadow-utils' setgid
# helper on Linux and must never be invoked in its place.
AST_GREP = "ast-grep"

_AST_GREP_DOC = """Structural search: match a code *pattern* by syntax rather than text.
    Patterns are written as ordinary code with `$VAR` metavariables, e.g.
    `$A == $A` or `def $NAME($$$ARGS)`. Prefer this over search_files when
    looking for a shape of code rather than a literal string."""


def docker_available() -> bool:
    """True when the bindings are installed and a daemon looks reachable."""
    try:
        import aiodocker  # noqa: F401
    except ImportError:
        return False
    return Path("/var/run/docker.sock").exists()


def ast_grep_available() -> bool:
    return shutil.which(AST_GREP) is not None


def _ast_grep_argv(pattern: str, path: str, lang: str | None) -> list[str]:
    argv = [AST_GREP, "run", "--pattern", pattern, "--heading", "never"]
    if lang:
        argv += ["--lang", lang]
    argv.append(path)
    return argv


def _truncate(text: str, max_results: int) -> str:
    lines = text.splitlines()
    if not lines:
        return "No matches"
    if len(lines) <= max_results:
        return text
    return "\n".join(lines[:max_results]) + f"\n[truncated at {max_results} lines]"


def _make_host_ast_grep() -> Any:
    async def ast_grep(pattern: str, path: str = ".", lang: str | None = None, max_results: int = 100) -> str:
        proc = await asyncio.create_subprocess_exec(
            *_ast_grep_argv(pattern, path, lang),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return _truncate(out.decode("utf-8", "replace").strip(), max_results)

    ast_grep.__doc__ = _AST_GREP_DOC
    return ast_grep


def _make_sandbox_overrides(sandbox: Any) -> list[Tool[Any]]:
    """search_files and ast_grep re-expressed against the container.

    axio-tools-docker has no counterpart for either, and left alone they would
    keep reading the host filesystem — the one thing the sandbox exists to stop.
    """

    async def search_files(query: str, path: str = ".", regex: bool = False, max_results: int = 100) -> str:
        script = "/tmp/.axio_search.py"
        await sandbox.write_file(script, Path(_search.__file__).read_text(encoding="utf-8"))
        params = json.dumps({"query": query, "path": path, "regex": regex, "max_results": max_results})
        out: str = await sandbox.exec(f"python3 {script}", stdin=params)
        return out

    async def ast_grep(pattern: str, path: str = ".", lang: str | None = None, max_results: int = 100) -> str:
        cmd = shlex.join(_ast_grep_argv(pattern, path, lang))
        return _truncate(await sandbox.exec(cmd), max_results)

    search_files.__doc__ = _search.search.__doc__
    ast_grep.__doc__ = _AST_GREP_DOC
    return [Tool(name="search_files", handler=search_files), Tool(name="ast_grep", handler=ast_grep)]


async def build_tools(
    stack: AsyncExitStack,
    tools: list[Tool[Any]],
    mode: str,
    image: str,
    workspace: Path,
) -> tuple[list[Tool[Any]], str, Path]:
    """Return the toolset, a one-line description, and the root the tools see.

    In a container the workspace is mounted elsewhere than on the host, and the
    system prompt has to state the path the model can actually act on.
    """
    if mode == "none" or (mode == "auto" and not docker_available()):
        result = [t for t in tools if t.name not in SANDBOX_ONLY_TOOL_NAMES]
        if ast_grep_available():
            result.append(Tool(name="ast_grep", handler=_make_host_ast_grep()))
        return result, "host — tools run directly on this machine", workspace

    from axio_tools_docker.sandbox import DockerSandbox

    sandbox = await stack.enter_async_context(
        DockerSandbox(image=image, volumes={WORKDIR: str(workspace)}, workdir=WORKDIR, network=False)
    )
    available = {t.name: t for t in sandbox.tools}
    overrides = {t.name: t for t in _make_sandbox_overrides(sandbox)}

    merged: list[Tool[Any]] = []
    for tool in tools:
        if tool.name in overrides:
            merged.append(overrides.pop(tool.name))
        else:
            merged.append(available.get(tool.name, tool))
    merged.extend(overrides.values())
    for name in SANDBOX_ONLY_TOOL_NAMES & available.keys():
        merged.append(available[name])
    return merged, f"docker — {image}, no network, {workspace} mounted at {WORKDIR}", Path(WORKDIR)
