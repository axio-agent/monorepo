# axio-tui-guards

[![PyPI](https://img.shields.io/pypi/v/axio-tui-guards)](https://pypi.org/project/axio-tui-guards/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tui-guards)](https://pypi.org/project/axio-tui-guards/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Permission guard plugins for [axio-tui](https://github.com/mosquito/axio-agent).

Guards intercept tool calls before execution and can allow, modify, or deny them. Ships two guards: a path-access guard that asks the user before touching filesystem locations, and an LLM-based guard that reviews tool calls for safety.

## Features

- **PathGuard** — intercepts tools that touch the filesystem; prompts the user once per directory and remembers the decision for the session
- **LLMGuard** — runs a secondary LLM call to review each tool invocation before allowing it
- **PermissionGuard protocol** — both guards implement `axio.PermissionGuard` and compose cleanly
- **TUI-aware** — prompts appear as native `axio-tui` dialogs, not blocking stdin reads

## Installation

```bash
pip install axio-tui-guards
```

Or install as part of the TUI bundle:

```bash
pip install "axio-tui[guards]"
```

## Guards

### PathGuard

Intercepts tool calls that contain filesystem paths (`file_path`, `filename`, `directory`, `path`, `cwd`). On the first access to a new directory it asks the user to **allow**, **allow all** (subtree), or **deny**.

<!--
name: test_readme_path_guard
```python
from typing import Any
from axio.tool import ToolHandler

class WriteFile(ToolHandler[Any]):
    """Write content to a file."""
    file_path: str
    content: str
    async def __call__(self, context: Any) -> str:
        return "ok"

class Shell(ToolHandler[Any]):
    """Run a shell command."""
    command: str
    cwd: str = "."
    async def __call__(self, context: Any) -> str:
        return "ok"
```
-->
<!-- name: test_readme_path_guard -->
```python
from axio_tui_guards.guards import PathGuard
from axio.tool import Tool

guard = PathGuard()   # uses TUI prompt_fn by default

tool = Tool(
    name="write_file",
    description="Write a file",
    handler=WriteFile,
    guards=(guard,),
)
```

Decision caching:
- **Allow** — grants access to the parent directory for the current session
- **Allow all** — grants access to the directory and all subdirectories
- **Deny** — blocks this path; raises `GuardError` immediately on retry

### LLMGuard

Uses a secondary LLM call to review the tool handler arguments before execution. If the reviewer deems the call unsafe it raises `GuardError` with the reason.

<!-- name: test_readme_llm_guard -->
```python
from axio_tui_guards.guards import LLMGuard
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response

reviewer = Agent(system="", tools=[], transport=StubTransport([make_text_response("allow")]))
guard = LLMGuard(agent=reviewer, context=MemoryContextStore())
```

### Composing guards

Guards are applied in order — attach both for layered protection:

<!--
name: test_readme_composing
```python
from typing import Any
from axio.tool import ToolHandler

class Shell(ToolHandler[Any]):
    """Run a shell command."""
    command: str
    cwd: str = "."
    async def __call__(self, context: Any) -> str:
        return "ok"
```
-->
<!-- name: test_readme_composing -->
```python
from axio_tui_guards.guards import PathGuard, LLMGuard
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio.tool import Tool

reviewer = Agent(system="", tools=[], transport=StubTransport([make_text_response("allow")]))
tool = Tool(
    name="shell",
    description="Run shell commands",
    handler=Shell,
    guards=(PathGuard(), LLMGuard(agent=reviewer, context=MemoryContextStore())),
)
```

## Custom guards

Implement the `PermissionGuard` protocol to write your own:

<!-- name: test_readme_custom_guard -->
```python
from axio.permission import PermissionGuard
from axio.exceptions import GuardError

class MyGuard(PermissionGuard):
    async def check(self, handler):
        if "rm -rf" in getattr(handler, "command", ""):
            raise GuardError("Refusing to run rm -rf")
        return handler
```

## Plugin registration

```toml
[project.entry-points."axio.guards"]
path = "axio_tui_guards.guards:PathGuard"
llm  = "axio_tui_guards.guards:LLMGuard"
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
