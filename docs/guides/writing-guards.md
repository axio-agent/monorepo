# Writing Guards

Guards control whether a tool call is allowed to execute. They inspect (and
can modify) the raw keyword arguments before the handler runs.

## Subclassing PermissionGuard

<!-- name: test_max_length_guard -->
```python
from typing import Any
from axio import PermissionGuard, GuardError, Tool


class MaxLengthGuard(PermissionGuard):
    """Deny tool calls where any string field exceeds a length limit."""

    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        for name, value in kwargs.items():
            if isinstance(value, str) and len(value) > self.max_length:
                raise GuardError(
                    f"Field '{name}' exceeds {self.max_length} characters"
                )
        return kwargs
```

Key rules:

- **Return** a `dict` of (possibly modified) kwargs to allow execution.
- **Raise** `GuardError` to deny. The error message is sent to the model.
- You may return a **modified** dict (e.g., to sanitize inputs).
- The `tool` argument carries `.name`, `.description`, and `.input_schema`.

## Attaching guards to tools

<!--
name: test_attaching_guards
```python
from typing import Any
from axio import Tool, PermissionGuard, GuardError

async def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return "ok"

class MaxLengthGuard(PermissionGuard):
    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        return kwargs
```
-->
<!-- name: test_attaching_guards -->
```python
from axio import Tool

tool = Tool(
    name="write_file",
    handler=write_file,
    guards=(MaxLengthGuard(max_length=50000),),
)
assert tool.name == "write_file"
assert len(tool.guards) == 1
```

Guards run sequentially in tuple order. The kwargs returned by one guard are
passed as inputs to the next.

## ConcurrentGuard

If your guard calls an external service (e.g., an LLM for risk assessment),
use `ConcurrentGuard` to limit concurrent calls:

<!-- name: test_llm_risk_guard -->
```python
from typing import Any
from axio import GuardError, ConcurrentGuard, Tool


class LLMRiskGuard(ConcurrentGuard):
    """Use an LLM to assess whether a tool call is safe."""

    concurrency = 2  # at most 2 concurrent risk assessments

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        risk = await self._assess_risk(tool.name, kwargs)
        if risk > 0.8:
            raise GuardError(f"Tool call deemed too risky (score={risk:.2f})")
        return kwargs

    async def _assess_risk(self, tool_name: str, kwargs: dict[str, Any]) -> float:
        # Call a secondary LLM to evaluate the tool call
        ...
```

The semaphore is acquired automatically in `ConcurrentGuard.__call__`
before `check()` is invoked.

## Registering as a plugin

```toml
[project.entry-points."axio.guards"]
max_length = "my_package.guards:MaxLengthGuard"
```

After installation, the guard appears in `discover_guards()` and can be
configured in the TUI.

## Composing guards

Guards compose naturally. Combine fast checks first, expensive checks last:

<!--
name: test_composing_guards
```python
from axio.permission import AllowAllGuard

async def shell(command: str) -> str:
    """Run a shell command."""
    return command

AllowedCommandGuard = AllowAllGuard
PathGuard = AllowAllGuard

class LLMRiskGuard(AllowAllGuard):
    pass
```
-->
<!-- name: test_composing_guards -->
```python
from axio import Tool

tool = Tool(
    name="shell",
    handler=shell,
    guards=(
        AllowedCommandGuard(),  # Fast: check against allowlist
        PathGuard(),            # Fast: validate file paths
        LLMRiskGuard(),         # Slow: LLM assessment (only if fast checks pass)
    ),
)
assert tool.name == "shell"
assert len(tool.guards) == 3
```

If any guard raises `GuardError`, subsequent guards are skipped and the
error is returned to the model immediately.
