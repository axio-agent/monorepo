# Writing Guards

Guards control whether a tool call is allowed to execute. They inspect (and
can modify) the validated handler instance before it runs.

## Subclassing PermissionGuard

```python
from axio import PermissionGuard
from axio.exceptions import GuardError


class MaxLengthGuard(PermissionGuard):
    """Deny tool calls where any string field exceeds a length limit."""

    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length

    async def check(self, handler: Any) -> Any:
        for name, value in handler.model_dump().items():
            if isinstance(value, str) and len(value) > self.max_length:
                raise GuardError(
                    f"Field '{name}' exceeds {self.max_length} characters"
                )
        return handler
```

Key rules:

- **Return** the handler to allow execution.
- **Raise** `GuardError` to deny. The error message is sent to the model.
- You may return a **modified** handler instance (e.g., to sanitize inputs).

## Attaching guards to tools

```python
tool = Tool(
    name="write_file",
    description="Write a file",
    handler=WriteFile,
    guards=(MaxLengthGuard(max_length=50000),),
)
```

Guards run sequentially in tuple order. The output of one guard is the input
to the next.

## ConcurrentGuard

If your guard calls an external service (e.g., an LLM for risk assessment),
use `ConcurrentGuard` to limit concurrent calls:

```python
from axio import ConcurrentGuard


class LLMRiskGuard(ConcurrentGuard):
    """Use an LLM to assess whether a tool call is safe."""

    concurrency = 2  # at most 2 concurrent risk assessments

    async def check(self, handler: Any) -> Any:
        risk = await self._assess_risk(handler)
        if risk > 0.8:
            raise GuardError(f"Tool call deemed too risky (score={risk:.2f})")
        return handler

    async def _assess_risk(self, handler: Any) -> float:
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

```python
Tool(
    name="shell",
    description="Run a shell command",
    handler=Shell,
    guards=(
        AllowedCommandGuard(),  # Fast: check against allowlist
        PathGuard(),            # Fast: validate file paths
        LLMRiskGuard(),         # Slow: LLM assessment (only if fast checks pass)
    ),
)
```

If any guard raises `GuardError`, subsequent guards are skipped and the
error is returned to the model immediately.
