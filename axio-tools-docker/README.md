# axio-tools-docker

[![PyPI](https://img.shields.io/pypi/v/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Docker sandbox tools for [axio](https://github.com/axio-agent/axio).

Run agent-generated code and commands inside isolated Docker containers. The agent gets `sandbox_exec`, `sandbox_write`, and `sandbox_read` tools that operate entirely within the sandbox — the host filesystem stays untouched.

## Features

- **Isolated execution** — code runs inside a Docker container, not on the host
- **Configurable image** — use any Docker image as the sandbox environment
- **Three sandboxed tools** — execute commands, write files, read files — all inside the container
- **Persistent sandbox** — container is reused across tool calls within a session for faster execution
- **TUI integration** — configure image, memory limits, and CPU from the `axio-tui` settings screen

## Requirements

Docker must be installed and running:

```bash
docker info   # should succeed
```

## Installation

```bash
pip install axio-tools-docker
```

## Usage

```python
from axio import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport
from axio_tools_docker.plugin import DockerPlugin

async def main() -> None:
    plugin = DockerPlugin()
    await plugin.init()   # uses default config (python:3.12-slim)

    agent = Agent(
        system=(
            "You are a coding assistant. Use sandbox_exec to run code safely. "
            "Never attempt to access the host filesystem directly."
        ),
        tools=plugin.all_tools,
        transport=OpenAITransport(api_key="sk-...", model="gpt-4o"),
    )

    ctx = MemoryContextStore()
    result = await agent.run(
        "Write a Python script that computes the first 20 Fibonacci numbers and run it.",
        ctx,
    )
    print(result)
```

## Sandbox tools

| Tool | Description |
|---|---|
| `sandbox_exec` | Run a shell command inside the container; returns stdout + stderr |
| `sandbox_write` | Write a file into the container's filesystem |
| `sandbox_read` | Read a file from the container's filesystem |

## Configuration

```python
from axio_tools_docker.config import SandboxConfig

config = SandboxConfig(
    image="python:3.12-slim",
    memory_limit="512m",
    cpu_quota=100000,    # 1 CPU
    work_dir="/workspace",
)
```

## Plugin registration

```toml
[project.entry-points."axio.tools.settings"]
docker = "axio_tools_docker.plugin:DockerPlugin"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-tools-local](https://github.com/axio-agent/axio-tools-local) · [axio-tools-mcp](https://github.com/axio-agent/axio-tools-mcp) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
