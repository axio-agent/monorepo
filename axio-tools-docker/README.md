# axio-tools-docker

[![PyPI](https://img.shields.io/pypi/v/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Docker sandbox tools for [axio](https://github.com/mosquito/axio-agent).

`DockerSandbox` is an async context manager that spins up an isolated Docker
container on entry and removes it on exit. Inside the context it exposes six
`axio` tools — the same `shell`, `write_file`, `read_file`, `list_files`,
`run_python`, and `patch_file` as `axio-tools-local`, but every operation runs
inside the container, never on the host.

## Requirements

Docker must be installed and running:

```bash
docker info   # should succeed
```

The package talks to the Docker Engine API directly via
[aiodocker](https://aiodocker.readthedocs.io/) — the `docker` CLI is not
required.

## Installation

```bash
pip install axio-tools-docker
```

## Quick start

<!-- name: test_readme_usage; mark: skip -->
```python
import asyncio
import os
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport
from axio_tools_docker import DockerSandbox

async def main() -> None:
    async with DockerSandbox(
        os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock"),
        image="python:3.12-slim",
        volumes={"/workspace": "/tmp/agent-workspace"},
    ) as sandbox:
        agent = Agent(
            system="You are a coding assistant. Use the sandbox tools to run code safely.",
            tools=sandbox.tools,
            transport=OpenAITransport(api_key="sk-...", model="gpt-4o"),
        )
        ctx = MemoryContextStore()
        result = await agent.run(
            "Write a Python script that computes the first 20 Fibonacci numbers and run it.",
            ctx,
        )
        print(result)

asyncio.run(main())
```

## Sandbox tools

These mirror `axio-tools-local` exactly — same names and field schemas:

| Tool | Description |
|---|---|
| `shell` | Run a shell command; returns stdout + stderr. Supports `timeout`, `cwd`, `stdin`. |
| `write_file` | Create or overwrite a file. Parent directories are created automatically. |
| `read_file` | Read a file with optional `start_line`/`end_line`, `line_numbers`, `max_chars`. |
| `list_files` | List a directory; directories first with a trailing `/`. |
| `run_python` | Execute a Python snippet in a subprocess. Supports `timeout`, `cwd`, `stdin`. |
| `patch_file` | Replace lines `from_line`..`to_line` (1-indexed, inclusive). `to_line = from_line - 1` inserts. |

## Container lifecycle

The container is created on `__aenter__` and removed on `__aexit__` (`docker rm -f`).
Cleanup runs even when the body raises an exception:

```python
async def run(ctx):
    async with DockerSandbox(image="alpine:latest") as sandbox:
        await agent.run("...", ctx)
    # container removed here (unless remove=False)
```

The image is pulled automatically if not present locally. If the Docker daemon
is unreachable, `__aenter__` raises immediately:

```
RuntimeError: Docker daemon not available at 'unix:///var/run/docker.sock': ...
```

The running container's ID is available as `sandbox.container_id` inside the
`async with` block.

## Named containers and reuse

Pass `name=` to give the container a fixed name. If a running container with
that name already exists, the sandbox attaches to it instead of creating a new
one, and never removes it on exit — regardless of `remove`:

```python
import asyncio
from axio_tools_docker import DockerSandbox

async def first_session() -> None:
    async with DockerSandbox(image="python:3.12-slim", name="my-sandbox", remove=False) as sb:
        await sb.exec("pip install requests")

async def second_session() -> None:
    async with DockerSandbox(name="my-sandbox") as sb:
        result = await sb.exec("python3 -c 'import requests; print(requests.__version__)'")

asyncio.run(first_session())
asyncio.run(second_session())
```

## Configuration

<!-- name: test_readme_config -->
```python
from axio_tools_docker import DockerSandbox

sandbox = DockerSandbox(
    "unix:///var/run/docker.sock",   # Docker daemon URL (positional)
    image="python:3.12-slim",
    memory="512m",                   # memory limit: "256m", "1g", …
    cpus="2.0",                      # CPU limit
    network=False,                   # False=none, True=default, str=explicit mode
    workdir="/workspace",
    volumes={"/workspace": "/tmp/host-dir"},   # {container_path: host_path}
    env={"PYTHONPATH": "/app"},
    user="nobody",
    name="my-agent-sandbox",
    remove=False,
    read_only=True,                  # read-only root filesystem
    shm_size="64m",                  # /dev/shm size
    cap_add=["NET_ADMIN"],           # add Linux capabilities
    cap_drop=["ALL"],                # drop Linux capabilities
    privileged=False,
    ulimits={"nofile": (1024, 65536), "nproc": 512},
    tmpfs={"/tmp": "size=128m,mode=1777"},
    ports={8080: 8080},              # {container_port: host_port}
    platform="linux/amd64",
    extra_hosts={"host.docker.internal": "host-gateway"},
    devices=["/dev/net/tun", "/dev/sda:/dev/xvda:r"],
    dns=["8.8.8.8", "1.1.1.1"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"unix:///var/run/docker.sock"` | Docker daemon URL (unix socket or TCP). Positional. |
| `image` | `str` | `"python:latest"` | Container image. Pulled automatically if not present. |
| `memory` | `str` | `"256m"` | Memory limit. Accepts `k`/`m`/`g` suffixes. |
| `cpus` | `str` | `"1.0"` | CPU limit as a decimal string. |
| `network` | `bool \| str` | `False` | Network mode. `False` → `none`. `True` → Docker default. String → explicit `NetworkMode` (e.g. `"host"`, `"bridge"`, `"my-net"`). |
| `workdir` | `str` | `"/workspace"` | Working directory inside the container. |
| `volumes` | `dict[str, str]` | `{}` | Bind mounts as `{container_path: host_path}`. |
| `env` | `dict[str, str]` | `{}` | Environment variables passed to all commands. |
| `user` | `str` | `""` | User to run as (e.g. `"nobody"`, `"1000"`). |
| `name` | `str` | `""` | Container name. Attaches to existing container if found; creates new one otherwise. |
| `remove` | `bool` | `True` | Remove container on exit. No effect when attached to an existing container. |
| `read_only` | `bool` | `False` | Read-only root filesystem. Combine with `tmpfs` for writable scratch paths. |
| `shm_size` | `str` | `""` | `/dev/shm` size (e.g. `"64m"`). Useful for PyTorch / shared-memory IPC. |
| `cap_add` | `list[str]` | `[]` | Linux capabilities to add (e.g. `["NET_ADMIN", "SYS_PTRACE"]`). |
| `cap_drop` | `list[str]` | `[]` | Linux capabilities to drop (e.g. `["ALL"]`). |
| `privileged` | `bool` | `False` | Extended privileges — full capability set and device access. Use with care. |
| `ulimits` | `dict[str, int \| tuple[int, int]]` | `{}` | Resource limits. `{"nofile": 1024}` → soft=hard=1024. `{"nofile": (1024, 65536)}` → soft/hard split. |
| `tmpfs` | `dict[str, str]` | `{}` | Tmpfs mounts as `{path: options}` (e.g. `{"/tmp": "size=128m,mode=1777"}`). |
| `ports` | `dict[int, int]` | `{}` | Port bindings as `{container_port: host_port}`. Only meaningful when `network != False`. |
| `platform` | `str` | `""` | Platform override (e.g. `"linux/amd64"`, `"linux/arm64"`). |
| `extra_hosts` | `dict[str, str]` | `{}` | Extra `/etc/hosts` entries as `{hostname: ip}`. |
| `devices` | `list[str]` | `[]` | Host devices to expose. Format: `"/dev/sda"`, `"/dev/sda:/dev/xvda"`, `"/dev/sda:/dev/xvda:r"`. |
| `dns` | `list[str]` | `[]` | DNS servers (e.g. `["8.8.8.8"]`). Only meaningful when `network != False`. |

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-tools-local](https://github.com/mosquito/axio-agent) · [axio-tools-mcp](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
