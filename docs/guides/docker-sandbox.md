# Docker Sandbox

The `axio-tools-docker` package provides an isolated Docker container for
running agent-generated code and commands. `DockerSandbox` is an async context
manager: it creates a container on entry and removes it on exit. Inside the
context it exposes six tools that are drop-in replacements for
`axio-tools-local` — the agent gets the same `shell`, `write_file`,
`read_file`, `list_files`, `run_python`, and `patch_file` tools, but every
operation runs inside the container, not on the host.

## Installation

```bash
pip install axio-tools-docker
```

Docker must be installed and running on the host. The package communicates with
the Docker Engine API directly via
[aiodocker](https://aiodocker.readthedocs.io/) — the `docker` CLI is not
required.

## Quick start

<!-- name: test_docker_quick_start; mark: skip -->
```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport
from axio_tools_docker import DockerSandbox

async def main() -> None:
    async with DockerSandbox(image="python:3.12-slim") as sandbox:
        agent = Agent(
            system="You are a coding assistant. Use the sandbox tools.",
            tools=sandbox.tools,
            transport=OpenAITransport(api_key="sk-...", model="gpt-4o"),
        )
        result = await agent.run(
            "Write a script that prints the first 20 Fibonacci numbers, then run it.",
            MemoryContextStore(),
        )
        print(result)

asyncio.run(main())
```

## Sandbox tools

The six tools exposed by `sandbox.tools` have the same names and field schemas
as `axio-tools-local`, so switching between local and sandboxed execution
requires changing only the tool list passed to `Agent`.

| Tool | Description |
|------|-------------|
| `shell` | Run a shell command. Returns combined stdout/stderr. Supports `timeout`, `cwd`, and `stdin`. |
| `write_file` | Create or overwrite a file. Parent directories are created automatically. Accepts `file_path`, `content`, and optional `mode`. |
| `read_file` | Read a file with optional `start_line`/`end_line`, `line_numbers`, and `max_chars` truncation. Binary files return hex. |
| `list_files` | List directory contents. Directories appear first with a trailing `/`. |
| `run_python` | Execute a Python snippet in a subprocess inside the container. Supports `timeout`, `cwd`, and `stdin`. |
| `patch_file` | Replace lines `from_line`..`to_line` (1-indexed, inclusive). Set `to_line = from_line - 1` to insert without deleting. Always read the file first with `line_numbers=True`. |

The `tools` property is only valid inside the `async with` block. Accessing it
outside raises `RuntimeError`.

## Container lifecycle

`DockerSandbox` creates and starts the container in `__aenter__`. On
`__aexit__` the container is force-removed (`docker rm -f`) unless `remove=False`
was passed. Cleanup runs even when the body raises an exception.

The container runs `sleep infinity` as its main process; all tool operations
are executed via `docker exec`. The image is pulled automatically if not
present locally.

The `container_id` property returns the Docker ID of the running container and
is only valid inside the `async with` block:

<!-- name: test_docker_container_id; mark: docker -->
```python
import asyncio
from axio_tools_docker import DockerSandbox

async def main() -> None:
    async with DockerSandbox(image="alpine:latest") as sandbox:
        print(sandbox.container_id)   # e.g. "3f2a1b..."
        result = await sandbox.exec("uname -r")
        print(result)
    # container removed here

asyncio.run(main())
```

## Named containers and reuse

Pass `name=` to give the container a fixed name. When a running container with
that name already exists, the sandbox attaches to it instead of creating a new
one and skips removal on exit regardless of `remove`:

<!-- name: test_docker_named_reuse; mark: docker -->
```python
import asyncio
from axio_tools_docker import DockerSandbox

async def first_session() -> None:
    async with DockerSandbox(
        image="python:3.12-slim",
        name="my-sandbox",
        remove=False,
    ) as sandbox:
        await sandbox.exec("pip install requests")

async def second_session() -> None:
    # Attaches to the existing container — requests is already installed
    async with DockerSandbox(name="my-sandbox") as sandbox:
        result = await sandbox.exec(
            "python3 -c 'import requests; print(requests.__version__)'"
        )
        print(result)

asyncio.run(first_session())
asyncio.run(second_session())
```

If no container with the given name exists, a new one is created normally.

## Resource limits

Use `ulimits` to cap resource usage inside the container. A plain integer sets
soft and hard to the same value; a `(soft, hard)` tuple sets them
independently:

<!-- name: test_docker_ulimits -->
```python
from axio_tools_docker import DockerSandbox

sandbox = DockerSandbox(
    image="python:3.12-slim",
    ulimits={
        "nofile": (1024, 65536),   # open file descriptors: soft 1024, hard 65536
        "nproc": 512,              # max processes: soft=hard=512
    },
)
```

Combined with a memory cap and CPU limit this gives strong containment for
untrusted code:

<!-- name: test_docker_containment -->
```python
from axio_tools_docker import DockerSandbox

sandbox = DockerSandbox(
    image="python:3.12-slim",
    memory="256m",
    cpus="1.0",
    network=False,
    ulimits={"nofile": (256, 256), "nproc": 128},
    tmpfs={"/tmp": "size=64m,mode=1777"},
    read_only=True,
)
```

## Hardened sandbox

For maximum isolation combine `read_only`, `tmpfs`, `cap_drop`, and disabled
networking:

<!-- name: test_docker_hardened -->
```python
from axio_tools_docker import DockerSandbox

sandbox = DockerSandbox(
    image="python:3.12-slim",
    memory="256m",
    cpus="1.0",
    network=False,
    read_only=True,
    cap_drop=["ALL"],
    ulimits={"nofile": (256, 256), "nproc": 128},
    tmpfs={
        "/tmp": "size=64m,mode=1777",
        "/workspace": "size=512m",
    },
    workdir="/workspace",
)
```

With this configuration the agent can only write to `/tmp` and `/workspace`,
has no network access, no Linux capabilities, and cannot exceed the memory or
process limits.

## All parameters

<!-- name: test_docker_all_params -->
```python
from axio_tools_docker import DockerSandbox

sandbox = DockerSandbox(
    "unix:///var/run/docker.sock",   # Docker daemon URL (positional)
    image="python:3.12-slim",
    memory="512m",
    cpus="2.0",
    network=False,
    workdir="/workspace",
    volumes={"/workspace": "/tmp/host-dir"},
    env={"PYTHONPATH": "/app"},
    user="nobody",
    name="my-sandbox",
    remove=False,
    read_only=True,
    shm_size="64m",
    cap_add=["NET_ADMIN"],
    cap_drop=["ALL"],
    privileged=False,
    ulimits={"nofile": (1024, 65536), "nproc": 512},
    tmpfs={"/tmp": "size=128m,mode=1777"},
    ports={8080: 8080},
    platform="linux/amd64",
    extra_hosts={"host.docker.internal": "host-gateway"},
    devices=["/dev/net/tun"],
    dns=["8.8.8.8", "1.1.1.1"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"unix:///var/run/docker.sock"` | Docker daemon URL. Positional. |
| `image` | `str` | `"python:latest"` | Container image. Pulled automatically if not present locally. |
| `memory` | `str` | `"256m"` | Memory limit. Accepts `k`/`m`/`g` suffixes (e.g. `"512m"`, `"1g"`). |
| `cpus` | `str` | `"1.0"` | CPU limit as a decimal string. |
| `network` | `bool \| str` | `False` | Network mode. `False` → `none`. `True` → Docker default. String → explicit `NetworkMode` (e.g. `"host"`, `"bridge"`, `"my-project_default"`). |
| `workdir` | `str` | `"/workspace"` | Working directory inside the container. Relative paths in tool calls resolve against this. |
| `volumes` | `dict[str, str]` | `{}` | Bind mounts as `{container_path: host_path}`. |
| `env` | `dict[str, str]` | `{}` | Environment variables passed to all commands. |
| `user` | `str` | `""` | User to run as (e.g. `"nobody"`, `"1000"`). |
| `name` | `str` | `""` | Container name. Attaches to existing container if running; creates new one otherwise. |
| `remove` | `bool` | `True` | Remove container on exit. No effect when attached to an existing container. |
| `read_only` | `bool` | `False` | Read-only root filesystem. Combine with `tmpfs` for writable scratch space. |
| `shm_size` | `str` | `""` | `/dev/shm` size (e.g. `"64m"`). Useful for PyTorch and shared-memory IPC. |
| `cap_add` | `list[str]` | `[]` | Linux capabilities to add (e.g. `["NET_ADMIN", "SYS_PTRACE"]`). |
| `cap_drop` | `list[str]` | `[]` | Linux capabilities to drop (e.g. `["ALL"]`). |
| `privileged` | `bool` | `False` | Extended privileges — full capability set and device access. Use with care. |
| `ulimits` | `dict[str, int \| tuple[int, int]]` | `{}` | Resource limits. `{"nofile": 1024}` → soft=hard=1024. `{"nofile": (1024, 65536)}` → soft/hard split. |
| `tmpfs` | `dict[str, str]` | `{}` | Tmpfs mounts as `{path: options}` (e.g. `{"/tmp": "size=128m,mode=1777"}`). Empty string uses Docker defaults. |
| `ports` | `dict[int, int]` | `{}` | Port bindings as `{container_port: host_port}`. Only meaningful when `network != False`. |
| `platform` | `str` | `""` | Platform override (e.g. `"linux/amd64"`, `"linux/arm64"`). |
| `extra_hosts` | `dict[str, str]` | `{}` | Extra `/etc/hosts` entries as `{hostname: ip}` (e.g. `{"host.docker.internal": "host-gateway"}`). |
| `devices` | `list[str]` | `[]` | Host devices to expose. Format: `"/dev/sda"` (same container path, `rwm`), `"/dev/sda:/dev/xvda"` (custom path), `"/dev/sda:/dev/xvda:r"` (explicit permissions). |
| `dns` | `list[str]` | `[]` | DNS servers (e.g. `["8.8.8.8", "1.1.1.1"]`). Only meaningful when `network != False`. |

## Docker daemon not available

If the daemon is unreachable, `__aenter__` raises immediately with a clear message:

```
RuntimeError: Docker daemon not available at 'unix:///var/run/docker.sock': ...
```

Common causes:

- Docker Desktop is not running — start it and try again.
- Wrong socket path — pass the correct `url` or set `DOCKER_HOST`.
- Permission denied — on Linux, add your user to the `docker` group:
  ```bash
  sudo usermod -aG docker $USER
  ```

## Low-level API

`DockerSandbox` exposes the methods the built-in tools use internally. You can
call these directly for custom container interaction:

| Method | Description |
|--------|-------------|
| `await sandbox.exec(command, timeout=30, stdin=None)` | Run a shell command; returns stdout/stderr as a string. |
| `await sandbox.write_file(path, content, mode=0o644)` | Write a string to a file inside the container. |
| `await sandbox.read_file_bytes(path)` | Read a file and return raw bytes. |
| `await sandbox.get_archive(path)` | Fetch a path from the container as a `tarfile.TarFile`. |
