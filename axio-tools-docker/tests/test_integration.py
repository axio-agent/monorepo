"""Integration tests for DockerSandbox — require a running Docker daemon."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import aiodocker
import pytest

from axio_tools_docker.sandbox import DockerSandbox, ListFiles, PatchFile, ReadFile, RunPython

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCKER_URL = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
IMAGE = "python:3.12-alpine"


# ---------------------------------------------------------------------------
# Availability check (sync — no loop-scope concerns)
# ---------------------------------------------------------------------------


def _check_docker() -> bool:
    async def _probe() -> bool:
        try:
            async with aiodocker.Docker(url=DOCKER_URL) as client:
                await client.system.info()
            return True
        except Exception:
            return False

    return asyncio.run(_probe())


docker_available = pytest.mark.skipif(
    not _check_docker(),
    reason="Docker daemon not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def pull_image() -> None:
    """Pull the sandbox image once per session using an isolated event loop."""
    if not _check_docker():
        return

    async def _pull() -> None:
        async with aiodocker.Docker(url=DOCKER_URL) as client:
            try:
                await client.images.inspect(IMAGE)
            except aiodocker.exceptions.DockerError:
                await client.images.pull(IMAGE)

    asyncio.run(_pull())


@pytest.fixture
def container_name(request: pytest.FixtureRequest) -> str:
    """Unique container name: <test-name>-<pid>.
    Docker names allow [a-zA-Z0-9_.-]; everything else is collapsed to '-'.
    """
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", request.node.name).strip("-")
    return f"{slug}-{os.getpid()}"


@pytest.fixture
async def sandbox(pull_image: None) -> AsyncGenerator[DockerSandbox, None]:
    """Fresh container per test — avoids cross-test state and event-loop issues."""
    async with DockerSandbox(DOCKER_URL, image=IMAGE, workdir="/workspace") as sb:
        await sb.exec("mkdir -p /workspace")
        yield sb


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@docker_available
async def test_tools_exposed(sandbox: DockerSandbox) -> None:
    names = {t.name for t in sandbox.tools}
    assert names == {"shell", "write_file", "read_file", "list_files", "run_python", "patch_file"}


@docker_available
async def test_shell_basic(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("echo hello")
    assert result == "hello"


@docker_available
async def test_shell_stderr(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("echo err >&2")
    assert "[stderr]" in result
    assert "err" in result


@docker_available
async def test_shell_exit_code(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("exit 42")
    assert "[exit code: 42]" in result


@docker_available
async def test_shell_timeout(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("sleep 60", timeout=0.5)
    assert "[timeout" in result


@docker_available
async def test_shell_stdin(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("cat", stdin="hello from stdin\n")
    assert "hello from stdin" in result


@docker_available
async def test_write_and_read_file(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/test.txt", "line1\nline2\nline3\n")
    raw = await sandbox.read_file_bytes("/workspace/test.txt")
    assert raw == b"line1\nline2\nline3\n"


@docker_available
async def test_write_file_mode(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/exec.sh", "#!/bin/sh\necho ok\n", mode=0o755)
    result = await sandbox.exec("stat -c '%a' /workspace/exec.sh")
    assert "755" in result


@docker_available
async def test_read_file_line_range(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/lines.txt", "a\nb\nc\nd\ne\n")

    handler = ReadFile(filename="lines.txt", start_line=2, end_line=4)
    result = await handler(sandbox)
    assert result.strip() == "b\nc\nd"


@docker_available
async def test_read_file_line_numbers(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/numbered.txt", "foo\nbar\n")

    handler = ReadFile(filename="numbered.txt", line_numbers=True)
    result = await handler(sandbox)
    assert "1\tfoo" in result
    assert "2\tbar" in result


@docker_available
async def test_read_file_truncation(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/big.txt", "x" * 100)

    handler = ReadFile(filename="big.txt", max_chars=10)
    result = await handler(sandbox)
    assert "truncated" in result


@docker_available
async def test_list_files(sandbox: DockerSandbox) -> None:
    await sandbox.exec("mkdir -p /workspace/listing")
    await sandbox.write_file("/workspace/listing/a.py", "x")
    await sandbox.write_file("/workspace/listing/b.txt", "y")
    await sandbox.exec("mkdir -p /workspace/listing/subdir")

    handler = ListFiles(directory="/workspace/listing")
    result = await handler(sandbox)

    assert "a.py" in result
    assert "b.txt" in result
    assert "subdir/" in result
    # dirs come before files
    assert result.index("subdir/") < result.index("a.py")


@docker_available
async def test_patch_file(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/patch_me.txt", "line1\nline2\nline3\n")

    handler = PatchFile(file_path="/workspace/patch_me.txt", from_line=2, to_line=2, content="REPLACED")
    await handler(sandbox)

    raw = await sandbox.read_file_bytes("/workspace/patch_me.txt")
    assert raw == b"line1\nREPLACED\nline3\n"


@docker_available
async def test_patch_file_insert(sandbox: DockerSandbox) -> None:
    """to_line = from_line - 1 inserts without deleting."""
    await sandbox.write_file("/workspace/insert_me.txt", "line1\nline3\n")

    handler = PatchFile(file_path="/workspace/insert_me.txt", from_line=2, to_line=1, content="line2")
    await handler(sandbox)

    raw = await sandbox.read_file_bytes("/workspace/insert_me.txt")
    assert raw == b"line1\nline2\nline3\n"


@docker_available
async def test_run_python(sandbox: DockerSandbox) -> None:

    handler = RunPython(code="print(2 + 2)")
    result = await handler(sandbox)
    assert result.strip() == "4"


@docker_available
async def test_run_python_stdin(sandbox: DockerSandbox) -> None:

    handler = RunPython(code="import sys; print(sys.stdin.read().strip().upper())", stdin="hello")
    result = await handler(sandbox)
    assert result.strip() == "HELLO"


@docker_available
async def test_run_python_exit_code(sandbox: DockerSandbox) -> None:

    handler = RunPython(code="import sys; sys.exit(3)")
    result = await handler(sandbox)
    assert "[exit code: 3]" in result


@docker_available
async def test_volumes(tmp_path: Path) -> None:
    host_dir = tmp_path
    (host_dir / "from_host.txt").write_text("host content")

    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        volumes={"/mnt/host": str(host_dir)},
    ) as sb:
        raw = await sb.read_file_bytes("/mnt/host/from_host.txt")
    assert raw == b"host content"


@docker_available
async def test_name_creates_new_container(container_name: str) -> None:
    """Named container that doesn't exist yet is created fresh."""
    async with DockerSandbox(DOCKER_URL, image=IMAGE, workdir="/workspace", name=container_name, remove=True) as sb:
        cid = sb.container_id
        result = await sb.exec("echo hello")
    assert "hello" in result
    assert cid


@docker_available
async def test_name_attaches_to_existing_container(container_name: str) -> None:
    """A second sandbox with the same name reuses the running container."""
    try:
        async with DockerSandbox(
            DOCKER_URL, image=IMAGE, workdir="/workspace", name=container_name, remove=False
        ) as sb1:
            first_id = sb1.container_id
            await sb1.write_file("/workspace/marker.txt", "from-first-session")

        async with DockerSandbox(
            DOCKER_URL, image=IMAGE, workdir="/workspace", name=container_name, remove=False
        ) as sb2:
            second_id = sb2.container_id
            raw = await sb2.read_file_bytes("/workspace/marker.txt")

        assert first_id == second_id
        assert raw == b"from-first-session"
    finally:
        async with aiodocker.Docker(url=DOCKER_URL) as client:
            with contextlib.suppress(Exception):
                c = await client.containers.get(container_name)
                await c.delete(force=True)


@docker_available
async def test_name_container_id_property() -> None:
    """container_id is accessible inside the context."""
    async with DockerSandbox(DOCKER_URL, image=IMAGE, workdir="/workspace") as sb:
        cid = sb.container_id
    assert cid and len(cid) > 8


@docker_available
async def test_stopped_container_exec_raises(container_name: str) -> None:
    """If the container is stopped externally, exec raises DockerError."""
    try:
        async with DockerSandbox(
            DOCKER_URL, image=IMAGE, workdir="/workspace", name=container_name, remove=False
        ) as sb:
            async with aiodocker.Docker(url=DOCKER_URL) as client:
                c = await client.containers.get(container_name)
                await c.stop()

            with pytest.raises(aiodocker.exceptions.DockerError):
                await sb.exec("echo hello")
    finally:
        async with aiodocker.Docker(url=DOCKER_URL) as client:
            with contextlib.suppress(Exception):
                c = await client.containers.get(container_name)
                await c.delete(force=True)


@docker_available
async def test_env_vars() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        env={"MY_VAR": "hello_from_env"},
    ) as sb:
        result = await sb.exec("echo $MY_VAR")
    assert "hello_from_env" in result


@docker_available
async def test_user() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        user="nobody",
    ) as sb:
        result = await sb.exec("id -un")
    assert "nobody" in result


@docker_available
async def test_ulimits_nofile() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        ulimits={"nofile": (512, 512)},
    ) as sb:
        result = await sb.exec("ulimit -n")
    assert result.strip() == "512"


@docker_available
async def test_ulimits_nproc() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        ulimits={"nproc": 128},
    ) as sb:
        result = await sb.exec("ulimit -u")
    assert result.strip() == "128"


@docker_available
async def test_tmpfs_mounted() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        tmpfs={"/scratch": "size=8m"},
    ) as sb:
        result = await sb.exec("mount | grep /scratch")
    assert "tmpfs" in result


@docker_available
async def test_read_only_rootfs() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        read_only=True,
        tmpfs={"/workspace": "", "/tmp": ""},
    ) as sb:
        # writing to a non-tmpfs path should fail
        result = await sb.exec("echo x > /etc/nope 2>&1; echo $?")
    assert "1" in result


@docker_available
async def test_shm_size() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        shm_size="32m",
    ) as sb:
        result = await sb.exec("df -m /dev/shm | awk 'NR==2{print $2}'")
    assert int(result.strip()) >= 30


@docker_available
async def test_extra_hosts() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        network=True,
        extra_hosts={"myservice": "127.0.0.1"},
    ) as sb:
        result = await sb.exec("grep myservice /etc/hosts")
    assert "myservice" in result
    assert "127.0.0.1" in result


@docker_available
async def test_dns() -> None:
    async with DockerSandbox(
        DOCKER_URL,
        image=IMAGE,
        workdir="/workspace",
        network=True,
        dns=["1.2.3.4", "5.6.7.8"],
    ) as sb:
        result = await sb.exec("cat /etc/resolv.conf")
    assert "1.2.3.4" in result
    assert "5.6.7.8" in result
