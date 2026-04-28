"""Integration tests for DockerSandbox - require a running Docker daemon."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import aiodocker
import pytest

from axio_tools_docker.sandbox import DockerSandbox





@pytest.fixture(scope="session", autouse=True)
async def image(docker: str) -> str:
    """Pull the sandbox image once per session using an isolated event loop."""
    image = "python:3.12-alpine"
    async with aiodocker.Docker(url=docker) as client:
        try:
            await client.images.inspect(image)
        except aiodocker.exceptions.DockerError:
            await client.images.pull(image)
        return image


@pytest.fixture
def container_name(request: pytest.FixtureRequest) -> str:
    """Unique container name: <test-name>-<pid>.
    Docker names allow [a-zA-Z0-9_.-]; everything else is collapsed to '-'.
    """
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", request.node.name).strip("-")
    return f"{slug}-{os.getpid()}"


@pytest.fixture
async def sandbox(docker: str, image: str) -> AsyncGenerator[DockerSandbox, None]:
    """Fresh container per test - avoids cross-test state and event-loop issues."""
    async with DockerSandbox(docker, image=image, workdir="/workspace") as sb:
        await sb.exec("mkdir -p /workspace")
        yield sb


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


async def test_tools_exposed(sandbox: DockerSandbox) -> None:
    names = {t.name for t in sandbox.tools}
    assert names == {"shell", "write_file", "read_file", "list_files", "run_python", "patch_file"}


async def test_shell_basic(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("echo hello")
    assert result == "hello"


async def test_shell_stderr(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("echo err >&2")
    assert "[stderr]" in result
    assert "err" in result

async def test_shell_exit_code(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("exit 42")
    assert "[exit code: 42]" in result

async def test_shell_timeout(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("sleep 60", timeout=0.5)
    assert "[timeout" in result

async def test_shell_stdin(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("cat", stdin="hello from stdin\n")
    assert "hello from stdin" in result

async def test_write_and_read_file(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/test.txt", "line1\nline2\nline3\n")
    raw = await sandbox.read_file_bytes("/workspace/test.txt")
    assert raw == b"line1\nline2\nline3\n"


async def test_write_file_mode(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/exec.sh", "#!/bin/sh\necho ok\n", mode=0o755)
    result = await sandbox.exec("stat -c '%a' /workspace/exec.sh")
    assert "755" in result


async def test_read_file_line_range(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/lines.txt", "a\nb\nc\nd\ne\n")
    tool = next(t for t in sandbox.tools if t.name == "read_file")
    result = await tool(filename="lines.txt", start_line=2, end_line=4)
    assert result.strip() == "b\nc\nd"


async def test_read_file_line_numbers(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/numbered.txt", "foo\nbar\n")
    tool = next(t for t in sandbox.tools if t.name == "read_file")
    result = await tool(filename="numbered.txt", line_numbers=True)
    assert "1\tfoo" in result
    assert "2\tbar" in result


async def test_read_file_truncation(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/big.txt", "x" * 100)
    tool = next(t for t in sandbox.tools if t.name == "read_file")
    result = await tool(filename="big.txt", max_chars=10)
    assert "truncated" in result


async def test_list_files(sandbox: DockerSandbox) -> None:
    await sandbox.exec("mkdir -p /workspace/listing")
    await sandbox.write_file("/workspace/listing/a.py", "x")
    await sandbox.write_file("/workspace/listing/b.txt", "y")
    await sandbox.exec("mkdir -p /workspace/listing/subdir")

    tool = next(t for t in sandbox.tools if t.name == "list_files")
    result = await tool(directory="/workspace/listing")

    assert "a.py" in result
    assert "b.txt" in result
    assert "subdir/" in result
    assert result.index("subdir/") < result.index("a.py")


async def test_patch_file(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("/workspace/patch_me.txt", "line1\nline2\nline3\n")
    tool = next(t for t in sandbox.tools if t.name == "patch_file")
    await tool(file_path="/workspace/patch_me.txt", from_line=2, to_line=2, content="REPLACED")
    raw = await sandbox.read_file_bytes("/workspace/patch_me.txt")
    assert raw == b"line1\nREPLACED\nline3\n"


async def test_patch_file_insert(sandbox: DockerSandbox) -> None:
    """to_line = from_line - 1 inserts without deleting."""
    await sandbox.write_file("/workspace/insert_me.txt", "line1\nline3\n")
    tool = next(t for t in sandbox.tools if t.name == "patch_file")
    await tool(file_path="/workspace/insert_me.txt", from_line=2, to_line=1, content="line2")
    raw = await sandbox.read_file_bytes("/workspace/insert_me.txt")
    assert raw == b"line1\nline2\nline3\n"


async def test_run_python(sandbox: DockerSandbox) -> None:
    tool = next(t for t in sandbox.tools if t.name == "run_python")
    result = await tool(code="print(2 + 2)")
    assert result.strip() == "4"


async def test_run_python_stdin(sandbox: DockerSandbox) -> None:
    tool = next(t for t in sandbox.tools if t.name == "run_python")
    result = await tool(code="import sys; print(sys.stdin.read().strip().upper())", stdin="hello")
    assert result.strip() == "HELLO"


async def test_run_python_exit_code(sandbox: DockerSandbox) -> None:
    tool = next(t for t in sandbox.tools if t.name == "run_python")
    result = await tool(code="import sys; sys.exit(3)")
    assert "[exit code: 3]" in result


async def test_volumes(docker: str, image: str, tmp_path: Path) -> None:
    host_dir = tmp_path
    (host_dir / "from_host.txt").write_text("host content")

    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        volumes={"/mnt/host": str(host_dir)},
    ) as sb:
        raw = await sb.read_file_bytes("/mnt/host/from_host.txt")
    assert raw == b"host content"


async def test_name_creates_new_container(docker: str, image: str, container_name: str) -> None:
    """Named container that doesn't exist yet is created fresh."""
    async with DockerSandbox(docker, image=image, workdir="/workspace", name=container_name, remove=True) as sb:
        cid = sb.container_id
        result = await sb.exec("echo hello")
    assert "hello" in result
    assert cid


async def test_name_attaches_to_existing_container(docker: str, image: str, container_name: str) -> None:
    """A second sandbox with the same name reuses the running container."""
    try:
        async with DockerSandbox(
            docker, image=image, workdir="/workspace", name=container_name, remove=False
        ) as sb1:
            first_id = sb1.container_id
            await sb1.write_file("/workspace/marker.txt", "from-first-session")

        async with DockerSandbox(
            docker, image=image, workdir="/workspace", name=container_name, remove=False
        ) as sb2:
            second_id = sb2.container_id
            raw = await sb2.read_file_bytes("/workspace/marker.txt")

        assert first_id == second_id
        assert raw == b"from-first-session"
    finally:
        async with aiodocker.Docker(url=docker) as client:
            with contextlib.suppress(Exception):
                c = await client.containers.get(container_name)
                await c.delete(force=True)


async def test_name_container_id_property(docker: str, image: str) -> None:
    """container_id is accessible inside the context."""
    async with DockerSandbox(docker, image=image, workdir="/workspace") as sb:
        cid = sb.container_id
    assert cid and len(cid) > 8


async def test_stopped_container_exec_raises(docker: str, image: str, container_name: str) -> None:
    """If the container is stopped externally, exec raises DockerError."""
    try:
        async with DockerSandbox(
            docker, image=image, workdir="/workspace", name=container_name, remove=False
        ) as sb:
            async with aiodocker.Docker(url=docker) as client:
                c = await client.containers.get(container_name)
                await c.stop()

            with pytest.raises(aiodocker.exceptions.DockerError):
                await sb.exec("echo hello")
    finally:
        async with aiodocker.Docker(url=docker) as client:
            with contextlib.suppress(Exception):
                c = await client.containers.get(container_name)
                await c.delete(force=True)


async def test_env_vars(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        env={"MY_VAR": "hello_from_env"},
    ) as sb:
        result = await sb.exec("echo $MY_VAR")
    assert "hello_from_env" in result


async def test_user(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        user="nobody",
    ) as sb:
        result = await sb.exec("id -un")
    assert "nobody" in result


async def test_ulimits_nofile(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        ulimits={"nofile": (512, 512)},
    ) as sb:
        result = await sb.exec("ulimit -n")
    assert result.strip() == "512"


async def test_ulimits_nproc(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        ulimits={"nproc": 128},
    ) as sb:
        result = await sb.exec("ulimit -u")
    assert result.strip() == "128"


async def test_tmpfs_mounted(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        tmpfs={"/scratch": "size=8m"},
    ) as sb:
        result = await sb.exec("mount | grep /scratch")
    assert "tmpfs" in result


async def test_read_only_rootfs(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        read_only=True,
        tmpfs={"/workspace": "", "/tmp": ""},
    ) as sb:
        # writing to a non-tmpfs path should fail
        result = await sb.exec("echo x > /etc/nope 2>&1; echo $?")
    assert "1" in result


async def test_shm_size(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        shm_size="32m",
    ) as sb:
        result = await sb.exec("df -m /dev/shm | awk 'NR==2{print $2}'")
    assert int(result.strip()) >= 30


async def test_extra_hosts(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        network=True,
        extra_hosts={"myservice": "127.0.0.1"},
    ) as sb:
        result = await sb.exec("grep myservice /etc/hosts")
    assert "myservice" in result
    assert "127.0.0.1" in result


async def test_dns(docker: str, image: str) -> None:
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        network=True,
        dns=["1.2.3.4", "5.6.7.8"],
    ) as sb:
        result = await sb.exec("cat /etc/resolv.conf")
    assert "1.2.3.4" in result
    assert "5.6.7.8" in result


async def test_named_volume_persists_across_containers(docker: str, image: str) -> None:
    """Data written to a named volume survives container removal and is visible in a new container."""
    vol_name = f"axio-test-vol-{__import__('uuid').uuid4().hex[:8]}"
    try:
        async with DockerSandbox(
            docker,
            image=image,
            workdir="/workspace",
            named_volumes={"/data": vol_name},
        ) as sb:
            await sb.write_file("/data/hello.txt", "persistent-content")

        async with DockerSandbox(
            docker,
            image=image,
            workdir="/workspace",
            named_volumes={"/data": vol_name},
            volumes_remove=True,
        ) as sb2:
            raw = await sb2.read_file_bytes("/data/hello.txt")

        assert raw == b"persistent-content"
    finally:
        async with aiodocker.Docker(url=docker) as client:
            with contextlib.suppress(Exception):
                vol = await client.volumes.get(vol_name)  # type: ignore[no-untyped-call]
                await vol.delete()


async def test_volumes_remove_cleans_up(docker: str, image: str) -> None:
    """volumes_remove=True deletes the named volume on exit."""
    vol_name = f"axio-test-vol-{__import__('uuid').uuid4().hex[:8]}"
    async with DockerSandbox(
        docker,
        image=image,
        workdir="/workspace",
        named_volumes={"/data": vol_name},
        volumes_remove=True,
    ) as sb:
        await sb.exec("echo hello > /data/x.txt")

    async with aiodocker.Docker(url=docker) as client:
        with pytest.raises(aiodocker.exceptions.DockerError):
            await client.volumes.get(vol_name)  # type: ignore[no-untyped-call]
