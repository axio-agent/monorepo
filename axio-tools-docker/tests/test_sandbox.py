"""Tests for DockerSandbox."""

from __future__ import annotations

import asyncio
import io
import tarfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiodocker
import pytest

from axio_tools_docker.sandbox import DockerSandbox, parse_cpus, parse_device, parse_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tar_bytes(filename: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def make_tar_file(filename: str, content: bytes) -> tarfile.TarFile:
    return tarfile.open(fileobj=io.BytesIO(make_tar_bytes(filename, content)))


def mock_docker_factory(
    exec_messages: list[tuple[int, bytes]] | None = None,
    exec_exit_code: int = 0,
    archive_content: tarfile.TarFile | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_docker_class, mock_client, mock_container)."""
    if exec_messages is None:
        exec_messages = [(1, b"hello\n")]

    messages = list(exec_messages)

    async def read_out() -> Any:
        if messages:
            stream_type, data = messages.pop(0)
            msg = MagicMock()
            msg.stream = stream_type
            msg.data = data
            return msg
        return None

    mock_stream = MagicMock()
    mock_stream.read_out = read_out
    mock_stream.close = AsyncMock()

    mock_exec = MagicMock()
    mock_exec.start = MagicMock(return_value=mock_stream)
    mock_exec.inspect = AsyncMock(return_value={"ExitCode": exec_exit_code})

    mock_container = MagicMock()
    mock_container.start = AsyncMock()
    mock_container.delete = AsyncMock()
    mock_container.exec = AsyncMock(return_value=mock_exec)
    mock_container.put_archive = AsyncMock()
    mock_container.get_archive = AsyncMock(
        return_value=archive_content if archive_content is not None else make_tar_file("file.txt", b"content")
    )

    mock_containers = MagicMock()
    captured_config: list[dict[str, Any]] = []

    async def create_container(config: dict[str, Any], **_: Any) -> MagicMock:
        captured_config.append(config)
        return mock_container

    mock_containers.create = create_container
    # Default: no named container exists - callers can override per test.
    mock_containers.get = AsyncMock(side_effect=aiodocker.exceptions.DockerError(404, "Not found"))

    mock_images = MagicMock()
    mock_images.inspect = AsyncMock()  # image present by default - no pull needed
    mock_images.pull = AsyncMock()

    mock_system = MagicMock()
    mock_system.info = AsyncMock(return_value={})  # daemon available by default

    mock_client = MagicMock()
    mock_client.containers = mock_containers
    mock_client.images = mock_images
    mock_client.system = mock_system
    mock_client.close = AsyncMock()
    mock_client._captured_config = captured_config

    mock_docker_class = MagicMock(return_value=mock_client)

    return mock_docker_class, mock_client, mock_container


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_daemon_unavailable_raises() -> None:
    cls, client, container = mock_docker_factory()
    client.system.info = AsyncMock(side_effect=OSError("connection refused"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError, match="Docker daemon not available"):
            async with DockerSandbox():
                pass


async def test_daemon_unavailable_closes_client() -> None:
    cls, client, container = mock_docker_factory()
    client.system.info = AsyncMock(side_effect=OSError("connection refused"))
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError):
            async with DockerSandbox():
                pass
    client.close.assert_awaited_once()


async def test_context_manager_creates_and_starts() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(image="alpine:latest"):
            pass
    container.start.assert_awaited_once()


async def test_context_manager_deletes_on_exit() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    container.delete.assert_awaited_once_with(force=True)


async def test_context_manager_deletes_on_error() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        with pytest.raises(RuntimeError):
            async with DockerSandbox():
                raise RuntimeError("boom")
    container.delete.assert_awaited_once_with(force=True)


async def test_client_closed_on_exit() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    client.close.assert_awaited_once()


async def test_named_existing_container_attaches() -> None:
    """name= reuses an existing container - no create, no start."""
    cls, client, container = mock_docker_factory()
    client.containers.get = AsyncMock(return_value=container)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="my-sandbox"):
            pass
    client.containers.get.assert_awaited_once_with("my-sandbox")
    container.start.assert_not_awaited()


async def test_named_existing_container_not_deleted() -> None:
    """Attached container is never removed even with remove=True."""
    cls, client, container = mock_docker_factory()
    client.containers.get = AsyncMock(return_value=container)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="my-sandbox", remove=True):
            pass
    container.delete.assert_not_awaited()


async def test_named_missing_container_creates_new() -> None:
    """If no container with the name exists, a new one is created."""
    cls, client, container = mock_docker_factory()
    # mock_docker_factory already sets get to raise DockerError by default
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="new-sandbox"):
            pass
    container.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# tools property
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = {"shell", "write_file", "read_file", "list_files", "run_python", "patch_file"}


async def test_tools_returns_six_tools() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            tools = sb.tools
    assert len(tools) == 6
    assert {t.name for t in tools} == EXPECTED_TOOL_NAMES


async def test_tools_names_match_axio_tools_local() -> None:
    """Tool names must be identical to axio-tools-local for drop-in use."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            names = {t.name for t in sb.tools}
    assert names == EXPECTED_TOOL_NAMES


async def test_tools_raises_outside_context() -> None:
    sb = DockerSandbox()
    with pytest.raises(RuntimeError, match="async context manager"):
        _ = sb.tools


async def test_container_id_inside_context() -> None:
    cls, client, container = mock_docker_factory()
    container.id = "abc123def456"
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            cid = sb.container_id
    assert cid == "abc123def456"


async def test_container_id_raises_outside_context() -> None:
    sb = DockerSandbox()
    with pytest.raises(RuntimeError, match="async context manager"):
        _ = sb.container_id


async def test_tools_unavailable_after_exit() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            pass
    with pytest.raises(RuntimeError):
        _ = sb.tools


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------


async def test_exec_stdout() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(1, b"hello\n")])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("echo hello")
    assert result == "hello"


async def test_exec_stderr() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(2, b"oops\n")])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("bad_cmd")
    assert "[stderr]" in result
    assert "oops" in result


async def test_exec_nonzero_exit_code() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[(2, b"fail\n")], exec_exit_code=1)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("false")
    assert "[exit code: 1]" in result


async def test_exec_no_output() -> None:
    cls, client, container = mock_docker_factory(exec_messages=[])
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("true")
    assert result == "(no output)"


async def test_exec_timeout() -> None:
    async def hanging_read_out() -> Any:
        await asyncio.sleep(999)

    cls, client, container = mock_docker_factory()
    mock_stream_slow = MagicMock()
    mock_stream_slow.read_out = hanging_read_out
    mock_stream_slow.close = AsyncMock()
    mock_exec_slow = MagicMock()
    mock_exec_slow.start = MagicMock(return_value=mock_stream_slow)
    mock_exec_slow.inspect = AsyncMock(return_value={"ExitCode": 0})
    container.exec = AsyncMock(return_value=mock_exec_slow)

    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.exec("sleep 999", timeout=0.01)
    assert "[timeout after 0.01s]" in result


async def test_exec_stdin_writes_temp_file() -> None:
    """When stdin is provided, a temp file is written before the command."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            await sb.exec("cat", stdin="hello stdin")
    # put_archive must have been called at least once (for the stdin temp file)
    assert container.put_archive.await_count >= 1


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


async def test_write_file_calls_put_archive() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.write_file("/workspace/hello.py", "print('hi')")
    assert "hello.py" in result
    container.put_archive.assert_awaited()


async def test_write_file_tar_contains_content() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            await sb.write_file("/workspace/hello.py", "print('hi')")

    call_kwargs = container.put_archive.call_args
    tar_bytes: bytes = call_kwargs.kwargs.get("data") or call_kwargs.args[1]
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
        member = tar.next()
        assert member is not None
        assert member.name == "hello.py"
        f = tar.extractfile(member)
        assert f is not None
        assert f.read() == b"print('hi')"


async def test_write_file_correct_parent_dir() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            await sb.write_file("/workspace/hello.py", "x")

    call_kwargs = container.put_archive.call_args
    path_arg: str = call_kwargs.kwargs.get("path") or call_kwargs.args[0]
    assert path_arg == "/workspace"


# ---------------------------------------------------------------------------
# read_file_bytes
# ---------------------------------------------------------------------------


async def test_read_file_bytes_extracts_content() -> None:
    tar_file = make_tar_file("hello.py", b"print('hi')")
    cls, client, container = mock_docker_factory(archive_content=tar_file)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox() as sb:
            result = await sb.read_file_bytes("/workspace/hello.py")
    assert result == b"print('hi')"


# ---------------------------------------------------------------------------
# Container config
# ---------------------------------------------------------------------------


async def test_volumes_binds() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(volumes={"/container/path": "/host/path"}):
            pass
    config = client._captured_config[0]
    assert "/host/path:/container/path" in config["HostConfig"]["Binds"]


async def test_named_volumes_binds() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(named_volumes={"/data": "myvolume"}):
            pass
    config = client._captured_config[0]
    assert "myvolume:/data" in config["HostConfig"]["Binds"]


async def test_named_volumes_combined_with_bind_mounts() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(
            volumes={"/app": "/host/app"},
            named_volumes={"/data": "myvolume"},
        ):
            pass
    config = client._captured_config[0]
    binds = config["HostConfig"]["Binds"]
    assert "/host/app:/app" in binds
    assert "myvolume:/data" in binds


async def test_volumes_remove_deletes_named_volumes_on_exit() -> None:
    cls, client, container = mock_docker_factory()
    mock_volume = MagicMock()
    mock_volume.delete = AsyncMock()
    client.volumes = MagicMock()
    client.volumes.get = AsyncMock(return_value=mock_volume)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(named_volumes={"/data": "myvolume"}, volumes_remove=True):
            pass
    client.volumes.get.assert_awaited_once_with("myvolume")
    mock_volume.delete.assert_awaited_once()


async def test_volumes_remove_not_called_when_attached() -> None:
    cls, client, container = mock_docker_factory()
    client.containers.get = AsyncMock(return_value=container)
    mock_volume = MagicMock()
    mock_volume.delete = AsyncMock()
    client.volumes = MagicMock()
    client.volumes.get = AsyncMock(return_value=mock_volume)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(
            name="existing",
            named_volumes={"/data": "myvolume"},
            volumes_remove=True,
        ):
            pass
    mock_volume.delete.assert_not_awaited()


async def test_volumes_remove_false_does_not_delete() -> None:
    cls, client, container = mock_docker_factory()
    mock_volume = MagicMock()
    mock_volume.delete = AsyncMock()
    client.volumes = MagicMock()
    client.volumes.get = AsyncMock(return_value=mock_volume)
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(named_volumes={"/data": "myvolume"}, volumes_remove=False):
            pass
    mock_volume.delete.assert_not_awaited()


async def test_network_mode_none_when_disabled() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network=False):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NetworkMode"] == "none"


async def test_network_mode_absent_when_true() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network=True):
            pass
    config = client._captured_config[0]
    assert "NetworkMode" not in config["HostConfig"]


async def test_network_mode_string() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network="my-project_default"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NetworkMode"] == "my-project_default"


async def test_network_mode_host() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(network="host"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NetworkMode"] == "host"


async def test_init_always_true() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Init"] is True


async def test_memory_parsed() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(memory="256m"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Memory"] == 256 * 1024 * 1024


async def test_cpus_as_nanocpus() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(cpus="1.0"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["NanoCPUs"] == 1_000_000_000


async def test_custom_url() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls) as docker_cls:
        async with DockerSandbox("tcp://localhost:2375"):
            pass
    docker_cls.assert_called_once_with(url="tcp://localhost:2375")


async def test_env_vars() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(env={"FOO": "bar", "BAZ": "qux"}):
            pass
    config = client._captured_config[0]
    assert "FOO=bar" in config["Env"]
    assert "BAZ=qux" in config["Env"]


async def test_no_env_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert config["Env"] == []


async def test_user_set() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(user="nobody"):
            pass
    config = client._captured_config[0]
    assert config["User"] == "nobody"


async def test_user_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "User" not in config


async def test_container_name_passed() -> None:
    cls, client, container = mock_docker_factory()

    captured_kwargs: list[dict[str, Any]] = []

    original_create = client.containers.create

    async def create_with_capture(**kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return await original_create(**kwargs)

    client.containers.create = create_with_capture

    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(name="my-sandbox"):
            pass
    assert any(kw.get("name") == "my-sandbox" for kw in captured_kwargs)


async def test_no_name_by_default() -> None:
    cls, client, container = mock_docker_factory()

    captured_kwargs: list[dict[str, Any]] = []

    original_create = client.containers.create

    async def create_with_capture(**kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return await original_create(**kwargs)

    client.containers.create = create_with_capture

    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    assert all("name" not in kw for kw in captured_kwargs)


async def test_remove_true_deletes_container() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(remove=True):
            pass
    container.delete.assert_awaited_once_with(force=True)


async def test_remove_false_keeps_container() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(remove=False):
            pass
    container.delete.assert_not_awaited()


# ---------------------------------------------------------------------------
# read_only
# ---------------------------------------------------------------------------


async def test_read_only_sets_flag() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(read_only=True):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["ReadonlyRootfs"] is True


async def test_read_only_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "ReadonlyRootfs" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# shm_size
# ---------------------------------------------------------------------------


async def test_shm_size_parsed() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(shm_size="64m"):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["ShmSize"] == 64 * 1024 * 1024


async def test_shm_size_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "ShmSize" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# cap_add / cap_drop
# ---------------------------------------------------------------------------


async def test_cap_add() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(cap_add=["NET_ADMIN", "SYS_PTRACE"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["CapAdd"] == ["NET_ADMIN", "SYS_PTRACE"]


async def test_cap_drop() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(cap_drop=["ALL"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["CapDrop"] == ["ALL"]


async def test_cap_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "CapAdd" not in config["HostConfig"]
    assert "CapDrop" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# privileged
# ---------------------------------------------------------------------------


async def test_privileged() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(privileged=True):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Privileged"] is True


async def test_privileged_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Privileged" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# ulimits
# ---------------------------------------------------------------------------


async def test_ulimits_single_value() -> None:
    """A plain int means soft == hard."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ulimits={"nofile": 1024}):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Ulimits"] == [{"Name": "nofile", "Soft": 1024, "Hard": 1024}]


async def test_ulimits_tuple() -> None:
    """A (soft, hard) tuple sets them independently."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ulimits={"nofile": (1024, 65536)}):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Ulimits"] == [{"Name": "nofile", "Soft": 1024, "Hard": 65536}]


async def test_ulimits_multiple() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ulimits={"nofile": (1024, 65536), "nproc": 512}):
            pass
    config = client._captured_config[0]
    entries = {e["Name"]: e for e in config["HostConfig"]["Ulimits"]}
    assert entries["nofile"] == {"Name": "nofile", "Soft": 1024, "Hard": 65536}
    assert entries["nproc"] == {"Name": "nproc", "Soft": 512, "Hard": 512}


async def test_ulimits_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Ulimits" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# tmpfs
# ---------------------------------------------------------------------------


async def test_tmpfs() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(tmpfs={"/tmp": "size=128m,mode=1777"}):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Tmpfs"] == {"/tmp": "size=128m,mode=1777"}


async def test_tmpfs_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Tmpfs" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# ports
# ---------------------------------------------------------------------------


async def test_ports_bindings() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(ports={8080: 8080, 5432: 15432}):
            pass
    config = client._captured_config[0]
    bindings = config["HostConfig"]["PortBindings"]
    assert bindings["8080/tcp"] == [{"HostPort": "8080"}]
    assert bindings["5432/tcp"] == [{"HostPort": "15432"}]
    exposed = config["ExposedPorts"]
    assert "8080/tcp" in exposed
    assert "5432/tcp" in exposed


async def test_ports_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "PortBindings" not in config["HostConfig"]
    assert "ExposedPorts" not in config


# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------


async def test_platform() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(platform="linux/amd64"):
            pass
    config = client._captured_config[0]
    assert config["Platform"] == "linux/amd64"


async def test_platform_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Platform" not in config


# ---------------------------------------------------------------------------
# extra_hosts
# ---------------------------------------------------------------------------


async def test_extra_hosts() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(extra_hosts={"myhost": "1.2.3.4", "other": "5.6.7.8"}):
            pass
    config = client._captured_config[0]
    assert "myhost:1.2.3.4" in config["HostConfig"]["ExtraHosts"]
    assert "other:5.6.7.8" in config["HostConfig"]["ExtraHosts"]


async def test_extra_hosts_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "ExtraHosts" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------


async def test_devices_full_format() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(devices=["/dev/sda:/dev/xvda:r"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Devices"] == [
        {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/xvda", "CgroupPermissions": "r"}
    ]


async def test_devices_short_format() -> None:
    """Just the host path - maps to same container path with rwm."""
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(devices=["/dev/net/tun"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Devices"] == [
        {"PathOnHost": "/dev/net/tun", "PathInContainer": "/dev/net/tun", "CgroupPermissions": "rwm"}
    ]


async def test_devices_multiple() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(devices=["/dev/sda", "/dev/sdb:/dev/xvdb:rw"]):
            pass
    config = client._captured_config[0]
    assert len(config["HostConfig"]["Devices"]) == 2


async def test_devices_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Devices" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# dns
# ---------------------------------------------------------------------------


async def test_dns() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox(dns=["8.8.8.8", "1.1.1.1"]):
            pass
    config = client._captured_config[0]
    assert config["HostConfig"]["Dns"] == ["8.8.8.8", "1.1.1.1"]


async def test_dns_absent_by_default() -> None:
    cls, client, container = mock_docker_factory()
    with patch("axio_tools_docker.sandbox.aiodocker.Docker", cls):
        async with DockerSandbox():
            pass
    config = client._captured_config[0]
    assert "Dns" not in config["HostConfig"]


# ---------------------------------------------------------------------------
# parse_device unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "/dev/sda",
            {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/sda", "CgroupPermissions": "rwm"},
        ),
        (
            "/dev/sda:/dev/xvda",
            {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/xvda", "CgroupPermissions": "rwm"},
        ),
        (
            "/dev/sda:/dev/xvda:r",
            {"PathOnHost": "/dev/sda", "PathInContainer": "/dev/xvda", "CgroupPermissions": "r"},
        ),
    ],
)
def test_parse_device(value: str, expected: dict[str, str]) -> None:
    assert parse_device(value) == expected


# ---------------------------------------------------------------------------
# parse_memory / parse_cpus unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("256m", 256 * 1024**2),
        ("1g", 1024**3),
        ("512k", 512 * 1024),
        ("1048576", 1048576),
    ],
)
def test_parse_memory(value: str, expected: int) -> None:
    assert parse_memory(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.0", 1_000_000_000),
        ("0.5", 500_000_000),
        ("2.0", 2_000_000_000),
    ],
)
def test_parse_cpus(value: str, expected: int) -> None:
    assert parse_cpus(value) == expected
