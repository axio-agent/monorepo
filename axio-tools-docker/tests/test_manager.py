"""Tests for SandboxManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axio_tools_docker.config import SandboxConfig
from axio_tools_docker.manager import SandboxManager


def _mock_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


@pytest.fixture()
def manager() -> SandboxManager:
    return SandboxManager()


async def test_docker_available_true() -> None:
    with patch("axio_tools_docker.manager.shutil.which", return_value="/usr/bin/docker"):
        assert SandboxManager.docker_available() is True


async def test_docker_available_false() -> None:
    with patch("axio_tools_docker.manager.shutil.which", return_value=None):
        assert SandboxManager.docker_available() is False


async def test_container_created_once(manager: SandboxManager) -> None:
    """Container is created on first call and reused."""
    create_proc = _mock_proc(stdout=b"abc123def456xyz\n")
    exec_proc = _mock_proc(stdout=b"hello\n")

    call_count = 0

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return create_proc
        return exec_proc

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        result1 = await manager.exec("echo hello")
        result2 = await manager.exec("echo hello")

    assert result1 == "hello"
    assert result2 == "hello"
    # 1 create + 2 exec = 3 total calls
    assert call_count == 3


async def test_exec_stdout_stderr(manager: SandboxManager) -> None:
    create_proc = _mock_proc(stdout=b"container123\n")
    exec_proc = _mock_proc(stdout=b"out\n", stderr=b"err\n", returncode=0)

    calls = [create_proc, exec_proc]

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return calls.pop(0)

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        result = await manager.exec("cmd")

    assert "out" in result
    assert "[stderr]" in result
    assert "err" in result


async def test_exec_nonzero_exit(manager: SandboxManager) -> None:
    create_proc = _mock_proc(stdout=b"container123\n")
    exec_proc = _mock_proc(stdout=b"", stderr=b"fail\n", returncode=1)

    calls = [create_proc, exec_proc]

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return calls.pop(0)

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        result = await manager.exec("bad_cmd")

    assert "[exit code: 1]" in result


async def test_exec_timeout(manager: SandboxManager) -> None:
    create_proc = _mock_proc(stdout=b"container123\n")
    timeout_proc = MagicMock()
    timeout_proc.communicate = AsyncMock(side_effect=TimeoutError)
    timeout_proc.kill = MagicMock()

    calls = [create_proc, timeout_proc]

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return calls.pop(0)

    with (
        patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess),
        patch("axio_tools_docker.manager.asyncio.wait_for", side_effect=TimeoutError),
    ):
        result = await manager.exec("sleep 100", timeout=5)

    assert "[timeout after 5s]" in result


async def test_write_file(manager: SandboxManager) -> None:
    create_proc = _mock_proc(stdout=b"container123\n")
    write_proc = _mock_proc()

    calls = [create_proc, write_proc]

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return calls.pop(0)

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        result = await manager.write_file("/workspace/test.py", "print('hi')")

    assert "Wrote /workspace/test.py" in result
    # Verify stdin was passed
    write_proc.communicate.assert_awaited_once_with(input=b"print('hi')")


async def test_read_file(manager: SandboxManager) -> None:
    create_proc = _mock_proc(stdout=b"container123\n")
    read_proc = _mock_proc(stdout=b"file content here")

    calls = [create_proc, read_proc]

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return calls.pop(0)

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        result = await manager.read_file("/workspace/test.py")

    assert result == "file content here"


async def test_close_removes_container(manager: SandboxManager) -> None:
    create_proc = _mock_proc(stdout=b"container123\n")
    rm_proc = _mock_proc()

    calls = [create_proc, rm_proc]
    captured_args: list[tuple[object, ...]] = []

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        captured_args.append(args)
        return calls.pop(0)

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        await manager._ensure_container()
        await manager.close()

    # Second call should be docker rm -f
    assert "rm" in captured_args[1]
    assert "-f" in captured_args[1]
    assert manager._container_id is None


async def test_close_noop_without_container(manager: SandboxManager) -> None:
    """close() is a no-op when no container was ever created."""
    await manager.close()  # Should not raise


async def test_creation_failure_raises(manager: SandboxManager) -> None:
    fail_proc = _mock_proc(stderr=b"no such image", returncode=1)

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return fail_proc

    with (
        patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess),
        pytest.raises(RuntimeError, match="Failed to create container"),
    ):
        await manager.exec("echo hi")


async def test_container_running_property(manager: SandboxManager) -> None:
    """container_running reflects internal state."""
    assert manager.container_running is False

    create_proc = _mock_proc(stdout=b"container123\n")

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return create_proc

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        await manager._ensure_container()

    assert manager.container_running is True
    manager._container_id = None
    assert manager.container_running is False


async def test_recreate_destroys_and_creates(manager: SandboxManager) -> None:
    """recreate() closes the old container and creates a new one."""
    create_proc_1 = _mock_proc(stdout=b"old_container\n")
    rm_proc = _mock_proc()
    create_proc_2 = _mock_proc(stdout=b"new_container\n")

    calls = [create_proc_1, rm_proc, create_proc_2]
    captured_args: list[tuple[object, ...]] = []

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        captured_args.append(args)
        return calls.pop(0)

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        await manager._ensure_container()
        assert manager._container_id == "old_containe"  # truncated to 12 chars
        new_id = await manager.recreate()

    assert new_id == "new_containe"
    assert manager._container_id == "new_containe"
    # Second call was docker rm -f
    assert "rm" in captured_args[1]
    assert "-f" in captured_args[1]


async def test_recreate_without_running_container(manager: SandboxManager) -> None:
    """recreate() creates a container even when none was running."""
    create_proc = _mock_proc(stdout=b"fresh_container\n")

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        return create_proc

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        new_id = await manager.recreate()

    assert new_id == "fresh_contai"
    assert manager.container_running is True


async def test_config_applied(manager: SandboxManager) -> None:
    """Container creation uses config values."""
    manager.config = SandboxConfig(image="ubuntu:22.04", memory="512m", cpus="2.0", network=True)
    create_proc = _mock_proc(stdout=b"container123\n")

    captured_args: list[tuple[object, ...]] = []

    async def mock_subprocess(*args: object, **kwargs: object) -> MagicMock:
        captured_args.append(args)
        return create_proc

    with patch("axio_tools_docker.manager.asyncio.create_subprocess_exec", side_effect=mock_subprocess):
        await manager._ensure_container()

    create_args = captured_args[0]
    assert "ubuntu:22.04" in create_args
    assert "512m" in create_args
    assert "2.0" in create_args
    # network=True means --network none should NOT be present
    assert "--network" not in create_args
