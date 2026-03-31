"""SandboxManager: Docker container lifecycle for sandbox execution."""

from __future__ import annotations

import asyncio
import logging
import shutil

from .config import SandboxConfig

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages a session-persistent Docker container for sandboxed execution."""

    def __init__(self) -> None:
        self.config = SandboxConfig()
        self._container_id: str | None = None
        self._lock = asyncio.Lock()

    @property
    def container_running(self) -> bool:
        """Whether a sandbox container is currently active."""
        return self._container_id is not None

    @staticmethod
    def docker_available() -> bool:
        """Check whether the docker CLI is on PATH."""
        return shutil.which("docker") is not None

    async def _create_container(self) -> str:
        """Create a new detached container and return its ID."""
        cmd: list[str] = [
            "docker",
            "run",
            "-d",
            "--memory",
            self.config.memory,
            "--cpus",
            self.config.cpus,
            "-w",
            self.config.workdir,
        ]
        if not self.config.network:
            cmd.extend(["--network", "none"])
        cmd.extend([self.config.image, "sleep", "infinity"])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create container: {stderr.decode().strip()}")
        container_id = stdout.decode().strip()[:12]
        logger.info("Created sandbox container %s (image=%s)", container_id, self.config.image)
        return container_id

    async def _ensure_container(self) -> str:
        """Return the running container ID, creating one if needed."""
        if self._container_id is not None:
            return self._container_id
        async with self._lock:
            if self._container_id is not None:
                return self._container_id
            self._container_id = await self._create_container()
            return self._container_id

    async def exec(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command inside the container."""
        container_id = await self._ensure_container()
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            container_id,
            "sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"[timeout after {timeout}s]"

        output = ""
        if stdout:
            output += stdout.decode()
        if stderr:
            output += f"\n[stderr]\n{stderr.decode()}"
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
        return output.strip() or "(no output)"

    async def write_file(self, path: str, content: str) -> str:
        """Write content to a file inside the container via stdin piping."""
        container_id = await self._ensure_container()
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            container_id,
            "sh",
            "-c",
            f"cat > {path}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=content.encode())
        if proc.returncode != 0:
            return f"Error writing {path}: {stderr.decode().strip()}"
        return f"Wrote {path}"

    async def read_file(self, path: str) -> str:
        """Read a file from inside the container."""
        container_id = await self._ensure_container()
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            container_id,
            "cat",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return f"Error reading {path}: {stderr.decode().strip()}"
        return stdout.decode()

    async def recreate(self) -> str:
        """Destroy the current container and create a fresh one."""
        await self.close()
        return await self._ensure_container()

    async def close(self) -> None:
        """Stop and remove the container if running."""
        if self._container_id is None:
            return
        container_id = self._container_id
        self._container_id = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception:
            logger.warning("Failed to remove container %s", container_id, exc_info=True)
