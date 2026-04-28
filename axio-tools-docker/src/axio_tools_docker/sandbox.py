"""DockerSandbox: async context manager for sandboxed Docker execution."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import shlex
import stat as stat_module
import tarfile
import uuid
from datetime import datetime
from typing import Any

import aiodocker
from axio.tool import CONTEXT, Tool

logger = logging.getLogger(__name__)


def parse_memory(s: str) -> int:
    """Parse human-readable memory string to bytes: "256m" → 268435456."""
    units = {"k": 1024, "m": 1024**2, "g": 1024**3}
    s = s.lower().strip()
    if s[-1] in units:
        return int(s[:-1]) * units[s[-1]]
    return int(s)


def parse_cpus(s: str) -> int:
    """Parse CPU string to NanoCPUs: "1.0" → 1_000_000_000."""
    return int(float(s) * 1_000_000_000)


def _resolve_path(workdir: str, path: str) -> str:
    """Resolve a possibly-relative path against the container workdir."""
    if os.path.isabs(path):
        return path
    return os.path.join(workdir, path)


def parse_device(s: str) -> dict[str, str]:
    """Parse a device string into a Docker device mapping dict.

    Accepted formats (mirrors ``docker run --device``):
    - ``/dev/sda`` → host=/dev/sda, container=/dev/sda, perms=rwm
    - ``/dev/sda:/dev/xvda`` → host=/dev/sda, container=/dev/xvda, perms=rwm
    - ``/dev/sda:/dev/xvda:r`` → explicit permissions
    """
    parts = s.split(":")
    host = parts[0]
    container = parts[1] if len(parts) > 1 else host
    perms = parts[2] if len(parts) > 2 else "rwm"
    return {"PathOnHost": host, "PathInContainer": container, "CgroupPermissions": perms}


# ---------------------------------------------------------------------------
# Tool handlers - plain async functions, context via CONTEXT.get()
# ---------------------------------------------------------------------------


async def shell(command: str, timeout: float = 5, cwd: str = ".", stdin: str | None = None) -> str:
    """Run a shell command and return combined stdout/stderr. Use for git,
    build tools, grep, tests, or any CLI operation. Non-zero exit codes
    are reported. Optionally pass stdin data for commands that read from
    standard input. Prefer short timeouts and avoid interactive commands."""
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, cwd)
    cmd = f"cd {shlex.quote(resolved)} && {command}"
    return await sandbox.exec(cmd, timeout=timeout, stdin=stdin)


async def write_file(file_path: str, content: str, mode: int = 0o644) -> str:
    """Create or overwrite a file with the given content. Parent directories
    are created automatically. Use this for new files or full rewrites.
    For partial edits prefer patch_file instead."""
    sandbox: DockerSandbox = CONTEXT.get()
    path = _resolve_path(sandbox.workdir, file_path)
    return await sandbox.write_file(path, content, mode=mode)


async def read_file(
    filename: str,
    max_chars: int = 32768,
    binary_as_hex: bool = True,
    start_line: int | None = None,
    end_line: int | None = None,
    line_numbers: bool = False,
) -> str:
    """Read file contents. Returns text for text files, hex for binaries.
    Lines are 1-indexed: start_line=1 is the first line, end_line=3 includes
    line 3. Pass line_numbers=True to prefix each line with its 1-based line
    number (tab-separated) - required before calling patch_file. Large files
    are truncated to max_chars. Always read the file before editing it with
    write_file or patch_file."""
    sandbox: DockerSandbox = CONTEXT.get()
    path = _resolve_path(sandbox.workdir, filename)
    raw = await sandbox.read_file_bytes(path)
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        if binary_as_hex:
            return "Encoded binary data HEX: " + raw[:max_chars].hex()
        raise
    all_lines = text.splitlines(keepends=True)
    start = 0 if start_line is None else start_line - 1
    end = len(all_lines) if end_line is None else end_line
    selected = all_lines[start:end]
    if line_numbers:
        result = "".join(f"{start + 1 + i}\t{line}" for i, line in enumerate(selected))
    else:
        result = "".join(selected)
    if len(result) > max_chars:
        return result[:max_chars] + "\n...[truncated]"
    return result


async def list_files(directory: str = ".") -> str:
    """List files and directories. Shows permissions, size, modification time,
    and name for each entry. Directories are listed first and marked with
    a trailing slash. Use this to explore the project structure before
    reading or editing files."""
    sandbox: DockerSandbox = CONTEXT.get()
    path = _resolve_path(sandbox.workdir, directory)
    tar = await sandbox.get_archive(path)

    members = tar.getmembers()
    if not members:
        return "(empty directory)"

    prefix = members[0].name.rstrip("/") + "/"
    entries: list[tarfile.TarInfo] = []
    for member in members:
        if not member.name.startswith(prefix):
            continue
        rel = member.name[len(prefix) :]
        if not rel or "/" in rel.rstrip("/"):
            continue
        entries.append(member)

    entries.sort(key=lambda m: (not m.isdir(), m.name))
    if not entries:
        return "(empty directory)"

    lines: list[str] = []
    for m in entries:
        full_mode = m.mode
        if m.isdir():
            full_mode |= stat_module.S_IFDIR
        elif m.issym():
            full_mode |= stat_module.S_IFLNK
        else:
            full_mode |= stat_module.S_IFREG
        mode_str = stat_module.filemode(full_mode)
        mtime = datetime.fromtimestamp(m.mtime).strftime("%b %d %H:%M")
        base = m.name.rstrip("/").split("/")[-1] + ("/" if m.isdir() else "")
        lines.append(f"{mode_str} {m.size:>8} {mtime} {base}")
    return "\n".join(lines)


async def run_python(code: str, cwd: str = ".", timeout: float = 5, stdin: str | None = None) -> str:
    """Run a Python code snippet in a subprocess and return stdout/stderr.
    The code is written to a temp file and executed with the current
    interpreter. Use for calculations, data processing, or testing
    small scripts. Optionally pass stdin data. Non-zero exit codes
    and tracebacks are returned as-is."""
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, cwd)
    tmp = f"/tmp/.axio_{uuid.uuid4().hex}.py"
    await sandbox.write_file(tmp, code)
    cmd = f"cd {shlex.quote(resolved)} && python3 {tmp}; _rc=$?; rm -f {tmp}; exit $_rc"
    return await sandbox.exec(cmd, timeout=timeout, stdin=stdin)


async def patch_file(file_path: str, from_line: int, to_line: int, content: str, mode: int = 0o644) -> str:
    """Replace a range of lines in an existing file. Lines are 1-indexed:
    from_line and to_line are both inclusive (from_line=2, to_line=4 replaces
    lines 2, 3, 4). To insert without deleting, set to_line = from_line - 1.
    Always read the file first with line_numbers=True to get correct line numbers.
    Use this for surgical edits instead of rewriting the whole file with
    write_file."""
    sandbox: DockerSandbox = CONTEXT.get()
    path = _resolve_path(sandbox.workdir, file_path)
    raw = await sandbox.read_file_bytes(path)
    lines = raw.decode().splitlines(keepends=True)
    content_lines = content.splitlines(keepends=True)
    if content_lines and not content_lines[-1].endswith("\n"):
        content_lines[-1] += "\n"
    new_lines = lines[: from_line - 1] + content_lines + lines[to_line:]
    await sandbox.write_file(path, "".join(new_lines), mode=mode)
    return f"{len(new_lines)} lines written to {path}"


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------


class DockerSandbox:
    """Async context manager that provides a sandboxed Docker container with axio tools."""

    def __init__(
        self,
        url: str = "unix:///var/run/docker.sock",
        *,
        image: str = "python:latest",
        memory: str = "256m",
        cpus: str = "1.0",
        network: bool | str = False,
        workdir: str = "/workspace",
        volumes: dict[str, str] | None = None,
        named_volumes: dict[str, str] | None = None,
        volumes_remove: bool = False,
        env: dict[str, str] | None = None,
        user: str = "",
        name: str = "",
        remove: bool = True,
        read_only: bool = False,
        shm_size: str = "",
        cap_add: list[str] | None = None,
        cap_drop: list[str] | None = None,
        privileged: bool = False,
        ulimits: dict[str, int | tuple[int, int]] | None = None,
        tmpfs: dict[str, str] | None = None,
        ports: dict[int, int] | None = None,
        platform: str = "",
        extra_hosts: dict[str, str] | None = None,
        devices: list[str] | None = None,
        dns: list[str] | None = None,
    ) -> None:
        """Create a DockerSandbox.

        Args:
            url: Docker daemon URL (unix socket or TCP).
            image: Container image to use.
            memory: Memory limit, e.g. "256m", "1g".
            cpus: CPU limit, e.g. "1.0".
            network: Network mode. ``False`` disables networking entirely
                (``NetworkMode: none``). ``True`` uses the Docker default.
                A string sets ``NetworkMode`` explicitly, e.g. ``"host"``,
                ``"bridge"``, or a named network like ``"my-project_default"``.
            workdir: Working directory inside the container.
            volumes: Mapping of {container_path: host_path} bind mounts.
            named_volumes: Mapping of {container_path: volume_name} named Docker volumes.
                Docker creates the volume automatically if it does not exist. Named volumes
                persist across container restarts and can be shared between sandbox sessions.
            volumes_remove: Remove named volumes on exit. Has no effect when attaching to
                an existing container (``name=`` reuse) or when ``named_volumes`` is empty.
            env: Environment variables passed to all commands, e.g. {"PYTHONPATH": "/app"}.
            user: User to run as inside the container, e.g. "1000" or "nobody".
            name: Container name. If a container with this name already exists and
                is running, the sandbox attaches to it instead of creating a new one
                and will not remove it on exit. If no container exists, a new one is
                created (and removed on exit if ``remove=True``).
            remove: Remove the container on exit (default: True). Has no effect when
                attaching to an existing container.
            read_only: Mount the container's root filesystem as read-only.
            shm_size: Size of ``/dev/shm``, e.g. ``"64m"``, ``"1g"``.
            cap_add: Linux capabilities to add, e.g. ``["NET_ADMIN", "SYS_PTRACE"]``.
            cap_drop: Linux capabilities to drop, e.g. ``["ALL"]``.
            privileged: Give extended privileges to the container (implies full
                capability set and device access). Use with care.
            ulimits: Resource limits as ``{name: value}`` or ``{name: (soft, hard)}``.
                A single integer sets soft == hard. Examples: ``{"nofile": 1024}``,
                ``{"nofile": (1024, 65536), "nproc": 512}``.
            tmpfs: Tmpfs mounts as ``{path: options}``, e.g.
                ``{"/tmp": "size=128m,mode=1777"}``. An empty string for options
                uses Docker defaults.
            ports: Port bindings as ``{container_port: host_port}``, e.g.
                ``{8080: 8080}``. Only meaningful when ``network`` is not ``False``.
            platform: Platform string for the container image, e.g.
                ``"linux/amd64"`` or ``"linux/arm64"``.
            extra_hosts: Additional ``/etc/hosts`` entries as ``{hostname: ip}``,
                e.g. ``{"host.docker.internal": "host-gateway"}``.
            devices: Host devices to expose inside the container. Each entry
                follows the ``docker run --device`` format:
                ``"/dev/sda"`` (maps to same path, permissions ``rwm``),
                ``"/dev/sda:/dev/xvda"`` (custom container path),
                ``"/dev/sda:/dev/xvda:r"`` (read-only).
            dns: DNS servers to use inside the container, e.g.
                ``["8.8.8.8", "1.1.1.1"]``.
        """
        self.url = url
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network: bool | str = network
        self.workdir = workdir
        self.volumes: dict[str, str] = volumes or {}
        self.named_volumes: dict[str, str] = named_volumes or {}
        self.volumes_remove = volumes_remove
        self.env: dict[str, str] = env or {}
        self.user = user
        self.name = name
        self.remove = remove
        self.read_only = read_only
        self.shm_size = shm_size
        self.cap_add: list[str] = cap_add or []
        self.cap_drop: list[str] = cap_drop or []
        self.privileged = privileged
        self.ulimits: dict[str, int | tuple[int, int]] = ulimits or {}
        self.tmpfs: dict[str, str] = tmpfs or {}
        self.ports: dict[int, int] = ports or {}
        self.platform = platform
        self.extra_hosts: dict[str, str] = extra_hosts or {}
        self.devices: list[str] = devices or []
        self.dns: list[str] = dns or []
        self._client: aiodocker.Docker | None = None
        self._container: aiodocker.containers.DockerContainer | None = None
        self._attached: bool = False  # True when we reused an existing container
        self._tools: list[Tool[Any]] | None = None

    async def __aenter__(self) -> DockerSandbox:
        self._client = aiodocker.Docker(url=self.url)
        try:
            await self._client.system.info()
        except Exception as exc:
            await self._client.close()
            self._client = None
            raise RuntimeError(f"Docker daemon not available at {self.url!r}: {exc}") from exc

        if self.name:
            try:
                self._container = await self._client.containers.get(self.name)
                self._attached = True
                logger.info("Attached to existing container (name=%s)", self.name)
            except aiodocker.exceptions.DockerError:
                self._attached = False

        if not self._attached:
            await self._ensure_image()
            binds = [f"{host}:{container}" for container, host in self.volumes.items()]
            binds += [f"{vol}:{path}" for path, vol in self.named_volumes.items()]
            host_config: dict[str, Any] = {
                "Init": True,
                "Memory": parse_memory(self.memory),
                "NanoCPUs": parse_cpus(self.cpus),
                "Binds": binds,
            }
            if self.network is False:
                host_config["NetworkMode"] = "none"
            elif isinstance(self.network, str):
                host_config["NetworkMode"] = self.network
            if self.read_only:
                host_config["ReadonlyRootfs"] = True
            if self.shm_size:
                host_config["ShmSize"] = parse_memory(self.shm_size)
            if self.cap_add:
                host_config["CapAdd"] = self.cap_add
            if self.cap_drop:
                host_config["CapDrop"] = self.cap_drop
            if self.privileged:
                host_config["Privileged"] = True
            if self.ulimits:
                host_config["Ulimits"] = [
                    {
                        "Name": limit_name,
                        "Soft": val if isinstance(val, int) else val[0],
                        "Hard": val if isinstance(val, int) else val[1],
                    }
                    for limit_name, val in self.ulimits.items()
                ]
            if self.tmpfs:
                host_config["Tmpfs"] = self.tmpfs
            if self.ports:
                host_config["PortBindings"] = {
                    f"{port}/tcp": [{"HostPort": str(host_port)}] for port, host_port in self.ports.items()
                }
            if self.extra_hosts:
                host_config["ExtraHosts"] = [f"{host}:{ip}" for host, ip in self.extra_hosts.items()]
            if self.devices:
                host_config["Devices"] = [parse_device(d) for d in self.devices]
            if self.dns:
                host_config["Dns"] = self.dns

            config: dict[str, Any] = {
                "Image": self.image,
                "Cmd": ["sleep", "infinity"],
                "WorkingDir": self.workdir,
                "Env": [f"{k}={v}" for k, v in self.env.items()],
                "HostConfig": host_config,
            }
            if self.user:
                config["User"] = self.user
            if self.ports:
                config["ExposedPorts"] = {f"{port}/tcp": {} for port in self.ports}
            if self.platform:
                config["Platform"] = self.platform
            create_kwargs: dict[str, Any] = {"config": config}
            if self.name:
                create_kwargs["name"] = self.name
            self._container = await self._client.containers.create(**create_kwargs)
            await self._container.start()
            logger.info("Started sandbox container (image=%s)", self.image)

        self._tools = [
            Tool(name="shell", handler=shell, context=self),
            Tool(name="write_file", handler=write_file, context=self),
            Tool(name="read_file", handler=read_file, context=self),
            Tool(name="list_files", handler=list_files, context=self),
            Tool(name="run_python", handler=run_python, context=self),
            Tool(name="patch_file", handler=patch_file, context=self),
        ]
        return self

    async def __aexit__(self, *exc: object) -> None:
        was_attached = self._attached
        if self._container is not None:
            if self.remove and not was_attached:
                with contextlib.suppress(Exception):
                    await self._container.delete(force=True)
                logger.info("Removed sandbox container")
            else:
                logger.info("Kept sandbox container (attached=%r, remove=%r)", was_attached, self.remove)
            self._container = None
            self._attached = False
        if self._client is not None:
            if self.named_volumes and self.volumes_remove and not was_attached:
                for vol_name in self.named_volumes.values():
                    with contextlib.suppress(Exception):
                        vol = await self._client.volumes.get(vol_name)  # type: ignore[no-untyped-call]
                        await vol.delete()
                logger.info("Removed %d named volume(s)", len(self.named_volumes))
            await self._client.close()
            self._client = None
        self._tools = None

    @property
    def tools(self) -> list[Tool[Any]]:
        """Return the axio Tool instances for this sandbox. Only valid inside `async with`."""
        if self._tools is None:
            raise RuntimeError("DockerSandbox must be used as an async context manager")
        return list(self._tools)

    @property
    def container_id(self) -> str:
        """Return the ID of the running container. Only valid inside `async with`."""
        if self._container is None:
            raise RuntimeError("DockerSandbox must be used as an async context manager")
        return self._container.id

    async def _ensure_image(self) -> None:
        """Pull the image if it is not present locally."""
        assert self._client is not None
        try:
            await self._client.images.inspect(self.image)
            logger.debug("Image already present: %s", self.image)
        except aiodocker.exceptions.DockerError:
            logger.info("Pulling image %s ...", self.image)
            await self._client.images.pull(self.image)
            logger.info("Image pulled: %s", self.image)

    async def exec(self, command: str, timeout: float = 30, stdin: str | None = None) -> str:
        """Execute a shell command inside the container and return its output."""
        assert self._container is not None

        if stdin is not None:
            stdin_path = f"/tmp/.axio_stdin_{uuid.uuid4().hex}"
            await self.write_file(stdin_path, stdin)
            # Wrap in a subshell so the redirect applies to the whole command,
            # not just the last simple command when the caller's command already
            # uses semicolons (e.g. RunPython's "; exit $_rc" suffix).
            command = f"( {command} ) < {stdin_path}; _rc=$?; rm -f {stdin_path}; exit $_rc"

        exec_obj = await self._container.exec(
            cmd=["sh", "-c", command],
            stdout=True,
            stderr=True,
            tty=False,
        )
        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []

        async def consume() -> None:
            stream = exec_obj.start(detach=False)
            try:
                while True:
                    msg = await stream.read_out()
                    if msg is None:
                        break
                    if msg.stream == 1:
                        stdout_parts.append(msg.data)
                    else:
                        stderr_parts.append(msg.data)
            finally:
                await stream.close()

        try:
            await asyncio.wait_for(consume(), timeout=timeout)
        except TimeoutError:
            return f"[timeout after {timeout}s]"

        info = await exec_obj.inspect()
        exit_code: int = info["ExitCode"]

        output = b"".join(stdout_parts).decode()
        if stderr_parts:
            output += "\n[stderr]\n" + b"".join(stderr_parts).decode()
        if exit_code != 0:
            output += f"\n[exit code: {exit_code}]"
        return output.strip() or "(no output)"

    async def write_file(self, path: str, content: str, mode: int = 0o644) -> str:
        """Write a string to a file inside the container."""
        assert self._container is not None
        data = content.encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tar:
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
        parent = os.path.dirname(path) or "/"
        await self.exec(f"mkdir -p {shlex.quote(parent)}")
        await self._container.put_archive(  # type: ignore[no-untyped-call]
            path=parent,
            data=buf.getvalue(),
        )
        return f"Wrote {len(data)} bytes to {path}"

    async def get_archive(self, path: str) -> tarfile.TarFile:
        """Fetch a path from the container as a TarFile object."""
        assert self._container is not None
        return await self._container.get_archive(path=path)

    async def read_file_bytes(self, path: str) -> bytes:
        """Read a file from inside the container and return raw bytes."""
        tar = await self.get_archive(path)
        member = tar.next()
        if member is None:
            return b""
        f = tar.extractfile(member)
        return f.read() if f else b""
