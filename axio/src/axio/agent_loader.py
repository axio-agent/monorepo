"""Load Agent prototypes from declarative config files (TOML, JSON, INI).

Each file describes one agent. The loader scans a directory, parses every
recognised file, resolves tool names against a caller-supplied toolbox, and
returns a ``dict[str, tuple[str, Agent]]`` that matches the shape used by
``agent_swarm`` and similar orchestrators.

Supported formats
-----------------
TOML::

    name = "architect"
    description = "System design and interface specs"
    max_iterations = 100
    tools = ["read_file", "write_file"]

    [system]
    text = \"\"\"
    You are an expert software architect...
    \"\"\"

JSON::

    {
      "name": "architect",
      "description": "System design and interface specs",
      "max_iterations": 100,
      "tools": ["read_file", "write_file"],
      "system": "You are an expert software architect..."
    }

INI::

    [agent]
    name = architect
    description = System design and interface specs
    max_iterations = 100
    tools = read_file, write_file

    [system]
    text = You are an expert software architect...

In all formats ``name`` falls back to the file stem when omitted.  ``system``
may be a plain string or a ``{"text": "..."}`` mapping (TOML/JSON).  INI tools
are comma-separated.

Custom sources
--------------
Subclass :class:`AgentLoader` and implement :meth:`~AgentLoader.load` - the
base :meth:`~AgentLoader.load_file` will handle reading the file and calling
your implementation automatically::

    class DbAgentLoader(AgentLoader):
        def load(self, content: str) -> AgentSpec:
            row = json.loads(content)
            return AgentSpec(name=row["name"], ...)
"""

from __future__ import annotations

import configparser
import dataclasses
import json
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, TypedDict

from .agent import Agent
from .context import ContextStore, MemoryContextStore
from .events import StreamEvent, TextDelta
from .field import Field
from .tool import CONTEXT, Tool
from .transport import CompletionTransport, DummyCompletionTransport


@dataclass(frozen=True)
class AgentSpec:
    """Parsed agent definition before transport/tools are injected."""

    name: str
    description: str
    system: str
    max_iterations: int = 50
    tools: tuple[str, ...] = ()
    model: str | None = None

    def to_agent(self, toolbox: Mapping[str, Tool[Any]] = MappingProxyType({})) -> Agent:
        """Return a prototype Agent with *toolbox* tools attached.

        The agent uses :class:`~axio.transport.DummyCompletionTransport` -
        call ``agent.copy(transport=real_transport)`` before running it.

        Raises :exc:`KeyError` if any name in ``self.tools`` is absent from
        *toolbox*.
        """
        resolved: list[Tool[Any]] = []
        for name in self.tools:
            if name not in toolbox:
                raise KeyError(f"Tool {name!r} not found in toolbox")
            resolved.append(toolbox[name])
        return Agent(
            system=self.system,
            transport=DummyCompletionTransport(),
            tools=resolved,
            max_iterations=self.max_iterations,
        )


class AgentLoader:
    """Base class for format-specific agent loaders.

    Subclasses implement :meth:`load` to parse a raw string.  The source of
    that string is entirely up to the caller - files, databases, HTTP, etc.
    :meth:`load_file` is provided on the base class and calls :meth:`load`
    automatically.
    """

    extensions: tuple[str, ...] = ()

    def load(self, content: str) -> AgentSpec:
        """Parse *content* and return an :class:`AgentSpec`.

        ``name`` defaults to ``""`` when not present; :meth:`load_file`
        patches it from the file stem after calling this method.
        """
        raise NotImplementedError

    def load_file(self, path: Path) -> AgentSpec:
        """Read *path* and delegate to :meth:`load`, patching ``name`` from the stem.

        Any :exc:`ValueError` raised by :meth:`load` is re-raised with *path*
        prepended so errors are easy to locate.
        """
        try:
            spec = self.load(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if not spec.name:
            return dataclasses.replace(spec, name=path.stem)
        return spec

    def scan(
        self,
        directory: Path,
        toolbox: Mapping[str, Tool[Any]] = MappingProxyType({}),
    ) -> dict[str, tuple[str, Agent]]:
        """Scan *directory* for agent files and return a name→(description, agent) dict."""
        result: dict[str, tuple[str, Agent]] = {}
        for ext in self.extensions:
            for path in sorted(directory.glob(f"*.{ext}")):
                spec = self.load_file(path)
                result[spec.name] = (spec.description, spec.to_agent(toolbox))
        return result

    def _parse_dict(self, data: dict[str, object]) -> AgentSpec:
        """Normalise a parsed dict (any format) into an :class:`AgentSpec`."""
        name = str(data.get("name", ""))
        description = str(data.get("description", ""))

        system_raw = data.get("system", "")
        if isinstance(system_raw, dict):
            system = str(system_raw.get("text", ""))
        else:
            system = str(system_raw)

        raw_iter = data.get("max_iterations", 50)
        max_iterations = int(raw_iter)  # type: ignore[call-overload]

        tools_raw = data.get("tools", ())
        tools = tuple(tools_raw) if isinstance(tools_raw, (list, tuple)) else ()

        model_raw = data.get("model")
        model = str(model_raw) if model_raw is not None else None

        return AgentSpec(
            name=name,
            description=description,
            system=system,
            max_iterations=max_iterations,
            tools=tools,
            model=model,
        )


class TomlAgentLoader(AgentLoader):
    """Load agent definitions from ``.toml`` files."""

    extensions = ("toml",)

    def load(self, content: str) -> AgentSpec:
        try:
            data = tomllib.loads(content)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Invalid TOML: {exc}") from exc
        return self._parse_dict(data)


class JsonAgentLoader(AgentLoader):
    """Load agent definitions from ``.json`` files."""

    extensions = ("json",)

    def load(self, content: str) -> AgentSpec:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
        return self._parse_dict(data)


class IniAgentLoader(AgentLoader):
    """Load agent definitions from ``.ini`` files.

    Expected sections: ``[agent]`` for metadata, ``[system]`` for the prompt.
    The ``tools`` key is comma-separated.  Multiline system prompts work via
    standard ConfigParser continuation (indent subsequent lines).
    """

    extensions = ("ini",)

    def load(self, content: str) -> AgentSpec:
        cp = configparser.ConfigParser(interpolation=None)
        cp.read_string(content)

        agent_section = cp["agent"] if cp.has_section("agent") else {}
        system_section = cp["system"] if cp.has_section("system") else {}

        name = agent_section.get("name", "")
        description = agent_section.get("description", "")
        max_iterations = int(agent_section.get("max_iterations", 50))
        model_raw = agent_section.get("model")
        model = model_raw if model_raw else None
        system = system_section.get("text", agent_section.get("system", ""))
        tools_raw = agent_section.get("tools", "")
        tools = tuple(t.strip() for t in tools_raw.split(",") if t.strip()) if tools_raw else ()

        return AgentSpec(
            name=name,
            description=description,
            system=system,
            max_iterations=max_iterations,
            tools=tools,
            model=model,
        )


@dataclass
class MultiFormatLoader:
    """Aggregate loader that handles TOML, JSON, and INI files."""

    loaders: list[AgentLoader] = field(
        default_factory=lambda: [TomlAgentLoader(), JsonAgentLoader(), IniAgentLoader()]
    )

    def scan(
        self,
        directory: Path,
        toolbox: Mapping[str, Tool[Any]] = MappingProxyType({}),
    ) -> dict[str, tuple[str, Agent]]:
        """Scan *directory* with all registered loaders, later loaders win on name collision."""
        result: dict[str, tuple[str, Agent]] = {}
        for loader in self.loaders:
            result.update(loader.scan(directory, toolbox))
        return result


def load_agents(
    directory: Path,
    toolbox: Mapping[str, Tool[Any]] = MappingProxyType({}),
) -> dict[str, tuple[str, Agent]]:
    """Scan *directory* for ``.toml``, ``.json``, and ``.ini`` agent files.

    Returns ``dict[name, (description, agent)]`` - same shape as the
    ``AGENTS`` registry used in ``agent_swarm`` and similar examples.

    Example::

        from pathlib import Path
        from axio.agent_loader import load_agents

        AGENTS = load_agents(
            Path(__file__).parent / "roles",
            toolbox={"read_file": read_file_tool, "write_file": write_file_tool},
        )
    """
    return MultiFormatLoader().scan(directory, toolbox)


class AgentContext(TypedDict):
    agent_name: str
    proto: Agent
    transport: CompletionTransport
    context_factory: Callable[[], ContextStore]
    on_event: Callable[[str, StreamEvent], None] | None


async def agent_tool(task: Annotated[str, Field(description="Full task instructions.")]) -> str:
    """Delegate a task to a sub-agent."""
    context: AgentContext = CONTEXT.get()
    agent = context["proto"].copy(transport=context["transport"])
    ctx = context["context_factory"]()
    if context["on_event"] is None:
        return await agent.run(task, ctx)
    parts: list[str] = []
    async for event in agent.run_stream(task, ctx):
        context["on_event"](context["agent_name"], event)
        if isinstance(event, TextDelta):
            parts.append(event.delta)
    return "".join(parts)


def make_agent_tools(
    agents: dict[str, tuple[str, Agent]],
    transport: CompletionTransport,
    context_factory: Callable[[], ContextStore] = MemoryContextStore,
    on_event: Callable[[str, StreamEvent], None] | None = None,
    agent_name_prefix: str = "",
) -> list[Tool[AgentContext]]:
    """Convert each agent into its own :class:`~axio.tool.Tool`.

    Each tool is named after the agent and accepts a single ``task`` field.
    Runtime dependencies (transport, context factory, event callback) are
    stored in :attr:`~axio.tool.Tool.context` and injected on each call.

    Parameters
    ----------
    agents:
        ``dict[name, (description, prototype_agent)]`` - e.g. the value
        returned by :func:`load_agents`.
    transport:
        Transport assigned to the selected agent via ``agent.copy()``.
    context_factory:
        Called once per invocation to produce a fresh
        :class:`~axio.context.ContextStore`.  Defaults to
        :class:`~axio.context.MemoryContextStore`.
    on_event:
        Optional callback receiving ``(agent_name, event)`` for every
        :class:`~axio.events.StreamEvent` the agent emits.
    agent_name_prefix:
        Prefix to prepend to each agent name.

    Example::

        from axio.agent_loader import load_agents, make_agent_tools

        agents = load_agents(Path("roles"), toolbox={"read_file": read_file_tool})
        tools = make_agent_tools(agents, transport=my_transport)
        orchestrator = Agent(system="...", transport=my_transport, tools=tools)
    """
    result = []
    for agent_name, (desc, proto) in agents.items():
        tool = Tool(
            name=f"{agent_name_prefix}{agent_name}",
            description=desc,
            handler=agent_tool,
            context=AgentContext(
                proto=proto,
                transport=transport,
                context_factory=context_factory,
                on_event=on_event,
                agent_name=agent_name,
            ),
        )
        result.append(tool)
    return result


def load_agents_from_dir(
    directory: Path,
    transport: CompletionTransport,
    context_factory: Callable[[], ContextStore] = MemoryContextStore,
    on_event: Callable[[str, StreamEvent], None] | None = None,
) -> list[Tool[AgentContext]]:
    return make_agent_tools(load_agents(directory), transport, context_factory, on_event)
