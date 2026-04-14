"""TransportRegistry — discovers, initialises and manages transport plugins."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import aiohttp
from axio.models import Capability, ModelRegistry, ModelSpec

from .plugin import discover_transport_settings, discover_transports
from .sqlite_config import ProjectConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """Associates a model role with a specific transport and model."""

    transport: str
    model: ModelSpec


@dataclass(slots=True)
class TransportRegistry:
    """Discovers transports, creates instances, and manages them."""

    _transports: dict[str, Any] = field(default_factory=dict, repr=False)
    _classes: dict[str, type] = field(default_factory=dict, repr=False)
    _screens: dict[str, type] = field(default_factory=dict)
    _saved: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)
    _config: ProjectConfig | None = field(default=None, repr=False)
    _session: aiohttp.ClientSession | None = field(default=None, repr=False)

    async def init(
        self,
        session: aiohttp.ClientSession,
        config: ProjectConfig | None = None,
        global_config: ProjectConfig | None = None,
    ) -> None:
        """Discover transports, create instances for those that initialise successfully."""
        self._session = session
        self._config = global_config or config
        cls_map = discover_transports()
        self._screens = discover_transport_settings()

        pending: list[tuple[str, Any]] = []
        for name, cls in cls_map.items():
            self._classes[name] = cls
            saved: dict[str, str] = {}
            if self._config is not None:
                saved = await self._load_settings(self._config, name)
            self._saved[name] = saved

            kwargs: dict[str, Any] = {"session": session}
            for k, v in saved.items():
                if v:
                    kwargs[k] = v
            try:
                instance = cls(**kwargs)
            except Exception:
                logger.warning("Transport %r failed to instantiate, skipping", name, exc_info=True)
                continue
            pending.append((name, instance))

        async def _fetch_one(name: str, transport: Any) -> None:
            if hasattr(transport, "on_auth_refresh"):
                transport.on_auth_refresh = partial(self._persist_auth, name)
            try:
                await transport.fetch_models()
                self._transports[name] = transport
            except Exception:
                logger.warning("Transport %r: fetch_models failed, skipping", name, exc_info=True)

        await asyncio.gather(*[_fetch_one(n, t) for n, t in pending])

        # Bulk-load providers from hub screens that expose load_config()
        # (e.g. CustomHubScreen for openai-custom.* providers)
        for screen_name, screen_cls in self._screens.items():
            load = getattr(screen_cls, "load_config", None)
            if not callable(load):
                continue
            try:
                providers = await asyncio.to_thread(load, session)
                for t in providers:
                    self.register_dynamic(f"{screen_name}.{t.name}", t)
            except Exception:
                logger.warning("Failed to bulk-load providers from %r", screen_name, exc_info=True)

    @staticmethod
    async def _load_settings(config: ProjectConfig, name: str) -> dict[str, str]:
        """Load saved transport settings from config, stripping the key prefix."""
        prefix = f"transport.{name}."
        raw = await config.get_prefix(prefix)
        return {k[len(prefix) :]: v for k, v in raw.items()}

    async def get_settings(self, name: str) -> dict[str, str]:
        """Return saved settings for a transport from the config DB."""
        if self._config is None:
            return dict(self._saved.get(name, {}))
        return await self._load_settings(self._config, name)

    async def save_settings(self, name: str, settings: dict[str, str]) -> None:
        """Persist settings to config and re-create the transport instance."""
        if self._config is not None:
            prefix = f"transport.{name}."
            await self._config.delete_prefix(prefix)
            for key, value in settings.items():
                if value:
                    await self._config.set(f"{prefix}{key}", value)

        self._saved[name] = settings

        # Re-create transport with new settings
        if name in self._classes and self._session is not None:
            cls = self._classes[name]
            kwargs: dict[str, Any] = {"session": self._session}
            for k, v in settings.items():
                if v:
                    kwargs[k] = v
            transport = cls(**kwargs)
            if hasattr(transport, "on_auth_refresh"):
                transport.on_auth_refresh = partial(self._persist_auth, name)
            try:
                await transport.fetch_models()
            except Exception:
                logger.warning("Transport %r: fetch_models failed after reconfigure", name, exc_info=True)
            self._transports[name] = transport

    @property
    def available(self) -> list[str]:
        """Names of transports that were successfully initialised."""
        return list(self._transports)

    @property
    def discovered(self) -> list[str]:
        """Names of all discovered transports (including those that failed to initialise)."""
        return list(self._classes)

    def get_transport(self, name: str) -> Any:
        """Return the initialised transport instance by name."""
        return self._transports[name]

    def all_models(self, *caps: Capability) -> list[tuple[str, ModelSpec]]:
        """Return (transport_name, ModelSpec) across all transports, optionally filtered."""
        result: list[tuple[str, ModelSpec]] = []
        required = frozenset(caps)
        for name, transport in self._transports.items():
            registry: ModelRegistry = transport.models
            for spec in registry.values():
                if required and not (required <= spec.capabilities):
                    continue
                result.append((name, spec))
        return result

    def make_transport(self, name: str, model: ModelSpec) -> Any:
        """Create a new transport instance for the given name and model."""
        src = self._transports[name]
        cls = self._classes[name]
        kwargs: dict[str, Any] = {
            "api_key": src.api_key,
            "base_url": src.base_url,
            "model": model,
            "models": src.models,
            "session": src.session,
        }
        # Apply saved settings that aren't already set
        for k, v in self._saved.get(name, {}).items():
            if k not in kwargs and v:
                kwargs[k] = v
        transport = cls(**kwargs)
        if hasattr(transport, "on_auth_refresh"):
            transport.on_auth_refresh = partial(self._persist_auth, name)
        return transport

    async def _persist_auth(self, name: str, tokens: dict[str, str]) -> None:
        if self._config is None:
            return
        prefix = f"transport.{name}."
        for key, value in tokens.items():
            if value:
                await self._config.set(f"{prefix}{key}", value)

    def register_dynamic(self, name: str, instance: Any) -> None:
        """Register a dynamically-created transport instance (e.g. custom providers)."""
        self._transports[name] = instance
        self._classes[name] = type(instance)

    def unregister_by_prefix(self, prefix: str) -> None:
        """Remove all transports whose names start with *prefix*."""
        names = [n for n in list(self._transports) if n.startswith(prefix)]
        for n in names:
            del self._transports[n]
            self._classes.pop(n, None)
            self._saved.pop(n, None)

    def resolve(self, config_value: str) -> RoleBinding | None:
        """Parse a config value like ``"nebius:model_id"`` into a RoleBinding.

        Falls back to searching all transports if no prefix is present (migration).
        Returns None if the model cannot be found in any transport.
        """
        if ":" in config_value:
            transport_name, model_id = config_value.split(":", 1)
            if transport_name in self._transports:
                transport = self._transports[transport_name]
                if model_id in transport.models:
                    return RoleBinding(transport=transport_name, model=transport.models[model_id])
            return None

        # Migration: bare model ID — search all transports
        for name, transport in self._transports.items():
            if config_value in transport.models:
                return RoleBinding(transport=name, model=transport.models[config_value])
        return None

    def encode(self, name: str, model_id: str) -> str:
        """Encode a transport name and model ID for config persistence."""
        return f"{name}:{model_id}"

    def model_counts(self) -> dict[str, int]:
        """Return number of models per available transport."""
        return {name: len(t.models) for name, t in self._transports.items()}

    async def reload_models(self, name: str | None = None) -> None:
        """Re-fetch model catalogues for one transport (or all available if name is None)."""
        names = [name] if (name and name in self._transports) else list(self._transports)
        await asyncio.gather(*[self._reload_one(n) for n in names])

    async def _reload_one(self, name: str) -> None:
        transport = self._transports[name]
        transport.models.clear()
        try:
            await transport.fetch_models()
        except Exception:
            logger.warning("Transport %r: fetch_models failed on reload", name, exc_info=True)

    def settings_screens(self) -> dict[str, type]:
        """Return discovered settings screen classes keyed by transport name."""
        return dict(self._screens)
