"""TransportRegistry — discovers, initialises and manages transport plugins."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from axio.models import Capability, ModelRegistry, ModelSpec, TransportMeta

from .plugin import discover_transport_settings, discover_transports
from .sqlite_context import ProjectConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """Associates a model role with a specific transport and model."""

    transport: str
    model: ModelSpec


@dataclass(slots=True)
class TransportRegistry:
    """Discovers transports, checks API keys, fetches model catalogues."""

    _transports: dict[str, Any] = field(default_factory=dict, repr=False)
    _classes: dict[str, type] = field(default_factory=dict, repr=False)
    _metas: dict[str, TransportMeta] = field(default_factory=dict)
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
        """Discover transports, create instances for those with API keys, fetch models."""
        self._session = session
        self._config = global_config or config
        cls_map = discover_transports()
        self._screens = discover_transport_settings()

        for name, cls in cls_map.items():
            meta: TransportMeta | None = getattr(cls, "META", None)
            if meta is None:
                logger.warning("Transport %r has no META, skipping", name)
                continue

            # Always store class and meta so settings screens work for all transports
            self._classes[name] = cls
            self._metas[name] = meta

            # Load saved settings from config
            saved: dict[str, str] = {}
            if self._config is not None:
                saved = await self._load_settings(self._config, name)
            self._saved[name] = saved

            # Determine API key: saved settings first, env second
            env_key = os.environ.get(meta.api_key_env, "") if meta.api_key_env else ""
            api_key = saved.get("api_key", "") or env_key
            if not api_key:
                logger.info("Transport %r: no API key (env %s), skipping", name, meta.api_key_env)
                continue

            # Build transport with saved settings as extra kwargs
            kwargs: dict[str, Any] = {"api_key": api_key, "session": session}
            for k, v in saved.items():
                if k != "api_key" and v:
                    kwargs[k] = v

            transport = cls(**kwargs)
            try:
                await transport.fetch_models()
            except Exception:
                logger.warning("Transport %r: fetch_models failed", name, exc_info=True)
            self._transports[name] = transport

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
            meta = self._metas[name]
            env_key = os.environ.get(meta.api_key_env, "") if meta.api_key_env else ""
            api_key = settings.get("api_key", "") or env_key
            if api_key:
                kwargs: dict[str, Any] = {"api_key": api_key, "session": self._session}
                for k, v in settings.items():
                    if k != "api_key" and v:
                        kwargs[k] = v
                transport = cls(**kwargs)
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
        """Names of all discovered transports (including those without API keys)."""
        return list(self._classes)

    def get_transport(self, name: str) -> Any:
        """Return the initialised transport instance by name."""
        return self._transports[name]

    def get_meta(self, name: str) -> TransportMeta:
        return self._metas[name]

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
            "model": model,
            "models": src.models,
            "session": src.session,
        }
        # Apply saved settings (e.g. base_url) that aren't already set
        for k, v in self._saved.get(name, {}).items():
            if k not in kwargs and v:
                kwargs[k] = v
        return cls(**kwargs)

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

    def resolve_default(self, role: str) -> RoleBinding | None:
        """Find the first transport that has a default for the given role."""
        for name, meta in self._metas.items():
            if name not in self._transports:
                continue
            model_id = meta.role_defaults.get(role)
            if model_id and model_id in self._transports[name].models:
                return RoleBinding(transport=name, model=self._transports[name].models[model_id])
        return None

    def encode(self, name: str, model_id: str) -> str:
        """Encode a transport name and model ID for config persistence."""
        return f"{name}:{model_id}"

    def settings_screens(self) -> dict[str, type]:
        """Return discovered settings screen classes keyed by transport name."""
        return dict(self._screens)
