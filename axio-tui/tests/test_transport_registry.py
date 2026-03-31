"""Tests for axio_tui.transport_registry."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import aiohttp
from axio.models import Capability, ModelRegistry, ModelSpec, TransportMeta

from axio_tui.sqlite_context import ProjectConfig
from axio_tui.transport_registry import TransportRegistry

_TT = frozenset({Capability.text, Capability.tool_use})
_VT = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_ET = frozenset({Capability.embedding})

_SPEC_CHAT = ModelSpec(id="test-chat", capabilities=_TT, context_window=100_000)
_SPEC_VISION = ModelSpec(id="test-vision", capabilities=_VT, context_window=100_000)
_SPEC_EMBED = ModelSpec(id="test-embed", capabilities=_ET, context_window=8_000)
_SPEC_OTHER = ModelSpec(id="other-chat", capabilities=_TT, context_window=50_000)


@dataclass(slots=True)
class _FakeTransport:
    META: ClassVar[TransportMeta] = TransportMeta(
        label="Fake",
        api_key_env="FAKE_API_KEY",
        role_defaults={"chat": "test-chat", "vision": "test-vision", "embedding": "test-embed"},
    )

    base_url: str = "https://fake.api/v1"
    api_key: str = ""
    model: ModelSpec = _SPEC_CHAT
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry([_SPEC_CHAT, _SPEC_VISION, _SPEC_EMBED]))
    session: aiohttp.ClientSession | None = None

    async def fetch_models(self) -> None:
        pass


@dataclass(slots=True)
class _OtherTransport:
    META: ClassVar[TransportMeta] = TransportMeta(
        label="Other",
        api_key_env="OTHER_API_KEY",
        role_defaults={"chat": "other-chat"},
    )

    api_key: str = ""
    model: ModelSpec = _SPEC_OTHER
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry([_SPEC_OTHER]))
    session: aiohttp.ClientSession | None = None

    async def fetch_models(self) -> None:
        pass


@dataclass(slots=True)
class _NoMetaTransport:
    api_key: str = ""
    model: ModelSpec = _SPEC_CHAT
    models: ModelRegistry = field(default_factory=ModelRegistry)
    session: aiohttp.ClientSession | None = None

    async def fetch_models(self) -> None:
        pass


def _patch_discover(
    transports: dict[str, type], settings: dict[str, type] | None = None
) -> AbstractContextManager[None]:
    """Return a context manager that patches discover_transports and discover_transport_settings."""

    @contextlib.contextmanager
    def _ctx() -> Generator[None, None, None]:
        with (
            patch("axio_tui.transport_registry.discover_transports", return_value=transports),
            patch("axio_tui.transport_registry.discover_transport_settings", return_value=settings or {}),
            patch.dict(
                "os.environ",
                {m.api_key_env: "key" for cls in transports.values() if (m := getattr(cls, "META", None))},
                clear=False,
            ),
        ):
            yield

    return _ctx()


class TestTransportRegistryInit:
    async def test_discovers_transport_with_api_key(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        assert reg.available == ["fake"]

    async def test_skips_transport_without_api_key(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with (
            patch("axio_tui.transport_registry.discover_transports", return_value={"fake": _FakeTransport}),
            patch("axio_tui.transport_registry.discover_transport_settings", return_value={}),
            patch.dict("os.environ", {}, clear=True),
        ):
            await reg.init(session)
        assert reg.available == []

    async def test_skips_transport_without_meta(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"nometa": _NoMetaTransport}):
            await reg.init(session)
        assert reg.available == []

    async def test_multiple_transports(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport, "other": _OtherTransport}):
            await reg.init(session)
        assert sorted(reg.available) == ["fake", "other"]


class TestAllModels:
    async def test_returns_all_models(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        models = reg.all_models()
        assert len(models) == 3
        assert all(name == "fake" for name, _ in models)

    async def test_filters_by_capability(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        vision_models = reg.all_models(Capability.vision)
        assert len(vision_models) == 1
        assert vision_models[0][1].id == "test-vision"

    async def test_multi_transport_models(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport, "other": _OtherTransport}):
            await reg.init(session)
        models = reg.all_models(Capability.tool_use)
        transport_names = {name for name, _ in models}
        assert "fake" in transport_names
        assert "other" in transport_names


class TestResolve:
    async def test_resolve_prefixed(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        binding = reg.resolve("fake:test-chat")
        assert binding is not None
        assert binding.transport == "fake"
        assert binding.model.id == "test-chat"

    async def test_resolve_bare_model_id_migration(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        binding = reg.resolve("test-chat")
        assert binding is not None
        assert binding.transport == "fake"
        assert binding.model.id == "test-chat"

    async def test_resolve_unknown_transport(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        assert reg.resolve("unknown:test-chat") is None

    async def test_resolve_unknown_model(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        assert reg.resolve("fake:nonexistent") is None

    async def test_resolve_bare_unknown(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        assert reg.resolve("nonexistent-model") is None


class TestResolveDefault:
    async def test_resolves_role_default(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        binding = reg.resolve_default("chat")
        assert binding is not None
        assert binding.transport == "fake"
        assert binding.model.id == "test-chat"

    async def test_returns_none_for_unknown_role(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        assert reg.resolve_default("unknown_role") is None


class TestEncode:
    def test_encode(self) -> None:
        reg = TransportRegistry()
        assert reg.encode("nebius", "deepseek-ai/DeepSeek-V3") == "nebius:deepseek-ai/DeepSeek-V3"


class TestMakeTransport:
    async def test_creates_new_transport_instance(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        transport = reg.make_transport("fake", _SPEC_VISION)
        assert isinstance(transport, _FakeTransport)
        assert transport.model == _SPEC_VISION
        assert transport.api_key == "key"

    async def test_make_transport_passes_saved_base_url(self, tmp_path: Path) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        config = ProjectConfig(tmp_path / "test.db", project="test")
        await config.set("transport.fake.base_url", "https://custom.api/v1")
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session, global_config=config)
        transport = reg.make_transport("fake", _SPEC_VISION)
        assert transport.base_url == "https://custom.api/v1"
        await config.close()


class TestSavedSettings:
    async def test_init_loads_saved_base_url(self, tmp_path: Path) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        config = ProjectConfig(tmp_path / "test.db", project="test")
        await config.set("transport.fake.base_url", "https://custom.api/v1")
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session, global_config=config)
        transport = reg.get_transport("fake")
        assert transport.base_url == "https://custom.api/v1"
        await config.close()

    async def test_init_saved_api_key_overrides_env(self, tmp_path: Path) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        config = ProjectConfig(tmp_path / "test.db", project="test")
        await config.set("transport.fake.api_key", "saved-key")
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session, global_config=config)
        transport = reg.get_transport("fake")
        assert transport.api_key == "saved-key"
        await config.close()

    async def test_init_saved_api_key_without_env(self, tmp_path: Path) -> None:
        """A transport can be bootstrapped with a saved api_key even without env var."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        config = ProjectConfig(tmp_path / "test.db", project="test")
        await config.set("transport.fake.api_key", "saved-key")
        reg = TransportRegistry()
        with (
            patch("axio_tui.transport_registry.discover_transports", return_value={"fake": _FakeTransport}),
            patch("axio_tui.transport_registry.discover_transport_settings", return_value={}),
            patch.dict("os.environ", {}, clear=True),
        ):
            await reg.init(session, global_config=config)
        assert reg.available == ["fake"]
        assert reg.get_transport("fake").api_key == "saved-key"
        await config.close()

    async def test_get_settings_returns_saved(self, tmp_path: Path) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        config = ProjectConfig(tmp_path / "test.db", project="test")
        await config.set("transport.fake.base_url", "https://custom.api/v1")
        await config.set("transport.fake.api_key", "saved-key")
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session, global_config=config)
        settings = await reg.get_settings("fake")
        assert settings == {"base_url": "https://custom.api/v1", "api_key": "saved-key"}
        await config.close()

    async def test_get_settings_empty_without_config(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        settings = await reg.get_settings("fake")
        assert settings == {}

    async def test_save_settings_persists_and_recreates(self, tmp_path: Path) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        config = ProjectConfig(tmp_path / "test.db", project="test")
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session, global_config=config)

            # Original transport has default base_url
            original = reg.get_transport("fake")
            assert original.base_url == "https://fake.api/v1"

            # Save new settings (env var still available inside patch)
            await reg.save_settings("fake", {"base_url": "https://new.api/v1"})

            # Transport instance was replaced
            updated = reg.get_transport("fake")
            assert updated is not original
            assert updated.base_url == "https://new.api/v1"

        # Persisted in config DB
        raw = await config.get("transport.fake.base_url")
        assert raw == "https://new.api/v1"
        await config.close()

    async def test_save_settings_clears_old_keys(self, tmp_path: Path) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        config = ProjectConfig(tmp_path / "test.db", project="test")
        await config.set("transport.fake.base_url", "https://old.api/v1")
        await config.set("transport.fake.api_key", "old-key")

        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session, global_config=config)

            # Save settings with only base_url (no api_key)
            await reg.save_settings("fake", {"base_url": "https://new.api/v1"})

        # Old api_key should be deleted
        assert await config.get("transport.fake.api_key") is None
        assert await config.get("transport.fake.base_url") == "https://new.api/v1"
        await config.close()

    async def test_discovered_includes_unavailable_transports(self) -> None:
        """Transports without API keys are in `discovered` but not in `available`."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with (
            patch("axio_tui.transport_registry.discover_transports", return_value={"fake": _FakeTransport}),
            patch("axio_tui.transport_registry.discover_transport_settings", return_value={}),
            patch.dict("os.environ", {}, clear=True),
        ):
            await reg.init(session)
        assert reg.available == []
        assert reg.discovered == ["fake"]

    async def test_init_without_config_uses_defaults(self) -> None:
        session = AsyncMock(spec=aiohttp.ClientSession)
        reg = TransportRegistry()
        with _patch_discover({"fake": _FakeTransport}):
            await reg.init(session)
        transport = reg.get_transport("fake")
        assert transport.base_url == "https://fake.api/v1"
        assert transport.api_key == "key"
