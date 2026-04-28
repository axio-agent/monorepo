"""TUI screens for OpenAI-compatible custom providers."""

from __future__ import annotations

import asyncio as _asyncio
import json
import logging
from pathlib import Path

import aiohttp
from axio.models import Capability, ModelRegistry, ModelSpec
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static

from axio_transport_openai.custom import OpenAICompatibleTransport

logger = logging.getLogger(__name__)


class _DeleteSentinel:
    pass


_DELETE = _DeleteSentinel()


def _is_delete(obj: object) -> bool:
    return isinstance(obj, _DeleteSentinel)


_VALID_CAPS = sorted(c.value for c in Capability)
_CAPS_HINT = ", ".join(_VALID_CAPS)


class ModelEditScreen(ModalScreen["ModelSpec | _DeleteSentinel | None"]):
    """Add / edit one ModelSpec within a custom provider."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    ModelEditScreen { align: center middle; }
    #me-edit {
        width: 72; height: auto; max-height: 90%;
        border: heavy $accent; background: $panel; padding: 1 2;
    }
    #me-edit Input { margin-bottom: 1; }
    .me-buttons { height: auto; margin-top: 1; }
    .me-buttons Button { margin: 0 1; }
    """

    def __init__(self, model: ModelSpec | None) -> None:
        super().__init__()
        self._editing = model

    def compose(self) -> ComposeResult:
        m = self._editing
        title = "Edit Model" if m else "Add Model"
        with Container(id="me-edit"):
            yield Static(f"[bold]{title}[/]")
            yield Static("Model ID:")
            yield Input(value=m.id if m else "", placeholder="e.g. llama3.2", id="me-id")
            yield Static("Context window (tokens):")
            yield Input(value=str(m.context_window) if m else "128000", id="me-ctx")
            yield Static("Max output tokens:")
            yield Input(value=str(m.max_output_tokens) if m else "8000", id="me-out")
            yield Static(f"Capabilities (comma-sep: {_CAPS_HINT}):")
            caps_str = ", ".join(c.value for c in m.capabilities) if m else "text, tool_use"
            yield Input(value=caps_str, id="me-caps")
            yield Static("Input cost ($/1M tokens, 0 for free):")
            yield Input(value=str(m.input_cost) if m else "0", id="me-incost")
            yield Static("Output cost ($/1M tokens, 0 for free):")
            yield Input(value=str(m.output_cost) if m else "0", id="me-outcost")
            with Horizontal(classes="me-buttons"):
                yield Button("Save", id="btn-save", variant="primary")
                if m:
                    yield Button("Delete", id="btn-delete", variant="error")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#me-id", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-delete":
            self.dismiss(_DELETE)
        elif event.button.id == "btn-save":
            self._save()

    def _save(self) -> None:
        model_id = self.query_one("#me-id", Input).value.strip()
        if not model_id:
            self.notify("Model ID is required", severity="error")
            return
        if "/" in model_id:
            self.notify("Model ID must not contain '/'", severity="error")
            return
        try:
            ctx = int(self.query_one("#me-ctx", Input).value.strip())
            out = int(self.query_one("#me-out", Input).value.strip())
            in_cost = float(self.query_one("#me-incost", Input).value.strip() or "0")
            out_cost = float(self.query_one("#me-outcost", Input).value.strip() or "0")
        except ValueError:
            self.notify("Context window and max output must be integers", severity="error")
            return
        raw_caps = self.query_one("#me-caps", Input).value
        caps = frozenset(Capability(c.strip()) for c in raw_caps.split(",") if c.strip() in Capability.__members__)
        self.dismiss(
            ModelSpec(
                id=model_id,
                context_window=ctx,
                max_output_tokens=out,
                capabilities=caps,
                input_cost=in_cost,
                output_cost=out_cost,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProviderEditScreen(ModalScreen["OpenAICompatibleTransport | _DeleteSentinel | None"]):
    """Add / edit a custom provider and manage its model list."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    ProviderEditScreen { align: center middle; }
    #pe-edit {
        width: 80; height: auto; max-height: 90%;
        border: heavy $accent; background: $panel; padding: 1 2;
    }
    #pe-edit Input { margin-bottom: 1; }
    #pe-models { height: 8; margin-bottom: 1; }
    .pe-model-buttons { height: auto; margin-bottom: 1; }
    .pe-model-buttons Button { margin: 0 1; }
    .pe-buttons { height: auto; margin-top: 1; }
    .pe-buttons Button { margin: 0 1; }
    """

    def __init__(self, provider: OpenAICompatibleTransport | None) -> None:
        super().__init__()
        self._editing: OpenAICompatibleTransport | None = provider
        self._models: list[ModelSpec] = list(provider.models.values()) if provider else []

    def _model_entries(self) -> list[str]:
        if not self._models:
            return ["  (no models - add one below)"]
        return [f"  {m.id:<32} ctx={m.context_window}  out={m.max_output_tokens}" for m in self._models]

    def _refresh_models(self) -> None:
        ol = self.query_one("#pe-models", OptionList)
        ol.clear_options()
        for entry in self._model_entries():
            ol.add_option(entry)

    def compose(self) -> ComposeResult:
        p = self._editing
        title = "Edit Provider" if p else "Add OpenAI-Compatible Provider"
        with ScrollableContainer(id="pe-edit"):
            yield Static(f"[bold]{title}[/]")
            yield Static("Name (used as registry key):")
            yield Input(value=p.name if p else "", placeholder="e.g. localai", id="pe-name")
            yield Static("Base URL:")
            yield Input(
                value=p.base_url if p else "",
                placeholder="e.g. http://localhost:8080/v1",
                id="pe-url",
            )
            yield Static("API Key (leave blank if not required):")
            yield Input(value=p.api_key if p else "", id="pe-apikey", password=True)
            yield Static("Models:")
            yield OptionList(*self._model_entries(), id="pe-models")
            with Horizontal(classes="pe-model-buttons"):
                yield Button("+ Add Model", id="btn-add-model")
            with Horizontal(classes="pe-buttons"):
                yield Button("Save", id="btn-save", variant="primary")
                if p:
                    yield Button("Delete Provider", id="btn-delete", variant="error")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#pe-name", Input).focus()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        idx = message.option_index
        if idx < len(self._models):
            self.app.push_screen(
                ModelEditScreen(self._models[idx]),
                lambda r, i=idx: self._on_model_edit(i, r),
            )

    def _on_model_add(self, result: object) -> None:
        if result is None or _is_delete(result):
            return
        assert isinstance(result, ModelSpec)
        self._models.append(result)
        self._refresh_models()

    def _on_model_edit(self, idx: int, result: object) -> None:
        if result is None:
            return
        if _is_delete(result):
            self._models.pop(idx)
        else:
            assert isinstance(result, ModelSpec)
            self._models[idx] = result
        self._refresh_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-delete":
            self.dismiss(_DELETE)
        elif event.button.id == "btn-add-model":
            self.app.push_screen(ModelEditScreen(None), self._on_model_add)
        elif event.button.id == "btn-save":
            self._save()

    def _save(self) -> None:
        name = self.query_one("#pe-name", Input).value.strip()
        if not name:
            self.notify("Name is required", severity="error")
            return
        if "/" in name:
            self.notify("Provider name must not contain '/'", severity="error")
            return
        url = self.query_one("#pe-url", Input).value.strip()
        if not url:
            self.notify("Base URL is required", severity="error")
            return
        api_key = self.query_one("#pe-apikey", Input).value.strip()
        self.dismiss(
            OpenAICompatibleTransport(
                name=name,
                base_url=url,
                api_key=api_key,
                models=ModelRegistry(self._models),
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class CustomHubScreen(ModalScreen["dict[str, str] | None"]):
    """Hub: list configured OpenAI-compatible providers; add / edit / delete them.

    On close, if providers were changed the hub saves the JSON config and
    re-registers per-provider transport instances in the app's transport
    registry under the names ``openai-custom.<provider-name>``.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    CustomHubScreen { align: center middle; }
    #custom-hub {
        width: 84; height: 80%;
        border: heavy $accent; background: $panel; padding: 1 2;
    }
    #custom-list { height: 1fr; }
    """

    CONFIG_PATH: Path = Path.home() / ".local" / "share" / "axio" / "openai-custom.json"

    def __init__(self, settings: dict[str, str]) -> None:  # settings unused
        super().__init__()
        self._providers: list[OpenAICompatibleTransport] = []
        self._changed = False

    @classmethod
    def load_config(cls, session: aiohttp.ClientSession | None = None) -> list[OpenAICompatibleTransport]:
        """Read provider list from CONFIG_PATH; returns [] on any error."""
        if not cls.CONFIG_PATH.exists():
            return []
        try:
            raw = json.loads(cls.CONFIG_PATH.read_text("utf-8"))
            return [OpenAICompatibleTransport.from_dict(p, session=session) for p in raw]
        except Exception:
            logger.warning("Failed to load %s", cls.CONFIG_PATH, exc_info=True)
            return []

    @classmethod
    def save_config(cls, providers: list[OpenAICompatibleTransport]) -> None:
        """Write provider list to CONFIG_PATH (creates parent dirs as needed)."""
        cls.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.CONFIG_PATH.write_text(
            json.dumps([p.to_dict() for p in providers], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def compose(self) -> ComposeResult:
        with Container(id="custom-hub"):
            yield Static("[bold]OpenAI-Compatible Providers[/]")
            yield OptionList(id="custom-list")

    async def on_mount(self) -> None:
        self._providers = await _asyncio.to_thread(self.load_config)
        self._refresh_list()
        self.query_one("#custom-list", OptionList).focus()

    def _format_entries(self) -> list[str]:
        entries = ["  + Add Provider"]
        for p in self._providers:
            n = len(p.models)
            entries.append(f"  {p.name:<20}  {p.base_url:<40}  ({n} model{'s' if n != 1 else ''})")
        return entries

    def _refresh_list(self) -> None:
        ol = self.query_one("#custom-list", OptionList)
        ol.clear_options()
        for entry in self._format_entries():
            ol.add_option(entry)

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        idx = message.option_index
        if idx == 0:
            self.app.push_screen(ProviderEditScreen(None), self._on_add)
        else:
            provider = self._providers[idx - 1]
            self.app.push_screen(
                ProviderEditScreen(provider),
                lambda r, p=provider: self._on_edit(p, r),
            )

    def _on_add(self, result: object) -> None:
        if result is None or _is_delete(result):
            return
        assert isinstance(result, OpenAICompatibleTransport)
        self._providers.append(result)
        self._changed = True
        self._refresh_list()

    def _on_edit(self, old: OpenAICompatibleTransport, result: object) -> None:
        if result is None:
            return
        idx = next((i for i, p in enumerate(self._providers) if p.name == old.name), None)
        if idx is None:
            return
        if _is_delete(result):
            self._providers.pop(idx)
        else:
            assert isinstance(result, OpenAICompatibleTransport)
            self._providers[idx] = result
        self._changed = True
        self._refresh_list()

    async def _apply_changes(self) -> None:
        """Persist JSON and re-register per-provider transports in the registry."""
        await _asyncio.to_thread(self.save_config, self._providers)
        registry = getattr(self.app, "_transports", None)
        if registry is None:
            return
        session = getattr(registry, "_session", None)
        registry.unregister_by_prefix("openai-custom.")
        for t in self._providers:
            t.session = session
            registry.register_dynamic(f"openai-custom.{t.name}", t)

    async def action_cancel(self) -> None:
        if self._changed:
            await self._apply_changes()
        await self.dismiss(None)
