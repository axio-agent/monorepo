"""Nebius AI Studio CompletionTransport — inherits from OpenAI-compatible transport."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from axio.exceptions import StreamError
from axio.models import Capability, ModelSpec, TransportMeta
from axio_transport_openai import OpenAITransport

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NebiusTransport(OpenAITransport):
    META: ClassVar[TransportMeta] = TransportMeta(
        label="Nebius AI Studio",
        api_key_env="NEBIUS_API_KEY",
        role_defaults={
            "chat": "moonshotai/Kimi-K2.5",
            "compact": "openai/gpt-oss-120b",
            "subagent": "openai/gpt-oss-120b",
            "guard": "openai/gpt-oss-20b",
            "vision": "nvidia/Nemotron-Nano-V2-12b",
            "embedding": "Qwen/Qwen3-Embedding-8B",
            "reasoning": "deepseek-ai/DeepSeek-R1-0528",
        },
    )

    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    model: ModelSpec = ModelSpec(id="deepseek-ai/DeepSeek-V3-0324")

    async def fetch_models(self) -> None:
        """Fetch available models from Nebius ``/v1/models?verbose=true``."""
        assert self.session is not None, "session is required for fetch_models"
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.get(url, params={"verbose": "true"}, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise StreamError(f"Nebius API error {resp.status}: {body}")
            payload: dict[str, Any] = await resp.json()

        self.models.clear()
        for entry in payload.get("data", []):
            m = self._parse_model(entry)
            self.models[m.id] = m
        logger.info("Loaded %d models from %s", len(self.models), url)

    @staticmethod
    def _parse_model(entry: dict[str, Any]) -> ModelSpec:
        caps: set[Capability] = set()
        for feat in entry.get("supported_features", []):
            name = "tool_use" if feat == "tools" else feat
            if name in Capability.__members__:
                caps.add(Capability(name))

        modality = entry.get("architecture", {}).get("modality", "")
        parts = modality.split("->") if "->" in modality else [modality]
        input_modality = parts[0]
        output_modality = parts[1] if len(parts) > 1 else ""
        if "image" in input_modality:
            caps.add(Capability.vision)
        if "embedding" in output_modality:
            caps.add(Capability.embedding)

        # Heuristic for known embedding model families
        model_id: str = entry["id"]
        _embed_prefixes = ("BAAI/bge-", "intfloat/e5-", "intfloat/multilingual-e5-")
        if any(model_id.startswith(p) for p in _embed_prefixes) or "/Embedding-" in model_id:
            caps.add(Capability.embedding)

        pricing = entry.get("pricing", {})
        return ModelSpec(
            id=entry["id"],
            context_window=int(entry.get("context_length", 128_000)),
            max_output_tokens=int(entry.get("max_output_tokens", 25_000)),
            capabilities=frozenset(caps),
            input_cost=float(pricing.get("prompt", 0)) * 1_000_000,
            output_cost=float(pricing.get("completion", 0)) * 1_000_000,
        )


try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal
    from textual.screen import ModalScreen
    from textual.widgets import Button, Input, Static

    class NebiusSettingsScreen(ModalScreen[dict[str, str] | None]):
        """Editable settings form for Nebius transport: base_url and api_key."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]
        CSS = """
        NebiusSettingsScreen { align: center middle; }
        #nebius-settings {
            width: 70; height: auto; border: heavy $accent;
            background: $panel; padding: 1 2;
        }
        .field-label { margin-top: 1; }
        .settings-buttons { height: auto; margin-top: 1; }
        .settings-buttons Button { margin: 0 1; }
        """
        DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

        def __init__(self, settings: dict[str, str]) -> None:
            super().__init__()
            self._settings = settings

        def compose(self) -> ComposeResult:
            with Container(id="nebius-settings"):
                yield Static("[bold]Nebius AI Studio Settings[/]")
                yield Static("Base URL:", classes="field-label")
                yield Input(
                    value=self._settings.get("base_url", self.DEFAULT_BASE_URL),
                    id="base-url",
                )
                yield Static("API Key (leave blank to use env var):", classes="field-label")
                yield Input(
                    value=self._settings.get("api_key", ""),
                    id="api-key",
                    password=True,
                )
                with Horizontal(classes="settings-buttons"):
                    yield Button("Save", id="btn-save", variant="primary")
                    yield Button("Cancel", id="btn-cancel")

        def on_mount(self) -> None:
            self.query_one("#base-url", Input).focus()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-save":
                result: dict[str, str] = {}
                base_url = self.query_one("#base-url", Input).value.strip()
                api_key = self.query_one("#api-key", Input).value.strip()
                if base_url:
                    result["base_url"] = base_url
                if api_key:
                    result["api_key"] = api_key
                self.dismiss(result)
            else:
                self.dismiss(None)

        def action_cancel(self) -> None:
            self.dismiss(None)

except ImportError:
    pass
