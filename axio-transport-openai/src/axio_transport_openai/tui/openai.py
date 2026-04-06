"""Settings screen for OpenAI transport."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class OpenAISettingsScreen(ModalScreen[dict[str, str] | None]):
    """Editable settings form for OpenAI transport: base_url and api_key."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    OpenAISettingsScreen { align: center middle; }
    #openai-settings {
        width: 70; height: auto; border: heavy $accent;
        background: $panel; padding: 1 2;
    }
    .field-label { margin-top: 1; }
    .settings-buttons { height: auto; margin-top: 1; }
    .settings-buttons Button { margin: 0 1; }
    """

    def __init__(self, settings: dict[str, str]) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        with Container(id="openai-settings"):
            yield Static("[bold]OpenAI Settings[/]")
            yield Static("Base URL (leave blank to use OPENAI_BASE_URL or default):", classes="field-label")
            yield Input(
                value=self._settings.get("base_url", ""),
                placeholder="https://api.openai.com/v1",
                id="base-url",
            )
            yield Static("API Key (leave blank to use OPENAI_API_KEY):", classes="field-label")
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
