"""Settings screen for Anthropic transport."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class AnthropicSettingsScreen(ModalScreen[dict[str, str] | None]):
    """Editable settings form for Anthropic transport: api_key, base_url."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    AnthropicSettingsScreen { align: center middle; }
    #anthropic-settings {
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
        with Container(id="anthropic-settings"):
            yield Static("[bold]Anthropic Settings[/]")
            yield Static("API Key (leave blank to use ANTHROPIC_API_KEY):", classes="field-label")
            yield Input(
                value=self._settings.get("api_key", ""),
                id="api-key",
                password=True,
            )
            yield Static("Base URL (leave blank to use ANTHROPIC_BASE_URL or default):", classes="field-label")
            yield Input(
                value=self._settings.get("base_url", ""),
                placeholder="https://api.anthropic.com/v1",
                id="base-url",
            )
            with Horizontal(classes="settings-buttons"):
                yield Button("Save", id="btn-save", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#api-key", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            result: dict[str, str] = {}
            api_key = self.query_one("#api-key", Input).value.strip()
            if api_key:
                result["api_key"] = api_key
            base_url = self.query_one("#base-url", Input).value.strip()
            if base_url:
                result["base_url"] = base_url
            self.dismiss(result)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
