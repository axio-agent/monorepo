"""Guard dialog screens for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class PathGuardDialog(ModalScreen[str]):
    """Modal dialog for path access permission - buttons only."""

    BINDINGS = [
        Binding("a", "allow", "Allow", show=False),
        Binding("d", "deny", "Deny", show=False),
        Binding("escape", "deny", "Deny", show=False),
        Binding("left", "focus_prev_button", show=False),
        Binding("right", "focus_next_button", show=False),
    ]
    CSS = """
    PathGuardDialog { align: center middle; }
    #path-guard-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        border: heavy $warning;
        background: $panel;
        padding: 1 2;
    }
    .guard-buttons { height: auto; margin-top: 1; }
    .guard-buttons Button { margin: 0 1; }
    .guard-prompt { color: $warning; }
    """

    def __init__(self, prompt_text: str) -> None:
        super().__init__()
        self._prompt_text = prompt_text

    def compose(self) -> ComposeResult:
        prompt = self._prompt_text[:5000] + "..." if len(self._prompt_text) > 5000 else self._prompt_text
        with Container(id="path-guard-dialog"):
            yield Static("[bold]Path Access Request[/]")
            yield Static(prompt, markup=False, classes="guard-prompt")
            with Horizontal(classes="guard-buttons"):
                yield Button("Allow", id="btn-allow", variant="success")
                yield Button("Deny", id="btn-deny", variant="error")
                yield Button("Always Deny", id="btn-always-deny", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#btn-allow", Button).focus()

    def _cycle_buttons(self, direction: int) -> None:
        buttons = list(self.query(Button))
        if not buttons:
            return
        try:
            idx = buttons.index(self.focused)  # type: ignore[arg-type]
        except ValueError:
            buttons[0].focus()
            return
        buttons[(idx + direction) % len(buttons)].focus()

    def action_focus_next_button(self) -> None:
        self._cycle_buttons(1)

    def action_focus_prev_button(self) -> None:
        self._cycle_buttons(-1)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-allow":
                self.dismiss("y")
            case "btn-deny":
                self.dismiss("n")
            case "btn-always-deny":
                self.dismiss("deny")

    def action_allow(self) -> None:
        self.dismiss("y")

    def action_deny(self) -> None:
        self.dismiss("n")


class LLMGuardDialog(ModalScreen[str]):
    """Modal dialog for LLM safety review - buttons + text input."""

    BINDINGS = [
        Binding("escape", "deny", "Deny", show=False),
        Binding("left", "focus_prev_button", show=False),
        Binding("right", "focus_next_button", show=False),
    ]
    CSS = """
    LLMGuardDialog { align: center middle; }
    #llm-guard-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        border: heavy $warning;
        background: $panel;
        padding: 1 2;
    }
    .guard-buttons { height: auto; margin-top: 1; }
    .guard-buttons Button { margin: 0 1; }
    #guard-reason { margin-top: 1; }
    .guard-prompt { color: $warning; }
    """

    def __init__(self, prompt_text: str) -> None:
        super().__init__()
        self._prompt_text = prompt_text

    def compose(self) -> ComposeResult:
        prompt = self._prompt_text[:5000] + "..." if len(self._prompt_text) > 5000 else self._prompt_text
        with Container(id="llm-guard-dialog"):
            yield Static("[bold]Safety Review[/]")
            yield Static(prompt, markup=False, classes="guard-prompt")
            with Horizontal(classes="guard-buttons"):
                yield Button("Allow", id="btn-allow", variant="success")
                yield Button("Always Allow", id="btn-always", variant="primary")
                yield Button("Deny", id="btn-deny", variant="error")
            yield Input(placeholder="Custom reason...", id="guard-reason")

    def on_mount(self) -> None:
        self.query_one("#btn-allow", Button).focus()

    def _cycle_buttons(self, direction: int) -> None:
        buttons = list(self.query(Button))
        if not buttons:
            return
        try:
            idx = buttons.index(self.focused)  # type: ignore[arg-type]
        except ValueError:
            buttons[0].focus()
            return
        buttons[(idx + direction) % len(buttons)].focus()

    def action_focus_next_button(self) -> None:
        self._cycle_buttons(1)

    def action_focus_prev_button(self) -> None:
        self._cycle_buttons(-1)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-allow":
                self.dismiss("y")
            case "btn-always":
                self.dismiss("always")
            case "btn-deny":
                self.dismiss("n")

    def on_input_submitted(self, message: Input.Submitted) -> None:
        if message.value.strip():
            self.dismiss(message.value.strip())

    def action_deny(self) -> None:
        self.dismiss("n")
