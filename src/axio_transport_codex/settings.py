"""Textual settings screen for ChatGPT (Codex) transport."""

from __future__ import annotations

try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal
    from textual.screen import ModalScreen
    from textual.widgets import Button, Static

    class CodexSettingsScreen(ModalScreen[dict[str, str] | None]):
        """Sign in / sign out screen for ChatGPT OAuth."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]
        CSS = """
        CodexSettingsScreen { align: center middle; }
        #codex-settings {
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
            self._signed_in = bool(settings.get("api_key"))

        def compose(self) -> ComposeResult:
            with Container(id="codex-settings"):
                yield Static("[bold]ChatGPT (Codex) Settings[/]")
                if self._signed_in:
                    account = self._settings.get("account_id", "unknown")
                    yield Static(f"Signed in (account: {account[:16]}...)", classes="field-label")
                    with Horizontal(classes="settings-buttons"):
                        yield Button("Sign Out", id="btn-signout", variant="warning")
                        yield Button("Cancel", id="btn-cancel")
                else:
                    yield Static(
                        "Sign in with your ChatGPT account to use this transport.",
                        classes="field-label",
                    )
                    with Horizontal(classes="settings-buttons"):
                        yield Button("Sign in with ChatGPT", id="btn-signin", variant="primary")
                        yield Button("Cancel", id="btn-cancel")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-signin":
                self.run_worker(self._do_signin(), exclusive=True)
            elif event.button.id == "btn-signout":
                self.dismiss({})
            else:
                self.dismiss(None)

        async def _do_signin(self) -> None:
            from .oauth import run_oauth_flow

            try:
                tokens = await run_oauth_flow()
                # Map access_token → api_key for registry compatibility
                settings: dict[str, str] = {
                    "api_key": tokens["access_token"],
                    "refresh_token": tokens.get("refresh_token", ""),
                    "expires_at": tokens.get("expires_at", ""),
                    "account_id": tokens.get("account_id", ""),
                }
                self.dismiss(settings)
            except Exception as exc:
                self.notify(f"Sign-in failed: {exc}", severity="error")

        def action_cancel(self) -> None:
            self.dismiss(None)

except ImportError:
    pass
