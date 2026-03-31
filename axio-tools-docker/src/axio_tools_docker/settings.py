"""TUI settings screen for Docker sandbox configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal
    from textual.screen import ModalScreen
    from textual.widgets import Button, Checkbox, Input, Static

    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False

if TYPE_CHECKING:
    from .config import SandboxConfig
    from .manager import SandboxManager


if _HAS_TEXTUAL:

    class DockerSettingsScreen(ModalScreen[None]):
        """Settings screen for Docker sandbox configuration."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]
        CSS = """
        DockerSettingsScreen { align: center middle; }
        #docker-settings {
            width: 70;
            height: auto;
            max-height: 90%;
            border: heavy $accent;
            background: $panel;
            padding: 1 2;
        }
        #docker-settings Input { margin-bottom: 1; }
        #docker-settings Checkbox { margin-bottom: 1; }
        .docker-buttons { height: auto; margin-top: 1; }
        .docker-buttons Button { margin: 0 1; }
        .docker-container-actions { height: auto; margin-top: 1; margin-bottom: 1; }
        .docker-container-actions Button { margin: 0 1; }
        """

        def __init__(
            self,
            manager: SandboxManager,
            config: Any = None,
            global_config: Any = None,
        ) -> None:
            super().__init__()
            self._manager = manager
            self._config = config
            self._global_config = global_config

        def compose(self) -> ComposeResult:
            cfg = self._manager.config
            docker_ok = self._manager.docker_available()
            status = "[green]Docker available[/]" if docker_ok else "[red]Docker not found[/]"
            running = self._manager.container_running

            with Container(id="docker-settings"):
                yield Static("[bold]Docker Sandbox Settings[/]")
                yield Static(f"Status: {status}", id="docker-status")
                yield Static(
                    "Container: [green]running[/]" if running else "Container: [dim]stopped[/]",
                    id="docker-container-status",
                )
                with Horizontal(classes="docker-container-actions"):
                    yield Button("Prepare", id="btn-prepare", variant="success", disabled=running)
                    yield Button("Stop", id="btn-stop", variant="error", disabled=not running)
                    yield Button("Recreate", id="btn-recreate", variant="warning", disabled=not running)
                yield Static("Image:")
                yield Input(value=cfg.image, placeholder="e.g. python:latest", id="docker-image")
                yield Static("Memory limit:")
                yield Input(value=cfg.memory, placeholder="e.g. 256m", id="docker-memory")
                yield Static("CPU limit:")
                yield Input(value=cfg.cpus, placeholder="e.g. 1.0", id="docker-cpus")
                yield Checkbox("Allow network access", cfg.network, id="docker-network")
                with Horizontal(classes="docker-buttons"):
                    yield Button("Save", id="btn-save", variant="primary")
                    yield Button("Cancel", id="btn-cancel")

        def on_mount(self) -> None:
            self.query_one("#docker-image", Input).focus()

        def _update_container_status(self) -> None:
            running = self._manager.container_running
            status_widget = self.query_one("#docker-container-status", Static)
            status_widget.update(
                "Container: [green]running[/]" if running else "Container: [dim]stopped[/]",
            )
            self.query_one("#btn-prepare", Button).disabled = running
            self.query_one("#btn-stop", Button).disabled = not running
            self.query_one("#btn-recreate", Button).disabled = not running

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-cancel":
                self.dismiss(None)
            elif event.button.id == "btn-save":
                self._save()
            elif event.button.id == "btn-prepare":
                self.app.run_worker(self._do_prepare())
            elif event.button.id == "btn-stop":
                self.app.run_worker(self._do_stop())
            elif event.button.id == "btn-recreate":
                self.app.run_worker(self._do_recreate())

        async def _do_prepare(self) -> None:
            self.notify("Preparing container...")
            try:
                await self._manager._ensure_container()
                self._update_container_status()
                self.notify("Container ready")
            except RuntimeError as exc:
                self.notify(str(exc), severity="error")

        async def _do_stop(self) -> None:
            await self._manager.close()
            self._update_container_status()
            self.notify("Container stopped")

        async def _do_recreate(self) -> None:
            self.notify("Recreating container...")
            try:
                await self._manager.recreate()
                self._update_container_status()
                self.notify("Container recreated")
            except RuntimeError as exc:
                self.notify(str(exc), severity="error")
                self._update_container_status()

        def _save(self) -> None:
            from .config import SandboxConfig

            image = self.query_one("#docker-image", Input).value.strip()
            memory = self.query_one("#docker-memory", Input).value.strip()
            cpus = self.query_one("#docker-cpus", Input).value.strip()
            network = self.query_one("#docker-network", Checkbox).value

            cfg = SandboxConfig(
                image=image or "python:latest",
                memory=memory or "256m",
                cpus=cpus or "1.0",
                network=network,
            )
            self._manager.config = cfg
            self.app.run_worker(self._persist(cfg))
            self.dismiss(None)

        async def _persist(self, cfg: SandboxConfig) -> None:
            db = self._config or self._global_config
            if db is None:
                return
            await db.delete_prefix("docker.")
            for key, value in cfg.to_dict().items():
                await db.set(f"docker.{key}", value)

        def action_cancel(self) -> None:
            self.dismiss(None)

else:

    class DockerSettingsScreen:  # type: ignore[no-redef]
        """Stub when textual is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("textual is required for Docker settings screen")
