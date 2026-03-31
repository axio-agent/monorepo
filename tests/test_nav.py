"""Tests for chat message navigation (Alt+Up/Down)."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, Static

from axio_tui.app import _ToolStatusWidget
from axio_tui.screens import ToolDetailScreen


class _NavApp(App[None]):
    """Minimal app with navigation logic matching AgentApp."""

    CSS = """
    #log { height: 1fr; }
    #log > .nav-selected {
        background: $accent 10%;
        border-left: thick $accent;
    }
    """

    def __init__(self, child_count: int = 3, tool_at: int | None = None) -> None:
        super().__init__()
        self._child_count = child_count
        self._tool_at = tool_at
        self._nav_index: int | None = None

    def compose(self) -> ComposeResult:
        log = VerticalScroll(id="log")
        log.can_focus = False
        yield log
        yield Input(id="input")

    async def on_mount(self) -> None:
        scroll = self.query_one("#log", VerticalScroll)
        for i in range(self._child_count):
            if i == self._tool_at:
                w = _ToolStatusWidget()
                await scroll.mount(w)
                w.track("t1", "shell")
                w.complete("t1", is_error=False, content="ok", tool_input={"cmd": "ls"})
            else:
                await scroll.mount(Static(f"Message {i}", classes="meta"))
        self.query_one("#input", Input).focus()

    def _nav_children(self) -> list[Widget]:
        return list(self.query_one("#log").children)

    def _nav_select(self, index: int | None) -> None:
        children = self._nav_children()
        if self._nav_index is not None and self._nav_index < len(children):
            children[self._nav_index].remove_class("nav-selected")
        self._nav_index = index
        if index is not None and index < len(children):
            children[index].add_class("nav-selected")
            children[index].scroll_visible()

    def _nav_exit(self) -> None:
        self._nav_select(None)
        self.query_one("#input", Input).focus()

    def _nav_activate(self) -> None:
        if self._nav_index is None:
            return
        children = self._nav_children()
        if not children or self._nav_index >= len(children):
            return
        widget = children[self._nav_index]
        if isinstance(widget, _ToolStatusWidget):
            for tid, info in widget._tools.items():
                if info.status is not None:
                    widget.action_show_detail(tid)
                    return

    def action_nav_up(self) -> None:
        children = self._nav_children()
        if not children:
            return
        if self._nav_index is None:
            self._nav_select(len(children) - 1)
        elif self._nav_index > 0:
            self._nav_select(self._nav_index - 1)

    def action_nav_down(self) -> None:
        children = self._nav_children()
        if not children:
            return
        if self._nav_index is None:
            self._nav_select(0)
        elif self._nav_index < len(children) - 1:
            self._nav_select(self._nav_index + 1)
        else:
            self._nav_exit()

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._nav_index is None:
            return
        if event.key == "escape":
            self._nav_exit()
            event.prevent_default()
            event.stop()
        elif event.key == "enter":
            self._nav_activate()
            event.prevent_default()
            event.stop()


class TestNavUp:
    async def test_selects_last_from_idle(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app.action_nav_up()
            assert app._nav_index == 2

    async def test_moves_up(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app.action_nav_up()
            app.action_nav_up()
            assert app._nav_index == 1

    async def test_stays_at_top(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(0)
            app.action_nav_up()
            assert app._nav_index == 0

    async def test_noop_on_empty(self) -> None:
        app = _NavApp(0)
        async with app.run_test():
            app.action_nav_up()
            assert app._nav_index is None


class TestNavDown:
    async def test_selects_first_from_idle(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app.action_nav_down()
            assert app._nav_index == 0

    async def test_moves_down(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app.action_nav_down()
            app.action_nav_down()
            assert app._nav_index == 1

    async def test_exits_past_last(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(2)
            app.action_nav_down()
            assert app._nav_index is None

    async def test_noop_on_empty(self) -> None:
        app = _NavApp(0)
        async with app.run_test():
            app.action_nav_down()
            assert app._nav_index is None


class TestNavSelect:
    async def test_adds_class(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(1)
            children = app._nav_children()
            assert children[1].has_class("nav-selected")

    async def test_removes_old_class(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(0)
            app._nav_select(1)
            children = app._nav_children()
            assert not children[0].has_class("nav-selected")
            assert children[1].has_class("nav-selected")

    async def test_none_clears(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(1)
            app._nav_select(None)
            assert not any(c.has_class("nav-selected") for c in app._nav_children())


class TestNavExit:
    async def test_resets_index(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(1)
            app._nav_exit()
            assert app._nav_index is None

    async def test_refocuses_input(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(1)
            app._nav_exit()
            assert app.query_one("#input", Input).has_focus


class TestNavActivate:
    async def test_opens_tool_detail(self) -> None:
        app = _NavApp(3, tool_at=1)
        async with app.run_test():
            app._nav_select(1)
            app._nav_activate()
            assert isinstance(app.screen, ToolDetailScreen)

    async def test_noop_on_text(self) -> None:
        app = _NavApp(3)
        async with app.run_test():
            app._nav_select(0)
            app._nav_activate()
            assert not isinstance(app.screen, ToolDetailScreen)


class TestNavKeyboard:
    async def test_escape_exits_nav(self) -> None:
        app = _NavApp(3)
        async with app.run_test() as pilot:
            app._nav_select(1)
            await pilot.press("escape")
            assert app._nav_index is None

    async def test_enter_activates_tool(self) -> None:
        app = _NavApp(3, tool_at=1)
        async with app.run_test() as pilot:
            app._nav_select(1)
            await pilot.press("enter")
            assert isinstance(app.screen, ToolDetailScreen)
