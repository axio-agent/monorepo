"""Tests for _ToolStatusWidget and _ToolCallInfo."""

from __future__ import annotations

from axio_tui.app import _ToolCallInfo, _ToolStatusWidget


class TestToolCallInfo:
    def test_defaults(self) -> None:
        info = _ToolCallInfo()
        assert info.name == ""
        assert info.status is None
        assert info.input == {}
        assert info.content == ""

    def test_mutable(self) -> None:
        info = _ToolCallInfo(name="shell")
        info.status = True
        info.content = "output"
        info.input = {"cmd": "ls"}
        assert info.status is True
        assert info.content == "output"
        assert info.input == {"cmd": "ls"}


class TestToolStatusWidget:
    def test_track_creates_pending_entry(self) -> None:
        w = _ToolStatusWidget()
        w.track("c1", "shell")
        assert "c1" in w._tools
        assert w._tools["c1"].name == "shell"
        assert w._tools["c1"].status is None

    def test_complete_updates_status(self) -> None:
        w = _ToolStatusWidget()
        w.track("c1", "shell")
        w.complete("c1", is_error=False, content="ok", tool_input={"cmd": "ls"})
        info = w._tools["c1"]
        assert info.status is True
        assert info.content == "ok"
        assert info.input == {"cmd": "ls"}

    def test_complete_error(self) -> None:
        w = _ToolStatusWidget()
        w.track("c1", "bad")
        w.complete("c1", is_error=True, content="boom")
        info = w._tools["c1"]
        assert info.status is False
        assert info.content == "boom"

    def test_complete_unknown_id_ignored(self) -> None:
        w = _ToolStatusWidget()
        w.complete("unknown", is_error=False, content="ok")
        assert "unknown" not in w._tools

    def test_multiple_tools(self) -> None:
        w = _ToolStatusWidget()
        w.track("c1", "shell")
        w.track("c2", "read_file")
        w.complete("c1", is_error=False, content="ok", tool_input={"cmd": "ls"})
        w.complete("c2", is_error=True, content="not found")
        assert w._tools["c1"].status is True
        assert w._tools["c2"].status is False

    def test_complete_default_input(self) -> None:
        w = _ToolStatusWidget()
        w.track("c1", "shell")
        w.complete("c1", is_error=False)
        assert w._tools["c1"].input == {}
        assert w._tools["c1"].content == ""
