import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from axio import background
from axio.models import ModelSpec
from axio.types import Usage

from axio_repl import _panel


class _Record:
    def __init__(self, agent_id: str, name: str) -> None:
        self.id = agent_id
        self.name = name


@pytest.fixture(autouse=True)
async def clean_registry() -> AsyncGenerator[None, None]:
    yield
    await background.cancel_all()


def test_silent_when_nothing_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)
    # A status line that always says something wastes a row of terminal.
    assert _panel.agent_summary() == ""


def test_counts_agents_by_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _panel,
        "local_background_agent_records",
        lambda: [_Record("a", "one"), _Record("b", "two"), _Record("c", "three")],
    )
    states = {"a": ("running", None), "b": ("running", None), "c": ("idle", None)}
    monkeypatch.setattr(_panel, "background_agent_state", lambda i: states[i])
    assert _panel.agent_summary() == "agents: 1 idle, 2 running"


def test_names_the_failed_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", lambda: [_Record("a", "analyst")])
    monkeypatch.setattr(_panel, "background_agent_state", lambda i: ("idle", "StreamError: boom"))
    summary = _panel.agent_summary()
    assert "failed: analyst" in summary


@pytest.mark.asyncio
async def test_reports_detached_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)

    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(60)
        return "never"

    background.start("shell", slow())
    await started.wait()
    assert "tasks: 1 running" in _panel.agent_summary()


@pytest.mark.asyncio
async def test_finished_calls_are_flagged_until_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)

    async def quick() -> str:
        return "done"

    handle = background.start("shell", quick())
    await background.get(handle).task  # type: ignore[union-attr]
    assert "1 ready to collect" in _panel.agent_summary()

    background.describe(handle)  # reading it marks it collected
    assert "ready to collect" not in _panel.agent_summary()


_M3 = ModelSpec(id="MiniMaxAI/MiniMax-M3", context_window=1_000_000, input_cost=0.3, output_cost=1.2)


def test_a_fresh_session_still_says_which_model() -> None:
    line = _panel.status_line(_M3, _panel.SessionStats())

    assert "MiniMaxAI/MiniMax-M3" in line
    assert "ctx 0/1M" in line
    assert "$0" in line


def test_tokens_of_every_agent_are_counted() -> None:
    stats = _panel.SessionStats()

    stats.record("main", Usage(88_622, 5_169), _M3)
    stats.record("child-1", Usage(19_666, 1_908), _M3)

    assert stats.input_tokens == 108_288
    assert stats.output_tokens == 7_077
    assert "108.3k in / 7.1k out" in _panel.status_line(_M3, stats)


def test_only_the_main_agent_sets_how_full_the_window_is() -> None:
    # A child's prompt is its own context, not the one the user is filling.
    stats = _panel.SessionStats()

    stats.record("main", Usage(30_000, 10), _M3)
    stats.record("child-1", Usage(900_000, 10), _M3)

    assert stats.context_tokens == 30_000


def test_the_window_shrinks_when_the_context_does() -> None:
    # Compaction is the point: the latest prompt is what occupies the window,
    # not the largest one ever sent.
    stats = _panel.SessionStats()

    stats.record("main", Usage(500_000, 10), _M3)
    stats.record("main", Usage(20_000, 10), _M3)

    assert stats.context_tokens == 20_000


def test_cost_follows_the_model_that_was_charged() -> None:
    stats = _panel.SessionStats()
    cheap = ModelSpec(id="cheap", input_cost=0.1, output_cost=0.2)

    stats.record("main", Usage(1_000_000, 1_000_000), _M3)
    stats.record("child", Usage(1_000_000, 1_000_000), cheap)

    assert stats.cost == pytest.approx(0.3 + 1.2 + 0.1 + 0.2)
    assert stats.per_model["cheap"] == Usage(1_000_000, 1_000_000)


def test_usage_without_a_model_still_counts_tokens() -> None:
    stats = _panel.SessionStats()

    stats.record("main", Usage(10, 20), None)

    assert (stats.input_tokens, stats.output_tokens, stats.cost) == (10, 20, 0.0)


def test_counts_stay_short_enough_for_one_line() -> None:
    assert _panel.compact(0) == "0"
    assert _panel.compact(999) == "999"
    assert _panel.compact(1_000) == "1k"
    assert _panel.compact(8_587) == "8.6k"
    assert _panel.compact(1_000_000) == "1M"
    assert _panel.compact(10_123_456) == "10.12M"


def test_a_cost_too_small_to_show_is_still_shown() -> None:
    assert _panel.format_cost(0) == "$0"
    assert _panel.format_cost(0.0026) == "$0.0026"
    assert _panel.format_cost(4.0067) == "$4.007"
    assert _panel.format_cost(123.456) == "$123.46"


def test_the_toolbar_is_not_reverse_video() -> None:
    # Its default is, which paints a solid white band across the terminal.
    from prompt_toolkit.styles import default_ui_style, merge_styles

    session = _panel.make_session(lambda: "x")
    merged = merge_styles([default_ui_style(), session.style])

    assert default_ui_style().get_attrs_for_style_str("class:bottom-toolbar").reverse is True
    for cls in ("class:bottom-toolbar", "class:bottom-toolbar.text"):
        attrs = merged.get_attrs_for_style_str(cls)
        assert attrs.reverse is False, cls
        assert attrs.bgcolor == "default", cls


async def test_ctrl_c_at_the_prompt_interrupts_instead_of_ending_the_read() -> None:
    # The prompt is up for the whole session now, and it puts the terminal in
    # raw mode - so Ctrl+C arrives as a keypress, not as SIGINT, and the handler
    # that used to stop a running turn never fires.
    from axio_repl import ReplRenderer, _read_input_async

    interrupts: list[int] = []
    answers: list[object] = [KeyboardInterrupt(), KeyboardInterrupt(), "carry on"]

    class _Session:
        async def prompt_async(self, prompt: str) -> str:
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return str(answer)

    result = await _read_input_async(_Session(), ReplRenderer(), lambda: interrupts.append(1))

    assert result == "carry on"
    assert len(interrupts) == 2


async def test_escape_interrupts_and_sends_while_enter_only_sends() -> None:
    # Enter is "and then this", which belongs after the answer being written.
    # Escape is "stop, do this instead", worth nothing once the thing being
    # stopped has finished.
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    interrupts: list[int] = []

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", on_interrupt=lambda: interrupts.append(1))

            pipe.send_text("sent with enter\r")
            assert await session.prompt_async("repl> ") == "sent with enter"
            assert interrupts == []

            pipe.send_text("stop, do this\x1b")
            assert await session.prompt_async("repl> ") == "stop, do this"
            assert len(interrupts) == 1

            pipe.send_text("\x1b")
            assert await session.prompt_async("repl> ") == ""
            assert len(interrupts) == 2


async def test_up_recalls_the_last_message_for_editing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    # Never the user's own history file.
    monkeypatch.setattr(_panel, "HISTORY_PATH", tmp_path / "history")

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status")

            pipe.send_text("first message\x1b")
            assert await session.prompt_async("repl> ") == "first message"

            pipe.send_text("\x1b[A")
            pipe.send_text(" and more\x1b")
            assert await session.prompt_async("repl> ") == "first message and more"
