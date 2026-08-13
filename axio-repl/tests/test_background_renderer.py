from __future__ import annotations

import pytest
from axio.events import Error, SessionEndEvent, TextDelta
from axio.types import StopReason, Usage

from axio_repl import ReplRenderer


async def test_one_shot_renderer_replays_buffered_background_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(buffer_background_events=True)

    await renderer.render("child", TextDelta(index=0, delta="background report"))
    await renderer.render(
        "child",
        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(input_tokens=1, output_tokens=2)),
    )

    assert capsys.readouterr().out == ""

    renderer.set_focus("child")

    output = capsys.readouterr().out
    assert "background report" in output
    assert "[1in/2out tokens]" in output


async def test_one_shot_renderer_replays_each_buffered_background_agent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(buffer_background_events=True)

    await renderer.render("child-a", TextDelta(index=0, delta="first report"))
    await renderer.render("child-b", TextDelta(index=0, delta="second report"))

    assert capsys.readouterr().out == ""

    renderer.set_focus("child-a")
    first_output = capsys.readouterr().out
    assert "first report" in first_output
    assert "second report" not in first_output

    renderer.set_focus("child-b")
    second_output = capsys.readouterr().out
    assert "second report" in second_output


async def test_a_finished_background_agent_reports_its_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The failure this closes: the agent wrote 6000 characters, the parent was
    # told only how many, and had to ask it to say the whole thing again.
    reports: list[tuple[str, str]] = []
    renderer = ReplRenderer(on_background_report=lambda agent_id, text: reports.append((agent_id, text)))

    await renderer.render("child", TextDelta(index=0, delta="## Report\n"))
    await renderer.render("child", TextDelta(index=0, delta="all good"))
    assert reports == []

    await renderer.render(
        "child",
        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(input_tokens=1, output_tokens=2)),
    )

    assert reports == [("child", "## Report\nall good")]
    assert "reported 18 chars" in capsys.readouterr().out


async def test_a_silent_background_agent_reports_nothing() -> None:
    reports: list[tuple[str, str]] = []
    renderer = ReplRenderer(on_background_report=lambda agent_id, text: reports.append((agent_id, text)))

    await renderer.render(
        "child",
        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(input_tokens=1, output_tokens=0)),
    )

    assert reports == []


async def test_a_background_failure_is_reported_with_its_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # "errors=1" says something went wrong and not what, which is the half that
    # decides whether to retry, fix, or give up.
    renderer = ReplRenderer()

    await renderer.render("child", Error(exception=RuntimeError("Stopped after 25 iterations")))
    await renderer.render(
        "child",
        SessionEndEvent(stop_reason=StopReason.error, total_usage=Usage(input_tokens=1, output_tokens=0)),
    )

    assert "Stopped after 25 iterations" in capsys.readouterr().out
