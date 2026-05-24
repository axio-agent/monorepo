from __future__ import annotations

import pytest
from axio.events import SessionEndEvent, TextDelta
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
