"""Tests for axio.types: Usage, StopReason, ToolName, ToolCallID."""

import pytest

from axio.types import StopReason, Usage


class TestUsage:
    def test_add(self) -> None:
        a = Usage(10, 5)
        b = Usage(3, 7)
        assert a + b == Usage(13, 12)

    def test_add_associative(self) -> None:
        a, b, c = Usage(1, 2), Usage(3, 4), Usage(5, 6)
        assert (a + b) + c == a + (b + c)

    def test_add_commutative(self) -> None:
        a = Usage(10, 5)
        b = Usage(3, 7)
        assert a + b == b + a

    def test_frozen(self) -> None:
        u = Usage(1, 2)
        with pytest.raises(AttributeError):
            u.input_tokens = 99  # type: ignore[misc]

    def test_identity(self) -> None:
        zero = Usage(0, 0)
        a = Usage(10, 5)
        assert a + zero == a


class TestStopReason:
    def test_values(self) -> None:
        assert set(StopReason) == {
            StopReason.end_turn,
            StopReason.tool_use,
            StopReason.max_tokens,
            StopReason.error,
            StopReason.refusal,
            StopReason.pause_turn,
            StopReason.context_window_exceeded,
            StopReason.cancelled,
        }

    def test_is_str(self) -> None:
        assert isinstance(StopReason.end_turn, str)

    def test_str_values(self) -> None:
        assert StopReason.end_turn == "end_turn"
        assert StopReason.tool_use == "tool_use"
        assert StopReason.max_tokens == "max_tokens"
        assert StopReason.error == "error"


class TestAliases:
    def test_tool_name_is_str(self) -> None:
        name: str = "my_tool"
        assert isinstance(name, str)

    def test_tool_call_id_is_str(self) -> None:
        call_id: str = "call_123"
        assert isinstance(call_id, str)


class TestUsageDetail:
    def test_the_slices_stay_inside_their_totals(self) -> None:
        # The one rule every transport converts into. A slice that escaped its total would make
        # uncached_input_tokens negative, and any cost computed from it nonsense.
        usage = Usage(100, 50, cache_read_tokens=70, cache_write_tokens=10, reasoning_tokens=40)

        # The derived figures, which are what a cost is computed from. Asserting the slices against
        # the totals instead was arithmetic on this test's own literals and could never fail.
        assert usage.uncached_input_tokens == 20
        assert usage.answer_tokens == 10
        assert usage.total_tokens == 150

    def test_a_slice_that_escaped_its_total_shows_up_in_what_is_derived(self) -> None:
        # A transport that reads a provider's cache as outside the input when it is inside reports
        # a hundred-thousand-token prompt as a handful, and this is where that surfaces.
        wrong = Usage(100, 50, cache_read_tokens=900)

        assert wrong.uncached_input_tokens < 0, "a negative remainder is how the mistake is visible"

    def test_addition_carries_every_slice(self) -> None:
        first = Usage(10, 5, cache_read_tokens=4, cache_write_tokens=2, reasoning_tokens=3)
        second = Usage(20, 7, cache_read_tokens=1, cache_write_tokens=6, reasoning_tokens=2)
        assert first + second == Usage(30, 12, cache_read_tokens=5, cache_write_tokens=8, reasoning_tokens=5)

    def test_the_invariant_survives_accumulation(self) -> None:
        # Componentwise sums preserve a linear inequality, so the derived counts never go negative.
        total = Usage(0, 0)
        for _ in range(50):
            total = total + Usage(10, 5, cache_read_tokens=7, cache_write_tokens=1, reasoning_tokens=4)
        assert total.uncached_input_tokens == 100
        assert total.answer_tokens == 50

    def test_a_two_argument_usage_still_means_what_it_did(self) -> None:
        # Every existing caller builds Usage this way; the slices default to nothing known.
        usage = Usage(input_tokens=10, output_tokens=5)
        assert (usage.cache_read_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (0, 0, 0)
        assert usage.uncached_input_tokens == 10 and usage.answer_tokens == 5
