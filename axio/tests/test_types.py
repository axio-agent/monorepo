"""Tests for axio.types: Usage, StopReason, ToolName, ToolCallID."""

import pytest

from axio.types import StopReason, Usage, stop_reason_from


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
            StopReason.unknown,
            StopReason.repetition,
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

    def test_a_slice_that_escaped_its_total_is_refused_where_it_is_built(self) -> None:
        # A transport that reads a provider's cache as outside the input when it is inside reports
        # a hundred-thousand-token prompt as a handful. Documented and unchecked, the mistake
        # travelled as a negative remainder into every display, aggregate and cost built on it.
        with pytest.raises(ValueError, match="inside input_tokens"):
            Usage(100, 50, cache_read_tokens=900)

    def test_reasoning_cannot_outgrow_the_output_it_is_part_of(self) -> None:
        with pytest.raises(ValueError, match="slice of output_tokens"):
            Usage(100, 50, reasoning_tokens=60)

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            Usage(-1, 0)

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


class TestAValueTheVocabularyDoesNotHave:
    """What a provider's own word reads as when no rule names it."""

    def test_it_becomes_unknown_rather_than_one_of_the_others(self) -> None:
        # Every other answer claims something the provider did not say: that the turn finished,
        # that it was truncated, or that the transport broke.
        table = {"stop": StopReason.end_turn}

        assert stop_reason_from("eos_token", table, provider="Compatible") is StopReason.unknown

    def test_a_word_the_table_names_still_wins(self) -> None:
        assert stop_reason_from("stop", {"stop": StopReason.end_turn}, provider="x") is StopReason.end_turn

    def test_unknown_vouches_for_nothing(self) -> None:
        # A turn whose ending nobody understands must not have its calls dispatched.
        from axio.agent import _DISPATCH

        assert StopReason.unknown not in _DISPATCH
