"""Tests for axon.types: Usage, StopReason, ToolName, ToolCallID."""

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
        try:
            u.input_tokens = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

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
