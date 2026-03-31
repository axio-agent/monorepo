"""Tests for axon.blocks: all content block types."""

import pytest

from axio.blocks import ContentBlock, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock, from_dict, to_dict
from axio.messages import Message


class TestTextBlock:
    def test_frozen(self) -> None:
        b = TextBlock(text="hello")
        with pytest.raises(AttributeError):
            b.text = "bye"  # type: ignore[misc]

    def test_hashable(self) -> None:
        b = TextBlock(text="hello")
        assert hash(b) is not None
        assert {b}  # usable in sets

    def test_equality(self) -> None:
        assert TextBlock(text="a") == TextBlock(text="a")
        assert TextBlock(text="a") != TextBlock(text="b")


class TestImageBlock:
    @pytest.mark.parametrize("media_type", ["image/jpeg", "image/png", "image/gif", "image/webp"])
    def test_media_types(self, media_type: str) -> None:
        b = ImageBlock(media_type=media_type, data=b"\x00")  # type: ignore[arg-type]
        assert b.media_type == media_type

    def test_frozen(self) -> None:
        b = ImageBlock(media_type="image/png", data=b"\x00")
        with pytest.raises(AttributeError):
            b.data = b"\x01"  # type: ignore[misc]

    def test_hashable(self) -> None:
        b = ImageBlock(media_type="image/png", data=b"\x00")
        assert hash(b) is not None


class TestToolUseBlock:
    def test_frozen(self) -> None:
        b = ToolUseBlock(id="c1", name="echo", input={"x": 1})
        with pytest.raises(AttributeError):
            b.name = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        b = ToolUseBlock(id="c1", name="echo", input={"x": 1})
        assert b.id == "c1"
        assert b.name == "echo"
        assert b.input == {"x": 1}


class TestToolResultBlock:
    def test_default_is_error(self) -> None:
        b = ToolResultBlock(tool_use_id="c1", content="ok")
        assert b.is_error is False

    def test_error(self) -> None:
        b = ToolResultBlock(tool_use_id="c1", content="fail", is_error=True)
        assert b.is_error is True

    def test_content_list(self) -> None:
        b = ToolResultBlock(tool_use_id="c1", content=[TextBlock(text="hello")])
        assert isinstance(b.content, list)

    def test_frozen(self) -> None:
        b = ToolResultBlock(tool_use_id="c1", content="ok")
        with pytest.raises(AttributeError):
            b.content = "new"  # type: ignore[misc]


class TestContentBlockBase:
    def test_all_types_are_subclass(self) -> None:
        blocks: list[ContentBlock] = [
            TextBlock(text="hi"),
            ImageBlock(media_type="image/png", data=b""),
            ToolUseBlock(id="c1", name="t", input={}),
            ToolResultBlock(tool_use_id="c1", content="ok"),
        ]
        for b in blocks:
            assert isinstance(b, ContentBlock)


class TestToDict:
    def test_text(self) -> None:
        assert to_dict(TextBlock(text="hi")) == {"type": "text", "text": "hi"}

    def test_image(self) -> None:
        d = to_dict(ImageBlock(media_type="image/png", data=b"\x89PNG"))
        assert d["type"] == "image"
        assert d["media_type"] == "image/png"
        assert isinstance(d["data"], str)  # base64 encoded

    def test_tool_use(self) -> None:
        d = to_dict(ToolUseBlock(id="c1", name="echo", input={"x": 1}))
        assert d == {"type": "tool_use", "id": "c1", "name": "echo", "input": {"x": 1}}

    def test_tool_result_str(self) -> None:
        d = to_dict(ToolResultBlock(tool_use_id="c1", content="ok"))
        assert d == {"type": "tool_result", "tool_use_id": "c1", "content": "ok", "is_error": False}

    def test_tool_result_nested(self) -> None:
        block = ToolResultBlock(
            tool_use_id="c1",
            content=[TextBlock(text="hi"), ImageBlock(media_type="image/png", data=b"\x00")],
        )
        d = to_dict(block)
        assert len(d["content"]) == 2
        assert d["content"][0] == {"type": "text", "text": "hi"}
        assert d["content"][1]["type"] == "image"

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unknown block type"):
            to_dict(ContentBlock())


class TestFromDict:
    def test_text(self) -> None:
        assert from_dict({"type": "text", "text": "hi"}) == TextBlock(text="hi")

    def test_image(self) -> None:
        block = ImageBlock(media_type="image/png", data=b"\x89PNG")
        assert from_dict(to_dict(block)) == block

    def test_tool_use(self) -> None:
        block = ToolUseBlock(id="c1", name="echo", input={"x": 1})
        assert from_dict(to_dict(block)) == block

    def test_tool_result_str(self) -> None:
        block = ToolResultBlock(tool_use_id="c1", content="ok", is_error=True)
        assert from_dict(to_dict(block)) == block

    def test_tool_result_nested(self) -> None:
        block = ToolResultBlock(
            tool_use_id="c1",
            content=[TextBlock(text="hi"), ImageBlock(media_type="image/png", data=b"\x00")],
        )
        assert from_dict(to_dict(block)) == block

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown block type"):
            from_dict({"type": "banana"})


class TestMessageSerialization:
    def test_roundtrip(self) -> None:
        msg = Message(role="user", content=[TextBlock(text="hello"), TextBlock(text="world")])
        assert Message.from_dict(msg.to_dict()) == msg

    def test_roundtrip_with_tool_blocks(self) -> None:
        msg = Message(
            role="assistant",
            content=[
                TextBlock(text="calling tool"),
                ToolUseBlock(id="c1", name="echo", input={"x": 1}),
            ],
        )
        assert Message.from_dict(msg.to_dict()) == msg

    def test_to_dict_structure(self) -> None:
        msg = Message(role="user", content=[TextBlock(text="hi")])
        d = msg.to_dict()
        assert d == {"role": "user", "content": [{"type": "text", "text": "hi"}]}

    def test_empty_content(self) -> None:
        msg = Message(role="user")
        assert Message.from_dict(msg.to_dict()) == msg
