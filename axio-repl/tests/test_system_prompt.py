"""Tests for build_system_prompt: capability-aware prompt generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axio.models import Capability, ModelSpec
from axio.tool import Tool

from axio_repl import build_system_prompt


async def _dummy_handler(x: str = "") -> str:
    return ""


def _tool(name: str) -> Tool[Any]:
    return Tool(name=name, description=f"{name} tool", handler=_dummy_handler)


_ROOT = Path("/tmp/test-workspace")

_CHAT_CAPS = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_VISION_VIDEO_CAPS = frozenset(
    {Capability.text, Capability.vision, Capability.video, Capability.tool_use, Capability.reasoning}
)
_IMAGE_GEN_CAPS = frozenset({Capability.text, Capability.vision, Capability.image_generation})
_NO_TOOLS_CAPS = frozenset({Capability.text, Capability.vision})


class TestPromptHeader:
    def test_contains_model_id(self) -> None:
        model = ModelSpec(id="gpt-test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [_tool("shell")])
        assert "gpt-test" in prompt

    def test_contains_context_window(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS, context_window=1_000_000)
        prompt = build_system_prompt(_ROOT, model, [_tool("shell")])
        assert "1000K context" in prompt

    def test_contains_output_limit(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS, max_output_tokens=65_536)
        prompt = build_system_prompt(_ROOT, model, [_tool("shell")])
        assert "65K max output" in prompt

    def test_contains_working_directory(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert str(_ROOT) in prompt


class TestToolListing:
    def test_tools_listed_when_tool_use_capable(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        tools = [_tool("read_file"), _tool("shell")]
        prompt = build_system_prompt(_ROOT, model, tools)
        assert "Tools: read_file, shell" in prompt

    def test_tools_not_listed_when_no_tool_use(self) -> None:
        model = ModelSpec(id="test", capabilities=_NO_TOOLS_CAPS)
        tools = [_tool("read_file"), _tool("shell")]
        prompt = build_system_prompt(_ROOT, model, tools)
        assert "Tools:" not in prompt


class TestCapabilityNotes:
    def test_vision_note_present(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "see images" in prompt

    def test_video_note_present(self) -> None:
        model = ModelSpec(id="test", capabilities=_VISION_VIDEO_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "video files" in prompt

    def test_video_note_absent_without_capability(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "video files" not in prompt

    def test_image_generation_note(self) -> None:
        model = ModelSpec(id="test", capabilities=_IMAGE_GEN_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "generate images inline" in prompt

    def test_reasoning_note(self) -> None:
        model = ModelSpec(id="test", capabilities=_VISION_VIDEO_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "thinking" in prompt.lower() or "reasoning" in prompt.lower()

    def test_no_tool_warning_absent(self) -> None:
        """Models without tool_use should NOT have a warning — tools are just omitted."""
        model = ModelSpec(id="test", capabilities=_NO_TOOLS_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "WARNING" not in prompt
        assert "cannot call tools" not in prompt


class TestToolRules:
    def test_tool_rules_present_with_tool_use(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [_tool("shell")])
        assert "Read files before editing" in prompt
        assert "destructive shell commands" in prompt

    def test_tool_rules_absent_without_tool_use(self) -> None:
        model = ModelSpec(id="test", capabilities=_NO_TOOLS_CAPS)
        prompt = build_system_prompt(_ROOT, model, [_tool("shell")])
        assert "Read files before editing" not in prompt
        assert "destructive shell commands" not in prompt

    def test_base_rules_always_present(self) -> None:
        model = ModelSpec(id="test", capabilities=_NO_TOOLS_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "Never refuse safe requests" in prompt

    def test_agent_parent_peer_id_included_for_spawn_tool(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(
            _ROOT,
            model,
            [_tool("spawn_agent"), _tool("send_message")],
            parent_peer_id="parent-123",
        )
        assert "parent-123" in prompt
        assert "send_message(agent_id='parent-123'" in prompt

    def test_agent_parent_peer_id_included_for_child_without_spawn_tool(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [_tool("send_message")], parent_peer_id="parent-123")
        assert "parent-123" in prompt
        assert "send_message(agent_id='parent-123'" in prompt


class TestAgentsText:
    def test_agents_text_appended(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [], agents_text="Custom agent rules here")
        assert "Custom agent rules here" in prompt
        assert "AGENTS.md instructions:" in prompt

    def test_empty_agents_text_omitted(self) -> None:
        model = ModelSpec(id="test", capabilities=_CHAT_CAPS)
        prompt = build_system_prompt(_ROOT, model, [])
        assert "AGENTS.md" not in prompt
