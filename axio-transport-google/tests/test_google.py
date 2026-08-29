"""Tests for Google GenAI transport — message conversion and tool handling."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from axio import Agent, MemoryContextStore
from axio.blocks import (
    AudioBlock,
    ImageBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from axio.events import IterationEnd, Refusal, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry
from axio.testing import assert_stream_contract
from axio.tool import Tool
from axio.types import StopReason
from axio_sse import Payload

from axio_transport_google import (
    _FINISH_REASON_MAP,
    GENAI_MODELS,
    ContentPart,
    GenerateContentChunk,
    GoogleTransport,
    UsageMetadata,
    _build_contents_json,
    _build_tools_json,
    _get_anthropic_models,
    _tool_name_from_id,
    _Turn,
    _usage,
)
from axio_transport_google._generated_types import Part


async def _dummy_tool(query: str) -> str:
    return "ok"


# ---------------------------------------------------------------------------
# _tool_name_from_id
# ---------------------------------------------------------------------------


def test_tool_name_from_id_found() -> None:
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="read_file", input={"filename": "a.txt"})]),
    ]
    assert _tool_name_from_id("c1", messages) == "read_file"


def test_tool_name_from_id_not_found() -> None:
    assert _tool_name_from_id("missing", []) == "unknown"


# ---------------------------------------------------------------------------
# _build_contents_json — basic user/assistant
# ---------------------------------------------------------------------------


def test_convert_user_text() -> None:
    messages = [Message(role="user", content=[TextBlock(text="Hello")])]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert any(p.get("text") == "Hello" for p in contents[0]["parts"])


def test_convert_assistant_text() -> None:
    messages = [Message(role="assistant", content=[TextBlock(text="Hi there")])]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "model"
    assert any(p.get("text") == "Hi there" for p in contents[0]["parts"])


# ---------------------------------------------------------------------------
# _build_contents_json — images, audio, and video in user messages
# ---------------------------------------------------------------------------


def test_convert_user_image() -> None:
    img_data = b"\x89PNG\r\n\x1a\nfake"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="What is this?"),
                ImageBlock(media_type="image/png", data=img_data),
            ],
        )
    ]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "What is this?"
    idata = parts[1]["inlineData"]
    assert idata["mimeType"] == "image/png"
    assert base64.b64decode(idata["data"]) == img_data


def test_convert_user_audio() -> None:
    audio_data = b"\xff\xfb\x90\x00fake-mp3"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="Transcribe this"),
                AudioBlock(media_type="audio/mp3", data=audio_data),
            ],
        )
    ]
    contents = _build_contents_json(messages)
    parts = contents[0]["parts"]
    assert len(parts) == 2
    idata = parts[1]["inlineData"]
    assert idata["mimeType"] == "audio/mp3"
    assert base64.b64decode(idata["data"]) == audio_data


def test_convert_user_video() -> None:
    video_data = b"\x00\x00\x00\x1cftypisom"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="Describe this video"),
                VideoBlock(media_type="video/mp4", data=video_data),
            ],
        )
    ]
    contents = _build_contents_json(messages)
    parts = contents[0]["parts"]
    assert len(parts) == 2
    idata = parts[1]["inlineData"]
    assert idata["mimeType"] == "video/mp4"
    assert base64.b64decode(idata["data"]) == video_data


# ---------------------------------------------------------------------------
# _build_contents_json — tool calls and results
# ---------------------------------------------------------------------------


def test_convert_assistant_tool_call() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="get_weather", input={"location": "Paris"})],
        )
    ]
    contents = _build_contents_json(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "model"
    part = contents[0]["parts"][0]
    fc = part["functionCall"]
    assert fc["name"] == "get_weather"
    assert fc["args"] == {"location": "Paris"}
    assert fc["id"] == "call_1"


def test_convert_tool_result_text() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="get_weather", input={"location": "Paris"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="22°C sunny")],
        ),
    ]
    contents = _build_contents_json(messages)
    assert len(contents) == 2
    fr = contents[1]["parts"][0]["functionResponse"]
    assert fr["name"] == "get_weather"
    assert fr["id"] == "call_1"
    assert fr["response"]["result"] == "22°C sunny"


def test_convert_tool_result_with_image() -> None:
    img_data = b"\xff\xd8\xff\xe0fake-jpeg"
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="read_file", input={"filename": "photo.jpg"})],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[
                        TextBlock(text="Image file: photo.jpg"),
                        ImageBlock(media_type="image/jpeg", data=img_data),
                    ],
                )
            ],
        ),
    ]
    contents = _build_contents_json(messages)
    fr = contents[1]["parts"][0]["functionResponse"]
    assert fr["response"] == {"result": "Image file: photo.jpg"}
    # Media is a sibling inlineData part, not nested inside functionResponse
    media_part = contents[1]["parts"][1]
    assert media_part["inlineData"]["mimeType"] == "image/jpeg"
    assert base64.b64decode(media_part["inlineData"]["data"]) == img_data


def test_convert_tool_result_with_audio() -> None:
    audio_data = b"OggS\x00\x02fake-ogg"
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="read_file", input={"filename": "audio.ogg"})],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[
                        TextBlock(text="Audio file: audio.ogg"),
                        AudioBlock(media_type="audio/ogg", data=audio_data),
                    ],
                )
            ],
        ),
    ]
    contents = _build_contents_json(messages)
    parts = contents[1]["parts"]
    assert parts[0]["functionResponse"]["response"] == {"result": "Audio file: audio.ogg"}
    assert parts[1]["inlineData"]["mimeType"] == "audio/ogg"
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == audio_data


def test_convert_tool_result_error() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="shell", input={"command": "bad"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="command failed", is_error=True)],
        ),
    ]
    contents = _build_contents_json(messages)
    fr = contents[1]["parts"][0]["functionResponse"]
    assert fr["response"]["error"] == "command failed"


# ---------------------------------------------------------------------------
# _build_contents_json — thought signatures
# ---------------------------------------------------------------------------


def test_convert_thought_signature_roundtrip() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="read_file", input={"filename": "a.txt"})],
        )
    ]
    sigs = {"call_1": "dGVzdF9zaWduYXR1cmU="}
    contents = _build_contents_json(messages, sigs)
    part = contents[0]["parts"][0]
    assert part["thoughtSignature"] == "dGVzdF9zaWduYXR1cmU="


# ---------------------------------------------------------------------------
# _build_tools_json
# ---------------------------------------------------------------------------


def test_build_tools_json() -> None:
    tool: Tool[Any] = Tool(name="search", description="Search the web", handler=_dummy_tool)
    result = _build_tools_json([tool])
    assert len(result) == 1
    declarations = result[0]["functionDeclarations"]
    assert len(declarations) == 1
    assert declarations[0]["name"] == "search"
    assert declarations[0]["description"] == "Search the web"
    assert "query" in declarations[0]["parameters"].get("properties", {})


# ---------------------------------------------------------------------------
# GoogleTransport basics
# ---------------------------------------------------------------------------


def test_transport_defaults() -> None:
    t = GoogleTransport()
    assert t.model.id == "gemini-3.1-flash-lite-preview"
    assert t.name == "Google GenAI"


def test_transport_to_from_dict() -> None:
    t = GoogleTransport(api_key="test-key")
    d = t.to_dict()
    assert d["api_key"] == "test-key"
    t2 = GoogleTransport.from_dict(d)
    assert t2.api_key == "test-key"


def test_transport_vertexai_to_from_dict() -> None:
    t = GoogleTransport(vertexai=True, project="my-project", location="us-central1")
    d = t.to_dict()
    assert d["vertexai"] is True
    assert d["project"] == "my-project"
    assert d["location"] == "us-central1"
    t2 = GoogleTransport.from_dict(d)
    assert t2.vertexai is True
    assert t2.project == "my-project"
    assert t2.location == "us-central1"


def test_transport_non_vertexai_no_extra_fields(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = GoogleTransport(api_key="test-key")
    d = t.to_dict()
    assert "vertexai" not in d
    assert "project" not in d


def test_genai_models_registry() -> None:
    assert isinstance(GENAI_MODELS, ModelRegistry)
    assert "gemini-3-flash-preview" in GENAI_MODELS
    assert "gemini-3.1-pro-preview" in GENAI_MODELS
    assert "gemini-3.1-flash-lite-preview" in GENAI_MODELS


def test_vertexai_anthropic_models_registry() -> None:
    models = _get_anthropic_models()
    assert isinstance(models, ModelRegistry)
    assert "anthropic/claude-sonnet-4-6" in models
    assert "anthropic/claude-opus-4-6" in models
    assert "anthropic/claude-haiku-4-5" in models


# ---------------------------------------------------------------------------
# Image / video generation model capabilities
# ---------------------------------------------------------------------------


def test_nano_banana_model_has_image_generation() -> None:
    spec = GENAI_MODELS["gemini-3.1-flash-image-preview"]
    assert Capability.image_generation in spec.capabilities
    assert Capability.vision in spec.capabilities


def test_dedicated_gen_models_not_in_chat_registry() -> None:
    """Imagen/Veo use dedicated :predict / :predictLongRunning endpoints —
    must not be selectable as chat models."""
    for model_id in GENAI_MODELS.keys():
        assert not model_id.startswith("imagen-"), f"{model_id} should not be in GENAI_MODELS"
        assert not model_id.startswith("veo-"), f"{model_id} should not be in GENAI_MODELS"


# ---------------------------------------------------------------------------
# Config parameters round-trip
# ---------------------------------------------------------------------------


def test_config_params_to_from_dict() -> None:
    t = GoogleTransport(temperature=0.7, top_p=0.9, seed=42, thinking_budget=4096, service_tier="flex")
    d = t.to_dict()
    assert d["temperature"] == 0.7
    assert d["seed"] == 42
    assert d["thinking_budget"] == 4096
    assert d["service_tier"] == "flex"
    t2 = GoogleTransport.from_dict(d)
    assert t2.temperature == 0.7
    assert t2.top_p == 0.9
    assert t2.seed == 42
    assert t2.thinking_budget == 4096
    assert t2.service_tier == "flex"


def test_string_settings_coercion() -> None:
    """Settings from TUI SQLite DB arrive as strings — __post_init__ must coerce."""
    t = GoogleTransport(
        vertexai="true",  # type: ignore[arg-type]
        temperature="0.7",  # type: ignore[arg-type]
        top_p="0.9",  # type: ignore[arg-type]
        seed="42",  # type: ignore[arg-type]
        thinking_budget="4096",  # type: ignore[arg-type]
    )
    assert t.vertexai is True
    assert t.temperature == 0.7
    assert t.top_p == 0.9
    assert t.seed == 42
    assert t.thinking_budget == 4096


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


def test_build_url_developer_api(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = GoogleTransport(api_key="test-key")
    url = t._build_url("models/gemini-test:streamGenerateContent", "alt=sse")
    assert "generativelanguage.googleapis.com" in url
    assert "key=test-key" in url
    assert "alt=sse" in url
    assert "gemini-test" in url


def test_build_url_vertex_ai() -> None:
    t = GoogleTransport(vertexai=True, project="my-proj", location="us-central1")
    url = t._build_url("publishers/google/models/gemini-test:streamGenerateContent", "alt=sse")
    assert "us-central1-aiplatform.googleapis.com" in url
    assert "projects/my-proj" in url
    assert "alt=sse" in url


def test_build_url_vertex_ai_global() -> None:
    t = GoogleTransport(vertexai=True, project="my-proj", location="global")
    url = t._build_url("publishers/google/models/gemini-test:streamGenerateContent")
    assert "aiplatform.googleapis.com" in url
    assert "global-aiplatform" not in url


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------


def test_generation_config_basic() -> None:
    t = GoogleTransport(temperature=0.5, top_p=0.8, seed=123)
    config = t._build_generation_config_json()
    assert config["temperature"] == 0.5
    assert config["topP"] == 0.8
    assert config["seed"] == 123
    assert config["maxOutputTokens"] == t.model.max_output_tokens


def test_generation_config_thinking() -> None:
    t = GoogleTransport(thinking_level="HIGH")
    config = t._build_generation_config_json()
    assert config["thinkingConfig"]["includeThoughts"] is True
    assert config["thinkingConfig"]["thinkingLevel"] == "HIGH"


class TestUsageAccounting:
    def test_thinking_and_tool_prompt_tokens_are_added_to_their_totals(self) -> None:
        """Gemini reports thinking beside the candidates, not inside them.

        Read as reported, a thinking model billed its reasoning to nobody. Cached content is the
        other way round and is already inside promptTokenCount, so adding it would double-count.
        """
        usage = _usage(
            UsageMetadata.read(
                Payload(
                    {
                        "promptTokenCount": 1000,
                        "cachedContentTokenCount": 800,
                        "toolUsePromptTokenCount": 30,
                        "candidatesTokenCount": 60,
                        "thoughtsTokenCount": 400,
                        "totalTokenCount": 1490,
                    }
                )
            )
        )
        assert usage.input_tokens == 1030, "tool-use prompt tokens stand outside promptTokenCount"
        assert usage.output_tokens == 460, "thinking was billed and not counted"
        assert usage.cache_read_tokens == 800
        assert usage.uncached_input_tokens == 230
        assert usage.answer_tokens == 60
        assert usage.total_tokens == 1490, "the provider's own total disagreed with the parts"

    def test_a_plain_response_needs_no_arithmetic(self) -> None:
        usage = _usage(
            UsageMetadata.read(Payload({"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}))
        )
        assert (usage.input_tokens, usage.output_tokens) == (10, 5)
        assert (usage.cache_read_tokens, usage.reasoning_tokens) == (0, 0)


class TestNothingIsDropped:
    def test_every_published_finish_reason_is_mapped(self) -> None:
        """Two of twenty-one were mapped; the rest were read as a transport error.

        The list is the one the discovery document publishes. A blocked answer is not the transport
        failing, and a malformed tool call is not a finished turn.
        """
        # The union of both surfaces: 21 in the developer discovery document, 17 in Vertex,
        # which publishes MODEL_ARMOR and lacks five of the others.
        published = {
            "FINISH_REASON_UNSPECIFIED",
            "MODEL_ARMOR",
            "STOP",
            "MAX_TOKENS",
            "SAFETY",
            "RECITATION",
            "LANGUAGE",
            "OTHER",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_OTHER",
            "NO_IMAGE",
            "IMAGE_RECITATION",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MISSING_THOUGHT_SIGNATURE",
            "MALFORMED_RESPONSE",
            "ESCALATION",
        }
        assert set(_FINISH_REASON_MAP) == published
        assert _FINISH_REASON_MAP["STOP"] == StopReason.end_turn
        # A blocked turn is not the transport failing.
        assert _FINISH_REASON_MAP["SAFETY"] == StopReason.refusal
        assert _FINISH_REASON_MAP["MODEL_ARMOR"] == StopReason.refusal
        # A call the API could not parse asks to be prompted again; too many of them does not.
        assert _FINISH_REASON_MAP["MALFORMED_FUNCTION_CALL"] == StopReason.tool_use
        assert _FINISH_REASON_MAP["TOO_MANY_TOOL_CALLS"] == StopReason.error

    def test_a_part_reads_its_declared_fields(self) -> None:
        part = ContentPart.read(
            Payload({"text": "thinking out loud", "thought": True, "thoughtSignature": "sig", "unknown": 1})
        )
        assert (part.text, part.thought, part.thoughtSignature) == ("thinking out loud", True, "sig")

    def test_a_thought_flag_that_is_a_number_does_not_pass_for_true(self) -> None:
        # bool is an int in Python, so a count of 1 must not read as a flag.
        assert ContentPart.read(Payload({"text": "x", "thought": 1})).thought is False

    def test_the_usage_slices_are_read_from_their_own_names(self) -> None:
        um = UsageMetadata.read(
            Payload({"promptTokenCount": 10, "thoughtsTokenCount": 4, "cachedContentTokenCount": 6})
        )
        assert (um.promptTokenCount, um.thoughtsTokenCount, um.cachedContentTokenCount) == (10, 4, 6)
        assert um.totalTokenCount is None

    def test_a_chunk_with_nothing_in_it_reads_as_empty_rather_than_failing(self) -> None:
        chunk = GenerateContentChunk.read(Payload({"candidates": None, "usageMetadata": "not an object"}))
        assert chunk.candidates == [] and chunk.usageMetadata == UsageMetadata()


class TestPromptFeedbackAndSignatureOrder:
    def test_safety_ratings_on_a_healthy_answer_are_not_a_blocked_prompt(self) -> None:
        """The developer API attaches promptFeedback to healthy answers too.

        A Payload is a dict, so gating on its truthiness called every one of them a blocked prompt:
        the user got the right answer with a red "prompt blocked" printed over it.
        """
        chunk = GenerateContentChunk.read(
            Payload(
                {
                    "candidates": [{"content": {"parts": [{"text": "Paris."}]}, "finishReason": "STOP"}],
                    "promptFeedback": {
                        "safetyRatings": [{"category": "HARM_CATEGORY_HATE", "probability": "NEGLIGIBLE"}]
                    },
                }
            )
        )
        assert chunk.promptFeedback, "the object is present"
        assert not chunk.promptFeedback.string("blockReason"), "and it is what the transport now gates on"

    def test_a_genuinely_blocked_prompt_still_says_so(self) -> None:
        chunk = GenerateContentChunk.read(Payload({"promptFeedback": {"blockReason": "SAFETY"}}))
        assert chunk.promptFeedback.string("blockReason") == "SAFETY"


class TestThoughtSignatureIsReplayed:
    def test_a_stored_thought_goes_back_with_its_signature(self) -> None:
        """Read from the stored block, not from the map on the transport instance.

        That map is per-transport and empty after a restart, so a resumed session used to send the
        turn back without the proof and come back MISSING_THOUGHT_SIGNATURE.
        """
        messages = [
            Message(
                role="assistant",
                content=[ReasoningBlock(text="weighing it", signature="SIG"), TextBlock(text="Paris.")],
            )
        ]
        parts = _build_contents_json(messages)[0]["parts"]
        assert parts[0] == {"text": "weighing it", "thought": True, "thoughtSignature": "SIG"}
        assert parts[1] == {"text": "Paris."}

    def test_a_stored_answer_goes_back_with_the_proof_that_signed_it(self) -> None:
        # Gemini signs the text part it issued the proof for, so the proof rides on that part.
        messages = [Message(role="assistant", content=[TextBlock(text="42", signature="SIG")])]
        assert _build_contents_json(messages)[0]["parts"] == [{"text": "42", "thoughtSignature": "SIG"}]

    def test_a_thought_with_no_signature_still_travels_as_a_thought(self) -> None:
        # Gemini does not refuse an unsigned thought the way Anthropic refuses an unsigned block.
        messages = [Message(role="assistant", content=[ReasoningBlock(text="unsigned")])]
        assert _build_contents_json(messages)[0]["parts"] == [{"text": "unsigned", "thought": True}]


class TestEmissionOrder:
    """Driven through the real stream, because the order is the whole point of the fix."""

    async def test_a_signature_arrives_after_the_reasoning_it_signs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The agent attaches a signature to the block it just built and refuses to extend a signed
        # one, so emitted first, one part became two blocks.
        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "weighing it", "thought": True, "thoughtSignature": "SIG"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        }
        events = await _stream_one(monkeypatch, chunk)
        kinds = [type(e).__name__ for e in events]
        assert kinds.index("ReasoningDelta") < kinds.index("ReasoningSignature")

    async def test_safety_ratings_on_a_healthy_answer_raise_no_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A Payload is a dict, so gating on its truthiness called every healthy answer that carried
        # safety ratings a blocked prompt, and the user saw "prompt blocked" over the right answer.
        chunk = {
            "candidates": [{"content": {"parts": [{"text": "Paris."}]}, "finishReason": "STOP"}],
            "promptFeedback": {"safetyRatings": [{"category": "HARM_CATEGORY_HATE", "probability": "NEGLIGIBLE"}]},
        }
        events = await _stream_one(monkeypatch, chunk)
        assert not [e for e in events if isinstance(e, Refusal)]
        assert [e.delta for e in events if isinstance(e, TextDelta)] == ["Paris."]

    async def test_a_blocked_prompt_still_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = await _stream_one(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}})
        refusals = [e for e in events if isinstance(e, Refusal)]
        assert [(r.blocked_input, r.category) for r in refusals] == [(True, "SAFETY")]


@asynccontextmanager
async def _serving(monkeypatch: pytest.MonkeyPatch, chunk: dict[str, Any]) -> AsyncIterator[None]:
    """Serve one SSE chunk from a local server while the block runs."""
    body = f"data: {json.dumps(chunk)}\n\n".encode()

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(body)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = site._server.sockets[0].getsockname()[:2]  # type: ignore[union-attr]
    monkeypatch.setattr("axio_transport_google._DEVELOPER_API_BASE", f"http://{host}:{port}/v1beta")
    try:
        yield
    finally:
        await runner.cleanup()


async def _stream_one(monkeypatch: pytest.MonkeyPatch, chunk: dict[str, Any]) -> list[Any]:
    """Serve one SSE chunk from a local server and collect what the transport made of it."""
    async with _serving(monkeypatch, chunk):
        async with aiohttp.ClientSession() as session:
            transport = GoogleTransport(
                api_key="test-key", model=GENAI_MODELS["gemini-3-flash-preview"], session=session, max_retries=1
            )
            return [event async for event in transport.stream([], [], "")]


class TestFunctionCallSignatureSurvivesARestart:
    """A signature that arrived on a functionCall part has to go back on that part.

    Held only in the transport's own map, it was lost the moment a session was restored into a new
    transport, and Gemini answered MISSING_THOUGHT_SIGNATURE.
    """

    @staticmethod
    def _restored() -> list[Message]:
        # What the agent stores for a part carrying both a call and its signature: ToolUseStart
        # goes to `pending` and materialises at IterationEnd, so the signature lands first.
        return [
            Message(
                role="assistant",
                content=[
                    ReasoningBlock(text="", signature="FCSIG"),
                    ToolUseBlock(id="call_1", name="get_weather", input={"city": "Paris"}),
                ],
            )
        ]

    def test_the_call_carries_its_signature_with_no_help_from_the_transport(self) -> None:
        parts = _build_contents_json(self._restored(), thought_signatures=None)[0]["parts"]
        assert len(parts) == 1, "an empty thought part was sent instead of signing the call"
        assert parts[0]["functionCall"]["id"] == "call_1"
        assert parts[0]["thoughtSignature"] == "FCSIG"

    def test_a_live_transport_map_still_works_for_a_call_with_no_stored_signature(self) -> None:
        messages = [
            Message(role="assistant", content=[ToolUseBlock(id="call_2", name="t", input={})]),
        ]
        parts = _build_contents_json(messages, thought_signatures={"call_2": "LIVE"})[0]["parts"]
        assert parts[0]["thoughtSignature"] == "LIVE"

    def test_a_signature_with_reasoning_beside_it_stays_on_the_reasoning(self) -> None:
        messages = [
            Message(
                role="assistant",
                content=[
                    ReasoningBlock(text="weighing it", signature="THOUGHTSIG"),
                    ToolUseBlock(id="call_3", name="t", input={}),
                ],
            )
        ]
        parts = _build_contents_json(messages, thought_signatures=None)[0]["parts"]
        assert parts[0] == {"text": "weighing it", "thought": True, "thoughtSignature": "THOUGHTSIG"}
        assert "thoughtSignature" not in parts[1], "the thought's own proof was moved onto the call"


class TestParallelCallSignatures:
    def test_each_parallel_call_gets_its_own_signature(self) -> None:
        """Several signed calls in one turn: the agent appends every signature before any of the
        calls, so a single slot gave the first call the last signature and left the rest unsigned."""
        stored = [
            Message(
                role="assistant",
                content=[
                    ReasoningBlock(text="", signature="SIG-A"),
                    ReasoningBlock(text="", signature="SIG-B"),
                    ToolUseBlock(id="call_a", name="weather", input={}),
                    ToolUseBlock(id="call_b", name="clock", input={}),
                ],
            )
        ]
        parts = _build_contents_json(stored, thought_signatures=None)[0]["parts"]
        assert [(p["functionCall"]["id"], p.get("thoughtSignature")) for p in parts] == [
            ("call_a", "SIG-A"),
            ("call_b", "SIG-B"),
        ]

    def test_a_call_with_no_stored_signature_falls_back_to_the_live_map(self) -> None:
        stored = [
            Message(
                role="assistant",
                content=[
                    ReasoningBlock(text="", signature="SIG-A"),
                    ToolUseBlock(id="call_a", name="weather", input={}),
                    ToolUseBlock(id="call_b", name="clock", input={}),
                ],
            )
        ]
        parts = _build_contents_json(stored, thought_signatures={"call_b": "LIVE-B"})[0]["parts"]
        assert [(p["functionCall"]["id"], p.get("thoughtSignature")) for p in parts] == [
            ("call_a", "SIG-A"),
            ("call_b", "LIVE-B"),
        ]

    def test_signatures_do_not_leak_across_messages(self) -> None:
        stored = [
            Message(role="assistant", content=[ReasoningBlock(text="", signature="SIG-A")]),
            Message(role="assistant", content=[ToolUseBlock(id="call_b", name="clock", input={})]),
        ]
        built = _build_contents_json(stored, thought_signatures=None)
        assert "thoughtSignature" not in built[-1]["parts"][0]


def test_each_signed_part_is_indexed_by_its_own_position() -> None:
    """The index on a signature says which block it proves.

    Fixed at zero, two parallel signed calls looked to the agent like two halves of one proof and
    were joined into a signature that matches neither.
    """
    chunk = GenerateContentChunk.read(
        Payload(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "a"}, "thoughtSignature": "SIG-A"},
                                {"functionCall": {"name": "b"}, "thoughtSignature": "SIG-B"},
                            ]
                        }
                    }
                ]
            }
        )
    )
    positions = [at for at, part in enumerate(chunk.candidates[0].content.parts) if part.thoughtSignature]
    assert positions == [0, 1], "the two parts must not share an index"


class TestACutStreamIsNotAFinishedAnswer:
    async def test_a_stream_that_ends_without_a_finish_reason_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every Gemini stream ends on a finishReason.

        Without one the connection was cut, and reporting end_turn stored a truncated answer as a
        whole one and returned it to the caller as the model's answer.
        """
        chunk = {"candidates": [{"content": {"parts": [{"text": "half an ans"}]}}]}
        with pytest.raises(StreamError, match="without a finishReason"):
            await _stream_one(monkeypatch, chunk)

    async def test_a_blocked_prompt_is_a_finished_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Nothing was generated, so no candidate and no finishReason ever arrive. That is not a cut.
        events = await _stream_one(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}})
        assert [e for e in events if isinstance(e, Refusal)]


class TestPartIndexes:
    async def test_every_event_of_a_part_carries_that_parts_position(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fixed at zero, the reasoning of one part and the signature of another shared an index.

        Nothing downstream could then group a part's events together.
        """
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "weighing it", "thought": True, "thoughtSignature": "SIG-0"},
                            {"text": "the answer"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        events = await _stream_one(monkeypatch, chunk)
        by_kind = {type(e).__name__: e for e in events if hasattr(e, "index")}
        assert by_kind["ReasoningDelta"].index == 0
        assert by_kind["ReasoningSignature"].index == 0
        assert by_kind["TextDelta"].index == 1

    async def test_a_tool_call_in_the_second_part_is_indexed_there(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "let me look"},
                            {"functionCall": {"id": "call_1", "name": "get_weather", "args": {}}},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        events = await _stream_one(monkeypatch, chunk)
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        deltas = [e for e in events if isinstance(e, ToolInputDelta)]
        assert [s.index for s in starts] == [1]
        assert [d.index for d in deltas] == [1]


class TestACallCarriesItsOwnProof:
    async def test_the_signature_of_a_call_part_goes_on_the_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sent as a bare signature it attached to whatever reasoning block was still unsigned.

        The thought took the call's proof and the call had none, so the replay was refused.
        """
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking", "thought": True},
                            {"functionCall": {"id": "call_1", "name": "t", "args": {}}, "thoughtSignature": "SIG"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        events = await _stream_one(monkeypatch, chunk)
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert [s.signature for s in starts] == ["SIG"]
        assert not [e for e in events if type(e).__name__ == "ReasoningSignature"]

    def test_a_stored_call_replays_its_own_proof_with_no_transport_state(self) -> None:
        messages = [
            Message(
                role="assistant",
                content=[ToolUseBlock(id="call_1", name="t", input={}, signature="SIG")],
            )
        ]
        parts = _build_contents_json(messages, thought_signatures=None)[0]["parts"]
        assert parts[0]["thoughtSignature"] == "SIG"


async def test_two_unnamed_calls_to_one_tool_get_different_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numbered by id() of a temporary part, two calls collided when CPython reused the address.

    The agent keys pending calls by id, so the second overwrote the first and one call was lost.
    """
    chunk = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "a"}}},
                        {"functionCall": {"name": "search", "args": {"q": "b"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    events = await _stream_one(monkeypatch, chunk)
    ids = [e.tool_use_id for e in events if isinstance(e, ToolUseStart)]
    assert len(ids) == len(set(ids)) == 2


class TestAProofOnAnswerTextSurvivesTheRoundTrip:
    """Gemini signs the answer part. That proof has to come back on the same part.

    Routed to a ProviderEvent the agent stored nothing, and the replay of the turn came back
    MISSING_THOUGHT_SIGNATURE.
    """

    @staticmethod
    async def _history(monkeypatch: pytest.MonkeyPatch, chunk: dict[str, Any]) -> list[Message]:
        """Run one real turn through the agent and return what it stored."""
        async with _serving(monkeypatch, chunk):
            async with aiohttp.ClientSession() as session:
                transport = GoogleTransport(
                    api_key="test-key", model=GENAI_MODELS["gemini-3-flash-preview"], session=session, max_retries=1
                )
                context = MemoryContextStore()
                async for _ in Agent(system="", tools=[], transport=transport).run_stream("what is it", context):
                    pass
                return await context.get_history()

    _CHUNK = {
        "candidates": [{"content": {"parts": [{"text": "42", "thoughtSignature": "SIG"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
    }

    async def test_the_agent_stores_the_answer_with_its_proof(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = await self._history(monkeypatch, self._CHUNK)

        assistant = [m for m in history if m.role == "assistant"][0]
        # The protocol that issued the proof is stored with it, so a session that later changes
        # transport does not replay a thoughtSignature to a provider that never made one.
        assert assistant.content == [TextBlock(text="42", signature="SIG", provider="google")]

    async def test_the_stored_answer_replays_on_the_part_gemini_signed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = await self._history(monkeypatch, self._CHUNK)

        # No transport state: the proof comes from the stored block, so it survives a restart.
        parts = _build_contents_json(history, thought_signatures=None)[-1]["parts"]
        assert parts[0] == {"text": "42", "thoughtSignature": "SIG"}


class TestWhatAPartProduces:
    """One part, one rule per shape it arrives in."""

    @staticmethod
    def _events(raw: dict[str, Any], *, seq: int = 1) -> list[Any]:
        transport = GoogleTransport(api_key="k", model=GENAI_MODELS["gemini-3-flash-preview"])
        return list(transport._part_events(ContentPart.read(Payload(raw)), _Turn(at=0, seq=seq)))

    def test_a_thought_is_signed_as_reasoning(self) -> None:
        events = self._events({"text": "hm", "thought": True, "thoughtSignature": "SIG"})
        assert [type(e).__name__ for e in events] == ["ReasoningDelta", "ReasoningSignature"]

    def test_a_bare_proof_is_still_reasoning(self) -> None:
        # It belongs to a call that follows, which is what the request side reads it as.
        assert [type(e).__name__ for e in self._events({"thoughtSignature": "SIG"})] == ["ReasoningSignature"]

    def test_a_proof_on_answer_text_signs_that_text(self) -> None:
        # Held as a ReasoningSignature it made a textless ReasoningBlock, and the next unsigned
        # call in the turn replayed with a proof that was never its own.
        events = self._events({"text": "42", "thoughtSignature": "SIG"})
        assert [type(e).__name__ for e in events] == ["TextDelta", "TextSignature"]
        assert events[1].signature == "SIG"

    def test_the_proof_arrives_after_the_text_it_signs(self) -> None:
        # The agent signs the block it has just built, so a proof emitted first signs nothing, and
        # a proof sent as anything but TextSignature is not stored at all.
        events = self._events({"text": "42", "thoughtSignature": "SIG"})

        assert [type(e).__name__ for e in events] == ["TextDelta", "TextSignature"]
        assert events[1].signature == "SIG"

    def test_a_signed_part_axio_has_no_type_for_is_kept_whole(self) -> None:
        # The signature suppressed the event carrying the content, so the code ran and vanished.
        # It is kept rather than watched: this API is stateless, and a part left out of the next
        # request is one the model answers the follow-up without.
        raw = {"executableCode": {"code": "print(1)"}, "thoughtSignature": "SIG"}

        events = self._events(raw)

        assert [type(e).__name__ for e in events] == ["ProviderOutput"]
        # One event, not two: the proof rides inside the part, which is where Gemini put it, so
        # replaying the part verbatim replays the proof with it.
        assert events[0].data == raw

    def test_a_signed_image_is_not_reasoning_either(self) -> None:
        events = self._events({"inlineData": {"mimeType": "image/png", "data": "aGk="}, "thoughtSignature": "SIG"})
        assert [type(e).__name__ for e in events] == ["ImageOutput", "ProviderEvent"]

    def test_a_synthesized_call_id_does_not_repeat_across_streams(self) -> None:
        # _thought_signatures outlives one stream, so a position alone let a later turn's proof
        # overwrite an earlier turn's under the same id.
        call = {"functionCall": {"name": "search", "args": {}}}
        assert self._events(call, seq=1)[0].tool_use_id != self._events(call, seq=2)[0].tool_use_id


def test_a_retry_does_not_reuse_the_part_indices_of_the_attempt_that_failed() -> None:
    # Seeded at -1 on every attempt, a retry repeated the indices already emitted, and the agent
    # merged parts of two different attempts into one block.
    turn = _Turn()
    turn.at = 2

    turn.restart()

    assert turn.at == 2
    assert (turn.stop_reason, turn.finished, turn.has_tool_calls) == (StopReason.end_turn, False, False)


class TestTheStreamContract:
    """What every transport must produce, checked here rather than left to the next review."""

    async def test_an_ordinary_turn_obeys_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chunk = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        }

        assert_stream_contract(await _stream_one(monkeypatch, chunk))

    @pytest.mark.parametrize("reason", ["MISSING_THOUGHT_SIGNATURE", "MALFORMED_RESPONSE", "TOO_MANY_TOOL_CALLS"])
    async def test_a_reason_the_caller_can_act_on_is_raised(
        self, monkeypatch: pytest.MonkeyPatch, reason: str
    ) -> None:
        # Yielded as IterationEnd(error) the agent reports `Transport stopped with: error`, and the
        # provider's own word for what went wrong never reaches the caller.
        chunk = {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": reason}]}

        with pytest.raises(StreamError, match=reason):
            await _stream_one(monkeypatch, chunk)

    async def test_a_reason_nobody_maps_reads_as_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Terminal, vouching for nothing, and claiming nothing the provider did not say.
        chunk = {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "INVENTED_LATER"}]}

        events = await _stream_one(monkeypatch, chunk)

        assert [e.stop_reason for e in events if isinstance(e, IterationEnd)] == [StopReason.unknown]


class TestWhichProofACallReplaysWith:
    """The call's own proof outranks the map on the transport, which a restart empties."""

    def test_the_stored_proof_wins_over_the_live_map(self) -> None:
        # Read from the map first, a session restored into a fresh transport replayed the call
        # with nothing and Gemini answered MISSING_THOUGHT_SIGNATURE.
        turn = Message(role="assistant", content=[ToolUseBlock(id="c1", name="search", input={}, signature="STORED")])

        parts = _build_contents_json([turn], {"c1": "FROM_THE_MAP"})[0]["parts"]

        assert parts[0]["thoughtSignature"] == "STORED"

    def test_the_map_still_answers_for_a_block_stored_without_one(self) -> None:
        turn = Message(role="assistant", content=[ToolUseBlock(id="c1", name="search", input={})])

        parts = _build_contents_json([turn], {"c1": "FROM_THE_MAP"})[0]["parts"]

        assert parts[0]["thoughtSignature"] == "FROM_THE_MAP"


class TestAProofFromAnotherProviderIsNotSentHere:
    """A thoughtSignature is Gemini's own. Read out of the same field by whichever converter runs,
    a session that changed transport put one provider's opaque data in another's protocol slot."""

    @staticmethod
    def _parts(block: Any) -> list[Part]:
        return _build_contents_json([Message(role="assistant", content=[block])])[-1]["parts"]

    def test_an_anthropic_proof_is_left_off_the_answer_text(self) -> None:
        assert self._parts(TextBlock(text="42", signature="SIG", provider="anthropic")) == [{"text": "42"}]

    def test_an_anthropic_proof_is_left_off_a_call(self) -> None:
        call = ToolUseBlock(id="c1", name="echo", input={}, signature="SIG", provider="anthropic")
        assert self._parts(call) == [{"functionCall": {"name": "echo", "args": {}, "id": "c1"}}]

    def test_a_thought_signed_elsewhere_is_not_moved_onto_the_next_part(self) -> None:
        # A signature with no text of its own is held for the call that follows. One from another
        # provider held that way would be attached to a part Gemini never signed.
        blocks = [
            ReasoningBlock(signature="SIG", provider="openai"),
            ToolUseBlock(id="c1", name="echo", input={}),
        ]
        parts = _build_contents_json([Message(role="assistant", content=blocks)])[-1]["parts"]
        assert parts == [{"functionCall": {"name": "echo", "args": {}, "id": "c1"}}]

    def test_this_provider_s_own_proof_still_travels(self) -> None:
        assert self._parts(TextBlock(text="42", signature="SIG", provider="google")) == [
            {"text": "42", "thoughtSignature": "SIG"}
        ]

    def test_a_proof_with_no_provider_recorded_still_travels(self) -> None:
        assert self._parts(TextBlock(text="42", signature="SIG")) == [{"text": "42", "thoughtSignature": "SIG"}]


class TestAPartGeminiRanItselfSurvivesTheTurn:
    """This API is stateless: the whole conversation goes back on every request.

    A part axio has no type for — executableCode, codeExecutionResult, and whatever is added next —
    was forwarded as news and never stored, so the next request did not carry the code the model
    had just run, and the follow-up was answered without it.
    """

    RAW = {"executableCode": {"language": "PYTHON", "code": "print(1)"}, "thoughtSignature": "SIG"}

    async def test_the_agent_stores_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chunk = {
            "candidates": [{"content": {"parts": [self.RAW, {"text": "1"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
        }
        async with _serving(monkeypatch, chunk):
            async with aiohttp.ClientSession() as session:
                transport = GoogleTransport(
                    api_key="k", model=GENAI_MODELS["gemini-3-flash-preview"], session=session, max_retries=1
                )
                context = MemoryContextStore()
                async for _ in Agent(system="", tools=[], transport=transport).run_stream("go", context):
                    pass
                history = await context.get_history()

        assistant = [m for m in history if m.role == "assistant"][0]
        assert assistant.content[0] == ProviderBlock(provider="google", kind="part", data=self.RAW)

    def test_it_replays_exactly_as_it_arrived(self) -> None:
        block = ProviderBlock(provider="google", kind="part", data=self.RAW)

        parts = _build_contents_json([Message(role="assistant", content=[block])])[-1]["parts"]

        assert parts == [self.RAW], "rebuilt instead, a part with no type here goes back as something else"

    def test_a_part_from_another_provider_is_not_sent_here(self) -> None:
        block = ProviderBlock(provider="openai", kind="web_search_call", data={"type": "web_search_call"})

        content = _build_contents_json([Message(role="assistant", content=[block, TextBlock(text="hi")])])

        assert content[-1]["parts"] == [{"text": "hi"}]


async def test_last_usage_reports_this_turn_and_not_the_one_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent reads it when a turn ends with no IterationEnd, so a stale value is counted twice.

    It reached the session total and, once repetition-cut turns started counting their tokens,
    the store's idea of the context size as well.
    """
    counted = {
        "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16},
    }
    silent = {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}

    async with aiohttp.ClientSession() as session:
        transport = GoogleTransport(
            api_key="k", model=GENAI_MODELS["gemini-3-flash-preview"], session=session, max_retries=1
        )
        async with _serving(monkeypatch, counted):
            async for _ in transport.stream([Message(role="user", content=[TextBlock(text="go")])], [], ""):
                pass
        assert transport.last_usage is not None

        async with _serving(monkeypatch, silent):
            async for _ in transport.stream([Message(role="user", content=[TextBlock(text="go")])], [], ""):
                pass

    assert transport.last_usage is None, "a turn that reported no usage must not report the last one's"


class TestABlockedAnswerSaysSo:
    """Eleven finish reasons map to a refusal, and only a blocked prompt announced it."""

    @staticmethod
    def _events(chunk: dict[str, Any]) -> list[Any]:
        transport = GoogleTransport(api_key="k", model=GENAI_MODELS["gemini-3-flash-preview"])
        return list(transport._chunk_events(GenerateContentChunk.read(Payload(chunk)), _Turn(at=0, seq=1)))

    def test_a_candidate_blocked_for_safety_emits_a_refusal(self) -> None:
        made = self._events(
            {"candidates": [{"content": {"parts": [{"text": "half an ans"}]}, "finishReason": "SAFETY"}]}
        )

        refusals = [e for e in made if isinstance(e, Refusal)]
        assert [r.category for r in refusals] == ["SAFETY"]
        assert refusals[0].text, "stored by the agent, or the turn keeps no record of the block"
        assert not refusals[0].spoken and not refusals[0].blocked_input

    def test_it_comes_after_the_text_the_chunk_carried(self) -> None:
        # Whatever the chunk carried was generated before the block.
        made = self._events(
            {"candidates": [{"content": {"parts": [{"text": "half an ans"}]}, "finishReason": "SAFETY"}]}
        )

        kinds = [type(e).__name__ for e in made]
        assert kinds.index("TextDelta") < kinds.index("Refusal")

    def test_a_blocked_prompt_and_a_blocked_candidate_announce_one_refusal(self) -> None:
        # Two events for one response disagree about blocked_input, and the second says the
        # opposite of the first about what was blocked.
        made = self._events(
            {
                "promptFeedback": {"blockReason": "SAFETY"},
                "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
            }
        )

        refusals = [e for e in made if isinstance(e, Refusal)]
        assert len(refusals) == 1
        assert refusals[0].blocked_input

    def test_an_ordinary_finish_announces_nothing(self) -> None:
        made = self._events({"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]})

        assert not [e for e in made if isinstance(e, Refusal)]
