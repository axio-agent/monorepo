"""Stream events: all variants emitted by AgentStream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .blocks import AudioMediaType, ImageMediaType, VideoMediaType
from .exceptions import StreamError
from .types import StopReason, ToolCallID, ToolName, Usage


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class ToolUseStart:
    index: int
    tool_use_id: ToolCallID
    name: ToolName

    signature: str = ""
    """Opaque proof that this call is the model's own, where the provider issues one for the call
    rather than for the reasoning beside it. Replayed on the call itself: attached to a reasoning
    block instead, the call comes back unsigned and the provider refuses the turn."""


@dataclass(frozen=True, slots=True)
class ToolInputDelta:
    index: int
    tool_use_id: ToolCallID
    partial_json: str


@dataclass(frozen=True, slots=True)
class ToolFieldStart:
    index: int
    tool_use_id: ToolCallID
    key: str


@dataclass(frozen=True, slots=True)
class ToolFieldDelta:
    index: int
    tool_use_id: ToolCallID
    key: str
    text: str


@dataclass(frozen=True, slots=True)
class ToolFieldEnd:
    index: int
    tool_use_id: ToolCallID
    key: str


@dataclass(frozen=True, slots=True)
class ToolOutputDelta:
    tool_use_id: ToolCallID
    name: ToolName
    key: str
    delta: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_use_id: ToolCallID
    name: ToolName
    is_error: bool
    content: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageOutput:
    """Model generated an image inline (e.g. Nano Banana / Gemini Image)."""

    index: int
    data: bytes
    media_type: ImageMediaType


@dataclass(frozen=True, slots=True)
class AudioOutput:
    """Audio content from a tool result (e.g. read_file on an audio file)."""

    index: int
    data: bytes
    media_type: AudioMediaType


@dataclass(frozen=True, slots=True)
class VideoOutput:
    """Model generated a video inline."""

    index: int
    data: bytes
    media_type: VideoMediaType


@dataclass(frozen=True, slots=True)
class IterationEnd:
    iteration: int
    stop_reason: StopReason
    usage: Usage

    def __post_init__(self) -> None:
        """Refuse ``StopReason.error``, which the agent can only report as a bare RuntimeError.

        The rule was a habit each transport had to acquire, and the shared reader every OpenAI
        turn goes through never acquired it. Raise ``StreamError`` with the provider's own message.
        """
        if self.stop_reason is StopReason.error:
            raise StreamError("IterationEnd cannot carry StopReason.error; raise StreamError instead")


@dataclass(frozen=True, slots=True)
class Error:
    exception: BaseException


@dataclass(frozen=True, slots=True)
class SessionEndEvent:
    stop_reason: StopReason
    total_usage: Usage


# ── Realtime (duplex) events ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AudioOutputDelta:
    """Streaming audio chunk from the assistant in a realtime session."""

    data: bytes
    media_type: str = "audio/pcm;rate=24000"


@dataclass(frozen=True, slots=True)
class TranscriptDelta:
    """Live transcript delta — server-side STT of user mic, or assistant
    speech transcription, depending on ``role``."""

    role: Literal["user", "assistant"]
    delta: str


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """Server VAD detected the user started speaking (realtime)."""


@dataclass(frozen=True, slots=True)
class SpeechStopped:
    """Server VAD detected the user stopped speaking (realtime)."""


@dataclass(frozen=True, slots=True)
class TurnComplete:
    """Assistant turn finished in a realtime session.  ``stop_reason`` may be
    :class:`StopReason.tool_use` to signal that pending tool calls should run
    before the next turn starts."""

    stop_reason: StopReason
    usage: Usage | None = None


# ── Provider passthrough ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    """A provider payload axio does not model, forwarded verbatim.

    ``data`` is the provider's own JSON object exactly as it was parsed: no renaming, no coercion,
    no filtering.

    How completely a transport forwards depends on what its stream names. A stream that names each
    event has a reader. That reader forwards every payload it does not interpret, so nothing is
    dropped. A stream with no discriminator — one shape per payload, read field by field — has no
    such catch. Those transports forward the parts they know how to name. A field the provider adds
    inside a payload reaches nobody until someone reads it.

    A consumer that does not recognise ``(provider, kind)`` ignores it.
    """

    provider: str
    """Which transport produced it: ``"anthropic"``, ``"openai"``, ``"google"``. The Codex
    transport reads its stream through the shared Responses reader, so its events say
    ``"openai"`` rather than naming a fourth provider."""

    kind: str
    """The provider's own discriminator, verbatim, never a name axio invented: matching on it is
    matching the vocabulary the provider publishes."""

    data: dict[str, Any]

    index: int | None = None
    """The content-block or output index, where the payload carries one."""


# ── Block lifecycle ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BlockEnd:
    """The content block at ``index`` is complete.

    The point at which accumulated :class:`ToolInputDelta` fragments are guaranteed to parse. Every
    provider marks it and axio has had no terminator for it until now.
    """

    index: int


# ── Attribution and refusal ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Citation:
    """A span of generated text attributed to a source.

    What Anthropic's citation shapes, OpenAI's annotations and Google's grounding metadata have in
    common. ``raw`` keeps the provider's whole object for the fields that differ.
    """

    index: int
    cited_text: str = ""
    title: str | None = None
    url: str | None = None

    source_id: str | None = None
    """Whatever identifies the source inside this request: a file id, a document index, a chunk id."""

    start: int | None = None
    end: int | None = None
    unit: Literal["char", "byte", "page", "block", "unknown"] = "unknown"
    """What ``start`` and ``end`` count. Stated because the providers disagree — OpenAI counts
    characters and Google counts bytes — so offsets from different units must never be compared."""

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Refusal:
    """The model declined, or the provider blocked the turn.

    This is deliberately not a :class:`TextDelta`. As ordinary assistant text, or as an empty turn
    that succeeded, a refusal is indistinguishable from an answer and no consumer can act on it.
    """

    index: int
    text: str = ""

    category: str | None = None
    """The provider's own category, verbatim. Not normalised: the taxonomies do not overlap, and a
    mapping between them would state something no provider says."""

    blocked_input: bool = False
    """True where the provider rejected the prompt rather than the answer, so nothing was generated
    and sending the same prompt again cannot succeed."""

    raw: dict[str, Any] = field(default_factory=dict)


# ── Provenance ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReasoningSignature:
    """Opaque proof that a reasoning block is the provider's own, to be replayed unaltered.

    Anthropic refuses a returned ``thinking`` block whose signature is missing or changed, and
    Google publishes a ``MISSING_THOUGHT_SIGNATURE`` finish reason for the same failure. Never
    inspect, decode, re-encode or truncate ``data``.
    """

    index: int
    signature: str

    redacted: bool = False
    """True where the payload replaces the reasoning text instead of accompanying it."""

    id: str = ""
    """How the provider names the block this proves, where it names them. Replayed beside the
    proof, because a provider that identifies reasoning by id refuses the pair without it."""


@dataclass(frozen=True, slots=True)
class TextSignature:
    """Opaque proof that a block of answer text is the provider's own, to be replayed unaltered.

    Google signs the part it issued the proof for, and answer text is one such part. The proof
    belongs to the text block, not to the reasoning or the call beside it: replayed on another part
    it proves nothing, and the turn fails with ``MISSING_THOUGHT_SIGNATURE``. Never inspect, decode,
    re-encode or truncate ``data``.

    Emitted after the text it signs, never before.
    """

    index: int
    signature: str


# ── Iteration lifecycle ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class IterationStart:
    """One provider request has begun.

    ``model`` is the model that actually served the turn, which need not be the one asked for.
    Server-side fallback, sticky routing and dated-snapshot resolution all substitute a different
    model at a different price. A cost lookup therefore keys off this rather than off the request.
    """

    iteration: int
    id: str | None = None
    model: str | None = None


type StreamEvent = (
    ReasoningDelta
    | ReasoningSignature
    | TextDelta
    | TextSignature
    | Refusal
    | Citation
    | ImageOutput
    | AudioOutput
    | VideoOutput
    | ToolUseStart
    | ToolInputDelta
    | ToolFieldStart
    | ToolFieldDelta
    | ToolFieldEnd
    | ToolOutputDelta
    | ToolResult
    | BlockEnd
    | IterationStart
    | IterationEnd
    | Error
    | ProviderEvent
    | SessionEndEvent
    | AudioOutputDelta
    | TranscriptDelta
    | SpeechStarted
    | SpeechStopped
    | TurnComplete
)
