(protocols)=

# Protocols

Axio's extensibility comes from a small set of **runtime-checkable protocols**
and abstract base classes. Implement any of them to plug your own components
into the framework. Subclassing the agent and monkey-patching are not needed.

```{mermaid}
classDiagram
    direction TB
    class CompletionTransport {
        <<Protocol>>
        +stream(messages, tools, system) AsyncIterator~StreamEvent~
    }
    class ContextStore {
        <<ABC>>
        +append(message)*
        +get_history() list~Message~*
        +session_id str
        +fork() ContextStore
        +clear()
        +close()
        +list_sessions() list~SessionInfo~
    }
    class PermissionGuard {
        <<ABC>>
        +check(tool, **kwargs) dict*
    }
    Agent --> CompletionTransport
    Agent --> ContextStore
    Tool --> PermissionGuard
```

## CompletionTransport

The transport protocol has a single method:

<!-- name: test_completion_transport_protocol -->
```python
from typing import Any, runtime_checkable, Protocol
from collections.abc import AsyncIterator
from axio.messages import Message
from axio import Tool, StreamEvent


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> AsyncIterator[StreamEvent]: ...
```

The agent calls `stream()` on every iteration, passing the full conversation
history, the available tools, and the system prompt. The transport yields
`StreamEvent` values as they arrive from the LLM.

Available transports (each in its own installable package):

| Transport | Package | Notes |
|---|---|---|
| {class}`~axio_transport_anthropic.AnthropicTransport` | `axio-transport-anthropic` | Anthropic Claude models |
| {class}`~axio_transport_openai.OpenAITransport` | `axio-transport-openai` | OpenAI and OpenAI-compatible APIs |
| {class}`~axio_transport_google.GoogleTransport` | `axio-transport-google` | Gemini Developer API |
| {class}`~axio_transport_google.VertexAITransport` | `axio-transport-google` | Gemini and Anthropic on Vertex AI |
| {class}`~axio_transport_codex.CodexTransport` | `axio-transport-codex` | ChatGPT via OAuth |

`OpenAITransport` speaks two endpoints. Its `api` field picks between them, and
defaults to `"responses"` (`/v1/responses`). The OpenAI-compatible subclasses -
`OpenAICompatibleTransport`, `NebiusTransport`, `OpenRouterTransport` - set
`api="chat"`, because compatible servers rarely implement `/v1/responses`.

The core `axio` package does not bundle any transport implementation. Install
the appropriate package for your model provider.

Every shipped transport reads its `text/event-stream` response through
`axio-sse` rather than parsing SSE by hand. `axio-transport-openai` and
`axio-transport-codex` also share `axio-responses`, which holds the Responses
API's request builders and stream reader - the vocabulary those two
transports speak. Neither package is a transport on its own; install them
transitively as a transport's dependency, not directly. See {doc}`../api/sse`
and {doc}`../api/responses`.

A transport is also where two contracts outside the protocol are honoured. Every
published provider stop reason is mapped onto a `StopReason`. An unmapped one
becomes `StopReason.error`, which the agent's wildcard turns into a terminated
run. Provider token counts are converted into the inclusive-totals rule
described in {doc}`models`.

See {doc}`../guides/writing-transports` for a step-by-step guide.

## ContextStore

The context store manages conversation history. It is an abstract base class
with async methods. Only `append` and `get_history` are truly abstract.
Everything else has a working default implementation:

<!-- name: test_context_store_abc -->
```python
from axio.messages import Message
from axio import ContextStore


class MyContextStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    # Default implementations provided by ContextStore (override as needed):
    #   session_id          - lazy UUID hex property
    #   clear()             - raises NotImplementedError by default
    #   fork()              - deep-copies history into a MemoryContextStore
    #   close()             - no-op by default
    #   set_context_tokens(input, output)  - no-op by default
    #   get_context_tokens()               - returns (0, 0) by default
    #   add_context_tokens(input, output)  - increments via get/set above
    #   list_sessions()     - returns a single SessionInfo for the current session

store = MyContextStore()
assert store.session_id  # auto-generated UUID hex
```

Built-in implementations:

- `MemoryContextStore` (in `axio`) - in-memory, with no persistence. It
  suits short-lived agents, tests, and prototypes.
- `SQLiteContextStore` (in `axio-context-sqlite`) - persistent and
  SQLite-backed. It survives process restarts and supports multiple named
  sessions per project.

Implement your own `ContextStore` to back conversations with Redis, a database,
or any other storage layer.

See {doc}`context` for details on the message model.

## PermissionGuard

Guards gate tool execution. They sit between parameter validation and handler
invocation. `PermissionGuard` is an abstract base class (ABC) - not a
Protocol. Subclass it and implement `check()`:

<!-- name: test_permission_guard_abc -->
```python
from typing import Any
from axio import PermissionGuard, Tool


class MyGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        # return kwargs to allow, raise GuardError to deny
        return kwargs
```

A guard receives the `Tool` object and the raw keyword arguments. It must
either return a `dict` of (possibly modified) kwargs to allow execution, or
raise `GuardError` to deny. Guards can modify the kwargs before returning them.

Tool calls are made via `await guard(tool, **kwargs)`, which delegates to `check()`.
The `ConcurrentGuard` subclass additionally wraps `check()` in an
`asyncio.Semaphore` to control parallelism.

Axio ships three built-in guards:

`AllowAllGuard`
: Always returns kwargs unchanged. Useful as a no-op default.

`DenyAllGuard`
: Always raises `GuardError("denied")`. Useful for locked-down environments.

`ConcurrentGuard`
: Abstract base that serializes (or rate-limits) concurrent `check()` calls
  via a semaphore. Set the `concurrency` class attribute to control
  parallelism (default: 1).

Multiple guards compose sequentially. Each guard's output is passed to the
next.

See {doc}`guards` for the full guard system and
{doc}`../guides/writing-guards` for a how-to guide.

## Additional transport protocols

Beyond `CompletionTransport`, axio defines protocols for other AI modalities.
Some are implemented by a shipped transport, and some are interfaces waiting
for an implementation. Each section below says which.

### ImageGenTransport

Generates images from text prompts.

<!-- name: test_imagegen_transport_protocol -->
```python
from typing import runtime_checkable, Protocol


@runtime_checkable
class ImageGenTransport(Protocol):
    async def generate_images(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
    ) -> list[bytes]:
        """Generate ``n`` image samples for a text prompt.

        Args:
            prompt: Text description of the image to generate.
            model: Optional model id, overriding the transport's default.
            n: Number of samples to generate (default: 1).

        Returns:
            List of raw image bytes - PNG, JPEG or WebP, provider-defined.
        """
        ...
```

There is no `size` argument. Aspect ratio, resolution and the rest are
provider-specific knobs that live as extra keyword arguments on the
implementation rather than in the protocol.

**Usage pattern**:

<!-- name: test_imagegen_transport_protocol -->
```python
async def example():
    from axio.transport import ImageGenTransport

    transport: ImageGenTransport = ...  # your implementation
    images = await transport.generate_images("a red moon", n=1)
    assert len(images) == 1
    assert isinstance(images[0], bytes)
```

**Implementation status**: implemented by
{class}`~axio_transport_google.GoogleTransport`, which declares
`CompletionTransport`, `ImageGenTransport` and `VideoGenTransport` together.

### VideoGenTransport

Generates videos from text prompts, optionally seeded with an image.

<!-- name: test_videogen_transport_protocol -->
```python
from typing import runtime_checkable, Protocol


@runtime_checkable
class VideoGenTransport(Protocol):
    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        image: bytes | None = None,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]:
        """Generate ``n`` video samples. Returns raw MP4 / WebM bytes."""
        ...
```

**Implementation status**: implemented by
{class}`~axio_transport_google.GoogleTransport` (Veo).

### AudioGenTransport

Generates non-speech audio - music, sound effects, ambience. Distinct from
`TTSTransport`, which is text-to-speech.

<!-- name: test_audiogen_transport_protocol -->
```python
from typing import runtime_checkable, Protocol


@runtime_checkable
class AudioGenTransport(Protocol):
    async def generate_audios(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
    ) -> list[bytes]:
        """Generate ``n`` audio samples. Returns raw MP3 / WAV / OGG bytes."""
        ...
```

**Implementation status**: Protocol-only. No package that ships with axio
implements it.

### TTSTransport

Synthesizes speech from text (text-to-speech).

<!-- name: test_tts_transport_protocol -->
```python
from typing import runtime_checkable, Protocol
from collections.abc import AsyncIterator


@runtime_checkable
class TTSTransport(Protocol):
    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesize speech from text.

        Args:
            text: Text to convert to speech.
            voice: Optional voice identifier (provider-specific).

        Yields:
            Chunks of audio data (e.g., WAV or MP3 bytes).
        """
        ...
```

**Purpose**: Convert agent responses to audio for voice assistants or
accessibility features.

**Usage pattern**:

```python
async def example():
    from axio.transport import TTSTransport

    transport: TTSTransport = ...  # your implementation
    audio_chunks = [
        chunk
        async for chunk in transport.synthesize("Hello world", voice="alloy")
    ]
    audio_data = b"".join(audio_chunks)
    assert isinstance(audio_data, bytes)
```

**Implementation status**: Protocol-only. No official implementation package
ships with Axio. Implement your own or use a third-party package.

### STTTransport

Transcribes audio to text (speech-to-text).

<!-- name: test_stt_transport_protocol -->
```python
from typing import runtime_checkable, Protocol


@runtime_checkable
class STTTransport(Protocol):
    async def transcribe(
        self,
        audio: bytes,
        media_type: str = "audio/wav",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw audio bytes.
            media_type: MIME type of the audio (e.g., "audio/wav", "audio/mp3").

        Returns:
            Transcribed text.
        """
        ...
```

**Purpose**: Convert user voice input to text for processing by the agent.

**Usage pattern**:

```python
async def example():
    from axio.transport import STTTransport

    transport: STTTransport = ...  # your implementation
    audio_data = b"..."  # raw audio bytes
    text = await transport.transcribe(audio_data, media_type="audio/wav")
    assert isinstance(text, str)
```

**Implementation status**: Protocol-only. No official implementation package
ships with Axio. Implement your own or use a third-party package.

### EmbeddingTransport

Generates vector embeddings from text.

<!-- name: test_embedding_transport_protocol -->
```python
from typing import runtime_checkable, Protocol


@runtime_checkable
class EmbeddingTransport(Protocol):
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...
```

**Purpose**: Generate embeddings for RAG (retrieval-augmented generation),
semantic search, or similarity comparisons.

**Usage pattern**:

```python
async def example():
    from axio.transport import EmbeddingTransport

    transport: EmbeddingTransport = ...  # your implementation
    embeddings = await transport.embed(["hello", "world"])
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], list)
    assert isinstance(embeddings[0][0], float)
```

**Implementation status**: implemented by
{class}`~axio_transport_openai.OpenAITransport`, which declares
`CompletionTransport` and `EmbeddingTransport` together and calls
`/v1/embeddings` with its own `model`.

### RealtimeTransport and RealtimeSession

A duplex session - audio, text and tools travelling both ways at once - rather
than a request that yields a response. `RealtimeTransport.connect()` opens one:

<!-- name: test_realtime_transport_protocol -->
```python
from typing import Any, runtime_checkable, Protocol
from collections.abc import AsyncIterator

from axio import StreamEvent, Tool
from axio.blocks import ContentBlock
from axio.types import ToolCallID, ToolName


@runtime_checkable
class RealtimeSession(Protocol):
    async def send(self, content: ContentBlock | list[ContentBlock]) -> None: ...
    async def commit(self) -> None: ...
    async def interrupt(self) -> None: ...
    async def send_tool_result(
        self, tool_use_id: ToolCallID, name: ToolName, content: str | list[ContentBlock]
    ) -> None: ...
    def events(self) -> AsyncIterator[StreamEvent]: ...
    async def close(self) -> None: ...


@runtime_checkable
class RealtimeTransport(Protocol):
    async def connect(
        self,
        *,
        system: str,
        tools: list[Tool[Any]],
        voice: str | None = None,
        input_audio_format: str = "audio/pcm;rate=16000",
        output_audio_format: str = "audio/pcm;rate=24000",
    ) -> RealtimeSession: ...
```

`send_tool_result` takes the tool `name` beside the call id because some
providers (Gemini Live) require it. OpenAI realtime ignores it. `commit()`
signals end-of-utterance for manual voice-activity detection. It is a no-op
under server VAD.

A session yields the realtime events described in {doc}`events` -
`AudioOutputDelta`, `TranscriptDelta`, `SpeechStarted`, `SpeechStopped`,
`TurnComplete` - alongside the ordinary content events.

**Implementation status**: implemented by `OpenAIRealtimeTransport`
(`axio-transport-openai`) and by `GeminiLiveTransport` / `VertexLiveTransport`
(`axio-transport-google`). See {doc}`../guides/realtime-audio`.

## Placeholder transports

`axio.transport` also ships a `Dummy*` implementation of each protocol -
`DummyCompletionTransport`, `DummyImageGenTransport`, `DummyVideoGenTransport`,
`DummyAudioGenTransport`, `DummyTTSTransport`, `DummySTTTransport`,
`DummyEmbeddingTransport`. Each one raises `RuntimeError` when called.

They exist so an agent prototype can be declared before a provider is chosen.
Construct the agent with a placeholder. Swap the real transport in with
`agent.copy(transport=...)` at runtime. A prototype that was never configured
then fails loudly at the call site, instead of quietly doing nothing.

---

**Note**: All transport protocols follow the same design principles:

- **Stateless**: All state is passed via arguments. No state is hidden between calls.
- **Type-safe**: Protocols are `@runtime_checkable` for isinstance checks.
- **Composable**: Multiple transports can be combined or wrapped.

To implement a custom transport, see {doc}`../guides/writing-transports`.
