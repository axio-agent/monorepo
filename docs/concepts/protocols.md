(protocols)=

# Protocols

Axio's extensibility comes from a small set of **runtime-checkable protocols**
and abstract base classes. Implement any of them to plug your own components
into the framework - no subclassing the agent, no monkey-patching.

```{mermaid}
classDiagram
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
        +check(handler) handler*
    }
    Agent --> CompletionTransport
    Agent --> ContextStore
    Tool --> PermissionGuard
```

## CompletionTransport

The transport protocol has a single method:

<!-- name: test_completion_transport_protocol -->
```python
from typing import runtime_checkable, Protocol
from collections.abc import AsyncIterator
from axio.messages import Message
from axio import Tool, StreamEvent


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]: ...
```

The agent calls `stream()` on every iteration, passing the full conversation
history, the available tools, and the system prompt. The transport yields
`StreamEvent` values as they arrive from the LLM.

Available transports (each in its own installable package):

| Transport | Package | Notes |
|---|---|---|
| `AnthropicTransport` | `axio-transport-anthropic` | Anthropic Claude models |
| `OpenAITransport` | `axio-transport-openai` | OpenAI and OpenAI-compatible APIs |
| `CodexTransport` | `axio-transport-codex` | ChatGPT via OAuth |

The core `axio` package does not bundle any transport implementation - install
the appropriate package for your model provider.

See {doc}`../guides/writing-transports` for a step-by-step guide.

## ContextStore

The context store manages conversation history. It is an abstract base class
with async methods. Only `append` and `get_history` are truly abstract -
everything else has a working default implementation:

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

- `MemoryContextStore` (in `axio`) - in-memory, no persistence; ideal for
  short-lived agents, tests, and prototypes.
- `SQLiteContextStore` (in `axio-context-sqlite`) - persistent, SQLite-backed;
  survives process restarts and supports multiple named sessions per project.

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
: Always returns kwargs unchanged - useful as a no-op default.

`DenyAllGuard`
: Always raises `GuardError("denied")` - useful for locked-down environments.

`ConcurrentGuard`
: Abstract base that serializes (or rate-limits) concurrent `check()` calls
  via a semaphore. Set the `concurrency` class attribute to control
  parallelism (default: 1).

Multiple guards compose sequentially - each guard's output is passed to the
next.

See {doc}`guards` for the full guard system and
{doc}`../guides/writing-guards` for a how-to guide.

## Additional transport protocols

Beyond `CompletionTransport`, Axio defines protocols for other AI modalities.
These protocols are **protocol-only** - the core `axio` package defines the
interfaces, but does not ship with implementations. You can implement them
yourself or use third-party packages when available.

### ImageGenTransport

Generates images from text prompts.

<!-- name: test_imagegen_transport_protocol -->
```python
from typing import runtime_checkable, Protocol


@runtime_checkable
class ImageGenTransport(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        size: tuple[int, int] | None = None,
        n: int = 1,
    ) -> list[bytes]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            size: Optional (width, height) tuple in pixels.
            n: Number of images to generate (default: 1).

        Returns:
            List of raw image bytes (one per generated image).
        """
        ...
```

**Purpose**: Text-to-image generation for tasks like creating diagrams,
illustrations, or visual content on demand.

**Usage pattern**:

```python
async def example():
    from axio.transport import ImageGenTransport

    transport: ImageGenTransport = ...  # your implementation
    images = await transport.generate("a red moon", size=(1024, 1024), n=1)
    assert len(images) == 1
    assert isinstance(images[0], bytes)
```

**Implementation status**: Protocol-only. No official implementation package
ships with Axio. Implement your own or use a third-party package.

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

**Implementation status**: Protocol-only. No official implementation package
ships with Axio. Implement your own or use a third-party package.

---

**Note**: All transport protocols follow the same design principles:

- **Stateless**: All state is passed via arguments; no hidden state between calls.
- **Type-safe**: Protocols are `@runtime_checkable` for isinstance checks.
- **Composable**: Multiple transports can be combined or wrapped.

To implement a custom transport, see {doc}`../guides/writing-transports`.
