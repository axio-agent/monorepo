# Context & Messages

The context store holds conversation history. Messages contain typed content
blocks that represent text, images, tool calls, and tool results.

## Content blocks

```{mermaid}
classDiagram
    class ContentBlock {
        <<base>>
    }
    class TextBlock {
        +text: str
    }
    class ImageBlock {
        +media_type: str
        +data: bytes
    }
    class ToolUseBlock {
        +id: ToolCallID
        +name: ToolName
        +input: dict
    }
    class ToolResultBlock {
        +tool_use_id: ToolCallID
        +content: str | list
        +is_error: bool
    }
    ContentBlock <|-- TextBlock
    ContentBlock <|-- ImageBlock
    ContentBlock <|-- ToolUseBlock
    ContentBlock <|-- ToolResultBlock
```

All content blocks are frozen dataclasses:

`TextBlock(text)`
: Plain text content.

`ImageBlock(media_type, data)`
: Binary image data with MIME type (jpeg, png, gif, webp).

`ToolUseBlock(id, name, input)`
: A tool call issued by the model, with its ID, tool name, and input dict.

`ToolResultBlock(tool_use_id, content, is_error)`
: The result of a tool call. `content` can be a string or a list of
  `TextBlock` / `ImageBlock` values.

### Serialization

Every block can be serialized to and from a dict:

```python
from axio.blocks import to_dict, from_dict

d = to_dict(TextBlock(text="hello"))   # {"type": "text", "text": "hello"}
block = from_dict(d)                    # TextBlock(text="hello")
```

## Message

A `Message` pairs a role with a list of content blocks:

```python
@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]
```

User messages typically contain `TextBlock` values. Assistant messages may
contain `TextBlock` and `ToolUseBlock` values. Tool results go into a
separate user message with `ToolResultBlock` values.

## ContextStore

`ContextStore` is an abstract base class — implement it to store conversations
anywhere.

```python
class ContextStore(ABC):
    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @abstractmethod
    async def append(self, message: Message) -> None: ...

    @abstractmethod
    async def get_history(self) -> list[Message]: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def fork(self) -> ContextStore: ...

    async def set_context_tokens(
        self, input_tokens: int, output_tokens: int,
    ) -> None: ...

    async def get_context_tokens(self) -> tuple[int, int]: ...
```

### Built-in implementations

`MemoryContextStore`
: In-memory list of messages. `fork()` returns a deep copy. No persistence.

The `axio-tui` package includes a **SQLite-backed context store** for
persistence across sessions.

### Extension point

Implement `ContextStore` to use any backend:

- Redis for shared state across processes
- PostgreSQL for durable, queryable history
- A vector database for retrieval-augmented context

### Factory methods

`ContextStore` provides two class-method factories:

```python
# Create from existing messages
ctx = await MemoryContextStore.from_history(messages)

# Clone another context store
ctx2 = await MemoryContextStore.from_context(ctx)
```

## Context compaction

Long conversations can exceed the model's context window. The
`compact_context()` function summarizes older messages while keeping recent
ones verbatim:

```python
async def compact_context(
    context: ContextStore,
    transport: CompletionTransport,
    *,
    max_messages: int = 20,
    keep_recent: int = 6,
    system_prompt: str | None = None,
) -> list[Message] | None:
```

It uses a separate one-shot agent call to generate the summary. The function
returns the compacted message list if the history exceeded `max_messages`, or
`None` if no compaction was needed.

The helper `_find_safe_boundary()` ensures that `ToolUseBlock` /
`ToolResultBlock` pairs are never split across the boundary.
