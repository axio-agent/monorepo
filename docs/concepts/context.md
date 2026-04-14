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

<!-- name: test_serialization -->
```python
from axio.blocks import TextBlock, to_dict, from_dict

d = to_dict(TextBlock(text="hello"))
assert d == {"type": "text", "text": "hello"}
block = from_dict(d)
assert block == TextBlock(text="hello")
```

## Message

A `Message` pairs a role with a list of content blocks:

<!-- name: test_message_dataclass -->
```python
from dataclasses import dataclass
from typing import Literal
from axio.blocks import ContentBlock

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

<!-- name: test_context_store_abc -->
```python
from abc import ABC, abstractmethod
from axio.context import ContextStore
from axio.messages import Message

class ContextStore(ABC):
    @abstractmethod
    async def append(self, message: Message) -> None: ...

    @abstractmethod
    async def get_history(self) -> list[Message]: ...

    # Everything else — session_id, close(), fork(), clear(),
    # get/set_context_tokens() — has a default implementation.
```

### Built-in implementations

#### MemoryContextStore

In-memory list of messages. No persistence — use it for short-lived agents,
tests, and prototypes. `fork()` returns an independent deep copy.

<!-- name: test_memory_context_store -->
```python
import asyncio
from axio.context import MemoryContextStore
from axio.messages import Message
from axio.blocks import TextBlock

async def main():
    ctx = MemoryContextStore()
    await ctx.append(Message(role="user", content=[TextBlock(text="Hello")]))
    await ctx.append(Message(role="assistant", content=[TextBlock(text="Hi!")]))

    history = await ctx.get_history()
    assert len(history) == 2
    assert history[0].role == "user"

    # fork() creates an independent deep copy — useful for branching
    fork = await ctx.fork()
    await fork.append(Message(role="user", content=[TextBlock(text="(branch)")]))
    assert len(await ctx.get_history()) == 2   # original unchanged
    assert len(await fork.get_history()) == 3

    await ctx.close()

asyncio.run(main())
```

#### SQLiteContextStore

Persistent storage backed by SQLite. Survives process restarts and supports
multiple named sessions within a project. Install the `axio-context-sqlite`
package to use it.

<!-- name: test_sqlite_context_store -->
```python
import asyncio, tempfile, pathlib
from axio_context_sqlite import SQLiteContextStore, connect
from axio.messages import Message
from axio.blocks import TextBlock

async def main():
    tmp = pathlib.Path(tempfile.mkdtemp()) / "ctx.db"
    conn = await connect(tmp)
    try:
        store = SQLiteContextStore(conn, session_id="my-session")
        await store.append(Message(role="user", content=[TextBlock(text="Hello")]))
        history = await store.get_history()
        assert len(history) == 1

        # fork() copies messages into a new session
        forked = await store.fork()
        assert len(await forked.get_history()) == 1
        assert forked.session_id != store.session_id
    finally:
        await conn.close()

asyncio.run(main())
```

### Extension point

Implement `ContextStore` to use any backend:

- Redis for shared state across processes
- PostgreSQL for durable, queryable history
- A vector database for retrieval-augmented context

### Factory methods

`ContextStore` provides two class-method factories:

<!-- name: test_context_factory_methods -->
```python
import asyncio
from axio.context import MemoryContextStore
from axio.messages import Message
from axio.blocks import TextBlock

async def main():
    messages = [Message(role="user", content=[TextBlock(text="hello")])]
    # Create from existing messages
    ctx = await MemoryContextStore.from_history(messages)
    # Clone another context store
    ctx2 = await MemoryContextStore.from_context(ctx)

asyncio.run(main())
```

## Context compaction

Long conversations can exceed the model's context window. The
`compact_context()` function summarizes older messages while keeping recent
ones verbatim:

<!-- name: test_compact_context -->
```python
from axio.context import ContextStore
from axio.transport import CompletionTransport
from axio.messages import Message

async def compact_context(
    context: ContextStore,
    transport: CompletionTransport,
    *,
    max_messages: int = 20,
    keep_recent: int = 6,
    system_prompt: str | None = None,
) -> list[Message] | None:
    ...
```

It uses a separate one-shot agent call to generate the summary. The function
returns the compacted message list if the history exceeded `max_messages`, or
`None` if no compaction was needed.

The helper `_find_safe_boundary()` ensures that `ToolUseBlock` /
`ToolResultBlock` pairs are never split across the boundary.

When `compact_context` returns a list, populate a fresh store and continue
from there:

<!--
name: test_compact_context_usage
```python
from axio.testing import StubTransport, make_text_response
transport = StubTransport([make_text_response("Earlier: user asked about deployment.")])
```
-->
<!-- name: test_compact_context_usage -->
```python
import asyncio
from axio.compaction import compact_context
from axio.context import MemoryContextStore
from axio.messages import Message
from axio.blocks import TextBlock

async def main():
    ctx = MemoryContextStore()
    for i in range(22):
        role = "user" if i % 2 == 0 else "assistant"
        await ctx.append(Message(role=role, content=[TextBlock(text=f"Message {i}")]))

    # compact: keep 4 recent messages verbatim, summarize the rest
    compacted = await compact_context(ctx, transport, max_messages=20, keep_recent=4)
    assert compacted is not None
    # [summary_user, "Understood" assistant] + 4 recent messages
    assert len(compacted) == 6

    new_ctx = await MemoryContextStore.from_history(compacted)
    assert len(await new_ctx.get_history()) == 6

asyncio.run(main())
```
