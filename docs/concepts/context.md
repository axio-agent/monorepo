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
from dataclasses import dataclass, field
from typing import Literal
from axio.blocks import ContentBlock

@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock] = field(default_factory=list)
```

User messages typically contain `TextBlock` values. Assistant messages may
contain `TextBlock` and `ToolUseBlock` values. Tool results go into a
separate user message with `ToolResultBlock` values. The `"system"` role is
supported for representing system-level messages in history.

## ContextStore

`ContextStore` is an abstract base class — implement it to store conversations
anywhere. Only two methods are truly abstract and must be overridden:

<!-- name: test_context_store_abc -->
```python
from axio.context import ContextStore
from axio.messages import Message

class MyContextStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    # All other methods have default implementations in ContextStore:
    #   session_id       — lazy UUID hex property (no __init__ required)
    #   clear()          — raises NotImplementedError by default
    #   fork()           — deep-copies history into a MemoryContextStore
    #   close()          — no-op by default
    #   set_context_tokens(input, output)  — no-op by default
    #   get_context_tokens()               — returns (0, 0) by default
    #   add_context_tokens(input, output)  — increments via get/set above
    #   list_sessions()  — returns a single SessionInfo for the current session

store = MyContextStore()
assert store.session_id  # auto-generated UUID hex
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
    # empty store, or pre-populate with existing messages:
    # ctx = MemoryContextStore([existing_message, ...])
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

For long-running agents wrap `MemoryContextStore` with
`AutoCompactStore` to automatically summarize old history
when the context window fills up.

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

`SQLiteContextStore` is the natural choice for production TUI sessions. Pair it
with `AutoCompactStore` to keep context within model limits
across long conversations.

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

### Token tracking

`ContextStore` includes optional token tracking. The agent calls
`add_context_tokens()` after every LLM iteration to accumulate usage:

`add_context_tokens(input_tokens, output_tokens)`
: Increment the stored token counts by the given amounts. The base
  implementation delegates to `get_context_tokens()` and
  `set_context_tokens()`.

`set_context_tokens(input_tokens, output_tokens)`
: Overwrite the stored counts. No-op in the base class.

`get_context_tokens() -> tuple[int, int]`
: Return `(input_tokens, output_tokens)`. Returns `(0, 0)` in the base class.

Both `MemoryContextStore` and `SQLiteContextStore` provide real storage for
these values. Custom stores may override `set_context_tokens` and
`get_context_tokens` to persist usage data.

<!-- name: test_token_tracking -->
```python
import asyncio
from axio.context import MemoryContextStore

async def main():
    ctx = MemoryContextStore()
    await ctx.add_context_tokens(100, 50)
    await ctx.add_context_tokens(200, 80)
    in_tok, out_tok = await ctx.get_context_tokens()
    assert in_tok == 300
    assert out_tok == 130

asyncio.run(main())
```

### Session listing

`list_sessions() -> list[SessionInfo]`
: Returns a list of `SessionInfo` dataclasses describing available sessions.
  The base implementation returns a single entry for the current session.
  `SQLiteContextStore` overrides this to list all sessions for the project,
  ordered newest-first.

`SessionInfo` is a frozen dataclass:

<!-- name: test_session_info -->
```python
from axio.context import SessionInfo

info = SessionInfo(
    session_id="abc123",
    message_count=10,
    preview="What is the capital of France?",
    created_at="2024-01-15 10:30:00",
    input_tokens=1500,
    output_tokens=300,
)
assert info.session_id == "abc123"
assert info.message_count == 10
```

`session_id`
: The unique identifier for the session.

`message_count`
: Total number of messages in the session.

`preview`
: A short excerpt (up to 80 characters) from the first user message.

`created_at`
: Creation timestamp as a string. `MemoryContextStore` returns an empty
  string; `SQLiteContextStore` returns an ISO-format datetime.

`input_tokens`
: Cumulative input token count for the session. Defaults to 0.

`output_tokens`
: Cumulative output token count for the session. Defaults to 0.

## Context compaction

Long conversations can exceed the model's context window. Axio provides
`AutoCompactStore` — a delegating wrapper that automatically summarizes old
history when token usage crosses a threshold.

### AutoCompactStore

`AutoCompactStore` wraps any `ContextStore` backend and compacts it
transparently. It intercepts `add_context_tokens()`, which the agent loop
calls after every `IterationEnd` with the real context size for that
iteration.

<!--
name: test_auto_compact_store
```python
from axio.agent import Agent
from axio.testing import StubTransport, make_text_response
transport = StubTransport([make_text_response("ok")])
agent = Agent(system="you are helpful", transport=transport)
```
-->
<!-- name: test_auto_compact_store -->
```python
import asyncio
from axio.compaction import AutoCompactStore
from axio.context import MemoryContextStore
from axio.messages import Message
from axio.blocks import TextBlock

async def main():
    store = AutoCompactStore(
        MemoryContextStore(),
        transport,          # same transport as the agent
        keep_recent=6,      # keep this many messages verbatim
        threshold=0.75,     # compact at 75 % of context_window (default)
    )
    result = await agent.run("Build a rate limiter", store)

asyncio.run(main())
```

The threshold is read from `transport.model.context_window` via duck typing;
falls back to 128 000 if the transport has no `model` attribute. Pass
`max_tokens` explicitly to override:

<!-- name: test_auto_compact_explicit_max_tokens -->
```python
from axio.compaction import AutoCompactStore
from axio.context import MemoryContextStore
from axio.testing import StubTransport
inner_store = MemoryContextStore()
transport = StubTransport([])
store = AutoCompactStore(inner_store, transport, max_tokens=60_000)
```

`AutoCompactStore` works with any `ContextStore` — `MemoryContextStore`,
`SQLiteContextStore`, or custom implementations.  `fork()` returns an
`AutoCompactStore` wrapping a fork of the inner store, preserving the same
threshold and `keep_recent` settings.

### How it works

1. The agent loop calls `context.add_context_tokens(usage.input_tokens, ...)`
   after each `IterationEnd`. `input_tokens` here is the actual context size
   sent to the model — not a cumulative sum.
2. If `input_tokens > max_tokens`, `_do_compact()` fires.
3. `_do_compact()` forks the inner store first, giving `compact_context` a
   stable snapshot while the live store remains writable.
4. After the (async) summarization agent returns, the live store is cleared
   and repopulated with the compacted messages. Cumulative token counts are
   preserved.

### `compact_context` — low-level function

`AutoCompactStore` uses this internally. Call it directly if you need custom
compaction logic:

<!-- name: test_compact_context -->
```python
from axio.context import ContextStore
from axio.transport import CompletionTransport
from axio.messages import Message

async def compact_context(
    context: ContextStore,
    transport: CompletionTransport,
    *,
    keep_recent: int = 6,
    system_prompt: str = "...",  # defaults to a built-in summarization prompt
) -> list[Message] | None:
    ...
```

Returns the compacted message list, or `None` if the history is too short to
split (`len(history) <= keep_recent`). Does not modify the store — the caller
applies the result.

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
    compacted = await compact_context(ctx, transport, keep_recent=4)
    assert compacted is not None
    # [summary_user, "Understood" assistant] + 4 recent messages
    assert len(compacted) == 6

    new_ctx = await MemoryContextStore.from_history(compacted)
    assert len(await new_ctx.get_history()) == 6

asyncio.run(main())
```
