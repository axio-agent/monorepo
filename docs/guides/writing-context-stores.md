# Writing Context Stores

A context store holds the conversation history for an agent session. Axio
ships two built-in implementations - `MemoryContextStore` (in-memory,
ephemeral) and `SQLiteContextStore` (persistent, file-backed) - that cover
most single-process use cases. This guide explains when you need something
different and how to implement it.

For a full description of the API and the built-in stores see
[Context & Messages](../concepts/context.md).

## When to write a custom store

Consider a custom context store when:

- **Shared state across processes.** Multiple agent workers need to read and
  write the same session (e.g. Redis, Memcached, or a database accessed over
  the network).
- **Durable, queryable history.** You want to keep conversation history in
  PostgreSQL alongside other application data, or you need full-text search
  over past sessions.
- **Existing infrastructure.** Your application already has a message store
  (a chat service, a ticket system) and you want the agent to read and write
  it directly.
- **Custom retention or compaction policies.** You need to cap history at N
  tokens, archive old messages to cold storage, or apply per-tenant data
  residency rules at the storage layer.

If none of the above applies, `MemoryContextStore` or `SQLiteContextStore`
are the right choices.

## The contract

`ContextStore` is an abstract base class defined in `axio.context`. Only two
methods are abstract and **must** be provided:

| Method | Guarantee |
|--------|-----------|
| `append(message)` | Append one `Message` to the end of the session history. Must be atomic - a caller that awaits `append` and then calls `get_history` must see the message. |
| `get_history()` | Return a list of all messages in insertion order. Must not mutate the store's internal state. |

Every other method has a default implementation on `ContextStore` that is
correct for simple cases, but may need overriding for production backends:

| Method | Default behaviour |
|--------|-------------------|
| `session_id` | Lazy UUID hex property; generated once per instance. |
| `clear()` | Raises `NotImplementedError`. |
| `fork()` | Deep-copies the history into a fresh `MemoryContextStore`. |
| `close()` | No-op. |
| `set_context_tokens(in, out)` | No-op; tokens are silently dropped. |
| `get_context_tokens()` | Returns `(0, 0)`. |
| `add_context_tokens(in, out)` | Calls `get_context_tokens` then `set_context_tokens`. |
| `list_sessions()` | Returns a single `SessionInfo` for the current session. |

## Minimal implementation

The simplest possible custom store wraps an in-memory list. This is
functionally identical to `MemoryContextStore`, but it is a useful starting
point to demonstrate the required interface before moving to a remote backend.

<!-- name: test_minimal_context_store -->
```python
import asyncio
from axio import ContextStore
from axio.messages import Message
from axio.blocks import TextBlock


class MinimalStore(ContextStore):
    """Minimal in-process context store."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        # Return a copy so callers can't mutate internal state.
        return list(self._messages)


async def main():
    store = MinimalStore()
    assert store.session_id  # lazy UUID hex, no __init__ call needed

    await store.append(Message(role="user", content=[TextBlock(text="Hello")]))
    await store.append(Message(role="assistant", content=[TextBlock(text="Hi!")]))

    history = await store.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"

asyncio.run(main())
```

Key rules:

- Return a **copy** of the internal list from `get_history()`, not a
  reference. Callers (including the agent loop) may iterate and modify the
  list independently.
- The `session_id` property is provided by the base class - you do not need
  to set it if you do not call `super().__init__()`. It is lazily initialised
  the first time it is accessed.

## Overriding lifecycle methods

### `clear()`

The base class raises `NotImplementedError`. Override it if your backend
supports clearing a session:

<!-- name: test_store_clear -->
```python
import asyncio
from axio import ContextStore
from axio.messages import Message
from axio.blocks import TextBlock


class ClearableStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    async def clear(self) -> None:
        """Remove all messages and reset token counts."""
        self._messages.clear()
        self._input_tokens = 0
        self._output_tokens = 0


async def main():
    store = ClearableStore()
    await store.append(Message(role="user", content=[TextBlock(text="Hello")]))
    assert len(await store.get_history()) == 1

    await store.clear()
    assert len(await store.get_history()) == 0

asyncio.run(main())
```

### `fork()`

The default `fork()` deep-copies the history into a new `MemoryContextStore`.
That is sufficient for most in-process scenarios, but override it when:

- You want the forked session to also be persisted (e.g. `SQLiteContextStore`
  does a SQL `INSERT … SELECT` to create a new session row).
- The deep-copy is expensive (large history) and you prefer copy-on-write
  or a pointer/reference to a snapshot.

<!--
name: test_store_fork
```python
import asyncio, copy
from axio import ContextStore
from axio.messages import Message
from axio.blocks import TextBlock
```
-->
<!-- name: test_store_fork -->
```python
import asyncio
import copy
from axio import ContextStore
from axio.messages import Message
from axio.blocks import TextBlock


class ForkableStore(ContextStore):
    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages: list[Message] = list(messages or [])

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    async def fork(self) -> "ForkableStore":
        """Return an independent copy of this session."""
        return ForkableStore(copy.deepcopy(self._messages))


async def main():
    store = ForkableStore()
    await store.append(Message(role="user", content=[TextBlock(text="Hello")]))

    forked = await store.fork()
    await forked.append(Message(role="assistant", content=[TextBlock(text="Hi!")]))

    # Original is unchanged.
    assert len(await store.get_history()) == 1
    assert len(await forked.get_history()) == 2

asyncio.run(main())
```

### `close()`

Override `close()` when your store holds resources that must be released
explicitly: database connections, file handles, network sockets. The default
is a no-op.

```python
async def close(self) -> None:
    await self._conn.close()
```

The `Agent` does **not** call `close()` automatically - the caller that
creates the store is responsible for closing it, typically with
`try / finally` or an `asynccontextmanager`.

## Token tracking

The agent calls `add_context_tokens(input_tokens, output_tokens)` after every
LLM iteration to accumulate usage data. The base class's default
implementation delegates to `get_context_tokens()` and `set_context_tokens()`,
both of which are no-ops - so tokens are silently discarded unless you
override at least `set_context_tokens` and `get_context_tokens`.

Override both methods together:

<!-- name: test_store_token_tracking -->
```python
import asyncio
from axio import ContextStore
from axio.messages import Message


class TokenTrackingStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    async def get_context_tokens(self) -> tuple[int, int]:
        return self._input_tokens, self._output_tokens


async def main():
    store = TokenTrackingStore()

    # The agent calls add_context_tokens after each iteration.
    await store.add_context_tokens(100, 50)
    await store.add_context_tokens(200, 75)

    in_tok, out_tok = await store.get_context_tokens()
    assert in_tok == 300
    assert out_tok == 125

asyncio.run(main())
```

Method summary:

`set_context_tokens(input_tokens, output_tokens)`
: Overwrite the stored counts with the given values. No-op in the base class.

`get_context_tokens() -> tuple[int, int]`
: Return `(input_tokens, output_tokens)`. Returns `(0, 0)` in the base class.

`add_context_tokens(input_tokens, output_tokens)`
: Increment the stored counts. The base class reads the current value via
  `get_context_tokens()` and writes the sum via `set_context_tokens()`. You
  can override this directly for backends (like `SQLiteContextStore`) that
  support atomic increment in a single query.

## Registering a context store

Context stores are not discovered through entry points - they are ordinary
Python classes that you instantiate yourself and pass to the `Agent`.

### Passing to Agent

`Agent.run()` and `Agent.run_stream()` accept any `ContextStore` instance as
their second argument:

<!--
name: test_store_with_agent
```python
import asyncio
from axio import ContextStore
from axio.messages import Message
from axio.testing import StubTransport, make_text_response


class MinimalStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []
    async def append(self, message: Message) -> None:
        self._messages.append(message)
    async def get_history(self) -> list[Message]:
        return list(self._messages)

transport = StubTransport([make_text_response("ok")])
```
-->
<!-- name: test_store_with_agent -->
```python
import asyncio
from axio import Agent

store = MinimalStore()
agent = Agent(
    system="You are a helpful assistant.",
    transport=transport,
)

async def main():
    result = await agent.run("Hello", store)
    assert result == "ok"

asyncio.run(main())
```

A single store instance represents one session. Pass a new store (or call
`fork()`) to start a fresh session while keeping the agent configuration.

### Using with SQLiteContextStore

`SQLiteContextStore` takes a connection created by `axio_context_sqlite.connect`
and can share that connection across many session instances:

```python
import asyncio
from axio_context_sqlite import SQLiteContextStore, connect

async def main():
    conn = await connect("./my_db.sqlite")
    try:
        store = SQLiteContextStore(conn, session_id="session-abc")
        # pass store to Agent.run(...)
    finally:
        await conn.close()

asyncio.run(main())
```

The connection is owned by the caller; `SQLiteContextStore.close()` is a
no-op by design. Close the underlying `aiosqlite.Connection` explicitly when
you are done with the session or on application shutdown.

## Testing

### Testing the store directly

Test `append` and `get_history` first, then each lifecycle method you
override:

<!-- name: test_custom_store_basic -->
```python
import asyncio
from axio import ContextStore
from axio.messages import Message
from axio.blocks import TextBlock


class SimpleStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    async def clear(self) -> None:
        self._messages.clear()


async def test_append_and_history():
    store = SimpleStore()
    msg = Message(role="user", content=[TextBlock(text="hello")])
    await store.append(msg)
    history = await store.get_history()
    assert len(history) == 1
    assert history[0].role == "user"


async def test_get_history_returns_copy():
    store = SimpleStore()
    await store.append(Message(role="user", content=[TextBlock(text="hi")]))
    h1 = await store.get_history()
    h2 = await store.get_history()
    # Mutations to the returned list do not affect the store.
    h1.clear()
    assert len(await store.get_history()) == 1


async def test_clear():
    store = SimpleStore()
    await store.append(Message(role="user", content=[TextBlock(text="hi")]))
    await store.clear()
    assert len(await store.get_history()) == 0


asyncio.run(test_append_and_history())
asyncio.run(test_get_history_returns_copy())
asyncio.run(test_clear())
```

### Testing fork isolation

<!-- name: test_custom_store_fork -->
```python
import asyncio
import copy
from axio import ContextStore
from axio.messages import Message
from axio.blocks import TextBlock


class ForkedStore(ContextStore):
    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages: list[Message] = list(messages or [])

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    async def fork(self) -> "ForkedStore":
        return ForkedStore(copy.deepcopy(self._messages))


async def test_fork_isolation():
    store = ForkedStore()
    await store.append(Message(role="user", content=[TextBlock(text="original")]))

    fork = await store.fork()
    await fork.append(Message(role="assistant", content=[TextBlock(text="branched")]))

    # Original is unchanged.
    assert len(await store.get_history()) == 1
    # Fork has the extra message.
    assert len(await fork.get_history()) == 2
    # Session IDs are independent.
    assert store.session_id != fork.session_id


asyncio.run(test_fork_isolation())
```

### Testing token tracking

<!-- name: test_custom_store_tokens -->
```python
import asyncio
from axio import ContextStore
from axio.messages import Message


class TokenStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._in: int = 0
        self._out: int = 0

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._in = input_tokens
        self._out = output_tokens

    async def get_context_tokens(self) -> tuple[int, int]:
        return self._in, self._out


async def test_token_accumulation():
    store = TokenStore()
    await store.add_context_tokens(100, 40)
    await store.add_context_tokens(50, 20)
    in_tok, out_tok = await store.get_context_tokens()
    assert in_tok == 150
    assert out_tok == 60


asyncio.run(test_token_accumulation())
```

### Testing with the agent using StubTransport

Use `axio.testing.StubTransport` to run an agent against your store without
making real LLM calls:

<!-- name: test_custom_store_with_agent -->
```python
import asyncio
from axio import ContextStore, Agent
from axio.messages import Message
from axio.testing import StubTransport, make_text_response


class RecordingStore(ContextStore):
    """Store that records all messages appended to it."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self.append_count: int = 0

    async def append(self, message: Message) -> None:
        self._messages.append(message)
        self.append_count += 1

    async def get_history(self) -> list[Message]:
        return list(self._messages)


async def test_agent_uses_store():
    store = RecordingStore()
    transport = StubTransport([make_text_response("Done!")])
    agent = Agent(system="You are a test agent.", transport=transport)

    result = await agent.run("Hello", store)

    assert result == "Done!"
    # Agent appends the user message and then the assistant reply.
    assert store.append_count == 2
    history = await store.get_history()
    assert history[0].role == "user"
    assert history[1].role == "assistant"


asyncio.run(test_agent_uses_store())
```

## Tips

- Always return a **copy** from `get_history()`. The agent loop iterates the
  returned list and may pass it to the transport while appending proceeds in
  the background.
- Keep `append()` atomic. If your backend supports transactions, commit
  inside `append` so that a failure in a subsequent call does not leave the
  history in a partially-written state.
- For remote backends, consider whether `fork()` should clone data in the
  backend (like `SQLiteContextStore` does) or fall back to the default
  in-memory deep-copy. The default is safe but does not persist the fork.
- Override `close()` whenever your store holds a connection or file handle.
  The agent does not call it - make the caller responsible, using a
  `try / finally` block or an `asynccontextmanager` wrapper.
