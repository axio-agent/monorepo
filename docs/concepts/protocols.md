(protocols)=

# Protocols

Axio's extensibility comes from a small set of **runtime-checkable protocols**
and abstract base classes. Implement any of them to plug your own components
into the framework — no subclassing the agent, no monkey-patching.

```{mermaid}
classDiagram
    class CompletionTransport {
        <<Protocol>>
        +stream(messages, tools, system) AsyncIterator~StreamEvent~
    }
    class ContextStore {
        <<ABC>>
        +session_id str
        +append(message)
        +get_history() list~Message~
        +clear()
        +fork() ContextStore
    }
    class PermissionGuard {
        <<ABC>>
        +check(handler) handler
    }
    Agent --> CompletionTransport
    Agent --> ContextStore
    Tool --> PermissionGuard
```

## CompletionTransport

The transport protocol has a single method:

```python
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

Built-in transports: `OpenAITransport`, `NebiusTransport`, `CodexTransport`.

See [Writing Transports](../guides/writing-transports.md) for a step-by-step guide.

## ContextStore

The context store manages conversation history. It is an abstract base class
with async methods:

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
```

`MemoryContextStore` is the built-in in-memory implementation. The `axio-tui`
package includes a SQLite-backed store for persistence across sessions.

Implement your own `ContextStore` to back conversations with Redis, a database,
or any other storage layer.

See [Context & Messages](context.md) for details on the message model.

## PermissionGuard

Guards gate tool execution. They sit between parameter validation and handler
invocation:

```python
class PermissionGuard(ABC):
    @abstractmethod
    async def check(self, handler: Any) -> Any: ...
```

A guard receives the validated handler instance and either returns it (allowing
execution) or raises `GuardError` to deny. Guards can also modify the handler
before returning it.

Multiple guards compose sequentially — each guard's output is passed to the
next.

See [Guards](guards.md) for the full guard system and
[Writing Guards](../guides/writing-guards.md) for a how-to guide.

## Additional transport protocols

Beyond `CompletionTransport`, Axio defines protocols for other AI modalities:

`ImageGenTransport`
: `async generate(prompt, *, size, n) -> list[bytes]`

`TTSTransport`
: `synthesize(text, *, voice) -> AsyncIterator[bytes]`

`STTTransport`
: `async transcribe(audio, media_type) -> str`

`EmbeddingTransport`
: `async embed(texts) -> list[list[float]]`

These follow the same pattern — implement the protocol, pass the object in,
and the framework handles the rest.
