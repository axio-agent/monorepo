# AGENTS.md

Guidance for AI coding agents working in this repository.

---

## Repository layout

This is a **uv workspace** monorepo. Each subdirectory is an independent Python package with its own `pyproject.toml`, `src/` layout, and `tests/`. They share a single `uv.lock` and a single `.venv`.

- `axio/` - core library; start here when unsure
- `axio-sse/` - `text/event-stream` reader. No dependencies at all
- `axio-responses/` - the OpenAI Responses API, shared by two transports
- `axio-transport-*/` - transport implementations (`openai`, `anthropic`, `google`, `codex`)
- `axio-tools-*/` - tool providers (`local`, `docker`, `mcp`)
- `axio-context-sqlite/` - SQLite context store
- `axio-tui/`, `axio-tui-guards/` - Textual application and its guard plugins
- `axio-repl/` - terminal REPL assistant
- `axio-audio/` - microphone and speaker helpers for realtime agents
- `docs/` - Sphinx sources and markdown-pytest doc tests
- `examples/` - runnable example scripts: `agent_swarm`, `gas_town`, `realtime_smoke`, `realtime_chat`

Fifteen distributable packages, plus `docs/` and the four examples, make up `[tool.uv.workspace] members` in the root `pyproject.toml`:

| Package | Purpose |
|---|---|
| `axio` | Agent loop, Tool, transport protocols, ContextStore, PermissionGuard, events, blocks, types |
| `axio-sse` | `text/event-stream`: `Decoder`, `Wire` payload shapes, `Reader`. Stdlib only |
| `axio-responses` | The OpenAI Responses API: request builders and the stream reader |
| `axio-transport-openai` | `/v1/responses` (default) and `/v1/chat/completions`; also `EmbeddingTransport` and OpenAI realtime. Subclasses point at compatible servers: Nebius, OpenRouter, custom |
| `axio-transport-anthropic` | Anthropic Claude, direct API and Vertex, with prompt caching |
| `axio-transport-google` | Gemini: completion, image generation, video generation, Gemini Live realtime |
| `axio-transport-codex` | ChatGPT via OAuth, over the Responses API |
| `axio-tools-local` | File, shell, Python execution tools |
| `axio-tools-mcp` | MCP server bridge |
| `axio-tools-docker` | Docker sandbox tool provider |
| `axio-context-sqlite` | SQLite-backed persistent context store |
| `axio-tui` | Textual TUI and entry-point plugin discovery. Uses `axio-context-sqlite` for history; its own `sqlite_config.py` is project config, not a context store |
| `axio-tui-guards` | PathGuard + LLMGuard plugins |
| `axio-repl` | Interactive REPL assistant (`axio-repl` console script) |
| `axio-audio` | `Microphone`, `Speaker`, `DuplexAudio` for `RealtimeAgent` |

---

## Commands

Always use `make`. Never invoke `uv run pytest`, `ruff`, or `mypy` directly at the repo root.

```bash
make              # lint + type-check + tests for all packages + doc tests
make linter       # ruff check + ruff format --check on all packages
make typing       # mypy --strict on all packages
make pytest       # pytest on all packages
make test-docs    # markdown-pytest on docs/
```

Run a single package's tests or checks:

```bash
make PACKAGES=axio-transport-anthropic
```

Run a specific test file inside a package:

```bash
uv run --directory axio pytest tests/test_agent_run.py -v
```

Run doc tests for a single file:

```bash
uv run --directory docs pytest -v guides/best-practices.md
```

---

## Development setup

```bash
git clone https://github.com/mosquito/axio-agent.git
cd axio-agent
uv sync --all-packages   # installs all workspace members + dev deps into .venv
```

After sync, all local packages resolve to their workspace sources via `[tool.uv.sources]`. No `pip install -e` or PYTHONPATH hacks are needed.

---

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/), `line-length = 119`, `target-version = "py312"`
- **Type checker**: [mypy](https://mypy.readthedocs.io/) strict mode (`--strict`), `python_version = "3.12"`
- Enabled ruff rules: `E`, `F`, `I`, `UP`
- All new code must pass `mypy --strict` with zero errors
- Use `from __future__ import annotations` at the top of every module
- Prefer `dataclass(frozen=True, slots=True)` for value types
- `Tool` is generic. Write `list[Tool[Any]]`, never a bare `list[Tool]`. Strict mode rejects the missing parameter

Always run `make linter` and `make typing` before considering a task done.

---

## Architecture

### Public API (`axio/__init__.py`)

42 symbols are importable directly from `axio`:

```python
from axio import Agent, Tool, CONTEXT, ContextStore, MemoryContextStore, CompletionTransport
from axio import Message, TextBlock, ReasoningBlock, ToolUseBlock, ToolResultBlock
from axio import StreamEvent, TextDelta, Refusal, Citation, ReasoningSignature
from axio import BlockEnd, IterationStart, IterationEnd, ProviderEvent
from axio import ToolUseStart, ToolInputDelta, ToolResult
from axio import RealtimeAgent, RealtimeTransport, RealtimeSession
from axio import AudioOutputDelta, TranscriptDelta, SpeechStarted, SpeechStopped, TurnComplete
from axio import StopReason, Usage, GuardError, HandlerError
from axio import PermissionGuard, ConcurrentGuard
from axio import Field, FieldInfo, StrictStr, ToolSelector, AgentStream
```

Not re-exported at top level - import from the submodule:

```python
from axio.events import SessionEndEvent, Error, ReasoningDelta, ToolOutputDelta, ImageOutput
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.schema import build_tool_schema
from axio.agent_loader import AgentSpec, load_agents
from axio.compaction import AutoCompactStore, compact_context
```

### Types (`axio/types.py`)

```python
type ToolName = str          # Unique tool identifier
type ToolCallID = str        # Opaque ID for a tool invocation

class StopReason(StrEnum):
    end_turn = "end_turn"                              # Assistant finished responding
    tool_use = "tool_use"                              # Assistant wants to call a tool
    max_tokens = "max_tokens"                          # Output truncated
    error = "error"                                    # Something went wrong
    refusal = "refusal"                                # Declined or blocked. Terminal, not an error
    pause_turn = "pause_turn"                          # Server-side tool loop paused. Resumable
    context_window_exceeded = "context_window_exceeded"  # Outgrew the window. Truncated
    cancelled = "cancelled"                            # Stopped before it finished
```

`refusal` is terminal and deliberately not an error. The same prompt sent again is declined again. A caller told "error" cannot tell a decline from a broken connection. `pause_turn` is the only reason that does not end the run.

```python
@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = field(default=0, kw_only=True)
    cache_write_tokens: int = field(default=0, kw_only=True)
    reasoning_tokens: int = field(default=0, kw_only=True)

    # __add__ sums all five fields.
    # Properties: total_tokens, uncached_input_tokens, answer_tokens
```

**The Usage rule.** `input_tokens` and `output_tokens` are always inclusive grand totals. Every other field is a disjoint slice of one of them:

```
cache_read_tokens + cache_write_tokens  <=  input_tokens
reasoning_tokens                        <=  output_tokens
```

Providers disagree about this, and they disagree in opposite directions. **Each transport converts into the rule**, so nothing downstream has to know which provider answered:

- Anthropic counts only the tokens after the last cache breakpoint, so `axio-transport-anthropic` adds the cache counts back into `input_tokens`.
- Gemini reports thinking beside the candidates and tool-use prompt tokens outside the prompt count, so `axio-transport-google` adds both. `cachedContentTokenCount` is already inside the prompt count and is not added. The transport also warns when Gemini's own `totalTokenCount` disagrees with the parts.
- The Responses API and `/v1/chat/completions` already nest both slices inside their totals, so nothing is added there.

Counts only, never money. A cached token and a written one bill at different multipliers. `ModelSpec` carries only `input_cost` and `output_cost`, with no cached-input or reasoning rate in the registry. A caller that wants cost therefore multiplies these slices by its own per-model rates. A zero slice means the provider billed none of it, or reported no breakdown at all. Axio cannot tell those apart.

### Messages (`axio/messages.py`)

The fundamental unit of conversation history:

```python
@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message: ...
```

Messages are serialized for transport to LLM providers and for persistence in context stores.

### Content Blocks (`axio/blocks.py`)

Seven `ContentBlock` subclasses represent all content in messages. All are `dataclass(frozen=True, slots=True)`.

| Block | Fields | Purpose |
|---|---|---|
| `TextBlock` | `text: str` | Plain text content |
| `ImageBlock` | `media_type: ImageMediaType`, `data: bytes` | Image attachment (base64 in serialization) |
| `AudioBlock` | `media_type: AudioMediaType`, `data: bytes` | Audio attachment |
| `VideoBlock` | `media_type: VideoMediaType`, `data: bytes` | Video attachment |
| `ReasoningBlock` | `text: str = ""`, `signature: str = ""`, `redacted: bool = False`, `id: str = ""` | The model's own reasoning, kept so the turn can be replayed |
| `ToolUseBlock` | `id: ToolCallID`, `name: ToolName`, `input: dict[str, Any]` | A tool call request |
| `ToolResultBlock` | `tool_use_id: ToolCallID`, `content: str \| list[TextBlock \| ImageBlock \| AudioBlock \| VideoBlock]`, `is_error: bool = False` | Result of tool execution |

```python
# Serialization helpers
def to_dict(block: ContentBlock) -> dict[str, Any]: ...   # singledispatch, one registration per type
def from_dict(data: dict[str, Any]) -> ContentBlock: ...
```

Both handle the full round-trip including nested blocks in `ToolResultBlock`.

`ReasoningBlock.signature` is the provider's proof that the block is its own. **Never inspect, decode, re-encode or truncate it.** Anthropic refuses a returned thinking block whose signature is missing or changed. Google publishes `MISSING_THOUGHT_SIGNATURE` for the same failure. `axio-transport-google` maps that status to `StopReason.error`, because prompting again does not fix it. A context store that round-trips a `ReasoningBlock` and loses `signature` fails on the *next* turn, not on the one that dropped it.

### Events (`axio/events.py`)

All events are `dataclass(frozen=True, slots=True)`. `StreamEvent` is a **type alias union** of 26 members, not a base class. Nothing inherits from it. `isinstance(x, StreamEvent)` raises. A `match` over it needs a `case _`.

Content:

| Event | Fields |
|---|---|
| `ReasoningDelta` | `index: int`, `delta: str` |
| `ReasoningSignature` | `index: int`, `data: str`, `redacted: bool = False`, `id: str = ""` |
| `TextDelta` | `index: int`, `delta: str` |
| `Refusal` | `index: int`, `text: str = ""`, `category: str \| None = None`, `blocked_input: bool = False`, `raw: dict = {}` |
| `Citation` | `index: int`, `cited_text: str = ""`, `title`, `url`, `source_id`, `start`, `end`, `unit: Literal["char","byte","page","block","unknown"] = "unknown"`, `raw` |
| `ImageOutput` / `AudioOutput` / `VideoOutput` | `index: int`, `data: bytes`, `media_type` |

Tool calls:

| Event | Fields |
|---|---|
| `ToolUseStart` | `index: int`, `tool_use_id: ToolCallID`, `name: ToolName` |
| `ToolInputDelta` | `index: int`, `tool_use_id: ToolCallID`, `partial_json: str` |
| `ToolFieldStart` / `ToolFieldEnd` | `index: int`, `tool_use_id: ToolCallID`, `key: str` |
| `ToolFieldDelta` | `index: int`, `tool_use_id: ToolCallID`, `key: str`, `text: str` |
| `ToolOutputDelta` | `tool_use_id: ToolCallID`, `name: ToolName`, `key: str`, `delta: str` |
| `ToolResult` | `tool_use_id`, `name`, `is_error: bool`, `content: str = ""`, `input: dict = {}` |

Lifecycle:

| Event | Fields |
|---|---|
| `IterationStart` | `iteration: int`, `id: str \| None = None`, `model: str \| None = None` |
| `BlockEnd` | `index: int` |
| `IterationEnd` | `iteration: int`, `stop_reason: StopReason`, `usage: Usage` |
| `Error` | `exception: BaseException` (**not** `Exception` - narrowing to `Exception` fails `mypy --strict`) |
| `ProviderEvent` | `provider: str`, `kind: str`, `data: dict[str, Any]`, `index: int \| None = None` |
| `SessionEndEvent` | `stop_reason: StopReason`, `total_usage: Usage` |

Realtime (duplex sessions): `AudioOutputDelta(data, media_type="audio/pcm;rate=24000")`, `TranscriptDelta(role, delta)`, `SpeechStarted()`, `SpeechStopped()`, `TurnComplete(stop_reason, usage=None)`.

Who emits what, where it is not uniform:

- `IterationStart` is emitted by all four transports, always with `iteration=0`. A transport does not know the agent's iteration number. The agent does not renumber it. Its `model` is the model that *actually served* the turn, which need not be the one requested. Server-side fallback, sticky routing and dated-snapshot resolution all substitute a different model at a different price. A cost lookup therefore keys off `IterationStart.model`. `IterationEnd.iteration` is `0` from every transport for the same reason.
- `BlockEnd` comes only from Anthropic and the Responses reader. Google and the chat-completions path emit none, so a consumer must not treat it as universal. The agent does not act on it. Tool JSON is finalized at `IterationEnd`.
- `Citation` comes from Anthropic and the Responses reader. Google forwards its grounding metadata as `ProviderEvent` instead. The agent never stores citations.
- `Refusal.text` is not always populated. Anthropic sends one whole refusal with a category. OpenAI (both endpoints) streams it fragment by fragment. Google's refusal is prompt-level, carries `blocked_input=True` and no text at all. A Google candidate blocked for SAFETY/RECITATION maps to `StopReason.refusal` while emitting no `Refusal` event.
- `ToolField*` is emitted by no transport. It comes from `axio.tool_args.ToolArgStream`, which the caller drives. The agent does not instantiate it.
- `ToolOutputDelta` is emitted by the agent for tools where `Tool.supports_streaming` is true.

**Important**: the stream **must** end with exactly one `IterationEnd` event per LLM call.

### Reading a provider stream (`axio-sse`)

Three of the four transports import `axio_sse` directly. The fourth reads through `axio-responses`, which is built on it. No transport parses `text/event-stream` itself any more. The package has no dependencies at all. It takes an async iterable of bytes and yields events. Whoever produced the bytes is not its business, which is why it is a separate distribution.

| Name | What it is |
|---|---|
| `Decoder` | The format as a sans-io state machine, shaped like `codecs.IncrementalDecoder`: `decode(chunk, final=False) -> list[Event]` and `reset()` |
| `events(chunks, *, until="")` | Async wrapper over `Decoder`, yielding `Event` |
| `payloads(chunks, *, until="")` | The same, yielding the JSON object of each event as `Payload` |
| `Event` | `data`, `event`, `id`, `retry`, plus `payload() -> Payload \| None` |
| `Payload` | A `dict` subclass with four path readers: `string()`, `number()`, `obj()`, `objs()` |
| `Wire` | One payload shape per wire name, read field-by-declared-name-and-type |
| `Reader` | One endpoint's vocabulary, as one `@on(...)` method per event |
| `EVENT_NAME` | `by=EVENT_NAME` dispatches on the format's own `event:` field |
| `Handled[T]` | What an `@on` method returns: `Iterable[T] \| None` |
| `UnknownEvent` | Raised for an unclaimed name when reading with `strict=True` |

`Decoder` takes **chunks, never lines**. `aiohttp`'s `readuntil` raises `LineTooLong` past 131072 bytes. `LineTooLong` is not a `ClientError`. One large reasoning event killed a turn with no answer. For the same reason, chunks must carry their line terminators. `aiter_lines()` strips them, and nothing dispatches. `until` names the non-JSON sentinel that closes a stream (`until="[DONE]"`), so it never reaches a caller.

`Wire` declares one payload shape, named on the class line:

```python
@dataclass(frozen=True, slots=True)
class OutputTextDelta(Wire, name="response.output_text.delta"):
    delta: str = ""
    output_index: int = 0
```

Every field is read by its declared name and type, so a misspelled key is a type error where it is used rather than a default quietly standing in. A field the provider omitted, sent as null, or sent as the wrong type takes its default. `also=` adds further names. A field declared `raw: Payload` receives the whole payload, for shapes that vary too much to declare whole.

A stream whose events are all one shape needs nothing above `payloads()`. `axio-transport-google` and the chat-completions path read that way, and both emit `ProviderEvent` inline. A stream that names its events subclasses `Reader`. `axio-transport-anthropic` (`by=EVENT_NAME`) and `axio_responses.Responses` do that.

**The `unmatched()` rule, which is not obvious.** A `Reader` names only the events it *interprets*. Everything else goes through `unmatched()`, which returns nothing by default. The OpenAI and Anthropic readers override it to forward those events as `ProviderEvent` under the provider's own name. An endpoint that runs tools publishes one event family per tool. That set therefore depends on which tools exist and which the caller declared, not on the protocol. Listing it goes stale the day a tool is added, and reports a new tool as news about the protocol. `strict=True` still raises `UnknownEvent` for anything unnamed (`read()` calls `unknown()` before `unmatched()`), so a test can hold `Reader.names()` against the schema the provider publishes. `unknown()` is also called by hand for a second discriminator nested inside one event - the delta type inside a content block - so a nested name nobody reads fails the same replay.

The natural instinct, adding a handler for every event in the provider's docs, is the wrong one here.

### The Responses API (`axio-responses`)

Both halves of the OpenAI Responses API live in one package because two transports speak it: `axio-transport-openai` (the public `/v1/responses` endpoint) and `axio-transport-codex` (the ChatGPT backend). Depends on `axio` and `axio-sse`.

- `convert_messages(messages, system) -> tuple[str, list[dict]]` - returns `(instructions, input_items)`
- `convert_tools(tools) -> list[dict]`
- `strip_title(schema)` - drops pydantic `title` keys recursively
- `STOP_REASONS: dict[str, StopReason]` - the published-status map
- `Responses` - an `axio_sse.Reader[StreamEvent]` that reads the stream
- 22 `Wire` shapes (`TextDeltaEvent`, `ReasoningDeltaEvent`, `RefusalDeltaEvent`, `ItemAdded`, `OutputItem`, `ResponseUsage`, `IncompleteDetails`, `StreamFailure`, ...) - the declared surface of the stream

`Responses.names()` is a **subset** of the published `ResponseStreamEvent` union, not the whole of it. The events the API runs its own tools with are forwarded through `unmatched()` rather than named. The test that fixes this asserts `Responses.names() <= PUBLISHED_EVENTS`, one-directionally.

### Transport protocol (`axio/transport.py`)

Nine `@runtime_checkable` Protocols. A transport implements as many as it supports. `OpenAITransport` is `CompletionTransport, EmbeddingTransport`. `GoogleTransport` is `CompletionTransport, ImageGenTransport, VideoGenTransport`.

```python
@runtime_checkable
class CompletionTransport(Protocol):
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]: ...
```

| Protocol | Method |
|---|---|
| `CompletionTransport` | `stream(messages, tools, system) -> AsyncIterator[StreamEvent]` |
| `ImageGenTransport` | `async generate_images(prompt, *, model=None, n=1) -> list[bytes]` |
| `VideoGenTransport` | `async generate_videos(prompt, *, model=None, n=1, image=None, duration_seconds=None, aspect_ratio=None) -> list[bytes]` |
| `AudioGenTransport` | `async generate_audios(prompt, *, model=None, n=1) -> list[bytes]` |
| `RealtimeTransport` | `async connect(*, system, tools, voice=None, input_audio_format=..., output_audio_format=...) -> RealtimeSession` |
| `RealtimeSession` | `send`, `commit`, `interrupt`, `send_tool_result`, `events`, `close` |
| `TTSTransport` | `synthesize(text, *, voice=None) -> AsyncIterator[bytes]` |
| `STTTransport` | `async transcribe(audio, media_type="audio/wav") -> str` |
| `EmbeddingTransport` | `async embed(texts) -> list[list[float]]` |

`DummyCompletionTransport` and its six siblings (`DummyImageGenTransport`, `DummyVideoGenTransport`, `DummyAudioGenTransport`, `DummyTTSTransport`, `DummySTTTransport`, `DummyEmbeddingTransport`) fail loudly when called. Use one as the transport of an agent prototype. Swap the real one in with `agent.copy(transport=...)`.

Contract for `stream()`:

- The stream **must** end with exactly one `IterationEnd` event.
- Do not suppress exceptions. The reference transports raise `StreamError`. `Agent._run_loop` catches it and yields `Error` + `SessionEndEvent(error)`. Yielding `IterationEnd(stop_reason=error)` instead reaches the agent's `case _` wildcard, which reports `RuntimeError("Transport stopped with: error")` and names nothing.
- `messages` **includes** the current user input. The agent appends it before reading history. Its text is not the caller's string but `f"[{ts}] {user_message}"` with a local timestamp, which is visible in stored history.
- On the final iteration only, the agent hands `[*history, last_iteration_message]` rather than the stored history. That extra message is never appended to the store.
- `tools` is `list[Tool[Any]]`, already filtered by the selector. It is empty when the model lacks `Capability.tool_use`.
- The agent reads three *optional* attributes off the transport with `getattr`, so a transport may provide them or not: `model` (for `model.capabilities`), `last_usage` (added to the total when repetition truncates a turn), and `nudge_on_media_tool_result`.

### Writing a transport

The rules a new transport has to satisfy. The walkthrough is in `docs/guides/writing-transports.md`.

1. **Read the stream through `axio-sse`.** Do not parse `text/event-stream` by hand and do not read lines.
2. **Convert token counts into the Usage rule** (see Types above). Pick your side before reporting anything.
3. **Map every stop reason the provider publishes**, in a module-level dict, and fall back to `StopReason.error` with a warning. Coverage is not uniform, and the docs should not claim it is. `context_window_exceeded` is mapped only by Anthropic. `cancelled` is mapped only by the Responses map. Google makes two interesting choices. `MALFORMED_FUNCTION_CALL` reads as `tool_use`, because prompting again is the recovery. `MISSING_THOUGHT_SIGNATURE` reads as `error`, because prompting again is not.
4. **Replay reasoning** if the provider requires it. All three that do, and how:
   - Anthropic: send the thinking block back with its signature (`redacted_thinking` when there is no text). An unsigned block is dropped rather than sent, because the API would refuse it.
   - Google: send the thought part with `thought: True` and its `thoughtSignature`. A signature that arrived on a *function-call* part goes back on that part, taken in arrival order from a `deque`. A single slot gave the first of several parallel calls the last signature.
   - Responses: send a reasoning item with `encrypted_content` and `id`. It arrives only because the request asked for it with `include: ["reasoning.encrypted_content"]`. It matters only because `store: False` means nothing is kept provider-side. Codex does not set `include`, so nothing comes back there.
5. **Emit `IterationStart` before any content**, with `iteration=0` and the model that served the turn.
6. **Forward what you do not interpret** as `ProviderEvent`, rather than dropping it.

`OpenAITransport.api` is `Literal["responses", "chat"] = "responses"`. The default endpoint is therefore `/v1/responses`. `_path()` routes on it. `OpenAICompatibleTransport`, `NebiusTransport` and `OpenRouterTransport` each override it to `"chat"`, because compatible servers rarely implement `/v1/responses`. The default moved because `/v1/chat/completions` refuses function tools beside any reasoning effort other than `"none"` for a model that reasons. `build_chat_payload` therefore sets `reasoning_effort: "none"` and warns. The warning marks a paid reasoning model asked not to reason.

`extra_params` **merges** `tools` rather than replacing them. A caller adding a hosted tool would otherwise take away the function declarations the agent needs dispatched. The turn would then read as the model choosing to call nothing. A later declaration with a matching name wins.

### Agent loop (`axio/agent.py`)

`Agent` is a `dataclass(slots=True)` - not frozen - with fields:

```python
@dataclass(slots=True)
class Agent:
    system: str                                          # System prompt
    transport: CompletionTransport                       # LLM backend
    tools: list[Tool[Any]] = field(default_factory=list)  # Available tools
    selector: ToolSelector | None = None                 # Optional tool selection logic
    max_iterations: int = 50                             # Iteration limit
    last_iteration_message: Message | None = None        # Injected on the final iteration only
```

Public API:

- `agent.run_stream(user_message, context) -> AgentStream` - streaming entry point, yields `StreamEvent` plus `SessionEndEvent`
- `await agent.run(user_message, context) -> str` - convenience wrapper over `AgentStream.get_final_text()`
- `agent.copy(**overrides) -> Agent` - `dataclasses.replace`; the documented way to configure a prototype
- `await agent.dispatch_tools(blocks, iteration) -> list[ToolResultBlock]`

`get_final_text()` concatenates `TextDelta.delta` **and** `Refusal.text`, so `run()` returns the decline for a declined turn instead of the empty string it used to. It raises `StreamError` on an `Error` event.

**Loop per iteration**:

1. Call `transport.stream(effective_history, active_tools, system)` → `AsyncIterator[StreamEvent]`.
2. Yield every event verbatim, then accumulate into the turn being built:
   - `TextDelta` → merged into the trailing `TextBlock`
   - `Refusal` → its text folded into the turn's text. A refusal is what the assistant said. Left out, the stored turn is empty. The next request then carries a blank assistant message the provider rejects
   - `ReasoningDelta` → merged into the trailing `ReasoningBlock`. A **signed** block is finished. A later delta starts a new block, because the provider computed the signature over the text it had
   - `ReasoningSignature` → attached to that block. A repeated index continues one proof rather than starting another
   - `ImageOutput` / `VideoOutput` → appended as `ImageBlock` / `VideoBlock`
   - `ToolUseStart` / `ToolInputDelta` → buffered as pending tool calls
   - `IterationEnd` → pending calls parsed into `ToolUseBlock`s, usage totalled, `context.add_context_tokens(input, output)`
3. If any `ToolUseBlock` was produced:
   - Dispatch all valid calls **concurrently** via `asyncio.gather()`. Malformed JSON is answered with a retry message instead of being executed
   - Append the assistant message and the tool results
   - `continue`
4. Otherwise append the assistant message - **on every path, before the match** - and branch on `stop_reason`:
   - `end_turn` → `SessionEndEvent(end_turn)`, return
   - `refusal` → `SessionEndEvent(refusal)`, return. Terminal, and deliberately not an error
   - `pause_turn` → `continue`. The provider stopped its own tool loop and expects the assistant content back. That content was just appended, so going round again *is* the resume. Bounded by `max_iterations` like any other iteration
   - `case _` → `Error(RuntimeError(...))` then `SessionEndEvent(error)`, return. The wildcard is on purpose. Named one by one, a reason added later matches nothing and falls out of the match. The model is then re-prompted with unchanged history until `max_iterations`, and the caller pays for every one of those turns

Two exits are easy to miss. `_RepetitionDetector` breaks the stream when the output loops, appends `[Output truncated: repetitive content detected]` to the turn, and ends the session with `end_turn`. Reaching `max_iterations` emits `SessionEndEvent(error)` and no `Error` event.

Tool dispatch happens **before** appending to context. This prevents orphaned `ToolUseBlock`s if a task is cancelled mid-loop.

### Tool system (`axio/tool.py`)

A tool handler is a plain `async def` function. Parameters become the input JSON schema. The docstring becomes the description. Use `Annotated` + `Field` from `axio.field` to add per-parameter descriptions, defaults, or numeric bounds.

```python
async def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    Path(path).write_text(content)
    return f"wrote {len(content)} bytes"
```

`Tool` is a `dataclass(frozen=True, slots=True)`:

```python
@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: ToolName                          # Unique identifier
    handler: Callable[..., Awaitable[Any]]  # Plain async function
    description: str = ""                   # Defaults to handler.__doc__
    guards: tuple[PermissionGuard, ...] = ()   # Run sequentially; any GuardError denies
    concurrency: int | None = None          # Optional per-tool semaphore limit
    context: T = field(default=MappingProxyType({}), compare=False)   # Runtime state for CONTEXT.get()
    schema: MappingProxyType[str, Any] = field(default=MappingProxyType({}), ...)  # Explicit schema override
```

The return annotation is `Awaitable[Any]`, not `Awaitable[str]`. The widening is load-bearing. The agent accepts a list of `ContentBlock`s, so a handler may return an image or audio and not only a string. Anything else is `str()`-ed.

`schema` left empty is derived from the handler's type hints. Set explicitly, it replaces the derived one. The field table is then synthesised from its `properties`. `input_schema` is a deep copy of it.

Streaming tools: `supports_streaming` is true when the handler exposes a `.stream` async-generator attribute. The agent then dispatches through `call_streaming()` and emits `ToolOutputDelta`, aggregating with `format_stream_result(chunks)`.

`Tool.__call__(**kwargs)` pipeline:

1. Acquire semaphore (if `concurrency` is set)
2. Field validation from type hints / `FieldInfo`
3. Guards run **sequentially**: each receives `(tool, **kwargs)` and returns modified kwargs or raises `GuardError`
4. `await handler(**kwargs)` - execute handler

### ContextStore (`axio/context.py`)

`ContextStore` is an ABC with exactly **two** abstract methods. Everything else has a working default. Override only what your backend can do better.

```python
class ContextStore(ABC):
    @abstractmethod
    async def append(self, message: Message) -> None: ...

    @abstractmethod
    async def get_history(self) -> list[Message]: ...
```

Concrete, with defaults: `session_id` (lazy UUID hex), `clear()` (raises `NotImplementedError`), `fork()` (deep copy into a `MemoryContextStore`), `set_context_tokens()` (no-op), `get_context_tokens()` (returns `(0, 0)`), `close()` (no-op), `list_sessions()` (one `SessionInfo` for the current session), `add_context_tokens()` (get + set), and the classmethods `from_history()` and `from_context()`.

There is no `compact()` on `ContextStore`. Compaction lives in `axio/compaction.py`, as `compact_context()` and `AutoCompactStore`. `AutoCompactStore` is a wrapper store that summarises once usage passes 75 % of `transport.model.context_window` (128 000 where the transport has no `model`).

The one method a store must not silently drop is `add_context_tokens()`. The agent calls it after every `IterationEnd`. `AutoCompactStore` fires from it.

**Implementations**:
- `MemoryContextStore` - in-process, ephemeral storage
- `axio-context-sqlite` - `SQLiteContextStore` for persistence across sessions

### PermissionGuard (`axio/permission.py`)

`PermissionGuard` is an ABC. Implement:

```python
class PermissionGuard(ABC):
    @abstractmethod
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        """Return (possibly modified) kwargs to allow, raise GuardError to deny."""
        ...
```

Guards are not limited to access control. Because `check()` receives the `Tool` object
and the raw kwargs before the handler executes, guards are also the right place for
**logging, auditing, and display**:

```python
class AuditGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        logger.info("tool=%s args=%s", tool.name, kwargs)
        return kwargs  # always allow; raise GuardError to deny
```

See `examples/agent_swarm/agent_swarm/__main__.py` (`RoleGuard`) for a production example.

**ConcurrentGuard**: Use as base when the guard itself must be rate-limited (e.g. LLM-based approval). Provides internal semaphore management.

### Exceptions (`axio/exceptions.py`)

Full exception hierarchy:

```python
class AxioError(Exception):
    """Base exception for all axio errors."""

class ToolError(AxioError):
    """Base for tool-related errors."""

class GuardError(ToolError):
    """Guard denied or crashed during permission check."""

class HandlerError(ToolError):
    """Handler raised during execution."""

class StreamError(AxioError):
    """Error during stream collection."""
```

### Testing (`axio/testing.py`)

Helper classes and functions for testing:

| Function | Returns | Purpose |
|---|---|---|
| `StubTransport(responses)` | `StubTransport` | Yields pre-configured event sequences per `stream()` call |
| `make_text_response(text="Done", iteration=2, usage=None)` | `list[StreamEvent]` | Build a simple end_turn response |
| `make_tool_use_response(tool_name="echo", tool_id="call_1", tool_input=None, iteration=1, usage=None)` | `list[StreamEvent]` | Build a tool_use response sequence |
| `make_stub_transport()` | `StubTransport` | Pre-configured with a "Hello world" text response |
| `make_ephemeral_context()` | `MemoryContextStore` | Fresh empty context |
| `make_echo_tool()` | `Tool[Any]` | `Tool(name="echo", description="Returns input as JSON", handler=_msg_input)` |

```python
# Example: StubTransport with multiple responses
async def example(my_tool: Tool) -> str:
    transport = StubTransport([
        make_tool_use_response("my_tool", tool_input={"x": 1}),
        make_text_response("Done"),
    ])
    agent = Agent(system="...", transport=transport, tools=[my_tool])
    return await agent.run("go", make_ephemeral_context())
```

`StubTransport` pops the next event sequence on each `stream()` call. If there are fewer sequences than calls, it repeats the last one.

`StubTransport` takes an arbitrary `list[list[StreamEvent]]`, so anything the builders do not cover - a `Refusal`, a `ReasoningSignature`, a `pause_turn`, a `Usage` with slices - is written out event by event. It discards `messages`, `tools` and `system`, so it cannot assert what was *sent*. A test that checks reasoning replay needs a recording transport of its own.

### Plugin system (entry points)

Plugins register via `pyproject.toml` entry points. `axio-tui/src/axio_tui/plugin.py` discovers six groups at startup:

| Group | Registers | Registered by |
|---|---|---|
| `axio.tools` | plain async handler functions | `axio-tools-local`, `axio-transport-google`, `axio-tui` |
| `axio.tools.settings` | `ToolsPlugin` (dynamic tool sets) | `axio-tools-mcp` |
| `axio.transport` | `CompletionTransport` implementations | all four transports |
| `axio.transport.settings` | TUI settings screens (Textual `Screen` subclasses) | all four transports |
| `axio.guards` | `PermissionGuard` subclasses | `axio-tui-guards` |
| `axio.selector` | `ToolSelector` implementations | nothing yet |

A seventh group, `axio.transport.realtime`, is registered by `axio-transport-openai` (`openai`) and `axio-transport-google` (`gemini`, `vertex`). `examples/realtime_chat/chat.py` reads it, rather than the TUI.

---

## Testing

### Unit tests

Each package has `tests/` with pytest. Test files follow the pattern `test_<module>.py`. Build agent-level tests out of `axio.testing` (table above).

A `Reader` subclass is tested by holding its claimed set against the provider's published event list, and by reading a recorded stream with `strict=True`:

```python
assert Responses.names() <= PUBLISHED_EVENTS   # subset: unclaimed names are forwarded, not named
```

### Doc tests

Documentation in `docs/` is tested with [markdown-pytest](https://github.com/mosquito/markdown-pytest), and so are the package READMEs. Most packages set `testpaths = ["tests", "README.md"]`. Annotate code blocks with HTML comments:

```markdown
<!-- name: test_my_example -->
```python
import asyncio
from axio.agent import Agent
# ... asyncio.run() for async code
```
```

Hidden setup (stubs that must not appear in rendered docs):

```markdown
<!--
name: test_my_example
```python
# This block is invisible in docs but runs before the named block
from axio.testing import StubTransport, make_text_response
```
-->
```

A named block is executed, so it must run with no network and no API keys. Use `axio.testing.StubTransport`. Leave the name off a block you do not want run. This file is not collected by any `testpaths`, so blocks here are illustrative only.

Run doc tests:

```bash
make test-docs
# or for a single file:
uv run --directory docs pytest -v guides/writing-transports.md
```

---

## Adding a new package

1. Create `axio-<name>/` with a `src/axio_<name>/` layout, including `py.typed`, and a `pyproject.toml` matching the style of existing packages: hatchling build backend, `[tool.hatch.build.targets.wheel] packages = ["src/axio_<name>"]`, ruff + mypy + pytest dev deps, `asyncio_mode = "auto"`, and `testpaths = ["tests", "README.md"]` with `markdown-pytest>=0.6.0` in dev deps. The README's code blocks are then executed by `make pytest`.
2. Add to `[tool.uv.workspace] members` and `[tool.uv.sources]` in the root `pyproject.toml`.
3. Add to `PACKAGES` in `Makefile`.
4. Register it with the docs build, or it gets no API reference. Add it to `dependencies` in `docs/pyproject.toml`, add a `docs/api/<name>.md` page, and list that page in a toctree in `docs/api/index.md`. (Of the fifteen packages, only `axio-tui` and `axio-tui-guards` have no API page.)
5. Run `uv sync --all-packages` to update `uv.lock`.

---

## What not to do

- **Do not** run `uv run pytest` or `ruff` from the repo root. Use `make`.
- **Do not** add dependencies to the root `pyproject.toml`. It is a workspace manifest only.
- **Do not** edit `uv.lock` manually. `uv sync` generates it.
- **Do not** use `asyncio_mode = "auto"` in ad-hoc scripts. It is only for pytest.
- **Do not** add a guard's blocking I/O in the hot path without subclassing `ConcurrentGuard`.
- **Do not** add a `Usage` field that is not a disjoint slice of `input_tokens` or `output_tokens`. Do not pass a provider's counts through without converting into that rule.
- **Do not** name provider tool events one by one in a `Reader`. Forward them through `unmatched()`. That set tracks the tools, not the protocol.
- **Do not** inspect, decode, re-encode or truncate a reasoning signature. Do not drop it when serializing a turn.
- **Do not** replace the `case _` in a `StopReason` match with an exhaustive list of members. An unhandled reason must end the run, not re-prompt.
