# Glossary

Terms used throughout Axio documentation.

## A

**Agent**
: The core component that orchestrates LLM interactions. Receives messages, decides on tool calls, and returns responses.

**AsyncIterator**
: A Python protocol for asynchronous streaming. Axio transports yield `StreamEvent` values via this protocol.

## B

**BlockEnd**
: Event marking the content block at `index` complete. The point at which accumulated `ToolInputDelta` fragments are guaranteed to parse. It is not universal. Anthropic and the Responses reader emit it. Gemini and the chat-completions path do not.

## C

**Chat Completions**
: OpenAI's `/v1/chat/completions` endpoint. `OpenAITransport` speaks it when `api="chat"`. `OpenAICompatibleTransport`, `NebiusTransport` and `OpenRouterTransport` set that. It refuses function tools beside any reasoning effort other than `"none"`. See **Responses API**.

**Citation**
: Event attributing a span of generated text to a source. `unit` says what `start` and `end` count, because the providers disagree. OpenAI counts characters and Google counts bytes. Offsets from different units must never be compared.

**CompletionTransport**
: Protocol defining how Axio talks to LLM providers. Implement this to add support for new APIs.

**Context**
: The conversation history passed to the LLM. Includes system prompt, user messages, and assistant responses.

**ContextStore**
: Protocol for persisting conversation history. Implementations: `MemoryContextStore`, `SQLiteContextStore`.

**Context Compaction**
: Technique for reducing context size when it approaches token limits. Uses summarization or truncation.

## D

**Decoder**
: `axio_sse.Decoder`. The `text/event-stream` format as a synchronous state machine, shaped like `codecs.IncrementalDecoder`. `decode(chunk, final=False)` returns the events the chunk completed. `reset()` forgets a half-read one. It takes chunks and never lines, because `aiohttp`'s `readuntil` raises `LineTooLong` past 131072 bytes. One large reasoning event otherwise kills the turn.

## E

**Event**
: `axio_sse.Event`. One dispatched server-sent event, with the four fields the format defines: `data`, `event`, `id`, `retry`. `payload()` returns its JSON object as a `Payload`, or `None` where it carries none.

**EVENT_NAME**
: The `by=` sentinel that tells a `Reader` to dispatch on the format's own `event:` field rather than on a key inside the payload.

**Event Stream**
: The flow of typed events from transport to agent. Includes tokens, tool calls, reasoning, and completion signals.

## G

**Guard**
: A permission check that runs before tool execution. Can allow, deny, or modify the handler input.

## H

**Handled**
: What a `Reader` handler returns: `Iterable[T] | None`. One rule for none, one, and many.

## I

**IterationEnd**
: An event signaling the end of one LLM call iteration. Contains usage statistics and stop reason.

**IterationStart**
: An event signaling that one provider request has begun. Its `model` is the model that actually served the turn, which need not be the one asked for. Server-side fallback, sticky routing and dated-snapshot resolution all substitute a different model at a different price. A cost lookup therefore keys off this. Every transport passes `iteration=0`. A transport cannot know the agent's iteration number.

## L

**LLM**
: Large Language Model. The AI model that powers agent reasoning (OpenAI, Anthropic, etc.).

## M

**MemoryContextStore**
: In-memory context storage. Fast but loses data on shutdown.

**ModelSpec**
: A specification for an LLM model: `id`, `capabilities`, `max_output_tokens`, `context_window`, `input_cost`, `output_cost`. It carries no per-slice rate, so cached-input and reasoning tokens cannot be priced from the registry.

## P

**PermissionGuard**
: Abstract class for implementing guards. Define `check()` method to allow/deny tool calls.

**Protocol**
: Runtime-checkable interface (Python `Protocol` or ABC). Axio uses protocols for pluggability.

**Parameter annotation**
: Type hint on a tool handler parameter. Axio reads annotations to build the JSON schema sent to the LLM. Use `Annotated[T, Field(...)]` from `axio.field` to attach descriptions, defaults, or numeric bounds.

**Payload**
: `axio_sse.Payload`. The JSON object inside one event. A `dict` subclass with four readers that take a path: `string()`, `number()`, `obj()`, `objs()`. Each returns its default wherever a step is missing, null, or the wrong type, which is what an optional provider field is.

**ProviderEvent**
: A provider payload axio does not model, forwarded verbatim under the provider's own name. `data` is the parsed JSON exactly as it arrived. A consumer that does not recognise `(provider, kind)` ignores it. See **unmatched()**.

## R

**Reader**
: `axio_sse.Reader`. What one endpoint sends, as one `@on(...)` method per event. `by=` on the class line names the payload key holding the event name, or `EVENT_NAME` for the format's own `event:` field. One instance reads one response, because the turn's running totals live on it. `names()` exposes the claimed set for a test.

**ReasoningBlock**
: A content block holding the model's own reasoning: `text`, `signature`, `redacted`, `id`. The agent stores one in the assistant turn so the turn can be replayed. See **reasoning signature**.

**ReasoningDelta**
: An event containing model reasoning/thinking tokens. Every completion transport emits it.

**Reasoning replay**
: Sending a stored `ReasoningBlock` back on the next request. That is why the block exists. Anthropic wants a `thinking` block with its signature. Google wants a thought part with its `thoughtSignature`. The Responses API wants a `reasoning` item with `encrypted_content` and `id`.

**Reasoning signature**
: The provider's opaque proof that a reasoning block is its own. `ReasoningSignature` carries it, and `ReasoningBlock.signature` stores it. Never inspect, decode, re-encode or truncate it. Anthropic refuses a returned thinking block whose signature is missing or changed. Google reports `MISSING_THOUGHT_SIGNATURE` for the same failure.

**Redacted reasoning**
: A reasoning block whose text the provider withheld. `redacted` is true and `text` is empty. The signature still has to travel.

**Refusal**
: An event saying the model declined, or the provider blocked the turn. Deliberately not a `TextDelta`: as ordinary assistant text a decline is indistinguishable from an answer. `blocked_input` is true where the prompt was rejected and nothing was generated. Terminal but not an error — the same prompt sent again will be declined again.

**Responses API**
: OpenAI's `/v1/responses` endpoint, the default for `OpenAITransport` (`api="responses"`) and the API `axio-transport-codex` speaks. It takes function tools and reasoning together, which `/v1/chat/completions` refuses. `axio-responses` holds both halves of it.

## S

**SSE**
: Server-Sent Events, the `text/event-stream` media type. How every completion transport in Axio receives a streamed response. The format itself lives in `axio-sse`, which is a separate distribution with no dependencies at all.

**StopReason**
: Why the provider stopped generating: `end_turn`, `tool_use`, `max_tokens`, `error`, `refusal`, `pause_turn`, `context_window_exceeded`, `cancelled`. Everything except `tool_use` and `pause_turn` ends the run. `pause_turn` is the one that resumes. The provider stopped its own server-side tool loop and expects the assistant content back. The agent therefore appends the turn and goes round again.

**StreamEvent**
: The union of every event a transport or the agent can yield. A type alias, not a base class. Nothing inherits from it, and `isinstance` against it raises. Match on the member types.

**Sub-agent**
: A child agent spawned from a parent agent. Used for parallel task execution or delegation.

## T

**Tool**
: The glue object that declares an async Python handler as an agent tool. It
  combines the handler with a stable name, generated input schema, description,
  and optional guards.

**Tool call**
: One concrete request from the model to invoke a named Tool with structured
  arguments. A Tool is the definition. A tool call is an invocation of it.

**Tool handler**
: The executable logic for a tool. A plain `async def` function. Its parameters define the input schema, and its body implements execution.

**ToolUseStart**
: Event signaling the start of a tool call. Contains tool name and unique ID.

**ToolInputDelta**
: Event containing partial JSON input for a tool call. Streamed for tools with large arguments.

**Transport**
: The bridge between Axio and an LLM provider. Handles API calls, streaming, and authentication.

## U

**UnknownEvent**
: What a `Reader` raises for a name no method claims, when reading with `strict=True`. What a test holds against the schema a provider publishes.

**unmatched()**
: The `Reader` hook for a payload no method claims. It returns nothing by default. A reader overrides it to forward rather than drop. The OpenAI and Anthropic readers do that as `ProviderEvent` under the provider's own name. The reason is that an endpoint that runs tools publishes one event family per tool. That set therefore depends on which tools exist and which the caller declared, not on the protocol. Named one by one, the list goes stale the day a tool is added. `strict=True` still refuses anything unnamed.

**Usage**
: Token counts for one provider request. `input_tokens` and `output_tokens` are always inclusive grand totals. `cache_read_tokens`, `cache_write_tokens` and `reasoning_tokens` are disjoint slices of one of them. Providers disagree about this in opposite directions, so each transport converts into the rule. The properties `total_tokens`, `uncached_input_tokens` and `answer_tokens` derive from the five fields. Counts only, never money — see {doc}`quick-start`.

## W

**Wire**
: `axio_sse.Wire`. One payload shape per wire name, named on the class line with `name=` and optional `also=`. Every field is read by its declared name and type. A misspelled key is therefore a type error where it is used, rather than a default quietly standing in. A field the provider omitted, sent as null, or sent as the wrong type takes its default. A field declared `raw: Payload` receives the whole payload, for a shape that varies too much to declare.

## Other

**GuardError**
: Exception raised by guards to deny tool execution. The error message is sent back to the model.

**HandlerError**
: Exception raised by tool handlers for expected failures. Distinguishes from unexpected crashes.

**to_thread()**
: Python asyncio function for running blocking code in a thread pool. Used for CPU-bound tools.
