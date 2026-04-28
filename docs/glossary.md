# Glossary

Terms used throughout Axio documentation.

## A

**Agent**
: The core component that orchestrates LLM interactions. Receives messages, decides on tool calls, and returns responses.

**AsyncIterator**
: A Python protocol for asynchronous streaming. Axio transports yield `StreamEvent` values via this protocol.

## C

**CompletionTransport**
: Protocol defining how Axio talks to LLM providers. Implement this to add support for new APIs.

**Context**
: The conversation history passed to the LLM. Includes system prompt, user messages, and assistant responses.

**ContextStore**
: Protocol for persisting conversation history. Implementations: `MemoryContextStore`, `SQLiteContextStore`.

**Context Compaction**
: Technique for reducing context size when it approaches token limits. Uses summarization or truncation.

## E

**Event Stream**
: The flow of typed events from transport to agent. Includes tokens, tool calls, reasoning, and completion signals.

## G

**Guard**
: A permission check that runs before tool execution. Can allow, deny, or modify the handler input.

## I

**IterationEnd**
: An event signaling the end of one LLM call iteration. Contains usage statistics and stop reason.

## L

**LLM**
: Large Language Model. The AI model that powers agent reasoning (OpenAI, Anthropic, etc.).

## M

**MemoryContextStore**
: In-memory context storage. Fast but loses data on shutdown.

**ModelSpec**
: A specification for an LLM model (name, provider, capabilities, context window).

## P

**PermissionGuard**
: Abstract class for implementing guards. Define `check()` method to allow/deny tool calls.

**Protocol**
: Runtime-checkable interface (Python `Protocol` or ABC). Axio uses protocols for pluggability.

**Parameter annotation**
: Type hint on a tool handler parameter. Axio reads annotations to build the JSON schema sent to the LLM. Use `Annotated[T, Field(...)]` from `axio.field` to attach descriptions, defaults, or numeric bounds.

## R

**ReasoningDelta**
: An event containing model reasoning/thinking tokens. Some providers (Anthropic) stream reasoning separately.

## S

**SSE**
: Server-Sent Events. Mechanism used by OpenAI/Anthropic APIs for streaming responses.

**StreamEvent**
: Base type for all events in the agent loop. Includes text deltas, tool calls, reasoning, and iteration end.

**Sub-agent**
: A child agent spawned from a parent agent. Used for parallel task execution or delegation.

## T

**Tool**
: A callable that the LLM can invoke. Combines a name, description, handler, and optional guards.

**Tool handler**
: The executable logic for a tool. A plain `async def` function whose parameters define the input schema and whose body implements execution.

**ToolUseStart**
: Event signaling the start of a tool call. Contains tool name and unique ID.

**ToolInputDelta**
: Event containing partial JSON input for a tool call. Streamed for tools with large arguments.

**Transport**
: The bridge between Axio and an LLM provider. Handles API calls, streaming, and authentication.

## U

**Usage**
: Token consumption statistics from an LLM call. Includes input and output token counts.

## Other

**GuardError**
: Exception raised by guards to deny tool execution. The error message is sent back to the model.

**HandlerError**
: Exception raised by tool handlers for expected failures. Distinguishes from unexpected crashes.

**to_thread()**
: Python asyncio function for running blocking code in a thread pool. Used for CPU-bound tools.