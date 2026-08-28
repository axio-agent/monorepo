# From Chat to Agent

Ask a chatbot to summarize `README.md` and it can produce a plausible answer.
It cannot read the project unless the application gives it that capability.

An Axio Agent with `tools=[]` can only exchange messages with its transport.
You do not need a new loop. You need to add one controlled action to Axio's
existing loop.

## Outcome

You will build a project-document assistant that handles one `read_document`
tool call and then returns a final answer.

## Fast Track

1. Construct `Agent(..., tools=[])` and observe the chat-only boundary.
2. Write one plain async `read_document` handler.
3. Wrap the handler in `Tool` and pass it to the same Agent.
4. Use `StubTransport` to verify the tool round trip exactly.

{download}`Download the complete example <../../examples/tutorial/from_chat_to_agent.py>`.

## Hands-on delta

### 1. Start without tools

The transport below returns one text response. The Agent has no action it can
offer before that response ends.

```{literalinclude} ../../examples/tutorial/from_chat_to_agent.py
:language: python
:caption: examples/tutorial/from_chat_to_agent.py
:start-after: "# [docs:start-from-chat-chat-only]"
:end-before: "# [docs:end-from-chat-chat-only]"
```

With a live transport, the model might refuse, guess, or explain what it would
do. None of those responses can contain newly retrieved document data.

### 2. Add one async handler

A handler is ordinary application code. It does not know about messages,
provider formats, or the agent loop.

Keep the small `DOCUMENTS` mapping from the downloaded snapshot. Add the
handler, wrap it as a tool, and register that tool on the Agent:

```{literalinclude} ../../examples/tutorial/from_chat_to_agent.py
:language: python
:caption: examples/tutorial/from_chat_to_agent.py
:start-after: "# [docs:start-from-chat-first-tool]"
:end-before: "# [docs:end-from-chat-first-tool]"
```

`Tool` gives the handler a stable model-facing name. It also derives a schema
from the Python signature and uses the docstring as the default description.

Read {doc}`../concepts/tools` for the complete `Tool` contract.

### 3. Notice the missing loop code

Your harness did not implement `while`, parse tool-call JSON, or append a tool
result manually.

```{mermaid}
sequenceDiagram
    participant H as Harness
    participant A as Axio Agent
    participant T as Transport
    participant G as read_document Tool
    H->>A: run(prompt, context)
    A->>T: stream(messages, tools, system)
    T-->>A: read_document call
    A->>G: validated arguments
    G-->>A: document result
    A->>T: stream(history with result, tools, system)
    T-->>A: final text
    A-->>H: answer
```

Axio owns this sequence. A transport performs one model request; the Agent
decides whether the returned stop reason requires another iteration.

See {doc}`../concepts/agent` for exact iteration and completion behavior.

## Try It

Run `uv run python examples/tutorial/from_chat_to_agent.py` from the repository
root. The program first proves the chat-only failure, then proves the tool
round trip:

```text
chat-only: I cannot read README.md without a tool.
with-tool: README.md: Project overview.
```

Replace the stub with a provider transport when you want to observe live model
selection. The Agent, tool, and context interfaces do not change.

## Done when

- [ ] The chat-only Agent has `tools=[]`.
- [ ] `read_document` is a plain async function.
- [ ] `read_document_tool` wraps that handler with a stable name.
- [ ] The deterministic run records one successful `ToolResultBlock`.
- [ ] The Agent returns the final text from the second transport call.

## Next failure

The assistant can read a document only when the user already knows its exact
path. If the user asks for “architecture documentation,” `read_document` is
the wrong shape. The next lesson adds search and gives the model enough
information to choose between two tools.

Continue with {doc}`tools-that-route`.
