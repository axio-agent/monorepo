# Troubleshooting

Solutions for common issues when working with Axio.

## Installation

### `uv: command not found`

Install [uv](https://docs.astral.sh/uv/) first:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

Or install Axio with pip: `pip install axio`.

## API Keys

### `StreamError: ... 401` on the first turn

No transport checks for a missing key. Each one defaults `api_key` to the environment
variable below, or to the empty string. An empty key reaches the provider and comes back
as a 401. The transport wraps that 401 in `axio.exceptions.StreamError`. Set the variable
for the transport you constructed:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google (Gemini developer API; the realtime transport also accepts GOOGLE_API_KEY)
export GEMINI_API_KEY="..."

# Nebius
export NEBIUS_API_KEY="..."

# OpenRouter
export OPENROUTER_API_KEY="..."
```

`axio-transport-codex` uses OAuth rather than a key. Vertex AI uses Google Application
Default Credentials.


### `Invalid API key` / `Authentication error`

- Verify the key is correct - no extra spaces or quotes in the env var
- Check the key has not expired
- For OpenAI: make sure you have active billing
- For Anthropic: ensure the key has the right permissions

## Transport Connection

### `Connection refused` / `Failed to connect`

- Check your internet connection
- Verify the API endpoint is correct (especially for custom endpoints)
- Some corporate networks block external APIs - try using a VPN
- For OpenAI-compatible APIs: verify the base URL in your transport config

### `Timeout error`

- The API may be slow or experiencing high load
- Try again in a few moments
- If persistent, increase the timeout in your transport settings

### `SSL certificate error`

- Update your Python version - newer versions have updated CA certificates
- On macOS: run `/Applications/Python\ 3.x/Install\ Certificates.command`
- On Linux: update ca-certificates: `sudo apt update && sudo apt install ca-certificates`

## Provider Requests

(tools-and-reasoning-400)=
### 400: function tools and `reasoning_effort` cannot be combined

`/v1/chat/completions` refuses function tools beside any reasoning effort other than
`"none"`. A reasoning model reasons by default. A request that carries tools therefore
fails with a 400 naming a parameter you never sent.

`OpenAITransport.build_chat_payload` heads that off. When the model has
`Capability.reasoning` and the request carries tools, it sets `reasoning_effort: "none"`
and warns. The second symptom is therefore the warning rather than the 400. You then pay
for a reasoning model that is asked not to reason.

This affects the transports that speak that endpoint. `OpenAICompatibleTransport`,
`NebiusTransport` and `OpenRouterTransport` all set `api="chat"`. `OpenAITransport`
defaults to `api="responses"`. `/v1/responses` takes both together.

- On OpenAI itself, leave `api` at its default.
- On a compatible server that implements `/v1/responses`, pass `api="responses"`.
- On one that does not, decide the trade explicitly. Either pass
  `extra_params={"reasoning_effort": "low"}` - which suppresses the override - and drop
  the tools, or keep the tools and accept `"none"`.

### 400 from Anthropic on a replayed thinking block

With extended thinking on, a turn that thought and then called a tool is refused unless
its thinking comes back with the signature the API issued for it. The transport replays a
`ReasoningBlock` only when `signature` is set, or when `redacted` is. An unsigned block
is dropped rather than sent, because there is nothing to prove it is the model's.

The failure lands one turn late. The request that fails is the one *after* the turn whose
signature was lost. Suspect a context store that round-trips blocks through
`to_dict`/`from_dict` and drops the field. Never inspect, decode, re-encode or truncate a
signature. It is opaque, and a changed one is as bad as a missing one.

### `MISSING_THOUGHT_SIGNATURE` from Gemini

This is the same failure on Google's side. Google publishes it as a
`finishReason`. The transport maps it to `StopReason.error`, which ends the run.

Two replay paths, easy to confuse:

- A thought that had text goes back as a part with `thought: true` and its
  `thoughtSignature`.
- A signature that arrived on a *function-call* part goes back on that part, not on a
  thought part. Gemini puts the proof on the part it signed. Sent as a text-less thought
  part, the call it belongs to comes back `MISSING_THOUGHT_SIGNATURE`.

Parallel calls consume unplaced signatures in arrival order, so the store has to preserve
the order of blocks within the assistant turn.

### A turn that reads as an empty answer

`run()` returning `""` with no exception, on a prompt the model declined. The decline
arrives as a `Refusal` event, not as `TextDelta`. A renderer or a `get_final_text()` that
only collects text prints nothing.

`AgentStream.get_final_text()` - and so `Agent.run()` - now collects `Refusal.text`, so
this is fixed for a transport that sends the text. Two cases still produce an empty
string. Both are visible in the stop reason:

- Gemini blocking a *prompt*, which sends a `Refusal` with `blocked_input=True`, a
  `category`, and no text at all. Nothing was generated.
- A Gemini candidate finishing with `SAFETY`, `RECITATION` and the rest, which maps to
  `StopReason.refusal` while emitting no `Refusal` event.

So read the stop reason rather than the text. A refusal is terminal and deliberately not
an error. Reported as one, it leaves a caller unable to tell a decline from a broken
connection. The caller then retries something that can never work.

<!-- name: test_troubleshooting_declined_turn -->
```python
import asyncio

from axio import Agent, MemoryContextStore, Refusal, StopReason, Usage
from axio.events import IterationEnd, SessionEndEvent
from axio.testing import StubTransport

transport = StubTransport([[
    Refusal(index=0, category="safety", blocked_input=True),
    IterationEnd(0, StopReason.refusal, Usage(10, 0)),
]])
agent = Agent(system="You are helpful.", transport=transport)


async def diagnose() -> tuple[str, StopReason]:
    text: list[str] = []
    stop = StopReason.error
    stream = agent.run_stream("...", MemoryContextStore())
    async for event in stream:
        match event:
            case Refusal(text=refused):
                text.append(refused)
            case SessionEndEvent(stop_reason=stop_reason):
                stop = stop_reason
    return "".join(text), stop


answer, stop = asyncio.run(diagnose())
assert answer == ""
assert stop is StopReason.refusal
```

### A long turn that stops mid-answer

A reasoning event large enough to blow past `aiohttp`'s 131072-byte line limit. Reading
the stream by lines raises `LineTooLong`. `LineTooLong` is not a `ClientError`, so it
escapes the retry paths. The turn ends with no answer.

`axio-sse` exists for this. It takes chunks cut anywhere and never lines, so no line
length is a limit. Every transport in this repository reads through it, from
`resp.content.iter_any()`. A transport of your own that reads the response line by line
has the bug. Feed `axio_sse.payloads()` or `Reader.over()` the chunks instead.

The mirror-image mistake is a chunk iterator that strips line terminators - `httpx`'s
`aiter_lines()`, for one. The decoder needs them to know an event ended. Nothing
therefore dispatches at all, and the turn is silent rather than truncated.

## Tools

### `Tool not found`

Tools must be passed explicitly to the Agent:

```python
from my_tool import my_tool

agent = Agent(
    system="You are helpful.",
    tools=[my_tool],  # explicitly pass
    transport=transport,
)
```

### `Tool execution failed`

Check the error message:

- **Timeout**: the tool took too long - consider async optimization
- **Permission denied**: a guard blocked the tool - see "Permission guards" below
- **Import error**: check the tool handler's dependencies are installed

### `Tool returned empty result`

- Verify the tool logic is correct
- Check logs for exceptions during execution
- Add debug output in your tool handler to see what's happening

## Permission Guards

### `Permission denied` for every tool call

Guards are blocking all tool calls. Check:

1. Check which guards are attached to your agent in your configuration
2. For `PathGuard`: it asks before a tool touches a path. A yes allows the parent
   directory, and a no remembers the path. One denial therefore keeps denying for the
   rest of the session. Its default prompt calls `input()`. Pass a `prompt_fn` where
   there is no terminal to read from, or every call blocks
3. For `LLMGuard`: it puts the decision to an agent, so that agent's transport needs a
   working API key of its own

Both ship in `axio-tui-guards`, and are registered under the `axio.guards` entry-point
group. See {doc}`packages`.

## Context & Storage

### `Database is locked` (SQLite)

Multiple processes are accessing the same SQLite database. Solutions:

- Use WAL mode (enabled by default in Axio)
- Ensure you're using a single process
- Increase busy timeout in connection string

### `Session not found`

- Check the `session_id` is correct
- For SQLite: verify the database file exists and has data
- The session may have been deleted or expired

## Development

### `Module not found` when importing axio

Ensure you're in the right environment:

```bash
uv run --directory axio python -c "import axio; print(axio.__file__)"
```

`uv run --directory <package>` is how every command in this workspace is run. There is no
`uv shell`. To get an interactive environment instead, create one with `uv venv` and
activate it.

### Type checking errors

Axio uses strict typing. Every package sets `strict = true`. Each is checked from its own
directory. `mypy axio/` from the repository root points at the distribution directory
rather than at the sources under `axio/src/axio`:

```bash
make typing                      # every package
uv run --directory axio mypy .   # one package
```

### Tests failing

Run tests with verbose output:

```bash
uv run --directory axio pytest -v
```

Every package sets `testpaths = ["tests", "README.md"]`, so its README's Python blocks are
executed too. The documentation has its own suite: `make test-docs`.

Check if the failure is in your code or the framework:

- If in framework: open an issue on GitHub
- If in your code: verify against the test examples in `docs/` and `axio/tests/`

## Getting Help

If your issue isn't listed here:

1. Search [GitHub issues](https://github.com/mosquito/axio-agent/issues)
2. Open a new issue with:
   - Python version
   - Axio version (`python -c "import importlib.metadata; print(importlib.metadata.version('axio'))"`)
   - Full error traceback
   - Minimal reproduction code
