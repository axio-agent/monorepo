# axio-responses

[![PyPI](https://img.shields.io/pypi/v/axio-responses)](https://pypi.org/project/axio-responses/)
[![Python](https://img.shields.io/pypi/pyversions/axio-responses)](https://pypi.org/project/axio-responses/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The OpenAI Responses API as [axio](https://github.com/mosquito/axio-agent) speaks it: request items
in, `StreamEvent`s out.

Both halves live here rather than in a transport because two transports speak this API — the public
`/v1/responses` endpoint and the ChatGPT backend Codex uses. It knows nothing about HTTP and opens
no connection.

## Installation

```bash
pip install axio-responses
```

## Usage

### Building the request

```python
from axio_responses import convert_messages, convert_tools

instructions, items = convert_messages(messages, system)
payload = {
    "model": "gpt-5.6",
    "instructions": instructions,
    "input": items,
    "stream": True,
    "tools": convert_tools(tools),
}
```

`convert_messages` returns the system prompt separately, because this API takes it as
`instructions` rather than as a message. Tool calls and their outputs become `function_call` and
`function_call_output` items beside the messages, not blocks inside them.

### Reading the stream

`Responses` is an `axio_sse.Reader`: one `@on(...)` method per event, dispatching on the payload's
own `type`. Its class body is the whole published `ResponseStreamEvent` union, so an event it does
not name is one the API added after it was written.

```python
from axio_responses import Responses
from axio_sse import events

turn = Responses()
async for made in turn.over(resp.content.iter_any(), until="[DONE]"):
    yield made
yield turn.finished()
```

Events axio has no type for — the API's own hosted tools, its audio, its bookkeeping — travel as
`ProviderEvent` under the provider's own name rather than being dropped.

### Holding it against the schema

```python
assert Responses.names() == PUBLISHED_EVENTS
```

`names()` answers what the reader claims, so a test can hold it against the union OpenAI publishes.
Reading with `strict=True` raises `UnknownEvent` on anything outside it.

## License

MIT
