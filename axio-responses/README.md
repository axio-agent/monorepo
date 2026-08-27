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

<!-- name: test_readme_building_a_request -->
```python
from axio.blocks import TextBlock
from axio.messages import Message
from axio_responses import convert_messages, convert_tools

messages, system, tools = [Message(role="user", content=[TextBlock(text="hi")])], "be brief", []

instructions, items = convert_messages(messages, system)
payload = {
    "model": "gpt-5.6",
    "instructions": instructions,
    "input": items,
    "stream": True,
    "tools": convert_tools(tools),
}
assert payload["instructions"] == "be brief"
```

`convert_messages` returns the system prompt separately, because this API takes it as
`instructions` rather than as a message. Tool calls and their outputs become `function_call` and
`function_call_output` items beside the messages, not blocks inside them.

### Reading the stream

`Responses` is an `axio_sse.Reader`: one `@on(...)` method per event, dispatching on the payload's
own `type`. Its class body names only the events it interprets. The API publishes one event family
per tool it can run, so that set grows with the tools and not with the protocol; everything else is
forwarded through `unmatched()` rather than dropped.

<!-- name: test_readme_reading_the_stream -->
```python
from collections.abc import AsyncIterator

import aiohttp
from axio.events import StreamEvent
from axio_responses import Responses


async def stream(resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
    turn = Responses()
    async for made in turn.over(resp.content.iter_any(), until="[DONE]"):
        yield made
    yield turn.finished()
```

Events axio has no type for — the API's own hosted tools, its audio, its bookkeeping — travel as
`ProviderEvent` under the provider's own name rather than being dropped.

### Holding it against the schema

<!-- name: test_readme_names_are_published -->
```python
from axio_responses import Responses

PUBLISHED_EVENTS = {"response.output_text.delta", "response.completed", "response.refusal.delta"}

# Every name the reader claims is one the schema publishes. A typo is a handler that never runs.
assert Responses.names() >= PUBLISHED_EVENTS
```

`names()` answers what the reader claims, so a test can hold it against the union OpenAI publishes.
The check is `<=`, not `==`: the reader deliberately names fewer events than the API sends. Reading
with `strict=True` raises `UnknownEvent` on a name it does not claim, which is how a test fails on
the day OpenAI adds one.

## License

MIT
