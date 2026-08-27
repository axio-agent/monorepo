# `axio-responses`

The OpenAI Responses API as axio speaks it: request items in, `StreamEvent`s out.

Both halves live here rather than in a transport because two transports speak this API —
the public `/v1/responses` endpoint and the ChatGPT backend Codex uses. The package knows
nothing about HTTP and opens no connection.

<!-- name: test_responses_request_and_stream -->
```python
import json

from axio import Message, ReasoningBlock, TextBlock
from axio_responses import Responses, convert_messages
from axio_sse import Event

# A stored reasoning block goes back as its own item. Without the encrypted content and
# the id the model starts the next round blind, so a block that has neither is dropped.
instructions, items = convert_messages(
    [
        Message(role="user", content=[TextBlock(text="Why?")]),
        Message(
            role="assistant",
            content=[
                ReasoningBlock(text="", signature="gAAAAA...", id="rs_1"),
                TextBlock(text="Because."),
            ],
        ),
    ],
    "You are helpful.",
)
assert instructions == "You are helpful."
assert items[1] == {
    "type": "reasoning",
    "id": "rs_1",
    "encrypted_content": "gAAAAA...",
    "summary": [],
}

reader = Responses()


def read(**payload: object) -> list[object]:
    return list(reader.read(Event(data=json.dumps(payload))))


[delta] = read(type="response.output_text.delta", delta="Hello")
assert delta.delta == "Hello"

# A hosted tool the reader does not name, forwarded rather than dropped.
[forwarded] = read(type="response.web_search_call.searching", output_index=0)
assert (forwarded.provider, forwarded.kind) == ("openai", "response.web_search_call.searching")

# The API sends no event meaning "the turn is over"; the reader adds one up.
assert reader.finished().stop_reason.value == "end_turn"
```

## Building the request

```{eval-rst}
.. autofunction:: axio_responses.convert_messages
```

```{eval-rst}
.. autofunction:: axio_responses.convert_tools
```

```{eval-rst}
.. autofunction:: axio_responses.strip_title
```

`STOP_REASONS` maps a published response status, or the reason an incomplete response
gives, onto a {class}`~axio.types.StopReason`:

A reason that is not in the map ends the turn as `StopReason.error`, never as `end_turn`. The event
is named `response.incomplete`. The response did not finish. A truncation reason the API adds later
must not be stored and reported as a whole answer.

| Published | `StopReason` |
|---|---|
| `completed`, `end_turn`, `stop` | `end_turn` |
| `max_output_tokens`, `length` | `max_tokens` |
| `cancelled` | `cancelled` |
| `content_filter` | `refusal` |

It is the only one of the four transport maps that names `cancelled`. A reason outside
the map ends the run as an error rather than passing for a finished answer.

## Reading the stream

```{eval-rst}
.. autoclass:: axio_responses.Responses
   :members:
```

The reader names only what it interprets. Everything else — almost all of it the API
running a tool on its own side — is forwarded by `unmatched()` as
`ProviderEvent(provider="openai", kind=<the API's own name>)`. See {doc}`sse` for why.

## Payload shapes

One `axio_sse.Wire` per payload, read field by declared name and type. A shape with no
wire name of its own is only ever nested inside another.

| Shape | Wire name |
|---|---|
| `TextDeltaEvent` | `response.output_text.delta` |
| `ReasoningDeltaEvent` | `response.reasoning_summary_text.delta`, `response.reasoning_text.delta` |
| `RefusalDeltaEvent` | `response.refusal.delta` |
| `AnnotationAdded` | `response.output_text.annotation.added` |
| `ItemAdded` | `response.output_item.added` |
| `ItemDone` | `response.output_item.done` |
| `BlockDone` | `response.content_part.done` |
| `ArgumentsDelta` | `response.function_call_arguments.delta` |
| `ArgumentsDone` | `response.function_call_arguments.done` |
| `Created` | `response.created` |
| `Completed` | `response.completed` |
| `Incomplete` | `response.incomplete` |
| `Failed` | `response.failed` |
| `StreamFailure` | `error` |
| `ResponseObject` | nested |
| `ResponseUsage` | nested |
| `InputDetails`, `OutputDetails` | nested |
| `OutputItem` | nested |
| `Annotation`, `AnnotationSource` | nested |
| `IncompleteDetails`, `ResponseError` | nested |

```{eval-rst}
.. autoclass:: axio_responses.ResponseUsage
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.OutputItem
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.ResponseObject
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.Annotation
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.AnnotationSource
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.InputDetails
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.OutputDetails
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.IncompleteDetails
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.ResponseError
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.TextDeltaEvent
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.ReasoningDeltaEvent
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.RefusalDeltaEvent
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.AnnotationAdded
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.ItemAdded
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.BlockDone
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.ArgumentsDelta
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.ArgumentsDone
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.Created
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.Completed
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.Incomplete
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.Failed
   :members:
```

```{eval-rst}
.. autoclass:: axio_responses.StreamFailure
   :members:
```
