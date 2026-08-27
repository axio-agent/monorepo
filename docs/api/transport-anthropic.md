# `axio-transport-anthropic`

Anthropic Claude completion transport for the direct API and Vertex AI.

It reads the stream with an {doc}`axio-sse <sse>` `Reader` keyed on the format's own
`event:` field. That reader is `axio_transport_anthropic.Messages`, with one `@on(...)`
method per published event. Its `unmatched()` forwards anything else as
`ProviderEvent(provider="anthropic")`.

Its usage counts are converted into the axio rule before they leave the transport. The API
reports `input_tokens` as only what follows the last cache breakpoint. The cache read and
cache write counts are therefore added back to make an inclusive total.

```{eval-rst}
.. autoclass:: axio_transport_anthropic.AnthropicTransport
   :members:
```
