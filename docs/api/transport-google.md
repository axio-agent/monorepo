# `axio-transport-google`

Google Gemini completion transports for the Developer API and Vertex AI.

Gemini's stream carries no per-event discriminator, so this transport reads it with
{doc}`axio_sse.payloads() <sse>` and `Wire` payload shapes rather than with a `Reader`.
Anything it has no typed event for — grounding and citation metadata, executable code,
parts added later — is emitted as `ProviderEvent(provider="google")`.

Its usage counts are converted into the axio rule. `toolUsePromptTokenCount` is not
inside `promptTokenCount`, and `thoughtsTokenCount` is not inside
`candidatesTokenCount`, so both are added. `cachedContentTokenCount` already is inside
its total, and is not added.

```{eval-rst}
.. autoclass:: axio_transport_google.GoogleTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_google.VertexAITransport
   :members:
```

## Realtime

```{eval-rst}
.. autoclass:: axio_transport_google.realtime.GeminiLiveTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_google.realtime.VertexLiveTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_google.realtime.GeminiLiveSession
   :members:
```

## Media tools

```{eval-rst}
.. autofunction:: axio_transport_google.tools.generate_image
```

```{eval-rst}
.. autofunction:: axio_transport_google.tools.generate_video
```

## Thinking levels

```{eval-rst}
.. autofunction:: axio_transport_google.valid_thinking_levels
```

A `thinking_level` the model does not accept is replaced by that family's highest level,
so an invented value buys maximum thinking rather than none.
