# `axio-transport-codex`

ChatGPT OAuth transport using the Responses API.

It reads the stream through {doc}`responses`, which it shares with
`axio-transport-openai`. Its `ProviderEvent`s therefore say `provider="openai"`.

```{eval-rst}
.. autoclass:: axio_transport_codex.CodexTransport
   :members:
```

`CODEX_MODELS` is the {class}`~axio.models.ModelRegistry` this transport ships, and the
default `models` for a `CodexTransport`. Query it like any other registry -
`CODEX_MODELS.by_capability(Capability.vision)`, `CODEX_MODELS.ids()`.

## OAuth

```{eval-rst}
.. autofunction:: axio_transport_codex.oauth.run_oauth_flow
```

```{eval-rst}
.. autofunction:: axio_transport_codex.oauth.refresh_access_token
```
