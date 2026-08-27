# `axio-transport-openai`

OpenAI and OpenAI-compatible completion transports.

`api` selects the endpoint. `OpenAITransport` defaults to `"responses"` and posts to
`/v1/responses`, reading it through {doc}`responses`. `NebiusTransport`,
`OpenRouterTransport` and `OpenAICompatibleTransport` set `"chat"` and post to
`/v1/chat/completions`. That endpoint refuses function tools beside any reasoning effort
other than `"none"` — see {ref}`the troubleshooting entry <tools-and-reasoning-400>`.

`extra_params` is folded into the request. Its `tools` are merged with the agent's
function declarations rather than replacing them, so adding a hosted tool does not take
away the calls the agent is there to dispatch.

```{eval-rst}
.. autoclass:: axio_transport_openai.OpenAITransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.nebius.NebiusTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.openrouter.OpenRouterTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.custom.OpenAICompatibleTransport
   :members:
```

## Realtime

```{eval-rst}
.. autoclass:: axio_transport_openai.OpenAIRealtimeTransport
   :members:
```

```{eval-rst}
.. autoclass:: axio_transport_openai.OpenAIRealtimeSession
   :members:
```
