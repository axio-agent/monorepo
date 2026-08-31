# How-To Guides

Practical, step-by-step guides for extending Axio with your own components.

Extending Axio means writing one of four things: a tool, a guard, a context store, or a transport.
{doc}`writing-transports` is the longest of them, because a transport is the only place that knows a
provider exists. It reads the provider's stream through [`axio-sse`](../api/sse.md), converts its
token counts, maps its stop reasons, and replays its reasoning.

```{toctree}
:maxdepth: 1

axio-repl
google-transport
realtime-audio
mcp-tools
multimodal
writing-tools
writing-transports
writing-guards
writing-context-stores
docker-sandbox
testing
cookbook
best-practices
agent-swarm
gas-town
```
