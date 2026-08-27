# Axio

[![GitHub](https://img.shields.io/badge/github-mosquito%2Faxio--agent-181717?logo=github&logoColor=white)](https://github.com/mosquito/axio-agent)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/mosquito/axio-agent/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/axio?label=axio&logo=pypi&logoColor=white)](https://pypi.org/project/axio/)

Axio is a small asynchronous foundation for building LLM agent harnesses in
Python. It provides the agent loop and explicit extension points for model
transports, tools, context storage, permissions, and streamed events.

Axio is a library, not an application framework. Your code keeps control of
sessions, input and output, resource lifetimes, persistence, observability, and
deployment.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Build an Agent Harness
:link: quick-start
:link-type: doc

The canonical tutorial: build a REPL, make it persistent, then turn it into a
multi-session cloud harness with a Docker sandbox per session.
:::

:::{grid-item-card} Core Concepts
:link: concepts/index
:link-type: doc

Reference documentation for the agent, transports, tools, events, and context.
:::

:::{grid-item-card} Extension Guides
:link: guides/index
:link-type: doc

Write transports, tools, guards, and context stores.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Public classes, events, and methods.
:::

::::

## The boundary Axio provides

~~~{mermaid}
flowchart TD
    H[Your harness] -->|prompt + context| A[Agent loop]
    A -->|one request| T[Transport]
    T <--> L[LLM provider]
    A -->|validated call| X[Async tools]
    A <--> C[Context store]
    A -->|typed events| H
~~~

The transport normalizes one provider request into typed events. The agent
consumes those events, executes requested tools, appends results to context,
and calls the transport again. Your harness consumes the same event stream and
turns it into application behavior.

Start with {doc}`quick-start`. It develops this entire flow as one program:

1. define a tool contract;
2. stream one agent turn in a terminal REPL;
3. replace memory context with SQLite;
4. separate sessions and serialize turns;
5. bind a distinct Docker sandbox to every session;
6. adapt typed events to an HTTP or WebSocket protocol.

## Design principles

Explicit composition
: Create an Agent from a transport and tools, then supply one context store per
  session. Dependencies remain visible in ordinary Python.

Provider isolation
: Provider-specific authentication, message encoding, streaming, and stop
  reasons stay behind CompletionTransport.

Typed streaming
: Text, reasoning, refusals, citations, tool requests, tool results, usage, and
  failures are observable while the turn runs. A payload axio has no type for is
  forwarded rather than dropped, so nothing a provider sends disappears on the way
  through.

Application-owned state
: Context is supplied by the caller. The same agent definition can serve many
  isolated sessions.

Plain async tools
: Tool handlers are async functions. Annotations become the input schema.
  Guards, runtime context, and concurrency limits are explicit.

```{toctree}
:maxdepth: 2
:hidden:

quick-start
concepts/index
guides/index
packages
api/index
troubleshooting
glossary
```
