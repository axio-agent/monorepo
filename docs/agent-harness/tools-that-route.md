# Tools That Route

`read_document` works only when the user supplies an exact path. A request such
as “find the architecture documentation” has no valid input for that tool.

Adding `search_documents` creates a new problem. The model now needs a clear
contract for choosing one tool and constructing its arguments.

## Outcome

You will add a second tool with a distinct description and a bounded input
schema. The same Agent can then dispatch exact-path and discovery-shaped calls.

## Fast Track

1. Add `Annotated` and `Field` metadata to each handler parameter.
2. Write `search_documents(query, limit=5)` for public document summaries.
3. Give both `Tool` objects explicit, contrasting descriptions.
4. Register both tools on the existing Agent.
5. Verify each dispatch path with `StubTransport`.

{download}`Download the complete example <../../examples/tutorial/tools_that_route.py>`.

## Hands-on delta

### 1. See the ambiguous contract

Descriptions such as “read documents” and “find documents” overlap. They do not
tell a model which input shape belongs to each action.

| Request shape | Exact lookup | Search |
|---|---:|---:|
| Contains `README.md` | yes | no |
| Describes documents without a path | no | yes |

This distinction must appear in both the descriptions and argument schemas.

### 2. Separate the argument shapes

Python types define the basic schema. `Field` adds model-facing descriptions,
defaults, and numeric bounds without creating a second schema definition.

Keep `DOCUMENTS` from the previous lesson. Replace the handlers with this
version:

```{literalinclude} ../../examples/tutorial/tools_that_route.py
:language: python
:caption: examples/tutorial/tools_that_route.py
:start-after: "# [docs:start-tools-routing-handlers]"
:end-before: "# [docs:end-tools-routing-handlers]"
```

`Tool.input_schema` is a JSON-compatible copy. Axio transports send the
read-only tool schema to the model with the description and name.

The bounds also affect execution. Axio validates model-produced values before
the handler runs.

See {doc}`../concepts/tools` for supported annotations and validation order.

### 3. Offer both tools

Give each `Tool` a contrasting description, then register both definitions on
the same Agent:

```{literalinclude} ../../examples/tutorial/tools_that_route.py
:language: python
:caption: examples/tutorial/tools_that_route.py
:start-after: "# [docs:start-tools-routing-definitions]"
:end-before: "# [docs:end-tools-routing-definitions]"
```

Change only the `tools` list in `build_agent`. The assertions confirm that the
Agent exposes both tools in the intended order:

```{literalinclude} ../../examples/tutorial/tools_that_route.py
:language: python
:caption: examples/tutorial/tools_that_route.py
:start-after: "# [docs:start-tools-routing-agent]"
:end-before: "# [docs:end-tools-routing-agent]"
```

The Agent passes both definitions to the transport. A live model chooses a
tool from their names, descriptions, and schemas. Axio resolves the returned
name and dispatches the validated arguments.

Keep descriptions specific. A description should state the action, its useful
input shape, and the nearby tool that handles a different request.

## Try It

Use canned model responses to test dispatch without model variance. This check
proves that Axio resolves each requested name to the correct handler. It does
not claim to measure a live model's selection quality.

Run `uv run python examples/tutorial/tools_that_route.py` from the repository
root.

```{literalinclude} ../../examples/tutorial/tools_that_route.py
:language: python
:caption: examples/tutorial/tools_that_route.py
:start-after: "# [docs:start-tools-routing-dispatch]"
:end-before: "# [docs:end-tools-routing-dispatch]"
```

The missing `limit` becomes `5` before `search_documents` runs. This verifies
default injection as well as name-based dispatch.

Use the same prompt set with your provider transport as an exploratory check:
one exact path, one discovery request, and one request near the boundary. Do
not use model output as the deterministic contract check.

## Done when

- [ ] `read_document` requires an exact `path`.
- [ ] `search_documents` accepts a described query and a bounded limit.
- [ ] The descriptions distinguish exact lookup from discovery.
- [ ] Both tools are registered on the same Agent.
- [ ] The stubbed calls reach different handlers by tool name.
- [ ] Axio injects the default search limit before execution.

## Next failure

The schema says that `path` is a string. It cannot say which documents the
current user may read. A model can still request private file `.env`, even when
the system prompt tells it not to.

The next lesson moves that decision into executable policy.

Continue with {doc}`guard-tool-calls`.
