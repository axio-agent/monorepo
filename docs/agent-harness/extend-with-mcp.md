# Extend the Tool Set without Changing the Loop

Your harness now isolates users, conversations, and execution. Then someone
asks for a documentation server, a source index, and browser automation.
Writing a custom handler for every service would couple the harness to every
integration.

MCP changes the source of the tools, not the agent loop. Axio discovers the
server definitions and wraps each one as an ordinary {class}`~axio.Tool`.

## Outcome

The harness loads tools from one MCP server, keeps its connection alive, and
closes that connection during shutdown.

## Fast Track

1. Install `axio-tools-mcp`.
2. Describe each server with `MCPServerConfig`.
3. Add the discovered tools to the agent's existing tool list.
4. Keep every returned session open until its tools are no longer used.

{download}`Download the complete example <../../examples/tutorial/extend_with_mcp.py>`.

## Hands-on delta

### 1. Add one server

Run `uv add axio-tools-mcp`.

`load_mcp_tools()` returns two related values. The tools belong to the agent;
the sessions belong to the harness lifecycle.

```{literalinclude} ../../examples/tutorial/extend_with_mcp.py
:language: python
:caption: examples/tutorial/extend_with_mcp.py
:start-after: "# [docs:start-mcp-load-tools]"
:end-before: "# [docs:end-mcp-load-tools]"
```

For a deterministic check, substitute an in-process server. Keep the same
lifecycle when the harness uses a stdio or HTTP MCP connection.

For a real server, replace the example executable with the command supplied by
that server. An HTTP configuration uses `url=` and optional `headers=` instead
of `command=` and `args=`.

### 2. Choose the connection scope

Load a service-wide MCP connection at application startup only when every
session may use the same server identity. Add those tools to the prototype
agent before constructing `CloudHarness`, then close the harness before its
shared MCP sessions:

```{literalinclude} ../../examples/tutorial/extend_with_mcp.py
:language: python
:caption: examples/tutorial/extend_with_mcp.py
:start-after: "# [docs:start-mcp-shared-scope]"
:end-before: "# [docs:end-mcp-shared-scope]"
```

If credentials or server state belong to one user, create that connection
inside the corresponding `CloudSession`. Register every `session.close`
callback on the same `AsyncExitStack` as its context and sandbox. Never reuse
one tenant's authenticated MCP tools in another tenant's agent copy.

### 3. Keep names inside the server boundary

Axio prefixes each discovered name with `{server}__`. A server named `docs`
that publishes `search` therefore becomes `docs__search`. The prefix prevents
collisions when two servers publish the same local name.

The model sees a normal tool schema and description. `Agent`, `ToolResult`,
guards, context storage, and event rendering do not need an MCP branch.

### 4. Keep execution location explicit

A stdio MCP server runs beside the harness unless you start it elsewhere. Its
tools do not automatically run in the session's Docker sandbox. Use one of
these arrangements when isolation matters:

- start the server inside the session sandbox;
- connect to a remote service that enforces its own tenant boundary;
- apply guards and downstream authorization before sensitive operations.

MCP standardizes discovery and invocation. It does not supply authorization or
resource isolation.

## Try It

Run `uv run python examples/tutorial/extend_with_mcp.py` from the repository
root. It completes without an MCP server or model API.

Then connect one real server and inspect the loaded names before creating the
agent. Confirm that shutdown closes every returned session, including the path
where agent construction or execution fails.

For server-specific configuration, failure handling, and multiple servers,
use {doc}`../guides/mcp-tools`.

## Done when

- [ ] Discovered MCP tools appear in the same list as local Axio tools.
- [ ] Every tool name includes its server prefix.
- [ ] MCP sessions remain open while their tools can run.
- [ ] Shutdown closes sessions after active turns finish.
- [ ] The deployment design states where each MCP server executes.

## Next failure

The harness can now acquire capabilities at runtime. The next problem is the
opposite boundary: typed Python events cannot cross a WebSocket unchanged.

Next: {doc}`adapt-the-event-stream`.
