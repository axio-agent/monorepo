# axio-tools-agents

Agent-to-agent tools for Axio.

## Tools

| Tool | Description |
|---|---|
| `list_peers(all_projects=False)` | List running local agents. By default only agents in the current project are shown. |
| `send_message(agent_id, message)` | Send a message to a running agent by global agent id. |
| `spawn_agent(task, inherit_context=False, name=None)` | Start an independent background child agent and return its global id. The child starts with an empty context unless `inherit_context=True`. |
| `interrupt_agent(agent_id, reason="")` | Cancel the agent's current response without stopping the agent. |
| `stop_agent(agent_id, reason="")` | Stop a background agent explicitly. |

Incoming peer messages are delivered by the host application, not by a receive
tool. In `axio-repl`, messages are queued into the dialog immediately after the
current response finishes. Background agents stay alive until `stop_agent` is
called or the host process exits.
