# axio-tools-agents

Agent-to-agent tools for Axio.

## Tools

| Tool | Description |
|---|---|
| `list_peers(all_projects=False)` | List running local agents. By default only agents in the current project are shown. |
| `send_message(agent_id, message)` | Send a message to a running agent by global agent id. |
| `spawn_agent(task, inherit_context=False, name=None)` | Run an independent child agent. The child starts with an empty context unless `inherit_context=True`. |

Incoming peer messages are delivered by the host application, not by a receive
tool. In `axio-repl`, messages are queued into the dialog immediately after the
current response finishes.
