# Isolate Execution

{doc}`Compact Long Context <compact-long-context>` made the project conversation
durable and bounded. The risk changes when you add file, Python, or shell tools
to inspect a public repository.

Handlers from `axio-tools-local` run as the harness process. They can access
every file, credential, process, and network destination available to that
operating-system user. A system prompt does not reduce those permissions.

## Outcome

Run execution tools inside a Docker container. Keep the agent, transport,
repository tools, context, streaming renderer, and compaction behavior unchanged.

## Fast Track

1. Install `axio-tools-docker`.
2. Define explicit network and resource limits.
3. Enter `DockerSandbox` before reading `sandbox.tools`.
4. Copy the agent and replace only its execution-tool list.
5. Keep the sandbox open until every agent turn and tool call finishes.

Install it with `uv add axio-tools-docker`.

{download}`Download the complete example <../../examples/tutorial/isolate_execution.py>`.

## Hands-on delta

### 1. Define the execution boundary

Constructing this object does not contact Docker. The daemon is contacted when
the async context manager starts.

```{literalinclude} ../../examples/tutorial/isolate_execution.py
:language: python
:caption: examples/tutorial/isolate_execution.py
:start-after: "# [docs:start-isolate-sandbox-factory]"
:end-before: "# [docs:end-isolate-sandbox-factory]"
```

`network=False` disables container networking beyond loopback. Memory, CPU,
file descriptor, process, and writable-filesystem limits bound common failure
modes.

See {doc}`Docker Sandbox <../guides/docker-sandbox>` for every option, including
named volumes, users, capabilities, and read-only filesystems.

### 2. Add tools at the isolation boundary

Keep the guarded repository tools from earlier lessons. Create the execution tools
only after the sandbox starts, then add both groups to a session-specific agent
copy:

```{literalinclude} ../../examples/tutorial/isolate_execution.py
:language: python
:caption: examples/tutorial/isolate_execution.py
:start-after: "# [docs:start-isolate-run-turn]"
:end-before: "# [docs:end-isolate-run-turn]"
```

`agent.tools` still contains `read_document`, `search_documents`, and their guards.
The new file and command tools exist only behind the container boundary. The
adapter from {doc}`Compact Long Context <compact-long-context>` also bounds
their text before Axio stores it in conversation history.

The Axio loop does not need a Docker branch. It receives ordinary `Tool`
objects and dispatches them through the same validation and event path.

### 3. Keep the async lifetime intact

`DockerSandbox` creates or attaches to its container in `__aenter__`. Its
`tools` property is valid only after entry and before exit.

Each returned tool has `Tool.context` set to that sandbox. Before calling a
handler, Axio binds the value to `CONTEXT`. The Docker shell handler can then
resolve the current container without a global variable:

```{literalinclude} ../../examples/tutorial/isolate_execution.py
:language: python
:caption: examples/tutorial/isolate_execution.py
:start-after: "# [docs:start-isolate-conceptual-shell]"
:end-before: "# [docs:end-isolate-conceptual-shell]"
```

Do not cache `sandbox.tools` globally. Do not return `isolated_agent` from the
context manager and use it later. Both mistakes leave tools bound to a closed
sandbox.

## Try It

This check requires a running Docker daemon. The container is temporary, and
the context manager removes it on exit.

```{literalinclude} ../../examples/tutorial/isolate_execution.py
:language: python
:caption: examples/tutorial/isolate_execution.py
:start-after: "# [docs:start-isolate-docker-check]"
:end-before: "# [docs:end-isolate-docker-check]"
```

The result proves that the bound shell reached the active container. It does
not prove that the container can resist every hostile workload.

Run `uv run python examples/tutorial/isolate_execution.py --docker` from the
repository root to execute this optional integration check.

## Treat Docker as one control

A container reduces the execution scope, but it is not a complete security
boundary by itself. Isolation also depends on the host kernel, Docker daemon,
image, mounts, capabilities, devices, and service configuration.

Never mount the Docker socket inside an untrusted container. Add host bind
mounts, devices, privileged mode, or network access only for a specific need.
Apply authentication, authorization, quotas, monitoring, and host patching as
separate controls.

## Done when

- [ ] The service agent has no local file or shell tools.
- [ ] Docker tools are created and used inside one `async with` block.
- [ ] `network=False` and resource limits are explicit.
- [ ] Existing repository guards, context, streaming, and compaction still work.
- [ ] The optional check returns `sandbox-ok` when Docker is available.

## Next failure

One context and one sandbox now work for one user. Sharing that pair between
users still mixes their messages and files. Continue with
{doc}`serve-many-sessions`, which gives each stable session its own history,
execution environment, and turn lock.
