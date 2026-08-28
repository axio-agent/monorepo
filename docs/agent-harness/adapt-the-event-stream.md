# Adapt the Event Stream to a Product Surface

The terminal renderer accepts Python dataclasses. A browser does not. Passing a
`SessionEndEvent` directly to `send_json()` fails on enums, exceptions, and
binary output.

Do not change the agent to solve a delivery problem. Add one adapter at the
edge of the harness.

## Outcome

The same `CloudHarness.stream_turn()` output can drive a terminal, WebSocket,
or HTTP stream through a versioned JSON envelope.

## Fast Track

1. Walk each event dataclass without copying unknown field values.
2. Normalize enums, exceptions, bytes, and nested containers.
3. Add the concrete event class name and a wire-format version.
4. Bind authenticated identity to the internal session ID.
5. Propagate disconnect cancellation to the active `AgentStream`.

{download}`Download the complete example <../../examples/tutorial/adapt_the_event_stream.py>`.

## Hands-on delta

### 1. Build the adapter

Axio intentionally does not define a public web protocol. Applications differ
in authentication, reconnect behavior, binary transport, and compatibility
requirements. This small adapter is one explicit application contract:

```{literalinclude} ../../examples/tutorial/adapt_the_event_stream.py
:language: python
:caption: examples/tutorial/adapt_the_event_stream.py
:start-after: "# [docs:start-adapt-event-adapter]"
:end-before: "# [docs:end-adapt-event-adapter]"
```

The field walk is intentionally shallow before `json_value()` recurses. In
contrast, `dataclasses.asdict()` deep-copies unknown values and can fail on a
provider exception that contains a lock, stream, or client object.

`Message.to_dict()` serializes persisted conversation messages. It is not an
event serializer. Keep message storage and the public stream protocol as two
separate contracts.

Do not send `str(exception)` to an untrusted client. Provider responses,
credentials, paths, or query data can appear in exception text. Record the
full exception only in access-controlled server logs, with the application's
normal secret-redaction policy.

### 2. Keep the endpoint small

The endpoint receives an authenticated identity, selects its server-issued
session ID, and forwards events:

```{literalinclude} ../../examples/tutorial/adapt_the_event_stream.py
:language: python
:caption: examples/tutorial/adapt_the_event_stream.py
:start-after: "# [docs:start-adapt-stream-endpoint]"
:end-before: "# [docs:end-adapt-stream-endpoint]"
```

`aclosing()` closes the async generator when `send_json()` fails or the request
is cancelled. This lets `CloudHarness.stream_turn()` close `AgentStream` and
release the per-session turn lock.

The authentication layer must issue and retain `session_id`. The example uses
a UUID hex value, which is also safe in Docker resource names. Never accept an
arbitrary client string as proof that the caller owns a context or sandbox.

### 3. Preserve meaning across surfaces

Different surfaces render the same event differently:

| Event | Terminal | Browser | Service log |
|---|---|---|---|
| `TextDelta` | append text | append to message | usually omit |
| `ToolUseStart` | show tool name | open tool card | record start time |
| `ToolResult` | show status | complete tool card | record duration and error flag |
| `Error` | show failure | show recoverable error | record exception metadata |
| `SessionEndEvent` | show usage | close turn | record stop reason and totals |

The adapter can omit internal fields or split binary data into another channel.
Once clients depend on that choice, change the `version` when compatibility
breaks.

## Try It

Run `uv run python examples/tutorial/adapt_the_event_stream.py` from the
repository root. It checks enum, exception, and binary serialization without a
server or model API.

Then send those envelopes through your selected HTTP or WebSocket framework.
Disconnect during a tool call and verify that the session lock is released for
the next request.

## Done when

- [ ] Every Axio event becomes JSON without provider-specific logic.
- [ ] Enum, exception, and binary values have explicit wire representations.
- [ ] The envelope has a compatibility version.
- [ ] Authentication, not client input, selects the internal session.

## Next failure

The product surface now receives stable events, but a later refactor can still
break dispatch, ordering, or cleanup silently. The final lesson turns the
complete harness boundary into a deterministic contract check.

Continue with {doc}`test-the-harness`.
