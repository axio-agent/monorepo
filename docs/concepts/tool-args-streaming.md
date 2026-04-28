# Streaming Tool Arguments

Tool arguments arrive incrementally as JSON fragments via `ToolInputDelta` events.
The [`ToolArgStream`][axio.tool_args.ToolArgStream] parser converts these
fragments into structured [`ToolField*`][axio.events.ToolFieldStart] events,
enabling real-time display of tool inputs as they stream in.

## Why streaming args?

When an LLM calls a tool, it generates the JSON arguments character-by-character.
Without streaming, you must wait for the complete JSON object before you can
display or process it. This creates latency - the user sees nothing until the
entire tool call is ready.

Streaming args enable:
- **Real-time display**: Show file content as it streams, similar to Claude Code's
  Edit tool live diffs.
- **Progressive validation**: Detect malformed JSON early.
- **Lower latency**: Start rendering the first fields before the last ones arrive.

## ToolArgStream API

`ToolArgStream` is a zero-dependency, O(1)-per-character streaming JSON parser.
It processes `ToolInputDelta.partial_json` chunks and emits `ToolFieldStart`,
`ToolFieldDelta`, and `ToolFieldEnd` events.

### Constructor

```python
from axio.tool_args import ToolArgStream

parser = ToolArgStream("call_123", index=0)
```

`tool_use_id`
: The unique identifier for this tool invocation (matches `ToolUseStart.tool_use_id`).

`index`
: Event index for ordering (defaults to 0).

### Methods

`feed(chunk: str) -> list[ToolFieldEvent]`
: Process a partial JSON chunk and return any field events produced.
  May return zero, one, or multiple events depending on the chunk content.

`current_key -> str`
: Property returning the field currently being parsed, or `""` if between fields.

<!-- name: test_tool_arg_stream_api -->
```python
from axio.tool_args import ToolArgStream
from axio.events import ToolFieldStart, ToolFieldDelta, ToolFieldEnd

stream = ToolArgStream("call_1", index=0)

# Feed first chunk
events = stream.feed('{"path":"/tmp/f')
assert len(events) == 2
assert isinstance(events[0], ToolFieldStart)
assert events[0].key == "path"
assert isinstance(events[1], ToolFieldDelta)
assert events[1].text == "/tmp/f"

# Feed second chunk
events = stream.feed('oo.py"}')
assert events[0].text == "oo.py"
assert isinstance(events[1], ToolFieldEnd)
assert stream.current_key == "path"  # retains last completed key
```

## Field-level events

The parser emits three event types for each top-level field:

### ToolFieldStart

Emitted when a new top-level field name is identified (after the closing quote
of the key, before any value content).

```python
@dataclass(frozen=True, slots=True)
class ToolFieldStart:
    index: int
    tool_use_id: ToolCallID
    key: str
```

### ToolFieldDelta

Emitted with chunks of the field's value. For **string values**, escape
sequences are resolved and surrounding quotes are stripped. For all other
types (numbers, booleans, objects, arrays), the raw JSON fragment is emitted.

```python
@dataclass(frozen=True, slots=True)
class ToolFieldDelta:
    index: int
    tool_use_id: ToolCallID
    key: str
    text: str
```

### ToolFieldEnd

Emitted when the field's value is complete.

```python
@dataclass(frozen=True, slots=True)
class ToolFieldEnd:
    index: int
    tool_use_id: ToolCallID
    key: str
```

## Streaming vs complete JSON

| Aspect | Complete JSON | Streaming Args |
|--------|--------------|----------------|
| **When available** | After full tool call | Incrementally, as characters arrive |
| **Parser state** | One-shot parse | Maintains state across chunks |
| **Escape handling** | Single `json.loads()` | Incremental escape resolution |
| **Display latency** | Wait for all | Show first chunk immediately |
| **Error detection** | At end | Potentially mid-stream |

### Example: String value

For `{"path": "/tmp/file"}` arriving as chunks `'{"pat'`, `'h":"/tm'`, `'p/file"}'`:

1. Chunk `'{"pat'`: No events yet (incomplete key)
2. Chunk `'h":"/tm'`: 
   - `ToolFieldStart(key="path")`
   - `ToolFieldDelta(text="/tm")`
3. Chunk `'p/file"}'`:
   - `ToolFieldDelta(text="p/file")`
   - `ToolFieldEnd(key="path")`

Note: The string value `/tmp/file` has no quotes - they are stripped by the parser.

### Example: Object value

For `{"config": {"retry": 3}}`:

```python
from axio.tool_args import ToolArgStream

stream = ToolArgStream("call_2")
events = stream.feed('{"config":{"re')
# ToolFieldStart(key="config")
# ToolFieldDelta(text='{"re')  # raw JSON, not parsed

events = stream.feed('try":3}}')
# ToolFieldDelta(text='try":3}}')  # raw JSON fragment
# ToolFieldEnd(key="config")
```

Object/array values are emitted as raw JSON fragments since they require
nested parsing beyond the scope of field-level events.

## Usage pattern

Typical usage - create one parser per active tool call and forward its output
to the display layer:

<!--
name: test_tool_arg_stream_basic
hidden:
```python
from axio.tool_args import ToolArgStream
```
-->
<!-- name: test_tool_arg_stream_basic -->
```python
from axio import ToolUseStart, ToolInputDelta
from axio.events import ToolFieldStart, ToolFieldDelta, ToolFieldEnd
from axio.tool_args import ToolArgStream

parsers: dict[str, ToolArgStream] = {}

async for event in agent.run_stream(prompt, ctx):
    match event:
        case ToolUseStart(tool_use_id=tid, name=name):
            # Create a new parser for this tool call
            parsers[tid] = ToolArgStream(tid)
            print(f"▶ {name}")

        case ToolInputDelta(tool_use_id=tid, partial_json=pj):
            # Parse streaming JSON into field events
            for field_event in parsers[tid].feed(pj):
                match field_event:
                    case ToolFieldStart(key=key):
                        print(f"\n  {key}: ", end="", flush=True)
                    case ToolFieldDelta(text=text):
                        print(text, end="", flush=True)
                    case ToolFieldEnd():
                        pass  # Field complete

        case ToolResult(tool_use_id=tid, content=content):
            print(f"\n  → {content}")
            parsers.pop(tid, None)  # Clean up parser
```

## String escaping

The parser handles JSON escape sequences in string values:

| Escape | Result |
|--------|--------|
| `\\n` | newline |
| `\\t` | tab |
| `\\r` | carriage return |
| `\\b` | backspace |
| `\\f` | form feed |
| `\\\"` | double quote |
| `\\\\` | backslash |
| `\\/` | forward slash |
| `\\uXXXX` | Unicode codepoint |

Example with escapes:

```python
stream = ToolArgStream("call_3")
events = stream.feed('{"msg":"Hello\\nWorld"}')
# ToolFieldStart(key="msg")
# ToolFieldDelta(text="Hello\nWorld")  # \n resolved to newline
# ToolFieldEnd(key="msg")
```

## Unicode surrogate pairs

The parser correctly handles UTF-16 surrogate pairs in `\uXXXX` escapes:

```python
stream = ToolArgStream("call_4")
# U+1F600 (grinning face) = \uD83D\uDE00
events = stream.feed('{"emoji":"\\uD83D\\uDE00"}')
# ToolFieldDelta(text='😀')  # Single Unicode character
```

If a high surrogate is not followed by a low surrogate, it emits the Unicode
replacement character `\ufffd`.

## See also

- {doc}`events` - Full event reference including `ToolField*` types
- {doc}`tools` - Tool system and handler functions
- [examples/stream_tool_args.py](https://github.com/mosquito/axio-agent/blob/master/examples/stream_tool_args.py) - Complete working example
