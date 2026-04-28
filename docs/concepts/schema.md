# JSON Schema Generation

The schema builder converts Python type annotations into JSON Schema objects.
This schema is sent to LLM providers in the tool definitions, enabling the model
to understand what arguments each tool expects and in what format.

## Purpose

Schema generation serves two critical functions in the axio framework:

1. **Tool input validation** - The JSON schema defines the expected structure,
   types, and constraints for tool arguments. LLM providers use this schema to
   validate and format tool calls before sending them back to the agent.

2. **Transport communication** - Each `CompletionTransport` sends the tool's
   `input_schema` property to the LLM backend. The schema must be valid JSON
   Schema compatible with the provider's expectations.

<!-- name: test_schema_purpose -->
```python
from axio import Tool, Field
from typing import Annotated

async def search(
    query: Annotated[str, Field(description="Search query")],
    limit: Annotated[int, Field(default=10, ge=1, le=100)] = 10,
) -> str:
    """Search the knowledge base."""
    return f"results for {query!r}"

tool = Tool(name="search", handler=search)
# tool.input_schema contains the JSON schema sent to the LLM
```

## Type Mappings

The schema builder maps Python types to JSON Schema types as follows:

| Python Type | JSON Schema |
|-------------|-------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[T]` | `{"type": "array", "items": {...}}` |
| `dict` | `{"type": "object"}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |
| `T | None` or `Optional[T]` | Schema of `T` (nullable) |

<!-- name: test_type_mappings -->
```python
from typing import Literal, Optional, get_type_hints
from axio.schema import build_tool_schema

def example(
    name: str,
    count: int,
    ratio: float,
    enabled: bool,
    tags: list[str],
    mode: Literal["read", "write"],
    description: Optional[str] = None,
) -> str:
    return "example"

schema = build_tool_schema(example)
assert schema["type"] == "object"
assert schema["properties"]["name"]["type"] == "string"
assert schema["properties"]["count"]["type"] == "integer"
assert schema["properties"]["ratio"]["type"] == "number"
assert schema["properties"]["enabled"]["type"] == "boolean"
assert schema["properties"]["tags"]["type"] == "array"
assert schema["properties"]["tags"]["items"]["type"] == "string"
assert schema["properties"]["mode"]["enum"] == ["read", "write"]
```

## Field Metadata

Use `Field()` to add descriptions, defaults, and constraints to parameters.
The schema builder extracts `FieldInfo` metadata and translates it into JSON
Schema keywords:

| Field Parameter | JSON Schema Key | Meaning |
|-----------------|-----------------|---------|
| `description` | `"description"` | Human-readable description |
| `ge` | `"minimum"` | Minimum value (inclusive) |
| `le` | `"maximum"` | Maximum value (inclusive) |
| `default` | (not in schema) | Used for required field detection |

<!-- name: test_field_metadata -->
```python
from typing import Annotated
from axio import Field
from axio.schema import build_tool_schema

def constrained(
    username: Annotated[str, Field(description="User identifier")],
    age: Annotated[int, Field(ge=0, le=150)],
    score: Annotated[float, Field(ge=0.0, le=100.0)] = 0.0,
) -> str:
    return f"{username}: {age}"

schema = build_tool_schema(constrained)
props = schema["properties"]
assert props["username"]["description"] == "User identifier"
assert props["age"]["minimum"] == 0
assert props["age"]["maximum"] == 150
assert props["score"]["minimum"] == 0.0
assert props["score"]["maximum"] == 100.0
```

## Required vs Optional Fields

A field is **required** when it has no default value. The schema builder checks
three sources for defaults:

1. `Field(default=...)` - Explicit field default
2. Function signature default - `param=default_value`
3. Class attribute default - For class-based handlers

Fields without any of these are added to the `"required"` list.

<!-- name: test_required_fields -->
```python
from typing import Annotated
from axio import Field
from axio.field import MISSING
from axio.schema import build_tool_schema

def required_vs_optional(
    required_str: str,                        # required
    annotated_required: Annotated[str, Field(description="no default")],  # required
    optional_default: str = "default",        # optional
    field_default: Annotated[str, Field(default="field")] = "x",  # optional (Field default)
) -> str:
    return "test"

schema = build_tool_schema(required_vs_optional)
assert "required_str" in schema.get("required", [])
assert "annotated_required" in schema.get("required", [])
assert "optional_default" not in schema.get("required", [])
assert "field_default" not in schema.get("required", [])
```

## Examples

### Basic Tool Schema

<!-- name: test_basic_schema -->
```python
from axio import Field, Tool
from typing import Annotated

async def greet(name: Annotated[str, Field(description="Name to greet")],
                repeat: Annotated[int, Field(default=1, ge=1, le=5)] = 1) -> str:
    """Greet someone by name."""
    return " ".join([f"Hello, {name}!" for _ in range(repeat)])

tool = Tool(name="greet", handler=greet)

# The generated schema:
# {
#     "type": "object",
#     "properties": {
#         "name": {
#             "type": "string",
#             "description": "Name to greet"
#         },
#         "repeat": {
#             "type": "integer",
#             "minimum": 1,
#             "maximum": 5
#         }
#     },
#     "required": ["name"]
# }
```

### Literal Enum

<!-- name: test_literal_schema -->
```python
from typing import Annotated, Literal
from axio import Field

def set_mode(mode: Annotated[Literal["on", "off", "toggle"], Field(description="Power mode")]) -> str:
    return f"Mode set to {mode}"

from axio.schema import build_tool_schema
schema = build_tool_schema(set_mode)
assert schema["properties"]["mode"]["enum"] == ["on", "off", "toggle"]
assert schema["properties"]["mode"]["description"] == "Power mode"
```

### Array of Strings

<!-- name: test_array_schema -->
```python
from typing import Annotated
from axio import Field

def create_tags(tags: Annotated[list[str], Field(description="List of tags")]) -> str:
    return f"Created tags: {tags}"

from axio.schema import build_tool_schema
schema = build_tool_schema(create_tags)
assert schema["properties"]["tags"]["type"] == "array"
assert schema["properties"]["tags"]["items"]["type"] == "string"
assert schema["properties"]["tags"]["description"] == "List of tags"
```

## Integration with Transports

Transports call `build_tool_schema()` internally to generate the `input_schema`
property sent to LLM providers. The schema is included in tool definitions
alongside the tool's name and description.

See {doc}`tools` for how tools use the schema and {doc}`field` for detailed
Field metadata options.

## See Also

- {doc}`field` - `FieldInfo` and `Field()` metadata system
- {doc}`tools` - Tool handlers and schema consumption
