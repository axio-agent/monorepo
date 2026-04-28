# Field Metadata System

The field metadata system provides a lightweight mechanism for annotating tool
parameters with descriptions, defaults, and validation constraints. It is used
to build JSON schemas that are sent to LLMs so they understand how to call tools.

## FieldInfo and Field()

{class}`FieldInfo` is a dataclass that holds metadata about a tool parameter:

<!--
name: test_fieldinfo_basics
```python
from axio import FieldInfo
```
-->
<!-- name: test_fieldinfo_fields -->
```python
from dataclasses import fields
from axio import FieldInfo

# FieldInfo has these fields:
fi = FieldInfo(
    description="Search query",  # Human-readable description
    default="MISSING",          # Default value (or MISSING if required)
    ge=1,                       # Minimum value (≥)
    le=100,                     # Maximum value (≤)
)
```

The `Field()` constructor function provides a convenient API:

<!-- name: test_field_constructor -->
```python
from axio import Field
from axio.field import MISSING

# Basic usage with description
query_field = Field(description="Search query")

# With default value
limit_field = Field(default=10)

# With numeric bounds
count_field = Field(ge=1, le=50)

# Combined
count_field = Field(description="Number of results", default=10, ge=1, le=100)
```

## Using Field in Tool Definitions

Use `Annotated` from the `typing` module to attach `FieldInfo` to parameters:

<!-- name: test_field_in_tool -->
```python
from typing import Annotated
from axio import Field, Tool


async def search(
    query: Annotated[str, Field(description="Search query")],
    limit: Annotated[int, Field(default=10, ge=1, le=100)] = 10,
) -> str:
    """Search the knowledge base."""
    return f"results for {query!r} (limit={limit})"


tool = Tool(name="search", handler=search)
```

The field metadata is extracted automatically when building the tool's input
schema. See {doc}`schema` for details on schema generation.

### `description`

A human-readable string describing what the parameter is for. This is sent
to the LLM as part of the JSON schema:

<!-- name: test_field_description -->
```python
from typing import Annotated
from axio import Field
from axio.schema import build_tool_schema


async def lookup(
    user_id: Annotated[str, Field(description="The unique identifier of the user")],
) -> str:
    """Look up user details."""
    return f"User {user_id}"


schema = build_tool_schema(lookup)
assert schema["properties"]["user_id"]["description"] == "The unique identifier of the user"
```

### `default`

Specify a default value to make a parameter optional:

<!-- name: test_field_default -->
```python
from typing import Annotated
from axio import Field
from axio.field import MISSING
from axio.schema import build_tool_schema


async def paginate(
    page: Annotated[int, Field(default=1)],
    page_size: Annotated[int, Field(default=20)],
) -> str:
    """Paginate results."""
    return f"page {page}, size {page_size}"


schema = build_tool_schema(paginate)
# No 'required' key since all params have defaults
assert "required" not in schema or len(schema.get("required", [])) == 0
```

The special `MISSING` sentinel indicates that a field has no default and is
required.

### `ge` and `le` (Numeric bounds)

Constrain numeric parameters with minimum and maximum values:

<!-- name: test_field_bounds -->
```python
from typing import Annotated
from axio import Field
from axio.schema import build_tool_schema


async def batch_process(
    count: Annotated[int, Field(ge=1, le=50)],
    threshold: Annotated[float, Field(ge=0.0, le=1.0)],
) -> str:
    """Process items in batches."""
    return f"Processing {count} items with threshold {threshold}"


schema = build_tool_schema(batch_process)
assert schema["properties"]["count"]["minimum"] == 1
assert schema["properties"]["count"]["maximum"] == 50
assert schema["properties"]["threshold"]["minimum"] == 0.0
assert schema["properties"]["threshold"]["maximum"] == 1.0
```

The LLM will see these constraints and (ideally) respect them when generating
tool calls. Axio also validates these constraints at runtime before calling
the handler.

### `strict` Mode

Some string parameters should reject implicit type coercion. Use `StrictStr`
to enforce strict string typing:

<!-- name: test_strict_str -->
```python
from typing import Annotated
from axio import StrictStr, FieldInfo
from axio.schema import build_tool_schema


async def set_name(name: StrictStr) -> str:
    """Set the name field."""
    return f"Name set to: {name}"


# StrictStr is Annotated[str, FieldInfo(strict=True)]
# The schema shows it's a string
schema = build_tool_schema(set_name)
assert schema["properties"]["name"]["type"] == "string"

# But at runtime, it will reject non-string values
fi = next(a for a in StrictStr.__metadata__ if isinstance(a, FieldInfo))
assert fi.strict is True
```

This is useful for file paths, identifiers, or any string where you want to
reject cases where an LLM might pass an integer like `0` or `1` instead of
`"0"` or `"1"`.

## Complete Examples

### File Operation Tool

<!-- name: test_field_file_example -->
```python
from typing import Annotated, Literal
from axio import Field, Tool


async def write_file(
    path: Annotated[str, Field(description="File path to write to")],
    content: Annotated[str, Field(description="Content to write to the file")],
    mode: Annotated[Literal["w", "a"], Field(default="w", description="Write mode: 'w' or 'a'")] = "w",
) -> str:
    """Write content to a file."""
    with open(path, mode) as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


tool = Tool(name="write_file", handler=write_file)
```

### HTTP Request Tool

<!-- name: test_field_http_example -->
```python
from typing import Annotated
from axio import Field, Tool


async def http_request(
    url: Annotated[str, Field(description="The URL to request")],
    method: Annotated[str, Field(default="GET", description="HTTP method")] = "GET",
    timeout: Annotated[float, Field(default=30.0, ge=0.1, le=300.0)] = 30.0,
) -> str:
    """Make an HTTP request."""
    # Implementation would use httpx or aiohttp
    return f"{method} {url} (timeout={timeout}s)"


tool = Tool(name="http_request", handler=http_request)
```

### Database Query Tool

<!-- name: test_field_db_example -->
```python
from typing import Annotated
from axio import Field, Tool


async def query_users(
    status: Annotated[str, Field(default="", description="Filter by status: 'active', 'inactive', or empty for all")],
    limit: Annotated[int, Field(default=100, ge=1, le=1000, description="Maximum number of rows to return")] = 100,
    offset: Annotated[int, Field(default=0, ge=0, description="Number of rows to skip")] = 0,
) -> str:
    """Query users from the database."""
    parts = [f"SELECT * FROM users"]
    if status:
        parts.append(f"WHERE status = '{status}'")
    parts.append(f"LIMIT {limit} OFFSET {offset}")
    return " ".join(parts)


tool = Tool(name="query_users", handler=query_users)
```

## API Reference

### FieldInfo

```python
@dataclass(frozen=True)
class FieldInfo:
    description: str = ""
    default: Any = MISSING
    ge: int | float | None = None
    le: int | float | None = None
    strict: bool = False
```

`validate(value, name, hint)`
: Validate a value against this field's constraints. Raises `TypeError`
  for type violations (when `strict=True`) or `ValueError` for bound
  violations.

### Field()

```python
def Field(
    description: str = "",
    default: Any = MISSING,
    ge: int | float | None = None,
    le: int | float | None = None,
) -> FieldInfo:
    """Create FieldInfo with metadata."""
```

### StrictStr

```python
StrictStr = Annotated[str, FieldInfo(strict=True)]
```

A type alias for strings that must be strictly typed (no coercion from int/etc).

### get_field_info()

```python
def get_field_info(annotation: Any) -> FieldInfo | None:
    """Extract FieldInfo from an Annotated type annotation."""
```

Utility function to extract `FieldInfo` from `Annotated[T, FieldInfo(...)]`
at runtime.

## Related

- {doc}`schema` - How field metadata is converted to JSON schema
- {doc}`tools` - Using Field in tool definitions
