# Model Registry

The model registry lets transports advertise which models they support and
what each model can do. This enables capability-based model selection and
cost-aware routing.

## ModelSpec

A frozen dataclass describing a single model:

```python
@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    capabilities: frozenset[Capability] = frozenset()
    max_output_tokens: int = 8192
    context_window: int = 128000
    input_cost: float = 0.0
    output_cost: float = 0.0
```

## Capability

Models declare their capabilities via a `StrEnum`:

```python
class Capability(StrEnum):
    text = "text"
    vision = "vision"
    reasoning = "reasoning"
    tool_use = "tool_use"
    json_mode = "json_mode"
    structured_outputs = "structured_outputs"
    embedding = "embedding"
```

## ModelRegistry

A dict-like container for `ModelSpec` values with powerful query methods:

```python
registry = ModelRegistry()
registry["gpt-4o"] = ModelSpec(
    id="gpt-4o",
    capabilities=frozenset({Capability.text, Capability.vision, Capability.tool_use}),
    context_window=128000,
    input_cost=2.50,
    output_cost=10.00,
)
```

### Query methods

All query methods return a new `ModelRegistry`, so they can be chained:

`by_prefix(prefix)`
: Filter models whose ID starts with a prefix.
  ```python
  registry.by_prefix("gpt-4")
  ```

`by_capability(*caps)`
: Keep only models that have **all** specified capabilities.
  ```python
  registry.by_capability(Capability.vision, Capability.tool_use)
  ```

`search(*q)`
: Keep models whose ID contains **all** query substrings.
  ```python
  registry.search("gpt", "4o")
  ```

`by_cost(*, output=False, desc=False)`
: Sort by input cost (default) or output cost, ascending or descending.
  ```python
  cheapest = registry.by_cost()            # cheapest input first
  priciest = registry.by_cost(desc=True)   # most expensive first
  ```

`ids()`
: Return a plain list of model ID strings.
  ```python
  registry.by_capability(Capability.vision).ids()
  # ["gpt-4o", "gpt-4o-mini", ...]
  ```

### Chaining example

```python
# Find the cheapest vision-capable model with tool use
model = (
    registry
    .by_capability(Capability.vision, Capability.tool_use)
    .by_cost()
    .ids()[0]
)
```
