# Model Registry

The model registry lets transports advertise which models they support and
what each model can do. This enables capability-based model selection and
cost-aware routing. The same page covers `Usage`, the token counts a turn
reports back. See {ref}`Token accounting <token-accounting>`.

## ModelSpec

A frozen dataclass describing a single model:

<!-- name: test_model_spec -->
```python
from dataclasses import dataclass
from axio.models import Capability


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    capabilities: frozenset[Capability] = frozenset()
    max_output_tokens: int = 8192
    context_window: int = 128000
    input_cost: float = 0.0
    output_cost: float = 0.0
```

`ModelSpec` fixes no unit for `input_cost` and `output_cost`. Whoever fills the
registry decides. Every registry that ships with axio uses price per million
tokens, so `input_cost=3.0` reads as $3 per million input tokens.

## Capability

Models declare their capabilities via a `StrEnum`:

<!-- name: test_capability_enum -->
```python
from enum import StrEnum


class Capability(StrEnum):
    # Input modalities
    text = "text"
    vision = "vision"
    video = "video"
    audio = "audio"
    # Output modalities
    image_generation = "image_generation"
    video_generation = "video_generation"
    # Processing capabilities
    reasoning = "reasoning"
    tool_use = "tool_use"
    json_mode = "json_mode"
    structured_outputs = "structured_outputs"
    embedding = "embedding"
```

| Capability | Meaning |
|---|---|
| `text` | Accepts text input |
| `vision` | Accepts image input |
| `video` | Accepts video input |
| `audio` | Accepts audio input |
| `image_generation` | Can generate images |
| `video_generation` | Can generate videos |
| `reasoning` | Supports extended thinking / chain-of-thought |
| `tool_use` | Supports function / tool calling |
| `json_mode` | Can be forced to output JSON |
| `structured_outputs` | Supports schema-constrained structured outputs |
| `embedding` | Can produce embedding vectors |

## ModelRegistry

A dict-like container for `ModelSpec` values with powerful query methods:

<!-- name: test_model_registry -->
```python
from axio.models import ModelRegistry, ModelSpec, Capability

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
  <!-- name: test_model_registry -->
  ```python
  assert "gpt-4o" in registry.by_prefix("gpt-4").ids()
  ```

`by_capability(*caps)`
: Keep only models that have **all** specified capabilities.
  <!-- name: test_model_registry -->
  ```python
  assert "gpt-4o" in registry.by_capability(Capability.vision, Capability.tool_use).ids()
  ```

`search(*q)`
: Keep models whose ID contains **all** query substrings.
  <!-- name: test_model_registry -->
  ```python
  assert "gpt-4o" in registry.search("gpt", "4o").ids()
  ```

`by_cost(*, output=False, desc=False)`
: Sort by input cost (default) or output cost, ascending or descending.
  <!-- name: test_model_registry -->
  ```python
  cheapest = registry.by_cost()            # cheapest input first
  priciest = registry.by_cost(desc=True)   # most expensive first
  assert cheapest.ids() == priciest.ids()[::-1]
  ```

`ids()`
: Return a plain list of model ID strings.
  <!-- name: test_model_registry -->
  ```python
  assert registry.by_capability(Capability.vision).ids() == ["gpt-4o"]
  ```

### Chaining example

<!-- name: test_model_registry -->
```python
# Find the cheapest vision-capable model with tool use
model = (
    registry
    .by_capability(Capability.vision, Capability.tool_use)
    .by_cost()
    .ids()[0]
)
assert model == "gpt-4o"
```

(token-accounting)=

## Token accounting

`Usage` is what a turn cost in tokens. It arrives on `IterationEnd.usage` for a
single provider request and on `SessionEndEvent.total_usage` for the whole run:

<!-- name: test_usage_shape -->
```python
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int

    cache_read_tokens: int = field(default=0, kw_only=True)
    cache_write_tokens: int = field(default=0, kw_only=True)
    reasoning_tokens: int = field(default=0, kw_only=True)
```

### The rule

`input_tokens` and `output_tokens` are always inclusive grand totals. Every
other field is a disjoint slice of one of them:

```text
cache_read_tokens + cache_write_tokens  <=  input_tokens
reasoning_tokens                        <=  output_tokens
```

Providers disagree about whether their own headline number already contains the
slices, and they disagree in opposite directions. Anthropic counts only the
tokens after the last cache breakpoint, so its transport adds the cache counts
back. Without that, a cached 100k prompt reports as the handful of tokens that
followed the breakpoint. Google reports thinking beside the candidates rather
than inside them, and tool-use prompt tokens outside the prompt count. Its
transport adds both in. The OpenAI Responses API and chat completions both nest
their slices inside their totals already, so those paths add nothing.

Each transport converts into the rule, so nothing downstream has to know which
provider answered. This is also the contract a new transport must satisfy. It is
the reason a new `Usage` field has to be a slice of one of the two totals,
rather than a sixth number beside them.

### Derived properties

<!-- name: test_usage_properties -->
```python
from axio import Usage

u = Usage(
    input_tokens=1000,
    output_tokens=400,
    cache_read_tokens=800,
    cache_write_tokens=50,
    reasoning_tokens=300,
)

assert u.total_tokens == 1400              # input + output
assert u.uncached_input_tokens == 150      # input - cache_read - cache_write
assert u.answer_tokens == 100              # output - reasoning
```

`Usage` supports `+`, and every field adds. A total accumulated across
iterations keeps its slices:

<!-- name: test_usage_properties -->
```python
total = Usage(10, 5, reasoning_tokens=4) + Usage(20, 8, cache_read_tokens=6)
assert (total.input_tokens, total.output_tokens) == (30, 13)
assert (total.reasoning_tokens, total.cache_read_tokens) == (4, 6)
```

### Counts, never money

`Usage` reports tokens and nothing else. A cached token and a written one bill
at different multipliers. `ModelSpec` carries only `input_cost` and
`output_cost`, with no cached-input, cache-write or reasoning rate in the
registry. A cached turn therefore cannot be priced from it. A caller that
wants cost multiplies each slice by its own per-model rates.

A zero slice means the provider billed none of it, or reported no breakdown at
all. Axio cannot tell those apart and does not pretend to.

## Agent capability checking

The agent reads `Capability.tool_use` from the active model before each
iteration. If the model lacks this capability, no tools are passed to the
transport:

```python
model = getattr(transport, "model", None)
model_caps = getattr(model, "capabilities", None)
if model_caps is not None and Capability.tool_use not in model_caps:
    active_tools = []   # embedding or image-gen models: no tools
```

This means you can safely point an `Agent` at an embedding or image-generation
model. It will not try to send tool definitions that the API would reject.
