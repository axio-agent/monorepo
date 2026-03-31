"""Transport-agnostic model types: Capability, ModelSpec, ModelRegistry."""

from __future__ import annotations

from collections.abc import ItemsView, Iterable, Iterator, KeysView, MutableMapping, ValuesView
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class TransportMeta:
    """Metadata a transport plugin declares about itself."""

    label: str
    api_key_env: str
    role_defaults: dict[str, str]


class Capability(StrEnum):
    text = "text"
    vision = "vision"
    reasoning = "reasoning"
    tool_use = "tool_use"
    json_mode = "json_mode"
    structured_outputs = "structured_outputs"
    embedding = "embedding"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    capabilities: frozenset[Capability] = frozenset()
    max_output_tokens: int = 8192
    context_window: int = 128000
    input_cost: float = 0.0
    output_cost: float = 0.0


class ModelRegistry(MutableMapping[str, ModelSpec]):
    __slots__ = ("_models",)

    def __init__(self, models: Iterable[ModelSpec] | None = None) -> None:
        self._models: dict[str, ModelSpec] = {m.id: m for m in (models or [])}

    def __setitem__(self, key: str, value: ModelSpec, /) -> None:
        if not isinstance(value, ModelSpec):
            raise ValueError("ModelRegistry values must be ModelSpec instances")
        self._models[key] = value

    def __delitem__(self, key: str, /) -> None:
        del self._models[key]

    def __getitem__(self, key: str, /) -> ModelSpec:
        return self._models[key]

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[ModelSpec]:  # type: ignore[override]
        return iter(self._models.values())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModelRegistry):
            return self._models == other._models
        if isinstance(other, dict):
            return self._models == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"ModelRegistry({self._models!r})"

    def clear(self) -> None:
        self._models.clear()

    def keys(self) -> KeysView[str]:
        return self._models.keys()

    def values(self) -> ValuesView[ModelSpec]:
        return self._models.values()

    def items(self) -> ItemsView[str, ModelSpec]:
        return self._models.items()

    def by_prefix(self, prefix: str) -> ModelRegistry:
        return ModelRegistry(v for k, v in self._models.items() if k.startswith(prefix))

    def by_capability(self, *caps: Capability) -> ModelRegistry:
        required = frozenset(caps)
        return ModelRegistry(v for v in self._models.values() if required <= v.capabilities)

    def search(self, *q: str) -> ModelRegistry:
        """search by parts of id"""
        return ModelRegistry(v for k, v in self._models.items() if all(part in k for part in q))

    def by_cost(self, *, output: bool = False, desc: bool = False) -> ModelRegistry:
        """Return registry ordered by cost (input by default, output if *output=True*)."""
        attr = "output_cost" if output else "input_cost"
        items = sorted(self._models.values(), key=lambda v: getattr(v, attr), reverse=desc)
        return ModelRegistry(items)

    def ids(self) -> list[str]:
        return list(self._models)
