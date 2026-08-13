"""Nebius AI Studio CompletionTransport - inherits from OpenAI-compatible transport."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.tool import Tool

from axio_transport_openai import OpenAITransport, ThinkingMixin

logger = logging.getLogger(__name__)

_UNSET = ModelSpec(id="<not initialized: call fetch_models() first>", context_window=0, max_output_tokens=0)

UNSET_CONTEXT_LENGTH = 8000
"""The value Nebius publishes when it has not filled the field in.

Sixteen of the twenty-nine models in the catalogue carry exactly this number,
among them a 428B MoE whose own description in the neighbouring field reads
"1M context". Taken literally it says the prompt overflowed before the first
token, which starves the answer or refuses the request outright.
"""

DEFAULT_CONTEXT_LENGTH = 128_000

CONTEXT_LENGTH_OVERRIDES: dict[str, int] = {
    # Stated by Nebius itself, in each model's own description field.
    "MiniMaxAI/MiniMax-M3": 1_000_000,
    "moonshotai/Kimi-K3": 1_000_000,
}
"""What to use instead, for models whose real window we can point at.

Consulted only where the published length is the unset marker: a number Nebius
actually filled in is theirs to be right about.
"""


def _context_window(model_id: str, published: Any) -> int:
    if not published or int(published) == UNSET_CONTEXT_LENGTH:
        return CONTEXT_LENGTH_OVERRIDES.get(model_id, DEFAULT_CONTEXT_LENGTH)
    return int(published)


@dataclass(slots=True)
class NebiusTransport(ThinkingMixin, OpenAITransport):
    name: str = "Nebius AI Studio"
    api_key: str = field(default_factory=lambda: os.environ.get("NEBIUS_API_KEY", ""))
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    model: ModelSpec = field(default_factory=lambda: _UNSET)
    thinking: bool = False

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[Any]:
        if self.model is _UNSET:
            raise RuntimeError("NebiusTransport: call fetch_models() before streaming")
        return OpenAITransport.stream(self, messages, tools, system)

    async def fetch_models(self) -> None:
        """Fetch available models from Nebius ``/v1/models?verbose=true``."""
        assert self.session is not None, "session is required for fetch_models"
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.get(url, params={"verbose": "true"}, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise StreamError(f"Nebius API error {resp.status}: {body}")
            payload: dict[str, Any] = await resp.json()

        self.models.clear()
        for entry in payload.get("data", []):
            m = self._parse_model(entry)
            self.models[m.id] = m
        logger.info("Loaded %d models from %s", len(self.models), url)

        if self.model.id in self.models:
            self.model = self.models[self.model.id]
        elif self.models:
            candidates = self.models.by_capability(Capability.tool_use).by_cost()
            self.model = candidates.first() if candidates else self.models.first()

    @staticmethod
    def _parse_model(entry: dict[str, Any]) -> ModelSpec:
        caps: set[Capability] = set()
        for feat in entry.get("supported_features", []):
            name = "tool_use" if feat == "tools" else feat
            if name in Capability.__members__:
                caps.add(Capability(name))

        modality = entry.get("architecture", {}).get("modality", "")
        parts = modality.split("->") if "->" in modality else [modality]
        input_modality = parts[0]
        output_modality = parts[1] if len(parts) > 1 else ""
        if "image" in input_modality:
            caps.add(Capability.vision)
        if "embedding" in output_modality:
            caps.add(Capability.embedding)

        model_id: str = entry["id"]
        _embed_prefixes = ("BAAI/bge-", "intfloat/e5-", "intfloat/multilingual-e5-")
        if any(model_id.startswith(p) for p in _embed_prefixes) or "/Embedding-" in model_id:
            caps.add(Capability.embedding)

        pricing = entry.get("pricing", {})
        return ModelSpec(
            id=entry["id"],
            context_window=_context_window(model_id, entry.get("context_length")),
            max_output_tokens=int(entry.get("max_output_tokens", 25_000)),
            capabilities=frozenset(caps),
            input_cost=float(pricing.get("prompt", 0)) * 1_000_000,
            output_cost=float(pricing.get("completion", 0)) * 1_000_000,
        )
