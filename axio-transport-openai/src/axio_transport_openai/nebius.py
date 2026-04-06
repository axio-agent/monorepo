"""Nebius AI Studio CompletionTransport — inherits from OpenAI-compatible transport."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from axio.exceptions import StreamError
from axio.models import Capability, ModelSpec

from axio_transport_openai import OpenAITransport

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NebiusTransport(OpenAITransport):
    name: str = "Nebius AI Studio"
    api_key: str = field(default_factory=lambda: os.environ.get("NEBIUS_API_KEY", ""))
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    model: ModelSpec = ModelSpec(id="deepseek-ai/DeepSeek-V3-0324")

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

        # Heuristic for known embedding model families
        model_id: str = entry["id"]
        _embed_prefixes = ("BAAI/bge-", "intfloat/e5-", "intfloat/multilingual-e5-")
        if any(model_id.startswith(p) for p in _embed_prefixes) or "/Embedding-" in model_id:
            caps.add(Capability.embedding)

        pricing = entry.get("pricing", {})
        return ModelSpec(
            id=entry["id"],
            context_window=int(entry.get("context_length", 128_000)),
            max_output_tokens=int(entry.get("max_output_tokens", 25_000)),
            capabilities=frozenset(caps),
            input_cost=float(pricing.get("prompt", 0)) * 1_000_000,
            output_cost=float(pricing.get("completion", 0)) * 1_000_000,
        )
