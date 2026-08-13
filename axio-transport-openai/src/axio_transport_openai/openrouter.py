"""OpenRouter CompletionTransport - inherits from OpenAI-compatible transport."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any

from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.tool import Tool

from axio_transport_openai import OpenAITransport, ThinkingMixin

logger = logging.getLogger(__name__)


def _split_provider(model_id: str) -> tuple[str, str | None]:
    """Split ``model@provider`` into its parts, leaving a bare id untouched."""
    base, sep, provider = model_id.rpartition("@")
    if not sep or not base or not provider:
        return model_id, None
    return base, provider


@dataclass(slots=True)
class OpenRouterTransport(ThinkingMixin, OpenAITransport):
    name: str = "OpenRouter"
    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    model: ModelSpec = ModelSpec(id="google/gemini-2.5-pro-preview")
    thinking: bool = False

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        payload = super().build_payload(messages, tools, system)
        # ThinkingMixin asks for enable_thinking, which OpenRouter does not
        # understand: it takes a reasoning object and returns the trace in
        # delta.reasoning.
        if payload.pop("enable_thinking", None):
            payload.setdefault("reasoning", {"enabled": True})
        model_id, provider = _split_provider(str(payload.get("model", "")))
        if provider:
            payload["model"] = model_id
            payload.setdefault("provider", {"only": [provider]})
        return payload

    def resolve_model(self, model_id: str) -> ModelSpec:
        """Resolve ``[<lab>/]<model>[:tier][@<provider>]``.

        The tier (``:nitro``, ``:free``) is OpenRouter's own routing suffix and
        the registry may not list the variant, so its metadata comes from the
        base model. The ``@provider`` suffix is ours: it is not part of any
        model id, it pins serving to one provider and is peeled off into
        ``provider.only`` when the request is built. Both stay in the resolved
        id so the selection survives a round trip through ``/model``.
        """
        base_id, provider = _split_provider(model_id)
        try:
            spec = self.models[base_id]
        except KeyError:
            stripped, sep, _tier = base_id.rpartition(":")
            if not sep or not stripped:
                raise KeyError(model_id) from None
            try:
                spec = self.models[stripped]
            except KeyError:
                raise KeyError(model_id) from None
        return spec if spec.id == model_id else replace(spec, id=model_id)

    async def fetch_models(self) -> None:
        """Fetch available models from OpenRouter ``/v1/models``."""
        assert self.session is not None, "session is required for fetch_models"
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.get(url, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise StreamError(f"OpenRouter API error {resp.status}: {body}")
            payload: dict[str, Any] = await resp.json()

        self.models.clear()
        for entry in payload.get("data", []):
            m = self._parse_model(entry)
            self.models[m.id] = m
        logger.info("Loaded %d models from %s", len(self.models), url)

    @staticmethod
    def _parse_model(entry: dict[str, Any]) -> ModelSpec:
        caps: set[Capability] = set()

        params: list[str] = entry.get("supported_parameters", [])
        if "tools" in params:
            caps.add(Capability.tool_use)
        # OpenRouter advertises reasoning support here; without reading it the
        # capability is never set, so thinking is never requested or announced.
        if "reasoning" in params or "include_reasoning" in params:
            caps.add(Capability.reasoning)

        arch: dict[str, Any] = entry.get("architecture", {})
        input_modalities: list[str] = arch.get("input_modalities", [])
        output_modalities: list[str] = arch.get("output_modalities", [])
        if "image" in input_modalities:
            caps.add(Capability.vision)
        if "embedding" in output_modalities:
            caps.add(Capability.embedding)

        top: dict[str, Any] = entry.get("top_provider", {})
        context_window = int(entry.get("context_length") or top.get("context_length") or 128_000)
        max_output_tokens = int(top.get("max_completion_tokens") or 8_000)

        pricing: dict[str, Any] = entry.get("pricing", {})
        return ModelSpec(
            id=entry["id"],
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            capabilities=frozenset(caps),
            input_cost=float(pricing.get("prompt", 0)) * 1_000_000,
            output_cost=float(pricing.get("completion", 0)) * 1_000_000,
        )
