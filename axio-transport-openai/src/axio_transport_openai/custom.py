"""Custom OpenAI-compatible provider transport.

Each custom provider is a separate :class:`OpenAICompatibleTransport` instance with its
own ``base_url``, ``api_key``, and ``models``.  Instances are created by the TUI hub
screen and registered dynamically in the transport registry.

Configuration is persisted to ``~/.local/share/axio/openai-custom.json``:

.. code-block:: json

    [
      {
        "name": "localai",
        "base_url": "http://localhost:8080/v1",
        "api_key": "",
        "models": [
          {
            "id": "llama3.2",
            "context_window": 131072,
            "max_output_tokens": 4096,
            "capabilities": ["text", "tool_use"],
            "input_cost": 0.0,
            "output_cost": 0.0
          }
        ]
      }
    ]
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import aiohttp
from axio.models import ModelRegistry

from axio_transport_openai import OpenAITransport


@dataclass
class OpenAICompatibleTransport(OpenAITransport):
    """OpenAI-compatible transport for a single user-defined provider.

    Instances are created by :class:`~axio_transport_openai.tui.custom.CustomHubScreen`
    with ``name``, ``base_url``, ``api_key``, and ``models`` populated from the JSON
    config.  Supports JSON round-trip via :meth:`to_dict` / :meth:`from_dict`.
    """

    # These point at servers that implement /v1/chat/completions and not /v1/responses.
    api: Literal["responses", "chat"] = field(default="chat", kw_only=True)

    base_url: str = ""  # override OpenAITransport default
    models: ModelRegistry = field(default_factory=ModelRegistry)  # empty default

    async def fetch_models(self) -> None:
        pass  # models passed in at construction

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, session: aiohttp.ClientSession | None = None) -> Self:
        """As :meth:`OpenAITransport.from_dict`, but ``base_url``/``api_key`` come verbatim from ``data``.

        The base implementation reads a value saved empty as empty, and falls back to the
        ``OPENAI_BASE_URL``/``OPENAI_API_KEY`` env vars only where the key is absent altogether.
        That is right for the built-in OpenAI provider, whose settings dict is written by hand and
        omits what it wants the default for.

        A custom provider's config is a full round-trip of :meth:`to_dict`, so an absent key means
        the same thing as an empty one: this server takes no credential. Reading it through the
        environment would point a local server at an unrelated real endpoint.
        """
        obj = super().from_dict(data, session=session)
        return dataclasses.replace(obj, base_url=str(data.get("base_url", "")), api_key=str(data.get("api_key", "")))
