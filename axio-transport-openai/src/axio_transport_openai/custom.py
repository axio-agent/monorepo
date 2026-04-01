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

from dataclasses import dataclass, field
from typing import ClassVar

from axio.models import ModelRegistry, TransportMeta

from axio_transport_openai import OpenAITransport


@dataclass
class OpenAICompatibleTransport(OpenAITransport):
    """OpenAI-compatible transport for a single user-defined provider.

    Instances are created by :class:`~axio_transport_openai.tui.custom.CustomHubScreen`
    with ``name``, ``base_url``, ``api_key``, and ``models`` populated from the JSON
    config.  Supports JSON round-trip via :meth:`to_dict` / :meth:`from_dict`.
    """

    META: ClassVar[TransportMeta] = TransportMeta(
        label="OpenAI-Compatible",
        api_key_env="",  # no env-var required; the hub manages activation
        role_defaults={},
    )

    base_url: str = ""  # override OpenAITransport default
    models: ModelRegistry = field(default_factory=ModelRegistry)  # empty default

    async def fetch_models(self) -> None:
        pass  # models passed in at construction
