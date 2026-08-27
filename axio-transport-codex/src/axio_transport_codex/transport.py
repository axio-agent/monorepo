"""ChatGPT (Codex) transport - Responses API over SSE."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from axio.events import (
    StreamEvent,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.retry import is_retryable, retry_delay
from axio.schema import strip_title
from axio.tool import Tool
from axio.transport import CompletionTransport
from axio_responses import STOP_REASONS, Responses, convert_messages, convert_tools

from .oauth import CLIENT_ID, ORIGINATOR, TOKEN_URL, _decode_jwt_payload

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_USER_AGENT = f"codex_cli_rs/0.1.0 ({platform.system()} {platform.release()}; {platform.machine()})"

_VT = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_RT = frozenset({Capability.text, Capability.reasoning, Capability.tool_use})
_TT = frozenset({Capability.text, Capability.tool_use})

CODEX_MODELS: ModelRegistry = ModelRegistry(
    {
        ModelSpec(id="o4-mini", context_window=200_000, max_output_tokens=100_000, capabilities=_RT),
        ModelSpec(id="gpt-4.1", context_window=1_047_576, max_output_tokens=32_768, capabilities=_VT),
        ModelSpec(id="gpt-4.1-mini", context_window=1_047_576, max_output_tokens=32_768, capabilities=_VT),
        ModelSpec(id="gpt-4.1-nano", context_window=1_047_576, max_output_tokens=32_768, capabilities=_TT),
        ModelSpec(id="gpt-4o", context_window=128_000, max_output_tokens=16_384, capabilities=_VT),
        ModelSpec(id="gpt-4o-mini", context_window=128_000, max_output_tokens=16_384, capabilities=_VT),
        ModelSpec(id="o3", context_window=200_000, max_output_tokens=100_000, capabilities=_RT),
        ModelSpec(id="o3-mini", context_window=200_000, max_output_tokens=100_000, capabilities=_RT),
    }
)

#: Every ``status`` the Responses API publishes, and every ``incomplete_details.reason``.
#: A status left out of this map ends the run as an error.
#: The Responses vocabulary lives in axio-responses: the public /v1/responses endpoint and
#: this ChatGPT backend speak it alike. Re-exported under this module's own names.
_STOP_REASON_MAP = STOP_REASONS
_strip_title = strip_title
_convert_tools = convert_tools
_convert_messages = convert_messages


@dataclass(slots=True)
class CodexTransport(CompletionTransport):
    name: str = "ChatGPT (Codex)"
    api_key: str = ""
    refresh_token: str = ""
    expires_at: str = ""
    account_id: str = ""
    base_url: str = CODEX_BASE_URL
    model: ModelSpec = field(default_factory=lambda: CODEX_MODELS["o4-mini"])
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry(CODEX_MODELS.values()))
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    on_auth_refresh: Callable[[dict[str, str]], Awaitable[None]] | None = field(
        default=None, repr=False, compare=False
    )
    max_retries: int = 10
    retry_base_delay: float = 5.0

    async def _ensure_token(self) -> None:
        """Refresh access token if expired or about to expire."""
        if not self.refresh_token or not self.expires_at:
            return
        try:
            expires_at = int(self.expires_at)
        except ValueError:
            return
        if time.time() < expires_at - 30:
            return

        logger.info("Access token expired or expiring soon, refreshing...")
        await self._refresh()

    async def _refresh(self) -> None:
        """Refresh the access token using the refresh token."""
        payload = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": self.refresh_token,
        }

        async with aiohttp.ClientSession() as sess:
            async with sess.post(TOKEN_URL, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise StreamError(f"Token refresh failed ({resp.status}): {body}")
                data: dict[str, Any] = await resp.json()

        self.api_key = data["access_token"]
        self.refresh_token = data.get("refresh_token", self.refresh_token)
        expires_in: int = data.get("expires_in", 3600)
        self.expires_at = str(int(time.time()) + expires_in)

        jwt_payload = _decode_jwt_payload(self.api_key)
        orgs = jwt_payload.get("organizations", [])
        if orgs and isinstance(orgs, list) and isinstance(orgs[0], dict):
            self.account_id = orgs[0].get("id", self.account_id)

        logger.info("Token refreshed, expires_at=%s", self.expires_at)

        if self.on_auth_refresh is not None:
            try:
                await self.on_auth_refresh(
                    {
                        "api_key": self.api_key,
                        "refresh_token": self.refresh_token,
                        "expires_at": self.expires_at,
                        "account_id": self.account_id,
                    }
                )
            except Exception:
                logger.warning("Failed to persist refreshed tokens", exc_info=True)

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        instructions, input_items = _convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model.id,
            "input": input_items,
            "stream": True,
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = _convert_tools(tools)
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True

        # Log input items summary for debugging
        fc = [i for i in input_items if i.get("type") == "function_call"]
        fco = [i for i in input_items if i.get("type") == "function_call_output"]
        if fc or fco:
            fc_ids = [i.get("call_id") for i in fc]
            fco_ids = [i.get("call_id") for i in fco]
            logger.info("Input: %d function_calls %s, %d outputs %s", len(fc), fc_ids, len(fco), fco_ids)

        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        """Parse Responses API SSE events into axio StreamEvents."""
        turn = Responses()
        async for made in turn.over(resp.content.iter_any(), until="[DONE]"):
            yield made
        yield turn.finished()

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        return self._do_stream(messages, tools, system)

    async def _do_stream(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncIterator[StreamEvent]:
        assert self.session is not None, "session is required for streaming"

        logger.debug("Stream start: model=%s, messages=%d, tools=%d", self.model.id, len(messages), len(tools))

        await self._ensure_token()
        logger.debug("Token check passed")

        url = f"{self.base_url}/responses"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": CODEX_USER_AGENT,
            "originator": ORIGINATOR,
        }
        if self.account_id:
            headers["ChatGPT-Account-ID"] = self.account_id

        payload = self.build_payload(messages, tools, system)
        logger.debug("Payload built: %d input items", len(payload.get("input", [])))

        if logger.getEffectiveLevel() <= logging.DEBUG:
            dumped = json.dumps(payload, indent=2)
            if len(dumped) > 4000:
                dumped = dumped[:4000] + f"\n... truncated ({len(dumped)} chars total)"
            logger.debug("Request payload:\n%s", dumped)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            retry_resp: aiohttp.ClientResponse | None = None
            try:
                logger.debug("POST %s (attempt %d/%d)", url, attempt, self.max_retries)
                async with self.session.post(url, json=payload, headers=headers) as resp:
                    logger.debug("HTTP response: status=%d content_type=%s", resp.status, resp.content_type)
                    if resp.status == 200:
                        logger.debug("SSE parsing started")
                        async for event in self._parse_sse(resp):
                            yield event
                        logger.debug("SSE parsing finished")
                        return

                    body = await resp.text()
                    if is_retryable(resp.status):
                        retry_resp = resp
                        last_exc = StreamError(f"Codex API error {resp.status}: {body}")
                        logger.warning(
                            "Retryable HTTP %d (attempt %d/%d): %s",
                            resp.status,
                            attempt,
                            self.max_retries,
                            body,
                        )
                    else:
                        logger.error("HTTP %d from %s: %s", resp.status, url, body)
                        raise StreamError(f"Codex API error {resp.status}: {body}")
            except aiohttp.ClientError as exc:
                last_exc = StreamError(str(exc))
                logger.warning("Connection error (attempt %d/%d): %s", attempt, self.max_retries, exc)

            if attempt < self.max_retries:
                delay = retry_delay(retry_resp, attempt, base=self.retry_base_delay)
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        raise last_exc or StreamError("Max retries exceeded")

    async def fetch_models(self) -> None:
        """Fetch available models from the Codex API."""
        if not self.api_key or self.session is None:
            self.models = ModelRegistry(CODEX_MODELS.values())
            return

        await self._ensure_token()

        url = f"{self.base_url}/models?client_version=0.0.1"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": CODEX_USER_AGENT,
            "originator": ORIGINATOR,
        }
        if self.account_id:
            headers["ChatGPT-Account-ID"] = self.account_id

        try:
            async with self.session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning("Failed to fetch models (%d), using defaults", resp.status)
                    self.models = ModelRegistry(CODEX_MODELS.values())
                    return
                data: dict[str, Any] = await resp.json()
        except Exception:
            logger.warning("Failed to fetch models, using defaults", exc_info=True)
            self.models = ModelRegistry(CODEX_MODELS.values())
            return

        # Parse model list - the WHAM endpoint may return different formats
        model_list = data.get("data", data.get("models", []))
        if not model_list:
            self.models = ModelRegistry(CODEX_MODELS.values())
            return

        specs: list[ModelSpec] = []
        for item in model_list:
            model_id = item.get("id", item.get("slug", ""))
            if not model_id:
                continue
            # Use known spec if available, otherwise build one from API data.
            if model_id in CODEX_MODELS:
                specs.append(CODEX_MODELS[model_id])
            else:
                specs.append(
                    ModelSpec(
                        id=model_id,
                        capabilities=_TT,
                        context_window=item.get("context_window", 128_000),
                        max_output_tokens=item.get("max_output_tokens", 8_192),
                    )
                )

        self.models = ModelRegistry(specs) if specs else ModelRegistry(CODEX_MODELS.values())

        # If the currently selected model was dropped from the API list, switch to
        # the first available one so the transport stays usable out of the box.
        if self.model.id not in self.models:
            self.model = self.models.first()
