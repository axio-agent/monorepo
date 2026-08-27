"""Gemini Live WebSocket realtime transport.

Implements :class:`axio.transport.RealtimeTransport` against the Gemini
Live API (``wss://generativelanguage.googleapis.com/ws/...BidiGenerateContent``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import aiohttp
from axio.blocks import AudioBlock, ContentBlock, TextBlock
from axio.events import (
    AudioOutputDelta,
    SpeechStarted,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from axio.tool import Tool
from axio.transport import RealtimeSession, RealtimeTransport
from axio.types import StopReason, ToolCallID, ToolName, Usage

logger = logging.getLogger(__name__)


class _RefreshableCredentials(Protocol):
    valid: bool
    expired: bool
    token: str | None

    def refresh(self, request: object) -> None: ...


GEMINI_LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
VERTEX_LIVE_PATH = "ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
DEFAULT_LIVE_MODEL = "models/gemini-live-2.5-flash-native-audio"
DEFAULT_VERTEX_LIVE_MODEL = "gemini-live-2.5-flash-native-audio"

# Vertex regions where the Live BidiGenerateContent endpoint is reachable.
# Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
# (regions listed under "Model availability" for gemini-live-* models).
SUPPORTED_VERTEX_LIVE_REGIONS: frozenset[str] = frozenset(
    {
        # United States
        "us-central1",
        "us-east1",
        "us-east4",
        "us-east5",
        "us-south1",
        "us-west1",
        "us-west4",
        # Europe
        "europe-central2",
        "europe-north1",
        "europe-southwest1",
        "europe-west1",
        "europe-west4",
        "europe-west8",
    }
)

# Per-geo preference order — first item is the "canonical" Live region for
# the zone.  Used by :func:`_nearest_vertex_live_region` when the caller's
# requested location isn't directly on the supported list.
_GEO_PREFERENCES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("europe", "africa", "me-"),
        (
            "europe-west4",
            "europe-west1",
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west8",
        ),
    ),
    (
        ("us", "northamerica", "southamerica"),
        ("us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west4"),
    ),
    # Asia / Australia have no Live region today; fall through to us-central1.
)


def _nearest_vertex_live_region(location: str) -> str:
    """Map an arbitrary GCP region to the closest Live-supported region.

    Pass-through when the requested region already supports Live; otherwise
    pick the first supported region in the same geo zone (e.g. ``asia-east1``
    or ``global`` → ``us-central1``; an unsupported European region →
    ``europe-west4``).
    """
    if location in SUPPORTED_VERTEX_LIVE_REGIONS:
        return location
    for prefixes, candidates in _GEO_PREFERENCES:
        if any(location.startswith(p) for p in prefixes):
            for cand in candidates:
                if cand in SUPPORTED_VERTEX_LIVE_REGIONS:
                    return cand
    return "us-central1"


async def probe_nearest_live_region(
    candidates: tuple[str, ...] | frozenset[str] = SUPPORTED_VERTEX_LIVE_REGIONS,
    *,
    timeout: float = 2.0,
) -> str:
    """Probe each candidate Vertex Live region and return the fastest.

    Hits the regional ``aiplatform.googleapis.com`` HTTPS endpoint with a
    HEAD request.  TLS handshake + first byte is a fair approximation of
    WebSocket connect latency from this network.  Probes run in parallel
    so the worst-case startup cost is ``timeout`` seconds, not the sum.

    Falls back to the geo-prefix lookup against ``GOOGLE_CLOUD_LOCATION``
    (or "us-central1" if that's also unset) when every probe fails.
    """
    import asyncio
    import time

    async def _probe(region: str) -> tuple[str, float]:
        url = f"https://{region}-aiplatform.googleapis.com/"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                t0 = time.monotonic()
                async with session.head(url, allow_redirects=False) as resp:
                    # Just consume the headers — we don't care about the
                    # status code, only that the TLS+HTTP round-trip
                    # completed.
                    _ = resp.status
                return region, time.monotonic() - t0
        except (TimeoutError, aiohttp.ClientError):
            return region, float("inf")

    results = await asyncio.gather(*(_probe(r) for r in candidates))
    reachable = [(r, t) for r, t in results if t != float("inf")]
    if not reachable:
        fallback = _nearest_vertex_live_region(os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1")
        logger.warning("All Vertex Live region probes failed; falling back to %s", fallback)
        return fallback
    reachable.sort(key=lambda r: r[1])
    best, best_latency = reachable[0]
    logger.info(
        "Vertex Live region probe: %s @ %.0f ms (probed %d/%d in parallel)",
        best,
        best_latency * 1000,
        len(reachable),
        len(results),
    )
    return best


# BCP-47 codes the Gemini Live speechConfig.languageCode parameter accepts.
# Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice
SUPPORTED_LIVE_LANGUAGES: frozenset[str] = frozenset(
    {
        "ar-XA",
        "bn-IN",
        "cmn-CN",
        "de-DE",
        "en-AU",
        "en-GB",
        "en-IN",
        "en-US",
        "es-ES",
        "es-US",
        "fr-CA",
        "fr-FR",
        "gu-IN",
        "hi-IN",
        "id-ID",
        "it-IT",
        "ja-JP",
        "kn-IN",
        "ko-KR",
        "ml-IN",
        "mr-IN",
        "nl-NL",
        "pl-PL",
        "pt-BR",
        "ru-RU",
        "ta-IN",
        "te-IN",
        "th-TH",
        "tr-TR",
        "vi-VN",
    }
)


def _api_key_from_env() -> str:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def detect_system_language() -> str | None:
    """Best-effort detection of the user's preferred BCP-47 locale.

    Returns ``None`` if the system locale isn't on the Gemini Live language
    list, so callers can leave the server's default (en-US) in place rather
    than send an unsupported value.
    """
    import locale

    candidates: list[str] = []
    try:
        loc, _ = locale.getlocale()
        if loc:
            candidates.append(loc)
    except (ValueError, TypeError):
        pass
    for env_key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(env_key)
        if value:
            candidates.append(value.split(".")[0])
    for raw in candidates:
        bcp47 = raw.replace("_", "-").split("@")[0]
        if bcp47 in SUPPORTED_LIVE_LANGUAGES:
            return bcp47
    return None


def _convert_realtime_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio ``Tool``s to Gemini Live ``tools`` config."""
    if not tools:
        return []
    return [
        {
            "functionDeclarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
                for t in tools
            ]
        }
    ]


@dataclass(slots=True)
class GeminiLiveSession(RealtimeSession):
    """RealtimeSession backed by an aiohttp WebSocket against Gemini Live.

    ``ws`` is any object that quacks like ``aiohttp.ClientWebSocketResponse``;
    tests inject a stub.
    """

    ws: Any
    output_audio_media_type: str = "audio/pcm;rate=24000"
    http_session: aiohttp.ClientSession | None = field(default=None, repr=False)
    own_http_session: bool = False
    _next_tool_index: int = field(default=0, init=False, repr=False)
    _turn_used_tools: bool = field(default=False, init=False, repr=False)

    async def _send_event(self, payload: dict[str, Any]) -> None:
        await self.ws.send_str(json.dumps(payload))

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            match block:
                case AudioBlock(data=data, media_type=mt):
                    # Gemini's realtimeInput.audio expects a Blob {data, mimeType}
                    # with PCM16 mono at 16 kHz.
                    await self._send_event(
                        {
                            "realtimeInput": {
                                "audio": {
                                    "data": base64.b64encode(data).decode(),
                                    "mimeType": _normalise_input_mime(mt),
                                }
                            }
                        }
                    )
                case TextBlock(text=text):
                    await self._send_event(
                        {
                            "clientContent": {
                                "turns": [
                                    {
                                        "role": "user",
                                        "parts": [{"text": text}],
                                    }
                                ],
                                "turnComplete": True,
                            }
                        }
                    )
                case _:
                    raise TypeError(f"Gemini Live: send() does not support {type(block).__name__}")

    async def commit(self) -> None:
        # Gemini Live commits user audio when audioStreamEnd is sent.  Server VAD
        # will normally trigger turn completion automatically; this is a manual
        # nudge for clients running with VAD off.
        await self._send_event({"realtimeInput": {"audioStreamEnd": True}})

    async def interrupt(self) -> None:
        # Gemini doesn't have an explicit cancel; signalling activityEnd lets
        # the server stop generating in non-VAD mode.  In server VAD mode
        # speaking again interrupts naturally.
        await self._send_event({"realtimeInput": {"activityEnd": {}}})

    async def send_tool_result(
        self, tool_use_id: ToolCallID, name: ToolName, content: str | list[ContentBlock]
    ) -> None:
        if isinstance(content, list):
            parts: list[str] = []
            for b in content:
                if isinstance(b, TextBlock):
                    parts.append(b.text)
                else:
                    raise TypeError(f"Gemini Live tool result only supports str / TextBlock, got {type(b).__name__}")
            output = "".join(parts)
        else:
            output = content
        await self._send_event(
            {
                "toolResponse": {
                    "functionResponses": [
                        {
                            "id": tool_use_id,
                            "name": name,
                            "response": {"output": output},
                        }
                    ]
                }
            }
        )

    async def events(self) -> AsyncIterator[StreamEvent]:
        async for msg in self.ws:
            mtype = getattr(msg, "type", None)
            if mtype is aiohttp.WSMsgType.TEXT:
                payload = json.loads(msg.data)
            elif mtype is aiohttp.WSMsgType.BINARY:
                payload = json.loads(msg.data.decode("utf-8"))
            elif mtype is aiohttp.WSMsgType.ERROR:
                logger.error("Gemini Live ws error: %s", self.ws.exception())
                break
            elif mtype in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                break
            else:
                continue
            for ev in self._translate(payload):
                yield ev

    def _translate(self, msg: dict[str, Any]) -> list[StreamEvent]:
        out: list[StreamEvent] = []
        if "setupComplete" in msg:
            return out
        if "serverContent" in msg:
            sc = msg["serverContent"]
            model_turn = sc.get("modelTurn")
            if model_turn:
                for part in model_turn.get("parts") or []:
                    inline = part.get("inlineData")
                    if inline and inline.get("mimeType", "").startswith("audio/pcm"):
                        out.append(
                            AudioOutputDelta(
                                data=base64.b64decode(inline["data"]),
                                media_type=inline.get("mimeType", self.output_audio_media_type),
                            )
                        )
                    elif "text" in part:
                        out.append(TextDelta(index=0, delta=part["text"]))
            it = sc.get("inputTranscription")
            if it and "text" in it:
                out.append(TranscriptDelta(role="user", delta=it["text"]))
            ot = sc.get("outputTranscription")
            if ot and "text" in ot:
                out.append(TranscriptDelta(role="assistant", delta=ot["text"]))
            if sc.get("interrupted"):
                # Server detected user speech and stopped generation — signal
                # the consumer so they can drop buffered audio.
                out.append(SpeechStarted())
            if sc.get("turnComplete"):
                stop_reason = StopReason.tool_use if self._turn_used_tools else StopReason.end_turn
                usage = _build_usage(msg.get("usageMetadata"))
                out.append(TurnComplete(stop_reason=stop_reason, usage=usage))
                self._turn_used_tools = False
        elif "toolCall" in msg:
            for fc in msg["toolCall"].get("functionCalls") or []:
                idx = self._next_tool_index
                self._next_tool_index += 1
                call_id = fc.get("id") or ""
                fname = fc["name"]
                out.append(ToolUseStart(index=idx, tool_use_id=call_id, name=fname))
                # Gemini delivers args atomically.  Emit one ToolInputDelta with
                # the serialized JSON, so downstream consumers can use the same
                # ToolInputDelta → ToolUseBlock pipeline.
                out.append(
                    ToolInputDelta(
                        index=idx,
                        tool_use_id=call_id,
                        partial_json=json.dumps(fc.get("args") or {}),
                    )
                )
                self._turn_used_tools = True
        elif "toolCallCancellation" in msg:
            ids = msg["toolCallCancellation"].get("ids") or []
            logger.info("Gemini Live cancelled tool calls: %s", ids)
        elif "goAway" in msg:
            logger.warning("Gemini Live goAway: %s", msg["goAway"])
        else:
            logger.debug("Gemini Live: ignoring message keys=%s", list(msg.keys()))
        return out

    async def close(self) -> None:
        if not self.ws.closed:
            await self.ws.close()
        if self.own_http_session and self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()


def _build_usage(meta: dict[str, Any] | None) -> Usage | None:
    if not meta:
        return None
    return Usage(
        input_tokens=int(meta.get("promptTokenCount", 0) or 0),
        output_tokens=int(meta.get("responseTokenCount", meta.get("candidatesTokenCount", 0)) or 0),
    )


def _normalise_input_mime(mt: str) -> str:
    """Coerce axio AudioBlock media types to a Gemini-compatible mime string.

    Gemini Live expects ``audio/pcm;rate=<sample_rate>`` with a 16 kHz default
    for input audio.  ``audio/pcm`` (no rate) is accepted as a 16 kHz stream.
    """
    return mt if mt.startswith("audio/pcm") else "audio/pcm;rate=16000"


@dataclass(slots=True)
class GeminiLiveTransport(RealtimeTransport):
    """RealtimeTransport for Gemini Live.

    Supports both backends:

    * **AI Studio / developer API** (default) — auth via ``GEMINI_API_KEY``
      query param, model id like ``"models/gemini-live-2.5-flash-native-audio"``.
    * **Vertex AI** — set ``vertexai=True`` (or env ``GOOGLE_GENAI_USE_VERTEXAI=1``).
      Requires ``project`` and ``location``.  Auth uses
      ``google.auth.default`` (gcloud user creds, service account, GCE
      metadata, etc.).  The full model resource path is built automatically.

    ``language_code`` is forwarded to ``generationConfig.speechConfig.languageCode``
    so the model picks the right TTS voice for the user's locale.
    """

    api_key: str = field(default_factory=_api_key_from_env)
    base_url: str | None = field(default=None)
    model: str | None = None
    http_session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)

    vertexai: bool = field(
        default_factory=lambda: os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")
    )
    project: str | None = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT"))
    location: str | None = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1")
    language_code: str | None = None
    """BCP-47 code passed to ``generationConfig.speechConfig.languageCode``.
    NOTE: this only steers the **TTS voice** the server uses for the audio
    output — it does *not* tell the model which language to think / write
    in.  Most callers will also want to inject a language instruction into
    the system prompt (chat.py does this automatically).
    """

    auto_region: bool = False
    """When True (Vertex only), probe every supported Live region at connect
    time and pick the lowest-latency one for THIS network — overrides
    ``location``.  Adds ~1 s of startup latency.  The result is cached for
    the lifetime of the transport instance."""

    _credentials: Any = field(default=None, init=False, repr=False)
    _probed_region: str | None = field(default=None, init=False, repr=False)

    async def _get_vertex_token(self) -> str:
        import google.auth
        import google.auth.transport.urllib3
        import urllib3

        if self._credentials is None:
            credentials, _ = await asyncio.to_thread(
                google.auth.default,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._credentials = credentials
        creds = cast(_RefreshableCredentials, self._credentials)
        if creds.valid and not creds.expired:
            if not creds.token:
                raise RuntimeError("Google credentials did not return an access token")
            return creds.token
        request_factory = cast(Any, google.auth.transport.urllib3.Request)
        await asyncio.to_thread(creds.refresh, request_factory(urllib3.PoolManager()))
        if not creds.token:
            raise RuntimeError("Google credentials did not return an access token")
        return creds.token

    def _resolve_url_and_model(self) -> tuple[str, str]:
        if self.vertexai:
            if not self.project:
                raise RuntimeError("vertexai=True requires `project` (or env GOOGLE_CLOUD_PROJECT).")
            # Vertex Live is region-specific.  The global endpoint
            # (`aiplatform.googleapis.com` with no prefix) returns 404 for
            # Live BidiGenerateContent.  Most regions don't yet expose
            # the BidiGenerateContent service.  Coerce the requested location
            # to the closest known-supported Live region so the demo just
            # works instead of erroring on the handshake.
            requested = self.location or "us-central1"
            location = _nearest_vertex_live_region(requested)
            if location != requested:
                logger.info(
                    "Vertex Live is not available in `%s`; routing to `%s` instead.",
                    requested,
                    location,
                )
            base = self.base_url or f"wss://{location}-aiplatform.googleapis.com/{VERTEX_LIVE_PATH}"
            model_id = self.model or DEFAULT_VERTEX_LIVE_MODEL
            full_model = (
                f"projects/{self.project}/locations/{location}/publishers/google/models/{model_id}"
                if "/" not in model_id
                else model_id
            )
            return base, full_model
        base = self.base_url or os.environ.get("GEMINI_LIVE_URL", GEMINI_LIVE_URL)
        model_id = self.model or DEFAULT_LIVE_MODEL
        return base, model_id

    async def connect(
        self,
        *,
        system: str,
        tools: list[Tool[Any]],
        voice: str | None = None,
        input_audio_format: str = "audio/pcm;rate=16000",
        output_audio_format: str = "audio/pcm;rate=24000",
    ) -> RealtimeSession:
        own_session = False
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession()
            own_session = True
        if self.vertexai and self.auto_region and self._probed_region is None:
            self._probed_region = await probe_nearest_live_region()
            self.location = self._probed_region
        base_url, model_path = self._resolve_url_and_model()
        if self.vertexai:
            token = await self._get_vertex_token()
            headers = {"Authorization": f"Bearer {token}"}
            if self.project:
                headers["x-goog-user-project"] = self.project
            ws = await self.http_session.ws_connect(base_url, headers=headers)
        else:
            ws = await self.http_session.ws_connect(f"{base_url}?key={self.api_key}")

        generation_config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        speech_config: dict[str, Any] = {}
        if voice:
            speech_config["voiceConfig"] = {"prebuiltVoiceConfig": {"voiceName": voice}}
        if self.language_code:
            speech_config["languageCode"] = self.language_code
        if speech_config:
            generation_config["speechConfig"] = speech_config

        setup: dict[str, Any] = {
            "model": model_path,
            "generationConfig": generation_config,
            "systemInstruction": {"parts": [{"text": system}]},
        }
        gemini_tools = _convert_realtime_tools(tools)
        if gemini_tools:
            setup["tools"] = gemini_tools
        await ws.send_str(json.dumps({"setup": setup}))
        return GeminiLiveSession(
            ws=ws,
            output_audio_media_type=output_audio_format,
            http_session=self.http_session,
            own_http_session=own_session,
        )


@dataclass(slots=True)
class VertexLiveTransport(GeminiLiveTransport):
    """GeminiLiveTransport pre-configured for Vertex AI (mirrors VertexAITransport)."""

    vertexai: bool = True
