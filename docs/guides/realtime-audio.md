# Realtime Audio

Build voice agents with `axio-audio` and `RealtimeAgent`. The audio package provides microphone
capture and speaker playback. `RealtimeAgent` (in the `axio` core) drives a `RealtimeTransport`
session, dispatching tool calls concurrently with streaming audio output.

## Install

```bash
pip install axio-audio axio-transport-google
```

`axio-audio` depends on `sounddevice` and `numpy`.

## Architecture

```
┌──────────────┐    PCM16 audio    ┌────────────────────────┐
│  Microphone  │──────────────────▶│                        │
└──────────────┘                   │     RealtimeAgent      │
                                   │  (axio core)           │
┌──────────────┐    PCM16 audio    │                        │
│   Speaker    │◀──────────────────│                        │
└──────────────┘                   └────────────┬───────────┘
                                                │  WebSocket
                                   ┌────────────▼───────────┐
                                   │  GeminiLiveTransport   │
                                   │  (Gemini Live API)     │
                                   └────────────────────────┘
```

The session is full-duplex: you send raw PCM16 chunks from the microphone and
receive `AudioOutputDelta` events to feed to the speaker.

## Minimal example

```python
import asyncio
from axio_audio import Microphone, Speaker
from axio_transport_google.realtime import GeminiLiveTransport
from axio.realtime import RealtimeAgent
from axio.events import AudioOutputDelta, TurnComplete


async def main() -> None:
    transport = GeminiLiveTransport()

    async with (
        RealtimeAgent(system="You are a helpful assistant.", transport=transport) as agent,
        Microphone() as mic,
        Speaker() as spk,
    ):
        async def send_mic() -> None:
            async for chunk in mic:
                await agent.send(chunk)

        async def play_output() -> None:
            async for ev in agent.events():
                if isinstance(ev, AudioOutputDelta):
                    await spk.feed(ev.data)

        await asyncio.gather(send_mic(), play_output())


asyncio.run(main())
```

## Microphone

`Microphone` is an async-iterable that yields `AudioBlock` chunks of PCM16 mono
audio at a configurable sample rate.

```python
from axio_audio import Microphone

async with Microphone(sample_rate=24000, chunk_ms=50) as mic:
    async for chunk in mic:
        # chunk is an AudioBlock with PCM16 bytes
        await agent.send(chunk)
```

| Parameter | Default | Description |
|---|---|---|
| `sample_rate` | `24000` | Sample rate in Hz |
| `chunk_ms` | `50` | Chunk duration in milliseconds |
| `device` | `None` | sounddevice device index or name |
| `queue_maxsize` | `100` | Internal queue size (chunks) |

## Speaker

`Speaker` is an async-friendly PCM16 playback buffer. Feed it raw bytes; the
audio callback drains the buffer as the device requests samples.

```python
from axio_audio import Speaker

async with Speaker(sample_rate=24000) as spk:
    async for ev in agent.events():
        if isinstance(ev, AudioOutputDelta):
            await spk.feed(ev.data)
```

| Parameter | Default | Description |
|---|---|---|
| `sample_rate` | `24000` | Sample rate in Hz |
| `device` | `None` | sounddevice device index or name |
| `playback_tap` | `None` | Optional `Callable[[bytes], None]` for echo-cancel reference |

Call `spk.stop()` to immediately clear the playback buffer. Use it to honour
user interruptions, so the assistant goes silent right away.

## DuplexAudio

Independent `Microphone` and `Speaker` open separate PortAudio streams on
different host clocks. On consumer audio stacks (PipeWire, PulseAudio) those
clocks drift by ~10-25 ms/sec, which destroys echo canceller quality.

`DuplexAudio` opens a **single** `sd.RawStream` so mic and speaker share one
PortAudio clock. Use it when echo cancellation matters.

```python
from axio_audio import DuplexAudio
from axio.events import AudioOutputDelta

async with DuplexAudio(sample_rate=48000, chunk_ms=20) as duplex:
    async def consume_mic() -> None:
        async for chunk in duplex.mic_chunks():
            await agent.send(chunk)

    async def play_output() -> None:
        async for ev in agent.events():
            if isinstance(ev, AudioOutputDelta):
                await duplex.feed_speaker(ev.data)

    await asyncio.gather(consume_mic(), play_output())
```

`DuplexAudio` also exposes `.mic` and `.speaker` properties. They satisfy
the same async context-manager interface as `Microphone` / `Speaker`, so you
can swap them with minimal changes to existing code.

| Parameter | Default | Description |
|---|---|---|
| `sample_rate` | `48000` | Sample rate in Hz |
| `chunk_ms` | `20` | Chunk duration in milliseconds |
| `device` | `None` | Device index, name, or `(input, output)` tuple |
| `mono_io` | `True` | Expose a mono API even on multi-channel devices |
| `queue_maxsize` | `100` | Mic chunk queue size |
| `playback_tap` | `None` | Optional `Callable[[bytes], None]` for echo-cancel reference |

## RealtimeAgent

`RealtimeAgent` (in the `axio` core) is the duplex counterpart to `Agent`. It
drives a `RealtimeSession` and dispatches tool calls as background tasks so
audio output is never blocked by slow tools.

```python
from axio.realtime import RealtimeAgent
from axio import Tool

agent = RealtimeAgent(
    system="You are a helpful assistant.",
    transport=transport,
    tools=[Tool(name="my_tool", handler=my_handler)],
    voice="Aoede",
    input_audio_format="audio/pcm;rate=16000",
    output_audio_format="audio/pcm;rate=24000",
)
```

| Parameter | Default | Description |
|---|---|---|
| `system` | required | System prompt |
| `transport` | required | A `RealtimeTransport` (e.g. `GeminiLiveTransport`) |
| `tools` | `[]` | Tools available to the model |
| `voice` | `None` | Voice name (transport-specific) |
| `input_audio_format` | `"audio/pcm;rate=16000"` | Audio format sent to the model |
| `output_audio_format` | `"audio/pcm;rate=24000"` | Audio format received from the model |
| `raise_on_error` | `True` | Re-raise exceptions from `Error` events; set `False` to handle them in-loop |

## Handling interruptions

When the user starts speaking while the assistant is talking, the model signals
a `SpeechStarted` event. Call `agent.interrupt()` to cancel in-flight tool tasks
and tell the session to stop generating:

```python
from axio.events import AudioOutputDelta, SpeechStarted

async for ev in agent.events():
    match ev:
        case SpeechStarted():
            spk.stop()                 # clear speaker buffer
            await agent.interrupt()    # cancel tools, stop generation
        case AudioOutputDelta(data=pcm):
            spk.feed(pcm)
```

## Audio format notes

- PCM16 = signed 16-bit integer, little-endian, mono
- Gemini Live default input: `audio/pcm;rate=16000`
- Gemini Live default output: `audio/pcm;rate=24000`
- `DuplexAudio` defaults to 48 kHz (higher quality, resampled by sounddevice)

## GeminiLiveTransport parameters

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `GEMINI_API_KEY`, then `GOOGLE_API_KEY` | Developer API key |
| `model` | `None`, resolved to `models/gemini-live-2.5-flash-native-audio` | Model for live sessions |
| `vertexai` | `GOOGLE_GENAI_USE_VERTEXAI` env var | Route through Vertex AI |
| `language_code` | `None` | BCP-47 language code (e.g. `"en-US"`) |

The live models are their own family, not the chat models. `VertexLiveTransport` resolves an
unset `model` to `gemini-live-2.5-flash-native-audio`, and expands it to a full publisher path.

30 languages are supported, including Arabic, French, German, Hindi, Japanese,
Korean, Portuguese, Spanish, and Vietnamese.

For Vertex AI, use `VertexLiveTransport` and optionally call
`probe_nearest_live_region()` to pick the fastest available region:

```python
from axio_transport_google.realtime import VertexLiveTransport, probe_nearest_live_region

region = await probe_nearest_live_region()
transport = VertexLiveTransport(location=region, auto_region=True)
```

## Examples

The repository includes two complete examples:

- [`examples/realtime_smoke/`](https://github.com/mosquito/axio-agent/tree/master/examples/realtime_smoke) -
  minimal smoke test: sends a text message, receives audio, plays it
- [`examples/realtime_chat/`](https://github.com/mosquito/axio-agent/tree/master/examples/realtime_chat) -
  full-featured voice chat with echo cancellation, volume metering, interruption
  handling, and Playwright-based screenshot tool
