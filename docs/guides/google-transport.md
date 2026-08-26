# Google Transport

`axio-transport-google` provides a Gemini transport for both standard
completion and realtime (Gemini Live) sessions. It supports the Google
GenAI Developer API and Vertex AI.

## Install

```bash
pip install axio-transport-google
```

For Vertex AI with application-default credentials:

```bash
pip install "axio-transport-google[vertexai]"
```

## Quick start

Set the API key and create a transport:

```bash
export GEMINI_API_KEY="..."
```

```python
from axio_transport_google import GoogleTransport

transport = GoogleTransport()
```

The transport auto-selects `gemini-3.1-pro-preview` as the default model.

## Models

| Model ID | Capabilities | Context | Notes |
|---|---|---|---|
| `gemini-3.1-pro-preview` | text, vision, audio, video, tools, reasoning | 1M tokens | Flagship |
| `gemini-3-flash-preview` | text, vision, audio, video, tools, reasoning | 1M tokens | Fast/cheap |
| `gemini-3.1-flash-lite-preview` | text, vision, audio, video, tools, reasoning | 1M tokens | Lightest |
| `gemini-3.1-flash-image-preview` | text, vision, image generation | 1M tokens | Nano Banana |
| `gemini-3-pro-image-preview` | text, vision, image generation | 1M tokens | Image gen |

## Switching models

```python
from axio_transport_google import GoogleTransport
from axio.models import Capability

transport = GoogleTransport()

# Switch to a specific model
transport.model = transport.models["gemini-3-flash-preview"]

# Find the cheapest reasoning model
transport.model = (
    transport.models
    .by_capability(Capability.reasoning)
    .by_cost()
    .first()
)
```

## Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `GEMINI_API_KEY` env var | API key for the Developer API |
| `model` | `gemini-3.1-pro-preview` | Active `ModelSpec` |
| `temperature` | `None` | Sampling temperature (uses model default if unset) |
| `top_p` | `None` | Nucleus sampling probability |
| `top_k` | `None` | Top-k sampling |
| `seed` | `None` | Random seed for deterministic outputs |
| `thinking_budget` | `None` | Token budget for chain-of-thought reasoning |
| `thinking_level` | `None` | Thinking level: `"LOW"`, `"MEDIUM"`, `"HIGH"`, or `"NONE"` |
| `max_output_tokens` | `None` | Override the model's default max output |
| `max_retries` | `5` | Retries on 429/503 with exponential backoff |
| `safety_settings` | `None` | List of `SafetySettingDict` (see below) |
| `debug` | `False` | Log raw request/response bodies |

## Vertex AI

Use `VertexAITransport` to route through Google Cloud Vertex AI instead of
the Developer API. It reads credentials from application-default credentials
(`gcloud auth application-default login`).

```python
from axio_transport_google import VertexAITransport

transport = VertexAITransport(
    project="my-gcp-project",
    location="us-central1",
)
```

Or set environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="1"
```

On Vertex AI you can also use Anthropic models with a `anthropic/` prefix:

```python
transport.model = transport.models["anthropic/claude-opus-4-6"]
```

## Safety settings

Override the default safety thresholds:

```python
from axio_transport_google import GoogleTransport
from axio_transport_google._generated_types import SafetySetting

transport = GoogleTransport(
    safety_settings=[
        SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]
)
```

## Image generation

When the selected model supports `Capability.image_generation`, the transport
exposes `generate_images`:

```python
import asyncio
from axio_transport_google import GoogleTransport


async def main() -> None:
    transport = GoogleTransport()
    images: list[bytes] = await transport.generate_images(
        "A photorealistic owl sitting on a branch",
        model="gemini-3.1-flash-image-preview",
        n=1,
    )
    with open("owl.png", "wb") as f:
        f.write(images[0])


asyncio.run(main())
```

## Video generation

```python
import asyncio
from axio_transport_google import GoogleTransport


async def main() -> None:
    transport = GoogleTransport()
    videos: list[bytes] = await transport.generate_videos(
        "Time-lapse of clouds moving over mountains",
        model="veo-3.1-fast-generate-001",
        duration_seconds=6,
        aspect_ratio="16:9",
    )
    with open("timelapse.mp4", "wb") as f:
        f.write(videos[0])


asyncio.run(main())
```

Video generation runs an async polling loop until the job completes.

## Tools registered as entry points

When installed, `axio-transport-google` registers two tools under `axio.tools`:

| Entry point | Tool | Description |
|---|---|---|
| `generate_image` | `generate_image` | Generate images via Gemini Nano Banana |
| `generate_video` | `generate_video` | Generate videos via Veo |

Pass these handlers to `Tool` explicitly, or use them through `axio-repl`.

## Realtime (Gemini Live)

For low-latency voice conversations, use `GeminiLiveTransport` with
`RealtimeAgent`. See the {doc}`realtime-audio` guide for the full setup.

```python
from axio_transport_google.realtime import GeminiLiveTransport
from axio.realtime import RealtimeAgent

transport = GeminiLiveTransport()

async with RealtimeAgent(system="You are a helpful assistant.", transport=transport) as agent:
    ...
```

For Vertex AI realtime, use `VertexLiveTransport`. If you have multiple Vertex
regions available, the transport can auto-select the nearest one:

```python
from axio_transport_google.realtime import VertexLiveTransport, probe_nearest_live_region

region = await probe_nearest_live_region()
transport = VertexLiveTransport(location=region)
```
