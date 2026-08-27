# Multimodal

Axio agents can send and receive images, audio, and video alongside text.
Content is represented as typed blocks throughout the stack - from tool results
to conversation history to LLM responses.

## Content blocks

All content is a block. Four of them carry media or text a message is made of:

| Block | Media types |
|---|---|
| `TextBlock` | - |
| `ImageBlock` | `image/jpeg`, `image/png`, `image/gif`, `image/webp` |
| `AudioBlock` | `audio/x-aac`, `audio/flac`, `audio/mp3`, `audio/m4a`, `audio/mpeg`, `audio/mpga`, `audio/mp4`, `audio/ogg`, `audio/pcm`, `audio/wav`, `audio/webm` |
| `VideoBlock` | `video/mp4`, `video/mpeg`, `video/mov`, `video/avi`, `video/x-flv`, `video/mpg`, `video/webm`, `video/wmv`, `video/3gpp` |

```python
from axio.blocks import TextBlock, ImageBlock, AudioBlock, VideoBlock
```

The three media blocks are frozen dataclasses with two fields, `media_type` and `data: bytes`.
`TextBlock` has `text`, and `signature` for the providers that sign the answer itself.

Three more `ContentBlock` types complete the set and carry no media: `ToolUseBlock`,
`ToolResultBlock`, and `ReasoningBlock`. `ReasoningBlock` holds the model's own reasoning,
stored with the provider's signature so the turn can be replayed unaltered. A stored assistant
turn can hold any of them, so code that walks `Message.content` must not assume it holds only
text.

```python
from axio.blocks import ReasoningBlock, ToolResultBlock, ToolUseBlock
```

## Sending images to the agent

`agent.run()` accepts a plain string as the user message. To include images or
other media, append a `Message` with the appropriate blocks to the context
**before** calling `run()`:

<!--
name: test_multimodal_send_image
```python
import builtins as _b
import io
_real_open = _b.open
_b.open = lambda p, m="r", **kw: (
    io.BytesIO(b"fake_png") if "screenshot.png" in str(p) and "b" in m
    else _real_open(p, m, **kw)
)
```
-->
<!-- name: test_multimodal_send_image -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio.messages import Message
from axio.blocks import TextBlock, ImageBlock
from axio.testing import StubTransport, make_text_response


async def main() -> None:
    image_data = open("screenshot.png", "rb").read()

    context = MemoryContextStore()
    await context.append(Message(
        role="user",
        content=[
            TextBlock(text="What is shown in this screenshot?"),
            ImageBlock(media_type="image/png", data=image_data),
        ],
    ))

    transport = StubTransport([make_text_response("A terminal window.")])
    agent = Agent(system="You are a helpful visual assistant.", tools=[], transport=transport)
    reply = await agent.run("Describe it in detail.", context)
    assert reply == "A terminal window."


asyncio.run(main())
```

The `run()` call appends its string argument as an additional user message.
The LLM therefore sees the image message followed by the "Describe it"
message, just as if two separate turns happened.

## Reading media files as tools

`read_file` from `axio-tools-local` detects file extensions and returns
multimodal blocks automatically - no extra configuration needed:

| Extension | Returned block |
|---|---|
| `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` | `ImageBlock` |
| `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.aac`, `.pcm` | `AudioBlock` |
| `.mp4`, `.webm`, `.mov`, `.avi`, `.mpg`, `.wmv`, `.3gp` | `VideoBlock` |

For a text file, `read_file` returns a plain `str`. For a media file it
returns `[TextBlock(text="...metadata..."), <MediaBlock>]`. The transport
forwards the blocks to the LLM as native multimodal content.

A vision-capable model (Gemini, Claude, GPT-4o) receiving `read_file` results
for images can describe, compare, and reason about the pixel content directly.

## Tools returning multimodal content

A tool handler can return a list of blocks to pass rich content back to the
model. This is how `read_file` works internally:

<!-- name: test_multimodal_tool_result -->
```python
import asyncio
from axio import Agent, MemoryContextStore, Tool
from axio.blocks import TextBlock, ImageBlock
from axio.testing import StubTransport, make_tool_use_response, make_text_response


async def capture_chart() -> list[TextBlock | ImageBlock]:
    """Capture the current chart as an image."""
    chart_bytes = b"\x89PNG..."   # real implementation would render a chart
    return [
        TextBlock(text="Chart captured."),
        ImageBlock(media_type="image/png", data=chart_bytes),
    ]


async def main() -> None:
    transport = StubTransport([
        make_tool_use_response("capture_chart", "t1", {}),
        make_text_response("The chart shows an upward trend."),
    ])
    agent = Agent(
        system="You are a data analyst.",
        tools=[Tool(name="capture_chart", handler=capture_chart)],
        transport=transport,
    )
    reply = await agent.run("Describe the current chart.", MemoryContextStore())
    assert reply == "The chart shows an upward trend."


asyncio.run(main())
```

## Realtime audio

For low-latency voice agents that stream raw PCM audio in both directions, use
`RealtimeAgent` with `axio-audio`. See the {doc}`realtime-audio` guide.

## Transport support

Not every transport supports every modality. Check the model's declared
capabilities before sending:

| Capability | Meaning |
|---|---|
| `vision` | Accepts `ImageBlock` |
| `audio` | Accepts `AudioBlock` |
| `video` | Accepts `VideoBlock` |
| `image_generation` | Can produce images |
| `video_generation` | Can produce videos |

```python
from axio.models import Capability

model = getattr(transport, "model", None)
caps = getattr(model, "capabilities", frozenset())
if Capability.vision in caps:
    print("This model can see images")
```

See the {doc}`../concepts/models` reference for the full capabilities list.
