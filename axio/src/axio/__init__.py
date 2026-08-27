"""axio - public API."""

from .agent import Agent
from .blocks import ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock
from .context import ContextStore, MemoryContextStore
from .events import (
    AudioOutputDelta,
    BlockEnd,
    Citation,
    IterationEnd,
    IterationStart,
    ProviderEvent,
    ReasoningSignature,
    Refusal,
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from .exceptions import GuardError, HandlerError
from .field import Field, FieldInfo, StrictStr
from .messages import Message
from .permission import ConcurrentGuard, PermissionGuard
from .realtime import RealtimeAgent
from .selector import ToolSelector
from .stream import AgentStream
from .tool import CONTEXT, Tool
from .transport import CompletionTransport, RealtimeSession, RealtimeTransport
from .types import StopReason, Usage

__all__ = [
    # core
    "Agent",
    "Tool",
    "CONTEXT",
    "ContextStore",
    "MemoryContextStore",
    "CompletionTransport",
    # events
    "StreamEvent",
    "TextDelta",
    "Refusal",
    "Citation",
    "ReasoningSignature",
    "BlockEnd",
    "IterationStart",
    "IterationEnd",
    "ProviderEvent",
    "ToolUseStart",
    "ToolInputDelta",
    "ToolResult",
    # realtime
    "RealtimeAgent",
    "RealtimeTransport",
    "RealtimeSession",
    "AudioOutputDelta",
    "TranscriptDelta",
    "SpeechStarted",
    "SpeechStopped",
    "TurnComplete",
    # messages & blocks
    "Message",
    "TextBlock",
    "ReasoningBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    # types & errors
    "StopReason",
    "Usage",
    "GuardError",
    "HandlerError",
    # permissions
    "PermissionGuard",
    "ConcurrentGuard",
    # field annotations
    "Field",
    "FieldInfo",
    "StrictStr",
    # advanced
    "ToolSelector",
    "AgentStream",
]
