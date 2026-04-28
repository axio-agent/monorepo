"""axio — public API."""

from .agent import Agent
from .blocks import TextBlock, ToolResultBlock, ToolUseBlock
from .context import ContextStore, MemoryContextStore
from .events import IterationEnd, StreamEvent, TextDelta, ToolInputDelta, ToolResult, ToolUseStart
from .exceptions import GuardError, HandlerError
from .field import Field, FieldInfo, StrictStr
from .messages import Message
from .permission import ConcurrentGuard, PermissionGuard
from .selector import ToolSelector
from .stream import AgentStream
from .tool import CONTEXT, Tool
from .transport import CompletionTransport
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
    "IterationEnd",
    "ToolUseStart",
    "ToolInputDelta",
    "ToolResult",
    # messages & blocks
    "Message",
    "TextBlock",
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
