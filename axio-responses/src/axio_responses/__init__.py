"""The OpenAI Responses API as axio speaks it.

``convert_messages`` and ``convert_tools`` build the request. ``Responses`` reads the stream it
answers with. Both halves are here rather than in a transport because two transports speak this
API: the public ``/v1/responses`` endpoint and the ChatGPT backend Codex uses.
"""

from __future__ import annotations

from .reader import (
    Annotation,
    AnnotationAdded,
    AnnotationSource,
    ArgumentsDelta,
    ArgumentsDone,
    Completed,
    ContentPartDone,
    Created,
    Failed,
    Incomplete,
    IncompleteDetails,
    InputDetails,
    ItemAdded,
    ItemDone,
    OutputDetails,
    OutputItem,
    ReasoningDeltaEvent,
    RefusalDeltaEvent,
    ResponseError,
    ResponseObject,
    Responses,
    ResponseUsage,
    StreamFailure,
    TextDeltaEvent,
)
from .request import STOP_REASONS, convert_messages, convert_tools, tool_output

__all__ = [
    "STOP_REASONS",
    "Annotation",
    "AnnotationAdded",
    "AnnotationSource",
    "ArgumentsDelta",
    "ArgumentsDone",
    "ContentPartDone",
    "Completed",
    "Created",
    "Failed",
    "Incomplete",
    "IncompleteDetails",
    "InputDetails",
    "ItemAdded",
    "ItemDone",
    "OutputDetails",
    "OutputItem",
    "ReasoningDeltaEvent",
    "RefusalDeltaEvent",
    "ResponseError",
    "ResponseObject",
    "ResponseUsage",
    "Responses",
    "StreamFailure",
    "TextDeltaEvent",
    "convert_messages",
    "convert_tools",
    "tool_output",
]
