"""Exception hierarchy for axio."""


class AxioError(Exception):
    """Base exception for all axio errors."""


class ToolError(AxioError):
    """Base for tool-related errors."""


class GuardError(ToolError):
    """Guard denied or crashed during permission check."""


class HandlerError(ToolError):
    """Handler raised during execution."""


class StreamError(AxioError):
    """Error during stream collection."""
