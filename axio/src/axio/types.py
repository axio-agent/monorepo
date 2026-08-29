"""Primitive types: ToolName, ToolCallID, StopReason, Usage."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

type ToolName = str
type ToolCallID = str

logger = logging.getLogger(__name__)


class StopReason(StrEnum):
    """Why the provider stopped generating.

    Anything that is not ``tool_use`` or ``pause_turn`` ends the run. See the match in
    ``Agent._run_loop``, whose wildcard keeps a member added here from falling through into
    another paid iteration.
    """

    end_turn = "end_turn"
    tool_use = "tool_use"
    max_tokens = "max_tokens"
    error = "error"
    #: The model declined, or the provider blocked the turn. Not an error: the same prompt sent
    #: again will be declined again.
    refusal = "refusal"
    #: A server-side tool loop reached its iteration limit. Resumable: the provider expects the
    #: assistant content back so it can finish. This is the one reason that does not end the run.
    pause_turn = "pause_turn"
    #: The conversation outgrew the model's window. Truncated, like ``max_tokens``.
    context_window_exceeded = "context_window_exceeded"
    #: The caller or the provider stopped the turn before it finished.
    cancelled = "cancelled"
    #: The provider said something this vocabulary does not have. Terminal, and it vouches for
    #: nothing. Named rather than folded into one of the others, because each of those claims
    #: something the provider did not say: that the turn finished, that it was truncated, or that
    #: the transport broke. ``IterationEnd.raw`` carries the word itself.
    unknown = "unknown"
    #: Axio stopped the turn, because the model was repeating itself. The only reason here the
    #: provider did not give. Reported as ``end_turn``, a caller could not tell an answer the model
    #: finished from one cut off mid-word, which is the same objection this vocabulary raises
    #: against reading a truncated response as a whole one.
    repetition = "repetition"


def stop_reason_from(raw: str, table: Mapping[str, StopReason], *, provider: str) -> StopReason:
    """What a provider's own stop value means here, or ``unknown`` where the table does not say.

    There is no fifth answer to give. Folded into ``end_turn`` it claims the turn finished, into
    ``max_tokens`` that it was truncated, into ``error`` that the transport broke; raising throws
    away an answer the caller has already read. ``unknown`` says only what is true, and
    ``IterationEnd.raw`` carries the provider's own word for the caller to act on.
    """
    if (known := table.get(raw)) is not None:
        return known
    logger.warning("Unknown %s stop reason %r", provider, raw)
    return StopReason.unknown


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one provider request.

    The rule: ``input_tokens`` and ``output_tokens`` are always inclusive grand totals, and every
    other field is a disjoint slice of one of them::

        cache_read_tokens + cache_write_tokens  <=  input_tokens
        reasoning_tokens                        <=  output_tokens

    Providers disagree about whether their own headline number already contains the slices, and
    they disagree in opposite directions. Anthropic counts only the tokens after the last cache
    breakpoint, so its cache counts have to be added. Google reports thinking beside the candidates
    rather than inside them. Each transport adds or does not add to satisfy the rule here, so
    nothing downstream has to know which provider answered.

    Counts only, never money. A cached token and a written one bill at different multipliers, so a
    caller that wants cost multiplies these slices by its own per-model rates. A zero slice means
    the provider billed none of it, or reported no breakdown at all. Axio cannot tell those apart.
    """

    input_tokens: int
    output_tokens: int

    #: The slice of ``input_tokens`` served from cache, billed at a discount.
    cache_read_tokens: int = field(default=0, kw_only=True)
    #: The slice of ``input_tokens`` written to cache, billed at a premium. Disjoint from the read.
    cache_write_tokens: int = field(default=0, kw_only=True)
    #: The slice of ``output_tokens`` spent on reasoning the caller never sees.
    reasoning_tokens: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        """Hold the rule above, so no derived count can come out negative.

        Documented and unchecked, a provider report the transport converted wrongly gave
        ``uncached_input_tokens`` or ``answer_tokens`` below zero, and every display, aggregate,
        quota check and cost built on them was wrong with nothing saying so.
        """
        if min(self.input_tokens, self.output_tokens, self.cache_read_tokens) < 0:
            raise ValueError(f"token counts cannot be negative: {self}")
        if min(self.cache_write_tokens, self.reasoning_tokens) < 0:
            raise ValueError(f"token counts cannot be negative: {self}")
        if self.cache_read_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError(f"the cache slices are inside input_tokens, which is the grand total: {self}")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError(f"reasoning is a slice of output_tokens, which is the grand total: {self}")

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        """Input the provider had to read in full, which is what most of the bill is."""
        return self.input_tokens - self.cache_read_tokens - self.cache_write_tokens

    @property
    def answer_tokens(self) -> int:
        """Output that was the answer rather than reasoning."""
        return self.output_tokens - self.reasoning_tokens
