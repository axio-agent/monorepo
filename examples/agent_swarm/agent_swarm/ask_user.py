from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Annotated

from axio import CONTEXT, Field, Tool
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

PromptFn = Callable[[], str] | None


def _default_prompt() -> str:
    return input("> ").strip() or "(no answer)"


async def ask_user(
    question: Annotated[str, Field(description="Question to ask the user, formatted as Markdown")],
) -> str:
    """Ask the user a question and return their answer.
    Use this when a decision requires human input before proceeding.
    The `question` field MUST be formatted as Markdown."""
    fn: PromptFn = CONTEXT.get()

    def _blocking() -> str:
        console = Console(stderr=True)
        console.print(Rule("[bold yellow]Question for you[/bold yellow]"))
        console.print(Markdown(question))
        console.print(Rule())
        sys.stderr.flush()
        prompt = fn if fn is not None else _default_prompt
        return prompt()

    return await asyncio.to_thread(_blocking)


def make_ask_user_tool(prompt_fn: PromptFn = None, guards: tuple = ()) -> Tool:
    """Create an ask_user tool with *prompt_fn* as its context.

    When *prompt_fn* is ``None`` the tool falls back to the default stdin prompt.
    """
    return Tool(
        name="ask_user",
        handler=ask_user,
        context=prompt_fn,
        guards=guards,
    )
