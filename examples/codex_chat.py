"""Minimal CLI chat using a ChatGPT subscription (Codex OAuth).

On first run the browser opens for ChatGPT sign-in; tokens are saved to
~/.config/axio-codex-example.ini and refreshed automatically on expiry.

Run:
    uv run python examples/codex_chat.py
"""

from __future__ import annotations

import asyncio
import configparser
from pathlib import Path

import aiohttp

from axio import Agent, MemoryContextStore, TextDelta
from axio_transport_codex.oauth import run_oauth_flow
from axio_transport_codex.transport import CodexTransport

CONFIG_PATH = Path.home() / ".config" / "axio-codex-example.ini"
SECTION = "auth"


def load_tokens() -> dict[str, str] | None:
    cfg = configparser.ConfigParser()
    if (
        not CONFIG_PATH.exists()
        or not cfg.read(CONFIG_PATH)
        or not cfg.has_section(SECTION)
    ):
        return None
    return dict(cfg[SECTION])


def save_tokens(tokens: dict[str, str]) -> None:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)  # preserve any other sections
    cfg[SECTION] = tokens
    with CONFIG_PATH.open("w") as fh:
        cfg.write(fh)


async def on_token_refresh(tokens: dict[str, str]) -> None:
    save_tokens(tokens)


async def main() -> None:
    tokens = load_tokens()
    if tokens is None:
        print("No saved credentials. Opening browser for sign-in...")
        tokens = await run_oauth_flow()
        save_tokens(tokens)
        print("Signed in.\n")

    async with aiohttp.ClientSession() as session:
        transport = CodexTransport(
            api_key=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", ""),
            expires_at=tokens.get("expires_at", "0"),
            account_id=tokens.get("account_id", ""),
            session=session,
            on_auth_refresh=on_token_refresh,
        )

        await transport.fetch_models()

        agent = Agent(
            system="You are a helpful assistant.",
            transport=transport,
            tools=[],
        )
        context = MemoryContextStore()

        print(f"Model: {transport.model.id}  (Ctrl-D or Ctrl-C to exit)\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            print("Assistant: ", end="", flush=True)
            try:
                async for event in agent.run_stream(user_input, context):
                    if isinstance(event, TextDelta):
                        print(event.delta, end="", flush=True)
            except Exception as exc:
                print(f"\n[error: {exc}]")
            else:
                print()


if __name__ == "__main__":
    asyncio.run(main())
