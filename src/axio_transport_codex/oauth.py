"""OAuth2 PKCE flow for ChatGPT (Codex) authentication."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
import webbrowser
from asyncio import Event
from typing import Any
from urllib.parse import urlencode

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

AUTH_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
SCOPES = "openid profile email offline_access api.connectors.read api.connectors.invoke"
ORIGINATOR = "codex_cli_rs"


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(96)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode JWT payload without verification (base64 only)."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    # Add padding
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    raw = base64.urlsafe_b64decode(payload)
    return json.loads(raw)  # type: ignore[no-any-return]


def _extract_account_id(access_token: str) -> str:
    """Extract account_id from JWT payload."""
    jwt_payload = _decode_jwt_payload(access_token)
    orgs = jwt_payload.get("organizations", [])
    if orgs and isinstance(orgs, list) and isinstance(orgs[0], dict):
        account_id: str = orgs[0].get("id", "")
        if account_id:
            return account_id
    return str(jwt_payload.get("sub", ""))


async def run_oauth_flow() -> dict[str, str]:
    """Run full OAuth2 PKCE flow with localhost callback.

    Opens browser for ChatGPT sign-in, waits for callback, exchanges code for tokens.

    Returns dict with keys: access_token, refresh_token, expires_at, account_id.
    """
    code_verifier, code_challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    result: dict[str, str] = {}
    error: str | None = None
    done = Event()

    async def callback_handler(request: web.Request) -> web.Response:
        nonlocal result, error

        received_state = request.query.get("state", "")
        if received_state != state:
            error = f"State mismatch: expected {state!r}, got {received_state!r}"
            done.set()
            return web.Response(text="Authentication failed: state mismatch", status=400)

        if "error" in request.query:
            error = request.query.get("error_description", request.query["error"])
            done.set()
            return web.Response(text=f"Authentication failed: {error}", status=400)

        code = request.query.get("code", "")
        if not code:
            error = "No authorization code received"
            done.set()
            return web.Response(text="Authentication failed: no code", status=400)

        # Exchange code for tokens
        try:
            async with aiohttp.ClientSession() as session:
                token_data = await _exchange_code(session, code, code_verifier)
            result = token_data
        except Exception as exc:
            error = str(exc)
            done.set()
            return web.Response(text=f"Token exchange failed: {exc}", status=500)

        done.set()
        return web.Response(
            text="<html><body><h2>Authentication successful!</h2>"
            "<p>You can close this tab and return to the app.</p></body></html>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/auth/callback", callback_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 1455)
    await site.start()

    redirect_uri = "http://localhost:1455/auth/callback"

    # Build authorization URL (matching codex-cli exactly)
    params = urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": ORIGINATOR,
        }
    )
    auth_url = f"{AUTH_URL}?{params}"

    logger.info("Opening browser for OAuth sign-in...")
    webbrowser.open(auth_url)

    try:
        await done.wait()
    finally:
        await runner.cleanup()

    if error:
        raise RuntimeError(f"OAuth flow failed: {error}")

    return result


async def _exchange_code(
    session: aiohttp.ClientSession,
    code: str,
    code_verifier: str,
) -> dict[str, str]:
    """Exchange authorization code for tokens via form-encoded POST."""
    redirect_uri = "http://localhost:1455/auth/callback"
    form_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": CLIENT_ID,
        "code_verifier": code_verifier,
    }

    async with session.post(
        TOKEN_URL,
        data=form_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Token exchange failed ({resp.status}): {body}")
        data: dict[str, Any] = await resp.json()

    access_token: str = data["access_token"]
    refresh_token: str = data.get("refresh_token", "")
    expires_in: int = data.get("expires_in", 3600)
    expires_at = str(int(time.time()) + expires_in)
    account_id = _extract_account_id(access_token)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "account_id": account_id,
    }


async def refresh_access_token(refresh_token: str) -> dict[str, str]:
    """Refresh an expired access token (JSON POST, matching codex-cli)."""
    payload = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(TOKEN_URL, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Token refresh failed ({resp.status}): {body}")
            data: dict[str, Any] = await resp.json()

    access_token: str = data["access_token"]
    new_refresh: str = data.get("refresh_token", refresh_token)
    expires_in: int = data.get("expires_in", 3600)
    expires_at = str(int(time.time()) + expires_in)
    account_id = _extract_account_id(access_token)

    return {
        "access_token": access_token,
        "refresh_token": new_refresh,
        "expires_at": expires_at,
        "account_id": account_id,
    }
