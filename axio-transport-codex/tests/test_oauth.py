"""Tests for OAuth PKCE flow."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from axio_transport_codex.oauth import (
    _decode_jwt_payload,
    _generate_pkce,
    refresh_access_token,
)

# ---------------------------------------------------------------------------
# PKCE generation
# ---------------------------------------------------------------------------


def test_pkce_verifier_length() -> None:
    verifier, _ = _generate_pkce()
    # 96 bytes → ~128 base64url chars
    assert len(verifier) >= 100


def test_pkce_challenge_is_sha256_of_verifier() -> None:
    verifier, challenge = _generate_pkce()
    expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
    assert challenge == expected_challenge


def test_pkce_no_padding_in_challenge() -> None:
    _, challenge = _generate_pkce()
    assert "=" not in challenge


def test_pkce_unique_each_call() -> None:
    v1, _ = _generate_pkce()
    v2, _ = _generate_pkce()
    assert v1 != v2


# ---------------------------------------------------------------------------
# JWT decode (no verification)
# ---------------------------------------------------------------------------


def _make_jwt(payload: dict[str, Any]) -> str:
    """Create a fake JWT with the given payload (no signature verification)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.fake_sig"


def test_decode_jwt_payload_basic() -> None:
    payload = {"sub": "user123", "email": "test@example.com"}
    token = _make_jwt(payload)
    decoded = _decode_jwt_payload(token)
    assert decoded["sub"] == "user123"
    assert decoded["email"] == "test@example.com"


def test_decode_jwt_payload_with_orgs() -> None:
    payload = {
        "sub": "user123",
        "organizations": [{"id": "org-abc", "name": "Test Org"}],
    }
    token = _make_jwt(payload)
    decoded = _decode_jwt_payload(token)
    assert decoded["organizations"][0]["id"] == "org-abc"


def test_decode_jwt_payload_invalid_token() -> None:
    result = _decode_jwt_payload("not-a-jwt")
    assert result == {}


def test_decode_jwt_payload_empty_string() -> None:
    result = _decode_jwt_payload("")
    assert result == {}


# ---------------------------------------------------------------------------
# Token refresh (mocked HTTP)
# ---------------------------------------------------------------------------


async def test_refresh_access_token_success() -> None:
    new_token = _make_jwt({"sub": "user123", "organizations": [{"id": "org-xyz"}]})
    response_data = {
        "access_token": new_token,
        "refresh_token": "new_refresh_token",
        "expires_in": 7200,
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=response_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = lambda *args, **kwargs: mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("axio_transport_codex.oauth.aiohttp.ClientSession", return_value=mock_session):
        result = await refresh_access_token("old_refresh_token")

    assert result["access_token"] == new_token
    assert result["refresh_token"] == "new_refresh_token"
    assert result["account_id"] == "org-xyz"
    assert int(result["expires_at"]) > int(time.time())


async def test_refresh_access_token_failure() -> None:
    mock_resp = AsyncMock()
    mock_resp.status = 401
    mock_resp.text = AsyncMock(return_value="Unauthorized")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = lambda *args, **kwargs: mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("axio_transport_codex.oauth.aiohttp.ClientSession", return_value=mock_session),
        pytest.raises(RuntimeError, match="Token refresh failed"),
    ):
        await refresh_access_token("bad_token")


async def test_refresh_preserves_refresh_token_if_not_returned() -> None:
    """If the server doesn't return a new refresh_token, keep the old one."""
    new_token = _make_jwt({"sub": "user123"})
    response_data = {
        "access_token": new_token,
        "expires_in": 3600,
        # No refresh_token in response
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=response_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = lambda *args, **kwargs: mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("axio_transport_codex.oauth.aiohttp.ClientSession", return_value=mock_session):
        result = await refresh_access_token("keep_this_token")

    assert result["refresh_token"] == "keep_this_token"
