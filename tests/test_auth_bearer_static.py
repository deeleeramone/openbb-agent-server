"""bearer_static auth backend tests."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from openbb_agent_server.plugins.auth.bearer_static import BearerStaticAuthBackend


def _make_request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {"type": "http", "headers": raw}
    return Request(scope)


def test_constructor_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENBB_AGENT_AUTH_BEARER", raising=False)
    with pytest.raises(RuntimeError):
        BearerStaticAuthBackend()


def test_constructor_reads_token_from_env(bearer_env: str) -> None:
    backend = BearerStaticAuthBackend()
    assert backend._token == bearer_env  # noqa: SLF001 — internal contract test


@pytest.mark.asyncio
async def test_missing_authorization_header_returns_401(bearer_env: str) -> None:
    backend = BearerStaticAuthBackend()
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_make_request({}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_returns_403(bearer_env: str) -> None:
    backend = BearerStaticAuthBackend()
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_make_request({"authorization": "Bearer wrong"}))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_valid_token_resolves_principal(bearer_env: str) -> None:
    backend = BearerStaticAuthBackend(user_id="dev")
    p = await backend.authenticate(
        _make_request({"authorization": f"Bearer {bearer_env}"})
    )
    assert p.user_id == "dev"
    assert "agent:query" in p.scopes


@pytest.mark.asyncio
async def test_non_bearer_scheme_returns_401(bearer_env: str) -> None:
    backend = BearerStaticAuthBackend()
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(
            _make_request({"authorization": f"Basic {bearer_env}"})
        )
    assert exc.value.status_code == 401
