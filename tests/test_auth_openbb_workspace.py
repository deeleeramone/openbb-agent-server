"""``openbb_workspace`` auth backend tests."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from openbb_agent_server.plugins.auth.openbb_workspace import (
    DEFAULT_HEADER,
    OpenBBWorkspaceAuthBackend,
)
from openbb_agent_server.runtime.identity import hash_user_id


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw})


@pytest.mark.asyncio
async def test_resolves_user_from_xopenbbuser_header() -> None:
    backend = OpenBBWorkspaceAuthBackend()
    p = await backend.authenticate(_request({DEFAULT_HEADER: "alice@example.com"}))
    assert p.user_id == hash_user_id("alice@example.com")
    assert p.user_id.startswith("u-")
    assert "@" not in p.user_id
    assert p.email == "alice@example.com"
    assert p.display_name is None


@pytest.mark.asyncio
async def test_lowercases_and_trims_the_header() -> None:
    backend = OpenBBWorkspaceAuthBackend()
    p = await backend.authenticate(_request({DEFAULT_HEADER: "  Alice@Example.COM  "}))
    assert p.user_id == hash_user_id("alice@example.com")
    assert p.email == "alice@example.com"


@pytest.mark.asyncio
async def test_user_id_is_stable_across_calls() -> None:
    backend = OpenBBWorkspaceAuthBackend()
    a = await backend.authenticate(_request({DEFAULT_HEADER: "alice@example.com"}))
    b = await backend.authenticate(_request({DEFAULT_HEADER: "ALICE@example.com"}))
    assert a.user_id == b.user_id


@pytest.mark.asyncio
async def test_default_scopes_grant_query_and_memory() -> None:
    backend = OpenBBWorkspaceAuthBackend()
    p = await backend.authenticate(_request({DEFAULT_HEADER: "u@x.com"}))
    assert "agent:query" in p.scopes
    assert "memory:read" in p.scopes
    assert "memory:write" in p.scopes


@pytest.mark.asyncio
async def test_missing_header_returns_401() -> None:
    backend = OpenBBWorkspaceAuthBackend()
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_request({}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_empty_header_returns_401() -> None:
    backend = OpenBBWorkspaceAuthBackend()
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_request({DEFAULT_HEADER: "   "}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_non_email_header_returns_403_when_required() -> None:
    backend = OpenBBWorkspaceAuthBackend()
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_request({DEFAULT_HEADER: "alice"}))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_non_email_header_accepted_when_require_email_false() -> None:
    backend = OpenBBWorkspaceAuthBackend(require_email=False)
    p = await backend.authenticate(_request({DEFAULT_HEADER: "alice"}))
    assert p.user_id == hash_user_id("alice")
    assert p.email is None


@pytest.mark.asyncio
async def test_custom_header_name() -> None:
    backend = OpenBBWorkspaceAuthBackend(header="X-Forwarded-User")
    p = await backend.authenticate(_request({"X-Forwarded-User": "bob@example.com"}))
    assert p.user_id == hash_user_id("bob@example.com")
    assert p.email == "bob@example.com"


@pytest.mark.asyncio
async def test_custom_scopes_are_used() -> None:
    backend = OpenBBWorkspaceAuthBackend(scopes=("agent:query",))
    p = await backend.authenticate(_request({DEFAULT_HEADER: "u@x.com"}))
    assert p.scopes == ("agent:query",)
