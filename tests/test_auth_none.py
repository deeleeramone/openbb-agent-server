"""none auth backend tests."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from openbb_agent_server.plugins.auth.none import (
    ANONYMOUS_USER_ID,
    NoneAuthBackend,
)


def _empty_request() -> Request:
    return Request({"type": "http", "headers": []})


@pytest.mark.asyncio
async def test_none_backend_returns_anonymous_principal() -> None:
    backend = NoneAuthBackend()
    p = await backend.authenticate(_empty_request())
    assert p.user_id == ANONYMOUS_USER_ID
    assert "agent:query" in p.scopes
