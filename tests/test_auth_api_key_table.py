"""api_key_table auth backend tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete
from starlette.requests import Request

from openbb_agent_server.persistence import models as m
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.plugins.auth.api_key_table import (
    ApiKeyTableAuthBackend,
    IssuedKey,
)


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw})


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncIterator[ApiKeyTableAuthBackend]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    backend = ApiKeyTableAuthBackend(db_url=url)
    try:
        yield backend
    finally:
        await backend.aclose()
        await history.aclose()


@pytest.mark.asyncio
async def test_issue_returns_issued_key_object(
    backend: ApiKeyTableAuthBackend,
) -> None:
    issued = await backend.issue(user_id="alice", label="laptop")
    assert isinstance(issued, IssuedKey)
    assert issued.user_id == "alice"
    assert issued.label == "laptop"
    assert issued.plaintext.startswith("oba_")
    assert "." in issued.plaintext


@pytest.mark.asyncio
async def test_issue_then_authenticate_round_trip(
    backend: ApiKeyTableAuthBackend,
) -> None:
    issued = await backend.issue(user_id="alice", scopes=("agent:query", "memory:read"))
    p = await backend.authenticate(_request({"x-api-key": issued.plaintext}))
    assert p.user_id == "alice"
    assert "agent:query" in p.scopes
    assert "memory:read" in p.scopes


@pytest.mark.asyncio
async def test_issue_auto_provisions_the_user_row(
    backend: ApiKeyTableAuthBackend,
) -> None:
    """Create the User row in-line on first call for an unknown user."""
    issued = await backend.issue(
        user_id="newcomer", display_name="Pat", email="pat@example.com"
    )
    p = await backend.authenticate(_request({"x-api-key": issued.plaintext}))
    assert p.user_id == "newcomer"
    assert p.display_name == "Pat"
    assert p.email == "pat@example.com"


@pytest.mark.asyncio
async def test_issue_default_scopes(backend: ApiKeyTableAuthBackend) -> None:
    issued = await backend.issue(user_id="alice")
    assert "agent:query" in issued.scopes
    assert "memory:read" in issued.scopes


@pytest.mark.asyncio
async def test_missing_key_returns_401(
    backend: ApiKeyTableAuthBackend,
) -> None:
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_request({}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_key_returns_403(
    backend: ApiKeyTableAuthBackend,
) -> None:
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_request({"x-api-key": "nonsense"}))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_revoked_key_is_rejected(
    backend: ApiKeyTableAuthBackend,
) -> None:
    issued = await backend.issue(user_id="alice")
    assert await backend.revoke(key_id=issued.key_id)
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_request({"x-api-key": issued.plaintext}))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_revoke_nonexistent_returns_false(
    backend: ApiKeyTableAuthBackend,
) -> None:
    assert await backend.revoke(key_id="does-not-exist") is False


@pytest.mark.asyncio
async def test_authorization_bearer_header_also_accepted(
    backend: ApiKeyTableAuthBackend,
) -> None:
    issued = await backend.issue(user_id="alice")
    p = await backend.authenticate(
        _request({"authorization": f"Bearer {issued.plaintext}"})
    )
    assert p.user_id == "alice"


@pytest.mark.asyncio
async def test_dangling_key_after_user_deleted_is_rejected(
    backend: ApiKeyTableAuthBackend,
    tmp_path: Path,
) -> None:
    """Reject a key whose User row was deleted out from under it."""
    issued = await backend.issue(user_id="ghost")

    async with backend._sessionmaker() as session:  # noqa: SLF001
        await session.execute(delete(m.User).where(m.User.user_id == "ghost"))
        await session.commit()

    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_request({"x-api-key": issued.plaintext}))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_list_keys_returns_metadata_no_secrets(
    backend: ApiKeyTableAuthBackend,
) -> None:
    issued = await backend.issue(user_id="alice", label="laptop")
    rows = await backend.list_keys()
    assert any(r["key_id"] == issued.key_id for r in rows)
    [row] = [r for r in rows if r["key_id"] == issued.key_id]
    assert row["user_id"] == "alice"
    assert row["label"] == "laptop"
    assert row["revoked_at"] is None
    assert "hashed_secret" not in row
    assert "key" not in row
    assert "secret" not in row


@pytest.mark.asyncio
async def test_list_keys_filters_by_user(
    backend: ApiKeyTableAuthBackend,
) -> None:
    a = await backend.issue(user_id="alice")
    b = await backend.issue(user_id="bob")
    alices = await backend.list_keys(user_id="alice")
    assert {r["key_id"] for r in alices} == {a.key_id}
    bobs = await backend.list_keys(user_id="bob")
    assert {r["key_id"] for r in bobs} == {b.key_id}


@pytest.mark.asyncio
async def test_list_keys_records_revoked_state(
    backend: ApiKeyTableAuthBackend,
) -> None:
    issued = await backend.issue(user_id="alice")
    await backend.revoke(key_id=issued.key_id)
    [row] = await backend.list_keys(user_id="alice")
    assert row["revoked_at"] is not None


@pytest_asyncio.fixture
async def api_key_backend(tmp_path: Path):
    from openbb_agent_server.plugins.auth.api_key_table import ApiKeyTableAuthBackend

    url = f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    backend = ApiKeyTableAuthBackend(db_url=url)
    try:
        yield backend
    finally:
        await backend.aclose()
        await history.aclose()


@pytest.mark.asyncio
async def test_api_key_table_rejects_empty_fragments(api_key_backend) -> None:
    """Reject an api key with empty fragments."""
    with pytest.raises(HTTPException) as exc:
        await api_key_backend.authenticate(_request({"x-api-key": "oba_."}))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_api_key_table_wrong_secret_returns_403(api_key_backend) -> None:
    issued = await api_key_backend.issue(user_id="alice")
    parts = issued.plaintext.split(".")
    forged = parts[0] + "." + "x" * 32
    with pytest.raises(HTTPException) as exc:
        await api_key_backend.authenticate(_request({"x-api-key": forged}))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_api_key_table_issue_updates_existing_user_metadata(
    api_key_backend,
) -> None:
    """Fill empty user metadata on a second issue() call."""
    await api_key_backend.issue(user_id="bob")
    await api_key_backend.issue(
        user_id="bob", display_name="Bob", email="bob@example.com"
    )
    p = await api_key_backend.authenticate(
        _request({"x-api-key": (await api_key_backend.issue(user_id="bob")).plaintext})
    )
    assert p.display_name == "Bob"
    assert p.email == "bob@example.com"
