"""Checkpointer plugin tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.plugins.checkpointers.inmemory import (
    InMemoryCheckpointerProvider,
)
from openbb_agent_server.plugins.checkpointers.postgres import (
    _normalise_pg_url,
)
from openbb_agent_server.plugins.checkpointers.sqlite import (
    SqliteCheckpointerProvider,
)
from openbb_agent_server.runtime import registry, services


def test_entrypoints_register_three_providers() -> None:
    names = registry.available("openbb_agent_server.checkpointers")
    assert {"inmemory", "sqlite", "postgres"}.issubset(names)


@pytest.mark.asyncio
async def test_inmemory_provider_open_and_close() -> None:
    provider = InMemoryCheckpointerProvider()
    saver = await provider.open(AgentServerSettings())
    assert saver is not None
    assert hasattr(saver, "aget")
    assert hasattr(saver, "aput")
    await provider.close(saver)


@pytest_asyncio.fixture
async def opened_sqlite(tmp_path: Path) -> AsyncIterator[tuple[Any, Any]]:
    settings = AgentServerSettings(data_dir=tmp_path)
    provider = SqliteCheckpointerProvider()
    saver = await provider.open(settings)
    try:
        yield provider, saver
    finally:
        await provider.close(saver)


@pytest.mark.asyncio
async def test_sqlite_provider_writes_a_db_file(
    tmp_path: Path,
    opened_sqlite: tuple[Any, Any],
) -> None:
    provider, saver = opened_sqlite
    assert saver is not None
    db_path = tmp_path / "checkpoints.db"
    assert db_path.exists()


@pytest.mark.asyncio
async def test_sqlite_provider_setup_creates_writes_table(
    opened_sqlite: tuple[Any, Any],
) -> None:
    _, saver = opened_sqlite
    out = []
    async for x in saver.alist({"configurable": {"thread_id": "doesnt-exist"}}):
        out.append(x)
    assert out == []


@pytest.mark.asyncio
async def test_sqlite_explicit_path_overrides_default(tmp_path: Path) -> None:
    custom = tmp_path / "custom-ckpt.db"
    provider = SqliteCheckpointerProvider(path=str(custom))
    saver = await provider.open(AgentServerSettings(data_dir=tmp_path))
    try:
        assert custom.exists()
    finally:
        await provider.close(saver)


@pytest.mark.asyncio
async def test_sqlite_close_is_idempotent(opened_sqlite: tuple[Any, Any]) -> None:
    provider, saver = opened_sqlite
    await provider.close(saver)
    await provider.close(saver)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("postgresql+psycopg://u:p@h:5432/db", "postgresql://u:p@h:5432/db"),
        ("postgresql+asyncpg://u:p@h:5432/db", "postgresql://u:p@h:5432/db"),
        ("postgresql://u:p@h:5432/db", "postgresql://u:p@h:5432/db"),
        ("sqlite+aiosqlite:///x.db", "sqlite+aiosqlite:///x.db"),
    ],
)
def test_postgres_url_normalisation(url: str, expected: str) -> None:
    assert _normalise_pg_url(url) == expected


def test_get_checkpointer_unbound_raises() -> None:
    services.reset()
    with pytest.raises(RuntimeError):
        services.get_checkpointer()


@pytest.mark.asyncio
async def test_set_get_checkpointer_round_trip() -> None:
    services.reset()
    sentinel = object()
    services.set_services(checkpointer=sentinel)
    assert services.get_checkpointer() is sentinel
    services.reset()


def test_postgres_checkpointer_constructor_stores_url() -> None:
    from openbb_agent_server.plugins.checkpointers.postgres import (
        PostgresCheckpointerProvider,
    )

    provider = PostgresCheckpointerProvider(url="postgresql://h/db", extra="ignored")
    assert provider._explicit_url == "postgresql://h/db"
    assert provider._cm is None


def test_postgres_resolve_url_normalises_sqlalchemy_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.checkpointers.postgres import (
        PostgresCheckpointerProvider,
    )

    class _Settings:
        def resolved_db_url(self) -> str:
            return "postgresql+psycopg://u:p@h/db"

    monkeypatch.delenv("OPENBB_AGENT_CHECKPOINTER_URL", raising=False)
    provider = PostgresCheckpointerProvider()
    assert provider._resolve_url(_Settings()) == "postgresql://u:p@h/db"


def test_postgres_resolve_url_uses_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.checkpointers.postgres import (
        PostgresCheckpointerProvider,
    )

    class _Settings:
        def resolved_db_url(self) -> str:
            return "sqlite:///nope.db"

    monkeypatch.setenv("OPENBB_AGENT_CHECKPOINTER_URL", "postgresql+asyncpg://u@h/db")
    provider = PostgresCheckpointerProvider()
    assert provider._resolve_url(_Settings()) == "postgresql://u@h/db"


def test_postgres_resolve_url_rejects_non_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.checkpointers.postgres import (
        PostgresCheckpointerProvider,
    )

    class _Settings:
        def resolved_db_url(self) -> str:
            return "sqlite:///nope.db"

    monkeypatch.delenv("OPENBB_AGENT_CHECKPOINTER_URL", raising=False)
    provider = PostgresCheckpointerProvider()
    with pytest.raises(RuntimeError, match="postgresql://"):
        provider._resolve_url(_Settings())
