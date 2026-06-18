"""Shared fixtures."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio

from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.runtime.principal import UserPrincipal


@pytest.fixture
def alice() -> UserPrincipal:
    return UserPrincipal(
        user_id="alice",
        display_name="Alice",
        scopes=("agent:query", "memory:read", "memory:write"),
    )


@pytest.fixture
def bob() -> UserPrincipal:
    return UserPrincipal(
        user_id="bob",
        display_name="Bob",
        scopes=("agent:query",),
    )


@pytest_asyncio.fixture
async def history(tmp_path: Path) -> AsyncIterator[SqliteHistoryStore]:
    store = SqliteHistoryStore(f"sqlite+aiosqlite:///{tmp_path / 'h.db'}")
    await store.init_schema()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture
def bearer_env(monkeypatch: pytest.MonkeyPatch) -> str:
    token = "test-bearer-token"
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BEARER", token)
    return token


@pytest.fixture
def settings_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> AgentServerSettings:
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_CHECKPOINTER_PROVIDER", "inmemory")
    return AgentServerSettings()


def _close_services() -> None:
    """Close any open stores/engines before dropping references."""
    from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
    from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
    from openbb_agent_server.runtime import services
    from openbb_agent_server.runtime.pdf_store import PdfStore

    try:
        mem = services.get_memory()
    except Exception:
        mem = None
    if isinstance(mem, SqliteMemoryStore):
        mem.close()

    try:
        hist = services.get_history()
    except Exception:
        hist = None
    if isinstance(hist, SqliteHistoryStore):
        hist._engine.sync_engine.dispose()

    pdf = services.get_pdf_store()
    if isinstance(pdf, PdfStore):
        if pdf._vec_conn is not None:
            pdf._vec_conn.close()
        pdf._engine.sync_engine.dispose()

    ws = services.get_widget_store()
    if ws is not None and hasattr(ws, "_engine"):
        ws._engine.sync_engine.dispose()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate the test from the host environment and global services."""
    from openbb_agent_server.runtime import canvas, services

    monkeypatch.setenv("HOME", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("OPENBB_AGENT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENBB_AGENT_EMBEDDINGS_PROVIDER", "hash")
    monkeypatch.setenv("OPENBB_AGENT_EMBEDDINGS_CODE_PROVIDER", "")
    monkeypatch.setenv("OPENBB_AGENT_RERANKER_PROVIDER", "")
    monkeypatch.setenv("OPENBB_AGENT_TRANSLATION_PROVIDER", "")
    monkeypatch.setenv("OPENBB_AGENT_PRUNE_INTERVAL_HOURS", "0")
    _close_services()
    services.reset()
    canvas.reset_canvas()
    yield
    _close_services()
    services.reset()
    canvas.reset_canvas()
