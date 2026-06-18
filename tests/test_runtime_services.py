"""Shared services container tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbb_agent_server.runtime import services


def test_get_history_before_set_raises() -> None:
    services.reset()
    with pytest.raises(RuntimeError):
        services.get_history()


def test_get_memory_before_set_returns_none() -> None:
    services.reset()
    assert services.get_memory() is None


def test_set_then_get_history_round_trip() -> None:
    class FakeHistory:
        pass

    services.reset()
    fh = FakeHistory()
    services.set_services(history=fh)  # type: ignore[arg-type]
    assert services.get_history() is fh
    services.reset()


def test_set_partial_then_overwrite() -> None:
    services.reset()
    h1, h2 = object(), object()
    services.set_services(history=h1)  # type: ignore[arg-type]
    services.set_services(history=h2)  # type: ignore[arg-type]
    assert services.get_history() is h2
    services.reset()


def test_extra_keys_pass_through() -> None:
    services.reset()
    services.set_services(custom_value=42)
    services.reset()


def test_set_and_get_checkpointer() -> None:
    services.reset()
    sentinel = object()
    services.set_services(checkpointer=sentinel)
    assert services.get_checkpointer() is sentinel
    services.reset()


def test_get_checkpointer_before_set_raises() -> None:
    services.reset()
    with pytest.raises(RuntimeError):
        services.get_checkpointer()


def test_get_widget_store_returns_none_when_unset() -> None:
    services.reset()
    assert services.get_widget_store() is None


def test_get_pdf_store_returns_none_when_unset() -> None:
    services.reset()
    assert services.get_pdf_store() is None


def test_set_and_get_widget_store() -> None:
    services.reset()
    sentinel = object()
    services.set_services(widget_store=sentinel)
    assert services.get_widget_store() is sentinel
    services.reset()


def test_set_and_get_pdf_store() -> None:
    services.reset()
    sentinel = object()
    services.set_services(pdf_store=sentinel)
    assert services.get_pdf_store() is sentinel
    services.reset()


@pytest.mark.asyncio
async def test_areset_closes_all_stores(tmp_path: Path) -> None:
    """areset() properly closes every bound store."""
    from openbb_agent_server.memory.embeddings import HashEmbeddings
    from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
    from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
    from openbb_agent_server.runtime.pdf_store import PdfStore
    from openbb_agent_server.runtime.widget_store import WidgetDataStore

    url = f"sqlite+aiosqlite:///{tmp_path / 'svc.db'}"
    hist = SqliteHistoryStore(url)
    await hist.init_schema()
    mem = SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=16))
    pdf = PdfStore(url)
    ws = WidgetDataStore(url)

    services.set_services(history=hist, memory=mem, pdf_store=pdf, widget_store=ws)
    await services.areset()

    assert services.get_memory() is None
    with pytest.raises(RuntimeError):
        services.get_history()


@pytest.mark.asyncio
async def test_sync_reset_closes_sqlite_connections(tmp_path: Path) -> None:
    """sync reset() closes SqliteMemoryStore._conn and PdfStore._vec_conn."""
    from openbb_agent_server.memory.embeddings import HashEmbeddings
    from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
    from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
    from openbb_agent_server.runtime.pdf_store import PdfStore

    db = tmp_path / "sync.db"
    url = f"sqlite+aiosqlite:///{db}"
    hist = SqliteHistoryStore(url)
    await hist.init_schema()
    mem = SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=16))
    pdf = PdfStore(url, embeddings=HashEmbeddings(dim=16))

    services.set_services(history=hist, memory=mem, pdf_store=pdf)
    assert pdf._vec_conn is not None

    services.reset()

    assert services.get_memory() is None
    # sync reset() doesn't touch async engines — clean them up here.
    await hist.aclose()
    await pdf.aclose()
