"""Tests for the BaseRetriever adapters over memory and widget stores."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from langchain_core.documents import Document

from openbb_agent_server.memory.embeddings import HashEmbeddings
from openbb_agent_server.memory.retrievers import (
    MemoryStoreRetriever,
    WidgetDataRetriever,
)
from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
from openbb_agent_server.persistence import models as m
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.runtime.principal import UserPrincipal
from openbb_agent_server.runtime.widget_store import WidgetDataStore


@pytest_asyncio.fixture
async def memory(tmp_path: Path) -> AsyncIterator[SqliteMemoryStore]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'm.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    store = SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=64))
    try:
        yield store
    finally:
        store.close()
        await history.aclose()


@pytest_asyncio.fixture
async def widget_store() -> AsyncIterator[WidgetDataStore]:
    s = WidgetDataStore("sqlite+aiosqlite:///:memory:")
    async with s._engine.begin() as conn:
        await conn.run_sync(m.Base.metadata.create_all)
    try:
        yield s
    finally:
        await s.aclose()


@pytest.fixture
def alice() -> UserPrincipal:
    return UserPrincipal(user_id="alice", scopes=("memory:read", "memory:write"))


@pytest.mark.asyncio
async def test_memory_retriever_returns_documents(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    await memory.write(principal=alice, text="user prefers dark mode")
    await memory.write(principal=alice, text="user tracks AAPL fundamentals")
    retriever = MemoryStoreRetriever(store=memory, principal=alice, k=5)
    docs = await retriever.ainvoke("AAPL")
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    assert any("AAPL" in d.page_content for d in docs)


@pytest.mark.asyncio
async def test_memory_retriever_metadata_includes_principal_and_score(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    await memory.write(principal=alice, text="hello", source_trace_id="trace-1")
    retriever = MemoryStoreRetriever(store=memory, principal=alice)
    [doc] = await retriever.ainvoke("hello")
    assert doc.metadata["user_id"] == "alice"
    assert doc.metadata["source_trace_id"] == "trace-1"
    assert "memory_id" in doc.metadata


@pytest.mark.asyncio
async def test_memory_retriever_is_strictly_principal_scoped(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    bob = UserPrincipal(user_id="bob", scopes=("memory:write", "memory:read"))
    await memory.write(principal=bob, text="bob's secret note")
    retriever = MemoryStoreRetriever(store=memory, principal=alice)
    docs = await retriever.ainvoke("secret")
    assert docs == []


@pytest.mark.asyncio
async def test_memory_retriever_respects_k(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    for i in range(10):
        await memory.write(principal=alice, text=f"fact {i}")
    retriever = MemoryStoreRetriever(store=memory, principal=alice, k=3)
    docs = await retriever.ainvoke("fact")
    assert len(docs) <= 3


def test_memory_retriever_clamps_k_to_at_least_one(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Clamp k to at least 1."""
    retriever = MemoryStoreRetriever(store=memory, principal=alice, k=0)
    assert retriever._k == 1


def test_memory_retriever_sync_path_works(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Route the sync invoke path through the async core."""
    import asyncio

    asyncio.run(memory.write(principal=alice, text="sync entry"))
    retriever = MemoryStoreRetriever(store=memory, principal=alice)
    docs = retriever.invoke("sync entry")
    assert len(docs) > 0


@pytest.mark.asyncio
async def test_widget_retriever_substring_path(
    widget_store: WidgetDataStore,
    alice: UserPrincipal,
) -> None:
    await widget_store.record(
        principal=alice,
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="prices",
        origin="market",
        input_args={},
        rows=[{"sym": "AAPL", "px": 100}, {"sym": "GOOG", "px": 200}],
        columns=["sym", "px"],
    )
    retriever = WidgetDataRetriever(
        store=widget_store,
        principal=alice,
        conversation_id="c1",
        k=10,
    )
    docs = await retriever.ainvoke("AAPL")
    assert len(docs) == 1
    assert "AAPL" in docs[0].page_content


@pytest.mark.asyncio
async def test_widget_retriever_metadata_carries_widget_id_and_row(
    widget_store: WidgetDataStore,
    alice: UserPrincipal,
) -> None:
    await widget_store.record(
        principal=alice,
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="prices",
        origin="market",
        input_args={},
        rows=[{"sym": "AAPL"}],
        columns=["sym"],
    )
    retriever = WidgetDataRetriever(
        store=widget_store, principal=alice, conversation_id="c1"
    )
    [doc] = await retriever.ainvoke("AAPL")
    assert doc.metadata["widget_uuid"] == "w-1"
    assert doc.metadata["widget_name"] == "prices"
    assert doc.metadata["row"] == {"sym": "AAPL"}


@pytest.mark.asyncio
async def test_widget_retriever_widget_uuid_filter(
    widget_store: WidgetDataStore,
    alice: UserPrincipal,
) -> None:
    await widget_store.record(
        principal=alice,
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"v": "match"}],
        columns=["v"],
    )
    await widget_store.record(
        principal=alice,
        conversation_id="c1",
        widget_uuid="w-b",
        widget_name="B",
        origin="o",
        input_args={},
        rows=[{"v": "match"}],
        columns=["v"],
    )
    retriever = WidgetDataRetriever(
        store=widget_store,
        principal=alice,
        conversation_id="c1",
        widget_uuid="w-a",
    )
    docs = await retriever.ainvoke("match")
    assert all(d.metadata["widget_uuid"] == "w-a" for d in docs)


@pytest.mark.asyncio
async def test_widget_retriever_empty_for_no_data(
    widget_store: WidgetDataStore,
    alice: UserPrincipal,
) -> None:
    retriever = WidgetDataRetriever(
        store=widget_store, principal=alice, conversation_id="cx"
    )
    docs = await retriever.ainvoke("anything")
    assert docs == []


def test_widget_retriever_clamps_k(
    widget_store: WidgetDataStore,
    alice: UserPrincipal,
) -> None:
    retriever = WidgetDataRetriever(
        store=widget_store, principal=alice, conversation_id="c1", k=-3
    )
    assert retriever._k == 1


def test_widget_retriever_sync_path_works(
    widget_store: WidgetDataStore,
    alice: UserPrincipal,
) -> None:
    """Run the sync invoke path through the async core."""
    retriever = WidgetDataRetriever(
        store=widget_store, principal=alice, conversation_id="cx"
    )
    assert retriever.invoke("nothing") == []


def test_widget_hit_to_doc_skips_none_values() -> None:
    from openbb_agent_server.memory.retrievers import _widget_hit_to_doc

    doc = _widget_hit_to_doc(
        {"row": {"a": 1, "b": None, "c": "x"}, "widget_uuid": "w", "widget_name": "n"}
    )
    assert doc.page_content == "a: 1 | c: x"
    assert doc.metadata["widget_uuid"] == "w"
