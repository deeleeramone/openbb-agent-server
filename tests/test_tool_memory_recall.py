"""recall_user_memory tool source tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from openbb_agent_server.memory.embeddings import HashEmbeddings
from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.plugins.tools.memory_recall import (
    MemoryRecallToolSource,
)
from openbb_agent_server.runtime import context as run_context
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


@pytest_asyncio.fixture
async def memory_store(tmp_path: Path) -> AsyncIterator[SqliteMemoryStore]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'm.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    store = SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=64))
    try:
        yield store
    finally:
        store.close()
        await history.aclose()


def _ctx(user_id: str) -> RunContext:
    return RunContext(
        principal=UserPrincipal(
            user_id=user_id,
            scopes=("agent:query", "memory:read", "memory:write"),
        ),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


@pytest.mark.asyncio
async def test_no_store_yields_no_tool() -> None:
    src = MemoryRecallToolSource()
    tools = await src.tools(_ctx("u"), {})
    assert tools == []


@pytest.mark.asyncio
async def test_with_store_yields_a_callable_tool(
    memory_store: SqliteMemoryStore,
) -> None:
    src = MemoryRecallToolSource(store=memory_store)
    tools = await src.tools(_ctx("alice"), {})
    assert len(tools) == 1
    assert tools[0].name == "recall_user_memory"


@pytest.mark.asyncio
async def test_recall_tool_returns_results_for_current_principal(
    memory_store: SqliteMemoryStore,
) -> None:
    alice = UserPrincipal(user_id="alice", scopes=("memory:read", "memory:write"))
    await memory_store.write(principal=alice, text="alice tracks AAPL")

    src = MemoryRecallToolSource(store=memory_store)
    [tool] = await src.tools(_ctx("alice"), {})
    with run_context.bind(_ctx("alice")):
        result = await tool.ainvoke({"query": "AAPL", "k": 5})
    assert any("AAPL" in r["text"] for r in result)


@pytest.mark.asyncio
async def test_recall_tool_does_not_leak_across_users(
    memory_store: SqliteMemoryStore,
) -> None:
    alice = UserPrincipal(user_id="alice", scopes=("memory:read", "memory:write"))
    await memory_store.write(principal=alice, text="ALICE_CANARY_DATA")

    src = MemoryRecallToolSource(store=memory_store)
    [tool] = await src.tools(_ctx("bob"), {})
    with run_context.bind(_ctx("bob")):
        result = await tool.ainvoke({"query": "ALICE_CANARY_DATA", "k": 10})
    assert all("ALICE_CANARY_DATA" not in r["text"] for r in result)


def test_memory_recall_bind_store_replaces_constructor_default() -> None:
    from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
    from openbb_agent_server.plugins.tools.memory_recall import MemoryRecallToolSource

    src = MemoryRecallToolSource()
    fake_store = SqliteMemoryStore.__new__(SqliteMemoryStore)
    src._bind_store(fake_store)
    assert src._store is fake_store
