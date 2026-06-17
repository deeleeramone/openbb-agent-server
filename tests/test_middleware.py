"""Middleware unit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage

from openbb_agent_server.memory.embeddings import HashEmbeddings
from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
from openbb_agent_server.memory.writer import write_memories
from openbb_agent_server.persistence import models as m
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.plugins.middleware.tool_call_ledger import (
    ToolCallLedgerMiddlewareFactory,
)
from openbb_agent_server.plugins.middleware.usage_recorder import (
    UsageRecorderMiddlewareFactory,
)
from openbb_agent_server.plugins.models.fake_provider import (
    _ToolAwareFakeChatModel,
)
from openbb_agent_server.runtime import (
    context as run_context,
    services,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


@pytest_asyncio.fixture
async def store_pair(
    tmp_path: Path,
) -> AsyncIterator[tuple[SqliteHistoryStore, SqliteMemoryStore]]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'mw.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    memory = SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=64))
    services.set_services(history=history, memory=memory)
    try:
        yield history, memory
    finally:
        await history.aclose()
        services.reset()


def _ctx(user_id: str = "u", scopes: tuple[str, ...] = ("memory:write",)) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id=user_id, scopes=scopes),
        trace_id="t1",
        run_id="r1",
        conversation_id="c1",
    )


@pytest.mark.asyncio
async def test_usage_recorder_persists_token_metadata(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    history, _ = store_pair
    factory = UsageRecorderMiddlewareFactory()
    mw = factory.build(_ctx(), {})

    ai = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_token_details": {"cache_read": 2, "cache_creation": 1},
        },
    )
    state = {"messages": [ai]}

    alice = UserPrincipal(user_id="u")
    await history.upsert_user(alice)
    await history.begin_trace(
        principal=alice, trace_id="t1", conversation_id=None, run_id=None
    )

    with run_context.bind(_ctx()):
        await mw.aafter_model(state, runtime=None)

    async with history._sessionmaker() as session:  # type: ignore[attr-defined]
        from sqlalchemy import select

        rows = (
            (await session.execute(select(m.Usage).where(m.Usage.trace_id == "t1")))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    row = rows[0]
    assert row.input_tokens == 10
    assert row.output_tokens == 5
    assert row.cache_read == 2
    assert row.cache_creation == 1


@pytest.mark.asyncio
async def test_usage_recorder_no_op_outside_run_context(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    factory = UsageRecorderMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    ai = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    await mw.aafter_model({"messages": [ai]}, runtime=None)


@pytest.mark.asyncio
async def test_usage_recorder_no_op_when_no_messages(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    factory = UsageRecorderMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    with run_context.bind(_ctx()):
        await mw.aafter_model({"messages": []}, runtime=None)


@pytest.mark.asyncio
async def test_usage_recorder_no_op_when_message_has_no_usage(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    factory = UsageRecorderMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    with run_context.bind(_ctx()):
        await mw.aafter_model({"messages": [AIMessage(content="x")]}, runtime=None)


class _FakeRequest:
    def __init__(self, *, tool_name: str, args: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.args = args


@pytest.mark.asyncio
async def test_tool_call_ledger_records_complete_call(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    history, _ = store_pair
    factory = ToolCallLedgerMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    alice = UserPrincipal(user_id="u")
    await history.upsert_user(alice)
    await history.begin_trace(
        principal=alice, trace_id="t1", conversation_id=None, run_id=None
    )

    async def handler(_req: Any) -> object:
        class R:
            content = {"ok": True}

        return R()

    with run_context.bind(_ctx()):
        await mw.awrap_tool_call(
            _FakeRequest(tool_name="search", args={"q": "x"}),
            handler,
        )

    async with history._sessionmaker() as session:  # type: ignore[attr-defined]
        from sqlalchemy import select

        rows = (
            (
                await session.execute(
                    select(m.ToolCall).where(m.ToolCall.trace_id == "t1")
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].tool_name == "search"
    assert rows[0].state == "complete"


@pytest.mark.asyncio
async def test_tool_call_ledger_records_error_path(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    history, _ = store_pair
    factory = ToolCallLedgerMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    alice = UserPrincipal(user_id="u")
    await history.upsert_user(alice)
    await history.begin_trace(
        principal=alice, trace_id="t1", conversation_id=None, run_id=None
    )

    async def handler(_req: Any) -> object:
        raise RuntimeError("boom")

    with run_context.bind(_ctx()):
        with pytest.raises(RuntimeError):
            await mw.awrap_tool_call(
                _FakeRequest(tool_name="failing", args={}),
                handler,
            )

    async with history._sessionmaker() as session:  # type: ignore[attr-defined]
        from sqlalchemy import select

        rows = (
            (
                await session.execute(
                    select(m.ToolCall).where(m.ToolCall.trace_id == "t1")
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].state == "error"
    assert rows[0].error == "boom"


@pytest.mark.asyncio
async def test_tool_call_ledger_passes_through_graph_bubble_up(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    """Pass GraphBubbleUp through as a control-flow signal."""
    from langgraph.errors import GraphBubbleUp

    history, _ = store_pair
    factory = ToolCallLedgerMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    alice = UserPrincipal(user_id="u")
    await history.upsert_user(alice)
    await history.begin_trace(
        principal=alice, trace_id="t1", conversation_id=None, run_id=None
    )

    async def handler(_req: Any) -> object:
        raise GraphBubbleUp("interrupt")

    with run_context.bind(_ctx()), pytest.raises(GraphBubbleUp):
        await mw.awrap_tool_call(
            _FakeRequest(tool_name="paused", args={}),
            handler,
        )
    async with history._sessionmaker() as session:  # type: ignore[attr-defined]
        from sqlalchemy import select

        rows = (
            (
                await session.execute(
                    select(m.ToolCall).where(m.ToolCall.trace_id == "t1")
                )
            )
            .scalars()
            .all()
        )
    assert rows == []


@pytest.mark.asyncio
async def test_tool_call_ledger_passes_through_when_no_run_context(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    factory = ToolCallLedgerMiddlewareFactory()
    mw = factory.build(_ctx(), {})

    async def handler(_req: Any) -> str:
        return "ok"

    out = await mw.awrap_tool_call(_FakeRequest(tool_name="t", args={}), handler)
    assert out == "ok"


@pytest.mark.asyncio
async def test_memory_writer_writes_extracted_facts(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    _, memory = store_pair
    extractor = _ToolAwareFakeChatModel(
        messages=iter([AIMessage(content="user prefers dark mode\nuser tracks AAPL")])
    )
    ctx = _ctx(scopes=("memory:read", "memory:write"))
    written = await write_memories(
        principal=ctx.principal,
        store=memory,
        extractor=extractor,
        human_text="hi",
        ai_text="OK then.",
        trace_id=ctx.trace_id,
    )
    assert written == 2
    listed = await memory.list_memories(principal=ctx.principal)
    assert len(listed) == 2


@pytest.mark.asyncio
async def test_memory_writer_no_op_without_scope(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    _, memory = store_pair
    extractor = _ToolAwareFakeChatModel(messages=iter([AIMessage(content="x")]))
    ctx = _ctx(scopes=("memory:read",))
    written = await write_memories(
        principal=ctx.principal,
        store=memory,
        extractor=extractor,
        human_text="hi",
        ai_text="bye",
        trace_id=ctx.trace_id,
    )
    assert written == 0
    assert await memory.list_memories(principal=ctx.principal) == []


@pytest.mark.asyncio
async def test_memory_writer_no_op_when_transcript_empty(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    _, memory = store_pair
    extractor = _ToolAwareFakeChatModel(messages=iter([AIMessage(content="x")]))
    ctx = _ctx(scopes=("memory:read", "memory:write"))
    written = await write_memories(
        principal=ctx.principal,
        store=memory,
        extractor=extractor,
        human_text="",
        ai_text="",
        trace_id=ctx.trace_id,
    )
    assert written == 0


@pytest.mark.asyncio
async def test_memory_writer_handles_none_extractor_output(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    _, memory = store_pair
    extractor = _ToolAwareFakeChatModel(messages=iter([AIMessage(content="NONE")]))
    ctx = _ctx(scopes=("memory:read", "memory:write"))
    written = await write_memories(
        principal=ctx.principal,
        store=memory,
        extractor=extractor,
        human_text="hi",
        ai_text="hello",
        trace_id=ctx.trace_id,
    )
    assert written == 0
    assert await memory.list_memories(principal=ctx.principal) == []


@pytest.mark.asyncio
async def test_memory_writer_swallows_extractor_failure(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    """Swallow extractor failures rather than aborting the request."""

    class _Boom:
        async def ainvoke(self, *_a: Any, **_k: Any) -> AIMessage:
            raise RuntimeError("model exploded")

    _, memory = store_pair
    ctx = _ctx(scopes=("memory:read", "memory:write"))
    written = await write_memories(
        principal=ctx.principal,
        store=memory,
        extractor=_Boom(),
        human_text="hi",
        ai_text="hello",
        trace_id=ctx.trace_id,
    )
    assert written == 0


@pytest.mark.asyncio
async def test_schedule_returns_none_when_store_or_extractor_absent() -> None:
    """Short-circuit schedule() when store or extractor is absent."""
    from openbb_agent_server.memory.writer import schedule

    ctx = _ctx(scopes=("memory:read", "memory:write"))
    assert (
        schedule(
            principal=ctx.principal,
            store=None,
            extractor=None,
            human_text="hi",
            ai_text="bye",
            trace_id=ctx.trace_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_schedule_returns_none_without_write_scope(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    from openbb_agent_server.memory.writer import schedule

    _, memory = store_pair
    extractor = _ToolAwareFakeChatModel(messages=iter([AIMessage(content="x")]))
    ctx = _ctx(scopes=("memory:read",))
    assert (
        schedule(
            principal=ctx.principal,
            store=memory,
            extractor=extractor,
            human_text="hi",
            ai_text="bye",
            trace_id=ctx.trace_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_schedule_runs_in_background_and_persists(
    store_pair: tuple[SqliteHistoryStore, SqliteMemoryStore],
) -> None:
    from openbb_agent_server.memory.writer import schedule

    _, memory = store_pair
    extractor = _ToolAwareFakeChatModel(
        messages=iter([AIMessage(content="user likes dark mode")])
    )
    ctx = _ctx(scopes=("memory:read", "memory:write"))
    task = schedule(
        principal=ctx.principal,
        store=memory,
        extractor=extractor,
        human_text="enable dark mode",
        ai_text="done",
        trace_id=ctx.trace_id,
    )
    assert task is not None
    written = await task
    assert written == 1
    assert len(await memory.list_memories(principal=ctx.principal)) == 1


def test_tool_call_ledger_safe_json_falls_back_for_unserializable() -> None:
    from openbb_agent_server.plugins.middleware.tool_call_ledger import _safe_json

    class _NotSerialisable:
        def __repr__(self) -> str:
            return "<obj>"

    out = _safe_json(_NotSerialisable())
    assert out == {"__str__": "<obj>"}
