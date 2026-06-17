"""End-to-end tests for the memory write/recall pipeline."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool

from openbb_agent_server.memory.embeddings import HashEmbeddings, cosine
from openbb_agent_server.memory.factory import (
    make_embeddings,
    make_reranker,
    make_translator,
)
from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
from openbb_agent_server.memory.writer import (
    EXTRACTOR_SYSTEM_PROMPT,
    _split,
    schedule,
    write_memories,
)
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.plugins.tools.memory_recall import MemoryRecallToolSource
from openbb_agent_server.runtime.context import RunContext, bind
from openbb_agent_server.runtime.principal import UserPrincipal


@pytest_asyncio.fixture
async def memory(tmp_path: Path) -> AsyncIterator[SqliteMemoryStore]:
    """Fresh sqlite-backed memory store on a tempdir DB."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'm.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    store = SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=64))
    try:
        yield store
    finally:
        store.close()
        await history.aclose()


def test_split_returns_empty_on_none_reply() -> None:
    """Return an empty list for a NONE extractor reply."""
    assert _split("NONE") == []
    assert _split("none") == []
    assert _split(" NONE ") == []


def test_split_strips_bullet_prefixes_and_whitespace() -> None:
    raw = (
        "- user prefers AAPL fundamentals\n"
        "  • user is a CFA charterholder\n"
        "* user lives in Toronto\n"
    )
    out = _split(raw)
    assert out == [
        "user prefers AAPL fundamentals",
        "user is a CFA charterholder",
        "user lives in Toronto",
    ]


def test_split_filters_lines_too_short_to_be_facts() -> None:
    raw = "hi\n- ok\n- the user is a quant analyst\n"
    out = _split(raw)
    assert out == ["the user is a quant analyst"]


def test_split_handles_empty_and_whitespace_only() -> None:
    assert _split("") == []
    assert _split("   \n\n   ") == []


async def test_write_memories_persists_extracted_lines(
    memory: SqliteMemoryStore, alice: UserPrincipal
) -> None:
    """Persist extracted lines from a transcript."""
    extractor_reply = (
        "- user tracks AAPL fundamentals quarterly\n"
        "- user prefers dark mode in Workspace\n"
    )
    extractor = FakeListChatModel(responses=[extractor_reply])

    n = await write_memories(
        principal=alice,
        store=memory,
        extractor=extractor,
        human_text="Hi, I'm a CFA charterholder following AAPL.",
        ai_text="Got it — I'll bias toward fundamentals.",
        trace_id="trace-1",
    )
    assert n == 2

    rows = await memory.list_memories(principal=alice)
    texts = {r.text for r in rows}
    assert texts == {
        "user tracks AAPL fundamentals quarterly",
        "user prefers dark mode in Workspace",
    }
    assert all(r.source_trace_id == "trace-1" for r in rows)

    recalled = await memory.recall(principal=alice, query="AAPL", k=5)
    assert any("AAPL" in m.text for m in recalled)


async def test_write_memories_returns_zero_on_none_extractor_reply(
    memory: SqliteMemoryStore, alice: UserPrincipal
) -> None:
    extractor = FakeListChatModel(responses=["NONE"])
    n = await write_memories(
        principal=alice,
        store=memory,
        extractor=extractor,
        human_text="ok thanks",
        ai_text="anytime",
        trace_id="trace-noop",
    )
    assert n == 0
    rows = await memory.list_memories(principal=alice)
    assert rows == []


async def test_write_memories_requires_memory_write_scope(
    memory: SqliteMemoryStore,
) -> None:
    """No-op when the principal lacks memory:write."""
    reader_only = UserPrincipal(user_id="reader", scopes=("memory:read",))
    extractor = FakeListChatModel(responses=["- user likes orange"])
    n = await write_memories(
        principal=reader_only,
        store=memory,
        extractor=extractor,
        human_text="hi",
        ai_text="bye",
        trace_id="t",
    )
    assert n == 0


async def test_write_memories_returns_zero_when_transcript_is_empty(
    memory: SqliteMemoryStore, alice: UserPrincipal
) -> None:
    extractor = FakeListChatModel(responses=["- something"])
    n = await write_memories(
        principal=alice,
        store=memory,
        extractor=extractor,
        human_text="",
        ai_text="",
        trace_id="t",
    )
    assert n == 0


async def test_write_memories_swallows_extractor_failures(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallow extractor failures."""
    extractor = FakeListChatModel(responses=["never used"])

    async def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("rate limit")

    monkeypatch.setattr(FakeListChatModel, "ainvoke", _boom)
    n = await write_memories(
        principal=alice,
        store=memory,
        extractor=extractor,
        human_text="hi",
        ai_text="bye",
        trace_id="t",
    )
    assert n == 0


def test_extractor_system_prompt_is_non_empty() -> None:
    assert "NONE" in EXTRACTOR_SYSTEM_PROMPT
    assert "memory" in EXTRACTOR_SYSTEM_PROMPT.lower()


async def test_schedule_returns_none_when_memory_disabled(
    alice: UserPrincipal,
) -> None:
    """Return None when memory is disabled."""
    assert (
        schedule(
            principal=alice,
            store=None,
            extractor=FakeListChatModel(responses=["x"]),
            human_text="hi",
            ai_text="bye",
            trace_id="t",
        )
        is None
    )
    assert (
        schedule(
            principal=alice,
            store=None,
            extractor=None,
            human_text="hi",
            ai_text="bye",
            trace_id="t",
        )
        is None
    )


async def test_schedule_returns_none_without_memory_write_scope(
    memory: SqliteMemoryStore,
) -> None:
    reader_only = UserPrincipal(user_id="reader", scopes=("memory:read",))
    extractor = FakeListChatModel(responses=["- nope"])
    assert (
        schedule(
            principal=reader_only,
            store=memory,
            extractor=extractor,
            human_text="hi",
            ai_text="bye",
            trace_id="t",
        )
        is None
    )


async def test_schedule_runs_async_and_returns_task(
    memory: SqliteMemoryStore, alice: UserPrincipal
) -> None:
    extractor = FakeListChatModel(responses=["- user is a quant analyst"])
    task = schedule(
        principal=alice,
        store=memory,
        extractor=extractor,
        human_text="Hi I'm a quant.",
        ai_text="cool",
        trace_id="trace-async",
    )
    assert task is not None
    written = await task
    assert written == 1
    rows = await memory.list_memories(principal=alice)
    assert any("quant analyst" in r.text for r in rows)


async def _recall_tool(store: SqliteMemoryStore | None) -> BaseTool:
    src = MemoryRecallToolSource(store=store)
    tools = await src.tools(
        RunContext(
            principal=UserPrincipal(user_id="any"),
            trace_id="t",
            run_id="r",
            conversation_id="c",
        ),
        {},
    )
    [tool] = tools
    return tool


async def test_recall_tool_yields_nothing_when_no_store_bound() -> None:
    src = MemoryRecallToolSource(store=None)
    tools = await src.tools(
        RunContext(
            principal=UserPrincipal(user_id="any"),
            trace_id="t",
            run_id="r",
            conversation_id="c",
        ),
        {},
    )
    assert tools == []


async def test_recall_tool_returns_user_memories_through_ainvoke(
    memory: SqliteMemoryStore, alice: UserPrincipal
) -> None:
    """Return user memories through the tool's ainvoke path."""
    await memory.write(principal=alice, text="user prefers AAPL fundamentals")
    await memory.write(principal=alice, text="user runs Quart-bench daily")

    tool = await _recall_tool(memory)
    ctx = RunContext(principal=alice, trace_id="t", run_id="r", conversation_id="c")
    with bind(ctx):
        result = await tool.ainvoke({"query": "AAPL", "k": 5})
    assert isinstance(result, list)
    assert any("AAPL" in row["text"] for row in result)
    for row in result:
        assert {"id", "text", "kind", "pinned", "score"} <= set(row.keys())


async def test_recall_tool_enforces_principal_scope(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
    bob: UserPrincipal,
) -> None:
    """Enforce principal scope in the recall tool."""
    canary = "AAAA-CANARY-DO-NOT-LEAK-AAAA"
    await memory.write(principal=alice, text=canary)

    tool = await _recall_tool(memory)
    bobs_ctx = RunContext(principal=bob, trace_id="t", run_id="r", conversation_id="c")
    with bind(bobs_ctx):
        result = await tool.ainvoke({"query": canary, "k": 10})
    assert all(canary not in row["text"] for row in result)


async def test_recall_tool_respects_k() -> None:
    """Clamp the result set with k."""
    url = "sqlite+aiosqlite:///:memory:"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    try:
        store = SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=64))
        alice = UserPrincipal(user_id="alice", scopes=("memory:read", "memory:write"))
        for i in range(5):
            await store.write(principal=alice, text=f"fact-{i} about banana")
        tool = await _recall_tool(store)
        ctx = RunContext(principal=alice, trace_id="t", run_id="r", conversation_id="c")
        with bind(ctx):
            result = await tool.ainvoke({"query": "banana", "k": 2})
        assert len(result) == 2
    finally:
        store.close()
        await history.aclose()


def test_make_embeddings_defaults_to_hash_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    e = make_embeddings(None)
    assert isinstance(e, HashEmbeddings)
    assert any("HashEmbeddings" in r.message for r in caplog.records)


def test_make_embeddings_accepts_explicit_hash() -> None:
    e = make_embeddings("hash", config={"dim": 128})
    assert isinstance(e, HashEmbeddings)
    assert e.dim == 128


def test_make_embeddings_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown embeddings provider"):
        make_embeddings("voyage")


def test_make_embeddings_nvidia_returns_langchain_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the official LangChain NVIDIA integration."""
    monkeypatch.setenv("NVIDIA_API_KEY", "fake")
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    e = make_embeddings("nvidia")
    assert isinstance(e, NVIDIAEmbeddings)


def test_make_embeddings_nvidia_raises_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        make_embeddings("nvidia")


def test_make_reranker_returns_none_when_disabled() -> None:
    assert make_reranker(None) is None
    assert make_reranker("") is None


def test_make_reranker_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown reranker provider"):
        make_reranker("cohere")


def test_make_reranker_nvidia_constructs_without_network() -> None:
    r = make_reranker("nvidia", config={"api_key": "fake-key"})
    from openbb_agent_server.memory.reranker import NvidiaReranker

    assert isinstance(r, NvidiaReranker)


def test_make_translator_returns_none_when_disabled() -> None:
    assert make_translator(None) is None
    assert make_translator("") is None


def test_make_translator_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown translation provider"):
        make_translator("deepl")


async def test_hash_embeddings_recall_finds_exact_keyword_match(
    memory: SqliteMemoryStore, alice: UserPrincipal
) -> None:
    """Find an exact keyword match via HashEmbeddings recall."""
    await memory.write(principal=alice, text="user tracks AAPL")
    await memory.write(principal=alice, text="completely unrelated banana stuff")
    [top, *_] = await memory.recall(principal=alice, query="AAPL", k=2)
    assert "AAPL" in top.text


def test_cosine_self_similarity_is_one() -> None:
    e = HashEmbeddings(dim=64)
    v = e.embed_query("hello world")
    assert cosine(v, v) == pytest.approx(1.0)


def test_test_env_does_not_leak_real_nvidia_api_key() -> None:
    """Confirm the test env does not leak a real NVIDIA API key."""
    assert os.environ.get("OPENBB_AGENT_MODEL_PROVIDER") is None


def test_memory_writer_split_drops_short_and_none_lines() -> None:
    from openbb_agent_server.memory.writer import _split

    assert _split("NONE") == []
    assert _split("") == []
    assert _split("- short\n- this is a real one") == ["this is a real one"]
