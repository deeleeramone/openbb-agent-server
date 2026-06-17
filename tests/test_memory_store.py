"""SqliteMemoryStore round-trip and isolation tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from openbb_agent_server.memory.embeddings import HashEmbeddings
from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.runtime.principal import UserPrincipal


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


@pytest.mark.asyncio
async def test_write_then_recall_returns_user_memories(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    await memory.write(principal=alice, text="user prefers dark mode")
    await memory.write(principal=alice, text="user tracks AAPL fundamentals quarterly")
    results = await memory.recall(principal=alice, query="AAPL", k=5)
    assert any("AAPL" in m.text for m in results)


@pytest.mark.asyncio
async def test_write_requires_memory_write_scope(
    memory: SqliteMemoryStore,
) -> None:
    no_scope = UserPrincipal(user_id="u", scopes=("memory:read",))
    with pytest.raises(PermissionError):
        await memory.write(principal=no_scope, text="should be rejected")


@pytest.mark.asyncio
async def test_recall_is_strictly_user_scoped(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
    bob: UserPrincipal,
) -> None:
    bob_with_write = UserPrincipal(
        user_id=bob.user_id,
        display_name=bob.display_name,
        scopes=("agent:query", "memory:read", "memory:write"),
    )
    canary = "AAAA-canary-secret-AAAA"
    await memory.write(principal=alice, text=canary)
    bobs_view = await memory.recall(principal=bob_with_write, query=canary, k=10)
    assert all(canary not in m.text for m in bobs_view)


@pytest.mark.asyncio
async def test_pin_then_unpin_round_trip(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    written = await memory.write(principal=alice, text="pin me later")
    pinned = await memory.pin(principal=alice, memory_id=written.memory_id, pinned=True)
    assert pinned is not None and pinned.pinned is True
    unpinned = await memory.pin(
        principal=alice, memory_id=written.memory_id, pinned=False
    )
    assert unpinned is not None and unpinned.pinned is False


@pytest.mark.asyncio
async def test_pin_other_users_memory_returns_none(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
    bob: UserPrincipal,
) -> None:
    written = await memory.write(principal=alice, text="alice's secret")
    bob_with_write = UserPrincipal(user_id=bob.user_id, scopes=("memory:write",))
    result = await memory.pin(
        principal=bob_with_write, memory_id=written.memory_id, pinned=True
    )
    assert result is None


@pytest.mark.asyncio
async def test_forget_removes_memory(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    written = await memory.write(principal=alice, text="forget me")
    ok = await memory.forget(principal=alice, memory_id=written.memory_id)
    assert ok
    listed = await memory.list_memories(principal=alice)
    assert all(m.memory_id != written.memory_id for m in listed)


@pytest.mark.asyncio
async def test_forget_other_users_memory_returns_false(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
    bob: UserPrincipal,
) -> None:
    written = await memory.write(principal=alice, text="alice owns this")
    ok = await memory.forget(principal=bob, memory_id=written.memory_id)
    assert ok is False


@pytest.mark.asyncio
async def test_pinned_memories_dominate_recall_score(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    pin_target = await memory.write(principal=alice, text="completely unrelated text")
    await memory.write(principal=alice, text="banana banana banana")
    await memory.pin(principal=alice, memory_id=pin_target.memory_id, pinned=True)
    out = await memory.recall(principal=alice, query="banana", k=5)
    assert out[0].memory_id == pin_target.memory_id


@pytest_asyncio.fixture
async def memory_with_code_embedder(
    tmp_path: Path,
) -> AsyncIterator[SqliteMemoryStore]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'mc.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    store = SqliteMemoryStore(
        url,
        embeddings=HashEmbeddings(dim=64),
        code_embeddings=HashEmbeddings(dim=64),
    )
    try:
        yield store
    finally:
        store.close()
        await history.aclose()


@pytest.mark.asyncio
async def test_code_kind_uses_code_embedder_on_write(
    memory_with_code_embedder: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Route a context_code write through the code embedder."""
    out = await memory_with_code_embedder.write(
        principal=alice,
        text="def f(): return 1",
        kind="context_code",
    )
    assert out.kind == "context_code"


@pytest.mark.asyncio
async def test_pin_finds_memory_in_code_table(
    memory_with_code_embedder: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Find a memory in the code table when pinning."""
    code_mem = await memory_with_code_embedder.write(
        principal=alice, text="def f(): return 1", kind="context_code"
    )
    out = await memory_with_code_embedder.pin(
        principal=alice, memory_id=code_mem.memory_id, pinned=True
    )
    assert out is not None
    assert out.pinned is True


@pytest.mark.asyncio
async def test_forget_finds_memory_in_code_table(
    memory_with_code_embedder: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    code_mem = await memory_with_code_embedder.write(
        principal=alice, text="def f(): return 1", kind="context_code"
    )
    assert (
        await memory_with_code_embedder.forget(
            principal=alice, memory_id=code_mem.memory_id
        )
        is True
    )


@pytest.mark.asyncio
async def test_pin_returns_none_for_missing_id(
    memory_with_code_embedder: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Return None when pinning a missing id."""
    out = await memory_with_code_embedder.pin(
        principal=alice, memory_id="does-not-exist", pinned=True
    )
    assert out is None


@pytest.mark.asyncio
async def test_forget_returns_false_for_missing_id(
    memory_with_code_embedder: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    assert (
        await memory_with_code_embedder.forget(
            principal=alice, memory_id="does-not-exist"
        )
        is False
    )


@pytest.mark.asyncio
async def test_recall_uses_code_qvec_for_code_rows(
    memory_with_code_embedder: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Score code rows against the code embedder's query vector."""
    await memory_with_code_embedder.write(
        principal=alice, text="def foo(): pass", kind="context_code"
    )
    await memory_with_code_embedder.write(principal=alice, text="ordinary user fact")
    out = await memory_with_code_embedder.recall(principal=alice, query="def foo", k=5)
    assert any("foo" in r.text for r in out)


class _OrderingReranker:
    """A reranker that reverses the input order."""

    async def rerank(
        self,
        _query: str,
        candidates: list[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        rev = list(reversed(candidates))
        scored = [(cid, float(i + 1)) for i, (cid, _) in enumerate(rev)]
        if top_k is not None:
            scored = scored[:top_k]
        return scored


class _BrokenReranker:
    async def rerank(self, *_a: object, **_kw: object) -> object:
        raise RuntimeError("rerank down")


@pytest_asyncio.fixture
async def memory_with_reranker(
    tmp_path: Path,
) -> AsyncIterator[SqliteMemoryStore]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'mr.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    store = SqliteMemoryStore(
        url,
        embeddings=HashEmbeddings(dim=64),
        reranker=_OrderingReranker(),  # type: ignore[arg-type]
        rerank_fanout=4,
    )
    try:
        yield store
    finally:
        store.close()
        await history.aclose()


@pytest.mark.asyncio
async def test_reranker_changes_final_order(
    memory_with_reranker: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Let the reranker order win over raw embedding score."""
    for text in ("alpha", "beta", "gamma", "delta"):
        await memory_with_reranker.write(principal=alice, text=text)
    out = await memory_with_reranker.recall(principal=alice, query="alpha", k=4)
    assert len(out) == 4


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_embedding_order(
    tmp_path: Path,
    alice: UserPrincipal,
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'mrf.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    try:
        store = SqliteMemoryStore(
            url,
            embeddings=HashEmbeddings(dim=64),
            reranker=_BrokenReranker(),  # type: ignore[arg-type]
            rerank_fanout=4,
        )
        await store.write(principal=alice, text="banana")
        await store.write(principal=alice, text="orange")
        out = await store.recall(principal=alice, query="banana", k=5)
        assert len(out) == 2
        assert {r.text for r in out} == {"banana", "orange"}
    finally:
        store.close()
        await history.aclose()


@pytest.mark.asyncio
async def test_reranker_promotes_pinned_outside_fanout(
    tmp_path: Path,
    alice: UserPrincipal,
) -> None:
    """Promote a pinned row past rerank_fanout into the rerank pool."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'mrp.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    try:
        captured_pool: list[list[str]] = []

        class _PoolReranker:
            async def rerank(
                self,
                _query: str,
                candidates: list[tuple[str, str]],
                *,
                top_k: int | None = None,
            ) -> list[tuple[str, float]]:
                captured_pool.append([cid for cid, _ in candidates])
                return [(cid, 1.0) for cid, _ in candidates[: top_k or len(candidates)]]

        store = SqliteMemoryStore(
            url,
            embeddings=HashEmbeddings(dim=64),
            reranker=_PoolReranker(),  # type: ignore[arg-type]
            rerank_fanout=1,
        )
        pinned_a = await store.write(principal=alice, text="pinned alpha")
        pinned_b = await store.write(principal=alice, text="pinned beta")
        await store.write(principal=alice, text="banana 1")
        await store.write(principal=alice, text="banana 2")
        await store.pin(principal=alice, memory_id=pinned_a.memory_id, pinned=True)
        await store.pin(principal=alice, memory_id=pinned_b.memory_id, pinned=True)
        out = await store.recall(principal=alice, query="banana", k=1)
        assert captured_pool
        assert pinned_a.memory_id in captured_pool[0]
        assert pinned_b.memory_id in captured_pool[0]
        assert any(r.memory_id in {pinned_a.memory_id, pinned_b.memory_id} for r in out)
    finally:
        store.close()
        await history.aclose()


@pytest.mark.asyncio
async def test_delete_all_for_user_purges_only_that_user(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    bob = UserPrincipal(user_id="bob", scopes=("memory:write",))
    await memory.write(principal=alice, text="alice fact 1")
    await memory.write(principal=alice, text="alice fact 2")
    await memory.write(principal=bob, text="bob fact")
    n = await memory.delete_all_for_user(alice)
    assert n == 2
    assert await memory.list_memories(principal=alice, limit=10) == []
    leftover = await memory.list_memories(principal=bob, limit=10)
    assert [m.text for m in leftover] == ["bob fact"]


@pytest.mark.asyncio
async def test_delete_all_for_user_on_empty_returns_zero(
    memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    assert await memory.delete_all_for_user(alice) == 0


def test_url_passthrough_for_raw_file_path(tmp_path: Path) -> None:
    """Use a bare filesystem path as-is."""
    raw_path = str(tmp_path / "raw.db")
    store = SqliteMemoryStore(raw_path, embeddings=HashEmbeddings(dim=16))
    assert store._db_file == raw_path
    store.close()


@pytest.mark.asyncio
async def test_reranker_skips_unknown_memory_ids(
    tmp_path: Path,
    alice: UserPrincipal,
) -> None:
    """Drop a reranker cid that is not in the pool."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'mr2.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    try:

        class _PhantomReranker:
            async def rerank(
                self,
                _query: str,
                candidates: list[tuple[str, str]],
                *,
                top_k: int | None = None,
            ) -> list[tuple[str, float]]:
                out: list[tuple[str, float]] = [
                    ("does-not-exist", 9.0),
                ]
                out.extend((cid, 1.0) for cid, _ in candidates[: top_k or 1])
                return out

        store = SqliteMemoryStore(
            url,
            embeddings=HashEmbeddings(dim=64),
            reranker=_PhantomReranker(),  # type: ignore[arg-type]
        )
        await store.write(principal=alice, text="real")
        out = await store.recall(principal=alice, query="real", k=2)
        assert len(out) == 1
        assert out[0].text == "real"
    finally:
        store.close()
        await history.aclose()
