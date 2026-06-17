"""Live integration tests for the embedding / reranker backends."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from openbb_agent_server.memory.embeddings import cosine
from openbb_agent_server.memory.reranker import NvidiaReranker
from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.plugins.tools.gemini_embeddings import (
    GeminiEmbeddingsToolSource,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal

_NIM_DEADLINE_S = 30.0
_HEAVY_NIM_DEADLINE_S = 90.0


pytestmark = pytest.mark.timeout(120)


nvidia_required = pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set — live NIM tests skipped",
)


@nvidia_required
async def test_nvidia_embeddings_returns_real_vectors() -> None:
    """Return real vectors from the public NIM endpoint."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
        vecs = await embedder.aembed_documents(["openbb workspace agent"])
    assert len(vecs) == 1
    assert isinstance(vecs[0], list)
    assert len(vecs[0]) > 0
    assert all(isinstance(x, float) for x in vecs[0])


@nvidia_required
async def test_nvidia_embeddings_returns_empty_for_empty_input() -> None:
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    assert await embedder.aembed_documents([]) == []


@nvidia_required
async def test_nvidia_embeddings_semantically_related_texts_cluster() -> None:
    """Cluster semantically related texts."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
        vecs = await embedder.aembed_documents(
            [
                "Apple Inc earnings",
                "AAPL quarterly fundamentals",
                "Toronto restaurant tips",
            ]
        )
    sim_related = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])
    assert sim_related > sim_unrelated, (
        f"semantic geometry failure: related={sim_related:.3f} "
        f"unrelated={sim_unrelated:.3f}"
    )


@nvidia_required
async def test_nvidia_embeddings_query_path() -> None:
    """Return a single vector from aembed_query."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
        vec = await embedder.aembed_query("openbb workspace agent")
    assert isinstance(vec, list)
    assert len(vec) >= 1024
    assert all(isinstance(x, float) for x in vec)


@pytest_asyncio.fixture
async def nim_memory(tmp_path: Path) -> AsyncIterator[SqliteMemoryStore]:
    """Sqlite-backed memory store wired to live NV-Embed."""
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set")
    url = f"sqlite+aiosqlite:///{tmp_path / 'nim-m.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    store = SqliteMemoryStore(url, embeddings=embedder)
    try:
        yield store
    finally:
        await history.aclose()


@nvidia_required
async def test_nim_backed_store_recalls_paraphrased_query(
    nim_memory: SqliteMemoryStore,
    alice: UserPrincipal,
) -> None:
    """Recall a paraphrased query against a NIM-backed store."""
    async with asyncio.timeout(_HEAVY_NIM_DEADLINE_S):
        facts = [
            "user follows Apple Inc quarterly results",
            "user lives in Toronto and prefers Cantonese food",
            "user's daily coffee order is a flat white with oat milk",
        ]
        for fact in facts:
            await nim_memory.write(principal=alice, text=fact)

        paraphrase = "What companies does this person track in their portfolio?"
        out = await nim_memory.recall(principal=alice, query=paraphrase, k=3)
    assert out, "recall returned no memories"
    assert "Apple" in out[0].text, (
        f"semantic recall regression — got top={out[0].text!r} for query={paraphrase!r}"
    )


@nvidia_required
async def test_nvidia_reranker_orders_candidates() -> None:
    """Order candidates by semantic relevance."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        reranker = NvidiaReranker(model="nv-rerank-qa-mistral-4b:1")
        candidates = [
            ("apple", "Apple Inc beat earnings expectations this quarter."),
            ("weather", "It's raining in Vancouver this weekend."),
            ("aapl", "AAPL revenue grew 8% year over year."),
        ]
        ranked = await reranker.rerank(
            query="How did Apple's quarterly earnings perform?",
            candidates=candidates,
            top_k=3,
        )
    assert len(ranked) == 3
    ids = [cid for cid, _ in ranked]
    weather_pos = ids.index("weather")
    assert weather_pos == 2, f"weather should be last; got {ids}"


@nvidia_required
async def test_nvidia_reranker_handles_empty_candidates() -> None:
    reranker = NvidiaReranker(model="nv-rerank-qa-mistral-4b:1")
    assert await reranker.rerank(query="x", candidates=[]) == []


@nvidia_required
async def test_nvidia_reranker_handles_empty_query() -> None:
    """Handle an empty query."""
    reranker = NvidiaReranker(model="nv-rerank-qa-mistral-4b:1")
    out = await reranker.rerank(
        query="   ",
        candidates=[("a", "first"), ("b", "second")],
        top_k=2,
    )
    assert out == [("a", 0.0), ("b", 0.0)]


@nvidia_required
async def test_nim_store_with_reranker_runs_two_stage_retrieval(
    tmp_path: Path,
    alice: UserPrincipal,
) -> None:
    """Run two-stage retrieval against a NIM-backed store."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'two-stage.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    try:
        async with asyncio.timeout(_HEAVY_NIM_DEADLINE_S):
            embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
            reranker = NvidiaReranker(model="nv-rerank-qa-mistral-4b:1")
            store = SqliteMemoryStore(
                url,
                embeddings=embedder,
                reranker=reranker,
                rerank_fanout=8,
            )
            for fact in [
                "user trades Apple options on quarterly earnings",
                "user reviews Tesla deliveries each month",
                "user's favourite hobby is sailing on Lake Ontario",
            ]:
                await store.write(principal=alice, text=fact)

            out = await store.recall(
                principal=alice,
                query="Which equities does the user actively trade options on?",
                k=2,
            )
        assert out, "two-stage recall returned nothing"
        assert "Apple" in out[0].text
        assert out[0].score is not None
    finally:
        await history.aclose()


gemini_required = pytest.mark.skipif(
    not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY / GEMINI_API_KEY not set — Gemini tests skipped",
)


def _gemini_key() -> str:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def _gemini_ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u-gem"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys={"GOOGLE_API_KEY": _gemini_key()},
    )


@gemini_required
async def test_gemini_embed_text_returns_real_vectors() -> None:
    """Return real vectors from the embed_text tool."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        src = GeminiEmbeddingsToolSource()
        tools = await src.tools(_gemini_ctx(), {})
        [tool] = tools

        out = await tool.ainvoke(
            {
                "texts": ["openbb workspace memory recall"],
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": 768,
            }
        )
    assert isinstance(out, dict)
    assert out["count"] == 1
    assert out["dimensions"] == 768
    assert out["task_type"] == "RETRIEVAL_DOCUMENT"
    [vec] = out["vectors"]
    assert isinstance(vec, list)
    assert len(vec) == 768
    assert all(isinstance(x, float) for x in vec)


@gemini_required
async def test_gemini_embed_text_semantically_separates_topics() -> None:
    """Separate topics semantically via Gemini embeddings."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        src = GeminiEmbeddingsToolSource()
        tools = await src.tools(_gemini_ctx(), {})
        [tool] = tools
        out = await tool.ainvoke(
            {
                "texts": [
                    "How did Apple report earnings this quarter?",
                    "AAPL beat consensus on EPS for Q2.",
                    "What's the weather like in Vancouver tomorrow?",
                ],
                "task_type": "SEMANTIC_SIMILARITY",
            }
        )
    v_apple, v_aapl, v_weather = out["vectors"]
    sim_pair = cosine(v_apple, v_aapl)
    sim_unrelated = cosine(v_apple, v_weather)
    assert sim_pair > sim_unrelated, (
        f"gemini semantic geometry failure: related={sim_pair:.3f} "
        f"unrelated={sim_unrelated:.3f}"
    )


@gemini_required
async def test_gemini_embed_text_handles_empty_input() -> None:
    """Handle empty input without a network call."""
    src = GeminiEmbeddingsToolSource()
    tools = await src.tools(_gemini_ctx(), {})
    [tool] = tools
    out = await tool.ainvoke({"texts": []})
    assert out == {
        "vectors": [],
        "model": "gemini-embedding-001",
        "dimensions": 0,
    }


@gemini_required
async def test_gemini_embed_text_rejects_unknown_task_type() -> None:
    """Reject an unknown task_type."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        src = GeminiEmbeddingsToolSource()
        tools = await src.tools(_gemini_ctx(), {})
        [tool] = tools
        with pytest.raises(Exception, match="task_type"):
            await tool.ainvoke({"texts": ["hi"], "task_type": "NOT_A_REAL_TASK"})
