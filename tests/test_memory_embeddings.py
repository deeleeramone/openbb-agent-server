"""Embedding adapter tests."""

from __future__ import annotations

import pytest

from openbb_agent_server.memory.embeddings import HashEmbeddings, cosine


def test_dim_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HashEmbeddings(dim=0)
    with pytest.raises(ValueError):
        HashEmbeddings(dim=-3)


def test_embed_documents_returns_vectors_of_configured_dim() -> None:
    e = HashEmbeddings(dim=32)
    [v1, v2] = e.embed_documents(["hello world", "different text"])
    assert len(v1) == 32
    assert len(v2) == 32


def test_embed_query_returns_one_vector() -> None:
    e = HashEmbeddings(dim=32)
    v = e.embed_query("hello world")
    assert len(v) == 32


def test_embed_is_deterministic() -> None:
    e = HashEmbeddings(dim=64)
    [a] = e.embed_documents(["repeatable"])
    [b] = e.embed_documents(["repeatable"])
    assert a == b


@pytest.mark.asyncio
async def test_aembed_documents_delegates_to_sync() -> None:
    """Run the sync path in an executor for the async API."""
    e = HashEmbeddings(dim=16)
    vecs = await e.aembed_documents(["a", "b"])
    assert len(vecs) == 2
    assert all(len(v) == 16 for v in vecs)


@pytest.mark.asyncio
async def test_aembed_query_delegates_to_sync() -> None:
    e = HashEmbeddings(dim=16)
    v = await e.aembed_query("anything")
    assert len(v) == 16


def test_cosine_handles_empty_inputs() -> None:
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0], []) == 0.0
    assert cosine([], []) == 0.0


def test_cosine_returns_one_for_parallel_vectors() -> None:
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


def test_cosine_returns_zero_for_orthogonal_vectors() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_truncates_to_shorter_length() -> None:
    val = cosine([1.0, 0.0, 0.0], [2.0, 0.0])
    assert val == pytest.approx(1.0)
