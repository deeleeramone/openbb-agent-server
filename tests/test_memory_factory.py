"""Tests for the memory factory."""

from __future__ import annotations

import pytest
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from openbb_agent_server.memory.embeddings import HashEmbeddings
from openbb_agent_server.memory.factory import (
    make_embeddings,
    make_reranker,
    make_translator,
)
from openbb_agent_server.memory.reranker import NvidiaReranker
from openbb_agent_server.memory.translation import NvidiaTranslator


def test_make_embeddings_default_is_hash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out = make_embeddings("hash")
    assert isinstance(out, HashEmbeddings)
    assert any("HashEmbeddings fallback" in r.message for r in caplog.records)


def test_make_embeddings_empty_provider_is_hash() -> None:
    out = make_embeddings("")
    assert isinstance(out, HashEmbeddings)


def test_make_embeddings_none_provider_is_hash() -> None:
    out = make_embeddings(None)
    assert isinstance(out, HashEmbeddings)


def test_make_embeddings_dim_passes_through_to_hash() -> None:
    out = make_embeddings("hash", config={"dim": 32})
    assert isinstance(out, HashEmbeddings)
    assert out.dim == 32


def test_make_embeddings_nvidia_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    out = make_embeddings("nvidia")
    assert isinstance(out, NVIDIAEmbeddings)
    assert out.model == "nvidia/nv-embed-v1"


def test_make_embeddings_nvidia_raises_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        make_embeddings("nvidia")


def test_make_embeddings_nvidia_code_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    out = make_embeddings("nvidia-code")
    assert isinstance(out, NVIDIAEmbeddings)
    assert out.model == "nvidia/nv-embedcode-7b-v1"


def test_make_embeddings_explicit_model_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    out = make_embeddings("nvidia", model="custom-embed-x")
    assert out.model == "custom-embed-x"


def test_make_embeddings_model_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    out = make_embeddings("nvidia", config={"model": "from-cfg"})
    assert out.model == "from-cfg"


def test_make_embeddings_propagates_optional_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    out = make_embeddings(
        "nvidia",
        config={"base_url": "https://x/v1", "dimensions": 512, "truncate": "START"},
    )
    assert "https://x" in out.base_url
    assert out.dimensions == 512
    assert out.truncate == "START"


def test_make_embeddings_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown embeddings provider"):
        make_embeddings("totally-fake")


def test_make_reranker_none_provider_returns_none() -> None:
    assert make_reranker(None) is None


def test_make_reranker_empty_provider_returns_none() -> None:
    assert make_reranker("") is None


def test_make_reranker_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown reranker provider"):
        make_reranker("nope")


def test_make_reranker_nvidia_with_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    r = make_reranker("nvidia")
    assert isinstance(r, NvidiaReranker)
    assert r._api_key == "env-key"


def test_make_reranker_nvidia_with_config_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fallback")
    r = make_reranker(
        "nvidia",
        model="custom-rerank",
        config={
            "api_key": "cfg-key",
            "base_url": "https://x",
            "truncate": "START",
            "top_n": 7,
        },
    )
    assert r is not None
    assert r._api_key == "cfg-key"
    assert r._model == "custom-rerank"
    assert r._base_url == "https://x"
    assert r._truncate == "START"
    assert r._top_n == 7


def test_make_translator_none_provider_returns_none() -> None:
    assert make_translator(None) is None


def test_make_translator_empty_provider_returns_none() -> None:
    assert make_translator("") is None


def test_make_translator_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown translation provider"):
        make_translator("nope")


def test_make_translator_nvidia_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    t = make_translator("nvidia")
    assert isinstance(t, NvidiaTranslator)
    assert t._model == "nvidia/riva-translate-4b-instruct-v1_1"
    assert t._api_key == "env-key"


def test_make_translator_nvidia_with_config_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    t = make_translator(
        "nvidia",
        model="custom-translate",
        config={
            "api_key": "cfg-key",
            "base_url": "https://x",
            "temperature": 0.3,
            "max_tokens": 128,
        },
    )
    assert t is not None
    assert t._api_key == "cfg-key"
    assert t._model == "custom-translate"
    assert t._base_url == "https://x"
    assert t._temperature == 0.3
    assert t._max_tokens == 128
