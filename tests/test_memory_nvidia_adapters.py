"""Unit tests for the NVIDIA reranker + translator adapters."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from openbb_agent_server.memory.reranker import NvidiaReranker
from openbb_agent_server.memory.translation import NvidiaTranslator


class _StubChatNVIDIA:
    instances: list[_StubChatNVIDIA] = []
    fake_reply: Any = "bonjour"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _StubChatNVIDIA.instances.append(self)

    async def ainvoke(self, _messages: Any) -> Any:
        class _Resp:
            def __init__(self, content: Any) -> None:
                self.content = content

        return _Resp(_StubChatNVIDIA.fake_reply)

    def invoke(self, _messages: Any) -> Any:
        class _Resp:
            def __init__(self, content: Any) -> None:
                self.content = content

        return _Resp(_StubChatNVIDIA.fake_reply)


class _NoAinvokeChatNVIDIA:
    instances: list[_NoAinvokeChatNVIDIA] = []
    fake_reply: Any = "from sync"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _NoAinvokeChatNVIDIA.instances.append(self)

    def invoke(self, _messages: Any) -> Any:
        class _Resp:
            def __init__(self, content: Any) -> None:
                self.content = content

        return _Resp(_NoAinvokeChatNVIDIA.fake_reply)


class _StubNVIDIARerank:
    instances: list[_StubNVIDIARerank] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _StubNVIDIARerank.instances.append(self)

    async def acompress_documents(
        self, *, documents: list[Any], query: str
    ) -> list[Any]:
        scored = list(documents)
        for i, d in enumerate(scored):
            d.metadata["relevance_score"] = float(len(scored) - i)
        return scored

    def compress_documents(self, *, documents: list[Any], query: str) -> list[Any]:
        return self.acompress_documents_sync(documents=documents, query=query)

    def acompress_documents_sync(
        self, *, documents: list[Any], query: str
    ) -> list[Any]:
        scored = list(documents)
        for i, d in enumerate(scored):
            d.metadata["relevance_score"] = float(len(scored) - i)
        return scored


class _NoAcompressNVIDIARerank:
    """Reranker without an async method."""

    instances: list[_NoAcompressNVIDIARerank] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _NoAcompressNVIDIARerank.instances.append(self)

    def compress_documents(self, *, documents: list[Any], query: str) -> list[Any]:
        scored = list(documents)
        for i, d in enumerate(scored):
            d.metadata["relevance_score"] = float(len(scored) - i)
        return scored


def _install_nvidia(monkeypatch: pytest.MonkeyPatch, **attrs: Any) -> None:
    mod = types.ModuleType("langchain_nvidia_ai_endpoints")
    for name, val in attrs.items():
        setattr(mod, name, val)
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", mod)


@pytest.mark.asyncio
async def test_translator_empty_input_returns_empty() -> None:
    t = NvidiaTranslator(api_key="k")
    assert await t.translate("") == ""
    assert await t.translate("   ") == ""


@pytest.mark.asyncio
async def test_translator_builds_client_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubChatNVIDIA.instances.clear()
    _install_nvidia(monkeypatch, ChatNVIDIA=_StubChatNVIDIA)
    t = NvidiaTranslator(api_key="k")
    await t.translate("hola")
    await t.translate("ola")
    assert len(_StubChatNVIDIA.instances) == 1


@pytest.mark.asyncio
async def test_translator_passes_optional_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubChatNVIDIA.instances.clear()
    _install_nvidia(monkeypatch, ChatNVIDIA=_StubChatNVIDIA)
    t = NvidiaTranslator(
        api_key="k",
        base_url="https://x.example",
        max_tokens=128,
        temperature=0.3,
    )
    await t.translate("hola")
    [c] = _StubChatNVIDIA.instances
    assert c.kwargs["base_url"] == "https://x.example"
    assert c.kwargs["max_tokens"] == 128
    assert c.kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_translator_omits_max_tokens_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubChatNVIDIA.instances.clear()
    _install_nvidia(monkeypatch, ChatNVIDIA=_StubChatNVIDIA)
    t = NvidiaTranslator(api_key="k", max_tokens=None)
    await t.translate("hola")
    [c] = _StubChatNVIDIA.instances
    assert "max_tokens" not in c.kwargs


@pytest.mark.asyncio
async def test_translator_raises_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    _install_nvidia(monkeypatch, ChatNVIDIA=_StubChatNVIDIA)
    t = NvidiaTranslator()
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        await t.translate("hola")


@pytest.mark.asyncio
async def test_translator_returns_string_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubChatNVIDIA.instances.clear()
    _StubChatNVIDIA.fake_reply = "hello"
    _install_nvidia(monkeypatch, ChatNVIDIA=_StubChatNVIDIA)
    t = NvidiaTranslator(api_key="k")
    out = await t.translate("bonjour")
    assert out == "hello"


@pytest.mark.asyncio
async def test_translator_extracts_text_from_block_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubChatNVIDIA.instances.clear()
    _StubChatNVIDIA.fake_reply = [
        {"type": "text", "text": "hel"},
        "lo",
        {"type": "image", "url": "skip"},
    ]
    _install_nvidia(monkeypatch, ChatNVIDIA=_StubChatNVIDIA)
    t = NvidiaTranslator(api_key="k")
    out = await t.translate("bonjour")
    assert out == "hello"


@pytest.mark.asyncio
async def test_translator_none_content_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubChatNVIDIA.instances.clear()
    _StubChatNVIDIA.fake_reply = None
    _install_nvidia(monkeypatch, ChatNVIDIA=_StubChatNVIDIA)
    t = NvidiaTranslator(api_key="k")
    out = await t.translate("bonjour")
    assert out == ""


@pytest.mark.asyncio
async def test_translator_falls_back_to_sync_invoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _NoAinvokeChatNVIDIA.instances.clear()
    _install_nvidia(monkeypatch, ChatNVIDIA=_NoAinvokeChatNVIDIA)
    t = NvidiaTranslator(api_key="k")
    out = await t.translate("bonjour")
    assert out == "from sync"


@pytest.mark.asyncio
async def test_reranker_empty_candidates_returns_empty() -> None:
    r = NvidiaReranker(api_key="k")
    assert await r.rerank("q", []) == []


@pytest.mark.asyncio
async def test_reranker_blank_query_returns_input_order_with_zero_scores() -> None:
    r = NvidiaReranker(api_key="k")
    out = await r.rerank("", [("a", "text-a"), ("b", "text-b")])
    assert out == [("a", 0.0), ("b", 0.0)]


@pytest.mark.asyncio
async def test_reranker_blank_query_respects_top_k() -> None:
    r = NvidiaReranker(api_key="k")
    out = await r.rerank("", [("a", "x"), ("b", "y"), ("c", "z")], top_k=2)
    assert out == [("a", 0.0), ("b", 0.0)]


@pytest.mark.asyncio
async def test_reranker_uses_acompress_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubNVIDIARerank.instances.clear()
    _install_nvidia(monkeypatch, NVIDIARerank=_StubNVIDIARerank)
    r = NvidiaReranker(api_key="k", base_url="https://x", top_n=10)
    out = await r.rerank("q", [("a", "ta"), ("b", "tb")])
    assert [cid for cid, _ in out] == ["a", "b"]
    [c] = _StubNVIDIARerank.instances
    assert c.kwargs["base_url"] == "https://x"
    assert c.kwargs["top_n"] == 10


@pytest.mark.asyncio
async def test_reranker_top_k_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    _StubNVIDIARerank.instances.clear()
    _install_nvidia(monkeypatch, NVIDIARerank=_StubNVIDIARerank)
    r = NvidiaReranker(api_key="k")
    out = await r.rerank("q", [("a", "x"), ("b", "y"), ("c", "z")], top_k=2)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_reranker_falls_back_to_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _NoAcompressNVIDIARerank.instances.clear()
    _install_nvidia(monkeypatch, NVIDIARerank=_NoAcompressNVIDIARerank)
    r = NvidiaReranker(api_key="k")
    out = await r.rerank("q", [("a", "x")])
    assert out == [("a", 1.0)]


@pytest.mark.asyncio
async def test_reranker_skips_docs_without_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingIdRerank:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def acompress_documents(
            self, *, documents: list[Any], query: str
        ) -> list[Any]:
            from langchain_core.documents import Document

            return [
                Document(
                    page_content="x",
                    metadata={"relevance_score": 9.0},
                )
            ]

    _install_nvidia(monkeypatch, NVIDIARerank=_MissingIdRerank)
    r = NvidiaReranker(api_key="k")
    out = await r.rerank("q", [("a", "ta")])
    assert out == []


@pytest.mark.asyncio
async def test_reranker_uses_score_when_relevance_score_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ScoreOnlyRerank:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def acompress_documents(
            self, *, documents: list[Any], query: str
        ) -> list[Any]:
            for d in documents:
                d.metadata["score"] = 0.5
            return list(documents)

    _install_nvidia(monkeypatch, NVIDIARerank=_ScoreOnlyRerank)
    r = NvidiaReranker(api_key="k")
    out = await r.rerank("q", [("a", "x")])
    assert out == [("a", 0.5)]


@pytest.mark.asyncio
async def test_reranker_returns_zero_when_score_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadScoreRerank:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def acompress_documents(
            self, *, documents: list[Any], query: str
        ) -> list[Any]:
            for d in documents:
                d.metadata["relevance_score"] = "not a number"
            return list(documents)

    _install_nvidia(monkeypatch, NVIDIARerank=_BadScoreRerank)
    r = NvidiaReranker(api_key="k")
    out = await r.rerank("q", [("a", "x")])
    assert out == [("a", 0.0)]


@pytest.mark.asyncio
async def test_reranker_raises_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    _install_nvidia(monkeypatch, NVIDIARerank=_StubNVIDIARerank)
    r = NvidiaReranker()
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        await r.rerank("q", [("a", "x")])
