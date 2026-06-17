"""Tests for the ``rerank`` and ``translate`` tool sources."""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.plugins.tools.rerank import (
    NvidiaRerankToolSource,
    _decode_if_string,
)
from openbb_agent_server.plugins.tools.translate import NvidiaTranslateToolSource
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(*, api_key: str | None = None) -> RunContext:
    keys: dict[str, str] = {}
    if api_key is not None:
        keys["NVIDIA_API_KEY"] = api_key
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=keys,
    )


def test_decode_if_string_passes_non_strings_through() -> None:
    assert _decode_if_string([1, 2, 3]) == [1, 2, 3]


def test_decode_if_string_parses_json_list() -> None:
    assert _decode_if_string('["a", "b"]') == ["a", "b"]


def test_decode_if_string_returns_original_on_bad_json() -> None:
    assert _decode_if_string("not json") == "not json"


@pytest.mark.asyncio
async def test_rerank_tool_returns_empty_for_no_candidates() -> None:
    src = NvidiaRerankToolSource(api_key="k")
    [tool] = await src.tools(_ctx(), {})
    out = await tool.ainvoke({"query": "x", "candidates": [], "top_k": 5})
    assert out == []


@pytest.mark.asyncio
async def test_rerank_tool_returns_ordered_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A working reranker produces ``{index, score, text}`` entries."""

    from openbb_agent_server.memory import reranker as reranker_mod

    async def fake_rerank(
        _self: Any,
        _q: str,
        candidates: list[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        rev = list(reversed(candidates))
        return [(cid, float(i + 1)) for i, (cid, _) in enumerate(rev)]

    monkeypatch.setattr(reranker_mod.NvidiaReranker, "rerank", fake_rerank)
    src = NvidiaRerankToolSource(api_key="k")
    [tool] = await src.tools(_ctx(), {})
    out = await tool.ainvoke(
        {"query": "q", "candidates": ["first", "second", "third"], "top_k": 5}
    )
    assert [r["text"] for r in out] == ["third", "second", "first"]
    assert all(isinstance(r["index"], int) for r in out)


@pytest.mark.asyncio
async def test_rerank_tool_falls_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from openbb_agent_server.memory import reranker as reranker_mod

    async def boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("nope")

    monkeypatch.setattr(reranker_mod.NvidiaReranker, "rerank", boom)
    src = NvidiaRerankToolSource(api_key="k")
    [tool] = await src.tools(_ctx(), {})
    out = await tool.ainvoke({"query": "q", "candidates": ["a", "b", "c"], "top_k": 2})
    assert len(out) == 2
    assert [r["text"] for r in out] == ["a", "b"]
    assert all(r["score"] == 0.0 for r in out)
    assert any("rerank tool failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_rerank_respects_config_overrides() -> None:
    src = NvidiaRerankToolSource()
    out = await src.tools(
        _ctx(api_key="ctx"),
        {"model": "custom", "base_url": "https://x", "truncate": "START"},
    )
    assert len(out) == 1


@pytest.mark.asyncio
async def test_rerank_pulls_key_from_ctx_priority_over_self() -> None:
    src = NvidiaRerankToolSource(api_key="default")
    tools = await src.tools(_ctx(api_key="from-ctx"), {})
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_translate_tool_returns_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.memory import translation

    async def fake_translate(
        _self: Any,
        text: str,
        *,
        source_language: str = "auto",
        target_language: str = "English",
    ) -> str:
        return f"TRANSLATED-{text}-{target_language}"

    monkeypatch.setattr(translation.NvidiaTranslator, "translate", fake_translate)
    src = NvidiaTranslateToolSource(api_key="k")
    [tool] = await src.tools(_ctx(), {})
    out = await tool.ainvoke(
        {"text": "bonjour", "source_language": "auto", "target_language": "English"}
    )
    assert out == "TRANSLATED-bonjour-English"


@pytest.mark.asyncio
async def test_translate_tool_handles_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from openbb_agent_server.memory import translation

    async def boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("translator down")

    monkeypatch.setattr(translation.NvidiaTranslator, "translate", boom)
    src = NvidiaTranslateToolSource(api_key="k")
    [tool] = await src.tools(_ctx(), {})
    out = await tool.ainvoke({"text": "x"})
    assert out.startswith("translation failed:")
    assert any("translate tool failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_translate_uses_config_overrides() -> None:
    src = NvidiaTranslateToolSource()
    out = await src.tools(
        _ctx(api_key="ctx"),
        {
            "model": "alt",
            "base_url": "https://x",
            "temperature": 0.1,
            "max_tokens": 256,
        },
    )
    assert len(out) == 1
