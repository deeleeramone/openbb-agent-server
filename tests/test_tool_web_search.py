"""web_search tool source tests."""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.plugins.tools import web_search
from openbb_agent_server.plugins.tools.web_search import WebSearchToolSource
from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(
    api_keys: dict[str, str] | None = None,
    *,
    search_enabled: bool = True,
) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=api_keys or {},
        workspace_options={"search-web": True} if search_enabled else {},
    )


@pytest.fixture
def captured_emits(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    monkeypatch.setattr(emit, "_writer", lambda: out.append)
    return out


@pytest.mark.asyncio
async def test_default_provider_is_duckduckgo() -> None:
    src = WebSearchToolSource()
    tools = await src.tools(_ctx(), {})
    assert len(tools) == 1
    assert tools[0].name == "web_search"


@pytest.mark.asyncio
async def test_web_search_filtered_when_user_has_not_opted_in() -> None:
    """Return no tools when the search-web feature is off."""
    src = WebSearchToolSource()
    tools = await src.tools(_ctx(search_enabled=False), {})
    assert tools == []


@pytest.mark.asyncio
async def test_search_emits_citations_for_results(
    monkeypatch: pytest.MonkeyPatch,
    captured_emits: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(
        web_search,
        "_ddg_search",
        lambda q, k: [
            {"title": "Page A", "url": "http://a", "snippet": "snippet a"},
            {"title": "Page B", "url": "http://b", "snippet": "snippet b"},
        ],
    )
    src = WebSearchToolSource()
    [tool] = await src.tools(_ctx(), {})
    results = tool.invoke({"query": "anything", "k": 2})
    assert len(results) == 2
    citations_emits = [e for e in captured_emits if e["type"] == "citations"]
    assert len(citations_emits) == 2


@pytest.mark.asyncio
async def test_tavily_provider_requires_api_key() -> None:
    src = WebSearchToolSource(provider="tavily")
    with pytest.raises(RuntimeError):
        await src.tools(_ctx(), {})


@pytest.mark.asyncio
async def test_provider_overridable_via_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = WebSearchToolSource(provider="tavily")
    monkeypatch.setattr(web_search, "_ddg_search", lambda q, k: [])
    tools = await src.tools(_ctx(api_keys={}), {"provider": "duckduckgo"})
    assert tools[0].name == "web_search"


def test_web_search_ddg_search_yields_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from openbb_agent_server.plugins.tools import web_search

    class _StubDDGS:
        def __enter__(self) -> _StubDDGS:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def text(self, query: str, max_results: int) -> list[dict[str, str]]:
            return [
                {"title": "T1", "href": "https://x.test/1", "body": "snip"},
                {"title": "T2", "link": "https://x.test/2", "snippet": "snip2"},
            ][:max_results]

    monkeypatch.setattr("ddgs.DDGS", _StubDDGS)
    out = web_search._ddg_search("apple", k=2)
    assert out == [
        {"title": "T1", "url": "https://x.test/1", "snippet": "snip"},
        {"title": "T2", "url": "https://x.test/2", "snippet": "snip2"},
    ]


@pytest.mark.asyncio
async def test_web_search_tool_runs_against_stubbed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import web_search
    from openbb_agent_server.plugins.tools.web_search import WebSearchToolSource
    from openbb_agent_server.runtime.context import (
        RunContext as _RC,
        bind,
        runtime_state,
    )

    monkeypatch.setattr(
        web_search,
        "_ddg_search",
        lambda q, k: [{"title": "T", "url": "u", "snippet": "s"}],
    )
    ctx = _RC(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        workspace_options={"search-web": True},
    )
    src = WebSearchToolSource()
    [tool] = await src.tools(ctx, {})

    with bind(ctx):
        runtime_state()  # ensure scratch dict exists for emit
        out = await tool.ainvoke({"query": "apple", "k": 1})
    assert out == [{"title": "T", "url": "u", "snippet": "s"}]


def test_web_search_tavily_requires_key() -> None:
    from openbb_agent_server.plugins.tools.web_search import _build_search_callable

    with pytest.raises(RuntimeError, match="tavily"):
        _build_search_callable("tavily", {})


def test_web_search_ddg_falls_back_to_duckduckgo_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through to duckduckgo_search when ddgs is unimportable."""
    import sys

    from openbb_agent_server.plugins.tools import web_search

    sentinel_results: list[dict[str, str]] = [{"title": "T", "href": "u", "body": "s"}]

    class _StubDDGS:
        def __enter__(self) -> _StubDDGS:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def text(self, query: str, max_results: int) -> list[dict[str, str]]:
            return sentinel_results

    fake_module = type("M", (), {"DDGS": _StubDDGS})
    monkeypatch.setitem(sys.modules, "ddgs", None)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_module)

    out = web_search._ddg_search("apple", k=1)
    assert out == [{"title": "T", "url": "u", "snippet": "s"}]


def test_web_search_tavily_returns_callable_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the tavily callable when a key is present."""
    from openbb_agent_server.plugins.tools.web_search import _build_search_callable

    callable_ = _build_search_callable("tavily", {"TAVILY_API_KEY": "stub"})
    assert callable(callable_)
