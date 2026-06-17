"""tool_filter middleware tests."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from openbb_agent_server.plugins.middleware.tool_filter import (
    ToolFilterMiddlewareFactory,
    _tool_name,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _Request:
    """Mimics langchain's model-call request with tools and override."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools

    def override(self, *, tools: list[Any]) -> _Request:
        return _Request(tools)


def test_tool_name_extracts_from_object() -> None:
    assert _tool_name(_Tool("x")) == "x"


def test_tool_name_extracts_from_dict() -> None:
    assert _tool_name({"name": "x"}) == "x"


def test_tool_name_returns_none_for_dict_without_name() -> None:
    assert _tool_name({"description": "x"}) is None


def test_tool_name_returns_none_for_dict_with_non_string_name() -> None:
    assert _tool_name({"name": 42}) is None


def test_tool_name_returns_none_for_object_without_name() -> None:
    class _Bare:
        pass

    assert _tool_name(_Bare()) is None


def test_tool_name_returns_none_for_object_with_non_string_name() -> None:
    t = _Tool(name="x")
    t.name = 42  # type: ignore[assignment]
    assert _tool_name(t) is None


def test_factory_name() -> None:
    assert ToolFilterMiddlewareFactory.name == "tool_filter"


def test_build_uses_default_excluded_when_config_missing() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {})
    assert "read_file" in mw._excluded
    assert "write_file" in mw._excluded
    assert "glob" in mw._excluded


def test_build_uses_config_excluded_list() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": ["foo", "bar"]})
    assert mw._excluded == frozenset({"foo", "bar"})


def test_build_uses_config_excluded_set() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": {"foo"}})
    assert mw._excluded == frozenset({"foo"})


def test_build_uses_config_excluded_tuple() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": ("a", "b")})
    assert mw._excluded == frozenset({"a", "b"})


def test_build_filters_non_string_entries() -> None:
    mw = ToolFilterMiddlewareFactory().build(
        _ctx(), {"excluded": ["foo", 42, None, "bar"]}
    )
    assert mw._excluded == frozenset({"foo", "bar"})


def test_build_invalid_excluded_falls_back_to_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": "not-a-list"})
    assert "read_file" in mw._excluded
    assert any("tool_filter.excluded" in r.message for r in caplog.records)


def test_build_empty_excluded_list_yields_no_filtering() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": []})
    assert mw._excluded == frozenset()


def test_wrap_model_call_drops_excluded_tool() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": ["bad"]})
    request = _Request([_Tool("good"), _Tool("bad"), _Tool("also_good")])

    seen: list[Any] = []

    def handler(req: Any) -> str:
        seen.append(req)
        return "ok"

    out = mw.wrap_model_call(request, handler)
    assert out == "ok"
    [used] = seen
    names = [t.name for t in used.tools]
    assert names == ["good", "also_good"]


def test_wrap_model_call_passes_through_when_nothing_excluded() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": ["bad"]})
    request = _Request([_Tool("good")])

    seen: list[Any] = []

    def handler(req: Any) -> str:
        seen.append(req)
        return "ok"

    mw.wrap_model_call(request, handler)
    [used] = seen
    assert used is request


def test_wrap_model_call_skips_filter_when_excluded_is_empty() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": []})
    request = _Request([_Tool("a"), _Tool("b")])

    seen: list[Any] = []

    def handler(req: Any) -> str:
        seen.append(req)
        return "ok"

    mw.wrap_model_call(request, handler)
    [used] = seen
    assert used is request


def test_filter_logs_only_once(caplog: pytest.LogCaptureFixture) -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": ["bad"]})

    def handler(req: Any) -> str:
        return "ok"

    with caplog.at_level(logging.DEBUG):
        mw.wrap_model_call(_Request([_Tool("good"), _Tool("bad")]), handler)
        mw.wrap_model_call(_Request([_Tool("good"), _Tool("bad")]), handler)

    removed_logs = [r for r in caplog.records if "tool_filter: removed" in r.message]
    assert len(removed_logs) == 1


@pytest.mark.asyncio
async def test_awrap_model_call_drops_excluded_tool() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": ["bad"]})
    request = _Request([_Tool("good"), _Tool("bad")])

    seen: list[Any] = []

    async def handler(req: Any) -> str:
        seen.append(req)
        return "ok"

    out = await mw.awrap_model_call(request, handler)
    assert out == "ok"
    [used] = seen
    assert [t.name for t in used.tools] == ["good"]


@pytest.mark.asyncio
async def test_awrap_model_call_skips_filter_when_excluded_empty() -> None:
    mw = ToolFilterMiddlewareFactory().build(_ctx(), {"excluded": []})
    request = _Request([_Tool("a")])

    seen: list[Any] = []

    async def handler(req: Any) -> str:
        seen.append(req)
        return "ok"

    await mw.awrap_model_call(request, handler)
    [used] = seen
    assert used is request
