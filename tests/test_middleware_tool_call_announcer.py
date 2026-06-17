"""tool_call_announcer middleware tests."""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.plugins.middleware.tool_call_announcer import (
    ToolCallAnnouncerMiddlewareFactory,
)
from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    monkeypatch.setattr(emit, "_writer", lambda: out.append)
    return out


class _Request:
    """Old-shape request with an object-style tool_call."""

    def __init__(self, *, tool_name: str, args: Any) -> None:
        self.tool_name = tool_name
        self.args = args


class _LcStyleRequest:
    """LangChain v1 shape where tool_call is a dict."""

    def __init__(self, *, name: str, args: dict[str, Any]) -> None:
        self.tool_call = {"name": name, "args": args, "id": "abc"}


@pytest.mark.asyncio
async def test_announces_tool_name_only_no_args(
    captured: list[dict[str, Any]],
) -> None:
    factory = ToolCallAnnouncerMiddlewareFactory()
    mw = factory.build(_ctx(), {})

    async def handler(_req: Any) -> str:
        return "ok"

    await mw.awrap_tool_call(
        _Request(tool_name="snowflake_query", args={"sql": "SELECT 1"}),
        handler,
    )
    [step] = captured
    assert step["type"] == "step"
    assert step["event_type"] == "INFO"
    assert step["message"] == "Calling tool: snowflake_query"
    assert step["details"] == {"tool_name": "snowflake_query"}
    assert "sql" not in str(step)
    assert "SELECT 1" not in str(step)


@pytest.mark.asyncio
async def test_announces_when_args_are_huge(
    captured: list[dict[str, Any]],
) -> None:
    """A large arg does not inflate the announce step."""
    mw = ToolCallAnnouncerMiddlewareFactory().build(_ctx(), {})

    async def handler(_req: Any) -> str:
        return "ok"

    await mw.awrap_tool_call(
        _Request(tool_name="cortex_complete", args={"text": "x" * 5000}),
        handler,
    )
    [step] = captured
    assert step["message"] == "Calling tool: cortex_complete"
    assert step["details"] == {"tool_name": "cortex_complete"}
    assert len(step["message"]) < 60


@pytest.mark.asyncio
async def test_announces_on_lc_v1_dict_shape(
    captured: list[dict[str, Any]],
) -> None:
    """Announce on the LangChain v1 dict-shaped tool_call."""
    mw = ToolCallAnnouncerMiddlewareFactory().build(_ctx(), {})

    async def handler(_req: Any) -> str:
        return "ok"

    await mw.awrap_tool_call(
        _LcStyleRequest(name="emit_table_artifact", args={"columns": ["a"]}),
        handler,
    )
    [step] = captured
    assert step["message"] == "Calling tool: emit_table_artifact"
    assert step["details"]["tool_name"] == "emit_table_artifact"


@pytest.mark.asyncio
async def test_announces_unknown_tool_when_request_is_bare(
    captured: list[dict[str, Any]],
) -> None:
    mw = ToolCallAnnouncerMiddlewareFactory().build(_ctx(), {})

    class _Bare:
        pass

    async def handler(_req: Any) -> str:
        return "ok"

    await mw.awrap_tool_call(_Bare(), handler)
    [step] = captured
    assert step["message"] == "Calling tool: unknown"


@pytest.mark.asyncio
async def test_announce_emits_error_step_on_failure(
    captured: list[dict[str, Any]],
) -> None:
    mw = ToolCallAnnouncerMiddlewareFactory().build(_ctx(), {})

    async def handler(_req: Any) -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await mw.awrap_tool_call(
            _Request(tool_name="failing_tool", args={"x": 1}), handler
        )
    kinds = [c.get("event_type") for c in captured]
    assert kinds == ["INFO", "ERROR"]
    err = captured[-1]
    assert err["message"] == "Tool failing_tool errored: boom"
    assert (
        "x" not in str(err["details"]) or err["details"]["tool_name"] == "failing_tool"
    )


@pytest.mark.asyncio
async def test_announce_runs_before_handler(
    captured: list[dict[str, Any]],
) -> None:
    mw = ToolCallAnnouncerMiddlewareFactory().build(_ctx(), {})
    seen_during: list[bool] = []

    async def handler(_req: Any) -> str:
        seen_during.append(any(c.get("type") == "step" for c in captured))
        return "ok"

    await mw.awrap_tool_call(_Request(tool_name="t", args={}), handler)
    assert seen_during == [True]


from openbb_agent_server.plugins.middleware.tool_call_announcer import (  # noqa: E402
    _args_as_detail,
    _from_tool_call,
    _stringify_arg,
)


def test_from_tool_call_none_returns_none() -> None:
    assert _from_tool_call(None, "name") is None


def test_from_tool_call_dict_returns_key() -> None:
    assert _from_tool_call({"name": "x"}, "name") == "x"


def test_from_tool_call_object_uses_getattr() -> None:
    class _Call:
        name = "y"

    assert _from_tool_call(_Call(), "name") == "y"
    assert _from_tool_call(_Call(), "missing") is None


def test_stringify_arg_none_returns_empty() -> None:
    assert _stringify_arg(None) == ""


def test_stringify_arg_string_passthrough() -> None:
    assert _stringify_arg("hello") == "hello"


def test_stringify_arg_int_passthrough() -> None:
    assert _stringify_arg(42) == "42"


def test_stringify_arg_bool_passthrough() -> None:
    assert _stringify_arg(True) == "True"


def test_stringify_arg_long_string_truncates() -> None:
    out = _stringify_arg("x" * 1000)
    assert len(out) == 401
    assert out.endswith("…")


def test_stringify_arg_json_fallback_for_non_serialisable() -> None:
    class _NotSerialisable:
        def __repr__(self) -> str:
            return "<NS>"

    out = _stringify_arg(_NotSerialisable())
    assert out == '"<NS>"'


def test_stringify_arg_uses_repr_on_json_failure() -> None:
    """Fall back to repr when json.dumps raises even with default=str."""

    import json as _json

    class _DefeatJson:
        pass

    obj = _DefeatJson()
    obj.__dict__["self_ref"] = obj

    real_dumps = _json.dumps

    def broken_dumps(*_a: object, **_kw: object) -> str:
        raise TypeError("circular")

    import openbb_agent_server.plugins.middleware.tool_call_announcer as mod

    mod.json.dumps = broken_dumps  # type: ignore[assignment]
    try:
        out = _stringify_arg({"k": "v"})
        assert "k" in out
    finally:
        mod.json.dumps = real_dumps  # type: ignore[assignment]


def test_args_as_detail_empty_dict() -> None:
    assert _args_as_detail({}) == {}


def test_args_as_detail_non_dict() -> None:
    assert _args_as_detail("not a dict") == {}
    assert _args_as_detail(None) == {}


def test_args_as_detail_stringifies_values() -> None:
    out = _args_as_detail({"a": 1, "b": "x", "c": [1, 2]})
    assert out == {"a": "1", "b": "x", "c": "[1, 2]"}


@pytest.mark.asyncio
async def test_graph_bubble_up_re_raises_without_error_step(
    captured: list[dict[str, Any]],
) -> None:
    """GraphBubbleUp is a control-flow signal and emits no ERROR step."""
    from langgraph.errors import GraphBubbleUp

    mw = ToolCallAnnouncerMiddlewareFactory().build(_ctx(), {})

    async def handler(_req: Any) -> str:
        raise GraphBubbleUp("interrupt")

    with pytest.raises(GraphBubbleUp):
        await mw.awrap_tool_call(_Request(tool_name="t", args={}), handler)
    kinds = [c.get("event_type") for c in captured]
    assert "ERROR" not in kinds


def test_tool_call_announcer_handles_object_shaped_tool_call() -> None:
    from openbb_agent_server.plugins.middleware.tool_call_announcer import (
        _from_tool_call,
        _tool_name,
    )

    class _ObjectToolCall:
        name = "transcribe_audio"

    request = type("R", (), {"tool_call": _ObjectToolCall()})()
    assert _from_tool_call(_ObjectToolCall(), "name") == "transcribe_audio"
    assert _tool_name(request) == "transcribe_audio"
