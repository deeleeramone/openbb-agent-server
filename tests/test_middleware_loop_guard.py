"""Unit tests for the ``loop_guard`` middleware."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from openbb_agent_server.plugins.middleware.loop_guard import (
    LoopGuardMiddlewareFactory,
    _args_hash,
    _LoopGuardMiddleware,
    _tool_args,
    _tool_call_field,
    _tool_call_id,
    _tool_name,
)
from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.principal import UserPrincipal


class _Req:
    """Fake request with a tool_call payload and flat attributes."""

    def __init__(
        self,
        *,
        tool_call: Any = None,
        tool_name: Any = None,
        args: Any = None,
    ) -> None:
        self.tool_call = tool_call
        if tool_name is not None:
            self.tool_name = tool_name
        if args is not None:
            self.args = args


class _ToolCallObj:
    """Object-shaped tool call for the getattr branch."""

    def __init__(self, *, name: str = "", args: Any = None, id: str = "") -> None:
        self.name = name
        self.args = args
        self.id = id


def test_args_hash_serialisable() -> None:
    assert _args_hash({"b": 2, "a": 1}) == _args_hash({"a": 1, "b": 2})
    assert len(_args_hash({"a": 1})) == 12


def test_args_hash_unserialisable_falls_back_to_repr() -> None:
    """Fall through to repr when json.dumps cannot handle a value."""

    class _NoJson:
        def __repr__(self) -> str:  # noqa: D105
            return "no-json"

    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    out = _args_hash(cyclic)
    assert len(out) == 12


def test_tool_call_field_none_when_no_tool_call() -> None:
    assert _tool_call_field(_Req(tool_call=None), "name") is None


def test_tool_call_field_reads_dict() -> None:
    assert _tool_call_field(_Req(tool_call={"name": "x"}), "name") == "x"


def test_tool_call_field_reads_object_attr() -> None:
    req = _Req(tool_call=_ToolCallObj(name="obj-tool"))
    assert _tool_call_field(req, "name") == "obj-tool"


def test_tool_name_prefers_request_attr() -> None:
    assert _tool_name(_Req(tool_name="flat")) == "flat"


def test_tool_name_falls_back_to_tool_call() -> None:
    assert _tool_name(_Req(tool_call={"name": "from-call"})) == "from-call"


def test_tool_name_defaults_to_unknown() -> None:
    assert _tool_name(_Req()) == "unknown"


def test_tool_args_from_tool_call() -> None:
    assert _tool_args(_Req(tool_call={"args": {"k": "v"}})) == {"k": "v"}


def test_tool_args_from_request_attr() -> None:
    assert _tool_args(_Req(args={"a": 1})) == {"a": 1}


def test_tool_args_defaults_to_empty_dict() -> None:
    assert _tool_args(_Req()) == {}


def test_tool_call_id_reads_dict() -> None:
    assert _tool_call_id(_Req(tool_call={"id": "cid"})) == "cid"


def test_tool_call_id_defaults_to_empty_string() -> None:
    assert _tool_call_id(_Req()) == ""


@pytest.mark.asyncio
async def test_awrap_runs_handler_below_threshold() -> None:
    """Distinct or first calls pass straight through to the handler."""
    mw = _LoopGuardMiddleware(max_repeats=2)
    calls: list[Any] = []

    async def handler(request: Any) -> str:
        calls.append(request)
        return "real-result"

    req = _Req(tool_call={"name": "fetch", "args": {"x": 1}, "id": "c1"})
    out = await mw.awrap_tool_call(req, handler)
    assert out == "real-result"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_awrap_short_circuits_after_max_repeats() -> None:
    """The third identical call returns a synthetic ToolMessage."""
    mw = _LoopGuardMiddleware(max_repeats=2)
    handler_calls = 0

    async def handler(request: Any) -> str:
        nonlocal handler_calls
        handler_calls += 1
        return "real"

    req = _Req(tool_call={"name": "fetch", "args": {"x": 1}, "id": "c1"})
    sink: list[dict[str, Any]] = []
    with emit.bind_writer(sink.append):
        assert await mw.awrap_tool_call(req, handler) == "real"
        assert await mw.awrap_tool_call(req, handler) == "real"
        tripped = await mw.awrap_tool_call(req, handler)
        await mw.awrap_tool_call(req, handler)

    assert handler_calls == 2
    assert isinstance(tripped, ToolMessage)
    assert "loop_guard" in tripped.content
    assert tripped.tool_call_id == "c1"
    assert tripped.name == "fetch"
    warnings = [s for s in sink if s.get("event_type") == "WARNING"]
    assert len(warnings) == 1
    assert warnings[0]["details"]["tool_name"] == "fetch"
    assert warnings[0]["details"]["repeat_count"] == 3


@pytest.mark.asyncio
async def test_awrap_resets_counter_on_different_key() -> None:
    """A different tool or args resets the repeat counter."""
    mw = _LoopGuardMiddleware(max_repeats=1)

    async def handler(request: Any) -> str:
        return "ok"

    req_a = _Req(tool_call={"name": "a", "args": {}, "id": "1"})
    req_b = _Req(tool_call={"name": "b", "args": {}, "id": "2"})
    assert await mw.awrap_tool_call(req_a, handler) == "ok"
    assert await mw.awrap_tool_call(req_b, handler) == "ok"
    assert await mw.awrap_tool_call(req_b, handler) != "ok"


@pytest.mark.asyncio
async def test_awrap_reraises_graph_bubble_up() -> None:
    """GraphBubbleUp from the handler propagates instead of looping."""
    from langgraph.errors import GraphBubbleUp

    mw = _LoopGuardMiddleware(max_repeats=2)

    async def handler(request: Any) -> str:
        raise GraphBubbleUp()

    req = _Req(tool_call={"name": "x", "args": {}, "id": "c"})
    with pytest.raises(GraphBubbleUp):
        await mw.awrap_tool_call(req, handler)


def test_factory_defaults_and_build() -> None:
    factory = LoopGuardMiddlewareFactory()
    assert factory.name == "loop_guard"
    ctx = _ctx()
    mw = factory.build(ctx, {})
    assert isinstance(mw, _LoopGuardMiddleware)
    assert mw._max_repeats == 2


def test_factory_build_honours_config_override() -> None:
    factory = LoopGuardMiddlewareFactory(max_repeats=5)
    mw = factory.build(_ctx(), {"max_repeats": 9})
    assert mw._max_repeats == 9


def _ctx() -> Any:
    from openbb_agent_server.runtime.context import RunContext

    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )
