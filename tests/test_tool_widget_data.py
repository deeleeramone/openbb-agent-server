"""widget_data tool source tests."""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.plugins.tools.widget_data import WidgetDataToolSource
from openbb_agent_server.runtime import context as run_context
from openbb_agent_server.runtime.context import RunContext, WidgetRef
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx_with_widgets() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        widgets=(
            WidgetRef(
                uuid="w-AAPL",
                widget_id="balance",
                origin="Test Provider",
                params={"ticker": "AAPL"},
                data={"price": 150.0},
            ),
            WidgetRef(
                uuid="w-MSFT",
                widget_id="balance",
                origin="Test Provider",
                params={"ticker": "MSFT"},
                data={"price": 380.0},
            ),
        ),
    )


@pytest.mark.asyncio
async def test_yields_two_tools() -> None:
    src = WidgetDataToolSource()
    tools = await src.tools(_ctx_with_widgets(), {})
    names = [t.name for t in tools]
    assert names == ["list_widgets", "get_widget_data"]


@pytest.mark.asyncio
async def test_list_widgets_returns_run_widgets() -> None:
    src = WidgetDataToolSource()
    [list_tool, _] = await src.tools(_ctx_with_widgets(), {})
    with run_context.bind(_ctx_with_widgets()):
        result = list_tool.invoke({})
    assert isinstance(result, dict)
    assert result["count"] == 2
    assert {r["widget_id"] for r in result["widgets"]} == {"w-AAPL", "w-MSFT"}


@pytest.mark.asyncio
async def test_list_widgets_with_empty_run_returns_populated_dict() -> None:
    """Return a populated dict, never an empty list, for an empty run."""
    from openbb_agent_server.runtime.context import RunContext
    from openbb_agent_server.runtime.principal import UserPrincipal

    empty_ctx = RunContext(
        principal=UserPrincipal(user_id="probe", scopes=("agent:query",)),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        widgets=(),
    )
    src = WidgetDataToolSource()
    [list_tool, _] = await src.tools(empty_ctx, {})
    with run_context.bind(empty_ctx):
        result = list_tool.invoke({})
    assert isinstance(result, dict)
    assert result == {"count": 0, "widgets": []}


@pytest.mark.asyncio
async def test_get_widget_data_unknown_id_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise for an unknown widget_id before reaching interrupt()."""
    monkeypatch.setattr(
        "openbb_agent_server.runtime.emit._writer", lambda: lambda *_: None
    )
    src = WidgetDataToolSource()
    [_, get_tool] = await src.tools(_ctx_with_widgets(), {})
    with run_context.bind(_ctx_with_widgets()):
        with pytest.raises(Exception):
            get_tool.invoke({"widget_ids": ["missing"]})


@pytest.mark.asyncio
async def test_get_widget_data_emits_workspace_data_sources_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit the copilotFunctionCall SSE from get_widget_data."""
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "openbb_agent_server.runtime.emit._writer",
        lambda: emitted.append,
    )

    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        widgets=(
            WidgetRef(
                uuid="0879a2d2-ffdc-47a0-8e14-48e0381289ed",
                widget_id="balance",
                origin="Financial Datasets Market Intelligence",
                params={"ticker": "NVDA", "period": "annual", "limit": "10"},
                data=None,
            ),
        ),
    )
    src = WidgetDataToolSource()
    [_, get_tool] = await src.tools(ctx, {})
    with run_context.bind(ctx):
        result = get_tool.invoke(
            {"widget_ids": ["0879a2d2-ffdc-47a0-8e14-48e0381289ed"]}
        )

    fn_calls = [e for e in emitted if e.get("type") == "function_call"]
    assert len(fn_calls) == 1
    assert fn_calls[0]["tool_name"] == "get_widget_data"
    assert fn_calls[0]["parameters"] == {
        "data_sources": [
            {
                "widget_uuid": "0879a2d2-ffdc-47a0-8e14-48e0381289ed",
                "origin": "Financial Datasets Market Intelligence",
                "id": "balance",
                "input_args": {
                    "ticker": "NVDA",
                    "period": "annual",
                    "limit": "10",
                },
            }
        ]
    }
    assert isinstance(result, str)
    assert "Dispatched" in result


def test_system_prompt_includes_widget_snapshot_when_widgets_attached() -> None:
    from openbb_agent_server.runtime.builder import _build_system_prompt

    prompt = _build_system_prompt(_ctx_with_widgets())
    assert "Attached widgets" in prompt
    assert "widget_uuid='w-AAPL'" in prompt
    assert "widget_id='balance'" in prompt
    assert "params_hash=" in prompt
    assert "data_hash=" in prompt


def test_system_prompt_omits_widget_section_when_no_widgets() -> None:
    from openbb_agent_server.runtime.builder import _build_system_prompt

    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )
    prompt = _build_system_prompt(ctx)
    assert "Attached widgets" not in prompt


def test_system_prompt_includes_file_snapshot_when_files_uploaded() -> None:
    from openbb_agent_server.runtime.builder import _build_system_prompt
    from openbb_agent_server.runtime.context import FileRef

    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        uploaded_files=(FileRef(name="10K.pdf", mime="application/pdf"),),
    )
    prompt = _build_system_prompt(ctx)
    assert "Uploaded files" in prompt
    assert "10K.pdf" in prompt


@pytest.mark.asyncio
async def test_get_widget_data_batches_multiple_ids_into_one_function_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit a single FunctionCallSSE carrying both data sources for one call."""
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "openbb_agent_server.runtime.emit._writer",
        lambda: emitted.append,
    )

    src = WidgetDataToolSource()
    ctx = _ctx_with_widgets()
    [_, get_tool] = await src.tools(ctx, {})
    with run_context.bind(ctx):
        get_tool.invoke({"widget_ids": ["w-AAPL", "w-MSFT"]})

    fn_calls = [e for e in emitted if e.get("type") == "function_call"]
    assert len(fn_calls) == 1
    sources = fn_calls[0]["parameters"]["data_sources"]
    assert [s["widget_uuid"] for s in sources] == ["w-AAPL", "w-MSFT"]


@pytest.mark.asyncio
async def test_get_widget_data_dedupes_repeated_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collapse repeated ids to a single data_source entry."""
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "openbb_agent_server.runtime.emit._writer",
        lambda: emitted.append,
    )

    src = WidgetDataToolSource()
    ctx = _ctx_with_widgets()
    [_, get_tool] = await src.tools(ctx, {})
    with run_context.bind(ctx):
        get_tool.invoke({"widget_ids": ["w-AAPL", "w-AAPL"]})

    fn_calls = [e for e in emitted if e.get("type") == "function_call"]
    assert len(fn_calls[0]["parameters"]["data_sources"]) == 1


def test_summarise_includes_name_and_description_when_present() -> None:
    """Add name and description when the widget has them."""
    from openbb_agent_server.plugins.tools.widget_data import _summarise

    w = WidgetRef(
        uuid="u-1",
        widget_id="balance",
        origin="market",
        params={"sym": "AAPL"},
        data=None,
        name="Balance Sheet",
        description="quarterly balance sheet",
    )
    out = _summarise(w)
    assert out["name"] == "Balance Sheet"
    assert out["description"] == "quarterly balance sheet"


def test_coerce_widget_ids_parses_json_string_list() -> None:
    """Decode a JSON-stringified list back into a real list."""
    from openbb_agent_server.plugins.tools.widget_data import _GetWidgetArgs

    args = _GetWidgetArgs(widget_ids='["a", "b"]')
    assert args.widget_ids == ["a", "b"]


def test_coerce_widget_ids_falls_back_to_comma_split_on_bad_json() -> None:
    """Fall back to comma splitting for a bracketed-but-invalid JSON string."""
    from openbb_agent_server.plugins.tools.widget_data import _GetWidgetArgs

    args = _GetWidgetArgs(widget_ids="[a, b, c]")
    assert args.widget_ids == ["[a", "b", "c]"]


def test_coerce_widget_ids_comma_splits_plain_string() -> None:
    """Split a plain comma-separated string into trimmed ids."""
    from openbb_agent_server.plugins.tools.widget_data import _GetWidgetArgs

    args = _GetWidgetArgs(widget_ids=" w-1 , w-2 ")
    assert args.widget_ids == ["w-1", "w-2"]


def test_coerce_widget_ids_passes_through_non_string_non_sequence() -> None:
    """Fall through to pydantic validation for a non-str/list/tuple value."""
    from openbb_agent_server.plugins.tools.widget_data import _GetWidgetArgs

    with pytest.raises(Exception):
        _GetWidgetArgs(widget_ids=123)


def test_stable_json_falls_back_to_str_on_circular_reference() -> None:
    """Return str(value) when JSON encoding raises."""
    from openbb_agent_server.plugins.tools.widget_data import _stable_json

    circular: dict[str, Any] = {}
    circular["self"] = circular
    out = _stable_json(circular)
    assert isinstance(out, str)


def test_widget_data_stable_json_falls_back_to_str_for_unserialisable() -> None:
    from openbb_agent_server.plugins.tools.widget_data import _stable_json

    class _Weird:
        def __repr__(self) -> str:
            return "<weird>"

        def __str__(self) -> str:
            return "<weird-str>"

    a: dict[str, Any] = {}
    a["self"] = a
    out = _stable_json(a)
    assert isinstance(out, str)
