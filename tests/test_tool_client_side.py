"""client_side tool source tests."""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from openbb_agent_server.plugins.tools.client_side import (
    ClientSideToolSource,
)
from openbb_agent_server.protocol.adapter import CLIENT_SIDE_TOOL_PREFIX
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


@pytest.mark.asyncio
async def test_emits_named_tools_with_client_prefix() -> None:
    src = ClientSideToolSource(
        tools=[
            {
                "name": "open_widget",
                "description": "Open a widget on the dashboard.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "widget_id": {"type": "string", "description": "Widget ID."},
                    },
                    "required": ["widget_id"],
                },
            }
        ]
    )
    tools = await src.tools(_ctx(), {})
    assert len(tools) == 1
    t = tools[0]
    assert isinstance(t, StructuredTool)
    assert t.name == f"{CLIENT_SIDE_TOOL_PREFIX}open_widget"


@pytest.mark.asyncio
async def test_args_schema_carries_required_fields() -> None:
    src = ClientSideToolSource(
        tools=[
            {
                "name": "select",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["ticker"],
                },
            }
        ]
    )
    [tool] = await src.tools(_ctx(), {})
    fields = tool.args_schema.model_fields
    assert "ticker" in fields
    assert "limit" in fields


@pytest.mark.asyncio
async def test_no_parameters_creates_empty_args_schema() -> None:
    src = ClientSideToolSource(tools=[{"name": "ping"}])
    [tool] = await src.tools(_ctx(), {})
    assert tool.name == f"{CLIENT_SIDE_TOOL_PREFIX}ping"
    assert tool.args_schema.model_fields == {}


@pytest.mark.asyncio
async def test_config_overrides_constructor_specs() -> None:
    src = ClientSideToolSource(tools=[{"name": "default"}])
    runtime_specs = [{"name": "override"}]
    tools = await src.tools(_ctx(), {"tools": runtime_specs})
    assert tools[0].name == f"{CLIENT_SIDE_TOOL_PREFIX}override"


def test_json_schema_to_python_unknown_type_returns_any() -> None:
    """Map a JSON-schema fragment with no type string to Any."""
    from typing import Any as _Any

    from openbb_agent_server.plugins.tools.client_side import (
        _json_schema_to_python,
    )

    assert _json_schema_to_python({}) is _Any
    assert _json_schema_to_python({"type": 42}) is _Any
    assert _json_schema_to_python({"type": "unmapped-foo"}) is _Any


def test_client_side_tool_invocation_raises_outside_langgraph_context() -> None:
    """Raise when the tool body's interrupt() runs outside an agent."""
    from openbb_agent_server.plugins.tools.client_side import _make_tool

    tool = _make_tool({"name": "ping"})
    with pytest.raises(Exception):
        tool.invoke({})
