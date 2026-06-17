"""dashboard (client-side) tool source tests."""

from __future__ import annotations

import pytest

from openbb_agent_server.plugins.tools.dashboard import DashboardToolSource
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
async def test_default_dashboard_tools_present() -> None:
    src = DashboardToolSource()
    tools = await src.tools(_ctx(), {})
    names = {t.name for t in tools}
    expected = {
        f"{CLIENT_SIDE_TOOL_PREFIX}open_widget",
        f"{CLIENT_SIDE_TOOL_PREFIX}highlight_widget",
        f"{CLIENT_SIDE_TOOL_PREFIX}change_dashboard",
        f"{CLIENT_SIDE_TOOL_PREFIX}add_widget_to_dashboard",
    }
    assert expected.issubset(names)


@pytest.mark.asyncio
async def test_open_widget_args_schema_has_widget_id() -> None:
    src = DashboardToolSource()
    tools = await src.tools(_ctx(), {})
    open_tool = next(
        t for t in tools if t.name == f"{CLIENT_SIDE_TOOL_PREFIX}open_widget"
    )
    assert "widget_id" in open_tool.args_schema.model_fields


@pytest.mark.asyncio
async def test_custom_tool_list_overrides_default() -> None:
    src = DashboardToolSource(tools=[{"name": "ping"}])
    tools = await src.tools(_ctx(), {})
    assert [t.name for t in tools] == [f"{CLIENT_SIDE_TOOL_PREFIX}ping"]
