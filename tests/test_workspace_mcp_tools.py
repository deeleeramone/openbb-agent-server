"""Workspace-MCP tool source + adapter routing + agents.json feature."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.plugins.tools.workspace_mcp import (
    WorkspaceMcpToolSource,
    _make_tool,
)
from openbb_agent_server.protocol.adapter import (
    WORKSPACE_MCP_TOOL_PREFIX,
    DeepAgentEventAdapter,
)
from openbb_agent_server.protocol.schemas import FunctionCallSSE
from openbb_agent_server.runtime import services
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(*, tools: tuple[dict[str, Any], ...] = ()) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        tools=tools,
    )


async def test_returns_one_stub_tool_per_agent_tool_in_ctx() -> None:
    src = WorkspaceMcpToolSource()
    tools = await src.tools(
        _ctx(
            tools=(
                {
                    "name": "list_databases",
                    "server_id": "snowflake-mcp",
                    "description": "List Snowflake databases",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "get_widget_data",
                    "server_id": "openbb-mcp",
                    "description": "Fetch widget data",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "widget_id": {
                                "type": "string",
                                "description": "Widget UUID",
                            },
                        },
                        "required": ["widget_id"],
                    },
                },
            )
        ),
        {},
    )
    names = sorted(t.name for t in tools)
    assert names == [
        "mcp:openbb-mcp:get_widget_data",
        "mcp:snowflake-mcp:list_databases",
    ]


async def test_handles_empty_tools_list() -> None:
    src = WorkspaceMcpToolSource()
    assert await src.tools(_ctx(), {}) == []


async def test_skips_entries_without_a_name() -> None:
    src = WorkspaceMcpToolSource()
    tools = await src.tools(
        _ctx(
            tools=(
                {"server_id": "x", "description": "no name → skipped"},
                {"name": "ok", "server_id": "y"},
            )
        ),
        {},
    )
    assert [t.name for t in tools] == ["mcp:y:ok"]


async def test_skips_entries_that_fail_to_make_a_tool() -> None:
    """Keep the tool list intact despite per-entry exceptions."""
    src = WorkspaceMcpToolSource()
    tools = await src.tools(
        _ctx(
            tools=(
                {"name": "", "server_id": "x"},
                {"name": "ok", "server_id": "y"},
            )
        ),
        {},
    )
    assert [t.name for t in tools] == ["mcp:y:ok"]


def test_make_tool_defaults_server_id_when_missing() -> None:
    tool = _make_tool({"name": "ping"})
    assert tool.name == "mcp:workspace:ping"


def test_make_tool_handles_non_dict_input_schema() -> None:
    tool = _make_tool({"name": "ping", "server_id": "s", "input_schema": "not-a-dict"})
    assert tool.name == "mcp:s:ping"


def test_make_tool_body_invokes_interrupt_when_called() -> None:
    """Raise when the tool body's interrupt() runs outside LangGraph."""
    tool = _make_tool({"name": "ping", "server_id": "s"})
    with pytest.raises(Exception):
        tool.invoke({})


def test_adapter_routes_mcp_prefix_into_function_call_sse() -> None:
    """Route mcp:<server>:<fn> into a FunctionCallSSE."""
    adapter = DeepAgentEventAdapter(client_tool_names=frozenset())
    out = adapter._translate_messages(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "name": f"{WORKSPACE_MCP_TOOL_PREFIX}snowflake-mcp:list_databases",
                        "args": {"limit": 5},
                        "id": "call-1",
                    }
                ],
            }
        },
        ns=("agent",),
    )
    assert len(out) == 1
    fc = out[0]
    assert isinstance(fc, FunctionCallSSE)
    assert fc.data.function == "execute_agent_tool"
    assert fc.data.input_arguments == {
        "server_id": "snowflake-mcp",
        "name": "list_databases",
        "arguments": {"limit": 5},
    }
    assert fc.data.extra_state == {"call_id": "call-1"}


def test_adapter_falls_back_to_namespace_when_server_id_missing() -> None:
    """Fall back to namespace[0] for mcp:tool_name with no server prefix."""
    adapter = DeepAgentEventAdapter(client_tool_names=frozenset())
    out = adapter._translate_messages(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "name": f"{WORKSPACE_MCP_TOOL_PREFIX}orphan_tool",
                        "args": {},
                        "id": "x",
                    }
                ],
            }
        },
        ns=("custom-ns",),
    )
    fc = out[0]
    assert fc.data.input_arguments["server_id"] == "custom-ns"
    assert fc.data.input_arguments["name"] == "orphan_tool"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    with TestClient(create_app(AgentServerSettings())) as client:
        yield client


def test_agents_json_advertises_mcp_tools_true_by_default(client: TestClient) -> None:
    body = client.get("/agents.json").json()
    assert "default" in body
    assert body["default"]["features"]["mcp-tools"] is True
