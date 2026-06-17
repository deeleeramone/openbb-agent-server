"""``workspace_mcp`` tool source."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from openbb_agent_server.plugins.tools.client_side import _args_model_from_schema
from openbb_agent_server.protocol.adapter import WORKSPACE_MCP_TOOL_PREFIX
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource


def _make_tool(spec: dict[str, Any]) -> StructuredTool:
    raw_name = str(spec.get("name") or "")
    if not raw_name:
        raise ValueError(f"workspace_mcp tool entry missing 'name': {spec}")
    server_id = str(spec.get("server_id") or "workspace")
    description = (
        spec.get("description")
        or f"Workspace MCP tool: {raw_name} (server={server_id})."
    )
    schema = spec.get("input_schema") or {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    args_model = _args_model_from_schema(raw_name, schema)

    def _client_call(**kwargs: Any) -> Any:
        """Pause the run; Workspace runs the MCP tool and POSTs the result."""
        from langgraph.types import interrupt

        return interrupt(
            {
                "type": "workspace_mcp_tool_call",
                "server_id": server_id,
                "name": raw_name,
                "arguments": dict(kwargs),
            }
        )

    return StructuredTool.from_function(
        _client_call,
        name=f"{WORKSPACE_MCP_TOOL_PREFIX}{server_id}:{raw_name}",
        description=description,
        args_schema=args_model,
    )


class WorkspaceMcpToolSource(ToolSource):
    """Surface ``ctx.tools`` (Workspace MCP) as client-side tools."""

    name = "workspace_mcp"

    async def tools(
        self, ctx: RunContext, config: dict[str, Any]
    ) -> list[StructuredTool]:
        """Build client-side tools from the Workspace MCP specs in ``ctx``.

        Convert each tool spec in ``ctx.tools`` into a ``StructuredTool`` whose
        invocation interrupts the run so the Workspace can execute the MCP tool
        and POST back the result. Specs that fail to convert (missing name or
        malformed schema) are skipped silently.

        Parameters
        ----------
        ctx : RunContext
            Active run context whose ``tools`` lists the Workspace MCP tool
            specs to surface.
        config : dict[str, Any]
            Plugin configuration for this tool source; currently unused.

        Returns
        -------
        list[StructuredTool]
            One structured tool per successfully converted spec.
        """
        tools: list[StructuredTool] = []
        for spec in ctx.tools:
            try:
                tools.append(_make_tool(spec))
            except (ValueError, KeyError, TypeError):
                continue
        return tools
