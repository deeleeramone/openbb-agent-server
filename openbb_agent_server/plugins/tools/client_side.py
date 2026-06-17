"""``client_side`` tool source — declares tools the *Workspace UI* executes."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from openbb_agent_server.protocol.adapter import CLIENT_SIDE_TOOL_PREFIX
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource


def _args_model_from_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a pydantic model from a JSON-Schema-ish dict."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    fields: dict[str, tuple[Any, Any]] = {}
    for prop, prop_schema in properties.items():
        py_type = _json_schema_to_python(prop_schema)
        default = ... if prop in required else None
        fields[prop] = (
            py_type if prop in required else py_type | None,
            Field(default, description=prop_schema.get("description", "")),
        )
    if not fields:
        return create_model(f"{name}Args", __base__=BaseModel)
    return create_model(f"{name}Args", **fields)  # ty: ignore[no-matching-overload]


_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_schema_to_python(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    if isinstance(t, str):
        return _JSON_TO_PY.get(t, Any)
    return Any


def _make_tool(spec: dict[str, Any]) -> StructuredTool:
    raw_name = spec["name"]
    description = spec.get("description") or f"Client-side tool: {raw_name}"
    parameters = spec.get("parameters") or {"type": "object", "properties": {}}
    args_model = _args_model_from_schema(raw_name, parameters)

    def _client_side(**kwargs: Any) -> Any:
        """Pause the run and hand control back to Workspace."""
        from langgraph.types import interrupt

        return interrupt(
            {
                "type": "client_side_tool_call",
                "name": raw_name,
                "arguments": dict(kwargs),
            }
        )

    return StructuredTool.from_function(
        _client_side,
        name=f"{CLIENT_SIDE_TOOL_PREFIX}{raw_name}",
        description=description,
        args_schema=args_model,
    )


class ClientSideToolSource(ToolSource):
    """Declare tools the Workspace UI must execute."""

    name = "client_side"

    def __init__(self, *, tools: list[dict[str, Any]] | None = None) -> None:
        """Store the default client-side tool specifications.

        Parameters
        ----------
        tools : list[dict[str, Any]] or None, optional
            Default tool specs, each a dict with ``name`` and optional
            ``description`` / ``parameters`` (JSON-Schema) keys. Used when
            no per-run ``tools`` override is supplied to :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools`.
            Defaults to an empty tuple.
        """
        self._specs = tuple(tools or ())

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Build the client-side tool stubs for a run.

        Each returned tool, when invoked by the agent, raises a LangGraph
        ``interrupt`` that pauses the run and hands the call back to the
        Workspace UI for execution rather than running server-side.

        Parameters
        ----------
        ctx : RunContext
            Active run context. Currently unused.
        config : dict[str, Any]
            Per-invocation config. A ``"tools"`` key overrides the specs
            passed to the constructor; otherwise those defaults are used.

        Returns
        -------
        list[Any]
            One ``StructuredTool`` per spec, each named with the
            client-side tool prefix.
        """
        specs = config.get("tools", self._specs)
        return [_make_tool(s) for s in specs]
