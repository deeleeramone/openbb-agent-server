"""Tool call announcer middleware."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import Middleware

logger = logging.getLogger("openbb_agent_server.middleware.tool_call_announcer")

_MAX_ARG_VALUE_LEN = 400


def _stringify_arg(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    elif isinstance(value, (int, float, bool)):
        s = str(value)
    else:
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            s = str(value)
    if len(s) > _MAX_ARG_VALUE_LEN:
        s = s[:_MAX_ARG_VALUE_LEN] + "…"
    return s


def _args_as_detail(args: Any) -> dict[str, str]:
    """Flatten a tool-call args dict to a string-only mapping."""
    if not isinstance(args, dict) or not args:
        return {}
    return {str(k): _stringify_arg(v) for k, v in args.items()}


def _from_tool_call(tool_call: Any, key: str) -> Any:
    """Read a field from a tool call dict or object."""
    if tool_call is None:
        return None
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _tool_name(request: Any) -> str:
    tool_call = getattr(request, "tool_call", None)
    return str(
        getattr(request, "tool_name", None)
        or _from_tool_call(tool_call, "name")
        or "unknown"
    )


class _ToolCallAnnouncerMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        from langgraph.errors import GraphBubbleUp

        name = _tool_name(request)
        tool_call = getattr(request, "tool_call", None)
        args = _from_tool_call(tool_call, "args") or {}

        details: dict[str, Any] = {"tool_name": name}
        for k, v in _args_as_detail(args).items():
            details[k] = v
        emit.reasoning_step(
            f"Calling tool: {name}",
            event_type="INFO",
            **details,
        )
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            emit.reasoning_step(
                f"Tool {name} errored: {exc}",
                event_type="ERROR",
                tool_name=name,
            )
            raise


class ToolCallAnnouncerMiddlewareFactory(Middleware):
    """Build the per-run tool-call announcer.

    A :class:`~openbb_agent_server.runtime.plugins.Middleware` factory whose :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build` returns middleware
    that emits a reasoning step before each tool call (with the tool
    name and stringified arguments) and an error step if the call
    raises.
    """

    name = "tool_call_announcer"

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        """Construct the announcer middleware for a run.

        The returned middleware is stateless and ignores ``ctx`` and
        ``config``; they are accepted to satisfy the factory contract.

        Parameters
        ----------
        ctx : RunContext
            The active run context (unused).
        config : dict[str, Any]
            Per-run middleware configuration (unused).

        Returns
        -------
        AgentMiddleware
            A middleware that announces each tool call and its outcome.
        """
        return _ToolCallAnnouncerMiddleware()
