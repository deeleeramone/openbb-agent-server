"""Tool call ledger middleware."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from openbb_agent_server.runtime import (
    context as run_context,
    services,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import Middleware

logger = logging.getLogger("openbb_agent_server.middleware.tool_call_ledger")


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return {"__str__": str(value)}


class _ToolCallLedgerMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Record each tool call to the history service, then return it.

        Wrap the downstream ``handler`` so that every tool invocation is
        timed and persisted via the history service. When no run context
        is active, the call is forwarded without recording. Successful
        calls are logged with ``state="complete"`` and the JSON-safe
        result; failures are logged with ``state="error"`` and the
        exception text before the exception is re-raised. ``GraphBubbleUp``
        control-flow signals are re-raised without being recorded.

        Parameters
        ----------
        request : Any
            The tool-call request. Its ``tool_name``/``args`` attributes
            (or the nested ``tool_call`` mapping/object) supply the name
            and arguments recorded to the ledger.
        handler : Any
            Async callable that executes the tool call and returns its
            response.

        Returns
        -------
        Any
            The response produced by ``handler(request)``.

        Raises
        ------
        Exception
            Re-raises any exception from ``handler`` after recording the
            failed tool call. ``GraphBubbleUp`` is re-raised unrecorded.
        """
        from langgraph.errors import GraphBubbleUp

        try:
            ctx: RunContext = run_context.current()
        except LookupError:
            return await handler(request)

        tool_call = getattr(request, "tool_call", None)
        tc_name = (
            tool_call.get("name")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "name", None)
        )
        tc_args = (
            tool_call.get("args")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "args", None)
        )
        tool_name = getattr(request, "tool_name", None) or tc_name or "unknown"
        args = getattr(request, "args", None) or tc_args or {}

        start = time.perf_counter()
        result_payload: dict[str, Any] | None = None
        error: str | None = None
        try:
            response = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            error = str(exc)
            await services.get_history().record_tool_call(
                principal=ctx.principal,
                trace_id=ctx.trace_id,
                tool_name=str(tool_name),
                args=_safe_json(args) or {},
                result=None,
                error=error,
                latency_ms=int((time.perf_counter() - start) * 1000),
                side="server",
                state="error",
            )
            raise
        result_payload = {"result": _safe_json(getattr(response, "content", response))}
        await services.get_history().record_tool_call(
            principal=ctx.principal,
            trace_id=ctx.trace_id,
            tool_name=str(tool_name),
            args=_safe_json(args) or {},
            result=result_payload,
            error=None,
            latency_ms=int((time.perf_counter() - start) * 1000),
            side="server",
            state="complete",
        )
        return response


class ToolCallLedgerMiddlewareFactory(Middleware):
    """Build the middleware that logs tool calls to the history service.

    Registered under the ``tool_call_ledger`` plugin name.
    """

    name = "tool_call_ledger"

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        """Return a new tool-call ledger middleware instance.

        Parameters
        ----------
        ctx : RunContext
            Per-run context (unused at build time; the middleware reads
            the active context at call time). Required by the
            ``Middleware`` interface.
        config : dict[str, Any]
            Per-build configuration. Unused; the middleware takes no
            options.

        Returns
        -------
        AgentMiddleware
            A middleware that records every tool call to the history
            service.
        """
        return _ToolCallLedgerMiddleware()
