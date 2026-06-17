"""Usage recorder middleware."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from openbb_agent_server.persistence.store import UsageRecord
from openbb_agent_server.runtime import (
    context as run_context,
    services,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import Middleware

logger = logging.getLogger("openbb_agent_server.middleware.usage_recorder")


class _UsageRecorderMiddleware(AgentMiddleware):
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        await self._record(state)
        return None

    async def _record(self, state: Any) -> None:
        messages = (state or {}).get("messages") or []
        if not messages:
            return
        last = messages[-1]
        usage = getattr(last, "usage_metadata", None)
        if not usage:
            return
        try:
            ctx: RunContext = run_context.current()
        except LookupError:
            logger.debug("usage_recorder: no RunContext bound, skipping")
            return
        details = usage.get("input_token_details", {}) or {}
        rmd = getattr(last, "response_metadata", {}) or {}
        addl = getattr(last, "additional_kwargs", {}) or {}
        model_name = (
            rmd.get("model")
            or rmd.get("model_name")
            or rmd.get("ls_model_name")
            or addl.get("model")
            or addl.get("model_name")
            or "unknown"
        )
        record = UsageRecord(
            trace_id=ctx.trace_id,
            user_id=ctx.principal.user_id,
            model=str(model_name),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cache_read=int(details.get("cache_read", 0)),
            cache_creation=int(details.get("cache_creation", 0)),
            cost_usd=0.0,
        )
        await services.get_history().record_usage(
            principal=ctx.principal,
            trace_id=ctx.trace_id,
            usage=record,
        )


class UsageRecorderMiddlewareFactory(Middleware):
    """Construct a usage recorder middleware.

    The built middleware runs after each model call and, when the latest
    message carries token usage metadata, persists a
    :class:`~openbb_agent_server.persistence.store.UsageRecord` (trace id, user id, model name, input/output
    tokens, and cache read/creation counts) via the history service.
    """

    name = "usage_recorder"

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        """Build a usage recorder middleware instance for a run.

        Parameters
        ----------
        ctx : RunContext
            The run context (unused; present for interface
            compatibility). The active context is read dynamically when a
            usage record is written.
        config : dict
            Per-run configuration (unused).

        Returns
        -------
        AgentMiddleware
            A usage recorder middleware instance.
        """
        return _UsageRecorderMiddleware()
