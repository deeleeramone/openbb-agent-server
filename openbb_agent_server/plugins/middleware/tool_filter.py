"""Tool filter middleware."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import Middleware

logger = logging.getLogger("openbb_agent_server.middleware.tool_filter")


_DEFAULT_EXCLUDED = frozenset(
    {
        "ls",
        "execute",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
    }
)


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        n = tool.get("name")
        return n if isinstance(n, str) else None
    n = getattr(tool, "name", None)
    return n if isinstance(n, str) else None


class _ToolFilterMiddleware(AgentMiddleware):
    """Drop named tools from every request tools payload."""

    def __init__(self, excluded: frozenset[str]) -> None:
        self._excluded = excluded
        self._logged_once = False

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        if self._excluded:
            request = self._filter(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        if self._excluded:
            request = self._filter(request)
        return await handler(request)

    def _filter(self, request: Any) -> Any:
        original = list(request.tools)
        filtered = [t for t in original if _tool_name(t) not in self._excluded]
        if len(filtered) == len(original):
            return request
        if not self._logged_once:
            removed = [
                _tool_name(t) for t in original if _tool_name(t) in self._excluded
            ]
            logger.debug("tool_filter: removed %s", removed)
            self._logged_once = True
        return request.override(tools=filtered)


class ToolFilterMiddlewareFactory(Middleware):
    """Build a tool-filter middleware from per-run config."""

    name = "tool_filter"

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        """Build a tool-filter middleware from per-run config.

        Reads the ``excluded`` key from ``config`` to decide which tool
        names are stripped from every model request. When absent, a default
        set of file-system and shell tools is excluded. A non-string-list
        value is rejected with a warning and the default set is used; from
        a provided iterable only string entries are kept.

        Parameters
        ----------
        ctx : RunContext
            Active run context. Unused, present for protocol conformance.
        config : dict[str, Any]
            Per-run config; the ``excluded`` key, if present, lists tool
            names to drop.

        Returns
        -------
        AgentMiddleware
            A middleware that removes the resolved tool names from each
            request's tools payload.
        """
        raw = config.get("excluded")
        if raw is None:
            excluded = _DEFAULT_EXCLUDED
        elif isinstance(raw, (list, tuple, set, frozenset)):
            excluded = frozenset(str(x) for x in raw if isinstance(x, str))
        else:
            logger.warning(
                "tool_filter.excluded must be a list of strings; "
                "ignoring %r and falling back to default",
                raw,
            )
            excluded = _DEFAULT_EXCLUDED
        return _ToolFilterMiddleware(excluded=excluded)
