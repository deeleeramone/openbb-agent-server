"""Loop guard middleware."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import Middleware

logger = logging.getLogger("openbb_agent_server.middleware.loop_guard")


def _args_hash(args: Any) -> str:
    try:
        s = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        s = repr(args)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _tool_call_field(request: Any, key: str) -> Any:
    tc = getattr(request, "tool_call", None)
    if tc is None:
        return None
    if isinstance(tc, dict):
        return tc.get(key)
    return getattr(tc, key, None)


def _tool_name(request: Any) -> str:
    return str(
        getattr(request, "tool_name", None)
        or _tool_call_field(request, "name")
        or "unknown"
    )


def _tool_args(request: Any) -> Any:
    return _tool_call_field(request, "args") or getattr(request, "args", None) or {}


def _tool_call_id(request: Any) -> str:
    return str(_tool_call_field(request, "id") or "")


class _LoopGuardMiddleware(AgentMiddleware):
    def __init__(self, *, max_repeats: int) -> None:
        self._max_repeats = max_repeats
        self._last_key: tuple[str, str] | None = None
        self._repeat_count: int = 0
        self._tripped: bool = False

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        from langgraph.errors import GraphBubbleUp

        name = _tool_name(request)
        key = (name, _args_hash(_tool_args(request)))

        if key == self._last_key:
            self._repeat_count += 1
        else:
            self._last_key = key
            self._repeat_count = 1
            self._tripped = False

        if self._repeat_count <= self._max_repeats:
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise

        if not self._tripped:
            self._tripped = True
            emit.reasoning_step(
                f"Loop guard: {name!r} called "
                f"{self._repeat_count} times with identical arguments — "
                f"halting further calls and forcing the model to "
                f"answer with what it has.",
                event_type="WARNING",
                tool_name=name,
                repeat_count=self._repeat_count,
            )
            logger.warning(
                "loop_guard tripped: tool=%r repeats=%d", name, self._repeat_count
            )

        return ToolMessage(
            content=(
                f"[loop_guard] Tool {name!r} has been called "
                f"{self._repeat_count} times in a row with identical "
                "arguments and is returning the same result. STOP "
                "calling this tool. Answer the user with the "
                "information you already have, or try a different "
                "tool / different arguments."
            ),
            tool_call_id=_tool_call_id(request),
            name=name,
        )


class LoopGuardMiddlewareFactory(Middleware):
    """Build the per-run loop guard middleware.

    The loop guard detects when the agent repeatedly calls the same tool
    with identical arguments. Once a call is repeated more than
    ``max_repeats`` times in a row, the guard stops invoking the tool and
    instead returns a synthetic :class:`ToolMessage` instructing the
    model to stop and answer with the information it already has,
    breaking the loop.
    """

    name = "loop_guard"

    def __init__(self, *, max_repeats: int = 2) -> None:
        """Set the default repeat threshold for guards built by this factory.

        Parameters
        ----------
        max_repeats : int, optional
            Number of consecutive identical tool calls allowed before the
            guard trips. This is the fallback used when a run's ``config``
            does not override ``max_repeats``. Defaults to ``2``.
        """
        self._max_repeats = max_repeats

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        """Build a loop guard middleware instance for a single run.

        Parameters
        ----------
        ctx : RunContext
            The run context (unused; present for interface compatibility).
        config : dict
            Per-run configuration. An integer ``"max_repeats"`` key, if
            present, overrides the factory default for this run.

        Returns
        -------
        AgentMiddleware
            A fresh loop guard middleware tracking repeat state for the
            run.
        """
        return _LoopGuardMiddleware(
            max_repeats=int(config.get("max_repeats", self._max_repeats))
        )
