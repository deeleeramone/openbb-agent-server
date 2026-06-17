"""Call limit middleware."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import AgentMiddleware

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import Middleware


class _Composite(AgentMiddleware):
    """Wrapper that owns two child middlewares."""


class CallLimitMiddlewareFactory(Middleware):
    """Build a ``ModelCallLimitMiddleware`` that caps model calls per run.

    Registered under the ``call_limit`` plugin name. Despite accepting a
    tool-run limit for symmetry, ``build`` produces only the model call
    limit middleware; per-tool caps are provided by
    :class:`ToolCallLimitMiddlewareFactory`.
    """

    name = "call_limit"

    def __init__(
        self,
        *,
        model_run_limit: int | None = 40,
        tool_run_limit: int | None = 80,
        exit_behavior: str = "end",
    ) -> None:
        """Store the default limits used when config omits overrides.

        Parameters
        ----------
        model_run_limit : int or None, optional
            Default maximum number of model calls allowed per run.
            Defaults to 40.
        tool_run_limit : int or None, optional
            Default per-tool call cap retained for parity with the tool
            factory. Not consumed by this factory's ``build``. Defaults
            to 80.
        exit_behavior : str, optional
            Default action taken once a limit is reached, e.g. ``"end"``
            to terminate the run. Defaults to ``"end"``.
        """
        self._model_run_limit = model_run_limit
        self._tool_run_limit = tool_run_limit
        self._exit_behavior = exit_behavior

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        """Return a configured ``ModelCallLimitMiddleware`` instance.

        Parameters
        ----------
        ctx : RunContext
            Per-run context (unused by this factory but required by the
            ``Middleware`` interface).
        config : dict[str, Any]
            Per-build overrides. Recognized keys: ``model_run_limit``
            and ``exit_behavior``; each falls back to the constructor
            default when absent.

        Returns
        -------
        AgentMiddleware
            A ``ModelCallLimitMiddleware`` enforcing the resolved model
            call limit and exit behavior.
        """
        from typing import cast

        return cast(
            AgentMiddleware,
            ModelCallLimitMiddleware(
                run_limit=int(config.get("model_run_limit", self._model_run_limit)),
                exit_behavior=config.get("exit_behavior", self._exit_behavior),
            ),
        )


class ToolCallLimitMiddlewareFactory(Middleware):
    """Build a ``ToolCallLimitMiddleware`` that caps tool calls per run.

    Registered under the ``tool_call_limit`` plugin name.
    """

    name = "tool_call_limit"

    def __init__(
        self,
        *,
        tool_run_limit: int | None = 80,
        exit_behavior: str = "end",
    ) -> None:
        """Store the default tool limit used when config omits overrides.

        Parameters
        ----------
        tool_run_limit : int or None, optional
            Default maximum number of tool calls allowed per run.
            Defaults to 80.
        exit_behavior : str, optional
            Default action taken once the limit is reached, e.g.
            ``"end"`` to terminate the run. Defaults to ``"end"``.
        """
        self._tool_run_limit = tool_run_limit
        self._exit_behavior = exit_behavior

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        """Return a configured ``ToolCallLimitMiddleware`` instance.

        Parameters
        ----------
        ctx : RunContext
            Per-run context (unused by this factory but required by the
            ``Middleware`` interface).
        config : dict[str, Any]
            Per-build overrides. Recognized keys: ``tool_run_limit`` and
            ``exit_behavior``; each falls back to the constructor default
            when absent.

        Returns
        -------
        AgentMiddleware
            A ``ToolCallLimitMiddleware`` enforcing the resolved tool
            call limit and exit behavior.
        """
        from typing import cast

        return cast(
            AgentMiddleware,
            ToolCallLimitMiddleware(
                run_limit=int(config.get("tool_run_limit", self._tool_run_limit)),
                exit_behavior=config.get("exit_behavior", self._exit_behavior),
            ),
        )
