"""Plugin ABCs — the runtime's swappable extension points.

Each plugin group is resolved from a Python entry point and must
subclass (or, for sub-agents, structurally match) the contract defined
here: :class:`AuthBackend` (identity resolution), :class:`ModelProvider`
(chat model construction), :class:`ToolSource` (agent tools),
:class:`Middleware` (deepagents middleware), :class:`CheckpointerProvider`
(LangGraph checkpointer lifecycle), and the :class:`SubAgentSpec` protocol
(sub-agent declarations). Every ABC carries a ``name`` class attribute
that matches the entry-point name operators select in configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import Request

from openbb_agent_server.runtime.principal import UserPrincipal

if TYPE_CHECKING:  # pragma: no cover — types-only
    from openbb_agent_server.runtime.context import RunContext


class AuthBackend(ABC):
    """Resolve a request's credentials into a ``UserPrincipal``."""

    name: str

    @abstractmethod
    async def authenticate(self, request: Request) -> UserPrincipal:
        """Resolve the request's credentials into a ``UserPrincipal``.

        Implementers must inspect the incoming request (headers, cookies,
        bearer token, etc.), verify the caller, and return the resolved
        principal. On missing or invalid credentials they must raise rather
        than return an anonymous principal.

        Parameters
        ----------
        request : fastapi.Request
            The inbound HTTP request whose credentials are to be validated.

        Returns
        -------
        UserPrincipal
            The authenticated caller's identity.

        Raises
        ------
        fastapi.HTTPException
            When authentication fails (e.g. missing or invalid credentials).
        """


class ModelProvider(ABC):
    """Build a LangChain chat model for the current run."""

    name: str

    @abstractmethod
    def build(self, ctx: RunContext, config: dict[str, Any]) -> Any:
        """Return a ``BaseChatModel``-compatible chat model for the run.

        Parameters
        ----------
        ctx : RunContext
            The active run's context, carrying per-request data such as
            resolved API keys.
        config : dict[str, Any]
            Provider-specific overrides for this run (e.g. ``model_name``);
            keys not understood by the provider are ignored.

        Returns
        -------
        Any
            A LangChain ``BaseChatModel``-compatible instance.
        """


class ToolSource(ABC):
    """Yield the agent's LangChain tools for one run."""

    name: str

    @abstractmethod
    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Return the agent's LangChain tools for one run.

        Parameters
        ----------
        ctx : RunContext
            The active run's context, used to scope or authorize the tools.
        config : dict[str, Any]
            Source-specific configuration for this run; unrecognised keys
            are ignored.

        Returns
        -------
        list[Any]
            LangChain ``BaseTool`` instances available to the agent.
        """


class SubAgentSpec(Protocol):
    """Structural type for one sub-agent declaration consumed by deepagents.

    Attributes
    ----------
    name : str
        Unique identifier for the sub-agent.
    description : str
        Human-readable summary of the sub-agent's purpose.
    system_prompt : str
        System prompt that defines the sub-agent's behavior.
    tools : tuple[str, ...]
        Names of the tools the sub-agent is permitted to use.
    model : str | None
        Optional model override; ``None`` falls back to the default model.
    """

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]
    model: str | None


class Middleware(ABC):
    """A deepagents middleware factory."""

    name: str

    @abstractmethod
    def build(self, ctx: RunContext, config: dict[str, Any]) -> Any:
        """Return a deepagents middleware instance for the run.

        Parameters
        ----------
        ctx : RunContext
            The active run's context.
        config : dict[str, Any]
            Middleware-specific configuration for this run; unrecognised
            keys are ignored.

        Returns
        -------
        Any
            A deepagents middleware instance.
        """


class CheckpointerProvider(ABC):
    """Build (and lifecycle-manage) the LangGraph checkpointer."""

    name: str

    @abstractmethod
    async def open(self, settings: Any) -> Any:
        """Open and ``setup()`` the LangGraph checkpointer saver.

        Implementers must allocate any backing resources (connections,
        files) and run the saver's ``setup()`` before returning the live,
        ready-to-use instance.

        Parameters
        ----------
        settings : Any
            Provider-specific settings describing where and how to persist
            checkpoints (e.g. a connection string or file path).

        Returns
        -------
        Any
            The live checkpointer saver instance.
        """

    @abstractmethod
    async def close(self, saver: Any) -> None:
        """Tear down any resources owned by the saver.

        Implementers must release connections, file handles, or other
        resources acquired in :meth:`open`.

        Parameters
        ----------
        saver : Any
            The saver instance previously returned by :meth:`open`.
        """
