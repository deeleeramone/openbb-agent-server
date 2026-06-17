"""Per-request ``RunContext`` and the contextvar that propagates it."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from openbb_agent_server.runtime.principal import UserPrincipal


class WidgetRef(BaseModel):
    """One Workspace-supplied widget the user has selected as context.

    Extra keys sent by Workspace are preserved (``extra="allow"``) so
    forward-compatible fields survive the round trip.

    Attributes
    ----------
    uuid : str
        Stable identifier of the selected widget instance.
    widget_id : str
        Identifier of the widget template/type. Empty when unset.
    origin : str
        Origin app or data source the widget belongs to. Empty when
        unset.
    params : dict[str, Any]
        Parameter values configured on the widget (symbol, interval,
        etc.).
    data : Any
        The widget's resolved data payload, when Workspace inlines it.
    """

    model_config = ConfigDict(extra="allow")

    uuid: str
    widget_id: str = ""
    origin: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    data: Any = None


class FileRef(BaseModel):
    """One uploaded file (PDF / image / spreadsheet / raw).

    Exactly one of ``data_base64`` or ``url`` is expected to carry the
    bytes; extra keys are preserved (``extra="allow"``).

    Attributes
    ----------
    name : str
        Original file name, used for display and type inference.
    mime : str or None
        MIME type when known; ``None`` falls back to extension/content
        sniffing.
    data_base64 : str or None
        Base64-encoded file contents when the bytes are inlined.
    url : str or None
        Location to fetch the file from when it is not inlined.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    mime: str | None = None
    data_base64: str | None = None
    url: str | None = None


class RunContext(BaseModel):
    """Bundles identity + request payload for one ``/v1/query`` exchange.

    Created per turn and exposed to the agent loop, tools and plugins
    through :func:`current` while :func:`bind` is active.

    Attributes
    ----------
    principal : UserPrincipal
        The authenticated caller this run acts on behalf of.
    trace_id : str
        Identifier of the trace record for this run, for history and
        observability.
    run_id : str
        Identifier of this single turn within the conversation.
    conversation_id : str
        Identifier of the conversation; also the checkpointer thread key.
    agent_name : str
        Name of the resolved agent profile driving the run. Defaults to
        ``"default"``.
    timezone : str or None
        IANA timezone used to localize the run, when supplied.
    widgets : tuple[WidgetRef, ...]
        Workspace widgets the user attached as context.
    uploaded_files : tuple[FileRef, ...]
        Files attached to the request.
    api_keys : dict[str, str]
        Provider API keys resolved for this run, keyed by provider.
    api_urls : dict[str, str]
        Provider base URLs resolved for this run, keyed by provider.
    tools : tuple[dict[str, Any], ...]
        Client-supplied tool declarations forwarded to the agent.
    workspace_options : dict[str, Any]
        Per-user custom feature toggles; query with
        :meth:`has_workspace_option`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    principal: UserPrincipal
    trace_id: str
    run_id: str
    conversation_id: str
    agent_name: str = "default"
    timezone: str | None = None
    widgets: tuple[WidgetRef, ...] = ()
    uploaded_files: tuple[FileRef, ...] = ()
    api_keys: dict[str, str] = Field(default_factory=dict)
    api_urls: dict[str, str] = Field(default_factory=dict)
    tools: tuple[dict[str, Any], ...] = ()
    workspace_options: dict[str, Any] = Field(default_factory=dict)

    def has_workspace_option(self, slug: str) -> bool:
        """Return True iff the user has enabled the named custom feature.

        ``workspace_options`` is option-id-keyed values; a feature counts
        as enabled only when its value is truthy — a toggle left off
        arrives as ``False``.
        """
        return bool(self.workspace_options.get(slug))


_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "openbb_agent_server.run_context",
    default=None,
)

_runtime_state: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "openbb_agent_server.runtime_state",
    default=None,
)


def current() -> RunContext:
    """Return the ``RunContext`` bound to the current task.

    Returns
    -------
    RunContext
        The context set by the enclosing :func:`bind` block.

    Raises
    ------
    LookupError
        If no context is bound on the current task.
    """
    ctx = _current.get()
    if ctx is None:
        raise LookupError("No RunContext bound on this task")
    return ctx


def runtime_state() -> dict[str, Any]:
    """Return the per-run mutable scratch dict, scoped to this run.

    The dict is created fresh by :func:`bind` and shared by everything
    running within that ``with`` block; use it to stash per-run state.

    Returns
    -------
    dict
        The mutable scratch dict for the current run.

    Raises
    ------
    LookupError
        If no runtime state is bound on the current task.
    """
    state = _runtime_state.get()
    if state is None:
        raise LookupError("No runtime state bound on this task")
    return state


@contextmanager
def bind(ctx: RunContext) -> Iterator[RunContext]:
    """Bind ``ctx`` and a fresh runtime-state dict for the ``with`` block.

    On exit the runtime state is cleaned up via
    :func:`~openbb_agent_server.runtime.jobs.cleanup_state` and both
    contextvars are reset to their prior values, even on error.

    Parameters
    ----------
    ctx : RunContext
        Context to expose through :func:`current` for the block's
        duration.

    Yields
    ------
    RunContext
        The same ``ctx`` that was bound.
    """
    state: dict[str, Any] = {}
    ctx_token = _current.set(ctx)
    state_token = _runtime_state.set(state)
    try:
        yield ctx
    finally:
        from openbb_agent_server.runtime.jobs import cleanup_state

        cleanup_state(state)
        _runtime_state.reset(state_token)
        _current.reset(ctx_token)
