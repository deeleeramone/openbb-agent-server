"""Shared service slots populated at app startup."""

from __future__ import annotations

from typing import Any

from openbb_agent_server.memory.store import MemoryStore
from openbb_agent_server.persistence.store import HistoryStore


class _Services:
    history: HistoryStore | None = None
    memory: MemoryStore | None = None
    checkpointer: Any = None
    extra: dict[str, Any] = {}


_services = _Services()


def set_services(
    *,
    history: HistoryStore | None = None,
    memory: MemoryStore | None = None,
    checkpointer: Any = None,
    **extra: Any,
) -> None:
    """Bind shared services at startup. Call exactly once.

    Only the arguments that are not ``None`` overwrite the corresponding slot,
    so the call is additive: a service already bound by an earlier call is left
    untouched when its argument is omitted. Keyword extras are merged into the
    ``extra`` mapping rather than replacing it.

    Parameters
    ----------
    history : HistoryStore or None, optional
        Conversation history store. Bound only when not ``None``.
    memory : MemoryStore or None, optional
        Long-term memory store. Bound only when not ``None``.
    checkpointer : Any, optional
        LangGraph checkpointer. Bound only when not ``None``.
    **extra : Any
        Additional named services (for example ``widget_store`` or
        ``pdf_store``) merged into the shared ``extra`` mapping.
    """
    if history is not None:
        _services.history = history
    if memory is not None:
        _services.memory = memory
    if checkpointer is not None:
        _services.checkpointer = checkpointer
    _services.extra.update(extra)


def get_history() -> HistoryStore:
    """Return the bound :class:`~openbb_agent_server.persistence.store.HistoryStore` or raise.

    Returns
    -------
    HistoryStore
        The history store bound via :func:`set_services`.

    Raises
    ------
    RuntimeError
        If no history store has been bound at startup.
    """
    if _services.history is None:
        raise RuntimeError("HistoryStore not bound; call set_services() at startup")
    return _services.history


def get_memory() -> MemoryStore | None:
    """Return the bound :class:`~openbb_agent_server.memory.store.MemoryStore`, or ``None`` if memory is disabled."""
    return _services.memory


def get_widget_store() -> Any:
    """Return the bound :class:`~openbb_agent_server.runtime.widget_store.WidgetDataStore`, or ``None`` if absent."""
    return _services.extra.get("widget_store")


def get_pdf_store() -> Any:
    """Return the bound :class:`~openbb_agent_server.runtime.pdf_store.PdfStore`, or ``None`` if absent."""
    return _services.extra.get("pdf_store")


def get_checkpointer() -> Any:
    """Return the bound LangGraph checkpointer.

    Returns
    -------
    Any
        The checkpointer bound via :func:`set_services`.

    Raises
    ------
    RuntimeError
        If no checkpointer has been configured at startup.
    """
    if _services.checkpointer is None:
        raise RuntimeError(
            "Checkpointer not bound; the app must configure one at startup"
        )
    return _services.checkpointer


def reset() -> None:
    """Test-only: forget all bound services."""
    _services.history = None
    _services.memory = None
    _services.checkpointer = None
    _services.extra.clear()
