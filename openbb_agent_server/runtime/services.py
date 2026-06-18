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


async def areset() -> None:
    """Close and forget all bound services (async version).

    Properly ``await``-disposes every async engine so that
    ``aiosqlite.Connection`` objects are closed cleanly.
    """
    import asyncio

    from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
    from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
    from openbb_agent_server.runtime.pdf_store import PdfStore
    from openbb_agent_server.runtime.widget_store import WidgetDataStore

    mem = _services.memory
    if isinstance(mem, SqliteMemoryStore):
        mem.close()

    hist = _services.history
    if isinstance(hist, SqliteHistoryStore):
        await hist.aclose()

    pdf = _services.extra.get("pdf_store")
    if isinstance(pdf, PdfStore):
        await pdf.aclose()

    ws = _services.extra.get("widget_store")
    if isinstance(ws, WidgetDataStore):
        await ws.aclose()

    # Give aiosqlite's background threads time to process close messages.
    await asyncio.sleep(0)

    _services.history = None
    _services.memory = None
    _services.checkpointer = None
    _services.extra.clear()


def reset() -> None:
    """Forget all bound services (sync).

    Closes synchronous ``sqlite3.Connection`` objects held by the
    memory store and PDF store. Does **not** dispose async engines —
    use :func:`areset` for that. Nulls all service slots so subsequent
    ``get_*`` calls fail cleanly.
    """
    from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
    from openbb_agent_server.runtime.pdf_store import PdfStore

    mem = _services.memory
    if isinstance(mem, SqliteMemoryStore):
        mem.close()

    pdf = _services.extra.get("pdf_store")
    if isinstance(pdf, PdfStore) and pdf._vec_conn is not None:
        pdf._vec_conn.close()

    _services.history = None
    _services.memory = None
    _services.checkpointer = None
    _services.extra.clear()
