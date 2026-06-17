"""Async SQLite checkpointer with proper pragmas."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from openbb_agent_server.runtime.plugins import CheckpointerProvider

logger = logging.getLogger("openbb_agent_server.checkpointer.sqlite")

_BUSY_TIMEOUT_MS = 30_000


async def _configure_connection(conn: aiosqlite.Connection) -> None:
    """Apply the pragmas every writer to this file relies on."""
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA temp_store=MEMORY")
    await conn.commit()


class SqliteCheckpointerProvider(CheckpointerProvider):
    """Persistent SQLite-backed checkpointer.

    Provide a LangGraph :class:`AsyncSqliteSaver` backed by a single
    on-disk SQLite database, configured with WAL journaling and the
    other concurrency pragmas writers to the file expect.
    """

    name = "sqlite"

    def __init__(self, *, path: str | None = None, **_config: Any) -> None:
        """Store the optional explicit database path.

        Parameters
        ----------
        path : str or None, optional
            Explicit filesystem path for the checkpoint database. When
            ``None``, :meth:`~openbb_agent_server.runtime.plugins.CheckpointerProvider.open` falls back to the
            ``OPENBB_AGENT_CHECKPOINTER_PATH`` environment variable and
            then to ``checkpoints.db`` under the settings data directory.
        **_config : Any
            Additional configuration keys, accepted and ignored so the
            provider can be constructed from a generic plugin config.
        """
        self._explicit_path = path
        self._conn: aiosqlite.Connection | None = None

    async def open(self, settings: Any) -> AsyncSqliteSaver:
        """Open the SQLite connection and return a ready saver.

        Resolve the database path, create its parent directory, connect
        with the busy-timeout applied, configure the WAL pragmas, run the
        saver's schema setup, and cache the connection for :meth:`~openbb_agent_server.runtime.plugins.CheckpointerProvider.close`.

        Parameters
        ----------
        settings : Any
            Runtime settings object; only ``settings.data_dir`` is read,
            and only when no explicit path or environment override is set.

        Returns
        -------
        AsyncSqliteSaver
            A LangGraph checkpointer whose tables have been created and
            that writes to the resolved database file.
        """
        path = (
            self._explicit_path
            or os.environ.get("OPENBB_AGENT_CHECKPOINTER_PATH")
            or str(Path(settings.data_dir) / "checkpoints.db")
        )
        await asyncio.to_thread(Path(path).parent.mkdir, parents=True, exist_ok=True)

        conn = await aiosqlite.connect(
            path,
            timeout=_BUSY_TIMEOUT_MS / 1000.0,
        )
        await _configure_connection(conn)
        self._conn = conn

        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        logger.info("sqlite checkpointer opened at %s", path)
        return saver

    async def close(self, saver: Any) -> None:
        """Close the cached SQLite connection.

        Parameters
        ----------
        saver : Any
            The saver returned by :meth:`~openbb_agent_server.runtime.plugins.CheckpointerProvider.open`. It is discarded; the
            underlying connection is the resource that gets closed.
        """
        del saver
        if self._conn is not None:
            try:
                await self._conn.close()
            finally:
                self._conn = None
