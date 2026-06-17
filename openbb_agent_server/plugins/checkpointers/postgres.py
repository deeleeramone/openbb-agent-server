"""Async Postgres checkpointer."""

from __future__ import annotations

import logging
import os
from typing import Any

from openbb_agent_server.runtime.plugins import CheckpointerProvider

logger = logging.getLogger("openbb_agent_server.checkpointer.postgres")


def _normalise_pg_url(url: str) -> str:
    """Strip SQLAlchemy-style driver suffixes that psycopg cannot parse."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://") :]
    return url


class PostgresCheckpointerProvider(CheckpointerProvider):
    """Persistent Postgres-backed checkpointer (production default).

    Wrap LangGraph's ``AsyncPostgresSaver`` so graph state is persisted to
    a Postgres database across runs. This is the recommended checkpointer
    for production deployments.

    Attributes
    ----------
    name : str
        Plugin registry key (``"postgres"``).
    """

    name = "postgres"

    def __init__(self, *, url: str | None = None, **_config: Any) -> None:
        """Initialize the provider with an optional explicit connection URL.

        Parameters
        ----------
        url : str or None, optional
            Explicit Postgres connection URL. When omitted, the URL is
            resolved at ``open`` time from the environment or settings.
        **_config : Any
            Extra plugin configuration keys, accepted and ignored so the
            provider can be constructed from a generic config mapping.
        """
        self._explicit_url = url
        self._cm: Any = None

    def _resolve_url(self, settings: Any) -> str:
        """Return the validated Postgres URL to connect against.

        Resolve the connection URL in priority order: the explicit URL
        passed to the constructor, then the ``OPENBB_AGENT_CHECKPOINTER_URL``
        environment variable, then ``settings.resolved_db_url()``. The
        result is normalised to strip SQLAlchemy driver suffixes.

        Parameters
        ----------
        settings : Any
            Settings object exposing ``resolved_db_url()`` as the
            final fallback source for the connection URL.

        Returns
        -------
        str
            A ``postgresql://`` connection URL accepted by psycopg.

        Raises
        ------
        RuntimeError
            If the resolved URL does not start with ``postgresql://``.
        """
        url = (
            self._explicit_url
            or os.environ.get("OPENBB_AGENT_CHECKPOINTER_URL")
            or settings.resolved_db_url()
        )
        url = _normalise_pg_url(url)
        if not url.startswith("postgresql://"):
            raise RuntimeError(
                f"Postgres checkpointer expects a postgresql:// URL, got {url!r}"
            )
        return url

    async def open(
        self, settings: Any
    ) -> Any:  # pragma: no cover — needs live Postgres
        """Open the saver and ensure its schema exists.

        Create an ``AsyncPostgresSaver`` from the resolved URL, enter its
        async context manager (retained for ``close``), run ``setup()`` to
        create the checkpoint tables if needed, and return the live saver.

        Parameters
        ----------
        settings : Any
            Settings object used to resolve the connection URL.

        Returns
        -------
        Any
            The opened ``AsyncPostgresSaver`` instance.

        Raises
        ------
        RuntimeError
            If ``langgraph-checkpoint-postgres`` (and psycopg) are not
            installed, or if the connection URL is not a valid
            ``postgresql://`` URL.
        """
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires langgraph-checkpoint-postgres "
                "and psycopg[binary]. Install the agent_server with the "
                "[postgres] extra."
            ) from exc

        url = self._resolve_url(settings)
        self._cm = AsyncPostgresSaver.from_conn_string(url)
        saver = await self._cm.__aenter__()
        await saver.setup()
        logger.info("postgres checkpointer opened against %s", url)
        return saver

    async def close(self, saver: Any) -> None:  # pragma: no cover — paired with open()
        """Close the saver opened by :meth:`~openbb_agent_server.runtime.plugins.CheckpointerProvider.open`.

        Exit the retained async context manager (releasing the connection
        pool) and clear the stored reference. Safe to call when ``open``
        was never invoked; in that case it is a no-op.

        Parameters
        ----------
        saver : Any
            The saver returned by :meth:`~openbb_agent_server.runtime.plugins.CheckpointerProvider.open`. Accepted for interface
            symmetry; teardown is driven by the stored context manager.
        """
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            finally:
                self._cm = None
