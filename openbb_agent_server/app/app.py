"""FastAPI app factory + lifespan."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, NamedTuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openbb_agent_server.app.config import (
    agent_section,
    bootstrap_launcher_config,
)
from openbb_agent_server.app.router import build_router
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.memory.factory import (
    make_embeddings,
    make_reranker,
    make_translator,
)
from openbb_agent_server.memory.sqlite_store import SqliteMemoryStore
from openbb_agent_server.observability.logging import install_trace_logging
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore
from openbb_agent_server.runtime import registry, services
from openbb_agent_server.runtime.identity import warn_if_pepper_unset
from openbb_agent_server.runtime.pdf_store import PdfStore
from openbb_agent_server.runtime.plugins import AuthBackend, CheckpointerProvider
from openbb_agent_server.runtime.widget_store import WidgetDataStore

logger = logging.getLogger("openbb_agent_server.app")


def _load_auth(settings: AgentServerSettings) -> AuthBackend:
    return registry.load(
        "openbb_agent_server.auth",
        settings.auth_backend,
        settings.auth_config,
    )


def _load_checkpointer_provider(
    settings: AgentServerSettings,
) -> CheckpointerProvider:
    return registry.load(
        "openbb_agent_server.checkpointers",
        settings.checkpointer_provider,
        settings.checkpointer_config,
    )


class AgentStores(NamedTuple):
    """Bundle the persistence and memory stores shared across runtimes.

    Returned by :func:`build_stores` and reused by both the HTTP app and
    the embedded runtime so the two compositions wire identical stores.

    Attributes
    ----------
    history : SqliteHistoryStore
        Conversation / run history persistence.
    memory : SqliteMemoryStore
        Long-term semantic memory store with embeddings and reranking.
    translator : Any or None
        Optional query translator, or ``None`` when not configured.
    widget_store : WidgetDataStore
        Store for widget payloads referenced by tool calls.
    pdf_store : PdfStore
        Store for ingested PDF content and its embeddings.
    """

    history: SqliteHistoryStore
    memory: SqliteMemoryStore
    translator: Any | None
    widget_store: WidgetDataStore
    pdf_store: PdfStore


def build_stores(settings: AgentServerSettings) -> AgentStores:
    """Construct the persistence + memory stores and bind shared services.

    Used by ``create_app`` and by ``runtime.embedded.EmbeddedRuntime`` so
    both compositions wire the exact same stores. Call once per process —
    the service slots are global.

    Parameters
    ----------
    settings : AgentServerSettings
        Resolved settings selecting the database URL, embeddings,
        reranker, and translation providers.

    Returns
    -------
    AgentStores
        The constructed history, memory, widget, and PDF stores plus the
        optional translator. The history/memory/widget/PDF stores are
        also bound into the shared service slots as a side effect.
    """
    db_url = settings.resolved_db_url()
    history = SqliteHistoryStore(db_url)
    embeddings = make_embeddings(
        settings.embeddings_provider,
        model=settings.embeddings_model,
        config=settings.embeddings_config,
    )
    code_embeddings = None
    if settings.embeddings_code_provider:
        code_embeddings = make_embeddings(
            settings.embeddings_code_provider,
            model=settings.embeddings_code_model,
            config=settings.embeddings_code_config,
        )
    reranker = (
        make_reranker(
            settings.reranker_provider,
            model=settings.reranker_model,
            config=settings.reranker_config,
        )
        if settings.reranker_provider
        else None
    )
    memory = SqliteMemoryStore(
        db_url,
        embeddings=embeddings,
        code_embeddings=code_embeddings,
        reranker=reranker,
        rerank_fanout=settings.rerank_fanout,
    )
    translator = (
        make_translator(
            settings.translation_provider,
            model=settings.translation_model,
            config=settings.translation_config,
        )
        if settings.translation_provider
        else None
    )
    widget_store = WidgetDataStore(db_url)
    pdf_store = PdfStore(db_url, embeddings=embeddings)
    services.set_services(widget_store=widget_store, pdf_store=pdf_store)
    services.set_services(history=history, memory=memory)
    return AgentStores(
        history=history,
        memory=memory,
        translator=translator,
        widget_store=widget_store,
        pdf_store=pdf_store,
    )


def create_app(settings: AgentServerSettings | None = None) -> FastAPI:
    """Build and wire the FastAPI application.

    Resolve settings (bootstrapping from TOML when none are passed),
    install trace logging, load the auth and checkpointer plugins, build
    the shared stores, register the API router, and attach a lifespan that
    initialises the history schema, opens the checkpointer, and runs the
    optional retention prune sweep. Optionally mounts the workspace-mcp
    sub-app when enabled.

    Parameters
    ----------
    settings : AgentServerSettings or None
        Server settings; when ``None`` they are loaded from the bootstrap
        TOML (path from ``OPENBB_AGENT_BOOTSTRAP_TOML`` if set).

    Returns
    -------
    FastAPI
        The configured application, with ``settings``, ``auth``,
        ``history``, and ``memory`` exposed on ``app.state``.
    """
    if settings is None:
        explicit = os.environ.get("OPENBB_AGENT_BOOTSTRAP_TOML") or None
        cfg = bootstrap_launcher_config(explicit_path=explicit)
        settings = AgentServerSettings.from_toml(agent_section(cfg))
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    install_trace_logging()
    warn_if_pepper_unset()

    auth = _load_auth(settings)
    checkpointer_provider = _load_checkpointer_provider(settings)
    stores = build_stores(settings)
    history = stores.history
    memory = stores.memory
    translator = stores.translator
    widget_store = stores.widget_store
    checkpointer: Any = None

    async def _prune_sweep() -> None:
        from openbb_agent_server.persistence.prune import run_prune

        interval = max(1, settings.prune_interval_hours) * 3600
        while True:
            try:
                await run_prune(
                    history=history,
                    checkpoint_path=settings.resolved_checkpoint_path(),
                    history_retention_days=settings.history_retention_days,
                    checkpoint_retention_days=settings.checkpoint_retention_days,
                    checkpoint_keep_last=settings.checkpoint_keep_last,
                )
            except Exception:  # noqa: BLE001 — a sweep failure must not kill the loop
                logger.warning(
                    "prune sweep failed; retrying next interval", exc_info=True
                )
            await asyncio.sleep(interval)

    retention_enabled = settings.prune_interval_hours > 0 and (
        settings.history_retention_days is not None
        or settings.checkpoint_retention_days is not None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal checkpointer
        await history.init_schema()
        checkpointer = await checkpointer_provider.open(settings)
        services.set_services(checkpointer=checkpointer)
        _app.state.checkpointer = checkpointer
        sweep: asyncio.Task[None] | None = (
            asyncio.create_task(_prune_sweep()) if retention_enabled else None
        )
        try:
            yield
        finally:
            if sweep is not None:
                sweep.cancel()
                with suppress(asyncio.CancelledError):
                    await sweep
            try:
                await checkpointer_provider.close(checkpointer)
            finally:
                await history.aclose()
                services.reset()

    app = FastAPI(
        title=settings.metadata.name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.include_router(
        build_router(
            settings=settings,
            auth=auth,
            history=history,
            memory=memory,
            translator=translator,
            widget_store=widget_store,
        ),
    )

    if settings.mount_workspace_mcp:
        _mount_workspace_mcp(app, settings)

    app.state.settings = settings
    app.state.auth = auth
    app.state.history = history
    app.state.memory = memory
    return app


def _mount_workspace_mcp(app: FastAPI, settings: AgentServerSettings) -> None:
    """Mount the optional workspace-mcp Starlette sub-app at /mcp/workspace.

    Soft-skips with a single info log when the ``[workspace-mcp]`` extra is
    not installed. Operators install it with::

        pip install 'openbb-agent-server[workspace-mcp]'
    """
    try:
        from workspace_mcp.app import create_app as _create_workspace_mcp
        from workspace_mcp.config import Settings as _WorkspaceMcpSettings
    except ImportError:
        logger.info(
            "workspace-mcp extra not installed; in-process mount skipped. "
            "Install with: pip install 'openbb-agent-server[workspace-mcp]'"
        )
        return

    try:
        wm_settings = _WorkspaceMcpSettings(**settings.workspace_mcp_config)
        app.mount("/mcp/workspace", _create_workspace_mcp(wm_settings))
        logger.info("Mounted workspace-mcp at /mcp/workspace")
    except Exception:  # noqa: BLE001 — mount failure must not break the parent app
        logger.warning(
            "workspace-mcp mount failed; continuing without it",
            exc_info=True,
        )
