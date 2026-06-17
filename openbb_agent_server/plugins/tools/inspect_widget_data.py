"""``inspect_widget_data`` tool source."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.runtime import (
    context as run_context,
    emit,
    services,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.inspect_widget_data")


class _ListArgs(BaseModel):
    pass


class _ReadArgs(BaseModel):
    widget_uuid: str | None = Field(
        default=None,
        description=(
            "Per-instance widget uuid (e.g. "
            "'759b7d14-2c9a-4046-9f04-56bd22c90068') — preferred. "
            "Take this verbatim from the widget snapshot's "
            "``widget_uuid=...`` field. Provide either widget_uuid OR "
            "widget_name."
        ),
    )
    widget_name: str | None = Field(
        default=None,
        description=(
            "Internal widget_id (e.g. 'blk_alloc_currency', "
            "'balance'). Case-insensitive match against the store's "
            "``widget_name`` column. DO NOT pass the human-readable "
            "display label like 'Currency Exposure' — the store does "
            "not key by display labels. Use widget_uuid whenever the "
            "snapshot gives you one."
        ),
    )
    max_rows: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="Optional cap on the number of rows returned.",
    )


class _SearchArgs(BaseModel):
    query: str = Field(description="Natural-language query to match rows against.")
    k: int = Field(
        default=8,
        ge=1,
        le=100,
        description="How many top results to return.",
    )
    widget_uuid: str | None = Field(
        default=None,
        description="Optional: restrict to one widget by uuid.",
    )


class _SchemaArgs(BaseModel):
    pass


class _QueryArgs(BaseModel):
    sql: str = Field(
        description=(
            "A single SQLite SELECT (or WITH … SELECT …) statement to "
            "run against the ingested widget tables. Table names come "
            "from describe_widget_data — use them verbatim. All column "
            "values are stored as TEXT; cast with CAST(col AS REAL) "
            "before arithmetic."
        ),
    )
    max_rows: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Cap on rows returned (default 200).",
    )


class InspectWidgetDataToolSource(ToolSource):
    """Provide tools for inspecting and querying ingested widget data.

    Exposes five LangChain tools over the per-conversation widget store:
    ``list_widget_data`` (what has been fetched), ``read_widget_data``
    (full rows for one widget), ``search_widget_data`` (semantic/substring
    row search), ``describe_widget_data`` (the SQL surface), and
    ``query_widget_data`` (read-only SQL). Each tool resolves the active
    run's principal and emits citations for the dashboard widgets it
    touches.
    """

    name = "inspect_widget_data"

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Bind the per-run widget-data inspection tools.

        Builds lookup maps from the run's pinned dashboard widgets (by uuid
        and by internal ``widget_id``) so the inner tools can emit a
        citation whenever they surface a stored widget's data. Returns
        freshly bound :class:`StructuredTool` coroutines closed over this
        run's context.

        Parameters
        ----------
        ctx : RunContext
            The active run context. Its ``widgets`` are used to map stored
            entries back to dashboard widgets for citation.
        config : dict[str, Any]
            Per-run tool configuration (currently unused).

        Returns
        -------
        list[BaseTool]
            The five widget-data inspection tools described in the class
            docstring.
        """
        cited: set[str] = set()
        dashboard_by_uuid: dict[str, Any] = {}
        dashboard_by_widget_id: dict[str, Any] = {}
        for w in ctx.widgets or []:
            if w.uuid:
                dashboard_by_uuid[w.uuid] = w
            wid = (getattr(w, "widget_id", "") or "").strip()
            if wid:
                dashboard_by_widget_id[wid] = w

        def _cite_widget(entry: dict[str, Any] | None) -> None:
            if not isinstance(entry, dict):
                return
            stored_uuid = str(entry.get("widget_uuid") or "")
            stored_name = str(entry.get("widget_name") or "")
            pinned = dashboard_by_uuid.get(stored_uuid) or dashboard_by_widget_id.get(
                stored_name
            )
            if pinned is None:
                return
            dashboard_uuid = pinned.uuid or stored_uuid
            if not dashboard_uuid or dashboard_uuid in cited:
                return
            cited.add(dashboard_uuid)
            display = (
                getattr(pinned, "name", None)
                or getattr(pinned, "widget_id", None)
                or stored_name
                or None
            )
            internal_id = (
                (getattr(pinned, "widget_id", "") or "").strip() or stored_name or None
            )
            emit.cite(
                widget=dashboard_uuid,
                widget_id=internal_id,
                source=display,
                source_url=str(entry.get("origin") or "")
                or (getattr(pinned, "origin", None) or None),
                input_arguments=entry.get("input_args") or {},
            )

        list_called = {"v": False}

        async def list_widget_data() -> dict[str, Any]:
            store = services.get_widget_store()
            current = run_context.current()
            entries: list[dict[str, Any]] = []
            if store is not None:
                entries = await store.list_entries(
                    principal=current.principal,
                    conversation_id=None,
                )
            if list_called["v"]:
                return {
                    "count": len(entries),
                    "widgets": entries,
                    "message": (
                        "list_widget_data was already called this turn — "
                        "the result is identical. STOP calling it and use "
                        "the widgets snapshot from the system prompt to "
                        "decide what to fetch via get_widget_data, or "
                        "answer from what you already have."
                    ),
                }
            list_called["v"] = True
            if not entries:
                attached = [
                    {
                        "widget_id": w.uuid,
                        "name": getattr(w, "name", None) or w.widget_id,
                    }
                    for w in current.widgets
                ]
                return {
                    "count": 0,
                    "widgets": [],
                    "attached_widgets": attached,
                    "message": (
                        "No widget data has been fetched yet this "
                        "conversation. Call get_widget_data(widget_ids=[..])"
                        " with the widget_id values from "
                        "``attached_widgets`` for the widgets you need, "
                        "then end the turn — Workspace returns the rows "
                        "on the next user message."
                    ),
                }
            return {"count": len(entries), "widgets": entries}

        async def read_widget_data(
            widget_uuid: str | None = None,
            widget_name: str | None = None,
            max_rows: int | None = None,
        ) -> dict[str, Any] | None:
            store = services.get_widget_store()
            if store is None:
                return None
            current = run_context.current()
            result = await store.read_latest(
                principal=current.principal,
                conversation_id=None,
                widget_uuid=widget_uuid,
                widget_name=widget_name,
                max_rows=max_rows,
            )
            _cite_widget(result)
            return result

        async def search_widget_data(
            query: str,
            k: int = 8,
            widget_uuid: str | None = None,
        ) -> list[dict[str, Any]]:
            store = services.get_widget_store()
            if store is None:
                return []
            current = run_context.current()
            hits = await store.search(
                principal=current.principal,
                conversation_id=None,
                query=query,
                k=k,
                widget_uuid=widget_uuid,
            )
            for h in hits:
                _cite_widget(h)
            return hits

        async def describe_widget_data() -> list[dict[str, Any]]:
            store = services.get_widget_store()
            if store is None:
                return []
            current = run_context.current()
            return await store.schema(
                principal=current.principal,
                conversation_id=None,
            )

        async def _cite_widgets_referenced_by_sql(sql_lower: str) -> None:
            store = services.get_widget_store()
            if store is None:
                return
            current = run_context.current()
            try:
                schema = await store.schema(
                    principal=current.principal,
                    conversation_id=None,
                )
            except Exception:
                return
            for entry in schema:
                table = str(entry.get("table") or "").lower()
                if table and table in sql_lower:
                    _cite_widget(entry)

        async def query_widget_data(
            sql: str,
            max_rows: int = 200,
        ) -> dict[str, Any]:
            store = services.get_widget_store()
            if store is None:
                return {
                    "error": "widget store unavailable",
                    "columns": [],
                    "rows": [],
                }
            current = run_context.current()
            try:
                result = await store.query(
                    principal=current.principal,
                    conversation_id=None,
                    sql=sql,
                    max_rows=max_rows,
                )
            except Exception as exc:
                logger.warning("query_widget_data failed: %s", exc)
                return {
                    "error": str(exc),
                    "columns": [],
                    "rows": [],
                }
            await _cite_widgets_referenced_by_sql(sql.lower())
            return result

        return [
            StructuredTool.from_function(
                coroutine=list_widget_data,
                name="list_widget_data",
                description=(
                    "List every widget whose data has ALREADY been "
                    "fetched this conversation. Returns "
                    "[{id, widget_uuid, widget_name, origin, input_args, "
                    "columns, row_count, ingested_at}]. Empty list means "
                    "nothing has been fetched yet — call "
                    "``get_widget_data(widget_ids=[...])`` ONCE with "
                    "every widget you need from the snapshot in the "
                    "system prompt."
                ),
                args_schema=_ListArgs,
            ),
            StructuredTool.from_function(
                coroutine=read_widget_data,
                name="read_widget_data",
                description=(
                    "Read the full row set for one previously-fetched "
                    "widget. PREFERRED: pass widget_uuid (the "
                    "per-instance UUID from the system-prompt widget "
                    "snapshot). Alternatively pass widget_name (the "
                    "internal widget_id like 'blk_alloc_currency', "
                    "NOT the human-readable display label). Returns "
                    "{widget_uuid, widget_name, origin, input_args, "
                    "columns, rows, ingested_at}, or None if no match."
                ),
                args_schema=_ReadArgs,
            ),
            StructuredTool.from_function(
                coroutine=search_widget_data,
                name="search_widget_data",
                description=(
                    "Semantic-search rows across all fetched widgets in "
                    "this conversation. Returns top-k "
                    "[{score, row, widget_uuid, widget_name}] sorted by "
                    "relevance to the query. Uses vector embeddings "
                    "when configured; falls back to substring match. "
                    "Prefer this when you only need a few specific rows "
                    "(e.g. 'cash and short-term debt') instead of the "
                    "full table."
                ),
                args_schema=_SearchArgs,
            ),
            StructuredTool.from_function(
                coroutine=describe_widget_data,
                name="describe_widget_data",
                description=(
                    "Return the SQL surface for query_widget_data: one "
                    "entry per ingested widget with [{table, widget_name, "
                    "widget_uuid, columns, row_count}]. ``table`` is the "
                    "name you reference in SQL; ``columns`` are the column "
                    'names (all stored as TEXT — cast with CAST("col" '
                    "AS REAL) for arithmetic). Call this before "
                    "query_widget_data."
                ),
                args_schema=_SchemaArgs,
            ),
            StructuredTool.from_function(
                coroutine=query_widget_data,
                name="query_widget_data",
                description=(
                    "Run a READ-ONLY SQL query (SELECT / WITH only) "
                    "against the persisted widget_data SQL table. Each "
                    "ingested widget is exposed as a temp view named "
                    "after the widget (see describe_widget_data for "
                    "names + columns). All columns are TEXT — cast with "
                    'CAST("col" AS REAL) before SUM / arithmetic. '
                    "Returns {columns, rows, table_count, truncated}. "
                    "Use this for aggregations, filters, joins across "
                    "widgets — anything beyond reading a row set."
                ),
                args_schema=_QueryArgs,
            ),
        ]
