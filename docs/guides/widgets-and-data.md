# Widgets and data

OpenBB Workspace dashboards are made of **widgets** — each is a structured data source (price chart, balance sheet, options chain, etc.). When a user has widgets pinned and chats with the agent, those widgets' data needs to make it into the conversation in a way the model can actually reason over.

The agent server handles this in three layers:

1. **Wire layer.** Workspace forwards selected widgets in `QueryRequest.widgets.{primary, secondary, extra}`.
2. **Fetch layer.** The agent either auto-fetches via `FunctionCallSSE(function="get_widget_data")` or calls the `get_widget_data` server-side tool.
3. **Inspect layer.** Once data has been ingested, the agent uses `inspect_widget_data` to `list / read / search / describe / query` it via SQL.

## The `widgets` field

```jsonc
"widgets": {
  "primary":   [{ "uuid": "...", "widget_id": "balance_sheet", "origin": "openbb", "params": [...] }],
  "secondary": [...],
  "extra":     [...]
}
```

`primary` is what the user pinned and selected. `secondary` is everything else on the dashboard. `extra` is widgets Workspace thinks might be relevant.

`WidgetRef` (`runtime/context.py`) is the runtime representation. Declared fields: `uuid`, `widget_id`, `origin`, `params`, and `data` (filled in only after a `get_widget_data` round-trip). The model has `extra="allow"`, so additional Workspace fields such as `name` and `description` ride through under `model_extra` without being part of the public schema.

## Auto-fetch flow

When the last user turn is a `human` message, no tool result has been seen yet, and `widgets.primary` is non-empty, the router emits two SSE frames before invoking the agent:

1. `StatusUpdateSSE` — "Fetching {widget names} data from Workspace…"
2. `FunctionCallSSE` — `{ "function": "get_widget_data", "input_arguments": { "data_sources": [...] } }`

The SSE stream closes. Workspace fetches the data and re-posts to `/v1/query` with the result as a `role:"tool"` message. The router's second-turn handling runs `parse_widget_data_messages`, which extracts the data-source ↔ payload pairs.

## Ingestion

`runtime/widget_store.py::WidgetDataStore.record` persists each ingest into the `widget_data` SQL table (`persistence/models.py::WidgetData`) — the row dicts plus metadata (`user_id`, `conversation_id`, `widget_uuid`, `widget_name`, `origin`, `input_args`, `columns`, `ingested_at`).

Tabular rows are **not** vector-indexed: structured data is queried with SQL (`query_widget_data`) and keyword-matched (`search_widget_data`), both of which beat semantic similarity over numbers and tickers.

## Agent-facing tools

`inspect_widget_data` (`plugins/tools/inspect_widget_data.py`) exposes the store as five `StructuredTool`s the model can call:

| Tool | Purpose |
| --- | --- |
| `list_widget_data` | Enumerate every widget ingested for this conversation. Returns `[{id, widget_uuid, widget_name, origin, input_args, columns, row_count, ingested_at}]`. |
| `read_widget_data` | Full row set for one widget by `widget_uuid` or `widget_name`. Honors `max_rows` for sampling. |
| `search_widget_data` | Keyword (substring) search across all rows in this conversation. Returns `[{score, row, widget_uuid, widget_name}]`. For filters / aggregates use `query_widget_data`. |
| `describe_widget_data` | Schema view for `query_widget_data`. Returns `[{table, widget_name, widget_uuid, columns, row_count}]` — `table` is the slug-cased name to reference in SQL. |
| `query_widget_data` | Read-only SQLite SQL over the ingested rows exposed as temp views. Supports `SELECT` and `WITH … SELECT …` only; columns are TEXT — use `CAST("col" AS REAL)` for arithmetic. |

The agent typically chains: `list_widget_data` → either `read_widget_data` (small sample) / `search_widget_data` (one or two rows by relevance) / `query_widget_data` (aggregates, joins, filters across widgets).

## SQL surface

`query_widget_data` builds one `CREATE TEMP VIEW` per ingested widget. The slug is derived from `widget_name` (falling back to `widget_uuid`, falling back to `widget_{id}`). Same slug for the same widget across turns → table names stable for the conversation lifetime.

```python
sql = """
WITH cap AS (
    SELECT CAST("market_cap" AS REAL) AS mc FROM "balance_sheet"
)
SELECT SUM(mc) FROM cap;
"""
```

Validation rules:

- `SELECT` or `WITH` only.
- Each column lives in `WidgetData.rows` as JSON; views unfold them via `json_each(wd.rows)` and `json_extract(je.value, '$."col"')`.
- Results are capped at `max_rows` (default 500); `truncated: true` flags when the cap hit.

## Direct retriever

For non-agent code, `WidgetDataRetriever` (`memory/retrievers.py`) wraps `WidgetDataStore.search` in a `BaseRetriever`:

```python
from openbb_agent_server.memory.retrievers import WidgetDataRetriever

retriever = WidgetDataRetriever(
    store=widget_store,
    principal=principal,
    conversation_id="...",
    widget_uuid="...",          # optional filter
    k=8,
)
docs = await retriever.ainvoke("cash and short-term debt")
```

Each `Document` carries `metadata = {widget_uuid, widget_name, score, row}` so downstream consumers can recover the original row dict.

## When to use which

| You need… | Use |
| --- | --- |
| The whole table | `read_widget_data(widget_name=…, max_rows=…)` |
| A few specific rows by name / keyword | `search_widget_data(query=…, k=…)` |
| Aggregates / filters / joins across widgets | `query_widget_data(sql=…)` |
| Compose with other LangChain retrievers | `WidgetDataRetriever` |

## Tests

The fixture `tests/test_tool_inspect_widget_data.py` exercises every tool end-to-end with a real SQLite store. `tests/test_widget_store.py` covers the storage layer (record / substring search / pinned-fanout / SQL query).

## Source

- [`runtime.widget_store`](../reference/runtime/widget_store.md)
- [`plugins.tools.inspect_widget_data`](../reference/plugins/tools/inspect_widget_data.md)
- [`plugins.tools.widget_data`](../reference/plugins/tools/widget_data.md)
- [`memory.retrievers`](../reference/memory/retrievers.md)
- [`app.router`](../reference/app/router.md) — `_run_query` ingestion path
