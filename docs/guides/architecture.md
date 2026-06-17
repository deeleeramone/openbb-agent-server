# Architecture

The agent server is a FastAPI app that translates the [OpenBB Workspace custom-agent SSE protocol](https://docs.openbb.co/workspace/developers/ai-features/share-step-by-step-reasoning) into a [LangChain `deepagents`](https://docs.langchain.com/oss/python/deepagents) loop, then translates the loop's stream events back into Workspace SSE frames.

## Process layout

```
                ┌───────────────────────────────────────────────────────┐
   Workspace ↔  │ FastAPI (app/router.py)                               │
   SSE          │   POST /v1/query  →  builder.run_agent(ctx, body)     │
                │   GET  /agents.json                                   │
                │   /v1/conversations, /v1/memory, /v1/traces, …        │
                ├───────────────────────────────────────────────────────┤
                │ runtime/builder.py                                    │
                │   loads plugins → create_deep_agent(…)                │
                │   agent.astream(stream_mode=[updates,messages,custom])│
                │   protocol/adapter.py: stream events → SSE            │
                ├───────────────────────────────────────────────────────┤
                │ Plugins (entry-point discovery)                       │
                │   auth │ models │ tools │ middleware │ subagents      │
                ├───────────────────────────────────────────────────────┤
                │ Stores                                                │
                │   HistoryStore   (SQLite/Postgres SQLAlchemy)         │
                │   MemoryStore    (SQLiteVec ANN + JSON metadata)      │
                │   WidgetDataStore(SQLAlchemy rows + SQLiteVec)        │
                └───────────────────────────────────────────────────────┘
```

## Wire protocol

The full request/response contract — every field, type, and artifact shape — is in [`reference/protocol/wire-contract.md`](../explanation/wire-contract.md). This section is the overview.

### `GET /agents.json`

Returns per-profile metadata:

```jsonc
{
  "openbb-agent": {
    "name": "OpenBB Agent",
    "description": "…",
    "endpoints": {"query": "/v1/query"},
    "features": {
      "streaming": true,
      "widget-dashboard-select": true,
      "widget-dashboard-search": true,
      "search-web": {"label": "Search Web", "description": "…", "default": false}
    }
  }
}
```

Feature flags map to `protocol/schemas.py` and are documented in the [Workspace AI-features reference](https://docs.openbb.co/workspace/developers/ai-features).

### `POST /v1/query`

Body is `QueryRequest` (`protocol/schemas.py`):

```jsonc
{
  "messages": [{"role": "human", "content": "..."} , …],
  "widgets": {"primary": [...], "secondary": [...], "extra": [...]},
  "uploaded_files": [{"name": "...", "mime": "...", "data_base64": "..."}],
  "api_keys": {"NVIDIA_API_KEY": "..."},
  "api_urls": {},
  "tools": [],
  "timezone": "America/New_York",
  "run_id": "uuidv7",
  "conversation_id": "uuid"
}
```

Response is `text/event-stream`. Each frame is one of:

| Event | Purpose |
| --- | --- |
| `MessageChunkSSE` | Token delta (`data.delta`). |
| `StatusUpdateSSE` | Reasoning step (`event_type ∈ {INFO,SUCCESS,WARNING,ERROR}`, `message`, `details`). |
| `FunctionCallSSE` | Client-side tool call — Workspace executes it, returns the result as a `role:"tool"` message on the next request. |
| `MessageArtifactSSE` | Inline artifact: table / chart / markdown / html / image / file. |
| `CitationCollectionSSE` | Citation chips, optionally with PDF bounding boxes. Emitted once at end-of-run. |

The response carries `X-Trace-ID` (the conversation id) and `X-Server-Trace-ID` (the per-exchange UUIDv7) headers. The body is the SSE event stream — there is no synthetic opening frame.

### Cancellation

A client disconnect mid-stream triggers the FastAPI handler's `asyncio.CancelledError`. `POST /v1/conversations/{id}/cancel` lets the same process cancel a run by setting an `asyncio.Event` keyed by `(user_id, run_id)` in `app/router.py::_cancellations`. Cancellation is in-process only — there is no cross-worker broadcast today.

## Request lifecycle

1. **Auth.** The router resolves a `UserPrincipal` via the configured `AuthBackend`. An anonymous request hits `none` or `bearer_static` (dev) or `oidc_jwt` / `api_key_table` (prod). Scopes (`agent:query`, `memory:read`, `memory:write`, …) drive endpoint authorization.
2. **Profile resolution.** The agent name (`/v1/query` → `default_profile`, `/agents/{name}/v1/query` → that name) is resolved to an `AgentProfile` — a frozen Pydantic model describing model + tools + middleware + sub-agents + features.
3. **Widget ingestion.** When a tool message carries `get_widget_data` results, `parse_widget_data_messages` extracts them and `WidgetDataStore.record` persists each row set; the rows are queried later with SQL via `query_widget_data`.
4. **Context ingestion.** `ingest_request_context` chunks any long uploaded file or human message (via `RecursiveCharacterTextSplitter`, language-aware for code) and writes the chunks to `MemoryStore` if the user has `memory:write` scope.
5. **Build the agent.** `runtime/builder.py::run_agent` calls `langchain.deepagents.create_deep_agent(model, tools, system_prompt, subagents, middleware)`. Every plugin is resolved from `runtime/registry.py` against the entry-point groups documented in [`docs/README.md`](../index.md).
6. **Stream.** `agent.astream(stream_mode=["updates","messages","custom"], subgraphs=True)` yields raw events. `protocol/adapter.py::DeepAgentEventAdapter` translates each into an OpenBB SSE event and the router writes it onto the wire.
7. **Persist.** As frames stream, the `UsageRecorder` middleware accumulates token usage, the `ToolCallLedger` middleware writes every server-side tool call to `tool_calls`, and the router persists the final assistant text to `messages`.
8. **Memory write-back.** Past-turn, the `MemoryWriter` middleware (when enabled + scope present) extracts durable facts from the assistant's reply and writes them to `memories`.

## Identity & trace ids

Four ids travel with every request:

| ID | Source | Lifetime | Purpose |
| --- | --- | --- | --- |
| `user_id` | resolved by `AuthBackend` | stable per user | partition key for every row |
| `trace_id` | `X-Trace-ID` header (or server-generated UUIDv7) | one HTTP exchange | join key across logs / usage / tool_calls |
| `run_id` | `QueryRequest.run_id` (or `<conversation>:turn<N>`) | one agent invocation | one `astream` call |
| `conversation_id` | `QueryRequest.conversation_id` | multi-turn thread | chat-history key |

`trace_id` and `conversation_id` show up in the response headers (`X-Server-Trace-ID`, `X-Trace-ID`) and in every log line via a contextvar filter.

## Plugin slots

See [the docs index plugin table](../index.md#plugin-slots) for the full list. Each slot is an ABC in `runtime/plugins.py`; the registry resolves names from entry points so third parties can ship their own without forking.

## Storage

| Surface | Default | Postgres |
| --- | --- | --- |
| Chat history / traces / usage / artifacts / citations / pending runs | SQLite at `~/.openbb_platform/agent/history.db` | `OPENBB_AGENT_DB_URL=postgresql+psycopg://…` |
| Vector memory + PDF page ANN index | `SQLiteVec` tables in the same SQLite file | — (SQLiteVec is SQLite-only) |
| Widget data rows | `widget_data` SQLAlchemy table | same |
| Background-job state | in-process `JobRegistry` | not persisted |
| Resume state after a client-side tool call | `pending_runs` SQLAlchemy table | same |

The `MemoryStore` ABC is principal-scoped — every method requires a `UserPrincipal` and stores reject mismatches before issuing SQL. See [Persistence](../operating/persistence.md) for schema details and [Memory and recall](memory-and-recall.md) for the recall flow.

## Where the code lives

```
openbb_agent_server/
├── app/         FastAPI app, router, settings.
├── runtime/     Builder, registry, ContextVars, JobRegistry, services.
├── protocol/    Pydantic wire models + SSE adapter.
├── memory/      Embeddings, retrievers, vector store, document loaders, splitter, writer.
├── persistence/ SQLAlchemy ORM + history store.
├── plugins/     Built-in auth / models / tools / middleware / sub-agents.
├── observability/ Structured logger + usage recorder.
└── prompts/     Packaged default system prompts.
```

Detailed module docs: [API reference](../reference/index.md).
