# OpenBB Agent Server Documentation

A pluggable, multi-tenant agent backend that speaks the [OpenBB Workspace](https://docs.openbb.co/workspace) custom-agent SSE protocol. The runtime is a thin wrapper over the LangChain `deepagents` harness: every model provider, tool source, sub-agent, middleware, auth backend, embeddings backend, vector store, and document loader is a swappable plugin.

## Where to start

- **First time?** Read [Getting Started](guides/getting-started.md), then [Architecture](guides/architecture.md).
- **Adding the agent to Workspace?** [Workspace integration](guides/workspace-integration.md).
- **Operating it in production?** [Configuration](operating/configuration.md) → [Auth](operating/auth.md) → [Persistence](operating/persistence.md).
- **Extending it with a custom plugin?** [Plugin system](developing/plugin-system.md), then the specific writing-a-* guide.
- **Looking up an exact function or class?** [API reference](reference/).

## Layout

| Folder | Audience | Purpose |
| --- | --- | --- |
| [`guides/`](guides/) | end users | Conceptual walk-throughs for running the agent through Workspace: architecture, widgets, memory, multimodal, background jobs, PyWry chat embedding. |
| [`developing/`](developing/) | plugin authors | Authoring tool sources, models, middleware, sub-agents, auth backends; conventions and the test harness. |
| [`operating/`](operating/) | operators / SREs | Configuration cascade, profiles, the five auth backends, the persistence schema, the memory pipeline, observability. |
| [`reference/`](reference/) | everyone | Module-by-module API reference that mirrors the package tree. |

## Plugin slots

The runtime resolves six plugin groups from Python entry points:

| Entry-point group | ABC | Built-ins | Docs |
| --- | --- | --- | --- |
| `openbb_agent_server.auth` | [`AuthBackend`](reference/runtime/plugins.md) | `none`, `bearer_static`, `api_key_table`, `oidc_jwt`, `openbb_workspace` | [Auth](operating/auth.md), [Writing an auth backend](developing/writing-an-auth-backend.md) |
| `openbb_agent_server.models` | [`ModelProvider`](reference/runtime/plugins.md) | `anthropic`, `openai`, `openai_compat`, `bedrock`, `vertex`, `google_genai`, `groq`, `nvidia`, `snowflake`, `fake` | [Writing a model provider](developing/writing-a-model-provider.md) |
| `openbb_agent_server.tools` | [`ToolSource`](reference/runtime/plugins.md) | `artifacts`, `web_search`, `fetch_url`, `widget_data`, `inspect_widget_data`, `pdf_extract`, `dashboard`, `recall_user_memory`, `translate`, `rerank`, `vision_qa`, `paligemma_vision`, `gemma_audio`, `gemini_image`, `gemini_embeddings`, `groq_audio`, `mcp_local`, `mcp_http`, `python_module`, `client_side`, `workspace_mcp`, `background_jobs`, `snowflake`, `pywry_canvas` | [Writing a tool source](developing/writing-a-tool-source.md) |
| `openbb_agent_server.middleware` | [`Middleware`](reference/runtime/plugins.md) | `call_limit`, `tool_call_limit`, `tool_call_announcer`, `tool_call_ledger`, `tool_filter`, `tool_message_normaliser`, `loop_guard`, `usage_recorder` | [Writing a middleware](developing/writing-a-middleware.md) |
| `openbb_agent_server.subagents` | [`SubAgentSpec`](reference/runtime/plugins.md) (Protocol) | `researcher`, `analyst`, `charter`, `pdf_reader` | [Writing a sub-agent](developing/writing-a-subagent.md) |
| `openbb_agent_server.checkpointers` | [`CheckpointerProvider`](reference/runtime/plugins.md) | `sqlite`, `postgres`, `inmemory` | — |

## Storage

| Surface | Default | Postgres |
| --- | --- | --- |
| Chat history, traces, usage, artifacts, citations, pending runs, widget data, PDF ingest, users, api keys | SQLite at `~/.openbb_platform/agent/history.db` | `OPENBB_AGENT_DB_URL=postgresql+psycopg://...` |
| Vector memory (`memories_text` / `memories_code`) + PDF page ANN (`pdf_pages_vec`) | `SQLiteVec` tables in the same SQLite file | SQLite-only (SQLiteVec is not available on Postgres) |
| Resume state after a client-side tool call | `pending_runs` SQLAlchemy table | same |
| Background-job state | in-process `JobRegistry` (run-scoped) | not persisted |

See [Persistence](operating/persistence.md) for the per-column schema and [Memory](operating/memory.md) for the vector pipeline.

## Wire protocol

`GET /agents.json` returns metadata. `POST /v1/query` returns a `text/event-stream` of `MessageChunkSSE`, `StatusUpdateSSE`, `FunctionCallSSE`, `MessageArtifactSSE`, and `CitationCollectionSSE` events. See [Architecture](guides/architecture.md#wire-protocol).

The same loop is also exposed without HTTP: [`EmbeddedRuntime`](reference/runtime/embedded.md) runs it in-process, and the [ACP shim](reference/acp/index.md) adapts it to PyWry's ACP chat component — see [PyWry chat (ACP)](guides/pywry-chat.md).
