---
slug: /
title: OpenBB Agent Server
sidebar_label: Home
---

# OpenBB Agent Server

A pluggable, multi-tenant agent backend that speaks the [OpenBB Workspace](https://docs.openbb.co/workspace) custom-agent SSE protocol. The runtime is a thin wrapper over the LangChain `deepagents` harness: every model provider, tool source, sub-agent, middleware, auth backend, embeddings backend, vector store, and document loader is a swappable plugin.

## Where to start

- **New here?** [Install it](installation.md), then run your first query in the [Quick Start](quick-start.md).
- **Using the agent?** The [User Guides](guides/index.md) cover architecture, Workspace integration, widgets, memory, multimodal, and the PyWry desktop canvas.
- **Operating it?** [Advanced Configuration](operating/index.md) covers the settings cascade, profiles, auth, persistence, memory, and observability.
- **Extending it?** [Developing Plugins](developing/index.md) walks through authoring a tool source, model provider, middleware, sub-agent, or auth backend.
- **Looking up a symbol?** The [API Reference](reference/index.md) is generated from the package docstrings.

## Plugin slots

The runtime resolves six plugin groups from Python entry points. Each ABC lives in [`runtime.plugins`](reference/runtime/plugins.md).

| Entry-point group | ABC | Built-ins |
| --- | --- | --- |
| `openbb_agent_server.auth` | `AuthBackend` | `none`, `bearer_static`, `api_key_table`, `oidc_jwt`, `openbb_workspace` |
| `openbb_agent_server.model_providers` | `ModelProvider` | `anthropic`, `openai`, `openai_compat`, `bedrock`, `vertex`, `google_genai`, `groq`, `nvidia`, `fake` |
| `openbb_agent_server.tools` | `ToolSource` | `artifacts`, `web_search`, `fetch_url`, `widget_data`, `pdf_extract`, `dashboard`, `vision_qa`, `gemini_image`, `mcp_local`, `mcp_http`, `pywry_canvas`, … |
| `openbb_agent_server.middleware` | `Middleware` | `call_limit`, `tool_call_limit`, `tool_call_announcer`, `tool_call_ledger`, `tool_filter`, `tool_message_normaliser`, `loop_guard`, `usage_recorder` |
| `openbb_agent_server.subagents` | `SubAgentSpec` | `researcher`, `analyst`, `charter`, `pdf_reader`, plus model-profile subagents (`deepseek-v4-flash`, `nemotron-3-super`, `qwen3.5`, `minimax-m3`, …) |
| `openbb_agent_server.checkpointers` | `CheckpointerProvider` | `sqlite`, `postgres`, `inmemory` |

See [Developing Plugins](developing/index.md) to add your own.

## Storage

| Surface | Default | Postgres |
| --- | --- | --- |
| Chat history, traces, usage, artifacts, citations, pending runs, widget data, PDF ingest, users, api keys | SQLite at `~/.openbb_platform/agent/history.db` | `OPENBB_AGENT_DB_URL=postgresql+psycopg://...` |
| Vector memory + PDF page ANN | `SQLiteVec` tables in the same SQLite file | SQLite-only (SQLiteVec is not available on Postgres) |
| Resume state after a client-side tool call | `pending_runs` SQLAlchemy table | same |
| Background-job state | in-process `JobRegistry` (run-scoped) | not persisted |

See [Persistence](operating/persistence.md) for the schema and [Memory](operating/memory.md) for the vector pipeline.

## Wire protocol

`GET /agents.json` returns metadata. `POST /v1/query` returns a `text/event-stream` of `MessageChunkSSE`, `StatusUpdateSSE`, `FunctionCallSSE`, `MessageArtifactSSE`, and `CitationCollectionSSE` events — the full field-by-field contract is in the [Wire-protocol contract](explanation/wire-contract.md).

The same loop is also exposed without HTTP: [`EmbeddedRuntime`](reference/runtime/embedded.md) runs it in-process, and the [ACP shim](reference/acp/index.md) adapts it to PyWry's chat component — see [PyWry chat (ACP)](guides/pywry-chat.md).
