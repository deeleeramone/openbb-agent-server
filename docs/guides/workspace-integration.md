# Workspace integration

OpenBB Workspace can talk to any HTTP backend that implements its [custom-agent contract](https://docs.openbb.co/workspace/developers/ai-features). This server implements the full surface — streaming, reasoning steps, widget data, file uploads, tables, charts, citations, and client-side tool calls — so adding it is a single URL paste.

## Registering the agent

1. In Workspace, open **AI Agents → Add Agent**.
2. Paste `http://<host>:<port>/agents.json`. For local dev: `http://127.0.0.1:6900/agents.json`.
3. Save. Workspace fetches `/agents.json`, parses each profile entry, and presents one selectable agent per profile.

The server can host **multiple profiles** in one process. The default profile binds to `POST /v1/query`; additional profiles bind to `POST /agents/{name}/v1/query`. See [Profiles](../operating/profiles.md).

## Auth

`/agents.json` is unauthenticated. `/v1/query` and the management endpoints honor the configured auth backend:

| Backend | Header | Source of identity |
| --- | --- | --- |
| `none` | _none_ | hard-coded `anonymous` user (dev only) |
| `bearer_static` | `Authorization: Bearer <token>` | shared secret in `OPENBB_AGENT_AUTH_BEARER` |
| `api_key_table` | `X-API-KEY: <oba_…>` or `Authorization: Bearer <oba_…>` | hashed key looked up in `api_keys` table |
| `oidc_jwt` | `Authorization: Bearer <jwt>` | JWT verified against a JWKS URL |
| `openbb_workspace` | `X-OpenBB-User: <email>` | hashed user_id; email kept in-memory only |

Workspace sends whichever header the configured backend wants. See [Auth](../operating/auth.md).

## Features

`/agents.json` reports a `features` map for each profile. The Workspace UI honors the reserved keys; everything else is rendered as a per-profile toggle the user can flip in the chat panel.

Reserved keys:

| Key | Effect when `true` |
| --- | --- |
| `streaming` | Token-by-token streaming. Always `true` here. |
| `widget-dashboard-select` | Workspace passes the user's selected dashboard widgets in `widgets.primary`. |
| `widget-dashboard-search` | Workspace lets the agent search across all dashboard widgets. |
| `widget-global-search` | Workspace exposes global search across the user's widget catalogue. |
| `mcp-tools` | Workspace forwards the user's enabled MCP tool list in `tools[]`. |
| `file-upload` | Workspace allows file uploads on the request. |
| `generative-ui` | Workspace renders client-side generative UI from the agent's output. |

Custom features are user-visible toggles whose state shows up in `workspace_options`:

```toml
# openbb.toml — per-profile features
[features."deep-research"]
label = "Deep Research"
description = "Branch into the researcher sub-agent before drafting."
default = false
```

Inside a tool / middleware:

```python
from openbb_agent_server.runtime import context as run_context
ctx = run_context.current()
if ctx.has_workspace_option("deep-research"):
    ...
```

`workspace_options` is a `dict` of option values keyed by id — `has_workspace_option` returns `True` only when the value is truthy. Read a non-boolean option's value directly with `ctx.workspace_options.get("...")`.

The built-in `search-web` and `fetch-url` features follow this same shape — each is a per-user toggle that gates the `web_search` and `fetch_url` tool sources respectively; see [Features](../operating/configuration.md#features).

## MCP tools (workspace-mcp)

The `mcp-tools` feature lets Workspace forward its enabled MCP tools to the agent in `tools[]`. The most common one is [`workspace-mcp`](https://github.com/OpenBB-finance/workspace-mcp) — the browser-bridge MCP that exposes widget create/update/delete, dashboard / navigation / backend / app management, and Workspace state snapshots as MCP tools.

Two ways to run it:

1. **In-process (opt-in)** — install the `[workspace-mcp]` extra and explicitly set `mount_workspace_mcp = true` in `openbb.toml`. The agent server then downloads and mounts the Starlette app at `/mcp/workspace`. Point the Workspace UI's MCP-servers setting at:

   ```
   http://<host>:<port>/mcp/workspace/mcp
   ```

   The mount defaults to `false` so operators opt in deliberately. When the setting is `true` but the extra is not installed, the mount no-ops with an info log — useful for catching install drift.

2. **Separate sidecar (default)** — run `workspace-mcp` standalone (e.g. `uv tool install --python 3.13 https://github.com/OpenBB-finance/workspace-mcp/archive/refs/heads/main.zip` then `workspace-mcp`). Point the UI at `http://127.0.0.1:8787/mcp`. Leave `mount_workspace_mcp` at its default `false`.

Either way, the agent server itself is unchanged downstream — the [`workspace_mcp`](../reference/plugins/tools/workspace_mcp.md) tool source iterates `ctx.tools` and surfaces whatever the UI advertises. The SSE adapter wraps any non-enum tool call in `execute_agent_tool`, which the UI routes back to the workspace-mcp server.

## Widget data

When the user has widgets pinned on the dashboard and `widget-dashboard-select` is on, Workspace lists them in `widgets.primary`. The agent can:

1. **Auto-fetch** — when the user's last message is a question grounded in the pinned widgets and no tool result has been seen yet, the server emits a `FunctionCallSSE(function="get_widget_data", input_arguments={"data_sources": […]})` proactively. Workspace fetches the data and resumes the conversation with the result as a `role:"tool"` message.
2. **Inspect** — once data has been ingested into `WidgetDataStore`, the agent uses the `inspect_widget_data` tool source to `list / read / search / describe / query` rows with full SQL.

Full flow: [Widgets and data](widgets-and-data.md).

## File uploads

Workspace serialises every upload as a `FileRef` with `name`, `mime`, and either `data_base64` or `url`. The agent sees them through `ctx.uploaded_files` and the per-modality tool sources:

| Modality | Tools |
| --- | --- |
| PDF | `pdf_extract` (text + bounding boxes), `pdf_reader` sub-agent |
| Image | `vision_qa.understand_image`, `paligemma_vision.caption_image` / `read_image_text` / `ask_about_image`, `gemini_image.*` |
| Audio | `gemma_audio.transcribe_audio`, `groq_audio.transcribe_audio` |
| Spreadsheet | upload via Workspace → server decodes CSV/TSV via `langchain_community.document_loaders.CSVLoader` |

See [Multimodal tools](multimodal.md).

## Citations & PDF highlights

The `cite` helper in `runtime/emit.py` queues citations during the run; the protocol adapter dedupes them and emits one `CitationCollectionSSE` frame at end-of-run. PDF citations carry optional `quote_bounding_boxes` payloads — Workspace renders those as inline highlights on the source PDF.

## Sharing one conversation across devices

`conversation_id` is the persistence key. The same id from any device (with the same `user_id`) hits the same history rows. Workspace generates this id automatically; the server never overrides it.

## Stopping a run

The user clicks the stop button in Workspace → the SSE connection closes → the FastAPI handler observes `request.is_disconnected()` and stops yielding new frames. The agent loop drains in the background up to its natural completion (or until `bind()`'s `finally` cancels any outstanding background jobs).

For programmatic cancellation: `POST /v1/conversations/{id}/cancel`. Cancels any in-flight run for the caller on the same worker process.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Workspace lists the agent but every query 404s | `agent_id` in `agents.json` doesn't match `^[a-z0-9-]+$`; the router drops profiles that don't match the Workspace spec. |
| 401 / 403 on `/v1/query` | wrong auth backend or missing `agent:query` scope. |
| `widget-dashboard-select` toggle has no effect | confirm `features.widget-dashboard-select=true` in the profile and that the user has pinned widgets in the dashboard. |
