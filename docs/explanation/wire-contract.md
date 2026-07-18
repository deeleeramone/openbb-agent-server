# Workspace wire contract

The exact request/response shapes exchanged between the OpenBB Workspace UI and this agent server. Every shape below is defined in [`protocol/schemas.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/protocol/schemas.py) (Pydantic models) — this page is the by-example contract; [`schemas.md`](../reference/protocol/schemas.md) is the field-level model reference.

All Pydantic models carry `ConfigDict(extra="allow")` unless noted, so Workspace may send fields not listed here without breaking validation — unknown fields land on `model_extra`.

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/agents.json` | none | Agent discovery — Workspace pulls this once when the agent URL is added. |
| `POST` | `/v1/query` | yes | Run the **default** profile. Returns an SSE stream. |
| `POST` | `/agents/{name}/v1/query` | yes | Run a named profile. Same body + stream. |
| `GET` | `/v1/me` | yes | Resolved principal (`user_id`, `display_name`, `email`, `scopes`). |
| `GET` | `/v1/conversations` | yes | List the caller's conversations. |
| `POST` | `/v1/conversations/{id}/cancel` | yes | Signal an in-flight run to stop. |

Auth depends on the configured backend (`none` / `bearer_static` / `api_key_table` / `oidc_jwt` / `openbb_workspace`); see [auth.md](../operating/auth.md). Every `/v1/query` response carries an `X-Server-Trace-ID` header.

---

## `GET /agents.json`

Returns a JSON object **keyed by agent id** (one entry per profile). Agent ids must match `^[a-z0-9-]+$`; non-matching profiles are dropped.

```json
{
  "default": {
    "name": "OpenBB · NVIDIA Stack",
    "description": "…",
    "image": "https://…",
    "endpoints": { "query": "/v1/query" },
    "features": {
      "streaming": true,
      "widget-dashboard-select": true,
      "search-web": {
        "label": "Search Web",
        "description": "Allow the agent to search the public web.",
        "default": false
      }
    }
  },
  "mistral-large-3": { "…": "…", "endpoints": { "query": "/agents/mistral-large-3/v1/query" } }
}
```

### Agent entry

| Field | Type | Req | Notes |
| --- | --- | --- | --- |
| `name` | `str` | yes | Display name in the Workspace agent picker. |
| `description` | `str` | yes | One-paragraph description. |
| `image` | `str` | no | Avatar URL. Emitted only when set. |
| `endpoints` | `{ "query": str }` | yes | Query path — `/v1/query` for the default profile, `/agents/{name}/v1/query` otherwise. |
| `features` | `dict[str, bool \| FeatureSpec]` | yes | Feature map (below). |

### `features` map

Two value shapes:

- **Reserved boolean capabilities** — a bare `bool`. Keys: `streaming`, `widget-dashboard-select`, `widget-dashboard-search`, `widget-global-search`, `file-upload`, `generative-ui`.
- **Custom user toggles** — a `FeatureSpec` object that Workspace renders as an opt-in switch in the chat-input settings menu:

| Field | Type | Req | Notes |
| --- | --- | --- | --- |
| `label` | `str` | yes | Switch label. |
| `description` | `str` | yes | Hover text. |
| `default` | `bool` | no | Initial state (default `false`). |

Built-in custom toggles: `search-web` (gates `web_search`), `fetch-url` (gates `fetch_url`). Each request's `workspace_options` carries the option's value keyed by its id — the feature is enabled only when that value is truthy.

---

## `POST /v1/query` — request

`Content-Type: application/json`. Body is `QueryRequest`.

```json
{
  "messages": [
    {"role": "human", "content": "What is AAPL trading at?"},
    {"role": "ai", "content": "{\"function\": \"get_widget_data\", \"input_arguments\": {…}}"},
    {"role": "tool", "tool_call_id": "call-1", "function": "get_widget_data", "data": [{"price": 192.3}]}
  ],
  "widgets": {"primary": [], "secondary": [], "extra": []},
  "uploaded_files": [],
  "api_keys": {}, "api_urls": {},
  "workspace_options": {"search-web": true, "fetch-url": false},
  "timezone": "America/New_York"
}
```

### `QueryRequest`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `messages` | `list[ChatMessage]` | required | Full multi-turn history, oldest first. |
| `widgets` | `WidgetsBag` | `{primary:[],secondary:[],extra:[]}` | Dashboard widget context. |
| `uploaded_files` | `list[UploadedFile]` | `[]` | Files attached to the chat input. |
| `api_keys` | `dict[str, Any]` | `{}` | Per-request provider keys forwarded by Workspace. |
| `api_urls` | `dict[str, Any]` | `{}` | Per-request provider base URLs. |
| `workspace_options` | `dict[str, Any]` | `{}` | Custom-feature option values, keyed by option id — `{"search-web": true, "model": "gpt-4o"}`. A toggle left off arrives as `false`. |
| `timezone` | `str \| None` | `None` | IANA tz of the user. |
| `context` | `list[dict] \| None` | `None` | Extra Workspace context blobs. |
| `urls` | `list[str] \| None` | `None` | URLs pasted into the chat input. |
| `force_web_search` | `bool \| None` | `None` | Workspace hint to force a web search. |
| `workspace_state` | `dict \| None` | `None` | Dashboard-state snapshot. |
| `tools` | `list[dict] \| None` | `None` | The user's enabled Workspace MCP tools. |

### `ChatMessage`

| Field | Type | Notes |
| --- | --- | --- |
| `role` | `"human" \| "ai" \| "tool"` | Required. |
| `content` | `str \| dict \| None` | Text, or a structured envelope dict on `ai` messages. |
| `tool_call_id` | `str \| None` | Pairs a `tool` message with the `ai` call that produced it. |
| `function` | `str \| None` | Tool name on `tool` messages. |
| `input_arguments` | `dict \| None` | Call arguments on `ai` tool-call messages. |
| `data` | `list \| None` | Result payload on `tool` messages (e.g. `get_widget_data` rows). |
| `agent_id` | `str \| None` | Originating agent for multi-agent threads. |

### `WidgetsBag` / `WidgetSpec` / `WidgetParam`

`WidgetsBag` has three `list[WidgetSpec]` lanes: `primary` (pinned to the active dashboard), `secondary` (other dashboards), `extra` (search results).

`WidgetSpec`: `uuid`, `widget_id`, `name`, `type`, `origin`, `description` (all `str \| None`); `params` (`list[WidgetParam] | dict`); `data` (`Any`). Lookup key is `uuid` when present else `widget_id`.

`WidgetParam`: `name` (`str`), `type` (`str | None`), `current_value` (`Any`).

### `UploadedFile`

| Field | Type | Notes |
| --- | --- | --- |
| `name` | `str` | Filename. |
| `mime` | `str \| None` | MIME type. |
| `data_base64` | `str \| None` | Base64 file bytes (inline upload). |
| `url` | `str \| None` | URL to fetch the file from (by-reference upload). |

---

## `POST /v1/query` — response

`Content-Type: text/event-stream`. Each event is one SSE frame:

```
event: <event-name>\n
data: <json>\n
\n
```

`<json>` is the event's `data` payload (the `data:` line only — the `event` field names the frame). Five event types:

### `copilotMessageChunk` — final-answer text

| `data` field | Type | Notes |
| --- | --- | --- |
| `delta` | `str` | A chunk of the chat-bubble answer. Concatenate all chunks for the full reply. |

### `copilotStatusUpdate` — reasoning step

| `data` field | Type | Req | Notes |
| --- | --- | --- | --- |
| `eventType` | `"INFO" \| "SUCCESS" \| "WARNING" \| "ERROR"` | yes | Severity. |
| `message` | `str` | yes | Reasoning-row text. |
| `group` | `"reasoning"` | yes | Always `"reasoning"`. |
| `details` | `list[dict \| str] \| None` | no | Structured sub-details. |
| `artifacts` | `list[ClientArtifact] \| None` | no | Artifacts attached to this row. |
| `hidden` | `bool` | no | Render collapsed when `true`. |

### `copilotFunctionCall` — client-side tool call

The server is asking Workspace to execute a tool **in the browser**. Workspace runs it and returns the result as a `role:"tool"` message in the *next* `/v1/query` request (see [round-trip](#client-side-tool-round-trip)).

| `data` field | Type | Req | Notes |
| --- | --- | --- | --- |
| `function` | enum | yes | One of `get_widget_data`, `get_extra_widget_data`, `get_params_options`, `add_widget_to_dashboard`, `add_generative_widget`, `update_widget_in_dashboard`, `assign_tasks_to_agents`, `execute_agent_tool`, `manage_navigation_bar`, `get_skill_content`. |
| `input_arguments` | `dict` | yes | Arguments for the function. Non-native tools are wrapped as `execute_agent_tool` with the real name nested inside. |
| `extra_state` | `dict \| None` | no | Opaque resume state echoed back by Workspace. |

### `copilotMessageArtifact` — inline artifact

`data` is a `ClientArtifact`:

| Field | Type | Req | Notes |
| --- | --- | --- | --- |
| `type` | `ArtifactType` | yes | See the artifact-type table below. |
| `name` | `str` | yes | Card title. |
| `description` | `str` | yes | Card subtitle. |
| `uuid` | `str` | yes | Stable id for the card (server-internal — do not echo it in prose). |
| `content` | `str \| list[dict]` | yes | Body — shape depends on `type`. |
| `chart_params` | `dict \| None` | no | Plotly figure JSON (chart artifacts only). |

#### Artifact types

| `type` | `content` | `chart_params` | Emitted by |
| --- | --- | --- | --- |
| `text` | Markdown string | — | `emit_markdown_artifact` (also the fallback for `markdown` and any unknown type) |
| `table` | `list[dict]` — one object per row, column name → cell value | — | `emit_table_artifact` |
| `chart` | `""` | Plotly figure JSON | `emit_chart_artifact` |
| `code` | Source code string | — | `emit_code_artifact` |
| `html` | Sanitised HTML string | — | `emit_html_artifact` |

`image` and `file` are **not** in the contract — markdown surfaces as `text`, other binary content is not emitted as an artifact.

### `copilotCitationCollection` — source attributions

Emitted once, at end of run. `data.citations` is a `list[Citation]`:

| `Citation` field | Type | Req | Notes |
| --- | --- | --- | --- |
| `id` | `str` | yes | Citation id. |
| `source_info` | `SourceInfo` | yes | Where it came from (below). |
| `details` | `list[dict] \| None` | no | Per-citation detail rows (e.g. `{text, url, title}`). |
| `quote_bounding_boxes` | `list[list[CitationHighlightBoundingBox]] \| None` | no | PDF highlight regions. Omitted from the wire payload when null. |

`SourceInfo`: `type` (`"widget" \| "direct retrieval" \| "web" \| "artifact"`), `uuid`, `origin`, `widget_id`, `name`, `description` (all `str | None`), `metadata` (`dict`), `citable` (`bool`, default `true`).

`CitationHighlightBoundingBox`: `text` (`str`), `page` (`int`), `x0` / `top` / `x1` / `bottom` (`float`) — pixel box on a PDF page.

---

## Emission & ordering contract

During a run, `copilotStatusUpdate` (reasoning) and `copilotFunctionCall` events stream as the agent works. At end of run the tail is drained in a fixed order:

1. **`copilotMessageChunk`** — the final chat-bubble answer flushes first.
2. **`copilotMessageArtifact`** — every artifact emitted during the run, in arrival order, so the cards stack *below* the answer.
3. **`copilotCitationCollection`** — one batch of every citation the final answer references.

This guarantees the chat bubble lands before its artifact cards and citation chips. See [`adapter.md`](adapter.md) for the translation internals.

## Client-side tool round-trip

Some tools execute in the Workspace browser, not on the server:

1. The server emits a `copilotFunctionCall` and **pauses** the run (resume state persisted in `pending_runs`).
2. Workspace executes the function and appends the result to the conversation.
3. Workspace sends a fresh `POST /v1/query` whose `messages` now end with a `role:"tool"` message carrying the result (`function`, `tool_call_id`, `data`).
4. The server resumes the paused run from the checkpoint and continues.

## See also

- [`schemas.md`](../reference/protocol/schemas.md) — field-level Pydantic model reference.
- [`adapter.md`](adapter.md) — how the DeepAgents event stream is translated into these frames.
- [`sse.md`](../reference/protocol/sse.md) — the SSE encoder.
- [Architecture: wire protocol](../guides/architecture.md#wire-protocol) — high-level overview.
