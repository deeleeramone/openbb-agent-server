# PyWry chat (ACP)

Embed the agent in a desktop window instead of (or alongside) the HTTP server. The ACP shim exposes the same agent loop, tool sources, middleware, and profiles to [PyWry](https://deeleeramone.github.io/PyWry/)'s ACP-native chat component — one `openbb.toml` drives both deployments.

## Install

```bash
pip install 'openbb-agent-server[pywry]'
```

`pywry` is deliberately not part of the `[all]` extra — it is a desktop GUI toolkit that server deployments don't want.

## The canvas app

The fastest path is the bundled desktop app — a window whose main content page is the agent's **live canvas**, with the chat attached as a side panel:

```bash
openbb-agent-canvas                        # layered openbb.toml cascade
openbb-agent-canvas --config-file /etc/openbb/agent.toml --profile default
```

`launch()` is the programmatic equivalent (`block=False` returns the wired `CanvasApp` for embedding in a larger PyWry application):

```python
from openbb_agent_server.acp import launch

canvas_app = launch(profile="default", block=False)
# canvas_app.app / .widget / .chat / .provider / .canvas
canvas_app.app.block()
```

The launcher appends the `pywry_canvas` tool source to the configured tools, so the agent can drive the page while the conversation runs in the chat panel:

| Tool | Renders |
| --- | --- |
| `canvas_html` | An HTML fragment (scripts do not execute). |
| `canvas_markdown` | Markdown (via `window.marked`, `<pre>` fallback). |
| `canvas_plotly` | An interactive Plotly figure — pywry's bundled plotly.js is injected on first use. |
| `canvas_table` | An escaped HTML table (capped at 2,000 rows). |
| `canvas_tvchart` | A **full PyWry TVChart** — engine chart with complete toolbar chrome (interval picker over the provided per-interval datasets, chart-type menu, client-side indicators, drawing tools, time-range tabs, OHLC legend, volume pane). Driven entirely through pywry's `tvchart:*` protocol; assets inject on first use. |
| `canvas_tvchart_*` | The **entire TVChart protocol** against the mounted chart: `update` / `stream_bar` (live data), `add_series` / `remove_series` (overlays & compares), `add_indicator` / `remove_indicator` / `list_indicators` (the engine's full built-in catalog — MAs, RSI, MACD, Bollinger, Ichimoku, ADX, ...), `add_volume_profile`, `add_markers`, `add_price_line`, `time_scale`, `apply_options`, `set_chart_type`, `get_state`, `destroy`. Backed by pywry's own `TVChartStateMixin` bound to the chart. |
| `canvas_image` | An image from an https URL or data URI. |
| `canvas_clear` | Back to the empty state. |

Each render **replaces** the canvas; the tools are only registered when a canvas host is bound, so the same TOML stays valid for the HTTP server (where they simply don't appear). To use the canvas from your own PyWry app instead of the launcher, put `build_canvas_html()` (or at least its `#openbb-canvas` container + bootstrap script) in your page and bind it:

```python
from openbb_agent_server.acp.canvas import PyWryCanvas
from openbb_agent_server.runtime import canvas as canvas_registry

canvas_registry.set_canvas(PyWryCanvas(widget))   # widget = app.show(...)
```

The canvas works on **all three PyWry rendering paths** — native window, anywidget/notebook, and the inline browser iframe. Every update travels as an `obb-canvas:*` event over `widget.emit(...)` (the one channel all paths share; no `eval_js`), and the page-side handlers ship in the canvas HTML's bootstrap script, which is additionally delivered through the anywidget `_asset_js` trait where initial-content scripts are inert. See [`reference/acp/canvas.md`](../explanation/canvas.md) for the support matrix.

## Attach a chat to any PyWry widget

```python
from pywry import PyWry
from pywry.models import HtmlContent

from openbb_agent_server.acp import create_chat_manager

app = PyWry(title="My App")

# Resolves the layered openbb.toml cascade — the same files the
# `openbb-agent-server` CLI reads. Pass explicit_path to pin one file.
chat = create_chat_manager(profile="default")

widget = app.show(
    HtmlContent(html="<h1>Dashboard</h1>"),
    toolbars=[chat.toolbar(position="right")],
    callbacks=chat.callbacks(),
)
chat.bind(widget)
app.block()
```

`create_chat_manager` accepts the usual `ChatManager` keyword arguments (`welcome_message`, `show_sidebar`, `toolbar_width`, ...) and forwards them.

## Or wire the provider yourself

```python
from pywry.chat import ChatManager

from openbb_agent_server.acp import OpenBBAgentProvider

provider = OpenBBAgentProvider.from_toml(
    "/etc/openbb/agent.toml",   # optional; cascade discovery otherwise
    profile="default",
    user_id="desk-1",
)
chat = ChatManager(provider=provider)
```

The provider implements PyWry's ACP `ChatProvider` lifecycle — `initialize`, `new_session`, `prompt`, `cancel`, `set_mode` — and streams typed `SessionUpdate` notifications (message deltas, thinking, tool status, artifacts, citations).

## What transfers from the TOML

| `[agent]` setting | Effect in the PyWry chat |
| --- | --- |
| `model` / `model_provider` / `model_name` | Same model plugin drives the loop. |
| `tool_sources` + `tool_source_config` | Same tools, including MCP tool sources (`mcp_local`, `mcp_http`). |
| `middleware` | Same middleware stack (usage recorder, loop guard, ...). |
| `subagents`, `skills` | Same sub-agent and skill wiring. |
| `profiles` | Surface as ACP **modes** — the chat's mode picker switches profiles per session. |
| `checkpointer_provider`, `db_url`, `data_dir` | Same persistence: history, memory, traces land in the same stores. |
| `metadata.description` | Seeds the chat welcome message. |

Things that do **not** apply outside OpenBB Workspace:

- `auth_backend` — the embedded chat is single-user; a local principal (`user_id="pywry-local"` by default) scopes history and memory.
- Client-side Workspace tools (`widget_data`, `client_side`, `dashboard`) — these dispatch `copilotFunctionCall` events that only the Workspace UI can execute. Leave them out of the profile you embed, or the agent announces the limitation and ends the turn.
- The retention prune sweep — embedding hosts manage their own retention.

## How it maps

```
PyWry chat UI ⇄ ChatManager ⇄ OpenBBAgentProvider (ACP)
                                   │ translate_sse()
                                   ▼
                          EmbeddedRuntime.run_turn()
                                   │
                                   ▼
                  run_agent() — the same loop /v1/query drives
```

See [`reference/acp/provider.md`](../reference/acp/provider.md) for the full SSE → `SessionUpdate` translation table and [`reference/runtime/embedded.md`](../reference/runtime/embedded.md) for the runtime underneath.
