# Canvas

This page explains what the PyWry canvas is, how it works, and when to use each rendering path.

`PyWryCanvas` implements the [LiveCanvas](../reference/runtime/canvas.md) protocol and powers the window's main content area in the desktop experience.

- Source code: [openbb_agent_server/acp/canvas.py](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/acp/canvas.py)
- Reference API: [acp/canvas](../reference/acp/canvas.md)
- App wiring: [acp/canvas_app](../reference/acp/canvas_app.md)

## What It Does

The canvas renders agent output as rich content:

- HTML and markdown
- Tables
- Plotly charts
- TradingView charts
- Images and documents

The agent tools in [plugins/tools/pywry_canvas.py](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/tools/pywry_canvas.py) call this layer.

## How Updates Flow

All updates are sent through widget events, not eval-style script execution.

1. Server emits `obb-canvas:*` events with JSON payloads.
2. Page-side bootstrap handlers receive events and update the DOM.
3. Optional assets (Plotly/TVChart) are loaded lazily when needed.

This keeps behavior consistent across supported PyWry hosting modes.

## Hosting Modes

| Capability | Native | Anywidget | Inline |
| --- | --- | --- | --- |
| `emit` to page handlers | yes | yes | yes |
| Bootstrap from page `<script>` | yes | no | yes |
| Bootstrap via `_asset_js` trait | n/a | yes (chat widget) | n/a |

Important: notebook usage expects the chat-capable widget (`PyWryChatWidget`) for script asset delivery.

## TVChart: Two Modes

Use one of two chart modes depending on your data source.

### Static dataset mode

Call `show_tvchart(datasets, ...)` when you already have interval-keyed OHLCV data.

- Fast and deterministic
- No symbol search/compare datafeed required

### Datafeed mode

Call `show_tvchart_symbol(symbol, ...)` when you need interactive symbol workflows.

- Enables symbol search and compare
- Requires a `tvchart_datafeed_provider`

## Document and Media Rendering

`show_document(src, mime, ...)` chooses rendering by MIME type:

| MIME family | Rendered as |
| --- | --- |
| `image/*` | `<img>` |
| `application/pdf` | `<iframe>` |
| `audio/*`, `video/*` | media player elements |
| `text/plain`, `application/json` | escaped `<pre>` |
| other | download link (+ optional extracted text) |

Agent tools normalize inputs to either:

- `https` URL, or
- `data:` URI

Large payloads should use `https` URLs.

## Key Methods

- `show_html`, `show_markdown`, `show_table`, `show_plotly`
- `show_tvchart`, `show_tvchart_symbol`
- `show_image`, `show_document`
- `clear`

For signatures and parameters, see [acp/canvas](../reference/acp/canvas.md).

## Troubleshooting

- Canvas not updating:
	- confirm widget handle supports `emit(type, data)`.
- Notebook canvas issues:
	- use `PyWryChatWidget`, not a plain base widget.
- TVChart symbol mode not working:
	- ensure `tvchart_datafeed_provider` is configured.
- Large media fails inline:
	- use an `https` URL instead of embedding bytes.
