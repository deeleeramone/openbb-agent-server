# `openbb_agent_server.acp.canvas`

The PyWry live canvas — the window's main content page.

`PyWryCanvas` implements the [`LiveCanvas`](../reference/runtime/canvas.md) protocol over any PyWry widget handle. It only needs the handle's `emit(type, data)` method (duck-typed), so this module imports nothing from pywry and is unit-testable everywhere; the pywry-specific wiring lives in [`canvas_app.md`](../reference/acp/canvas_app.md).

**Source:** [`openbb_agent_server/acp/canvas.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/acp/canvas.py)

## Cross-path rendering contract

PyWry has three rendering paths — native window, anywidget/notebook, and the inline browser iframe — and they do **not** share the same update surface: `eval_js` and the `pywry:set-content` handler exist only on the native path, and `<script>` tags inside content rendered via `innerHTML` are inert on the anywidget path. The one channel uniform across all three is `handle.emit(type, data)` delivered to page JavaScript through `window.pywry.on(type, cb)` — the same mechanism PyWry's own `ChatManager` relies on. The canvas therefore:

- sends every update as a plain `obb-canvas:*` event with a JSON payload — no `eval_js` anywhere;
- ships its page-side handlers in `CANVAS_BOOTSTRAP_JS`, embedded as a `<script>` tag by `build_canvas_html()` (executes on the native + inline paths) **and** pushed through the anywidget `_asset_js` trait when the handle exposes `set_trait` (where initial-content scripts are inert). The bootstrap is idempotent, so double delivery is harmless;
- loads Plotly and the TVChart stack lazily through an `obb-canvas:load-assets` event whose payload the page handler injects as a DOM-appended `<script>` element — which executes on every path;
- renders HTML / tables / images via `innerHTML` replacement inside the page handler (which also runs `initToolbarHandlers` over the new content when available), so `<script>` tags in agent-supplied markup do not execute on any path;
- JSON round-trips figure/bar payloads (`default=str`) so non-serializable leaves cannot break any path's transport encoding.

**TradingView charts are not rendered by the canvas bootstrap at all** — `show_tvchart` drives pywry's own TVChart engine protocol end to end, exactly like `app.show_tvchart`: the host's `tvchart_assets` supply lightweight-charts + the tvchart engine (which self-registers the `tvchart:*` handlers) + the toolbar handlers; the host's `tvchart_chrome` wraps the chart container with `build_tvchart_toolbars` chrome (header, drawing rail, time-range bar, OHLC legend); the chart mounts via `tvchart:create` (`chartId = TVCHART_CHART_ID`); and the canvas registers a `tvchart:data-request` handler on the widget so the toolbar's interval picker round-trips `tvchart:data-response` against the per-interval `datasets`. Every other `tvchart:*` protocol event (`tvchart:update`, `tvchart:stream`, `tvchart:add-series`, `tvchart:add-markers`, ...) addresses the same `chartId`.

| Operation | Native | Anywidget | Inline |
| --- | --- | --- | --- |
| `emit` → `window.pywry.on` | ✓ | ✓ | ✓ |
| Bootstrap via page `<script>` | ✓ | inert | ✓ |
| Bootstrap via `_asset_js` trait | n/a | ✓ (`PyWryChatWidget`) | n/a |

Note: only `PyWryChatWidget` carries the `_asset_js` trait — the base `PyWryWidget` has no script-execution channel at all (pywry's own `ChatManager` asset injection has the same constraint). Notebook canvas hosting therefore uses the chat-paired widget; binding a plain `PyWryWidget` degrades gracefully with a warning that names the supported class.

### TVChart toolbar interactivity (the dual-dispatch shim)

The TVChart chrome is fully interactive on every path: the chart-type menu, the indicators panel (including key-dispatched indicators like Moving Average and the Volume Profile entries), the left-rail drawing tools, the settings/undo/redo/screenshot/fullscreen buttons, the log-scale toggle, and the interval dropdown all act on the live chart. Those controls are wired by pywry's own engine — they register handlers with `window.pywry.on(...)` and trigger them by calling `window.pywry.emit('tvchart:...', ...)`.

The catch is that `emit` does not mean the same thing on every path. The **native** bridge dispatches an `emit` *both* to the host and to locally-registered `on` handlers, so a toolbar button's `emit` reaches the engine handler in the same page. The **inline** ws-bridge `emit`, by contrast, only sends to the server over the WebSocket — it never dispatches locally — so those same controls would fire into the void and appear dead.

`CANVAS_BOOTSTRAP_JS` closes this gap with a one-time `patchEmit` shim applied in `wire()`. It wraps `window.pywry.emit` so that, after the original send, it *also* locally fires (`_fire`) the event **when a handler is registered for that type**. It is:

- **shape-guarded** — it only patches when `emit`, `_fire`, and `_handlers` are all present (the ws-bridge shape), so the native and anywidget bridges are left untouched;
- **idempotent** — a `__obbDualEmit` flag means double bootstrap delivery patches once;
- **handler-gated** — local dispatch happens only when `_handlers[type]` is non-empty, so genuinely outbound-only events (`tvchart:data-request`, host notifications) are not echoed back into the page or queued in the bridge's `_pending` buffer.

This is verified live on both paths via real DOM clicks: on the native path the shim is inert (`__obbDualEmit` is set but `emit` already dispatched locally), and on the inline path it is the load-bearing reason every toolbar control and drawing tool reaches the engine.

## Classes

### `class PyWryCanvas`

#### `def __init__(handle, *, element_id=CANVAS_ELEMENT_ID, plotly_assets=None, tvchart_assets=None, tvchart_chrome=None, tvchart_controller_factory=None, tvchart_datafeed_provider=None)`

`handle` is the widget handle returned by `app.show()` — anything with `emit(event_type, data)` (plus `.on()` for TVChart interval answering). `plotly_assets` / `tvchart_assets` are zero-arg callables returning JS source to lazy-load; `tvchart_chrome` is `(container_html, intervals, selected) -> page_html` wrapping the chart with pywry's toolbar set; `tvchart_controller_factory` is `(handle, chart_id) -> controller` binding a pywry `TVChartStateMixin` instance scoped to `TVCHART_CHART_ID` (constructed on the first `show_tvchart`). `tvchart_datafeed_provider` is a pywry `DatafeedProvider` instance (or a zero-arg factory) that backs Symbol Search and Compare via `show_tvchart_symbol` (below); when present, the canvas wires it via `_wire_datafeed_provider` and skips its own static data-request handler. When the handle exposes `set_trait`, the constructor delivers the bootstrap through `_asset_js`.

#### `def tvchart_controller() -> Any`

The bound TVChart protocol controller, or `None`. Available after the first `show_tvchart` when the host supplied a `tvchart_controller_factory`; it carries pywry's entire `TVChartStateMixin` surface (`update_series`, `add_builtin_indicator`, `add_volume_profile`, `add_marker`, `add_price_line`, `set_visible_range`, `list_indicators`, ...) scoped to `TVCHART_CHART_ID`. The [`canvas_tvchart_*` agent tools](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/tools/pywry_canvas.py) drive the chart through this controller.

#### Renderers

`show_html` / `show_markdown` / `show_plotly` / `show_table` / `show_tvchart` / `show_tvchart_symbol` / `show_image` / `show_document` / `clear` — the `LiveCanvas` protocol. `show_tvchart(datasets, *, selected_interval=None, series_type="Candlestick", title=None, chart_options=None, height=None)` takes interval-keyed OHLCV bars. Titles render as a heading above the content (escaped server-side for HTML payloads, `textContent`-assigned in the page handler for markdown/plotly).

#### `def show_document(*, src, mime, filename=None, title=None, text=None)`

The one MIME-dispatching renderer for documents and media. `src` must already be renderable — an `https` URL or a `data:` URI (the [tool layer](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/tools/pywry_canvas.py) normalizes bytes / local paths / uploaded files to one of these before calling here). The MIME family picks the element, all built and escaped server-side then delivered as one `obb-canvas:set-html` emit (so it works on every path, with `<script>` inert):

| MIME family | Element |
| --- | --- |
| `image/*` | `<img>` (SVG goes here too — never `innerHTML` — so embedded scripts are neutered by the image context) |
| `application/pdf` | `<iframe>` with the `data:`/URL src (not `<embed>`, which some webviews block for `data:`). Inline PDF rendering depends on the webview having a built-in PDF viewer. |
| `audio/*` / `video/*` | `<audio controls>` / `<video controls>` |
| `text/plain`, `application/json` | escaped `<pre class="obb-canvas-text">` |
| anything else | a download-link anchor, optionally followed by an escaped `<pre>` of caller-supplied extracted `text` |

#### Image / document source normalization (tool layer)

`show_image` and `show_document` only ever receive a ready `src`. The normalization lives in the `canvas_image` / `canvas_document` tools, which accept (in resolution order) an uploaded-file `name` (looked up in `RunContext.current().uploaded_files`), agent-supplied `data_base64` (+ `mime`), or a `src` that is a `data:` URI / `https` URL (passed through) / local file path (read and encoded). Bytes are inlined as a `data:` URI via `_media.to_data_url`, capped at `_MAX_INLINE_BYTES` (8 MiB raw); larger media must be supplied as an `https` URL (passed through so the browser fetches it directly — the only PyWry path with a binary route is "let the browser do it"). `http://` and unresolvable inputs return a friendly `error:` string.

`canvas_document` additionally pre-converts rich text/data types to the existing renderers: CSV/TSV and tabular JSON → `show_table`; markdown → `show_markdown`; HTML → `show_html`; non-tabular JSON, plain text, and YAML → escaped text. Everything else embeds via `show_document`.

#### `def show_tvchart_symbol(symbol, *, intervals=None, selected_interval=None, series_type="Candlestick", title=None, chart_options=None, height=None)`

Mounts a **datafeed-backed** TVChart (`useDatafeed=True`) for `symbol` and wires the host's `tvchart_datafeed_provider` to the controller via pywry's `_wire_datafeed_provider`. The provider then answers every datafeed event — config, **symbol search**, resolve, history, and the `tvchart:data-request` that interval switches, search selections, and **Compare** emit — so the header's Symbol Search and Compare controls work end to end against any symbol the provider knows. Requires a provider (raises `RuntimeError` otherwise). Mirrors `pywry.PyWry.show_tvchart(provider=..., symbol=..., use_datafeed=True)`. Use `show_tvchart(datasets)` instead when you already hold the exact bars and don't need interactive search/compare.

## Functions

### `def build_canvas_html(*, heading="OpenBB Agent", subtitle="") -> str`

The main content page the chat toolbar attaches to: a header strip, the `#openbb-canvas` container with an empty-state hero, and the canvas bootstrap `<script>`. Pair with `CANVAS_CSS` (module constant) as the page's `inline_css`.

### `def render_table_html(rows, *, columns=None) -> str`

Escaped HTML table; explicit column order or first-row keys; capped at 2,000 rows with a "Showing N of M" note.

## Constants

| Name | Value | Purpose |
| --- | --- | --- |
| `CANVAS_ELEMENT_ID` | `"openbb-canvas"` | DOM id of the canvas container. |
| `TVCHART_CHART_ID` | `"openbb-canvas-tvchart"` | Chart + container id for the canvas's TVChart engine chart; all `tvchart:*` protocol events address it. |
| `CANVAS_BOOTSTRAP_JS` | script | Page-side `obb-canvas:*` event handlers — the single rendering implementation every path shares. |
| `CANVAS_CSS` | stylesheet | Layout + table/empty-state styling for the canvas page. |
