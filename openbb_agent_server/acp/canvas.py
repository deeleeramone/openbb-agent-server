"""The PyWry live canvas — the window's main content page.

``PyWryCanvas`` implements the
:class:`~openbb_agent_server.runtime.canvas.LiveCanvas` protocol over a
PyWry widget handle. It only needs the handle's ``emit(type, data)``
method (duck-typed), so this module imports nothing from pywry and is
unit-testable everywhere; the pywry-specific wiring lives in
:mod:`openbb_agent_server.acp.canvas_app`.

Cross-path rendering contract
-----------------------------

PyWry has three rendering paths — native window, anywidget/notebook,
and the inline browser iframe — and they do NOT share the same update
surface: ``eval_js`` and the ``pywry:set-content`` handler exist only
on the native path, and ``<script>`` tags inside content rendered via
``innerHTML`` are inert on the anywidget path. The one channel uniform
across all three is ``handle.emit(type, data)`` delivered to page
JavaScript through ``window.pywry.on(type, cb)``.

The canvas therefore mirrors the pattern PyWry's own ``ChatManager``
uses:

* All updates are plain events (``obb-canvas:*``) with JSON payloads —
  no ``eval_js`` anywhere.
* The page-side handlers live in :data:`CANVAS_BOOTSTRAP_JS`. It is
  embedded as a ``<script>`` tag by :func:`build_canvas_html` (executes
  on the native + inline paths, where initial content scripts run) and
  pushed through the anywidget ``_asset_js`` trait when the handle
  exposes ``set_trait`` (where initial-content scripts are inert). The
  bootstrap is idempotent, so double delivery is harmless.
* Plotly assets load lazily through an ``obb-canvas:load-assets`` event
  whose payload the page-side handler injects as a ``<script>`` element
  — DOM-appended scripts execute on every path.
* HTML, tables, and images render via ``innerHTML`` replacement inside
  the page handler, so ``<script>`` tags in agent-supplied markup do
  not execute on any path.
"""

from __future__ import annotations

import html as html_mod
import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("openbb_agent_server.acp.canvas")

CANVAS_ELEMENT_ID = "openbb-canvas"
# Chart + container id for the canvas's TVChart engine chart. All
# ``tvchart:*`` protocol events address it via ``chartId``.
TVCHART_CHART_ID = "openbb-canvas-tvchart"
_MAX_TABLE_ROWS = 2000

_EMPTY_STATE = (
    '<div class="obb-canvas-empty">'
    "<h2>Canvas</h2>"
    "<p>Ask the agent to chart, tabulate, or lay out anything here.</p>"
    "</div>"
)

# Page-side event handlers — the single rendering implementation every
# PyWry path shares. Registered against ``window.pywry.on`` with a
# short retry loop so it works whether the bridge or this script loads
# first. ``window.__obbCanvas`` is the idempotency marker: the script
# may arrive twice (page HTML + anywidget asset trait).
CANVAS_BOOTSTRAP_JS = """
(() => {
  if (window.__obbCanvas) { return; }
  const state = { ready: false };
  window.__obbCanvas = state;
  const byId = (id) => document.getElementById(id || 'openbb-canvas');
  const heading = (title) => {
    if (!title) { return null; }
    const h = document.createElement('h2');
    h.className = 'obb-canvas-title';
    h.textContent = title;
    return h;
  };
  const setHtml = (d) => {
    const el = byId(d.id);
    if (!el) { return; }
    el.innerHTML = d.html || '';
    // Wire any pywry toolbar markup (tooltips, icon buttons, dropdowns)
    // the same way the widget renderers do after innerHTML updates.
    if (typeof window.initToolbarHandlers === 'function') {
      try { window.initToolbarHandlers(el, window.pywry); } catch (e) {}
    }
  };
  const renderMarkdown = (d) => {
    const el = byId(d.id);
    if (!el) { return; }
    el.replaceChildren();
    const h = heading(d.title);
    if (h) { el.appendChild(h); }
    const box = document.createElement('div');
    box.className = 'obb-canvas-md';
    if (window.marked && typeof window.marked.parse === 'function') {
      box.innerHTML = window.marked.parse(d.text || '');
    } else {
      const pre = document.createElement('pre');
      pre.style.whiteSpace = 'pre-wrap';
      pre.textContent = d.text || '';
      box.appendChild(pre);
    }
    el.appendChild(box);
  };
  const renderPlotly = (d) => {
    const el = byId(d.id);
    if (!el) { return; }
    el.replaceChildren();
    const h = heading(d.title);
    if (h) { el.appendChild(h); }
    const plot = document.createElement('div');
    plot.style.cssText = 'width:100%;height:100%;min-height:420px;';
    el.appendChild(plot);
    const fig = d.figure || {};
    if (window.Plotly) {
      window.Plotly.newPlot(
        plot,
        fig.data || [],
        fig.layout || {},
        Object.assign({ responsive: true }, fig.config || {})
      );
    } else {
      plot.innerHTML = '<pre>Plotly.js is not loaded in this window.</pre>';
    }
  };
  const loadAssets = (d) => {
    (d.scripts || []).forEach((src) => {
      const s = document.createElement('script');
      s.textContent = src;
      document.head.appendChild(s);
    });
  };
  // Bridge the inline/browser ws-bridge gap. The tvchart engine's
  // client-side toolbar controls (chart type, drawing tools, indicators
  // panel, settings, undo/redo, screenshot, fullscreen, interval
  // dropdown, ...) register handlers via ``pywry.on`` and fire them by
  // calling ``pywry.emit``. The native bridge dispatches emit BOTH to
  // the host and to local handlers; the ws-bridge (browser iframe) only
  // sends to the server, so those controls would never reach their
  // local handlers and would appear dead. Mirror the native bridge:
  // when emit fires an event that has a local handler, also dispatch it
  // locally. Guarded to the ws-bridge shape (``_fire`` + ``_handlers``)
  // and applied once, so the native/anywidget paths are untouched. We
  // only local-dispatch when a handler is registered, so outbound-only
  // events (tvchart:data-request, notifications) are not queued in the
  // bridge's ``_pending`` buffer.
  const patchEmit = () => {
    const b = window.pywry;
    if (!b || b.__obbDualEmit) { return; }
    if (typeof b.emit !== 'function' || typeof b._fire !== 'function' || !b._handlers) {
      return;
    }
    const orig = b.emit.bind(b);
    b.emit = function(type, data) {
      orig(type, data);
      if ((b._handlers[type] || []).length > 0) {
        try { b._fire(type, data); } catch (e) {}
      }
    };
    b.__obbDualEmit = true;
  };
  const wire = () => {
    if (!window.pywry || typeof window.pywry.on !== 'function') { return false; }
    patchEmit();
    window.pywry.on('obb-canvas:set-html', setHtml);
    window.pywry.on('obb-canvas:markdown', renderMarkdown);
    window.pywry.on('obb-canvas:plotly', renderPlotly);
    window.pywry.on('obb-canvas:load-assets', loadAssets);
    state.ready = true;
    return true;
  };
  if (!wire()) {
    let tries = 0;
    const timer = setInterval(() => {
      tries += 1;
      if (wire() || tries > 100) { clearInterval(timer); }
    }, 50);
  }
})();
"""


def build_canvas_html(
    *,
    heading: str = "OpenBB Agent",
    subtitle: str = "",
) -> str:
    """Build the main content page the chat toolbar attaches to.

    A header strip, the ``#openbb-canvas`` container the agent draws
    into via the ``pywry_canvas`` tools, and the canvas bootstrap
    script (which executes on the native + inline paths; the anywidget
    path receives the same script through the ``_asset_js`` trait).

    Parameters
    ----------
    heading : str
        Bold title shown in the header strip; HTML-escaped before use.
    subtitle : str
        Optional dimmed text rendered beside the heading; HTML-escaped
        and omitted entirely when empty.

    Returns
    -------
    str
        A full HTML fragment: the canvas root with header, the
        ``#openbb-canvas`` main container seeded with the empty state,
        and the bootstrap ``<script>`` tag.
    """
    sub = (
        f'<span class="obb-canvas-subtitle">{html_mod.escape(subtitle)}</span>'
        if subtitle
        else ""
    )
    return f"""
<div class="obb-canvas-root">
  <header class="obb-canvas-header">
    <strong>{html_mod.escape(heading)}</strong>
    {sub}
  </header>
  <main id="{CANVAS_ELEMENT_ID}" class="obb-canvas-main">
    {_EMPTY_STATE}
  </main>
</div>
<script>{CANVAS_BOOTSTRAP_JS}</script>
"""


CANVAS_CSS = """
.obb-canvas-root { display: flex; flex-direction: column; height: 100vh; }
.obb-canvas-header {
  display: flex; align-items: baseline; gap: 12px;
  padding: 10px 16px; border-bottom: 1px solid rgba(128,128,128,0.25);
  flex: 0 0 auto;
}
.obb-canvas-subtitle { opacity: 0.65; font-size: 0.85em; }
.obb-canvas-main { flex: 1 1 auto; overflow: auto; padding: 16px; }
.obb-canvas-empty { opacity: 0.55; text-align: center; margin-top: 18vh; }
.obb-canvas-title { margin: 0 0 12px 0; }
.obb-canvas-table { border-collapse: collapse; width: 100%; }
.obb-canvas-table th, .obb-canvas-table td {
  border: 1px solid rgba(128,128,128,0.3);
  padding: 6px 10px; text-align: left; font-size: 0.9em;
}
.obb-canvas-table th { position: sticky; top: 0; }
.obb-canvas-img { max-width: 100%; height: auto; }
.obb-canvas-doc { width: 100%; height: 80vh; border: 0; }
.obb-canvas-audio, .obb-canvas-video { max-width: 100%; }
.obb-canvas-text {
  white-space: pre-wrap; word-break: break-word;
  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.9em;
}
.obb-canvas-download {
  display: inline-block; padding: 8px 14px; border-radius: 6px;
  border: 1px solid rgba(128,128,128,0.4); text-decoration: none;
  font-weight: 600; margin-bottom: 12px;
}
.obb-canvas-note { opacity: 0.65; font-size: 0.85em; }
.obb-canvas-tvchart-frame {
  position: relative; display: flex; flex-direction: column;
  min-height: 0; overflow: hidden;
}
.obb-canvas-tvchart-frame > * { flex: 1 1 auto; min-height: 0; }
"""


def _section(html: str, title: str | None) -> str:
    """Prefix content with an escaped heading when a title is given.

    Parameters
    ----------
    html : str
        The content fragment the heading is prepended to.
    title : str or None
        Section title; HTML-escaped and wrapped in an ``<h2>``. When
        falsy, ``html`` is returned unchanged.

    Returns
    -------
    str
        ``html`` optionally preceded by an ``<h2 class="obb-canvas-title">``
        heading.
    """
    if not title:
        return html
    return f'<h2 class="obb-canvas-title">{html_mod.escape(title)}</h2>{html}'


def render_table_html(
    rows: list[dict[str, Any]],
    *,
    columns: list[str] | None = None,
) -> str:
    """Render rows as an escaped HTML table (capped at ``_MAX_TABLE_ROWS``).

    Every header and cell value is HTML-escaped; ``dict``/``list`` cell
    values are JSON-encoded and ``None`` becomes an empty cell. When more
    rows are supplied than the cap allows, only the first
    ``_MAX_TABLE_ROWS`` are emitted and a note records the totals.

    Parameters
    ----------
    rows : list of dict
        Row records keyed by column name. When ``columns`` is omitted,
        the keys of the first row determine the column order.
    columns : list of str, optional
        Explicit column order. Cells absent from a row render empty.

    Returns
    -------
    str
        An ``<table class="obb-canvas-table">`` fragment, followed by a
        truncation note when rows were dropped.
    """
    cols = columns or (list(rows[0].keys()) if rows else [])
    shown = rows[:_MAX_TABLE_ROWS]
    head = "".join(f"<th>{html_mod.escape(str(c))}</th>" for c in cols)
    body_rows: list[str] = []
    for row in shown:
        cells = "".join(
            f"<td>{html_mod.escape(_cell_text(row.get(c)))}</td>" for c in cols
        )
        body_rows.append(f"<tr>{cells}</tr>")
    note = (
        f'<p class="obb-canvas-note">Showing {len(shown):,} of {len(rows):,} rows.</p>'
        if len(rows) > len(shown)
        else ""
    )
    return (
        f'<table class="obb-canvas-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table>{note}"
    )


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def _looks_like_datafeed_provider(obj: Any) -> bool:
    """Tell a provider INSTANCE from a zero-arg factory/class.

    A pywry ``DatafeedProvider`` instance exposes ``get_bars`` and is not
    a class; a factory is a plain callable (function or provider class)
    that must be invoked to produce the instance.
    """
    return hasattr(obj, "get_bars") and not isinstance(obj, type)


def _doc_kind_for_mime(mime: str | None) -> str:
    """Map a MIME type to the canvas render family used by ``show_document``.

    ``image`` / ``pdf`` / ``audio`` / ``video`` / ``text`` get a native
    HTML element; everything else (Office docs, archives, unknown binary)
    falls back to a ``download`` link.
    """
    m = (mime or "").split(";", 1)[0].strip().lower()
    if m.startswith("image/"):
        return "image"
    if m == "application/pdf":
        return "pdf"
    if m.startswith("audio/"):
        return "audio"
    if m.startswith("video/"):
        return "video"
    if m in ("text/plain", "application/json"):
        return "text"
    return "download"


def _embed_html(
    src: str,
    *,
    kind: str,
    filename: str | None = None,
    text: str | None = None,
) -> str:
    """Build the escaped HTML element for a ``show_document`` render.

    ``src`` is a ready-to-use URL or ``data:`` URI; it is escaped for the
    HTML attribute context. ``text`` (when given) is escaped for text
    context. ``<script>`` never appears, and PDFs use an ``<iframe>``
    (``data:`` in ``<embed>`` is blocked by some webviews) so the doc
    renders uniformly on every path.
    """
    esc = html_mod.escape(src, quote=True)
    if kind == "image":
        return f'<img class="obb-canvas-img" src="{esc}">'
    if kind == "pdf":
        name = html_mod.escape(filename or "document.pdf", quote=True)
        return f'<iframe class="obb-canvas-doc" src="{esc}" title="{name}"></iframe>'
    if kind == "audio":
        return f'<audio class="obb-canvas-audio" controls src="{esc}"></audio>'
    if kind == "video":
        return f'<video class="obb-canvas-video" controls src="{esc}"></video>'
    if kind == "text":
        return f'<pre class="obb-canvas-text">{html_mod.escape(text or "")}</pre>'
    # download fallback — a link plus any extracted text we were handed.
    name = html_mod.escape(filename or "file", quote=True)
    link = (
        f'<a class="obb-canvas-download" href="{esc}" download="{name}">'
        f"Download {name}</a>"
    )
    if text:
        link += f'<pre class="obb-canvas-text">{html_mod.escape(text)}</pre>'
    return link


class PyWryCanvas:
    """LiveCanvas over any PyWry widget handle.

    Every update is a plain ``handle.emit(event, data)`` — the one
    channel that behaves identically on PyWry's native-window,
    anywidget/notebook, and inline-iframe rendering paths. The
    page-side handlers come from :data:`CANVAS_BOOTSTRAP_JS`; on
    handles that expose ``set_trait`` (the anywidget path, where
    initial-content scripts are inert) the constructor delivers the
    bootstrap through the ``_asset_js`` trait as well.

    Parameters
    ----------
    handle : Any
        The widget handle returned by ``app.show()`` — anything with
        ``emit(event_type, data)``.
    element_id : str
        DOM id of the canvas container in the main page.
    plotly_assets : Callable[[], str] | None
        Returns the plotly.js source to load on first ``show_plotly``
        (sent via ``obb-canvas:load-assets`` and DOM-injected by the
        page handler). ``None`` skips loading — the figure still
        renders if the page already has ``window.Plotly``, and shows a
        notice otherwise.
    tvchart_assets : Callable[[], str] | None
        Same contract for first ``show_tvchart``: the lightweight-
        charts bundle PLUS pywry's tvchart engine modules (which
        self-register the ``tvchart:*`` protocol handlers) PLUS the
        toolbar handlers script.
    tvchart_chrome : Callable[[str, list[str] | None, str | None], str] | None
        ``(container_html, intervals, selected_interval) -> page_html``
        — wraps the chart container with pywry's TVChart toolbar set
        (``build_tvchart_toolbars`` + ``wrap_content_with_toolbars``).
        ``None`` renders the bare chart without chrome.
    tvchart_controller_factory : Callable[[Any, str], Any] | None
        ``(handle, chart_id) -> controller`` — binds pywry's
        ``TVChartStateMixin`` (the full protocol surface) to the canvas
        chart. Exposed via :meth:`tvchart_controller` for the
        ``canvas_tvchart_*`` tools.
    tvchart_datafeed_provider : Any | Callable[[], Any] | None
        A pywry ``DatafeedProvider`` instance — or a zero-arg factory
        returning one — that supplies symbol search, resolution, and
        historical/compare bars for ANY symbol. When present,
        :meth:`show_tvchart_symbol` mounts a datafeed-backed chart
        (``useDatafeed=True``) and the provider is wired to the
        controller via pywry's ``_wire_datafeed_provider``, so the
        header's Symbol Search and Compare controls work end to end
        (``symbol-search → resolve → data-request → data-response``).
        ``None`` leaves the canvas in static-data mode, where charts are
        mounted from caller-supplied bars and have no search/compare
        data source.
    """

    def __init__(
        self,
        handle: Any,
        *,
        element_id: str = CANVAS_ELEMENT_ID,
        plotly_assets: Callable[[], str] | None = None,
        tvchart_assets: Callable[[], str] | None = None,
        tvchart_chrome: Callable[[str, list[str] | None, str | None], str]
        | None = None,
        tvchart_controller_factory: Callable[[Any, str], Any] | None = None,
        tvchart_datafeed_provider: Any | Callable[[], Any] | None = None,
    ) -> None:
        """Bind the canvas to a PyWry widget handle.

        See the class docstring for the full parameter reference. The
        bootstrap is delivered immediately (and again via ``_asset_js``
        on handles that expose ``set_trait``).
        """
        self._handle = handle
        self._element_id = element_id
        self._plotly_assets = plotly_assets
        self._plotly_injected = False
        self._tvchart_assets = tvchart_assets
        self._tvchart_injected = False
        self._tvchart_chrome = tvchart_chrome
        self._tvchart_controller_factory = tvchart_controller_factory
        self._tvchart_controller: Any = None
        self._tvchart_datasets: dict[str, list[dict[str, Any]]] = {}
        self._tvchart_series_type = "Candlestick"
        self._tvchart_wired = False
        self._tvchart_datafeed_provider_spec = tvchart_datafeed_provider
        self._tvchart_datafeed_provider: Any = None
        self._tvchart_datafeed_resolved = False
        self._tvchart_datafeed_wired = False
        self._ensure_bootstrap()

    def tvchart_controller(self) -> Any:
        """Return the bound TVChart protocol controller, or ``None``.

        Available after the first :meth:`show_tvchart` when the host
        supplied a ``tvchart_controller_factory``. Carries pywry's
        entire ``TVChartStateMixin`` surface scoped to
        ``TVCHART_CHART_ID``.
        """
        return self._tvchart_controller

    def _ensure_bootstrap(self) -> None:
        """Deliver the page handlers on paths where page scripts are inert.

        The anywidget path renders content via ``innerHTML`` (scripts
        do not execute) but executes anything pushed through the
        ``_asset_js`` trait. Native / inline handles have no
        ``set_trait`` and already ran the bootstrap from the page HTML.

        Only ``PyWryChatWidget`` carries the ``_asset_js`` trait — the
        base ``PyWryWidget`` has no script-execution channel at all
        (verified live; pywry's own ``ChatManager`` asset injection has
        the same constraint), so notebook hosting must use the
        chat-paired widget.
        """
        set_trait = getattr(self._handle, "set_trait", None)
        if set_trait is None:
            return
        has_trait = getattr(self._handle, "has_trait", None)
        if callable(has_trait) and not has_trait("_asset_js"):
            logger.warning(
                "canvas: this anywidget handle has no _asset_js trait, so "
                "the canvas bootstrap cannot execute — plain PyWryWidget "
                "has no script channel. Host notebook canvases on "
                "PyWryChatWidget (the chat-paired widget) instead."
            )
            return
        try:
            set_trait("_asset_js", CANVAS_BOOTSTRAP_JS)
        except Exception:
            logger.warning(
                "canvas: bootstrap delivery via _asset_js failed", exc_info=True
            )

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        self._handle.emit(event, {"id": self._element_id, **data})

    def show_html(self, html: str, *, title: str | None = None) -> None:
        """Replace the canvas with an HTML fragment.

        The fragment is delivered verbatim to the page and assigned via
        ``innerHTML``, so any ``<script>`` tags it contains are inert.

        Parameters
        ----------
        html : str
            Raw HTML markup to drop into the canvas container.
        title : str or None, optional
            Optional section heading prepended above the fragment;
            HTML-escaped.
        """
        self._emit("obb-canvas:set-html", {"html": _section(html, title)})

    def show_markdown(self, text: str, *, title: str | None = None) -> None:
        """Render markdown via ``window.marked`` with a ``<pre>`` fallback.

        Parameters
        ----------
        text : str
            Markdown source. Parsed by ``window.marked`` when present;
            otherwise shown as preformatted text.
        title : str or None, optional
            Optional section heading rendered above the rendered markdown.
        """
        self._emit("obb-canvas:markdown", {"text": text, "title": title})

    def _ensure_plotly(self) -> None:
        if self._plotly_injected or self._plotly_assets is None:
            return
        try:
            bundle = self._plotly_assets()
        except Exception:
            logger.warning("canvas: plotly asset load failed", exc_info=True)
            return
        self._emit("obb-canvas:load-assets", {"scripts": [bundle]})
        self._plotly_injected = True

    def show_plotly(self, figure: dict[str, Any], *, title: str | None = None) -> None:
        """Render a Plotly figure into the canvas container.

        On first call the plotly.js bundle is injected (when the host
        supplied ``plotly_assets``). The figure is round-tripped through
        JSON with ``default=str`` so non-serializable leaves (dates,
        numpy scalars) cannot break the transport.

        Parameters
        ----------
        figure : dict
            A Plotly figure mapping; its ``data``, ``layout``, and
            optional ``config`` keys are forwarded to ``Plotly.newPlot``.
        title : str or None, optional
            Optional section heading rendered above the chart.
        """
        self._ensure_plotly()
        # Round-trip through JSON so non-serializable leaves (dates,
        # numpy scalars rendered via str) cannot break any path's
        # transport encoding.
        safe_figure = json.loads(json.dumps(figure, default=str))
        self._emit("obb-canvas:plotly", {"figure": safe_figure, "title": title})

    def show_table(
        self,
        rows: list[dict[str, Any]],
        *,
        title: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        """Replace the canvas with an escaped HTML table.

        Delegates to :func:`render_table_html`, so values are HTML-escaped
        and the row count is capped at ``_MAX_TABLE_ROWS``.

        Parameters
        ----------
        rows : list of dict
            Row records keyed by column name.
        title : str or None, optional
            Optional section heading rendered above the table.
        columns : list of str, optional
            Explicit column order; defaults to the first row's keys.
        """
        self._emit(
            "obb-canvas:set-html",
            {"html": _section(render_table_html(rows, columns=columns), title)},
        )

    def _ensure_tvchart(self) -> None:
        if self._tvchart_injected or self._tvchart_assets is None:
            return
        try:
            bundle = self._tvchart_assets()
        except Exception:
            logger.warning("canvas: tvchart asset load failed", exc_info=True)
            return
        self._emit("obb-canvas:load-assets", {"scripts": [bundle]})
        self._tvchart_injected = True

    def _resolve_datafeed_provider(self) -> Any:
        """Resolve the datafeed provider instance (calling a factory once)."""
        if self._tvchart_datafeed_resolved:
            return self._tvchart_datafeed_provider
        spec = self._tvchart_datafeed_provider_spec
        provider = spec
        if callable(spec) and not _looks_like_datafeed_provider(spec):
            # A zero-arg factory (provider instances expose get_bars etc.,
            # so the duck-type check tells a factory from an instance).
            try:
                provider = spec()
            except Exception:
                logger.warning(
                    "canvas: datafeed provider factory failed", exc_info=True
                )
                provider = None
        self._tvchart_datafeed_provider = provider
        self._tvchart_datafeed_resolved = True
        return provider

    def _ensure_tvchart_controller(self) -> Any:
        """Bind pywry's TVChart controller once (if a factory was given)."""
        if (
            self._tvchart_controller is None
            and self._tvchart_controller_factory is not None
        ):
            try:
                self._tvchart_controller = self._tvchart_controller_factory(
                    self._handle, TVCHART_CHART_ID
                )
            except Exception:
                logger.warning(
                    "canvas: tvchart controller binding failed", exc_info=True
                )
        return self._tvchart_controller

    def _ensure_datafeed_wired(self) -> bool:
        """Wire the datafeed provider to the controller once.

        Delegates to pywry's ``_wire_datafeed_provider`` (the documented
        auto-wiring), which registers the datafeed config/search/resolve/
        history handlers AND the ``tvchart:data-request`` handler that
        serves interval switches, symbol-search selections, and compare
        series — all through ``provider.get_bars``. Returns ``True`` when
        a provider is wired (so the static data-request handler is
        skipped to avoid double responses).
        """
        if self._tvchart_datafeed_wired:
            return True
        provider = self._resolve_datafeed_provider()
        if provider is None:
            return False
        controller = self._ensure_tvchart_controller()
        wire = getattr(controller, "_wire_datafeed_provider", None)
        if not callable(wire):
            logger.warning(
                "canvas: controller has no _wire_datafeed_provider — symbol "
                "search/compare will not be answered"
            )
            return False
        try:
            wire(provider)
        except Exception:
            logger.warning("canvas: datafeed provider wiring failed", exc_info=True)
            return False
        self._tvchart_datafeed_wired = True
        return True

    def _ensure_tvchart_wired(self) -> None:
        """Register the protocol's Python obligation once.

        Interval switches in the toolbar make the engine emit
        ``tvchart:data-request``; the host answers with
        ``tvchart:data-response`` (the exact contract
        ``examples/pywry_demo_tvchart.py`` implements).

        When a datafeed provider is configured, its pywry wiring already
        owns ``tvchart:data-request`` (serving every symbol/interval via
        ``get_bars``), so the static per-interval handler is skipped to
        avoid double responses.
        """
        if self._tvchart_wired:
            return
        if self._ensure_datafeed_wired():
            self._tvchart_wired = True
            return
        on = getattr(self._handle, "on", None)
        if not callable(on):
            logger.warning(
                "canvas: handle has no .on() — tvchart interval switching "
                "will not be answered"
            )
            self._tvchart_wired = True
            return
        on("tvchart:data-request", self._on_tvchart_data_request)
        self._tvchart_wired = True

    def _on_tvchart_data_request(self, data: Any, *_args: Any) -> None:
        """Serve the stashed per-interval dataset back to the engine."""
        try:
            payload = dict(data or {})
            if payload.get("chartId") not in (None, TVCHART_CHART_ID):
                return
            if payload.get("seriesId", "main") != "main":
                # Compare-series requests carry symbols we have no data
                # for; answering them with main bars would be wrong.
                return
            if not self._tvchart_datasets:
                return
            interval = str(payload.get("resolution") or payload.get("interval") or "")
            if interval not in self._tvchart_datasets:
                interval = next(iter(self._tvchart_datasets))
            self._handle.emit(
                "tvchart:data-response",
                {
                    "chartId": payload.get("chartId") or TVCHART_CHART_ID,
                    "seriesId": "main",
                    "bars": self._tvchart_datasets[interval],
                    "fitContent": True,
                    "interval": interval,
                },
            )
        except Exception:
            logger.warning("canvas: tvchart data-request failed", exc_info=True)

    def show_tvchart(
        self,
        datasets: dict[str, list[dict[str, Any]]],
        *,
        selected_interval: str | None = None,
        series_type: str = "Candlestick",
        title: str | None = None,
        chart_options: dict[str, Any] | None = None,
        height: str | None = None,
    ) -> None:
        """Mount a full PyWry TVChart — engine, toolbars, and data flow.

        This drives pywry's own TradingView chart protocol end to end,
        exactly like ``app.show_tvchart``:

        * the host's ``tvchart_assets`` supply lightweight-charts, the
          tvchart engine (which self-registers the ``tvchart:*``
          handlers), and the toolbar handlers;
        * the host's ``tvchart_chrome`` wraps the chart container with
          ``build_tvchart_toolbars`` chrome — header (symbol search,
          chart type, interval dropdown, indicators, settings, ...),
          drawing rail, time-range bar, and the OHLC legend overlay;
        * the chart mounts via the protocol's ``tvchart:create`` event;
        * interval switches round-trip through ``tvchart:data-request``
          → ``tvchart:data-response`` served from ``datasets``.

        ``datasets`` maps interval codes (``"1m"``, ``"1h"``, ``"1d"``,
        ``"1w"``, ...) to bar lists. Bars are
        ``{time, open, high, low, close, volume?}`` (volume embedded in
        the bars — the engine splits it into its own pane). Once
        created, every other ``tvchart:*`` protocol event
        (``tvchart:update``, ``tvchart:stream``, ``tvchart:add-series``,
        ``tvchart:add-markers``, ...) targets the chart via
        ``chartId=TVCHART_CHART_ID``.

        Parameters
        ----------
        datasets : dict of str to list of dict
            Interval code mapped to its bar list (see above). Must be
            non-empty.
        selected_interval : str or None, optional
            Interval to mount first; falls back to the first key in
            ``datasets`` when missing or not present.
        series_type : str, optional
            Main-series type passed to the engine (e.g. ``"Candlestick"``,
            ``"Line"``). Defaults to ``"Candlestick"``.
        title : str or None, optional
            Chart title shown in the chrome and section heading.
        chart_options : dict or None, optional
            Extra engine chart options forwarded as ``chartOptions``.
        height : str or None, optional
            CSS height for the chart frame; defaults to ``"560px"``.

        Raises
        ------
        ValueError
            If ``datasets`` is empty.
        """
        if not datasets:
            raise ValueError("datasets must contain at least one interval")
        sanitised: dict[str, list[dict[str, Any]]] = json.loads(
            json.dumps(datasets, default=str)
        )
        intervals = list(sanitised)
        selected = selected_interval if selected_interval in sanitised else intervals[0]
        self._tvchart_datasets = sanitised
        self._tvchart_series_type = series_type
        self._ensure_tvchart()
        self._ensure_tvchart_wired()
        self._ensure_tvchart_controller()
        self._mount_tvchart_chrome(intervals, selected, title=title, height=height)

        payload = {
            "containerId": TVCHART_CHART_ID,
            "chartId": TVCHART_CHART_ID,
            "chartOptions": chart_options or {},
            "title": title or "",
            "series": [
                {
                    "seriesId": "main",
                    "seriesType": series_type,
                    "bars": sanitised[selected],
                    "volume": [],
                    "seriesOptions": {},
                }
            ],
            "chartKind": "default",
            "useDatafeed": False,
            "interval": selected,
            "storage": {"backend": "localStorage"},
        }
        # Protocol event — handled by pywry's tvchart engine, not the
        # canvas bootstrap.
        self._handle.emit("tvchart:create", payload)

    def _mount_tvchart_chrome(
        self,
        intervals: list[str],
        selected: str,
        *,
        title: str | None,
        height: str | None,
    ) -> None:
        """Emit the chart container wrapped in pywry's toolbar chrome."""
        container = (
            f'<div id="{TVCHART_CHART_ID}" class="pywry-tvchart-container"></div>'
        )
        if self._tvchart_chrome is not None:
            try:
                body = self._tvchart_chrome(container, intervals, selected)
            except Exception:
                logger.warning(
                    "canvas: tvchart chrome build failed; rendering bare chart",
                    exc_info=True,
                )
                body = container
        else:
            body = container
        frame = (
            f'<div class="obb-canvas-tvchart-frame" '
            f'style="height:{html_mod.escape(height or "560px", quote=True)};">'
            f"{body}</div>"
        )
        self._emit("obb-canvas:set-html", {"html": _section(frame, title)})

    def show_tvchart_symbol(
        self,
        symbol: str,
        *,
        intervals: list[str] | None = None,
        selected_interval: str | None = None,
        series_type: str = "Candlestick",
        title: str | None = None,
        chart_options: dict[str, Any] | None = None,
        height: str | None = None,
    ) -> None:
        """Mount a DATAFEED-backed TVChart whose data comes from the provider.

        Unlike :meth:`show_tvchart` (which renders caller-supplied bars
        and has no search/compare data source), this mounts the chart in
        ``useDatafeed=True`` mode for ``symbol`` and wires the configured
        ``tvchart_datafeed_provider``. The provider then answers every
        datafeed event — config, symbol search, resolve, history, and the
        ``tvchart:data-request`` that interval switches, symbol-search
        selections, and Compare emit — so the header's Symbol Search and
        Compare controls work end to end against any symbol the provider
        knows. Mirrors ``pywry.PyWry.show_tvchart(provider=..., symbol=...,
        use_datafeed=True)``.

        Requires a ``tvchart_datafeed_provider``; raises ``RuntimeError``
        when none is configured (there is no data source to back search).

        Parameters
        ----------
        symbol : str
            Initial symbol to load; resolved through the datafeed
            provider. Required.
        intervals : list of str or None, optional
            Interval ladder offered in the toolbar; defaults to
            ``["1d", "1w", "1M"]``.
        selected_interval : str or None, optional
            Interval to mount first; falls back to the first ladder entry
            when missing or not present.
        series_type : str, optional
            Main-series type passed to the engine. Defaults to
            ``"Candlestick"``.
        title : str or None, optional
            Chart title; defaults to ``symbol``.
        chart_options : dict or None, optional
            Extra engine chart options forwarded as ``chartOptions``.
        height : str or None, optional
            CSS height for the chart frame; defaults to ``"560px"``.

        Raises
        ------
        ValueError
            If ``symbol`` is empty.
        RuntimeError
            If no ``tvchart_datafeed_provider`` is configured, since
            symbol search and compare would have no data source.
        """
        if not symbol:
            raise ValueError("symbol is required")
        self._ensure_tvchart()
        self._ensure_tvchart_controller()
        if not self._ensure_datafeed_wired():
            raise RuntimeError(
                "show_tvchart_symbol needs a tvchart_datafeed_provider; none is "
                "configured, so symbol search/compare have no data source. Use "
                "show_tvchart(datasets) for caller-supplied bars instead."
            )
        # The provider owns tvchart:data-request, so mark wiring done.
        self._tvchart_wired = True
        ladder = list(intervals) if intervals else ["1d", "1w", "1M"]
        selected = selected_interval if selected_interval in ladder else ladder[0]
        self._tvchart_series_type = series_type
        self._mount_tvchart_chrome(ladder, selected, title=title, height=height)

        payload = {
            "containerId": TVCHART_CHART_ID,
            "chartId": TVCHART_CHART_ID,
            "chartOptions": chart_options or {},
            "title": title or symbol,
            "series": [
                {
                    "seriesId": "main",
                    "symbol": symbol,
                    "resolution": selected,
                    "seriesType": series_type,
                    "seriesOptions": {},
                    "bars": [],
                    "volume": [],
                }
            ],
            "chartKind": "default",
            "useDatafeed": True,
            "interval": selected,
            "storage": {"backend": "localStorage"},
        }
        self._handle.emit("tvchart:create", payload)

    def show_image(self, src: str, *, title: str | None = None) -> None:
        """Replace the canvas with an image element.

        ``src`` must already be a renderable string — an ``https`` URL or
        a ``data:`` URI. The tool layer normalizes bytes / local paths /
        uploaded files to one of these before calling here.

        Parameters
        ----------
        src : str
            Renderable image source (``https`` URL or ``data:`` URI);
            escaped for the attribute context.
        title : str or None, optional
            Optional section heading rendered above the image.
        """
        img = f'<img class="obb-canvas-img" src="{html_mod.escape(src, quote=True)}">'
        self._emit("obb-canvas:set-html", {"html": _section(img, title)})

    def show_document(
        self,
        *,
        src: str,
        mime: str,
        filename: str | None = None,
        title: str | None = None,
        text: str | None = None,
    ) -> None:
        """Replace the canvas with a rendered document, dispatched by MIME.

        ``src`` is a ready ``https`` URL or ``data:`` URI (the tool layer
        normalizes bytes / paths / uploads). The MIME family selects the
        element: ``image/*`` → ``<img>``, ``application/pdf`` →
        ``<iframe>``, ``audio/*`` → ``<audio>``, ``video/*`` →
        ``<video>``, ``text/plain``/``application/json`` → ``<pre>``;
        anything else renders a download link (plus ``text`` when given as
        an extracted-content fallback). Like every renderer this is one
        ``obb-canvas:set-html`` emit, so it works on all three paths.

        Parameters
        ----------
        src : str
            Renderable document source (``https`` URL or ``data:`` URI);
            escaped for the attribute context.
        mime : str
            MIME type used to select the render element / family.
        filename : str or None, optional
            Display / download name for PDFs and the download fallback.
        title : str or None, optional
            Optional section heading rendered above the document.
        text : str or None, optional
            Extracted text shown inline for the text family and appended
            beneath the download link as a fallback.
        """
        kind = _doc_kind_for_mime(mime)
        fragment = _embed_html(src, kind=kind, filename=filename, text=text)
        self._emit("obb-canvas:set-html", {"html": _section(fragment, title)})

    def clear(self) -> None:
        """Reset the canvas to its empty state."""
        self._emit("obb-canvas:set-html", {"html": _EMPTY_STATE})
