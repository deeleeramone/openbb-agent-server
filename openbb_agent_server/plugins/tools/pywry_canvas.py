"""Agent tools that draw on the host's live canvas.

The *canvas* is the main content page of an embedding host — in the
PyWry desktop app (``openbb_agent_server.acp.canvas_app``) it is the
window the chat toolbar is attached to. These tools let the agent use
that page as a live display surface: render HTML, markdown, Plotly
figures, tables, and images for the user to look at while the
conversation continues in the chat panel.

The tool source itself is host-agnostic — it talks to the
:class:`~openbb_agent_server.runtime.canvas.LiveCanvas` protocol bound
in ``runtime.canvas``. When no host has bound a canvas (e.g. the plain
HTTP server) every tool returns an explanatory error string instead of
raising, so a misconfigured profile degrades gracefully.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.runtime import (
    canvas as canvas_registry,
    context as run_context,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.pywry_canvas")

_NO_CANVAS = (
    "error: no live canvas is bound in this deployment — the canvas tools "
    "only work when the agent runs inside a canvas host such as the PyWry "
    "desktop app. Use the chat artifact tools instead."
)

# Raw-byte cap for inlining a binary as a data: URI. The only delivery
# channel uniform across all PyWry paths is the emit() JSON event, so
# binaries become ``data:`` URIs; this keeps them under the WebSocket
# frame ceiling (~16 MiB after base64's +33%). Larger media must be
# supplied as an https URL (passed through, fetched by the browser).
_MAX_INLINE_BYTES = 8 * 1024 * 1024

# Text/data MIME types the document router converts to a RICH canvas
# view (table / markdown / html / text) instead of embedding verbatim.
_TEXT_TABLE_MIMES = ("text/csv", "text/tab-separated-values")
_TEXT_FAMILY_MIMES = (
    *_TEXT_TABLE_MIMES,
    "application/json",
    "application/yaml",
    "text/markdown",
    "text/html",
    "text/plain",
)


def _canvas() -> Any:
    return canvas_registry.get_canvas()


def _data_uri_bytes(uri: str) -> bytes:
    """Decode the base64 payload of a ``data:...;base64,...`` URI."""
    _, _, payload = uri.partition(",")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"malformed data: URI ({exc})") from exc


def _data_uri_mime(uri: str) -> str | None:
    """Pull the MIME type out of a ``data:<mime>[;...],...`` URI."""
    head = uri[5:].split(",", 1)[0]
    mime = head.split(";", 1)[0].strip()
    return mime or None


async def _resolve_media(
    src: str | None,
    data_base64: str | None,
    mime: str | None,
    name: str | None,
    *,
    default_mime: str,
) -> tuple[bytes | None, str | None, str]:
    """Resolve a media source to ``(raw_bytes, https_url, mime)``.

    Exactly one of ``raw_bytes`` / ``https_url`` is set: an https URL is
    passed through (the browser fetches it, sidestepping the inline
    frame-size ceiling); everything else (uploaded file, agent base64,
    ``data:`` URI, local path) resolves to raw bytes. Resolution order is
    uploaded-name → ``data_base64`` → ``src``. Raises ``ValueError`` (or
    a ``_media.MediaError``) with a friendly message on bad input.
    """
    from openbb_agent_server.plugins.tools._media import fetch_url

    if name:
        try:
            ctx = run_context.current()
        except Exception as exc:  # noqa: BLE001 — no active run context
            raise ValueError("name lookup needs an active run context") from exc
        target = next((f for f in ctx.uploaded_files if f.name == name), None)
        if target is None:
            raise ValueError(f"{name!r} is not among this run's uploaded files")
        fmime = (
            target.mime or mimetypes.guess_type(target.name or "")[0] or default_mime
        )
        if target.data_base64:
            b64 = target.data_base64
            raw = _data_uri_bytes(b64) if b64.startswith("data:") else _b64_bytes(b64)
            return raw, None, fmime
        if target.url:
            if target.url.startswith("https://"):
                return None, target.url, fmime
            fetched = await fetch_url(
                target.url, max_bytes=_MAX_INLINE_BYTES, fallback_mime=fmime
            )
            return fetched.data, None, fetched.mime
        raise ValueError(f"uploaded file {name!r} has no data_base64 or url")

    if data_base64:
        m = mime or default_mime
        if data_base64.startswith("data:"):
            return (
                _data_uri_bytes(data_base64),
                None,
                (_data_uri_mime(data_base64) or m),
            )
        return _b64_bytes(data_base64), None, m

    if src:
        if src.startswith("data:"):
            return (
                _data_uri_bytes(src),
                None,
                (_data_uri_mime(src) or mime or default_mime),
            )
        if src.startswith("https://"):
            return None, src, (mime or mimetypes.guess_type(src)[0] or default_mime)
        if src.startswith("http://"):
            raise ValueError("http:// is not allowed; use https:// or a data: URI")
        path = Path(src)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            raw = path.read_bytes()
            return raw, None, (mime or mimetypes.guess_type(src)[0] or default_mime)
        raise ValueError(
            f"{src!r} is not a data: URI, an https URL, or an existing file path"
        )

    raise ValueError(
        "provide one of: src, data_base64 (+mime), or name (uploaded file)"
    )


def _b64_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 data ({exc})") from exc


async def _renderable_src(raw: bytes | None, url: str | None, mime: str) -> str:
    """Return a renderable src — the https URL, or a data: URI of the bytes."""
    if url is not None:
        return url
    from openbb_agent_server.plugins.tools._media import to_data_url

    return await to_data_url(raw or b"", mime=mime, max_bytes=_MAX_INLINE_BYTES)


async def _resolved_text(raw: bytes | None, url: str | None) -> str:
    """Decode resolved media to text (fetching an https URL if needed)."""
    if raw is None:
        from openbb_agent_server.plugins.tools._media import fetch_url

        fetched = await fetch_url(url or "", max_bytes=_MAX_INLINE_BYTES)
        raw = fetched.data
    return raw.decode("utf-8", errors="replace")


class _HtmlArgs(BaseModel):
    """Argument schema for :func:`canvas_html`."""

    html: str = Field(
        description=(
            "HTML fragment to render. It becomes the canvas's content — "
            "use complete, self-contained markup (inline styles allowed; "
            "<script> tags will NOT execute)."
        )
    )
    title: str | None = Field(
        default=None,
        description="Optional heading shown above the content.",
    )


class _MarkdownArgs(BaseModel):
    """Argument schema for :func:`canvas_markdown`."""

    text: str = Field(description="Markdown source to render on the canvas.")
    title: str | None = Field(
        default=None,
        description="Optional heading shown above the content.",
    )


class _PlotlyArgs(BaseModel):
    """Argument schema for :func:`canvas_plotly`."""

    figure: dict[str, Any] = Field(
        description=(
            "A Plotly figure as JSON: {'data': [trace, ...], 'layout': "
            "{...}}. Standard plotly.js trace types (scatter, bar, "
            "candlestick, heatmap, pie, ...) are supported."
        )
    )
    title: str | None = Field(
        default=None,
        description="Optional heading shown above the chart.",
    )


class _TableArgs(BaseModel):
    """Argument schema for :func:`canvas_table`."""

    rows: list[dict[str, Any]] = Field(
        description="Table rows as a list of objects keyed by column name."
    )
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Optional explicit column order. Defaults to the keys of the first row."
        ),
    )
    title: str | None = Field(
        default=None,
        description="Optional heading shown above the table.",
    )


# pywry TVChart engine series types (payload key ``seriesType``).
_TV_SERIES_TYPES = {
    "candlestick": "Candlestick",
    "line": "Line",
    "area": "Area",
    "bar": "Bar",
    "baseline": "Baseline",
    "histogram": "Histogram",
}


class _TvChartArgs(BaseModel):
    """Argument schema for :func:`canvas_tvchart`."""

    datasets: dict[str, list[dict[str, Any]]] = Field(
        description=(
            "OHLCV bars keyed by interval code — e.g. {'1d': [...], "
            "'1w': [...], '1M': [...]}. Each bar is {'time', 'open', "
            "'high', 'low', 'close'} and should include 'volume' (the "
            "chart shows it in its own pane). 'time' is unix epoch "
            "seconds (int) or 'YYYY-MM-DD', ascending. The chart's "
            "toolbar interval picker switches between the intervals you "
            "provide, so include every interval you have data for."
        )
    )
    selected_interval: str | None = Field(
        default=None,
        description="Initially active interval. Defaults to the first key.",
    )
    series_type: str = Field(
        default="Candlestick",
        description=(
            "Main series style: Candlestick (default), Line, Area, Bar, "
            "Baseline, or Histogram. Non-OHLC types plot the bars' "
            "'close' (or 'value') field."
        ),
    )
    chart_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional chart-level lightweight-charts options "
            "(layout, grid, crosshair, timeScale, rightPriceScale, ...) — "
            "merged over pywry's theme defaults."
        ),
    )
    height: str | None = Field(
        default=None,
        description="CSS height of the chart area, e.g. '600px'. Default 560px.",
    )
    title: str | None = Field(
        default=None,
        description="Optional heading shown above the chart.",
    )


class _TvChartSymbolArgs(BaseModel):
    """Argument schema for :func:`canvas_tvchart_symbol`."""

    symbol: str = Field(
        description=(
            "Ticker to chart, e.g. 'AAPL'. The host's datafeed supplies "
            "its bars, and the chart header's Symbol Search and Compare "
            "controls resolve and load OTHER symbols from the same feed."
        )
    )
    intervals: list[str] | None = Field(
        default=None,
        description=(
            "Toolbar interval ladder, e.g. ['1d','1w','1M']. Defaults to "
            "daily/weekly/monthly. Use codes the datafeed supports "
            "(canonical pywry codes: 1m, 5m, 1h, 1d, 1w, 1M)."
        ),
    )
    selected_interval: str | None = Field(
        default=None,
        description="Initially active interval (must be one of intervals).",
    )
    series_type: str = Field(
        default="Candlestick",
        description="Main series style: Candlestick (default), Line, Area, Bar, ...",
    )
    chart_options: dict[str, Any] | None = Field(
        default=None,
        description="Optional chart-level lightweight-charts options.",
    )
    height: str | None = Field(
        default=None,
        description="CSS height of the chart area, e.g. '600px'. Default 560px.",
    )
    title: str | None = Field(
        default=None,
        description="Optional heading (defaults to the symbol).",
    )


class _ImageArgs(BaseModel):
    """Argument schema for :func:`canvas_image`."""

    src: str | None = Field(
        default=None,
        description=(
            "Image source: an https URL, a data: URI "
            "(data:image/png;base64,...), or a local file path."
        ),
    )
    data_base64: str | None = Field(
        default=None,
        description="Raw image bytes as base64 (pair with 'mime', e.g. image/png).",
    )
    mime: str | None = Field(
        default=None,
        description="MIME type for 'data_base64' (default image/png).",
    )
    name: str | None = Field(
        default=None,
        description="Name of a file uploaded to this run, to render its image.",
    )
    title: str | None = Field(
        default=None,
        description="Optional heading shown above the image.",
    )


class _DocumentArgs(BaseModel):
    """Argument schema for :func:`canvas_document`."""

    src: str | None = Field(
        default=None,
        description=(
            "Document source: an https URL, a data: URI, or a local file path."
        ),
    )
    data_base64: str | None = Field(
        default=None,
        description="Raw document bytes as base64 (pair with 'mime').",
    )
    mime: str | None = Field(
        default=None,
        description=(
            "MIME type of the document — e.g. application/pdf, text/csv, "
            "application/json, audio/mpeg, video/mp4. Drives how it renders."
        ),
    )
    name: str | None = Field(
        default=None,
        description="Name of a file uploaded to this run, to render it.",
    )
    filename: str | None = Field(
        default=None,
        description="Display name for the download-link / PDF-title fallback.",
    )
    title: str | None = Field(
        default=None,
        description="Optional heading shown above the document.",
    )
    text: str | None = Field(
        default=None,
        description=(
            "Optional extracted text shown alongside a download link for "
            "types the browser cannot render inline (e.g. .docx)."
        ),
    )


class _ClearArgs(BaseModel):
    """No arguments."""


def canvas_html(html: str, title: str | None = None) -> str:
    """Render an HTML fragment on the live canvas, replacing its content.

    Parameters
    ----------
    html : str
        Self-contained HTML markup. Inline styles are honored; ``<script>``
        tags do not execute.
    title : str or None, optional
        Heading shown above the content.

    Returns
    -------
    str
        A status line on success, or the no-canvas error string when no
        live canvas is bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    canvas.show_html(html, title=title)
    return "canvas updated: html rendered"


def canvas_markdown(text: str, title: str | None = None) -> str:
    """Render markdown on the live canvas, replacing its content.

    Parameters
    ----------
    text : str
        Markdown source to render.
    title : str or None, optional
        Heading shown above the content.

    Returns
    -------
    str
        A status line on success, or the no-canvas error string when no
        live canvas is bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    canvas.show_markdown(text, title=title)
    return "canvas updated: markdown rendered"


def canvas_plotly(figure: dict[str, Any], title: str | None = None) -> str:
    """Render a Plotly figure on the live canvas, replacing its content.

    Parameters
    ----------
    figure : dict
        A Plotly figure as JSON, with a ``data`` list of traces and an
        optional ``layout`` object.
    title : str or None, optional
        Heading shown above the chart.

    Returns
    -------
    str
        A status line on success; an error string when ``figure`` is not a
        dict with a ``data`` key, or the no-canvas error string when no
        live canvas is bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    if not isinstance(figure, dict) or "data" not in figure:
        return (
            "error: figure must be a Plotly JSON object with a 'data' "
            "list (and optionally 'layout')"
        )
    canvas.show_plotly(figure, title=title)
    return "canvas updated: chart rendered"


def canvas_table(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    title: str | None = None,
) -> str:
    """Render a data table on the live canvas, replacing its content.

    Parameters
    ----------
    rows : list of dict
        Table rows as objects keyed by column name.
    columns : list of str or None, optional
        Explicit column order. Defaults to the keys of the first row.
    title : str or None, optional
        Heading shown above the table.

    Returns
    -------
    str
        A status line naming the row count; an error string when ``rows``
        is empty, or the no-canvas error string when no live canvas is
        bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    if not rows:
        return "error: rows is empty — nothing to render"
    canvas.show_table(rows, title=title, columns=columns)
    return f"canvas updated: table with {len(rows)} row(s) rendered"


def canvas_tvchart(
    datasets: dict[str, list[dict[str, Any]]],
    selected_interval: str | None = None,
    series_type: str = "Candlestick",
    chart_options: dict[str, Any] | None = None,
    height: str | None = None,
    title: str | None = None,
) -> str:
    """Render a full PyWry TVChart (engine + toolbars) on the live canvas.

    Validates ``series_type`` against the supported series styles and
    ``selected_interval`` against the dataset keys before mounting.

    Parameters
    ----------
    datasets : dict of str to list of dict
        OHLCV bars keyed by interval code. Each bar needs at least a
        ``time`` field and should carry ``open``/``high``/``low``/``close``
        and ``volume``, ascending by ``time``.
    selected_interval : str or None, optional
        Interval active on mount. Defaults to the first dataset key.
    series_type : str, default "Candlestick"
        Main series style, one of Candlestick, Line, Area, Bar, Baseline,
        or Histogram (case-insensitive).
    chart_options : dict or None, optional
        Chart-level lightweight-charts options merged over theme defaults.
    height : str or None, optional
        CSS height of the chart area, e.g. ``"600px"``.
    title : str or None, optional
        Heading shown above the chart.

    Returns
    -------
    str
        A status line listing the toolbar intervals; an error string when
        ``datasets`` is empty, a dataset lacks valid bar rows,
        ``series_type`` is unknown, or ``selected_interval`` is not a
        dataset key; or the no-canvas error string when no live canvas is
        bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    if not datasets:
        return "error: datasets is empty — provide at least one interval"
    for interval, bars in datasets.items():
        if not bars or not all(
            isinstance(b, dict) and b.get("time") is not None for b in bars
        ):
            return (
                f"error: datasets[{interval!r}] needs a non-empty list of "
                "{time, ...} bar rows"
            )
    stype = _TV_SERIES_TYPES.get(str(series_type).lower())
    if stype is None:
        return (
            f"error: series_type {series_type!r} is not one of "
            f"{sorted(_TV_SERIES_TYPES.values())}"
        )
    if selected_interval is not None and selected_interval not in datasets:
        return (
            f"error: selected_interval {selected_interval!r} is not a key "
            f"of datasets {sorted(datasets)}"
        )
    canvas.show_tvchart(
        datasets,
        selected_interval=selected_interval,
        series_type=stype,
        title=title,
        chart_options=chart_options,
        height=height,
    )
    intervals = ", ".join(datasets)
    return f"canvas updated: tvchart rendered with toolbar intervals [{intervals}]"


def canvas_tvchart_symbol(
    symbol: str,
    intervals: list[str] | None = None,
    selected_interval: str | None = None,
    series_type: str = "Candlestick",
    chart_options: dict[str, Any] | None = None,
    height: str | None = None,
    title: str | None = None,
) -> str:
    """Render a datafeed-backed TVChart where Symbol Search and Compare work.

    Unlike :func:`canvas_tvchart`, bars come from the host's datafeed
    rather than the caller, so the header's Symbol Search and Compare
    controls can resolve and load any symbol the feed knows.

    Parameters
    ----------
    symbol : str
        Ticker to chart, e.g. ``"AAPL"``.
    intervals : list of str or None, optional
        Toolbar interval ladder. Defaults to daily/weekly/monthly.
    selected_interval : str or None, optional
        Interval active on mount; must be one of ``intervals``.
    series_type : str, default "Candlestick"
        Main series style, one of Candlestick, Line, Area, Bar, Baseline,
        or Histogram (case-insensitive).
    chart_options : dict or None, optional
        Chart-level lightweight-charts options.
    height : str or None, optional
        CSS height of the chart area, e.g. ``"600px"``.
    title : str or None, optional
        Heading shown above the chart. Defaults to the symbol.

    Returns
    -------
    str
        A status line confirming the datafeed chart; an error string when
        ``symbol`` is blank, ``series_type`` is unknown, the host does not
        support datafeed-backed charts, or the host raises while mounting;
        or the no-canvas error string when no live canvas is bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    if not symbol or not symbol.strip():
        return "error: symbol is required"
    stype = _TV_SERIES_TYPES.get(str(series_type).lower())
    if stype is None:
        return (
            f"error: series_type {series_type!r} is not one of "
            f"{sorted(_TV_SERIES_TYPES.values())}"
        )
    show = getattr(canvas, "show_tvchart_symbol", None)
    if not callable(show):
        return "error: this canvas host does not support datafeed-backed charts"
    try:
        show(
            symbol.strip(),
            intervals=intervals,
            selected_interval=selected_interval,
            series_type=stype,
            title=title,
            chart_options=chart_options,
            height=height,
        )
    except (RuntimeError, ValueError) as exc:
        return f"error: {exc}"
    return (
        f"canvas updated: datafeed tvchart rendered for {symbol.strip()!r} "
        "(Symbol Search + Compare enabled)"
    )


# ---------------------------------------------------------------------------
# Full TVChart protocol tools.
#
# Each tool wraps pywry's ``TVChartStateMixin`` — the library's own
# Python implementation of the ``tvchart:*`` protocol — bound to the
# canvas chart via the host's controller.
#
# Indicators are dispatched by the engine on a catalog ``key`` (see
# ``frontend/src/tvchart/09-indicators/00-helpers-catalog.js`` and the
# dispatch in ``04-series.js``). For most catalog entries the engine
# derives that key from the name (``name.lower().replace(' ', '-')``),
# so the name alone resolves. Three entries carry an explicit key that
# differs from the name — Moving Average and the two Volume Profiles —
# and the library's own indicators panel adds them by passing the
# catalog def (key included) to the engine. The ``tvchart:add-indicator``
# event exposes a first-class ``key`` field for exactly this, so for
# those entries the tools resolve the key from the catalog below (a
# verbatim mirror of ``_INDICATOR_CATALOG``'s explicit keys) and pass it.
# ---------------------------------------------------------------------------

_NO_TVCHART = (
    "error: no TVChart is mounted on the canvas — render one first with canvas_tvchart"
)

# Name -> catalog key, mirroring the explicit-key entries of pywry's
# ``_INDICATOR_CATALOG``. Names absent here dispatch by the engine's
# name-derived key and need no explicit key.
_TV_CATALOG_KEYS: dict[str, str] = {
    "Average Price": "average-price",
    "Correlation": "correlation",
    "Median Price": "median-price",
    "Momentum": "momentum",
    "Moving Average": "moving-average-ex",
    "Percent Change": "percent-change",
    "Product": "product",
    "Ratio": "ratio",
    "Spread": "spread",
    "Sum": "sum",
    "Weighted Close": "weighted-close",
    "Volume Profile Fixed Range": "volume-profile-fixed",
    "Volume Profile Visible Range": "volume-profile-visible",
}


def _tvchart_ctrl() -> tuple[Any, str | None]:
    """Resolve the mounted chart's protocol controller."""
    canvas = _canvas()
    if canvas is None:
        return None, _NO_CANVAS
    getter = getattr(canvas, "tvchart_controller", None)
    ctrl = getter() if callable(getter) else None
    if ctrl is None:
        return None, _NO_TVCHART
    return ctrl, None


def canvas_tvchart_update(
    bars: list[dict[str, Any]],
    series_id: str | None = None,
    fit_content: bool = True,
) -> str:
    """Replace all bar data for a series (TVChartStateMixin.update_series).

    Parameters
    ----------
    bars : list of dict
        Replacement bars, each ``{time, open, high, low, close, volume?}``
        or ``{time, value}``, ascending by ``time``.
    series_id : str or None, optional
        Target series; ``None`` updates the main series.
    fit_content : bool, default True
        Whether to auto-fit the visible range to the new data.

    Returns
    -------
    str
        A status line naming the bar count and series; an error string
        when ``bars`` is empty, the no-tvchart error string when no chart
        is mounted, or the no-canvas error string when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if not bars:
        return "error: bars is empty"
    ctrl.update_series(bars, series_id=series_id, fit_content=fit_content)
    return f"tvchart updated: {len(bars)} bars on '{series_id or 'main'}'"


def canvas_tvchart_stream_bar(
    bar: dict[str, Any],
    series_id: str | None = None,
) -> str:
    """Stream one live bar tick (TVChartStateMixin.update_bar).

    Updates the last bar when ``time`` matches it, otherwise appends a new
    bar; ``time`` must be greater than or equal to the last bar's time.

    Parameters
    ----------
    bar : dict
        A single bar, ``{time, open, high, low, close, volume?}`` or
        ``{time, value}``. Must contain a ``time`` field.
    series_id : str or None, optional
        Target series; ``None`` updates the main series.

    Returns
    -------
    str
        A status line naming the bar time; an error string when ``bar`` is
        not a dict with ``time``, the no-tvchart error string when no chart
        is mounted, or the no-canvas error string when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if not isinstance(bar, dict) or bar.get("time") is None:
        return "error: bar must be a {time, ...} dict"
    ctrl.update_bar(bar, series_id=series_id)
    return f"tvchart streamed bar at time={bar['time']}"


def canvas_tvchart_add_series(
    series_id: str,
    bars: list[dict[str, Any]],
    series_type: str = "Line",
    series_options: dict[str, Any] | None = None,
) -> str:
    """Add an overlay data series (TVChartStateMixin.add_indicator).

    Parameters
    ----------
    series_id : str
        Identifier for the new series; must be non-empty and not ``"main"``.
    bars : list of dict
        Series data, ``{time, value}`` or OHLC rows, ascending by ``time``.
    series_type : str, default "Line"
        Series style, e.g. Line, Histogram, Area, Candlestick, Bar, or
        Baseline.
    series_options : dict or None, optional
        lightweight-charts series options; defaults to an empty mapping.

    Returns
    -------
    str
        A status line describing the added series; an error string when
        ``series_id`` is empty or ``"main"`` or ``bars`` is empty, the
        no-tvchart error string when no chart is mounted, or the no-canvas
        error string when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if not series_id or series_id == "main":
        return "error: series_id must be a non-empty id other than 'main'"
    if not bars:
        return "error: bars is empty"
    ctrl.add_indicator(
        bars,
        series_id=series_id,
        series_type=series_type,
        series_options=series_options or {},
    )
    return f"tvchart series '{series_id}' added ({series_type}, {len(bars)} bars)"


def canvas_tvchart_remove_series(series_id: str) -> str:
    """Remove an overlay series (TVChartStateMixin.remove_indicator).

    Parameters
    ----------
    series_id : str
        Series to remove; the main series cannot be removed.

    Returns
    -------
    str
        A status line confirming removal; an error string when
        ``series_id`` is empty or ``"main"``, the no-tvchart error string
        when no chart is mounted, or the no-canvas error string when no
        canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if not series_id or series_id == "main":
        return "error: cannot remove the 'main' series"
    ctrl.remove_indicator(series_id)
    return f"tvchart series '{series_id}' removed"


def canvas_tvchart_add_indicator(
    name: str,
    period: int | None = None,
    color: str | None = None,
    source: str | None = None,
    method: str | None = None,
    multiplier: float | None = None,
    ma_type: str | None = None,
    offset: int | None = None,
) -> str:
    """Add a built-in indicator (engine-computed).

    Catalog names (see the mixin docstring): Moving Average (choose
    SMA/EMA/WMA/HMA/VWMA via ``method``), Ichimoku Cloud, Bollinger
    Bands, Keltner Channels, ATR, Historical Volatility, Parabolic SAR,
    RSI, MACD, Stochastic, Williams %R, CCI, ADX, Aroon, VWAP, Volume
    SMA, Accumulation/Distribution.

    Most names dispatch by the engine's name-derived key, so they go
    through ``add_builtin_indicator``. The few catalog entries with an
    explicit key that differs from the name (e.g. Moving Average →
    ``moving-average-ex``) are added via the ``tvchart:add-indicator``
    event with that ``key`` — the same way the library's own indicators
    panel adds them.

    Parameters
    ----------
    name : str
        Catalog indicator name, verbatim (e.g. ``"RSI"``, ``"Moving
        Average"``).
    period : int or None, optional
        Lookback period; defaults from the catalog when omitted.
    color : str or None, optional
        Line color; defaults from the catalog when omitted.
    source : str or None, optional
        Price source, one of close/open/high/low/hl2/hlc3/ohlc4.
    method : str or None, optional
        Moving-average method (SMA/EMA/WMA/HMA/VWMA) where applicable.
    multiplier : float or None, optional
        Band/channel multiplier (e.g. for Bollinger Bands).
    ma_type : str or None, optional
        Moving-average type for band-style indicators.
    offset : int or None, optional
        Plot offset in bars.

    Returns
    -------
    str
        A status line naming the indicator; an error string when ``name``
        is empty, the no-tvchart error string when no chart is mounted, or
        the no-canvas error string when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if not name:
        return "error: name is required"
    key = _TV_CATALOG_KEYS.get(name)
    if key is None:
        # Name-dispatched: the mixin convenience method is sufficient.
        ctrl.add_builtin_indicator(
            name,
            period=period,
            color=color,
            source=source,
            method=method,
            multiplier=multiplier,
            ma_type=ma_type,
            offset=offset,
        )
        return f"tvchart indicator added: {name}"
    # Key-dispatched: pass the catalog key through the documented event
    # field (add_builtin_indicator does not expose it).
    payload: dict[str, Any] = {"name": name, "key": key}
    if period is not None:
        payload["period"] = period
    if color is not None:
        payload["color"] = color
    if source is not None:
        payload["source"] = source
    if method is not None:
        payload["method"] = method
    if multiplier is not None:
        payload["multiplier"] = multiplier
    if ma_type is not None:
        payload["maType"] = ma_type
    if offset is not None:
        payload["offset"] = offset
    ctrl.emit("tvchart:add-indicator", payload)
    return f"tvchart indicator added: {name}"


def canvas_tvchart_remove_indicator(series_id: str) -> str:
    """Remove a built-in indicator (TVChartStateMixin.remove_builtin_indicator).

    Parameters
    ----------
    series_id : str
        The indicator group's series id, as reported by
        :func:`canvas_tvchart_list_indicators`.

    Returns
    -------
    str
        A status line confirming removal; the no-tvchart error string when
        no chart is mounted, or the no-canvas error string when no canvas
        is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    ctrl.remove_builtin_indicator(series_id)
    return f"tvchart indicator '{series_id}' removed"


def canvas_tvchart_list_indicators() -> str:
    """List active indicators (TVChartStateMixin.list_indicators, sync).

    Synchronously round-trips a request to the chart and returns its
    indicator list as JSON.

    Returns
    -------
    str
        A JSON array of indicator descriptors (seriesId/name/period/
        color/group); an error string on timeout, the no-tvchart error
        string when no chart is mounted, or the no-canvas error string
        when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    response = ctrl.list_indicators_sync()
    if response is None:
        return "error: no response from the chart (timeout)"
    return json.dumps(response.get("indicators", []), default=str)


def canvas_tvchart_add_volume_profile(
    mode: str = "visible",
    bucket_count: int = 24,
    from_index: int | None = None,
    to_index: int | None = None,
    placement: str = "right",
    width_percent: float = 25.0,
    value_area_pct: float = 0.70,
    show_poc: bool = True,
    show_value_area: bool = True,
    up_color: str | None = None,
    down_color: str | None = None,
    poc_color: str | None = None,
) -> str:
    """Add a volume-by-price profile (engine-computed).

    Volume Profile entries carry an explicit catalog key that differs
    from the name, so — like Moving Average — they are added via the
    ``tvchart:add-indicator`` event with the catalog ``key`` and the
    Volume Profile payload fields the engine reads.

    Parameters
    ----------
    mode : str, default "visible"
        ``"visible"`` tracks the viewport; ``"fixed"`` buckets a fixed
        bar-index range given by ``from_index``/``to_index``.
    bucket_count : int, default 24
        Number of price buckets in the profile.
    from_index : int or None, optional
        First bar index of the fixed range (``mode="fixed"`` only).
    to_index : int or None, optional
        Last bar index of the fixed range (``mode="fixed"`` only).
    placement : str, default "right"
        Side the profile is drawn on, ``"right"`` or ``"left"``.
    width_percent : float, default 25.0
        Profile width as a percentage of the chart width.
    value_area_pct : float, default 0.70
        Fraction of volume enclosed by the highlighted value area.
    show_poc : bool, default True
        Whether to draw the point-of-control line.
    show_value_area : bool, default True
        Whether to highlight the value area.
    up_color : str or None, optional
        Color for up-volume buckets.
    down_color : str or None, optional
        Color for down-volume buckets.
    poc_color : str or None, optional
        Color for the point-of-control line.

    Returns
    -------
    str
        A status line describing the profile; an error string when
        ``mode`` or ``placement`` is invalid, the no-tvchart error string
        when no chart is mounted, or the no-canvas error string when no
        canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if mode not in ("visible", "fixed"):
        return "error: mode must be 'visible' or 'fixed'"
    if placement not in ("right", "left"):
        return "error: placement must be 'right' or 'left'"
    name = (
        "Volume Profile Fixed Range"
        if mode == "fixed"
        else "Volume Profile Visible Range"
    )
    payload: dict[str, Any] = {
        "name": name,
        "key": _TV_CATALOG_KEYS[name],
        "period": int(bucket_count),
        "placement": placement,
        "widthPercent": float(width_percent),
        "valueAreaPct": float(value_area_pct),
        "showPOC": bool(show_poc),
        "showValueArea": bool(show_value_area),
    }
    if mode == "fixed" and from_index is not None and to_index is not None:
        payload["fromIndex"] = int(from_index)
        payload["toIndex"] = int(to_index)
    if up_color is not None:
        payload["upColor"] = up_color
    if down_color is not None:
        payload["downColor"] = down_color
    if poc_color is not None:
        payload["pocColor"] = poc_color
    ctrl.emit("tvchart:add-indicator", payload)
    return f"tvchart volume profile added ({mode}, {bucket_count} buckets)"


def canvas_tvchart_add_markers(
    markers: list[dict[str, Any]],
    series_id: str | None = None,
) -> str:
    """Add markers to a series (TVChartStateMixin.add_marker).

    Each marker is a lightweight-charts marker dict: ``time`` plus
    ``position`` ('aboveBar'/'belowBar'/'inBar'), ``shape``
    ('arrowUp'/'arrowDown'/'circle'/'square'), ``color``, ``text``.

    Parameters
    ----------
    markers : list of dict
        Marker descriptors; each must contain a ``time`` field.
    series_id : str or None, optional
        Series to attach the markers to; ``None`` uses the main series.

    Returns
    -------
    str
        A status line naming the marker count and series; an error string
        when ``markers`` is empty or a marker lacks ``time``, the
        no-tvchart error string when no chart is mounted, or the no-canvas
        error string when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if not markers:
        return "error: markers is empty"
    for i, marker in enumerate(markers):
        if not isinstance(marker, dict) or marker.get("time") is None:
            return f"error: markers[{i}] needs a 'time' field"
    ctrl.add_marker(markers, series_id=series_id)
    return f"tvchart markers added: {len(markers)} on '{series_id or 'main'}'"


def canvas_tvchart_add_price_line(
    price: float,
    color: str = "#2196F3",
    line_width: int = 1,
    title: str = "",
    series_id: str | None = None,
) -> str:
    """Add a horizontal price line (TVChartStateMixin.add_price_line).

    Parameters
    ----------
    price : float
        Price level at which to draw the line.
    color : str, default "#2196F3"
        Line color as a CSS color string.
    line_width : int, default 1
        Line width in pixels.
    title : str, default ""
        Label rendered on the price line.
    series_id : str or None, optional
        Series the line attaches to; ``None`` uses the main series.

    Returns
    -------
    str
        A status line naming the price; the no-tvchart error string when
        no chart is mounted, or the no-canvas error string when no canvas
        is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    ctrl.add_price_line(
        float(price),
        color=color,
        line_width=int(line_width),
        title=title,
        series_id=series_id,
    )
    return f"tvchart price line added at {price}"


def canvas_tvchart_set_visible_range(from_time: int, to_time: int) -> str:
    """Set the visible time range (TVChartStateMixin.set_visible_range).

    Parameters
    ----------
    from_time : int
        Start of the visible window, as unix epoch seconds.
    to_time : int
        End of the visible window, as unix epoch seconds.

    Returns
    -------
    str
        A status line naming the range; the no-tvchart error string when
        no chart is mounted, or the no-canvas error string when no canvas
        is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    ctrl.set_visible_range(int(from_time), int(to_time))
    return f"tvchart visible range set to {from_time}..{to_time}"


def canvas_tvchart_fit_content() -> str:
    """Auto-fit the chart to all data (TVChartStateMixin.fit_content).

    Returns
    -------
    str
        A status line on success; the no-tvchart error string when no
        chart is mounted, or the no-canvas error string when no canvas is
        bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    ctrl.fit_content()
    return "tvchart fit to content"


def canvas_tvchart_apply_options(
    chart_options: dict[str, Any] | None = None,
    series_options: dict[str, Any] | None = None,
    series_id: str | None = None,
) -> str:
    """Apply chart/series options (TVChartStateMixin.apply_chart_options).

    Parameters
    ----------
    chart_options : dict or None, optional
        Chart-level lightweight-charts options (layout/grid/crosshair/
        timeScale/rightPriceScale/...).
    series_options : dict or None, optional
        Options applied to a single series.
    series_id : str or None, optional
        Series targeted by ``series_options``; ``None`` uses the main
        series.

    Returns
    -------
    str
        A status line on success; an error string when neither options
        mapping is given, the no-tvchart error string when no chart is
        mounted, or the no-canvas error string when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    if not chart_options and not series_options:
        return "error: provide chart_options and/or series_options"
    ctrl.apply_chart_options(
        chart_options=chart_options,
        series_options=series_options,
        series_id=series_id,
    )
    return "tvchart options applied"


def canvas_tvchart_get_state() -> str:
    """Read the live chart state (TVChartStateMixin.request_tvchart_state, sync).

    Synchronously requests the chart's state and returns it as JSON. The
    bulky ``rawData`` array is dropped and replaced by a ``barCount``
    field, and the ``context`` key is removed, to keep the payload small.

    Returns
    -------
    str
        A JSON object describing the chart (symbol, interval, chartType,
        series map, visible range, indicators, drawings, ``barCount``); an
        error string on timeout, the no-tvchart error string when no chart
        is mounted, or the no-canvas error string when no canvas is bound.
    """
    ctrl, err = _tvchart_ctrl()
    if err:
        return err
    state = ctrl.get_state()
    if state is None:
        return "error: no response from the chart (timeout)"
    trimmed = dict(state)
    raw = trimmed.pop("rawData", None)
    trimmed["barCount"] = len(raw) if isinstance(raw, list) else 0
    trimmed.pop("context", None)
    return json.dumps(trimmed, default=str)


async def canvas_image(
    src: str | None = None,
    data_base64: str | None = None,
    mime: str | None = None,
    name: str | None = None,
    title: str | None = None,
) -> str:
    """Render an image on the live canvas.

    Accepts an https URL or ``data:`` URI in ``src``; raw image bytes as
    base64 in ``data_base64`` (with ``mime``); a local file path in
    ``src``; or the ``name`` of a file uploaded to this run. All are
    normalized to a single renderable source.

    Parameters
    ----------
    src : str or None, optional
        Image source: an https URL, a ``data:`` URI, or a local file path.
    data_base64 : str or None, optional
        Raw image bytes as base64, paired with ``mime``.
    mime : str or None, optional
        MIME type for ``data_base64`` (default ``image/png``).
    name : str or None, optional
        Name of a file uploaded to this run, whose image to render.
    title : str or None, optional
        Heading shown above the image.

    Returns
    -------
    str
        A status line on success; an error string when the source cannot
        be resolved, or the no-canvas error string when no canvas is bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    from openbb_agent_server.plugins.tools._media import MediaError

    try:
        raw, url, rmime = await _resolve_media(
            src, data_base64, mime, name, default_mime="image/png"
        )
        norm = await _renderable_src(raw, url, rmime)
    except (ValueError, MediaError) as exc:
        return f"error: {exc}"
    canvas.show_image(norm, title=title)
    return "canvas updated: image rendered"


async def canvas_document(
    src: str | None = None,
    data_base64: str | None = None,
    mime: str | None = None,
    name: str | None = None,
    filename: str | None = None,
    title: str | None = None,
    text: str | None = None,
) -> str:
    """Render ANY document/media on the canvas, dispatched by MIME type.

    Images, PDFs, audio, and video embed natively; CSV/TSV and tabular
    JSON render as a table; markdown/HTML/JSON/plain text render as rich
    text; anything else (Office docs, archives, unknown binary) becomes a
    download link (plus ``text`` as an extracted-content fallback).
    Source resolution matches :func:`canvas_image` (src URL/data:/path,
    ``data_base64`` + ``mime``, or uploaded ``name``).

    Parameters
    ----------
    src : str or None, optional
        Document source: an https URL, a ``data:`` URI, or a local path.
    data_base64 : str or None, optional
        Raw document bytes as base64, paired with ``mime``.
    mime : str or None, optional
        MIME type driving how the document renders; when omitted it is
        inferred from the resolved source.
    name : str or None, optional
        Name of a file uploaded to this run, to render.
    filename : str or None, optional
        Display name for the download-link / PDF-title fallback.
    title : str or None, optional
        Heading shown above the document.
    text : str or None, optional
        Extracted text shown alongside a download link for types the
        browser cannot render inline (e.g. ``.docx``).

    Returns
    -------
    str
        A status line naming the rendered kind; an error string when the
        source cannot be resolved or text/data content is malformed, or
        the no-canvas error string when no canvas is bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    from openbb_agent_server.plugins.tools._media import MediaError

    try:
        raw, url, rmime = await _resolve_media(
            src, data_base64, mime, name, default_mime="application/octet-stream"
        )
        eff = (mime or rmime).split(";", 1)[0].strip().lower()
        if eff in _TEXT_FAMILY_MIMES:
            return await _render_text_family(canvas, eff, raw, url, title)
        norm = await _renderable_src(raw, url, eff)
    except (ValueError, MediaError) as exc:
        return f"error: {exc}"
    if eff.startswith("image/"):
        canvas.show_image(norm, title=title)
        return "canvas updated: image rendered"
    canvas.show_document(src=norm, mime=eff, filename=filename, title=title, text=text)
    kind = "pdf" if eff == "application/pdf" else eff.split("/", 1)[0]
    return f"canvas updated: {kind} document rendered"


async def _render_text_family(
    canvas: Any,
    eff: str,
    raw: bytes | None,
    url: str | None,
    title: str | None,
) -> str:
    """Render a text/data MIME as a rich canvas view (table/markdown/...)."""
    body = await _resolved_text(raw, url)
    if eff in _TEXT_TABLE_MIMES:
        delimiter = "\t" if eff == "text/tab-separated-values" else ","
        try:
            rows = list(csv.DictReader(io.StringIO(body), delimiter=delimiter))
        except csv.Error as exc:
            return f"error: malformed delimited text ({exc})"
        canvas.show_table(rows, title=title)
        return f"canvas updated: table rendered from {len(rows)} rows"
    if eff == "application/json":
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            return f"error: malformed JSON ({exc})"
        if (
            isinstance(parsed, list)
            and parsed
            and all(isinstance(r, dict) for r in parsed)
        ):
            canvas.show_table(parsed, title=title)
            return f"canvas updated: table rendered from {len(parsed)} JSON rows"
        canvas.show_document(
            src="", mime="text/plain", title=title, text=json.dumps(parsed, indent=2)
        )
        return "canvas updated: JSON rendered"
    if eff == "text/markdown":
        canvas.show_markdown(body, title=title)
        return "canvas updated: markdown rendered"
    if eff == "text/html":
        canvas.show_html(body, title=title)
        return "canvas updated: html rendered"
    # text/plain and application/yaml — verbatim, escaped <pre>.
    canvas.show_document(src="", mime="text/plain", title=title, text=body)
    return "canvas updated: text rendered"


def canvas_clear() -> str:
    """Reset the live canvas to its empty state.

    Returns
    -------
    str
        A status line on success, or the no-canvas error string when no
        live canvas is bound.
    """
    canvas = _canvas()
    if canvas is None:
        return _NO_CANVAS
    canvas.clear()
    return "canvas cleared"


class PyWryCanvasToolSource(ToolSource):
    """Tool source that exposes the live-canvas drawing tools.

    Registered under the name ``pywry_canvas``; its :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools` builds
    the canvas tool set only when a host has bound a live canvas.
    """

    name = "pywry_canvas"

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Return the canvas tools when a canvas host is bound.

        Without a bound canvas the tools are not registered at all —
        the model never sees them, instead of seeing tools that always
        error.

        Parameters
        ----------
        ctx : RunContext
            The active run context (unused here; canvas binding is read
            from the process-global slot).
        config : dict
            Tool-source configuration (unused by this source).

        Returns
        -------
        list
            The canvas ``StructuredTool`` instances, or an empty list when
            no live canvas is bound.
        """
        if _canvas() is None:
            logger.debug(
                "pywry_canvas: no live canvas bound; skipping tool registration"
            )
            return []
        common = (
            "The canvas is the user's main content page next to the chat "
            "panel — a live display surface that persists until replaced. "
            "Rendering REPLACES the current canvas content. Prefer the "
            "canvas for anything worth keeping on screen (charts, tables, "
            "dashboards); use chat artifacts for inline transcript items."
        )
        return [
            StructuredTool.from_function(
                func=canvas_html,
                name="canvas_html",
                description=(
                    "Render an HTML fragment on the live canvas. "
                    "Scripts do not execute; use inline styles. " + common
                ),
                args_schema=_HtmlArgs,
            ),
            StructuredTool.from_function(
                func=canvas_markdown,
                name="canvas_markdown",
                description=("Render markdown on the live canvas. " + common),
                args_schema=_MarkdownArgs,
            ),
            StructuredTool.from_function(
                func=canvas_plotly,
                name="canvas_plotly",
                description=(
                    "Render an interactive Plotly figure on the live "
                    "canvas. Pass standard Plotly JSON: {'data': [...], "
                    "'layout': {...}}. " + common
                ),
                args_schema=_PlotlyArgs,
            ),
            StructuredTool.from_function(
                func=canvas_table,
                name="canvas_table",
                description=(
                    "Render a data table on the live canvas from a list "
                    "of row objects. " + common
                ),
                args_schema=_TableArgs,
            ),
            StructuredTool.from_function(
                func=canvas_tvchart,
                name="canvas_tvchart",
                description=(
                    "Render a full TradingView chart on the live canvas "
                    "via pywry's TVChart engine — complete with toolbars: "
                    "interval picker (switches between the datasets you "
                    "provide), chart-type menu, client-side indicators, "
                    "drawing tools, time-range tabs, OHLC legend, and a "
                    "volume pane from the bars' 'volume' field. THE tool "
                    "for price/OHLCV charts; use canvas_plotly for "
                    "non-price charts. " + common
                ),
                args_schema=_TvChartArgs,
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_symbol,
                name="canvas_tvchart_symbol",
                description=(
                    "Render a DATAFEED-BACKED TradingView chart for a "
                    "ticker. Same chrome as canvas_tvchart, but the host's "
                    "datafeed supplies the data, so the header's Symbol "
                    "Search and Compare controls fully work (resolve + load "
                    "any symbol the feed knows). Use this when the user may "
                    "want to search/compare symbols interactively; use "
                    "canvas_tvchart when you already hold the exact bars to "
                    "display. " + common
                ),
                args_schema=_TvChartSymbolArgs,
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_update,
                name="canvas_tvchart_update",
                description=(
                    "Replace ALL bar data of a series on the mounted "
                    "TVChart (default series 'main'). Bars: {time, open, "
                    "high, low, close, volume?} or {time, value}, "
                    "ascending. Volume embedded in bars refreshes the "
                    "volume pane."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_stream_bar,
                name="canvas_tvchart_stream_bar",
                description=(
                    "Stream ONE live bar tick into the mounted TVChart "
                    "(updates the last bar when 'time' matches, else "
                    "appends — 'time' must be >= the last bar). Use for "
                    "live/streaming updates instead of full replaces."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_add_series,
                name="canvas_tvchart_add_series",
                description=(
                    "Add an overlay data series to the mounted TVChart "
                    "(e.g. a precomputed line). series_type: Line, "
                    "Histogram, Area, Candlestick, Bar, Baseline. bars are "
                    "{time, value} or OHLC rows."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_remove_series,
                name="canvas_tvchart_remove_series",
                description="Remove a series from the mounted TVChart by its series_id.",
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_add_indicator,
                name="canvas_tvchart_add_indicator",
                description=(
                    "Add a built-in technical indicator to the mounted "
                    "TVChart, computed by pywry's indicator engine. "
                    "name (verbatim from the catalog): 'Moving Average' "
                    "(method=SMA/EMA/WMA/HMA/VWMA), 'RSI', 'MACD', "
                    "'Bollinger Bands' (multiplier/ma_type), 'Keltner "
                    "Channels', 'Stochastic', 'ATR', 'ADX', 'Aroon', "
                    "'CCI', 'Williams %R', 'VWAP', 'Volume SMA', "
                    "'Historical Volatility', 'Parabolic SAR', 'Ichimoku "
                    "Cloud', 'Accumulation/Distribution'. source: "
                    "close/open/high/low/hl2/hlc3/ohlc4. Color and period "
                    "default from the catalog when omitted."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_remove_indicator,
                name="canvas_tvchart_remove_indicator",
                description=(
                    "Remove an indicator (whole group) from the mounted "
                    "TVChart by its seriesId — get ids from "
                    "canvas_tvchart_list_indicators."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_list_indicators,
                name="canvas_tvchart_list_indicators",
                description=(
                    "List the active indicators on the mounted TVChart "
                    "(returns JSON with seriesId/name/period/color/group)."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_add_volume_profile,
                name="canvas_tvchart_add_volume_profile",
                description=(
                    "Add a volume-by-price profile to the mounted TVChart. "
                    "mode 'visible' tracks the viewport; 'fixed' buckets a "
                    "bar-index range (from_index/to_index). POC line and "
                    "value area included by default."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_add_markers,
                name="canvas_tvchart_add_markers",
                description=(
                    "Add buy/sell/annotation markers to a TVChart series. "
                    "Each marker: {time, position: aboveBar/belowBar/"
                    "inBar, shape: arrowUp/arrowDown/circle/square, "
                    "color?, text?}."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_add_price_line,
                name="canvas_tvchart_add_price_line",
                description=(
                    "Add a horizontal price line (support/resistance/"
                    "entry level) to a TVChart series."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_set_visible_range,
                name="canvas_tvchart_set_visible_range",
                description=(
                    "Set the TVChart visible time range "
                    "(from_time/to_time as unix epoch seconds)."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_fit_content,
                name="canvas_tvchart_fit_content",
                description="Auto-fit the mounted TVChart to show all data.",
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_apply_options,
                name="canvas_tvchart_apply_options",
                description=(
                    "Apply lightweight-charts options to the mounted "
                    "TVChart: chart_options (layout/grid/crosshair/"
                    "timeScale/rightPriceScale/...) and/or series_options "
                    "for one series."
                ),
            ),
            StructuredTool.from_function(
                func=canvas_tvchart_get_state,
                name="canvas_tvchart_get_state",
                description=(
                    "Read the live TVChart state as JSON: symbol, "
                    "interval, chartType, series map, visible range, "
                    "indicators, drawings, bar count. Use to inspect what "
                    "is currently on screen before mutating it."
                ),
            ),
            StructuredTool.from_function(
                coroutine=canvas_image,
                name="canvas_image",
                description=(
                    "Render an image on the live canvas. Source can be an "
                    "https URL, a data: URI, a local file path, raw bytes "
                    "(data_base64 + mime), or an uploaded file (name). " + common
                ),
                args_schema=_ImageArgs,
            ),
            StructuredTool.from_function(
                coroutine=canvas_document,
                name="canvas_document",
                description=(
                    "Render ANY document or media on the live canvas — "
                    "image, PDF, audio, video, CSV/JSON table, markdown, "
                    "HTML, plain text, or a download link for unsupported "
                    "types — dispatched by MIME. Provide one of: src (https "
                    "URL | data: URI | local path), data_base64 (+ mime), or "
                    "name (uploaded file). " + common
                ),
                args_schema=_DocumentArgs,
            ),
            StructuredTool.from_function(
                func=canvas_clear,
                name="canvas_clear",
                description="Reset the live canvas to its empty state.",
                args_schema=_ClearArgs,
            ),
        ]
