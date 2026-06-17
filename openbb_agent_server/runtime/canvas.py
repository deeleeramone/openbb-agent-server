"""Process-global live-canvas slot.

A *canvas* is a host-owned display surface the agent renders to — in
the PyWry desktop embedding it is the window's main content page. The
slot is deliberately host-agnostic: the ``pywry_canvas`` tool source
talks to the :class:`LiveCanvas` protocol only, and the host
(``openbb_agent_server.acp.canvas_app``) binds a concrete
implementation at startup. One canvas per process, matching the
embedded runtime's one-runtime-per-process constraint.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LiveCanvas(Protocol):
    """The surface the canvas tools draw on."""

    def show_html(self, html: str, *, title: str | None = None) -> None:
        """Replace the canvas with an HTML fragment.

        Parameters
        ----------
        html : str
            The HTML markup to render as the canvas content.
        title : str or None, optional
            Window/page title to display alongside the content.
        """

    def show_markdown(self, text: str, *, title: str | None = None) -> None:
        """Replace the canvas with rendered markdown.

        Parameters
        ----------
        text : str
            Markdown source to render into the canvas.
        title : str or None, optional
            Window/page title to display alongside the content.
        """

    def show_plotly(self, figure: dict[str, Any], *, title: str | None = None) -> None:
        """Replace the canvas with an interactive Plotly figure.

        Parameters
        ----------
        figure : dict[str, Any]
            A Plotly figure spec (``data``/``layout`` mapping) to render.
        title : str or None, optional
            Window/page title to display alongside the figure.
        """

    def show_table(
        self,
        rows: list[dict[str, Any]],
        *,
        title: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        """Replace the canvas with a data table.

        Parameters
        ----------
        rows : list[dict[str, Any]]
            Row records, each a mapping of column name to cell value.
        title : str or None, optional
            Window/page title to display alongside the table.
        columns : list[str] or None, optional
            Explicit column order/subset; when omitted the implementation
            infers columns from the row keys.
        """

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
        """Replace the canvas with a full PyWry TVChart (engine + chrome).

        Parameters
        ----------
        datasets : dict[str, list[dict[str, Any]]]
            Mapping of interval label to a list of OHLCV bar records the
            chart renders directly (no datafeed lookup).
        selected_interval : str or None, optional
            Interval key from ``datasets`` to display initially.
        series_type : str, default "Candlestick"
            Series style to render, e.g. ``"Candlestick"`` or ``"Line"``.
        title : str or None, optional
            Window/page title to display alongside the chart.
        chart_options : dict[str, Any] or None, optional
            Extra TVChart configuration forwarded to the engine.
        height : str or None, optional
            CSS height for the chart container.
        """

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
        """Replace the canvas with a datafeed-backed TVChart for a symbol.

        Search/compare work against the configured datafeed provider.

        Parameters
        ----------
        symbol : str
            Ticker/symbol the datafeed resolves bars for.
        intervals : list[str] or None, optional
            Intervals to make selectable; defaults to the implementation's
            built-in set when omitted.
        selected_interval : str or None, optional
            Interval to display initially.
        series_type : str, default "Candlestick"
            Series style to render, e.g. ``"Candlestick"`` or ``"Line"``.
        title : str or None, optional
            Window/page title to display alongside the chart.
        chart_options : dict[str, Any] or None, optional
            Extra TVChart configuration forwarded to the engine.
        height : str or None, optional
            CSS height for the chart container.
        """

    def show_image(self, src: str, *, title: str | None = None) -> None:
        """Replace the canvas with an image.

        ``src`` is an https URL or a data URI; the tool layer normalizes
        bytes / local paths / uploaded files to one of these.

        Parameters
        ----------
        src : str
            An https URL or a ``data:`` URI pointing at the image.
        title : str or None, optional
            Window/page title to display alongside the image.
        """

    def show_document(
        self,
        *,
        src: str,
        mime: str,
        filename: str | None = None,
        title: str | None = None,
        text: str | None = None,
    ) -> None:
        """Replace the canvas with a document rendered by MIME family.

        Handles image/pdf/audio/video/text natively and falls back to a
        download link for anything else.

        Parameters
        ----------
        src : str
            An https URL or a ``data:`` URI pointing at the document.
        mime : str
            MIME type that selects the renderer (image/pdf/audio/video/
            text families render natively; others get a download link).
        filename : str or None, optional
            Display/download name for the document.
        title : str or None, optional
            Window/page title to display alongside the document.
        text : str or None, optional
            Pre-extracted text content, used for text-family documents.
        """

    def tvchart_controller(self) -> Any:
        """Return the TVChart protocol controller, or None when unmounted.

        Returns
        -------
        Any
            The controller object for driving an already-mounted TVChart
            (search/compare/interval changes), or ``None`` when no chart
            is currently mounted.
        """

    def clear(self) -> None:
        """Reset the canvas to its empty state."""


class _CanvasSlot:
    canvas: LiveCanvas | None = None


_slot = _CanvasSlot()


def set_canvas(canvas: LiveCanvas) -> None:
    """Bind the process's live canvas.

    The host calls this once at startup so the canvas tools can reach the
    concrete surface. Replaces any previously bound canvas.

    Parameters
    ----------
    canvas : LiveCanvas
        The concrete canvas implementation to install as the process slot.
    """
    _slot.canvas = canvas


def get_canvas() -> LiveCanvas | None:
    """Return the bound canvas, or ``None`` when no host has bound one.

    Returns
    -------
    LiveCanvas or None
        The canvas installed by :func:`set_canvas`, or ``None`` if the
        host has not bound one (e.g. headless runs).
    """
    return _slot.canvas


def reset_canvas() -> None:
    """Unbind the canvas, clearing the process slot.

    Used during host teardown and in tests to drop any installed canvas
    so a later :func:`get_canvas` returns ``None``.
    """
    _slot.canvas = None
