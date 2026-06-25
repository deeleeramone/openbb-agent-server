"""Tests for the live-canvas registry, tools, and PyWry canvas surface.

Everything here is pywry-free: ``PyWryCanvas`` only needs a handle with
``emit`` / ``eval_js``, and the tool source talks to the registry
protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.acp.canvas import (
    CANVAS_ELEMENT_ID,
    PyWryCanvas,
    build_canvas_html,
    render_table_html,
)
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.plugins.tools.pywry_canvas import (
    PyWryCanvasToolSource,
    canvas_clear,
    canvas_document,
    canvas_html,
    canvas_image,
    canvas_markdown,
    canvas_plotly,
    canvas_table,
    canvas_tvchart,
)
from openbb_agent_server.runtime import canvas as canvas_registry
from openbb_agent_server.runtime.canvas import LiveCanvas
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


class _FakeHandle:
    """Capture emit calls like a PyWry widget handle.

    Deliberately exposes ONLY ``emit`` — no ``eval_js``, no
    ``set_trait`` — to pin the canvas to the lowest-common-denominator
    surface shared by all three PyWry rendering paths.
    """

    def __init__(self) -> None:
        self.emits: list[tuple[str, dict[str, Any]]] = []
        self.handlers: dict[str, Any] = {}

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.emits.append((event_type, data))

    def on(self, event_type: str, callback: Any) -> None:
        self.handlers[event_type] = callback


class _RecordingController:
    """Records the TVChartStateMixin methods each tool calls.

    Method names + signatures mirror pywry's ``TVChartStateMixin`` so a
    drift between the tools and the real protocol API surfaces as a
    ``TypeError`` here.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.state_response: dict[str, Any] | None = {
            "chartId": "openbb-canvas-tvchart",
            "interval": "1d",
            "chartType": "Candles",
            "series": {"main": {"type": "Candlestick"}},
            "rawData": [{"time": 1}, {"time": 2}],
            "indicators": [],
        }
        self.indicators_response: dict[str, Any] | None = {
            "indicators": [{"seriesId": "ind_rsi_1", "name": "RSI", "period": 14}]
        }

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.calls.append(("emit", (event, payload)))

    def update_series(self, data, *, series_id=None, fit_content=True) -> None:
        self.calls.append(("update_series", (data, series_id, fit_content)))

    def update_bar(self, bar, *, series_id=None) -> None:
        self.calls.append(("update_bar", (bar, series_id)))

    def add_indicator(
        self,
        indicator_data,
        *,
        series_id="indicator",
        series_type="Line",
        series_options=None,
    ) -> None:
        self.calls.append(
            ("add_indicator", (indicator_data, series_id, series_type, series_options))
        )

    def remove_indicator(self, series_id) -> None:
        self.calls.append(("remove_indicator", series_id))

    def add_builtin_indicator(
        self,
        name,
        *,
        period=None,
        color=None,
        source=None,
        method=None,
        multiplier=None,
        ma_type=None,
        offset=None,
    ) -> None:
        self.calls.append(
            (
                "add_builtin_indicator",
                {
                    "name": name,
                    "period": period,
                    "color": color,
                    "source": source,
                    "method": method,
                    "multiplier": multiplier,
                    "ma_type": ma_type,
                    "offset": offset,
                },
            )
        )

    def add_volume_profile(self, mode, **kwargs) -> None:
        self.calls.append(("add_volume_profile", (mode, kwargs)))

    def remove_builtin_indicator(self, series_id) -> None:
        self.calls.append(("remove_builtin_indicator", series_id))

    def add_marker(self, markers, *, series_id=None) -> None:
        self.calls.append(("add_marker", (markers, series_id)))

    def add_price_line(
        self, price, *, color="#2196F3", line_width=1, title="", series_id=None
    ) -> None:
        self.calls.append(
            ("add_price_line", (price, color, line_width, title, series_id))
        )

    def set_visible_range(self, from_time, to_time) -> None:
        self.calls.append(("set_visible_range", (from_time, to_time)))

    def fit_content(self) -> None:
        self.calls.append(("fit_content", None))

    def apply_chart_options(
        self, *, chart_options=None, series_options=None, series_id=None
    ) -> None:
        self.calls.append(
            ("apply_chart_options", (chart_options, series_options, series_id))
        )

    def list_indicators_sync(self, timeout: float = 10.0):
        return self.indicators_response

    def get_state(self, timeout: float = 10.0):
        return self.state_response


class _RecordingCanvas:
    """LiveCanvas stub that records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.tvchart: _RecordingController | None = None

    def tvchart_controller(self) -> _RecordingController | None:
        return self.tvchart

    def show_html(self, html: str, *, title: str | None = None) -> None:
        self.calls.append(("html", (html, title)))

    def show_markdown(self, text: str, *, title: str | None = None) -> None:
        self.calls.append(("markdown", (text, title)))

    def show_plotly(self, figure: dict, *, title: str | None = None) -> None:
        self.calls.append(("plotly", (figure, title)))

    def show_table(self, rows, *, title=None, columns=None) -> None:
        self.calls.append(("table", (rows, title, columns)))

    def show_tvchart(
        self,
        datasets,
        *,
        selected_interval=None,
        series_type="Candlestick",
        title=None,
        chart_options=None,
        height=None,
    ) -> None:
        self.calls.append(
            ("tvchart", (datasets, selected_interval, series_type, title))
        )

    def show_tvchart_symbol(
        self,
        symbol,
        *,
        intervals=None,
        selected_interval=None,
        series_type="Candlestick",
        title=None,
        chart_options=None,
        height=None,
    ) -> None:
        self.calls.append(
            ("tvchart_symbol", (symbol, intervals, selected_interval, series_type))
        )

    def show_image(self, src: str, *, title: str | None = None) -> None:
        self.calls.append(("image", (src, title)))

    def show_document(
        self,
        *,
        src: str,
        mime: str,
        filename: str | None = None,
        title: str | None = None,
        text: str | None = None,
    ) -> None:
        self.calls.append(("document", (src, mime, filename, title, text)))

    def clear(self) -> None:
        self.calls.append(("clear", None))


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u1", scopes=("agent:query",)),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_set_get_reset() -> None:
    assert canvas_registry.get_canvas() is None
    stub = _RecordingCanvas()
    canvas_registry.set_canvas(stub)
    assert canvas_registry.get_canvas() is stub
    assert isinstance(stub, LiveCanvas)
    canvas_registry.reset_canvas()
    assert canvas_registry.get_canvas() is None


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def test_tools_error_when_no_canvas_bound() -> None:
    for result in (
        canvas_html("<p>x</p>"),
        canvas_markdown("# x"),
        canvas_plotly({"data": []}),
        canvas_table([{"a": 1}]),
        canvas_tvchart(
            {"1d": [{"time": 1, "open": 1, "high": 2, "low": 0, "close": 1}]}
        ),
        await canvas_image("https://x/y.png"),
        await canvas_document(src="https://x/y.pdf", mime="application/pdf"),
        canvas_clear(),
    ):
        assert result.startswith("error: no live canvas")


async def test_tools_dispatch_to_bound_canvas() -> None:
    stub = _RecordingCanvas()
    canvas_registry.set_canvas(stub)

    assert "html rendered" in canvas_html("<p>x</p>", title="T")
    assert "markdown rendered" in canvas_markdown("# hi")
    assert "chart rendered" in canvas_plotly({"data": [{"y": [1]}]})
    assert "2 row(s)" in canvas_table([{"a": 1}, {"a": 2}])
    assert "toolbar intervals [1d]" in canvas_tvchart(
        {"1d": [{"time": 1, "open": 1, "high": 2, "low": 0, "close": 1}]}
    )
    assert "image rendered" in await canvas_image("https://x/y.png")
    assert canvas_clear() == "canvas cleared"

    kinds = [kind for kind, _ in stub.calls]
    assert kinds == ["html", "markdown", "plotly", "table", "tvchart", "image", "clear"]
    assert stub.calls[0][1] == ("<p>x</p>", "T")


def test_canvas_tvchart_validates_inputs() -> None:
    canvas_registry.set_canvas(_RecordingCanvas())
    bars = [{"time": 1, "open": 1, "high": 2, "low": 0, "close": 1}]
    assert canvas_tvchart({}).startswith("error: datasets is empty")
    assert "non-empty list" in canvas_tvchart({"1d": []})
    assert "non-empty list" in canvas_tvchart({"1d": [{"open": 1}]})  # no time
    out = canvas_tvchart({"1d": bars}, series_type="renko")
    assert out.startswith("error: series_type")
    out = canvas_tvchart({"1d": bars}, selected_interval="1w")
    assert out.startswith("error: selected_interval")


def test_canvas_tvchart_passes_datasets_through() -> None:
    stub = _RecordingCanvas()
    canvas_registry.set_canvas(stub)
    datasets = {
        "1d": [
            {
                "time": "2024-01-02",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
            }
        ],
        "1w": [
            {
                "time": "2024-01-05",
                "open": 1,
                "high": 3,
                "low": 0.5,
                "close": 2.5,
                "volume": 400,
            }
        ],
    }
    out = canvas_tvchart(
        datasets,
        selected_interval="1w",
        series_type="candlestick",  # lowercase normalised
        title="AAPL",
    )
    assert "toolbar intervals [1d, 1w]" in out
    kind, (got, selected, stype, title) = stub.calls[0]
    assert kind == "tvchart"
    assert got == datasets
    assert selected == "1w"
    assert stype == "Candlestick"
    assert title == "AAPL"


# ---------------------------------------------------------------------------
# Full TVChart protocol tools
# ---------------------------------------------------------------------------


def _mounted_chart() -> tuple[_RecordingCanvas, _RecordingController]:
    stub = _RecordingCanvas()
    stub.tvchart = _RecordingController()
    canvas_registry.set_canvas(stub)
    return stub, stub.tvchart


def test_tvchart_protocol_tool_errors_without_any_canvas() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_get_state,
    )

    assert canvas_tvchart_get_state().startswith("error: no live canvas")


def test_canvas_binds_and_exposes_tvchart_controller() -> None:
    built: list[tuple[Any, str]] = []

    def factory(handle: Any, chart_id: str) -> str:
        built.append((handle, chart_id))
        return "CONTROLLER"

    handle = _FakeHandle()
    canvas = PyWryCanvas(handle, tvchart_controller_factory=factory)
    assert canvas.tvchart_controller() is None
    canvas.show_tvchart({"1d": _BARS_1D})
    canvas.show_tvchart({"1w": _BARS_1W})  # bound once, reused
    assert canvas.tvchart_controller() == "CONTROLLER"
    assert len(built) == 1
    assert built[0][1] == "openbb-canvas-tvchart"


def test_canvas_controller_factory_failure_is_non_fatal() -> None:
    def bad_factory(handle: Any, chart_id: str) -> str:
        raise RuntimeError("mixin unavailable")

    canvas = PyWryCanvas(_FakeHandle(), tvchart_controller_factory=bad_factory)
    canvas.show_tvchart({"1d": _BARS_1D})  # must not raise
    assert canvas.tvchart_controller() is None


# Every protocol tool delegates to the matching TVChartStateMixin method.
# These tests assert the delegation (method + args), not any payload the
# library builds — the library owns the wire format.


def test_tvchart_protocol_tools_error_without_chart() -> None:
    from openbb_agent_server.plugins.tools import pywry_canvas as mod

    canvas_registry.set_canvas(_RecordingCanvas())  # canvas, no chart
    bars = [{"time": 1, "value": 2}]
    results = [
        mod.canvas_tvchart_update(bars),
        mod.canvas_tvchart_stream_bar({"time": 1, "close": 2}),
        mod.canvas_tvchart_add_series("s1", bars),
        mod.canvas_tvchart_remove_series("s1"),
        mod.canvas_tvchart_add_indicator("RSI"),
        mod.canvas_tvchart_remove_indicator("ind_x"),
        mod.canvas_tvchart_list_indicators(),
        mod.canvas_tvchart_add_volume_profile(),
        mod.canvas_tvchart_add_markers([{"time": 1}]),
        mod.canvas_tvchart_add_price_line(100.0),
        mod.canvas_tvchart_set_visible_range(1, 2),
        mod.canvas_tvchart_fit_content(),
        mod.canvas_tvchart_apply_options(chart_options={"grid": {}}),
        mod.canvas_tvchart_get_state(),
    ]
    assert all(r.startswith("error: no TVChart is mounted") for r in results)


def test_tvchart_update_and_stream() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_stream_bar,
        canvas_tvchart_update,
    )

    _, ctrl = _mounted_chart()
    bars = [{"time": 1, "open": 1, "high": 2, "low": 0, "close": 1, "volume": 5}]
    assert "1 bars" in canvas_tvchart_update(bars, series_id="main")
    assert canvas_tvchart_update([]).startswith("error: bars is empty")
    out = canvas_tvchart_stream_bar(
        {"time": 2, "open": 1, "high": 2, "low": 0, "close": 2}
    )
    assert "time=2" in out
    assert canvas_tvchart_stream_bar({"close": 2}).startswith("error: bar must")
    # update_series gets the bars verbatim — no Python-side volume split.
    assert ctrl.calls[0] == ("update_series", (bars, "main", True))
    assert ctrl.calls[1][0] == "update_bar"
    assert ctrl.calls[1][1][0]["close"] == 2


def test_tvchart_add_remove_series() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_add_series,
        canvas_tvchart_remove_series,
    )

    _, ctrl = _mounted_chart()
    bars = [{"time": 1, "value": 2}]
    assert canvas_tvchart_add_series("main", bars).startswith("error: series_id")
    assert canvas_tvchart_add_series("s1", []).startswith("error: bars is empty")
    out = canvas_tvchart_add_series("spy", bars, series_type="Line")
    assert "spy" in out
    # Delegates to mixin.add_indicator(bars, series_id=, series_type=, ...).
    name, (data, series_id, series_type, opts) = ctrl.calls[0]
    assert name == "add_indicator"
    assert (data, series_id, series_type, opts) == (bars, "spy", "Line", {})

    assert canvas_tvchart_remove_series("main").startswith("error: cannot remove")
    assert "removed" in canvas_tvchart_remove_series("spy")
    assert ctrl.calls[-1] == ("remove_indicator", "spy")


def test_tvchart_add_indicator_name_dispatched_uses_mixin() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_add_indicator,
    )

    _, ctrl = _mounted_chart()
    # Name-dispatched indicators go through the mixin convenience method.
    assert "RSI" in canvas_tvchart_add_indicator("RSI", period=14)
    name, kw = ctrl.calls[0]
    assert name == "add_builtin_indicator"
    assert kw["name"] == "RSI"
    assert kw["period"] == 14

    canvas_tvchart_add_indicator(
        "Bollinger Bands", period=20, multiplier=2.0, ma_type="SMA", offset=1
    )
    _, bb = ctrl.calls[1]
    assert bb["name"] == "Bollinger Bands"
    assert bb["multiplier"] == 2.0
    assert bb["ma_type"] == "SMA"
    assert bb["offset"] == 1

    assert canvas_tvchart_add_indicator("").startswith("error: name is required")


def test_tvchart_add_indicator_moving_average_carries_catalog_key() -> None:
    """Moving Average dispatches by key 'moving-average-ex' — the tool
    must pass it via the documented event field (the mixin omits it)."""
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_add_indicator,
    )

    _, ctrl = _mounted_chart()
    out = canvas_tvchart_add_indicator(
        "Moving Average",
        period=21,
        method="EMA",
        source="close",
        color="#f00",
        multiplier=2.5,
        ma_type="WMA",
        offset=3,
    )
    assert "Moving Average" in out
    kind, (evt_name, body) = ctrl.calls[0]
    assert kind == "emit"
    assert evt_name == "tvchart:add-indicator"
    assert body["name"] == "Moving Average"
    assert body["key"] == "moving-average-ex"  # the load-bearing field
    assert body["period"] == 21
    assert body["method"] == "EMA"
    assert body["source"] == "close"
    assert body["color"] == "#f00"
    # Remaining optional fields ride the documented event payload too.
    assert body["multiplier"] == 2.5
    assert body["maType"] == "WMA"
    assert body["offset"] == 3


def test_tvchart_indicator_lifecycle_and_listing() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_list_indicators,
        canvas_tvchart_remove_indicator,
    )

    _, ctrl = _mounted_chart()
    listing = canvas_tvchart_list_indicators()
    assert "ind_rsi_1" in listing
    ctrl.indicators_response = None
    assert canvas_tvchart_list_indicators().startswith("error: no response")
    assert "removed" in canvas_tvchart_remove_indicator("ind_rsi_1")
    assert ctrl.calls[-1] == ("remove_builtin_indicator", "ind_rsi_1")


def test_tvchart_volume_profile_carries_catalog_key() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_add_volume_profile,
    )

    _, ctrl = _mounted_chart()
    assert "mode must" in canvas_tvchart_add_volume_profile(mode="weird")
    assert "placement" in canvas_tvchart_add_volume_profile(placement="top")
    out = canvas_tvchart_add_volume_profile(
        mode="fixed",
        bucket_count=30,
        from_index=0,
        to_index=99,
        up_color="#0f0",
        down_color="#f00",
        poc_color="#ff0",
    )
    assert "fixed" in out
    # Volume Profile dispatches by catalog key — passed via the event.
    kind, (evt_name, body) = ctrl.calls[0]
    assert kind == "emit"
    assert evt_name == "tvchart:add-indicator"
    assert body["name"] == "Volume Profile Fixed Range"
    assert body["key"] == "volume-profile-fixed"
    assert body["period"] == 30
    assert body["fromIndex"] == 0
    assert body["toIndex"] == 99
    assert body["upColor"] == "#0f0"
    assert body["downColor"] == "#f00"
    assert body["pocColor"] == "#ff0"

    # Visible mode uses the visible key.
    canvas_tvchart_add_volume_profile(mode="visible")
    assert ctrl.calls[-1][1][1]["key"] == "volume-profile-visible"


def test_tvchart_markers_minimal_validation_and_delegation() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_add_markers,
    )

    _, ctrl = _mounted_chart()
    assert canvas_tvchart_add_markers([]).startswith("error: markers is empty")
    assert "'time'" in canvas_tvchart_add_markers([{"position": "aboveBar"}])
    # Marker dicts pass through to the library verbatim (time only checked).
    markers = [
        {"time": 2, "position": "belowBar", "shape": "arrowUp", "text": "B"},
        {"time": 1, "position": "aboveBar", "shape": "arrowDown"},
    ]
    assert "2 on 'main'" in canvas_tvchart_add_markers(markers)
    name, (sent, series_id) = ctrl.calls[0]
    assert name == "add_marker"
    assert series_id is None
    assert sent is markers


def test_tvchart_price_line_view_and_options() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_add_price_line,
        canvas_tvchart_apply_options,
        canvas_tvchart_fit_content,
        canvas_tvchart_set_visible_range,
    )

    _, ctrl = _mounted_chart()
    assert "187.5" in canvas_tvchart_add_price_line(
        187.5, color="#f00", line_width=2, title="resistance", series_id="main"
    )
    # Delegates to mixin.add_price_line — no lineStyle (not in the API).
    assert ctrl.calls[0] == ("add_price_line", (187.5, "#f00", 2, "resistance", "main"))

    canvas_tvchart_set_visible_range(1_700_000_000, 1_700_600_000)
    assert ctrl.calls[-1] == ("set_visible_range", (1_700_000_000, 1_700_600_000))

    canvas_tvchart_fit_content()
    assert ctrl.calls[-1] == ("fit_content", None)

    assert canvas_tvchart_apply_options().startswith("error: provide")
    canvas_tvchart_apply_options(
        chart_options={"grid": {"vertLines": {"visible": False}}},
        series_options={"upColor": "#0f0"},
        series_id="main",
    )
    assert ctrl.calls[-1][0] == "apply_chart_options"


def test_tvchart_get_state_trims_raw_data() -> None:
    from openbb_agent_server.plugins.tools.pywry_canvas import (
        canvas_tvchart_get_state,
    )

    _, ctrl = _mounted_chart()
    state = canvas_tvchart_get_state()
    assert '"interval": "1d"' in state
    assert '"barCount": 2' in state
    assert "rawData" not in state  # trimmed to a count
    ctrl.state_response = None
    assert canvas_tvchart_get_state().startswith("error: no response")


def test_canvas_plotly_rejects_non_figure() -> None:
    canvas_registry.set_canvas(_RecordingCanvas())
    assert canvas_plotly({"layout": {}}).startswith("error: figure must")


def test_canvas_table_rejects_empty_rows() -> None:
    canvas_registry.set_canvas(_RecordingCanvas())
    assert canvas_table([]).startswith("error: rows is empty")


async def test_canvas_image_rejects_unsafe_src() -> None:
    canvas_registry.set_canvas(_RecordingCanvas())
    js = await canvas_image("javascript:alert(1)")
    assert js.startswith("error:") and "is not a data: URI" in js
    http = await canvas_image("http://insecure/x.png")
    assert http.startswith("error:") and "http:// is not allowed" in http


# ---------------------------------------------------------------------------
# Tool source
# ---------------------------------------------------------------------------


async def test_tool_source_skips_without_canvas() -> None:
    source = PyWryCanvasToolSource()
    assert await source.tools(_ctx(), {}) == []


async def test_tool_source_registers_all_tools_with_canvas() -> None:
    canvas_registry.set_canvas(_RecordingCanvas())
    source = PyWryCanvasToolSource()
    tools = await source.tools(_ctx(), {})
    names = {t.name for t in tools}
    assert names == {
        "canvas_html",
        "canvas_markdown",
        "canvas_plotly",
        "canvas_table",
        "canvas_tvchart",
        "canvas_tvchart_symbol",
        "canvas_tvchart_update",
        "canvas_tvchart_stream_bar",
        "canvas_tvchart_add_series",
        "canvas_tvchart_remove_series",
        "canvas_tvchart_add_indicator",
        "canvas_tvchart_remove_indicator",
        "canvas_tvchart_list_indicators",
        "canvas_tvchart_add_volume_profile",
        "canvas_tvchart_add_markers",
        "canvas_tvchart_add_price_line",
        "canvas_tvchart_set_visible_range",
        "canvas_tvchart_fit_content",
        "canvas_tvchart_apply_options",
        "canvas_tvchart_get_state",
        "canvas_image",
        "canvas_document",
        "canvas_clear",
    }


async def test_tool_invocation_through_langchain_layer() -> None:
    stub = _RecordingCanvas()
    canvas_registry.set_canvas(stub)
    source = PyWryCanvasToolSource()
    tools = {t.name: t for t in await source.tools(_ctx(), {})}
    out = tools["canvas_table"].invoke(
        {"rows": [{"a": 1}], "title": "Numbers", "columns": ["a"]}
    )
    assert "1 row(s)" in out
    assert stub.calls == [("table", ([{"a": 1}], "Numbers", ["a"]))]
    assert tools["canvas_clear"].invoke({}) == "canvas cleared"


# ---------------------------------------------------------------------------
# Page builder + table renderer
# ---------------------------------------------------------------------------


def test_build_canvas_html_contains_container() -> None:
    html = build_canvas_html()
    assert f'id="{CANVAS_ELEMENT_ID}"' in html
    assert "obb-canvas-root" in html


def test_render_table_html_escapes_and_orders_columns() -> None:
    html = render_table_html(
        [{"name": "<b>x</b>", "n": 1, "extra": {"k": "v"}}],
        columns=["n", "name", "extra", "missing"],
    )
    assert "<th>n</th><th>name</th>" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "{&quot;k&quot;: &quot;v&quot;}" in html
    # Missing keys and None render as empty cells.
    assert "<td></td>" in html


def test_render_table_html_caps_rows() -> None:
    rows = [{"i": i} for i in range(2001)]
    html = render_table_html(rows)
    assert "Showing 2,000 of 2,001 rows." in html


def test_render_table_html_empty_rows() -> None:
    assert "<tbody></tbody>" in render_table_html([])


# ---------------------------------------------------------------------------
# PyWryCanvas over a fake handle
#
# The fake handle deliberately has NO eval_js and NO set_trait: every
# render must work through emit() alone, the one channel uniform across
# PyWry's native / anywidget / inline rendering paths.
# ---------------------------------------------------------------------------


def test_show_html_emits_set_html_with_title() -> None:
    handle = _FakeHandle()
    canvas = PyWryCanvas(handle)
    canvas.show_html("<p>hello</p>", title="Greeting <1>")
    event, data = handle.emits[0]
    assert event == "obb-canvas:set-html"
    assert data["id"] == CANVAS_ELEMENT_ID
    assert "Greeting &lt;1&gt;" in data["html"]
    assert "<p>hello</p>" in data["html"]


def test_canvas_never_requires_eval_js_or_set_trait() -> None:
    """A handle with only emit() supports every render operation."""
    handle = _FakeHandle()
    assert not hasattr(handle, "eval_js")
    assert not hasattr(handle, "set_trait")
    canvas = PyWryCanvas(
        handle,
        plotly_assets=lambda: "/* plotly */",
        tvchart_assets=lambda: "/* lwc + engine */",
    )
    canvas.show_html("<p>x</p>")
    canvas.show_markdown("# hi")
    canvas.show_plotly({"data": []})
    canvas.show_table([{"a": 1}])
    canvas.show_tvchart(
        {"1d": [{"time": 1, "open": 1, "high": 2, "low": 0, "close": 1}]}
    )
    canvas.show_image("https://x/y.png")
    canvas.clear()
    # 6 single-emit renders + tvchart (container + tvchart:create) +
    # 2 lazy asset loads.
    assert len(handle.emits) == 10


def test_show_markdown_emits_text_payload() -> None:
    handle = _FakeHandle()
    PyWryCanvas(handle).show_markdown("# Title\n*hi*", title="Doc")
    event, data = handle.emits[0]
    assert event == "obb-canvas:markdown"
    assert data["text"] == "# Title\n*hi*"
    assert data["title"] == "Doc"
    assert data["id"] == CANVAS_ELEMENT_ID


def test_show_plotly_loads_assets_once_then_plots() -> None:
    handle = _FakeHandle()
    calls = {"n": 0}

    def assets() -> str:
        calls["n"] += 1
        return "/* plotly bundle */"

    canvas = PyWryCanvas(handle, plotly_assets=assets)
    canvas.show_plotly({"data": [{"y": [1, 2]}]}, title="Chart")
    canvas.show_plotly({"data": []})
    assert calls["n"] == 1
    events = [e for e, _ in handle.emits]
    assert events == [
        "obb-canvas:load-assets",
        "obb-canvas:plotly",
        "obb-canvas:plotly",
    ]
    assert handle.emits[0][1]["scripts"] == ["/* plotly bundle */"]
    assert handle.emits[1][1]["figure"] == {"data": [{"y": [1, 2]}]}
    assert handle.emits[1][1]["title"] == "Chart"


def test_show_plotly_without_assets_still_emits_figure() -> None:
    handle = _FakeHandle()
    PyWryCanvas(handle).show_plotly({"data": []})
    events = [e for e, _ in handle.emits]
    assert events == ["obb-canvas:plotly"]


def test_show_plotly_sanitizes_non_json_leaves() -> None:
    import datetime

    handle = _FakeHandle()
    when = datetime.date(2026, 6, 12)
    PyWryCanvas(handle).show_plotly({"data": [{"x": [when], "y": [1]}]})
    figure = handle.emits[0][1]["figure"]
    assert figure["data"][0]["x"] == ["2026-06-12"]


def test_show_plotly_asset_failure_is_non_fatal_and_retried() -> None:
    def boom() -> str:
        raise RuntimeError("no bundle")

    handle = _FakeHandle()
    canvas = PyWryCanvas(handle, plotly_assets=boom)
    canvas.show_plotly({"data": []})
    # The figure still went out; assets retried on the next call.
    assert [e for e, _ in handle.emits] == ["obb-canvas:plotly"]
    assert not canvas._plotly_injected
    canvas.show_plotly({"data": []})
    assert [e for e, _ in handle.emits].count("obb-canvas:plotly") == 2


_BARS_1D = [
    {"time": 1700000000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
    {
        "time": 1700086400,
        "open": 1.5,
        "high": 2.5,
        "low": 1.0,
        "close": 2.0,
        "volume": 20,
    },
]
_BARS_1W = [
    {"time": 1700000000, "open": 1, "high": 3, "low": 0.5, "close": 2.5, "volume": 90},
]


def test_show_tvchart_drives_the_pywry_protocol() -> None:
    """show_tvchart mounts chrome + container and emits tvchart:create."""
    from openbb_agent_server.acp.canvas import TVCHART_CHART_ID

    handle = _FakeHandle()
    calls = {"assets": 0, "chrome": []}

    def assets() -> str:
        calls["assets"] += 1
        return "/* lwc + engine + toolbars */"

    def chrome(container: str, intervals, selected) -> str:
        calls["chrome"].append((intervals, selected))
        return f'<div class="pywry-wrapper-header">TOOLBARS{container}</div>'

    canvas = PyWryCanvas(handle, tvchart_assets=assets, tvchart_chrome=chrome)
    datasets = {"1d": _BARS_1D, "1w": _BARS_1W}
    canvas.show_tvchart(
        datasets,
        selected_interval="1w",
        title="SPY",
        chart_options={"crosshair": {"mode": 1}},
        height="600px",
    )
    canvas.show_tvchart(datasets)
    assert calls["assets"] == 1
    assert calls["chrome"][0] == (["1d", "1w"], "1w")
    events = [e for e, _ in handle.emits]
    assert events == [
        "obb-canvas:load-assets",
        "obb-canvas:set-html",
        "tvchart:create",
        "obb-canvas:set-html",
        "tvchart:create",
    ]
    # The page carries the toolbar chrome around the engine container.
    page_html = handle.emits[1][1]["html"]
    assert "TOOLBARS" in page_html
    assert f'id="{TVCHART_CHART_ID}"' in page_html
    assert "pywry-tvchart-container" in page_html
    assert "height:600px" in page_html
    # The protocol payload, exactly as app.show_tvchart shapes it.
    payload = handle.emits[2][1]
    assert payload["containerId"] == TVCHART_CHART_ID
    assert payload["chartId"] == TVCHART_CHART_ID
    assert payload["series"] == [
        {
            "seriesId": "main",
            "seriesType": "Candlestick",
            "bars": _BARS_1W,
            "volume": [],
            "seriesOptions": {},
        }
    ]
    assert payload["chartOptions"] == {"crosshair": {"mode": 1}}
    assert payload["chartKind"] == "default"
    assert payload["useDatafeed"] is False
    assert payload["interval"] == "1w"
    assert payload["storage"] == {"backend": "localStorage"}
    assert payload["title"] == "SPY"


def test_show_tvchart_answers_interval_data_requests() -> None:
    """The canvas serves tvchart:data-request like the pywry demo does."""
    from openbb_agent_server.acp.canvas import TVCHART_CHART_ID

    handle = _FakeHandle()
    canvas = PyWryCanvas(handle)
    canvas.show_tvchart({"1d": _BARS_1D, "1w": _BARS_1W})
    handler = handle.handlers["tvchart:data-request"]

    handler({"chartId": TVCHART_CHART_ID, "seriesId": "main", "resolution": "1w"})
    event, response = handle.emits[-1]
    assert event == "tvchart:data-response"
    assert response == {
        "chartId": TVCHART_CHART_ID,
        "seriesId": "main",
        "bars": _BARS_1W,
        "fitContent": True,
        "interval": "1w",
    }

    # Unknown interval falls back to the first dataset.
    handler({"chartId": TVCHART_CHART_ID, "interval": "5m"})
    assert handle.emits[-1][1]["interval"] == "1d"
    assert handle.emits[-1][1]["bars"] == _BARS_1D

    # Foreign charts and compare-series requests are left unanswered.
    before = len(handle.emits)
    handler({"chartId": "someone-elses-chart", "interval": "1d"})
    handler({"chartId": TVCHART_CHART_ID, "seriesId": "compare-1", "interval": "1d"})
    assert len(handle.emits) == before


def test_tvchart_data_request_edge_paths() -> None:
    """Empty stash and emit failures are swallowed, never raised."""
    from openbb_agent_server.acp.canvas import TVCHART_CHART_ID

    handle = _FakeHandle()
    canvas = PyWryCanvas(handle)
    canvas.show_tvchart({"1d": _BARS_1D})
    handler = handle.handlers["tvchart:data-request"]

    # Stash cleared (e.g. canvas re-used) -> request silently ignored.
    canvas._tvchart_datasets = {}
    before = len(handle.emits)
    handler({"chartId": TVCHART_CHART_ID, "interval": "1d"})
    assert len(handle.emits) == before

    # A broken emit during the response must not propagate into the
    # widget callback thread.
    canvas._tvchart_datasets = {"1d": _BARS_1D}

    def _boom_emit(event_type, data):
        raise RuntimeError("bridge gone")

    canvas._handle = type("H", (), {"emit": staticmethod(_boom_emit)})()
    handler({"chartId": TVCHART_CHART_ID, "interval": "1d"})  # no raise


def test_canvas_app_tvchart_helpers_use_real_pywry() -> None:
    """The host callables build real pywry assets + toolbar chrome."""
    pytest.importorskip("pywry.tvchart")
    from openbb_agent_server.acp.canvas_app import _tvchart_assets, _tvchart_chrome

    js = _tvchart_assets()
    assert "LightweightCharts" in js
    assert "initToolbarHandlers" in js  # toolbar handlers ship too

    html = _tvchart_chrome('<div id="c"></div>', ["1d", "1w"], "1d")
    assert '<div id="c"></div>' in html
    assert "tvchart-interval-btn" in html  # interval dropdown
    assert "tvchart-header" in html  # header toolbar
    assert "tvchart-legend" in html  # OHLC legend overlay


def test_show_tvchart_handler_registered_once_and_handles_no_on() -> None:
    handle = _FakeHandle()
    canvas = PyWryCanvas(handle)
    canvas.show_tvchart({"1d": _BARS_1D})
    canvas.show_tvchart({"1w": _BARS_1W})
    assert list(handle.handlers) == ["tvchart:data-request"]

    class _NoOnHandle:
        def __init__(self) -> None:
            self.emits: list = []

        def emit(self, event_type, data) -> None:
            self.emits.append((event_type, data))

    # A handle without .on degrades gracefully (chart renders, no
    # interval answering).
    PyWryCanvas(_NoOnHandle()).show_tvchart({"1d": _BARS_1D})


def test_show_tvchart_chrome_failure_falls_back_to_bare_chart() -> None:
    def bad_chrome(container, intervals, selected) -> str:
        raise RuntimeError("no chrome")

    handle = _FakeHandle()
    canvas = PyWryCanvas(handle, tvchart_chrome=bad_chrome)
    canvas.show_tvchart({"1d": _BARS_1D})
    page_html = handle.emits[0][1]["html"]
    assert "pywry-tvchart-container" in page_html  # bare chart still mounts


def test_show_tvchart_rejects_empty_datasets() -> None:
    with pytest.raises(ValueError, match="at least one interval"):
        PyWryCanvas(_FakeHandle()).show_tvchart({})


def test_show_tvchart_sanitizes_non_json_leaves() -> None:
    import datetime

    handle = _FakeHandle()
    when = datetime.date(2024, 1, 2)
    PyWryCanvas(handle).show_tvchart(
        {"1d": [{"time": when, "open": 1, "high": 2, "low": 0, "close": 1}]}
    )
    payload = handle.emits[1][1]
    assert payload["series"][0]["bars"][0]["time"] == "2024-01-02"


def test_show_tvchart_asset_failure_is_non_fatal_and_retried() -> None:
    def boom() -> str:
        raise RuntimeError("no bundle")

    handle = _FakeHandle()
    canvas = PyWryCanvas(handle, tvchart_assets=boom)
    canvas.show_tvchart({"1d": _BARS_1D})
    assert [e for e, _ in handle.emits].count("tvchart:create") == 1
    assert not canvas._tvchart_injected
    canvas.show_tvchart({"1d": _BARS_1D})
    assert [e for e, _ in handle.emits].count("tvchart:create") == 2


def test_show_table_and_image_and_clear() -> None:
    handle = _FakeHandle()
    canvas = PyWryCanvas(handle)
    canvas.show_table([{"a": 1}], title="T", columns=["a"])
    canvas.show_image('https://x/y.png" onerror="x', title="Img")
    canvas.clear()
    assert all(event == "obb-canvas:set-html" for event, _ in handle.emits)
    table_html = handle.emits[0][1]["html"]
    assert "obb-canvas-table" in table_html
    img_html = handle.emits[1][1]["html"]
    assert "&quot;" in img_html  # attribute-escaped
    assert handle.emits[2][1]["html"].startswith('<div class="obb-canvas-empty">')


def test_custom_element_id_is_used() -> None:
    handle = _FakeHandle()
    PyWryCanvas(handle, element_id="my-target").show_html("<p>x</p>")
    assert handle.emits[0][1]["id"] == "my-target"


def test_anywidget_handle_receives_bootstrap_via_asset_trait() -> None:
    """Handles exposing set_trait (anywidget) get the page handlers
    pushed through _asset_js, where initial-content scripts are inert."""
    from openbb_agent_server.acp.canvas import CANVAS_BOOTSTRAP_JS

    class _AnyWidgetHandle(_FakeHandle):
        def __init__(self) -> None:
            super().__init__()
            self.traits: list[tuple[str, str]] = []

        def set_trait(self, name: str, value: str) -> None:
            self.traits.append((name, value))

    handle = _AnyWidgetHandle()
    PyWryCanvas(handle)
    assert handle.traits == [("_asset_js", CANVAS_BOOTSTRAP_JS)]


def test_bootstrap_trait_failure_is_non_fatal() -> None:
    class _BrokenAnyWidget(_FakeHandle):
        def set_trait(self, name: str, value: str) -> None:
            raise RuntimeError("widget closed")

    canvas = PyWryCanvas(_BrokenAnyWidget())
    canvas.show_html("<p>still works</p>")  # must not raise


def test_real_chat_widget_receives_bootstrap_trait() -> None:
    """Against the REAL anywidget class: _asset_js lands on PyWryChatWidget."""
    pytest.importorskip("anywidget")
    from pywry.widget import PyWryChatWidget

    from openbb_agent_server.acp.canvas import CANVAS_BOOTSTRAP_JS

    widget = PyWryChatWidget()
    PyWryCanvas(widget)
    assert widget._asset_js == CANVAS_BOOTSTRAP_JS


def test_plain_pywry_widget_degrades_with_clear_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Base PyWryWidget has no _asset_js trait (verified live) — the
    canvas must not raise, and the warning names the supported widget."""
    pytest.importorskip("anywidget")
    from pywry.widget import PyWryWidget

    widget = PyWryWidget()
    with caplog.at_level("WARNING"):
        canvas = PyWryCanvas(widget)
    assert any("PyWryChatWidget" in r.message for r in caplog.records)
    # Rendering still emits — a future pywry adding the trait Just Works.
    canvas.show_html("<p>x</p>")


def test_bootstrap_script_ships_in_canvas_page() -> None:
    from openbb_agent_server.acp.canvas import CANVAS_BOOTSTRAP_JS

    html = build_canvas_html()
    assert "<script>" in html
    assert "window.__obbCanvas" in html
    # The page handler covers every obb-canvas event the Python side
    # emits. TV charts are NOT here by design — they go through pywry's
    # own tvchart engine protocol (tvchart:create), not a custom
    # canvas renderer.
    for event in (
        "obb-canvas:set-html",
        "obb-canvas:markdown",
        "obb-canvas:plotly",
        "obb-canvas:load-assets",
    ):
        assert event in CANVAS_BOOTSTRAP_JS
    # The bootstrap also installs initToolbarHandlers wiring and the
    # ws-bridge dual-dispatch shim so the tvchart engine's client-side
    # toolbar controls (which call pywry.emit) reach their local
    # handlers on the browser path.
    assert "initToolbarHandlers" in CANVAS_BOOTSTRAP_JS
    assert "__obbDualEmit" in CANVAS_BOOTSTRAP_JS
    assert "_fire" in CANVAS_BOOTSTRAP_JS


def test_bootstrap_dual_dispatch_shim_semantics() -> None:
    """The shim must: send to the host, locally dispatch only when a
    handler exists, guard the ws-bridge shape, and apply once."""
    from openbb_agent_server.acp.canvas import CANVAS_BOOTSTRAP_JS

    js = CANVAS_BOOTSTRAP_JS
    assert "b.__obbDualEmit" in js  # idempotency guard
    assert "typeof b._fire !== 'function' || !b._handlers" in js  # shape guard
    assert "(b._handlers[type] || []).length > 0" in js  # only when handler exists
    assert "orig(type, data)" in js  # still sends to the host


# ---------------------------------------------------------------------------
# Settings helper + agent integration
# ---------------------------------------------------------------------------


def test_with_canvas_tools_appends_once() -> None:
    from openbb_agent_server.acp.canvas_app import with_canvas_tools

    settings = AgentServerSettings(model_provider="fake", tool_sources=("artifacts",))
    updated = with_canvas_tools(settings)
    assert updated.tool_sources == ("artifacts", "pywry_canvas")
    assert with_canvas_tools(updated) is updated


async def test_canvas_tool_source_loads_via_entry_point_registry() -> None:
    """The full plugin path: registry → tool source → invoke → canvas."""
    from openbb_agent_server.runtime import registry

    stub = _RecordingCanvas()
    canvas_registry.set_canvas(stub)

    source = registry.load("openbb_agent_server.tools", "pywry_canvas", {})
    assert isinstance(source, PyWryCanvasToolSource)
    tools = {t.name: t for t in await source.tools(_ctx(), {})}
    out = tools["canvas_html"].invoke({"html": "<p>from-agent</p>", "title": "Demo"})
    assert "html rendered" in out
    assert ("html", ("<p>from-agent</p>", "Demo")) in stub.calls


def test_acp_package_getattr_rejects_unknown_names() -> None:
    import openbb_agent_server.acp as acp_pkg

    with pytest.raises(AttributeError, match="no attribute"):
        _ = acp_pkg.does_not_exist


async def test_runtime_registers_canvas_tools_in_agent_run(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
) -> None:
    """run_turn with pywry_canvas configured completes with a bound canvas."""
    from openbb_agent_server.acp.canvas_app import with_canvas_tools
    from openbb_agent_server.protocol.schemas import ChatMessage, MessageChunkSSE
    from openbb_agent_server.runtime.embedded import EmbeddedRuntime

    canvas_registry.set_canvas(_RecordingCanvas())
    settings = with_canvas_tools(settings_env)

    runtime = EmbeddedRuntime(settings)
    try:
        deltas: list[str] = []
        async for ev in runtime.run_turn(
            principal=alice,
            conversation_id="conv-canvas",
            messages=[ChatMessage(role="human", content="hello")],
        ):
            if isinstance(ev, MessageChunkSSE):
                deltas.append(ev.data.delta)
        assert "".join(deltas) == "OK."
    finally:
        await runtime.aclose()
