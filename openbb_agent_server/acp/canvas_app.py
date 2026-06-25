"""Desktop canvas app — a PyWry window the agent treats as a live canvas.

``launch()`` opens a PyWry window whose main content page is the
agent's `live canvas <openbb_agent_server.acp.canvas>`, with the
ACP chat toolbar attached. The ``pywry_canvas`` tool source is added to
the configured tool sources automatically, so the agent can render
HTML, markdown, Plotly figures, tables, and images onto the page while
the conversation runs in the chat panel.

CLI entry point (installed as ``openbb-agent-canvas``)::

    openbb-agent-canvas --config-file /etc/openbb/agent.toml --profile default

Or programmatically, non-blocking, to embed in a larger PyWry app::

    from openbb_agent_server.acp.canvas_app import launch

    canvas_app = launch(block=False)
    ...
    canvas_app.app.block()

This module keeps its top-level imports pywry-free so the console
script can print a friendly install hint instead of a bare
``ImportError`` when the ``pywry`` extra is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openbb_agent_server.acp.canvas import (
    CANVAS_CSS,
    PyWryCanvas,
    build_canvas_html,
)
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import canvas as canvas_registry

logger = logging.getLogger("openbb_agent_server.acp.canvas_app")

CANVAS_TOOL_SOURCE = "pywry_canvas"

# The chart header's Symbol Search and Compare controls need a data
# source for ARBITRARY symbols. That source is the OPERATOR's choice —
# there is NO default. When ``OPENBB_AGENT_CANVAS_DATAFEED_URL`` names a
# UDF server, pywry's ``UDFAdapter`` is wired to it; otherwise no
# datafeed is attached and charts stay in static caller-supplied-data
# mode. (Embedders can also pass their own ``DatafeedProvider`` to
# ``PyWryCanvas`` directly.)
CANVAS_DATAFEED_URL_ENV = "OPENBB_AGENT_CANVAS_DATAFEED_URL"


@dataclass
class CanvasApp:
    """Bundle the objects ``launch()`` wired together.

    Returned to non-blocking hosts so they can drive the window's event
    loop themselves and reach the wired components.

    Attributes
    ----------
    app : Any
        The ``PyWry`` application owning the window event loop.
    widget : Any
        The shown window/widget handle the canvas renders onto.
    chat : Any
        The ``ChatManager`` driving the attached chat toolbar.
    provider : Any
        The ``OpenBBAgentProvider`` backing the chat conversation.
    canvas : PyWryCanvas
        The live canvas bound to the agent's canvas tools.
    datafeed : Any, optional
        The TVChart datafeed provider, or ``None`` when no datafeed URL
        is configured.
    """

    app: Any
    widget: Any
    chat: Any
    provider: Any
    canvas: PyWryCanvas
    datafeed: Any = None


def _format_welcome(name: str | None, description: str | None) -> str:
    """Format a markdown welcome message from agent metadata."""
    parts: list[str] = []
    if name:
        parts.append(f"### {name}")
    if description:
        parts.append(f"_{description}_")
    parts.append("---")
    parts.append("Type a message to get started.")
    return "\n\n".join(parts)


def with_canvas_tools(settings: AgentServerSettings) -> AgentServerSettings:
    """Return settings with ``pywry_canvas`` appended to the tool sources.

    Only the base tool list is touched — a profile that overrides
    ``tool_sources`` must list ``pywry_canvas`` itself.
    """
    if CANVAS_TOOL_SOURCE in settings.tool_sources:
        return settings
    return settings.model_copy(
        update={"tool_sources": (*settings.tool_sources, CANVAS_TOOL_SOURCE)}
    )


def _plotly_assets() -> str:
    """Concatenate pywry's bundled plotly.js + defaults for injection."""
    from pywry.assets import (
        get_plotly_defaults_js,
        get_plotly_js,
        get_plotly_templates_js,
    )

    return "\n".join(
        [get_plotly_js(), get_plotly_templates_js(), get_plotly_defaults_js()]
    )


def _tvchart_assets() -> str:
    """Pywry's lightweight-charts bundle + the TVChart engine + toolbars.

    The engine (``get_tvchart_defaults_js``) self-registers the
    ``tvchart:*`` protocol handlers on the page bridge; the toolbar
    handlers script provides ``initToolbarHandlers`` for the chrome the
    canvas mounts around the chart.
    """
    from pywry.assets import get_tvchart_defaults_js, get_tvchart_js
    from pywry.toolbar import get_toolbar_script

    return "\n".join(
        [
            get_tvchart_js(),
            get_tvchart_defaults_js(),
            get_toolbar_script(with_script_tag=False),
        ]
    )


def _build_datafeed_provider() -> Any:
    """Build the canvas's TVChart datafeed provider (UDF-backed), or None.

    The data source is the OPERATOR's choice and there is NO default:
    connects pywry's ``UDFAdapter`` ONLY to an explicit UDF endpoint named
    in ``OPENBB_AGENT_CANVAS_DATAFEED_URL``. Returns ``None`` when that is
    unset (or pywry's adapter cannot be imported) — charts then stay in
    static caller-supplied-data mode (no symbol search / compare source).
    """
    import os

    url = os.environ.get(CANVAS_DATAFEED_URL_ENV, "").strip()
    if not url:
        logger.info(
            "canvas: no datafeed configured (%s unset) — symbol search/compare "
            "have no data source",
            CANVAS_DATAFEED_URL_ENV,
        )
        return None
    try:
        from pywry.tvchart.udf import UDFAdapter
    except Exception:
        logger.warning(
            "canvas: pywry UDFAdapter unavailable; symbol search/compare disabled",
            exc_info=True,
        )
        return None
    logger.info("canvas: datafeed via UDFAdapter(%s)", url)
    return UDFAdapter(url)


def _tvchart_controller(handle: Any, chart_id: str) -> Any:
    """Bind pywry's full TVChart protocol implementation to the canvas chart.

    Returns an instance of pywry's own ``TVChartStateMixin`` — the
    complete Python-side protocol surface (``update_series`` /
    ``update_bar``, series add/remove, the entire built-in indicator
    family, volume profiles, markers, price lines, view control, state
    export, the datafeed responder family, storage wiring) — scoped to
    the canvas chart and extended with a synchronous
    ``request(emit_event, payload, response_event)`` helper that
    correlates request/response pairs through the protocol's ``context``
    echo.
    """
    import threading
    import uuid as _uuid

    from pywry.tvchart.mixin import TVChartStateMixin

    class _CanvasTVChartController(TVChartStateMixin):
        """pywry's protocol mixin over a duck-typed widget handle."""

        _RESPONSE_EVENTS = (
            "tvchart:state-response",
            "tvchart:list-indicators-response",
        )

        def __init__(self) -> None:
            self.chart_id = chart_id
            self._handle = handle
            self._waiters: dict[str, dict[str, Any]] = {}
            self._waiter_lock = threading.Lock()
            handle_on = getattr(handle, "on", None)
            self._can_wait = callable(handle_on)
            if callable(handle_on):
                for event in self._RESPONSE_EVENTS:
                    handle_on(event, self._response_handler)

        def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
            payload = dict(data or {})
            if event_type.startswith("tvchart:"):
                payload.setdefault("chartId", self.chart_id)
            self._handle.emit(event_type, payload)

        def on(self, event_type: str, callback: Any, label: Any = None) -> Any:
            # pywry's _wire_datafeed_provider registers handlers with a
            # ``label=`` kwarg (window scoping); forward it when the handle
            # accepts it, else fall back to the 2-arg form.
            handle_on = getattr(self._handle, "on", None)
            if callable(handle_on):
                try:
                    handle_on(event_type, callback, label=label)
                except TypeError:
                    handle_on(event_type, callback)
            return self

        def _response_handler(self, data: Any, *_args: Any) -> None:
            payload = dict(data or {})
            context = payload.get("context")
            token = context.get("obb_request") if isinstance(context, dict) else None
            if token is None:
                return
            with self._waiter_lock:
                waiter = self._waiters.pop(token, None)
            if waiter is not None:
                waiter["payload"] = payload
                waiter["event"].set()

        def _await_response(
            self,
            call: Any,
            timeout: float,
        ) -> dict[str, Any] | None:
            """Invoke a mixin request method and await its response.

            ``call(context)`` runs one of the mixin's request methods
            with a correlation ``context`` — the echo mechanism the
            protocol documents for exactly this purpose — and the
            matching response event resolves the waiter.
            """
            if not self._can_wait:
                return None
            token = _uuid.uuid4().hex
            done = threading.Event()
            box: dict[str, Any] = {"event": done, "payload": None}
            with self._waiter_lock:
                self._waiters[token] = box
            call({"obb_request": token})
            if not done.wait(timeout):
                with self._waiter_lock:
                    self._waiters.pop(token, None)
                return None
            return box["payload"]

        def get_state(self, timeout: float = 10.0) -> dict[str, Any] | None:
            """Run ``request_tvchart_state`` and await ``state-response``."""
            return self._await_response(
                lambda ctx: self.request_tvchart_state(context=ctx), timeout
            )

        def list_indicators_sync(self, timeout: float = 10.0) -> dict[str, Any] | None:
            """Run ``list_indicators`` and await ``list-indicators-response``."""
            return self._await_response(
                lambda ctx: self.list_indicators(context=ctx), timeout
            )

    return _CanvasTVChartController()


def _tvchart_chrome(
    container_html: str,
    intervals: list[str] | None,
    selected_interval: str | None,
) -> str:
    """Wrap a chart container with pywry's full TVChart toolbar chrome.

    Uses pywry's own builders — ``build_tvchart_toolbars`` (header,
    drawing rail, time-range bar, OHLC legend overlay) and
    ``wrap_content_with_toolbars`` (the single source of truth for
    toolbar layout) — so the canvas chart carries the same chrome as
    ``app.show_tvchart``.
    """
    from pywry.toolbar import wrap_content_with_toolbars
    from pywry.tvchart import build_tvchart_toolbars

    toolbars = build_tvchart_toolbars(
        intervals=list(intervals) if intervals else None,
        selected_interval=selected_interval,
    )
    return wrap_content_with_toolbars(container_html, toolbars)


def launch(
    explicit_path: str | None = None,
    *,
    settings: AgentServerSettings | None = None,
    profile: str | None = None,
    user_id: str = "pywry-local",
    title: str | None = None,
    block: bool = True,
    **chat_kwargs: Any,
) -> CanvasApp:
    """Open the canvas window with the chat toolbar attached.

    Resolve the layered ``openbb.toml`` cascade (or take explicit
    ``settings``), append the ``pywry_canvas`` tool source, build the
    provider + ``ChatManager``, show the window, and bind the live
    canvas so the agent's canvas tools reach the page. With
    ``block=True`` (the default) this call runs the window's event loop
    until close; ``block=False`` returns immediately with the wired
    :class:`CanvasApp`.

    Parameters
    ----------
    explicit_path : str or None, optional
        Path to an explicit ``openbb.toml``; ignored when ``settings``
        is provided, otherwise it seeds the layered config cascade.
    settings : AgentServerSettings or None, optional
        Pre-resolved settings; when given, the config cascade is skipped.
    profile : str or None, optional
        Agent profile to start on; defaults to the configured default.
    user_id : str, default "pywry-local"
        Identity used for history and memory scoping.
    title : str or None, optional
        Window title override; falls back to the agent metadata name.
    block : bool, default True
        When ``True``, run the window event loop until close and tear
        down afterward. When ``False``, return immediately.
    **chat_kwargs : Any
        Extra keyword arguments forwarded to ``ChatManager``; the agent
        description seeds ``welcome_message`` if not already set.

    Returns
    -------
    CanvasApp
        The wired application bundle. When ``block=True`` it is returned
        only after the window has closed and been torn down.
    """
    from pywry import PyWry
    from pywry.models import HtmlContent

    from openbb_agent_server.acp.provider import (
        OpenBBAgentProvider,
        _build_settings_items,
        _make_settings_change_handler,
    )

    if settings is None:
        from openbb_agent_server.app.config import (
            agent_section,
            bootstrap_launcher_config,
        )

        cfg = bootstrap_launcher_config(explicit_path=explicit_path)
        settings = AgentServerSettings.from_toml(agent_section(cfg))
    settings = with_canvas_tools(settings)

    provider = OpenBBAgentProvider(settings, profile=profile, user_id=user_id)
    metadata = settings.metadata
    welcome = _format_welcome(metadata.name, metadata.description)
    if welcome:
        chat_kwargs.setdefault("welcome_message", welcome)

    active_profile = profile or settings.default_profile
    settings_items = _build_settings_items(settings, active_profile)
    chat_ref: list[Any] = []
    chat_kwargs.setdefault("settings", settings_items)
    chat_kwargs.setdefault(
        "on_settings_change",
        _make_settings_change_handler(provider, settings, chat_ref),
    )

    from pywry.chat.manager import ChatManager

    chat = ChatManager(provider=provider, **chat_kwargs)
    chat_ref.append(chat)

    window_title = title or metadata.name or "OpenBB Agent"
    app = PyWry(title=window_title)
    content = HtmlContent(
        html=build_canvas_html(),
        inline_css=CANVAS_CSS,
    )
    widget = app.show(
        content,
        toolbars=[chat.toolbar()],
        callbacks=chat.callbacks(),
    )
    chat.bind(widget)

    datafeed = _build_datafeed_provider()
    canvas = PyWryCanvas(
        widget,
        plotly_assets=_plotly_assets,
        tvchart_assets=_tvchart_assets,
        tvchart_chrome=_tvchart_chrome,
        tvchart_controller_factory=_tvchart_controller,
        tvchart_datafeed_provider=datafeed,
    )
    canvas_registry.set_canvas(canvas)
    logger.info(
        "canvas app up | title=%r profile=%r tools+=%s",
        window_title,
        profile or settings.default_profile,
        CANVAS_TOOL_SOURCE,
    )

    canvas_app = CanvasApp(
        app=app,
        widget=widget,
        chat=chat,
        provider=provider,
        canvas=canvas,
        datafeed=datafeed,
    )
    if block:
        try:
            app.block()
        finally:
            _teardown(canvas_app)
    return canvas_app


def _teardown(canvas_app: CanvasApp) -> None:
    """Best-effort cleanup after the window closes."""
    import asyncio
    from contextlib import suppress

    canvas_registry.reset_canvas()
    if canvas_app.datafeed is not None:
        with suppress(Exception):
            canvas_app.datafeed.close()
    with suppress(Exception):
        asyncio.run(canvas_app.provider.runtime.aclose())


def main(argv: list[str] | None = None) -> int:
    """Run the ``openbb-agent-canvas`` console entry point.

    Parse command-line arguments and call :func:`launch` in blocking
    mode. If a required optional dependency is missing, print the install
    hint and exit non-zero instead of raising.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector to parse; defaults to ``sys.argv`` when ``None``.

    Returns
    -------
    int
        ``0`` on a clean exit. On ``ImportError`` the process exits with
        status ``1`` via the argument parser.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="openbb-agent-canvas",
        description=(
            "Open a PyWry window with the configured agent attached as a "
            "chat panel and the main page as its live canvas."
        ),
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Explicit openbb.toml (otherwise the layered cascade applies).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Agent profile to start on (default: the configured default).",
    )
    parser.add_argument(
        "--user-id",
        default="pywry-local",
        help="Identity for history / memory scoping.",
    )
    parser.add_argument("--title", default=None, help="Window title override.")
    args = parser.parse_args(argv)

    try:
        launch(
            args.config_file,
            profile=args.profile,
            user_id=args.user_id,
            title=args.title,
            block=True,
        )
    except ImportError as exc:
        parser.exit(1, f"{exc}\n")
    return 0
