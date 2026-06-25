"""Tests for the desktop canvas app launcher (pywry faked at the seams)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pywry.chat")

from openbb_agent_server.acp import canvas_app as canvas_app_mod  # noqa: E402
from openbb_agent_server.acp.canvas import PyWryCanvas  # noqa: E402
from openbb_agent_server.acp.canvas_app import (  # noqa: E402
    CanvasApp,
    _plotly_assets,
    _teardown,
    launch,
    main,
)
from openbb_agent_server.app.settings import AgentServerSettings  # noqa: E402
from openbb_agent_server.runtime import canvas as canvas_registry  # noqa: E402


class _FakeWidget:
    def __init__(self) -> None:
        self.emits: list[tuple[str, dict[str, Any]]] = []
        self.scripts: list[str] = []

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.emits.append((event_type, data))

    def eval_js(self, script: str) -> None:
        self.scripts.append(script)


class _FakePyWry:
    instances: list[_FakePyWry] = []

    def __init__(self, title: str = "") -> None:
        self.title = title
        self.blocked = False
        self.shown: list[dict[str, Any]] = []
        _FakePyWry.instances.append(self)

    def show(self, content: Any, *, toolbars: Any = None, callbacks: Any = None):
        self.shown.append(
            {"content": content, "toolbars": toolbars, "callbacks": callbacks}
        )
        return _FakeWidget()

    def block(self) -> None:
        self.blocked = True


class _FakeChatManager:
    def __init__(self, provider: Any = None, **kwargs: Any) -> None:
        self.provider = provider
        self.kwargs = kwargs
        self.bound: Any = None

    def toolbar(self) -> str:
        return "TOOLBAR"

    def callbacks(self) -> dict[str, Any]:
        return {"chat:user-message": lambda *a, **k: None}

    def bind(self, widget: Any) -> None:
        self.bound = widget


@pytest.fixture
def fake_pywry(monkeypatch: pytest.MonkeyPatch) -> type[_FakePyWry]:
    _FakePyWry.instances = []
    monkeypatch.setattr("pywry.PyWry", _FakePyWry)
    monkeypatch.setattr("pywry.chat.manager.ChatManager", _FakeChatManager)
    return _FakePyWry


def test_launch_wires_window_chat_and_canvas(
    settings_env: AgentServerSettings,
    fake_pywry: type[_FakePyWry],
) -> None:
    result = launch(settings=settings_env, title="Test Window", block=False)

    assert isinstance(result, CanvasApp)
    app = fake_pywry.instances[0]
    assert app.title == "Test Window"
    shown = app.shown[0]
    assert shown["toolbars"] == ["TOOLBAR"]
    assert "chat:user-message" in shown["callbacks"]
    # The chat manager got the provider and was bound to the widget.
    assert isinstance(result.chat, _FakeChatManager)
    assert result.chat.provider is result.provider
    assert result.chat.bound is result.widget
    # The live canvas is registered and targets the shown widget.
    assert isinstance(result.canvas, PyWryCanvas)
    assert canvas_registry.get_canvas() is result.canvas
    # The canvas tool source was appended to the settings.
    assert "pywry_canvas" in result.provider.runtime.settings.tool_sources
    # Settings items include model params and feature toggles.
    assert "settings" in result.chat.kwargs
    setting_ids = [s.id for s in result.chat.kwargs["settings"]]
    assert "temperature" in setting_ids
    assert "on_settings_change" in result.chat.kwargs


def test_launch_block_true_runs_loop_and_tears_down(
    settings_env: AgentServerSettings,
    fake_pywry: type[_FakePyWry],
) -> None:
    result = launch(settings=settings_env, block=True)
    assert fake_pywry.instances[0].blocked is True
    # Teardown ran after block(): canvas unbound, runtime closed.
    assert canvas_registry.get_canvas() is None
    assert not result.provider.runtime.started


def test_launch_resolves_settings_from_toml_cascade(
    fake_pywry: type[_FakePyWry],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.app import config as cfg_mod

    cfg = tmp_path / "canvas.toml"
    cfg.write_text(
        '[agent]\nmodel_provider = "fake"\ncheckpointer_provider = "inmemory"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", tmp_path / "no_user_dir")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))

    result = launch(str(cfg), block=False)
    assert result.provider.runtime.settings.checkpointer_provider == "inmemory"


def test_teardown_is_best_effort(
    settings_env: AgentServerSettings,
    fake_pywry: type[_FakePyWry],
) -> None:
    result = launch(settings=settings_env, block=False)

    class _BoomRuntime:
        async def aclose(self) -> None:
            raise RuntimeError("already gone")

    class _Datafeed:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    feed = _Datafeed()
    broken = CanvasApp(
        app=result.app,
        widget=result.widget,
        chat=result.chat,
        provider=type("P", (), {"runtime": _BoomRuntime()})(),
        canvas=result.canvas,
        datafeed=feed,
    )
    _teardown(broken)  # must not raise
    assert canvas_registry.get_canvas() is None
    assert feed.closed is True  # the datafeed was closed on teardown


def test_plotly_assets_returns_bundle() -> None:
    js = _plotly_assets()
    assert isinstance(js, str)
    assert len(js) > 1000


def test_main_parses_args_and_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_launch(config_file: Any, **kwargs: Any) -> None:
        calls.append((config_file, kwargs))

    monkeypatch.setattr(canvas_app_mod, "launch", fake_launch)
    rc = main(
        [
            "--config-file",
            "/etc/openbb/agent.toml",
            "--profile",
            "research",
            "--user-id",
            "desk-9",
            "--title",
            "Desk",
        ]
    )
    assert rc == 0
    config_file, kwargs = calls[0]
    assert config_file == "/etc/openbb/agent.toml"
    assert kwargs == {
        "profile": "research",
        "user_id": "desk-9",
        "title": "Desk",
        "block": True,
    }


def test_main_exits_nonzero_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_launch(*args: Any, **kwargs: Any) -> None:
        raise ImportError("pywry missing")

    monkeypatch.setattr(canvas_app_mod, "launch", fake_launch)
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# The real TVChart protocol controller (pywry's TVChartStateMixin bound
# to a duck-typed handle).
# ---------------------------------------------------------------------------


class _ProtocolHandle:
    def __init__(self) -> None:
        self.emits: list[tuple[str, dict[str, Any]]] = []
        self.handlers: dict[str, Any] = {}

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.emits.append((event_type, data))

    def on(self, event_type: str, callback: Any) -> None:
        self.handlers[event_type] = callback


def _controller(handle: Any = None):
    from openbb_agent_server.acp.canvas_app import _tvchart_controller

    handle = handle or _ProtocolHandle()
    return handle, _tvchart_controller(handle, "openbb-canvas-tvchart")


def test_controller_is_pywrys_own_mixin() -> None:
    from pywry.tvchart.mixin import TVChartStateMixin

    _, ctrl = _controller()
    assert isinstance(ctrl, TVChartStateMixin)


def test_controller_scopes_protocol_events_to_the_canvas_chart() -> None:
    handle, ctrl = _controller()
    ctrl.update_series(
        [{"time": 1, "open": 1, "high": 2, "low": 0, "close": 1, "volume": 9}]
    )
    event, payload = handle.emits[-1]
    assert event == "tvchart:update"
    assert payload["chartId"] == "openbb-canvas-tvchart"
    # List input passes through with volume embedded in the bars — the
    # engine's update path extracts it client-side.
    assert payload["bars"][0]["close"] == 1
    assert payload["bars"][0]["volume"] == 9

    ctrl.update_bar(
        {"time": 2, "open": 1, "high": 2, "low": 0, "close": 2, "volume": 4}
    )
    event, payload = handle.emits[-1]
    assert event == "tvchart:stream"
    assert payload["chartId"] == "openbb-canvas-tvchart"
    assert payload["bar"]["close"] == 2
    assert payload["volume"]["value"] == 4

    ctrl.add_price_line(123.0, title="lvl")
    event, payload = handle.emits[-1]
    assert event == "tvchart:add-price-line"
    assert payload["price"] == 123.0
    assert payload["chartId"] == "openbb-canvas-tvchart"


def test_controller_request_round_trip_and_timeout() -> None:
    handle, ctrl = _controller()
    # Response handlers were registered at construction.
    assert "tvchart:state-response" in handle.handlers
    assert "tvchart:list-indicators-response" in handle.handlers

    import threading

    def _answer_state() -> None:
        # Find the emitted request and echo its context back.
        for event, payload in handle.emits:
            if event == "tvchart:request-state":
                handle.handlers["tvchart:state-response"](
                    {
                        "chartId": "openbb-canvas-tvchart",
                        "interval": "1d",
                        "context": payload["context"],
                    }
                )
                return

    timer = threading.Timer(0.1, _answer_state)
    timer.start()
    try:
        state = ctrl.get_state(timeout=5.0)
    finally:
        timer.cancel()
    assert state is not None
    assert state["interval"] == "1d"

    # No answer -> clean timeout, waiter cleaned up.
    assert ctrl.list_indicators_sync(timeout=0.2) is None
    assert ctrl._waiters == {}

    # Foreign responses (no token) are ignored without error.
    handle.handlers["tvchart:state-response"]({"chartId": "x"})
    handle.handlers["tvchart:state-response"](None)


def test_controller_on_delegates_to_handle() -> None:
    handle, ctrl = _controller()
    sentinel = object()
    assert ctrl.on("tvchart:data-settled", sentinel) is ctrl
    assert handle.handlers["tvchart:data-settled"] is sentinel
    # Non-tvchart emits pass through without chartId injection.
    ctrl.emit("pywry:update-theme", {"theme": "light"})
    assert handle.emits[-1] == ("pywry:update-theme", {"theme": "light"})


def test_controller_without_on_degrades() -> None:
    class _EmitOnly:
        def __init__(self) -> None:
            self.emits: list = []

        def emit(self, event_type: str, data: dict[str, Any]) -> None:
            self.emits.append((event_type, data))

    handle = _EmitOnly()
    _, ctrl = _controller(handle)
    assert ctrl.get_state(timeout=0.1) is None  # no .on -> no waiting
    ctrl.fit_content()  # plain emits still work
    assert handle.emits[-1][0] == "tvchart:time-scale"


def test_launch_binds_tvchart_controller_factory(
    settings_env: AgentServerSettings,
    fake_pywry: type[_FakePyWry],
) -> None:
    result = launch(settings=settings_env, block=False)
    assert result.canvas._tvchart_controller_factory is not None
