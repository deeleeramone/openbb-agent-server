"""Tests for the canvas document/image renderer and the TVChart datafeed.

Covers the MIME-dispatching ``show_document`` path, the ``canvas_image`` /
``canvas_document`` source normalization, and the datafeed-backed
``show_tvchart_symbol`` wiring — every branch, for 100% coverage.
"""

from __future__ import annotations

import base64
import sys
from typing import Any

import pytest

from openbb_agent_server.acp import canvas_app
from openbb_agent_server.acp.canvas import (
    PyWryCanvas,
    _doc_kind_for_mime,
    _embed_html,
    _looks_like_datafeed_provider,
)
from openbb_agent_server.plugins.tools import _media
from openbb_agent_server.plugins.tools.pywry_canvas import (
    _b64_bytes,
    _data_uri_bytes,
    _data_uri_mime,
    _renderable_src,
    _resolve_media,
    _resolved_text,
    canvas_document,
    canvas_image,
    canvas_tvchart_symbol,
)
from openbb_agent_server.runtime import (
    canvas as canvas_registry,
    context as run_context,
)
from openbb_agent_server.runtime.context import FileRef, RunContext
from openbb_agent_server.runtime.principal import UserPrincipal

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeHandle:
    def __init__(self) -> None:
        self.emits: list[tuple[str, dict[str, Any]]] = []
        self.handlers: dict[str, Any] = {}

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.emits.append((event_type, data))

    def on(self, event_type: str, callback: Any) -> None:
        self.handlers[event_type] = callback


class _RecCanvas:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def show_html(self, html: str, *, title: str | None = None) -> None:
        self.calls.append(("html", {"html": html, "title": title}))

    def show_markdown(self, text: str, *, title: str | None = None) -> None:
        self.calls.append(("markdown", {"text": text, "title": title}))

    def show_table(self, rows, *, title=None, columns=None) -> None:
        self.calls.append(("table", {"rows": rows, "title": title}))

    def show_image(self, src: str, *, title: str | None = None) -> None:
        self.calls.append(("image", {"src": src, "title": title}))

    def show_document(self, *, src, mime, filename=None, title=None, text=None) -> None:
        self.calls.append(
            ("document", {"src": src, "mime": mime, "filename": filename, "text": text})
        )


class _FakeProvider:
    """Looks like a DatafeedProvider instance (has get_bars, not a class)."""

    async def get_bars(self, *a: Any, **k: Any) -> dict[str, Any]:  # pragma: no cover
        return {"bars": [], "status": "no_data"}


class _RecController:
    def __init__(self, *, has_wire: bool = True, wire_raises: bool = False) -> None:
        self.wired: Any = None
        self._has_wire = has_wire
        self._wire_raises = wire_raises
        if has_wire:
            self._wire_datafeed_provider = self._wire  # type: ignore[attr-defined]

    def _wire(self, provider: Any) -> None:
        if self._wire_raises:
            raise RuntimeError("boom")
        self.wired = provider


@pytest.fixture(autouse=True)
def _reset_canvas():
    canvas_registry.reset_canvas()
    yield
    canvas_registry.reset_canvas()


def _ctx_with(files: tuple[FileRef, ...]) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u1", scopes=("agent:query",)),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        uploaded_files=files,
    )


# ---------------------------------------------------------------------------
# _doc_kind_for_mime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mime", "kind"),
    [
        ("image/png", "image"),
        ("image/svg+xml", "image"),
        ("application/pdf", "pdf"),
        ("audio/mpeg", "audio"),
        ("video/mp4", "video"),
        ("text/plain", "text"),
        ("application/json", "text"),
        ("text/csv", "download"),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "download",
        ),
        ("application/octet-stream", "download"),
        ("text/plain; charset=utf-8", "text"),
        ("", "download"),
        (None, "download"),
    ],
)
def test_doc_kind_for_mime(mime, kind) -> None:
    assert _doc_kind_for_mime(mime) == kind


# ---------------------------------------------------------------------------
# _embed_html
# ---------------------------------------------------------------------------


def test_embed_html_image() -> None:
    out = _embed_html("data:image/png;base64,AAAA", kind="image")
    assert out == '<img class="obb-canvas-img" src="data:image/png;base64,AAAA">'


def test_embed_html_pdf_default_and_named() -> None:
    assert "obb-canvas-doc" in _embed_html("data:application/pdf;base64,x", kind="pdf")
    out = _embed_html("u", kind="pdf", filename="report.pdf")
    assert "<iframe" in out and 'title="report.pdf"' in out


def test_embed_html_audio_video() -> None:
    assert "<audio" in _embed_html("u", kind="audio")
    assert "<video" in _embed_html("u", kind="video")


def test_embed_html_text_escapes() -> None:
    out = _embed_html("", kind="text", text="<b>x & y</b>")
    assert "&lt;b&gt;x &amp; y&lt;/b&gt;" in out and "obb-canvas-text" in out


def test_embed_html_download_with_and_without_text() -> None:
    bare = _embed_html("u", kind="download", filename="a.zip")
    assert 'download="a.zip"' in bare and "<pre" not in bare
    withtext = _embed_html("u", kind="download", filename="a.docx", text="hi")
    assert "<pre" in withtext and "hi" in withtext


def test_embed_html_escapes_hostile_src_and_filename() -> None:
    out = _embed_html('x" onerror="y', kind="image")
    assert "onerror" in out and '"' not in out.split("src=")[1].split(">")[0].strip('"')
    pdf = _embed_html("s", kind="pdf", filename='a" onload="b')
    assert "&quot;" in pdf


def test_embed_html_download_default_filename() -> None:
    out = _embed_html("u", kind="download")
    assert 'download="file"' in out


# ---------------------------------------------------------------------------
# PyWryCanvas.show_document
# ---------------------------------------------------------------------------


def _last_html(handle: _FakeHandle) -> str:
    event, data = handle.emits[-1]
    assert event == "obb-canvas:set-html"
    return data["html"]


def test_show_document_pdf_audio_video_text_download() -> None:
    handle = _FakeHandle()
    canvas = PyWryCanvas(handle)
    canvas.show_document(src="data:application/pdf;base64,x", mime="application/pdf")
    assert "obb-canvas-doc" in _last_html(handle)
    canvas.show_document(src="u", mime="audio/mpeg")
    assert "<audio" in _last_html(handle)
    canvas.show_document(src="u", mime="video/mp4")
    assert "<video" in _last_html(handle)
    canvas.show_document(src="", mime="text/plain", text="hello", title="T")
    html = _last_html(handle)
    assert "obb-canvas-text" in html and "hello" in html and "obb-canvas-title" in html
    canvas.show_document(src="u", mime="application/zip", filename="a.zip")
    assert "obb-canvas-download" in _last_html(handle)
    canvas.show_document(src="u", mime="image/png")
    assert "obb-canvas-img" in _last_html(handle)


# ---------------------------------------------------------------------------
# _looks_like_datafeed_provider
# ---------------------------------------------------------------------------


def test_looks_like_datafeed_provider() -> None:
    assert _looks_like_datafeed_provider(_FakeProvider()) is True
    assert _looks_like_datafeed_provider(lambda: None) is False
    assert _looks_like_datafeed_provider(_FakeProvider) is False  # a class/factory


# ---------------------------------------------------------------------------
# show_tvchart_symbol (datafeed mode)
# ---------------------------------------------------------------------------


def test_show_tvchart_symbol_wires_provider_and_mounts_datafeed() -> None:
    handle = _FakeHandle()
    ctrl = _RecController()
    provider = _FakeProvider()
    canvas = PyWryCanvas(
        handle,
        tvchart_controller_factory=lambda h, c: ctrl,
        tvchart_datafeed_provider=provider,
    )
    canvas.show_tvchart_symbol("AAPL", intervals=["1d", "1w"], selected_interval="1w")
    assert ctrl.wired is provider
    create = [d for e, d in handle.emits if e == "tvchart:create"][-1]
    assert create["useDatafeed"] is True
    assert create["series"][0]["symbol"] == "AAPL"
    assert create["interval"] == "1w"
    # provider owns data-request, so the static handler is NOT registered.
    assert "tvchart:data-request" not in handle.handlers


def test_show_tvchart_symbol_defaults_interval_ladder() -> None:
    handle = _FakeHandle()
    canvas = PyWryCanvas(
        handle,
        tvchart_controller_factory=lambda h, c: _RecController(),
        tvchart_datafeed_provider=_FakeProvider(),
    )
    canvas.show_tvchart_symbol("IBM")
    create = [d for e, d in handle.emits if e == "tvchart:create"][-1]
    assert create["interval"] == "1d"
    assert create["title"] == "IBM"


def test_show_tvchart_symbol_provider_via_factory() -> None:
    handle = _FakeHandle()
    ctrl = _RecController()
    provider = _FakeProvider()
    canvas = PyWryCanvas(
        handle,
        tvchart_controller_factory=lambda h, c: ctrl,
        tvchart_datafeed_provider=lambda: provider,
    )
    canvas.show_tvchart_symbol("AAPL")
    assert ctrl.wired is provider


def test_show_tvchart_symbol_requires_symbol() -> None:
    canvas = PyWryCanvas(_FakeHandle(), tvchart_datafeed_provider=_FakeProvider())
    with pytest.raises(ValueError, match="symbol is required"):
        canvas.show_tvchart_symbol("")


def test_show_tvchart_symbol_requires_provider() -> None:
    canvas = PyWryCanvas(_FakeHandle())
    with pytest.raises(RuntimeError, match="needs a tvchart_datafeed_provider"):
        canvas.show_tvchart_symbol("AAPL")


def test_show_tvchart_symbol_factory_returns_none() -> None:
    canvas = PyWryCanvas(_FakeHandle(), tvchart_datafeed_provider=lambda: None)
    with pytest.raises(RuntimeError, match="needs a tvchart_datafeed_provider"):
        canvas.show_tvchart_symbol("AAPL")


def test_show_tvchart_symbol_factory_raises() -> None:
    def boom() -> Any:
        raise RuntimeError("nope")

    canvas = PyWryCanvas(_FakeHandle(), tvchart_datafeed_provider=boom)
    with pytest.raises(RuntimeError, match="needs a tvchart_datafeed_provider"):
        canvas.show_tvchart_symbol("AAPL")


def test_show_tvchart_symbol_controller_without_wire() -> None:
    canvas = PyWryCanvas(
        _FakeHandle(),
        tvchart_controller_factory=lambda h, c: _RecController(has_wire=False),
        tvchart_datafeed_provider=_FakeProvider(),
    )
    with pytest.raises(RuntimeError, match="needs a tvchart_datafeed_provider"):
        canvas.show_tvchart_symbol("AAPL")


def test_show_tvchart_symbol_wire_raises() -> None:
    canvas = PyWryCanvas(
        _FakeHandle(),
        tvchart_controller_factory=lambda h, c: _RecController(wire_raises=True),
        tvchart_datafeed_provider=_FakeProvider(),
    )
    with pytest.raises(RuntimeError, match="needs a tvchart_datafeed_provider"):
        canvas.show_tvchart_symbol("AAPL")


def test_resolve_datafeed_provider_caches_none() -> None:
    canvas = PyWryCanvas(_FakeHandle(), tvchart_datafeed_provider=lambda: None)
    assert canvas._resolve_datafeed_provider() is None
    # second call returns the cached (resolved) value without re-invoking
    assert canvas._resolve_datafeed_provider() is None


def test_ensure_datafeed_wired_idempotent() -> None:
    ctrl = _RecController()
    canvas = PyWryCanvas(
        _FakeHandle(),
        tvchart_controller_factory=lambda h, c: ctrl,
        tvchart_datafeed_provider=_FakeProvider(),
    )
    assert canvas._ensure_datafeed_wired() is True
    # already wired -> early return True, no re-wire
    assert canvas._ensure_datafeed_wired() is True


async def test_resolve_media_path_is_file_oserror(monkeypatch) -> None:
    def boom(self) -> bool:
        raise OSError("bad path")

    monkeypatch.setattr("pathlib.Path.is_file", boom)
    with pytest.raises(ValueError, match="is not a data: URI"):
        await _resolve_media("weird-path", None, None, None, default_mime="image/png")


def test_static_show_tvchart_with_provider_skips_static_handler() -> None:
    handle = _FakeHandle()
    ctrl = _RecController()
    provider = _FakeProvider()
    canvas = PyWryCanvas(
        handle,
        tvchart_controller_factory=lambda h, c: ctrl,
        tvchart_datafeed_provider=provider,
    )
    canvas.show_tvchart(
        {"1d": [{"time": 1, "open": 1, "high": 2, "low": 0, "close": 1}]}
    )
    assert ctrl.wired is provider
    assert "tvchart:data-request" not in handle.handlers


# ---------------------------------------------------------------------------
# data-uri / base64 helpers
# ---------------------------------------------------------------------------


def test_data_uri_helpers() -> None:
    uri = "data:image/png;base64,QUJD"
    assert _data_uri_bytes(uri) == b"ABC"
    assert _data_uri_mime(uri) == "image/png"
    assert _data_uri_mime("data:,plain") is None


def test_data_uri_bytes_malformed() -> None:
    with pytest.raises(ValueError, match="malformed data: URI"):
        _data_uri_bytes("data:image/png;base64,QUJ")  # incorrect padding


def test_b64_bytes_valid_and_invalid() -> None:
    assert _b64_bytes("QUJD") == b"ABC"
    with pytest.raises(ValueError, match="invalid base64"):
        _b64_bytes("QUJ")  # incorrect padding


# ---------------------------------------------------------------------------
# _resolve_media
# ---------------------------------------------------------------------------


async def test_resolve_media_data_uri_and_base64_and_https() -> None:
    raw, url, mime = await _resolve_media(
        "data:image/png;base64,QUJD", None, None, None, default_mime="image/png"
    )
    assert raw == b"ABC" and url is None and mime == "image/png"

    raw, url, mime = await _resolve_media(
        None, "QUJD", "image/jpeg", None, default_mime="image/png"
    )
    assert raw == b"ABC" and mime == "image/jpeg"

    raw, url, mime = await _resolve_media(
        "https://x/y.png", None, None, None, default_mime="image/png"
    )
    assert raw is None and url == "https://x/y.png" and mime == "image/png"


async def test_resolve_media_data_base64_already_data_uri() -> None:
    raw, _url, mime = await _resolve_media(
        None, "data:image/gif;base64,QUJD", None, None, default_mime="image/png"
    )
    assert raw == b"ABC" and mime == "image/gif"


async def test_resolve_media_local_path(tmp_path) -> None:
    p = tmp_path / "pic.png"
    p.write_bytes(b"ABC")
    raw, url, mime = await _resolve_media(
        str(p), None, None, None, default_mime="image/png"
    )
    assert raw == b"ABC" and url is None and mime == "image/png"


async def test_resolve_media_http_rejected() -> None:
    with pytest.raises(ValueError, match="http:// is not allowed"):
        await _resolve_media("http://x/y", None, None, None, default_mime="image/png")


async def test_resolve_media_unknown_string() -> None:
    with pytest.raises(ValueError, match="is not a data: URI"):
        await _resolve_media(
            "javascript:alert(1)", None, None, None, default_mime="image/png"
        )


async def test_resolve_media_nothing_provided() -> None:
    with pytest.raises(ValueError, match="provide one of"):
        await _resolve_media(None, None, None, None, default_mime="image/png")


async def test_resolve_media_uploaded_name_data_base64() -> None:
    ctx = _ctx_with((FileRef(name="a.png", mime="image/png", data_base64="QUJD"),))
    with run_context.bind(ctx):
        raw, url, mime = await _resolve_media(
            None, None, None, "a.png", default_mime="image/png"
        )
    assert raw == b"ABC" and mime == "image/png"


async def test_resolve_media_uploaded_name_data_uri_value() -> None:
    ctx = _ctx_with(
        (FileRef(name="a", mime=None, data_base64="data:image/png;base64,QUJD"),)
    )
    with run_context.bind(ctx):
        raw, _url, _mime = await _resolve_media(
            None, None, None, "a", default_mime="image/png"
        )
    assert raw == b"ABC"


async def test_resolve_media_uploaded_name_https_url() -> None:
    ctx = _ctx_with((FileRef(name="a.png", url="https://x/a.png"),))
    with run_context.bind(ctx):
        raw, url, mime = await _resolve_media(
            None, None, None, "a.png", default_mime="image/png"
        )
    assert raw is None and url == "https://x/a.png" and mime == "image/png"


async def test_resolve_media_uploaded_name_nonhttps_url_fetches(monkeypatch) -> None:
    async def fake_fetch(
        url, *, max_bytes, fallback_mime="application/octet-stream", timeout_s=60.0
    ):
        return _media.FetchedMedia(data=b"XYZ", mime="image/webp")

    monkeypatch.setattr(_media, "fetch_url", fake_fetch)
    ctx = _ctx_with((FileRef(name="a", url="http://insecure/a"),))
    with run_context.bind(ctx):
        raw, url, mime = await _resolve_media(
            None, None, None, "a", default_mime="image/png"
        )
    assert raw == b"XYZ" and url is None and mime == "image/webp"


async def test_resolve_media_uploaded_name_no_data_or_url() -> None:
    ctx = _ctx_with((FileRef(name="a"),))
    with run_context.bind(ctx):
        with pytest.raises(ValueError, match="has no data_base64 or url"):
            await _resolve_media(None, None, None, "a", default_mime="image/png")


async def test_resolve_media_uploaded_name_not_found() -> None:
    ctx = _ctx_with(())
    with run_context.bind(ctx):
        with pytest.raises(ValueError, match="not among this run"):
            await _resolve_media(None, None, None, "missing", default_mime="image/png")


async def test_resolve_media_name_without_context() -> None:
    with pytest.raises(ValueError, match="needs an active run context"):
        await _resolve_media(None, None, None, "a", default_mime="image/png")


# ---------------------------------------------------------------------------
# _renderable_src / _resolved_text
# ---------------------------------------------------------------------------


async def test_renderable_src_url_passthrough_and_bytes() -> None:
    assert await _renderable_src(None, "https://x/y", "image/png") == "https://x/y"
    durl = await _renderable_src(b"ABC", None, "image/png")
    assert durl == "data:image/png;base64,QUJD"


async def test_resolved_text_raw_and_url(monkeypatch) -> None:
    assert await _resolved_text(b"hi", None) == "hi"

    async def fake_fetch(
        url, *, max_bytes, fallback_mime="application/octet-stream", timeout_s=60.0
    ):
        return _media.FetchedMedia(data=b"fetched", mime="text/plain")

    monkeypatch.setattr(_media, "fetch_url", fake_fetch)
    assert await _resolved_text(None, "https://x/y.txt") == "fetched"


# ---------------------------------------------------------------------------
# canvas_image (async)
# ---------------------------------------------------------------------------


async def test_canvas_image_no_canvas() -> None:
    assert (await canvas_image("https://x/y.png")).startswith("error: no live canvas")


async def test_canvas_image_https_data_and_bytes() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    assert "image rendered" in await canvas_image("https://x/y.png")
    assert cv.calls[-1][1]["src"] == "https://x/y.png"
    assert "image rendered" in await canvas_image(
        data_base64=_PNG_B64, mime="image/png"
    )
    assert cv.calls[-1][1]["src"].startswith("data:image/png;base64,")


async def test_canvas_image_uploaded_name() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    ctx = _ctx_with((FileRef(name="up.png", mime="image/png", data_base64="QUJD"),))
    with run_context.bind(ctx):
        assert "image rendered" in await canvas_image(name="up.png")
    assert cv.calls[-1][1]["src"] == "data:image/png;base64,QUJD"


async def test_canvas_image_path(tmp_path) -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    p = tmp_path / "i.png"
    p.write_bytes(base64.b64decode(_PNG_B64))
    assert "image rendered" in await canvas_image(str(p))
    assert cv.calls[-1][1]["src"].startswith("data:image/png;base64,")


async def test_canvas_image_error() -> None:
    canvas_registry.set_canvas(_RecCanvas())
    assert (await canvas_image("ftp://nope")).startswith("error:")


# ---------------------------------------------------------------------------
# canvas_document router
# ---------------------------------------------------------------------------


async def test_canvas_document_no_canvas() -> None:
    out = await canvas_document(src="https://x/y.pdf", mime="application/pdf")
    assert out.startswith("error: no live canvas")


async def test_canvas_document_resolve_error() -> None:
    canvas_registry.set_canvas(_RecCanvas())
    assert (await canvas_document(src="http://x", mime="application/pdf")).startswith(
        "error:"
    )


async def test_canvas_document_pdf_https_passthrough() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    out = await canvas_document(src="https://x/y.pdf", mime="application/pdf")
    assert "pdf document rendered" in out
    assert cv.calls[-1][0] == "document" and cv.calls[-1][1]["src"] == "https://x/y.pdf"


async def test_canvas_document_image_routes_to_show_image() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    await canvas_document(data_base64=_PNG_B64, mime="image/png")
    assert cv.calls[-1][0] == "image"


async def test_canvas_document_audio_video() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    await canvas_document(src="https://x/a.mp3", mime="audio/mpeg")
    assert cv.calls[-1][0] == "document" and cv.calls[-1][1]["mime"] == "audio/mpeg"
    await canvas_document(src="https://x/v.mp4", mime="video/mp4")
    assert cv.calls[-1][1]["mime"] == "video/mp4"


async def test_canvas_document_csv_and_tsv() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    csv_b64 = base64.b64encode(b"a,b\n1,2\n3,4\n").decode()
    out = await canvas_document(data_base64=csv_b64, mime="text/csv")
    assert "table rendered from 2 rows" in out
    assert cv.calls[-1][0] == "table" and cv.calls[-1][1]["rows"][0]["a"] == "1"
    tsv_b64 = base64.b64encode(b"a\tb\n1\t2\n").decode()
    await canvas_document(data_base64=tsv_b64, mime="text/tab-separated-values")
    assert cv.calls[-1][1]["rows"][0]["b"] == "2"


async def test_canvas_document_json_list_and_scalar() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    listing = base64.b64encode(b'[{"a":1},{"a":2}]').decode()
    out = await canvas_document(data_base64=listing, mime="application/json")
    assert "JSON rows" in out and cv.calls[-1][0] == "table"
    scalar = base64.b64encode(b'{"a":1}').decode()
    out = await canvas_document(data_base64=scalar, mime="application/json")
    assert "JSON rendered" in out and cv.calls[-1][0] == "document"


async def test_canvas_document_json_malformed() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    bad = base64.b64encode(b"{not json").decode()
    assert (await canvas_document(data_base64=bad, mime="application/json")).startswith(
        "error: malformed JSON"
    )


async def test_canvas_document_markdown_html_text_yaml() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    await canvas_document(
        data_base64=base64.b64encode(b"# h").decode(), mime="text/markdown"
    )
    assert cv.calls[-1][0] == "markdown"
    await canvas_document(
        data_base64=base64.b64encode(b"<p>x</p>").decode(), mime="text/html"
    )
    assert cv.calls[-1][0] == "html"
    await canvas_document(
        data_base64=base64.b64encode(b"plain").decode(), mime="text/plain"
    )
    assert cv.calls[-1][0] == "document" and cv.calls[-1][1]["text"] == "plain"
    await canvas_document(
        data_base64=base64.b64encode(b"k: v").decode(), mime="application/yaml"
    )
    assert cv.calls[-1][0] == "document" and cv.calls[-1][1]["text"] == "k: v"


async def test_canvas_document_download_fallback() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    out = await canvas_document(
        src="https://x/a.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="a.docx",
        text="extracted",
    )
    assert "document rendered" in out
    assert cv.calls[-1][0] == "document" and cv.calls[-1][1]["text"] == "extracted"


async def test_canvas_document_csv_malformed(monkeypatch) -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)

    class _Boom:
        def __iter__(self):
            raise __import__("csv").Error("bad")

    monkeypatch.setattr("csv.DictReader", lambda *a, **k: _Boom())
    out = await canvas_document(
        data_base64=base64.b64encode(b"a,b").decode(), mime="text/csv"
    )
    assert out.startswith("error: malformed delimited text")


# ---------------------------------------------------------------------------
# canvas_tvchart_symbol tool
# ---------------------------------------------------------------------------


async def test_canvas_tvchart_symbol_tool_no_canvas() -> None:
    assert canvas_tvchart_symbol("AAPL").startswith("error: no live canvas")


def test_canvas_tvchart_symbol_tool_validations() -> None:
    cv = _RecCanvas()
    canvas_registry.set_canvas(cv)
    assert canvas_tvchart_symbol("").startswith("error: symbol is required")
    assert canvas_tvchart_symbol("AAPL", series_type="weird").startswith(
        "error: series_type"
    )


def test_canvas_tvchart_symbol_tool_unsupported_host() -> None:
    cv = _RecCanvas()  # no show_tvchart_symbol
    canvas_registry.set_canvas(cv)
    assert "does not support datafeed" in canvas_tvchart_symbol("AAPL")


def test_canvas_tvchart_symbol_tool_success() -> None:
    handle = _FakeHandle()
    canvas = PyWryCanvas(
        handle,
        tvchart_controller_factory=lambda h, c: _RecController(),
        tvchart_datafeed_provider=_FakeProvider(),
    )
    canvas_registry.set_canvas(canvas)
    out = canvas_tvchart_symbol("AAPL", intervals=["1d", "1w"], selected_interval="1d")
    assert "datafeed tvchart rendered for 'AAPL'" in out


def test_canvas_tvchart_symbol_tool_runtime_error() -> None:
    canvas = PyWryCanvas(_FakeHandle())  # no provider -> RuntimeError -> error string
    canvas_registry.set_canvas(canvas)
    assert canvas_tvchart_symbol("AAPL").startswith("error:")


# ---------------------------------------------------------------------------
# canvas_app._build_datafeed_provider
# ---------------------------------------------------------------------------


def test_build_datafeed_provider_unset_returns_none(monkeypatch) -> None:
    # No default data source: unset env -> no datafeed (operator opt-in only).
    monkeypatch.delenv(canvas_app.CANVAS_DATAFEED_URL_ENV, raising=False)
    assert canvas_app._build_datafeed_provider() is None


def test_build_datafeed_provider_disabled(monkeypatch) -> None:
    monkeypatch.setenv(canvas_app.CANVAS_DATAFEED_URL_ENV, "   ")
    assert canvas_app._build_datafeed_provider() is None


def test_build_datafeed_provider_explicit_url(monkeypatch) -> None:
    # Operator explicitly opts in by naming their own UDF endpoint.
    monkeypatch.setenv(canvas_app.CANVAS_DATAFEED_URL_ENV, "https://feed.example")
    provider = canvas_app._build_datafeed_provider()
    assert provider is not None and type(provider).__name__ == "UDFAdapter"
    provider.close()


def test_build_datafeed_provider_import_failure(monkeypatch) -> None:
    monkeypatch.setenv(canvas_app.CANVAS_DATAFEED_URL_ENV, "https://feed.example")
    monkeypatch.setitem(sys.modules, "pywry.tvchart.udf", None)
    assert canvas_app._build_datafeed_provider() is None
