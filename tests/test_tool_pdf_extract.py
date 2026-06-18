"""pdf_extract tool source tests."""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest

from openbb_agent_server.persistence import models as m
from openbb_agent_server.plugins.tools import pdf_extract as pe
from openbb_agent_server.plugins.tools.pdf_extract import (
    PdfExtractToolSource,
    _bytes_from_data_url,
    _coerce_page_range,
    _coerce_str,
    _find_any_http_url,
    _find_any_pdf_b64,
    _is_pdf,
    _match_uploaded_pdf,
    _normalise_pdf_name,
    _resolve_pdf_bytes,
    _string_to_pdf_b64,
)
from openbb_agent_server.runtime import (
    context as run_context,
    emit,
    services,
)
from openbb_agent_server.runtime.context import FileRef, RunContext, WidgetRef, bind
from openbb_agent_server.runtime.pdf_store import PdfStore
from openbb_agent_server.runtime.principal import UserPrincipal


def _make_ctx(
    files: list[FileRef], widgets: list[WidgetRef] | None = None
) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        uploaded_files=tuple(files),
        widgets=tuple(widgets or ()),
    )


def _build_one_page_pdf(
    text: str = "Hello World", *, pages: int = 1, outline: bool = False
) -> bytes:
    """Build a minimal real PDF."""
    pdfplumber = pytest.importorskip("pdfplumber")  # noqa: F841
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for idx in range(pages):
        if outline:
            key = f"sec{idx}"
            c.bookmarkPage(key)
            c.addOutlineEntry(f"Section {idx + 1}", key, level=0)
        c.drawString(72, 720, f"{text} page {idx + 1}")
        c.showPage()
    if outline:
        c.setTitle("Annual Report")
        c.setAuthor("Analyst")
        c.setSubject("Filings")
    c.save()
    return buf.getvalue()


async def _ready_store(
    principal: UserPrincipal, name: str, pdf_bytes: bytes, *, data_base64: str
) -> PdfStore:
    """Build a PdfStore with one PDF already ingested to ready."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    async with store.engine.begin() as conn:
        await conn.run_sync(m.Base.metadata.create_all)
    await store.ingest_async(principal=principal, name=name, data_base64=data_base64)
    await store.await_pending()
    return store


def test_is_pdf_detects_extension_and_mime() -> None:
    assert _is_pdf(FileRef(name="x.PDF"))
    assert _is_pdf(FileRef(name="x", mime="application/pdf"))
    assert not _is_pdf(FileRef(name="x.txt"))


@pytest.mark.asyncio
async def test_list_pdfs_returns_only_pdfs() -> None:
    src = PdfExtractToolSource()
    files = [
        FileRef(name="a.pdf", data_base64=""),
        FileRef(name="b.txt", data_base64=""),
    ]
    tools = await src.tools(_make_ctx(files), {})
    list_pdfs = next(t for t in tools if t.name == "list_pdfs")
    with run_context.bind(_make_ctx(files)):
        result = list_pdfs.invoke({})
    assert {f["name"] for f in result["pdfs"]} == {"a.pdf"}


@pytest.mark.asyncio
async def test_pdf_extract_real_round_trip() -> None:
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Quick brown fox.")
    ref = FileRef(
        name="hello.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(pdf_bytes).decode(),
    )
    src = PdfExtractToolSource()
    tools = await src.tools(_make_ctx([ref]), {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    with run_context.bind(_make_ctx([ref])):
        out = await extract.ainvoke({"name": "hello.pdf", "include_words": True})
    assert out["name"] == "hello.pdf"
    assert len(out["pages"]) == 1
    assert "fox" in out["pages"][0]["text"].lower()
    assert all("x0" in w for w in out["pages"][0]["words"])


@pytest.mark.asyncio
async def test_pdf_extract_unknown_name_returns_error() -> None:
    src = PdfExtractToolSource()
    tools = await src.tools(_make_ctx([]), {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    with run_context.bind(_make_ctx([])):
        out = await extract.ainvoke({"name": "nope.pdf"})
    assert "error" in out


def test_coerce_page_range_all_shapes() -> None:
    assert _coerce_page_range(None) is None
    assert _coerce_page_range([1, 5]) == [1, 5]
    assert _coerce_page_range((1, 5)) == (1, 5)
    assert _coerce_page_range(7) == (7, 7)
    assert _coerce_page_range("  ") is None
    assert _coerce_page_range("[1, 5]") == [1, 5]
    assert _coerce_page_range("(2, 8)") == [2, 8]
    assert _coerce_page_range("[bad") == "[bad"
    assert _coerce_page_range("[1, 2, 3]") == "[1, 2, 3]"
    assert _coerce_page_range("1-5") == (1, 5)
    assert _coerce_page_range("3..9") == (3, 9)
    assert _coerce_page_range("4,6") == (4, 6)
    assert _coerce_page_range("a-b") == "a-b"
    assert _coerce_page_range("12") == (12, 12)
    assert _coerce_page_range("hello") == "hello"
    assert _coerce_page_range(3.5) == 3.5


def test_coerce_str_variants() -> None:
    assert _coerce_str(b"hi") == "hi"
    assert _coerce_str(None) == ""
    assert _coerce_str(42) == "42"
    assert isinstance(_coerce_str(b"\xff\xfe"), str)


def test_string_to_pdf_b64() -> None:
    assert _string_to_pdf_b64("") is None
    assert _string_to_pdf_b64(123) is None  # type: ignore[arg-type]
    assert _string_to_pdf_b64("  JVBERi0xLjQ=") == "JVBERi0xLjQ="
    data_url = "data:application/pdf;base64,JVBERi0xLjQ="
    assert _string_to_pdf_b64(data_url) == "JVBERi0xLjQ="
    assert _string_to_pdf_b64("data:application/pdf;base64,aGVsbG8=") is None
    raw = _string_to_pdf_b64("%PDF-1.4 body")
    assert raw is not None and base64.b64decode(raw).startswith(b"%PDF-")
    assert _string_to_pdf_b64("just text") is None


def test_find_any_http_url() -> None:
    assert _find_any_http_url("https://x.test/a.pdf") == "https://x.test/a.pdf"
    assert _find_any_http_url("data:application/pdf;base64,AAA").startswith("data:")
    assert _find_any_http_url("plain text") is None
    assert _find_any_http_url({"k": {"u": "http://nested.test/x"}}) == (
        "http://nested.test/x"
    )
    assert _find_any_http_url(["a", ["http://deep.test/y"]]) == "http://deep.test/y"
    assert _find_any_http_url({"k": "nothing"}) is None
    assert _find_any_http_url(42) is None
    nested: Any = "https://x.test/deep"
    for _ in range(8):
        nested = [nested]
    assert _find_any_http_url(nested) is None


def test_find_any_pdf_b64() -> None:
    assert _find_any_pdf_b64("JVBERi0xLjQ=") == "JVBERi0xLjQ="
    assert _find_any_pdf_b64({"a": {"b": "JVBERi0xLjQ="}}) == "JVBERi0xLjQ="
    assert _find_any_pdf_b64(["x", ["JVBERi0xLjQ="]]) == "JVBERi0xLjQ="
    assert _find_any_pdf_b64({"a": "no pdf"}) is None
    assert _find_any_pdf_b64(99) is None
    deep: Any = "JVBERi0xLjQ="
    for _ in range(8):
        deep = [deep]
    assert _find_any_pdf_b64(deep) is None


def test_bytes_from_data_url() -> None:
    payload = base64.b64encode(b"%PDF-x").decode()
    assert _bytes_from_data_url(f"data:application/pdf;base64,{payload}") == b"%PDF-x"
    assert _bytes_from_data_url(123) is None  # type: ignore[arg-type]
    assert _bytes_from_data_url("https://x.test/a.pdf") is None
    assert _bytes_from_data_url("data:application/pdf;charset=utf8,") is None
    assert _bytes_from_data_url("data:application/pdf;base64,ab") is None


def test_normalise_pdf_name() -> None:
    assert _normalise_pdf_name("  Report.PDF ") == "report"
    assert _normalise_pdf_name("plain") == "plain"


def test_match_uploaded_pdf() -> None:
    files = (
        FileRef(name="IEFA-Prospectus.pdf", mime="application/pdf"),
        FileRef(name="notes.txt"),
    )
    assert _match_uploaded_pdf(files, "IEFA-Prospectus.pdf").name == (
        "IEFA-Prospectus.pdf"
    )
    assert _match_uploaded_pdf(files, "iefa-prospectus").name == ("IEFA-Prospectus.pdf")
    assert _match_uploaded_pdf(files, "iefa").name == "IEFA-Prospectus.pdf"
    assert (
        _match_uploaded_pdf(files, "the iefa-prospectus document").name
        == "IEFA-Prospectus.pdf"
    )
    assert _match_uploaded_pdf((FileRef(name="x.txt"),), "x") is None
    assert _match_uploaded_pdf(files, "totally-different") is None
    assert _match_uploaded_pdf(files, ".pdf") is None


def test_resolve_pdf_bytes_from_data_base64() -> None:
    payload = base64.b64encode(b"%PDF-direct").decode()
    ref = FileRef(name="x.pdf", data_base64=payload)
    assert _resolve_pdf_bytes(ref, http_client=None) == b"%PDF-direct"


def test_resolve_pdf_bytes_from_data_url() -> None:
    payload = base64.b64encode(b"%PDF-url").decode()
    ref = FileRef(name="x.pdf", url=f"data:application/pdf;base64,{payload}")
    assert _resolve_pdf_bytes(ref, http_client=None) == b"%PDF-url"


def test_resolve_pdf_bytes_from_extras_b64() -> None:
    """Recover a bare base64 PDF payload sitting in an extra field."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Extras b64")
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    ref = FileRef(name="x.pdf", content=pdf_b64)
    out = _resolve_pdf_bytes(ref, http_client=None)
    assert out == pdf_bytes


def test_resolve_pdf_bytes_from_extras_url() -> None:
    class _Resp:
        content = b"%PDF-fetched"

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def get(self, url: str, headers: dict | None = None) -> _Resp:
            return _Resp()

    ref = FileRef(name="x.pdf", file_uri="https://x.test/remote.pdf")
    assert _resolve_pdf_bytes(ref, _Client()) == b"%PDF-fetched"


def test_resolve_pdf_bytes_extras_data_url() -> None:
    pdf_b64 = base64.b64encode(b"%PDF-inline").decode()
    ref = FileRef(name="x.pdf", blob=f"data:application/pdf;base64,{pdf_b64}")
    assert _resolve_pdf_bytes(ref, http_client=None) == b"%PDF-inline"


def test_resolve_pdf_bytes_extras_url_decodes_data_url() -> None:
    """Decode locally an extras URL that is itself a data: URL."""
    pdf_b64 = base64.b64encode(b"%PDF-payload-text").decode()
    ref = FileRef(
        name="x.pdf",
        somefield="data:application/octet-stream;base64," + pdf_b64,
    )
    out = _resolve_pdf_bytes(ref, http_client=None)
    assert out == b"%PDF-payload-text"


def test_extract_toc_and_metadata_from_outlined_pdf() -> None:
    """Extract TOC and metadata from a real outlined PDF."""
    import pdfplumber

    from openbb_agent_server.plugins.tools.pdf_extract import (
        _extract_pdf_metadata,
        _extract_toc,
    )

    pdf_bytes = _build_one_page_pdf("Outlined", pages=3, outline=True)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        toc = _extract_toc(pdf)
        meta = _extract_pdf_metadata(pdf)
    assert len(toc) == 3
    assert {e["title"] for e in toc} >= {"Section 1"}
    assert all(e["page"] for e in toc)
    assert meta["title"] == "Annual Report"
    assert meta["author"] == "Analyst"
    assert meta["total_pages"] == 3


def test_extract_toc_no_outline_returns_empty() -> None:
    """Yield an empty TOC for a PDF with no embedded outline."""
    import pdfplumber

    from openbb_agent_server.plugins.tools.pdf_extract import _extract_toc

    pdf_bytes = _build_one_page_pdf("Plain")
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        assert _extract_toc(pdf) == []


class _StubPageObj:
    """A pdfminer page object stub."""

    def __init__(self, pageid: int) -> None:
        self.pageid = pageid
        self.objid = pageid


class _StubPlumberPage:
    def __init__(self, idx: int, *, has_obj: bool = True) -> None:
        if has_obj:
            self.page_obj = _StubPageObj(idx)


class _StubDoc:
    def __init__(self, outlines: list) -> None:
        self._outlines = outlines

    def get_outlines(self) -> list:
        return self._outlines


class _StubPlumberPdf:
    def __init__(self, outlines: list, *, meta: Any = None) -> None:
        self.pages = [_StubPlumberPage(1), _StubPlumberPage(2, has_obj=False)]
        self.doc = _StubDoc(outlines)
        self._meta = meta

    @property
    def metadata(self) -> Any:
        if isinstance(self._meta, Exception):
            raise self._meta
        return self._meta


def test_extract_toc_handles_malformed_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle malformed outline entries defensively."""
    from pdfminer import pdftypes

    from openbb_agent_server.plugins.tools.pdf_extract import _extract_toc

    def _fake_resolve1(dest: Any) -> Any:
        if dest == "raise":
            raise RuntimeError("resolve boom")
        if dest == "list-objid":
            return [_StubPageObj(1)]
        if dest == "scalar-objid":
            return _StubPageObj(1)
        return None

    monkeypatch.setattr(pdftypes, "resolve1", _fake_resolve1)
    pdf = _StubPlumberPdf(
        [
            (1, "Resolved", "list-objid", None, None),
            (1, "ScalarObjid", "scalar-objid", None, None),
            ("bad", "tuple"),
            (1, "RaiseDest", "raise", None, None),
            (1, "Unresolvable", None, None, None),
        ]
    )
    toc = _extract_toc(pdf)
    titles = [e["title"] for e in toc]
    assert titles == ["Resolved", "ScalarObjid", "RaiseDest", "Unresolvable"]
    assert toc[0]["page"] == 1
    assert toc[1]["page"] == 1
    assert toc[3]["page"] is None


def test_extract_toc_outline_access_failure() -> None:
    """Yield an empty TOC when get_outlines raises."""
    from openbb_agent_server.plugins.tools.pdf_extract import _extract_toc

    class _Doc:
        def get_outlines(self) -> list:
            raise RuntimeError("no outline")

    class _Pdf:
        doc = _Doc()
        pages: list = []

    assert _extract_toc(_Pdf()) == []


def test_resolve_outline_page_without_pdfminer() -> None:
    """Return None for non-resolvable outline dests."""
    from openbb_agent_server.plugins.tools.pdf_extract import _resolve_outline_page

    assert _resolve_outline_page(None, {}) is None


def test_extract_pdf_metadata_access_failure() -> None:
    """Yield all-None metadata when the pdf.metadata accessor raises."""
    from openbb_agent_server.plugins.tools.pdf_extract import _extract_pdf_metadata

    pdf = _StubPlumberPdf([], meta=RuntimeError("metadata blew up"))
    meta = _extract_pdf_metadata(pdf)
    assert meta["title"] is None
    assert meta["total_pages"] == 2


def test_resolve_pdf_bytes_raises_without_any_source() -> None:
    """Raise for a FileRef with no url, data_base64, or usable extras."""
    ref = FileRef(name="orphan.pdf")
    with pytest.raises(RuntimeError, match="has no url or data_base64"):
        _resolve_pdf_bytes(ref, http_client=None)


def test_resolve_pdf_bytes_url_branch() -> None:
    class _Resp:
        content = b"%PDF-net"

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def get(self, url: str, headers: dict | None = None) -> _Resp:
            return _Resp()

    ref = FileRef(name="x.pdf", url="https://x.test/a.pdf")
    assert _resolve_pdf_bytes(ref, _Client()) == b"%PDF-net"


@pytest.mark.asyncio
async def test_list_pdfs_second_call_returns_stop_message() -> None:
    ctx = _make_ctx([FileRef(name="a.pdf", mime="application/pdf")])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    list_pdfs = next(t for t in tools if t.name == "list_pdfs")
    with run_context.bind(ctx):
        first = list_pdfs.invoke({})
        second = list_pdfs.invoke({})
    assert first["count"] == 1
    assert "already called" in second["message"]


@pytest.mark.asyncio
async def test_list_pdfs_empty_surfaces_document_widgets() -> None:
    widget = WidgetRef(uuid="w1", widget_id="blk_document_viewer", name="Docs")
    ctx = _make_ctx([], widgets=[widget])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    list_pdfs = next(t for t in tools if t.name == "list_pdfs")
    with run_context.bind(ctx):
        out = list_pdfs.invoke({})
    assert out["count"] == 0
    assert out["document_widgets"][0]["uuid"] == "w1"


@pytest.mark.asyncio
async def test_pdf_extract_preview_range_when_omitted() -> None:
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Multi", pages=5)
    ref = FileRef(
        name="big.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(pdf_bytes).decode(),
    )
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    with run_context.bind(ctx):
        out = await extract.ainvoke({"name": "big.pdf"})
    assert out["is_preview"] is True
    assert out["page_range"] == [1, 3]
    assert len(out["pages"]) == 3
    assert "words" not in out["pages"][0]


@pytest.mark.asyncio
async def test_pdf_extract_fetch_failure_returns_error() -> None:
    """Surface an error dict, not an exception, on a bytes-fetch failure."""

    class _Client:
        def get(self, url: str, headers: dict | None = None) -> Any:
            raise RuntimeError("network down")

    ref = FileRef(name="remote.pdf", mime="application/pdf", url="https://x/r.pdf")
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    import httpx

    orig = httpx.Client
    httpx.Client = lambda *a, **k: _Client()  # type: ignore[assignment,misc]
    try:
        tools = await src.tools(ctx, {})
        extract = next(t for t in tools if t.name == "pdf_extract")
        with run_context.bind(ctx):
            out = await extract.ainvoke({"name": "remote.pdf"})
    finally:
        httpx.Client = orig  # type: ignore[misc]
    assert "failed to fetch pdf bytes" in out["error"]


@pytest.mark.asyncio
async def test_pdf_extract_parse_failure_returns_error() -> None:
    """Surface a parse error dict for undecodable PDF bytes."""
    ref = FileRef(
        name="junk.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(b"not a pdf").decode(),
    )
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    with run_context.bind(ctx):
        out = await extract.ainvoke({"name": "junk.pdf"})
    assert "failed to parse PDF" in out["error"]


@pytest.mark.asyncio
async def test_pdf_extract_uses_ready_store() -> None:
    """Read from the store when it has the PDF ready."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Stored content", pages=2)
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    principal = UserPrincipal(user_id="u")
    store = await _ready_store(principal, "doc.pdf", pdf_bytes, data_base64=pdf_b64)
    services.set_services(pdf_store=store)
    ref = FileRef(name="doc.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await extract.ainvoke({"name": "doc.pdf", "include_words": True})
    assert out["name"] == "doc.pdf"
    assert out["total_pages"] == 2
    assert len(out["pages"]) == 2
    assert any(e["type"] == "citations" for e in sink)
    await store.aclose()


@pytest.mark.asyncio
async def test_pdf_extract_store_error_status() -> None:
    """Surface a background-ingest error for a store row in error status."""
    pdf_b64 = base64.b64encode(b"garbage bytes").decode()
    principal = UserPrincipal(user_id="u")
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    async with store.engine.begin() as conn:
        await conn.run_sync(m.Base.metadata.create_all)
    await store.ingest_async(principal=principal, name="bad.pdf", data_base64=pdf_b64)
    await store.await_pending()
    services.set_services(pdf_store=store)
    ref = FileRef(name="bad.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    with run_context.bind(ctx):
        out = await extract.ainvoke({"name": "bad.pdf"})
    assert "failed background ingestion" in out["error"]
    await store.aclose()


@pytest.mark.asyncio
async def test_pdf_extract_polls_then_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the inline parse path after status polling yields None."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Poll fallthrough")
    pdf_b64 = base64.b64encode(pdf_bytes).decode()

    class _EmptyStore:
        async def status(self, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(services, "get_pdf_store", lambda: _EmptyStore())
    monkeypatch.setattr(pe.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(pe, "_PREVIEW_PAGES", 3)
    ref = FileRef(name="poll.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await extract.ainvoke({"name": "poll.pdf"})
    assert out["name"] == "poll.pdf"
    assert any("ingestion" in str(e.get("message", "")) for e in sink)


async def _fast_sleep(_seconds: float) -> None:
    """Replace asyncio.sleep with a no-op to fast-forward poll loops."""
    return None


@pytest.mark.asyncio
async def test_pdf_extract_polls_until_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transition the poll loop pending to ready and read from the store."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Becomes ready", pages=2)
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    principal = UserPrincipal(user_id="u")
    store = await _ready_store(principal, "r.pdf", pdf_bytes, data_base64=pdf_b64)

    real_status = store.status
    calls = {"n": 0}

    async def _slow_status(**kwargs: Any) -> dict[str, Any] | None:
        calls["n"] += 1
        if calls["n"] < 3:
            return {"status": "pending", "total_pages": 0, "metadata": {}, "toc": []}
        return await real_status(**kwargs)

    store.status = _slow_status  # type: ignore[method-assign]
    services.set_services(pdf_store=store)
    monkeypatch.setattr(pe.asyncio, "sleep", _fast_sleep)
    ref = FileRef(name="r.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await extract.ainvoke({"name": "r.pdf"})
    assert out["total_pages"] == 2
    await store.aclose()


@pytest.mark.asyncio
async def test_pdf_extract_poll_errors_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface the error when status flips to error mid-poll."""
    calls = {"n": 0}

    class _Store:
        async def status(self, **kwargs: Any) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] < 2:
                return {
                    "status": "pending",
                    "total_pages": 0,
                    "metadata": {},
                    "toc": [],
                }
            return {
                "status": "error",
                "error": "boom",
                "total_pages": 0,
                "metadata": {},
                "toc": [],
            }

    monkeypatch.setattr(services, "get_pdf_store", lambda: _Store())
    monkeypatch.setattr(pe.asyncio, "sleep", _fast_sleep)
    ref = FileRef(
        name="e.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(b"x").decode(),
    )
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await extract.ainvoke({"name": "e.pdf"})
    assert "failed background ingestion" in out["error"]


@pytest.mark.asyncio
async def test_pdf_extract_poll_timeout_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit heartbeats then fall through to the inline parse on a stuck store."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Timeout then inline")
    pdf_b64 = base64.b64encode(pdf_bytes).decode()

    class _StuckStore:
        async def status(self, **kwargs: Any) -> dict[str, Any]:
            return {"status": "pending", "total_pages": 0, "metadata": {}, "toc": []}

    monkeypatch.setattr(services, "get_pdf_store", lambda: _StuckStore())
    monkeypatch.setattr(pe.asyncio, "sleep", _fast_sleep)
    ref = FileRef(name="t.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await extract.ainvoke({"name": "t.pdf"})
    assert out["name"] == "t.pdf"
    assert any("Still ingesting" in str(e.get("message", "")) for e in sink)


@pytest.mark.asyncio
async def test_search_pdf_no_store_returns_empty() -> None:
    ctx = _make_ctx([])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    search = next(t for t in tools if t.name == "search_pdf")
    with run_context.bind(ctx):
        out = await search.ainvoke({"query": "anything"})
    assert out == []


@pytest.mark.asyncio
async def test_search_pdf_with_matched_uploads() -> None:
    """Produce a widget citation for a hit whose name matches an upload."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Searchable revenue figures", pages=2)
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    principal = UserPrincipal(user_id="u")
    store = await _ready_store(principal, "fin.pdf", pdf_bytes, data_base64=pdf_b64)
    services.set_services(pdf_store=store)
    ref = FileRef(name="fin.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    search = next(t for t in tools if t.name == "search_pdf")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        hits = await search.ainvoke({"query": "revenue"})
    assert hits and hits[0]["name"] == "fin.pdf"
    assert any(e["type"] == "citations" for e in sink)
    await store.aclose()


@pytest.mark.asyncio
async def test_search_pdf_unmatched_name_falls_back_to_name_chip() -> None:
    """Give a plain name-only citation to hits whose name has no FileRef."""

    class _Store:
        async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
            return [
                {"name": "ghost.pdf", "page": 4, "text": "x"},
                {"name": "", "page": 1, "text": "skip"},
                {"name": "ghost.pdf", "page": 4, "text": "dup"},
                {"name": "ghost.pdf", "page": 0, "text": "no page"},
            ]

    services.set_services(pdf_store=_Store())
    ctx = _make_ctx([])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    search = next(t for t in tools if t.name == "search_pdf")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        hits = await search.ainvoke({"query": "q"})
    assert len(hits) == 4
    cites = [e for e in sink if e["type"] == "citations"]
    assert cites


@pytest.mark.asyncio
async def test_search_pdf_hit_with_no_page_data() -> None:
    """Emit a citation for a matched-upload hit whose page is missing."""

    class _Store:
        async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
            return [
                {"name": "doc.pdf", "page": 9, "text": "hit text"},
                {"name": "doc.pdf", "page": 0, "text": "no page"},
            ]

        async def get_pages(self, **kwargs: Any) -> list[dict[str, Any]]:
            return []

    services.set_services(pdf_store=_Store())
    ref = FileRef(name="doc.pdf", mime="application/pdf", data_base64="AAA")
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    search = next(t for t in tools if t.name == "search_pdf")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        hits = await search.ainvoke({"query": "q"})
    assert hits[0]["page"] == 9
    assert any(e["type"] == "citations" for e in sink)


@pytest.mark.asyncio
async def test_get_pdf_outline_unknown_name() -> None:
    ctx = _make_ctx([])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    outline = next(t for t in tools if t.name == "get_pdf_outline")
    with run_context.bind(ctx):
        out = await outline.ainvoke({"name": "nope.pdf"})
    assert "not found" in out["error"]


@pytest.mark.asyncio
async def test_get_pdf_outline_no_store() -> None:
    ref = FileRef(name="a.pdf", mime="application/pdf", data_base64="AAA")
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    outline = next(t for t in tools if t.name == "get_pdf_outline")
    with run_context.bind(ctx):
        out = await outline.ainvoke({"name": "a.pdf"})
    assert "not configured" in out["error"]


@pytest.mark.asyncio
async def test_get_pdf_outline_ready_fast_path() -> None:
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Outline ready", pages=2)
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    principal = UserPrincipal(user_id="u")
    store = await _ready_store(principal, "o.pdf", pdf_bytes, data_base64=pdf_b64)
    services.set_services(pdf_store=store)
    ref = FileRef(name="o.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    outline = next(t for t in tools if t.name == "get_pdf_outline")
    with run_context.bind(ctx):
        out = await outline.ainvoke({"name": "o.pdf"})
    assert out["total_pages"] == 2
    assert "toc" in out
    await store.aclose()


@pytest.mark.asyncio
async def test_get_pdf_outline_error_status() -> None:
    pdf_b64 = base64.b64encode(b"garbage").decode()
    principal = UserPrincipal(user_id="u")
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    async with store.engine.begin() as conn:
        await conn.run_sync(m.Base.metadata.create_all)
    await store.ingest_async(principal=principal, name="bad.pdf", data_base64=pdf_b64)
    await store.await_pending()
    services.set_services(pdf_store=store)
    ref = FileRef(name="bad.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    outline = next(t for t in tools if t.name == "get_pdf_outline")
    with run_context.bind(ctx):
        out = await outline.ainvoke({"name": "bad.pdf"})
    assert "failed background ingestion" in out["error"]
    await store.aclose()


@pytest.mark.asyncio
async def test_get_pdf_outline_polls_until_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transition the poll loop pending to ready, emitting reasoning steps."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Outline poll", pages=2)
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    principal = UserPrincipal(user_id="u")
    store = await _ready_store(principal, "p.pdf", pdf_bytes, data_base64=pdf_b64)
    real_status = store.status
    calls = {"n": 0}

    async def _slow_status(**kwargs: Any) -> dict[str, Any] | None:
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        if calls["n"] < 3:
            return {"status": "pending", "total_pages": 0, "metadata": {}, "toc": []}
        return await real_status(**kwargs)

    store.status = _slow_status  # type: ignore[method-assign]
    services.set_services(pdf_store=store)
    monkeypatch.setattr(pe.asyncio, "sleep", _fast_sleep)
    ref = FileRef(name="p.pdf", mime="application/pdf", data_base64=pdf_b64)
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    outline = next(t for t in tools if t.name == "get_pdf_outline")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await outline.ainvoke({"name": "p.pdf"})
    assert out["total_pages"] == 2
    await store.aclose()


@pytest.mark.asyncio
async def test_get_pdf_outline_poll_errors_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    class _Store:
        async def status(self, **kwargs: Any) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] < 2:
                return {
                    "status": "pending",
                    "total_pages": 0,
                    "metadata": {},
                    "toc": [],
                }
            return {
                "status": "error",
                "error": "kaboom",
                "total_pages": 0,
                "metadata": {},
                "toc": [],
            }

    monkeypatch.setattr(services, "get_pdf_store", lambda: _Store())
    monkeypatch.setattr(pe.asyncio, "sleep", _fast_sleep)
    ref = FileRef(name="e.pdf", mime="application/pdf", data_base64="AAA")
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    outline = next(t for t in tools if t.name == "get_pdf_outline")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await outline.ainvoke({"name": "e.pdf"})
    assert "failed background ingestion" in out["error"]


@pytest.mark.asyncio
async def test_get_pdf_outline_poll_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the timeout error for a store stuck in pending."""

    class _StuckStore:
        async def status(self, **kwargs: Any) -> dict[str, Any]:
            return {"status": "pending", "total_pages": 0, "metadata": {}, "toc": []}

    monkeypatch.setattr(services, "get_pdf_store", lambda: _StuckStore())
    monkeypatch.setattr(pe.asyncio, "sleep", _fast_sleep)
    ref = FileRef(name="t.pdf", mime="application/pdf", data_base64="AAA")
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    outline = next(t for t in tools if t.name == "get_pdf_outline")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await outline.ainvoke({"name": "t.pdf"})
    assert "still being indexed" in out["error"]
    assert any("Still ingesting" in str(e.get("message", "")) for e in sink)


@pytest.mark.asyncio
async def test_pdf_extract_citation_deduped_across_calls() -> None:
    """Emit one citation for two pdf_extract calls on the same PDF page."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Dedup me")
    ref = FileRef(
        name="d.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(pdf_bytes).decode(),
    )
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        await extract.ainvoke({"name": "d.pdf"})
        await extract.ainvoke({"name": "d.pdf"})
    assert sum(e["type"] == "citations" for e in sink) == 1


@pytest.mark.asyncio
async def test_pdf_extract_empty_page_range_emits_no_citation() -> None:
    """Emit no citation when page_range selects no pages."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Single page")
    ref = FileRef(
        name="one.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(pdf_bytes).decode(),
    )
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await extract.ainvoke({"name": "one.pdf", "page_range": [50, 60]})
    assert out["pages"] == []
    assert not any(e["type"] == "citations" for e in sink)


@pytest.mark.asyncio
async def test_pdf_extract_blank_page_anchors_first_page() -> None:
    """Anchor the citation to page one when no extracted page has text."""
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()
    c.save()
    ref = FileRef(
        name="blank.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(buf.getvalue()).decode(),
    )
    ctx = _make_ctx([ref])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        out = await extract.ainvoke({"name": "blank.pdf"})
    assert out["pages"][0]["text"] == ""
    assert any(e["type"] == "citations" for e in sink)


@pytest.mark.asyncio
async def test_pdf_extract_widget_param_extraction_failure() -> None:
    """Swallow input_args to an empty dict when widget params raise."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Apple inc")

    class _BadParams(dict):
        def __iter__(self) -> Any:
            raise RuntimeError("params iteration blew up")

    widget = WidgetRef(uuid="bw", widget_id="blk_docs", name="Docs")
    object.__setattr__(widget, "params", _BadParams({"apple": "apple"}))
    ref = FileRef(
        name="apple.pdf",
        mime="application/pdf",
        data_base64=base64.b64encode(pdf_bytes).decode(),
        source_widget_uuid="bw",
    )
    ctx = _make_ctx([ref], widgets=[widget])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        await extract.ainvoke({"name": "apple.pdf"})
    cite = next(e for e in sink if e["type"] == "citations")
    assert cite["citations"][0]["source_info"]["uuid"] == "bw"


@pytest.mark.asyncio
async def test_pdf_extract_citation_resolves_source_widget() -> None:
    """Resolve the citation to the widget whose params match PDF tokens."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Apple revenue")
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    widget = WidgetRef(
        uuid="wuuid-1",
        widget_id="blk_drill_fund_documents",
        name="Fund Docs",
        origin="custom-backend",
        params={"ticker": "apple-2024-prospectus", "tags": ["filing"]},
    )
    ref = FileRef(
        name="apple-2024-prospectus.pdf",
        mime="application/pdf",
        data_base64=pdf_b64,
        source_widget_uuid="wuuid-1",
        source_widget_id="blk_drill_fund_documents",
    )
    ctx = _make_ctx([ref], widgets=[widget])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        await extract.ainvoke({"name": "apple-2024-prospectus.pdf"})
    cite = next(e for e in sink if e["type"] == "citations")
    src_info = cite["citations"][0]["source_info"]
    assert src_info["type"] == "widget"
    assert src_info["uuid"] == "wuuid-1"


@pytest.mark.asyncio
async def test_pdf_extract_citation_stamp_and_list_params() -> None:
    """Fall back to the source_widget_uuid stamp when content match fails."""
    pytest.importorskip("reportlab")
    pdf_bytes = _build_one_page_pdf("Quarterly")
    pdf_b64 = base64.b64encode(pdf_bytes).decode()

    class _Param:
        def __init__(self, name: str, value: str) -> None:
            self.name = name
            self.current_value = value

    widget = WidgetRef(
        uuid="stamp-uuid",
        widget_id="blk_viewer",
        name="Viewer",
    )
    object.__setattr__(widget, "params", [_Param("region", "us")])
    ref = FileRef(
        name="zzz-unrelated-name.pdf",
        mime="application/pdf",
        data_base64=pdf_b64,
        source_widget_uuid="stamp-uuid",
    )
    ctx = _make_ctx([ref], widgets=[widget])
    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    sink: list[dict[str, Any]] = []
    with run_context.bind(ctx), emit.bind_writer(sink.append):
        await extract.ainvoke({"name": "zzz-unrelated-name.pdf"})
    cite = next(e for e in sink if e["type"] == "citations")
    assert cite["citations"][0]["source_info"]["uuid"] == "stamp-uuid"


def test_pdf_extract_resolve_pdf_bytes_fetches_url() -> None:
    from openbb_agent_server.plugins.tools.pdf_extract import _resolve_pdf_bytes

    class _FakeResp:
        content = b"PDF-BYTES"

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def get(self, url: str, headers: dict | None = None) -> _FakeResp:
            return _FakeResp()

    ref = FileRef(name="x.pdf", url="https://x.test/x.pdf")
    assert _resolve_pdf_bytes(ref, _FakeClient()) == b"PDF-BYTES"


def test_pdf_extract_resolve_pdf_bytes_raises_when_no_source() -> None:
    from openbb_agent_server.plugins.tools.pdf_extract import _resolve_pdf_bytes

    ref = FileRef(name="x.pdf")
    with pytest.raises(RuntimeError, match="has no url or data_base64"):
        _resolve_pdf_bytes(ref, http_client=None)


@pytest.mark.asyncio
async def test_pdf_extract_page_range_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    from openbb_agent_server.plugins.tools.pdf_extract import PdfExtractToolSource

    class _Page:
        def __init__(self, idx: int) -> None:
            self._idx = idx

        def extract_text(self) -> str:
            return f"page-{self._idx}"

        def extract_words(self) -> list[dict[str, Any]]:
            return []

    class _Pdf:
        pages = [_Page(1), _Page(2), _Page(3)]

        def __enter__(self) -> _Pdf:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    class _Plumber:
        def open(self, _stream: Any) -> _Pdf:
            return _Pdf()

    monkeypatch.setitem(__import__("sys").modules, "pdfplumber", _Plumber())

    pdf_b64 = base64.b64encode(b"%PDF-fake").decode()
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        uploaded_files=(
            FileRef(name="f.pdf", mime="application/pdf", data_base64=pdf_b64),
        ),
    )

    src = PdfExtractToolSource()
    tools = await src.tools(ctx, {})
    extract = next(t for t in tools if t.name == "pdf_extract")
    with bind(ctx):
        out = await extract.ainvoke({"name": "f.pdf", "page_range": [2, 2]})
    pages = out["pages"]
    assert len(pages) == 1 and pages[0]["page"] == 2 and pages[0]["text"] == "page-2"
