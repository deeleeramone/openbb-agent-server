"""PdfStore unit tests."""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

from openbb_agent_server.persistence import models as m
from openbb_agent_server.runtime.pdf_store import (
    PdfStore,
    _apply_sqlite_pragmas,
    _build_vec_connection,
    _file_key,
    _parse_pdf_sync,
    _url_to_file,
)
from openbb_agent_server.runtime.principal import UserPrincipal


def _build_pdf(
    text: str = "Hello World", *, pages: int = 1, outline: bool = False
) -> bytes:
    """Build a minimal real multi-page PDF via reportlab."""
    pytest.importorskip("pdfplumber")
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
        c.setTitle("Quarterly Report")
        c.setAuthor("Jane Analyst")
        c.setSubject("Earnings")
    c.save()
    return buf.getvalue()


async def _create_schema(store: PdfStore) -> None:
    async with store.engine.begin() as conn:
        await conn.run_sync(m.Base.metadata.create_all)


def _principal(uid: str = "u") -> UserPrincipal:
    return UserPrincipal(user_id=uid)


def test_url_to_file_variants() -> None:
    assert _url_to_file("sqlite+aiosqlite:///:memory:") is None
    assert _url_to_file("sqlite+aiosqlite:///") is None
    assert _url_to_file("sqlite+aiosqlite:////tmp/x.db") == "/tmp/x.db"
    assert _url_to_file("postgresql://host/db") == "postgresql://host/db"


def test_apply_sqlite_pragmas_skips_non_sqlite() -> None:
    """Non-sqlite URLs return before registering the connect listener."""

    class _Engine:
        pass

    _apply_sqlite_pragmas(_Engine(), "postgresql://host/db")


def test_apply_sqlite_pragmas_registers_listener(tmp_path: Path) -> None:
    """The ``connect`` listener runs the PRAGMA statements on connect."""
    from sqlalchemy import create_engine, text

    eng = create_engine(f"sqlite:///{tmp_path / 'p.db'}")

    class _AsyncShim:
        sync_engine = eng

    _apply_sqlite_pragmas(_AsyncShim(), f"sqlite:///{tmp_path / 'p.db'}")
    with eng.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert str(mode).lower() == "wal"
    eng.dispose()


def test_build_vec_connection(tmp_path: Path) -> None:
    conn = _build_vec_connection(str(tmp_path / "vec.db"))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_file_key_branches() -> None:
    by_url = _file_key("a.pdf", "https://x/a.pdf", None)
    by_b64 = _file_key("a.pdf", None, "JVBERi0xLjQ=")
    by_name = _file_key("a.pdf", None, None)
    assert by_url != by_b64 != by_name
    assert _file_key("renamed.pdf", "https://x/a.pdf", None) == by_url
    assert _file_key("b.pdf", None, None) != by_name


@pytest.mark.asyncio
async def test_engine_property_and_status_unknown() -> None:
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    assert store.engine is not None
    st = await store.status(principal=_principal(), name="missing.pdf")
    assert st is None
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_init_with_external_engine_skips_pragmas() -> None:
    """Passing an explicit engine bypasses _apply_sqlite_pragmas."""
    from sqlalchemy.ext.asyncio import create_async_engine

    eng = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    store = PdfStore("sqlite+aiosqlite:///:memory:", engine=eng)
    assert store.engine is eng
    await eng.dispose()


@pytest.mark.asyncio
async def test_get_pages_returns_none_when_not_ready() -> None:
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    pages = await store.get_pages(principal=_principal(), name="x.pdf")
    assert pages is None
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_ingest_status_get_pages_round_trip() -> None:
    """Full happy path: ingest a real PDF, poll to ready, read pages."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    pdf_b64 = base64.b64encode(_build_pdf("Round trip", pages=3)).decode()
    p = _principal()
    key = await store.ingest_async(
        principal=p, name="rt.pdf", data_base64=pdf_b64, mime="application/pdf"
    )
    assert isinstance(key, str)
    await store.await_pending()
    st = await store.status(principal=p, name="rt.pdf", data_base64=pdf_b64)
    assert st is not None and st["status"] == "ready"
    assert st["total_pages"] == 3
    assert st["metadata"]["total_pages"] == 3
    assert isinstance(st["toc"], list)
    all_pages = await store.get_pages(principal=p, name="rt.pdf", data_base64=pdf_b64)
    assert all_pages is not None and len(all_pages) == 3
    ranged = await store.get_pages(
        principal=p, name="rt.pdf", data_base64=pdf_b64, page_range=(2, 2)
    )
    assert ranged is not None and len(ranged) == 1 and ranged[0]["page"] == 2
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_ingest_skips_when_already_ready() -> None:
    """A second ingest of ready content returns early without rescheduling."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    pdf_b64 = base64.b64encode(_build_pdf()).decode()
    p = _principal()
    await store.ingest_async(principal=p, name="a.pdf", data_base64=pdf_b64)
    await store.await_pending()
    store._tasks.clear()
    await store.ingest_async(principal=p, name="a.pdf", data_base64=pdf_b64)
    assert not store._tasks
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_ingest_skips_when_in_flight() -> None:
    """A duplicate ingest while the first is still running is deduped."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    pdf_b64 = base64.b64encode(_build_pdf()).decode()
    p = _principal()

    started = asyncio.Event()
    release = asyncio.Event()
    original = store._ingest_in_background

    async def _slow(**kwargs: Any) -> None:
        started.set()
        await release.wait()
        await original(**kwargs)

    store._ingest_in_background = _slow  # type: ignore[method-assign]
    key1 = await store.ingest_async(principal=p, name="a.pdf", data_base64=pdf_b64)
    await started.wait()
    key2 = await store.ingest_async(principal=p, name="a.pdf", data_base64=pdf_b64)
    assert key1 == key2
    assert len(store._tasks) == 1
    release.set()
    await store.await_pending()
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_await_pending_noop_when_empty() -> None:
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await store.await_pending()
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_ingest_records_error_on_parse_failure() -> None:
    """Unparsable bytes land the document in ``error`` status."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    bad_b64 = base64.b64encode(b"not a pdf at all").decode()
    p = _principal()
    await store.ingest_async(principal=p, name="bad.pdf", data_base64=bad_b64)
    await store.await_pending()
    st = await store.status(principal=p, name="bad.pdf", data_base64=bad_b64)
    assert st is not None and st["status"] == "error"
    assert st["error"]
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_ingest_reuses_existing_pending_row() -> None:
    """A re-ingest after a prior error re-pends the existing row."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    bad_b64 = base64.b64encode(b"garbage").decode()
    p = _principal()
    await store.ingest_async(principal=p, name="bad.pdf", data_base64=bad_b64)
    await store.await_pending()
    key = _file_key("bad.pdf", None, bad_b64)
    good_b64 = base64.b64encode(_build_pdf()).decode()
    await store._ingest_in_background(
        principal=p,
        file_key=key,
        name="bad.pdf",
        url=None,
        data_base64=good_b64,
        mime=None,
    )
    st = await store.status(principal=p, name="bad.pdf", data_base64=bad_b64)
    assert st is not None and st["status"] == "ready"
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_ingest_background_returns_when_existing_ready() -> None:
    """_ingest_in_background returns early if the row is already ready."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    pdf_b64 = base64.b64encode(_build_pdf()).decode()
    p = _principal()
    await store.ingest_async(principal=p, name="a.pdf", data_base64=pdf_b64)
    await store.await_pending()
    key = _file_key("a.pdf", None, pdf_b64)
    await store._ingest_in_background(
        principal=p,
        file_key=key,
        name="a.pdf",
        url=None,
        data_base64=pdf_b64,
        mime=None,
    )
    st = await store.status(principal=p, name="a.pdf", data_base64=pdf_b64)
    assert st is not None and st["status"] == "ready"
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_substring_search_no_documents() -> None:
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    hits = await store.search(principal=_principal(), query="anything")
    assert hits == []
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_search_blank_query_returns_empty() -> None:
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    assert await store.search(principal=_principal(), query="   ") == []
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_substring_search_matches_page_text() -> None:
    """No vector index → ``search`` falls back to substring matching."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    pdf_b64 = base64.b64encode(_build_pdf("Searchable needle", pages=2)).decode()
    p = _principal()
    await store.ingest_async(principal=p, name="s.pdf", data_base64=pdf_b64)
    await store.await_pending()
    hits = await store.search(principal=p, query="needle", k=5)
    assert hits and hits[0]["name"] == "s.pdf"
    assert hits[0]["score"] == 1.0
    assert await store.search(principal=p, query="zzzznomatch", k=5) == []
    await store.engine.dispose()


class _StubEmbeddings(Embeddings):
    """Tiny deterministic embedding for the SQLiteVec path."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)

    @staticmethod
    def _one(text: str) -> list[float]:
        vec = [0.0] * 8
        for i, ch in enumerate(text.lower()):
            vec[ord(ch) % 8] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


@pytest.mark.asyncio
async def test_vector_index_and_search(tmp_path: Path) -> None:
    """With embeddings + a file DB, ingestion builds vectors and ANN runs."""
    db_path = tmp_path / "vec_store.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    store = PdfStore(url, embeddings=_StubEmbeddings())
    await _create_schema(store)
    assert store._vec is not None
    pdf_b64 = base64.b64encode(_build_pdf("Quarterly revenue report", pages=2)).decode()
    p = _principal()
    await store.ingest_async(principal=p, name="v.pdf", data_base64=pdf_b64)
    await store.await_pending()
    hits = await store.search(principal=p, query="revenue", k=3)
    assert isinstance(hits, list)
    assert all("score" in h and "page" in h for h in hits)
    other = await store.search(principal=_principal("other"), query="revenue", k=3)
    assert other == []
    await store.engine.dispose()
    if store._vec_conn is not None:
        store._vec_conn.close()


@pytest.mark.asyncio
async def test_vector_search_dedup_and_k_limit(tmp_path: Path) -> None:
    """ANN results are deduped per ``(doc_id, page)`` and capped at ``k``."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'dk.db'}"
    store = PdfStore(url, embeddings=_StubEmbeddings())
    await _create_schema(store)
    p = _principal()

    class _Doc:
        def __init__(self, doc_id: int, page: int, uid: str = "u") -> None:
            self.metadata = {
                "doc_id": doc_id,
                "page": page,
                "user_id": uid,
                "name": "d.pdf",
            }
            self.page_content = f"page {page}"

    scored = [
        (_Doc(9, 9, "other"), 0.1),
        (_Doc(1, 1), 0.2),
        (_Doc(1, 1), 0.3),
        (_Doc(1, 2), 0.4),
        (_Doc(1, 3), 0.5),
    ]
    store._vec.similarity_search_with_score = (  # type: ignore[union-attr]
        lambda *a, **k: scored
    )
    hits = await store.search(principal=p, query="anything", k=2)
    assert [h["page"] for h in hits] == [1, 2]
    await store.engine.dispose()
    if store._vec_conn is not None:
        store._vec_conn.close()


@pytest.mark.asyncio
async def test_substring_search_k_limit() -> None:
    """``_substring_search`` stops once ``k`` matches are collected."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    await _create_schema(store)
    pdf_b64 = base64.b64encode(_build_pdf("repeated keyword", pages=3)).decode()
    p = _principal()
    await store.ingest_async(principal=p, name="k.pdf", data_base64=pdf_b64)
    await store.await_pending()
    hits = await store.search(principal=p, query="keyword", k=1)
    assert len(hits) == 1
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_search_falls_back_when_ann_raises(tmp_path: Path) -> None:
    """When the ANN call raises, ``search`` falls back to substring."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'fb.db'}"
    store = PdfStore(url, embeddings=_StubEmbeddings())
    await _create_schema(store)
    pdf_b64 = base64.b64encode(_build_pdf("Fallback keyword here", pages=1)).decode()
    p = _principal()
    await store.ingest_async(principal=p, name="f.pdf", data_base64=pdf_b64)
    await store.await_pending()

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("ANN exploded")

    store._vec.similarity_search_with_score = _boom  # type: ignore[union-attr]
    hits = await store.search(principal=p, query="keyword", k=3)
    assert hits and hits[0]["name"] == "f.pdf"
    await store.engine.dispose()
    if store._vec_conn is not None:
        store._vec_conn.close()


@pytest.mark.asyncio
async def test_vector_index_error_is_swallowed(tmp_path: Path) -> None:
    """A failing ``_index_pages_sync`` does not break ingestion."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'ie.db'}"
    store = PdfStore(url, embeddings=_StubEmbeddings())
    await _create_schema(store)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("index failure")

    store._index_pages_sync = _boom  # type: ignore[method-assign]
    pdf_b64 = base64.b64encode(_build_pdf("Indexed text", pages=1)).decode()
    p = _principal()
    await store.ingest_async(principal=p, name="i.pdf", data_base64=pdf_b64)
    await store.await_pending()
    st = await store.status(principal=p, name="i.pdf", data_base64=pdf_b64)
    assert st is not None and st["status"] == "ready"
    await store.engine.dispose()
    if store._vec_conn is not None:
        store._vec_conn.close()


def test_index_pages_sync_returns_when_no_vec() -> None:
    """``_index_pages_sync`` no-ops when no vector store is configured."""
    store = PdfStore("sqlite+aiosqlite:///:memory:")
    store._index_pages_sync(doc_id=1, user_id="u", name="x.pdf", pages=[])


def test_parse_pdf_sync_from_base64() -> None:
    pdf_b64 = base64.b64encode(_build_pdf("Direct parse", pages=2)).decode()
    parsed = _parse_pdf_sync(url=None, data_base64=pdf_b64)
    assert parsed["total_pages"] == 2
    assert len(parsed["pages"]) == 2
    assert parsed["metadata"]["total_pages"] == 2
    assert isinstance(parsed["toc"], list)


def test_parse_pdf_sync_extracts_outline_and_metadata() -> None:
    """A PDF with bookmarks + metadata exercises the TOC / metadata paths."""
    pdf_b64 = base64.b64encode(_build_pdf("Outlined", pages=3, outline=True)).decode()
    parsed = _parse_pdf_sync(url=None, data_base64=pdf_b64)
    assert parsed["metadata"]["title"] == "Quarterly Report"
    assert parsed["metadata"]["author"] == "Jane Analyst"
    assert len(parsed["toc"]) == 3
    titles = {e["title"] for e in parsed["toc"]}
    assert "Section 1" in titles
    # Bookmark destinations resolve to real page numbers.
    assert all(e["page"] for e in parsed["toc"])


def test_parse_pdf_sync_from_data_url() -> None:
    pdf_b64 = base64.b64encode(_build_pdf("Data URL parse")).decode()
    parsed = _parse_pdf_sync(
        url=f"data:application/pdf;base64,{pdf_b64}", data_base64=None
    )
    assert parsed["total_pages"] == 1


def test_parse_pdf_sync_from_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_bytes = _build_pdf("HTTP parse")

    class _Resp:
        content = pdf_bytes

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def get(self, url: str, headers: dict | None = None) -> _Resp:
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)
    parsed = _parse_pdf_sync(url="https://x.test/doc.pdf", data_base64=None)
    assert parsed["total_pages"] == 1


def test_parse_pdf_sync_raises_without_source() -> None:
    with pytest.raises(RuntimeError, match="no url or data_base64"):
        _parse_pdf_sync(url=None, data_base64=None)


class _StubPage:
    """A pdfplumber-like page; has_obj=False drops page_obj."""

    def __init__(self, idx: int, *, has_obj: bool = True) -> None:
        self._idx = idx
        if has_obj:
            self.page_obj = type("_O", (), {"pageid": idx})()

    def extract_text(self) -> str:
        return f"text {self._idx}"

    def extract_words(self) -> list[dict[str, Any]]:
        return [{"text": "w", "x0": 0, "top": 0, "x1": 1, "bottom": 1}]


class _StubDoc:
    """A pdf.doc with a valid, a malformed, and a resolve-raising outline entry."""

    @staticmethod
    def get_outlines() -> list[Any]:
        return [
            (1, "Good", ["dest-ok"], None, None),
            ("malformed", "too", "few"),
            (1, "BadDest", "raise-me", None, None),
        ]


class _StubPdf:
    """A pdfplumber-like context manager."""

    def __init__(self, *, meta_raises: bool) -> None:
        self.pages = [_StubPage(1), _StubPage(2, has_obj=False)]
        self.doc = _StubDoc()
        self._meta_raises = meta_raises

    @property
    def metadata(self) -> dict[str, Any]:
        if self._meta_raises:
            raise RuntimeError("metadata access blew up")
        return {"Title": b"Bytes Title", "Author": "Plain"}

    def __enter__(self) -> _StubPdf:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _patch_plumber(monkeypatch: pytest.MonkeyPatch, *, meta_raises: bool) -> None:
    """Install a stub pdfplumber and resolve1 for _parse_pdf_sync."""
    import sys

    from pdfminer import pdftypes

    class _Plumber:
        @staticmethod
        def open(_stream: Any) -> _StubPdf:
            return _StubPdf(meta_raises=meta_raises)

    def _fake_resolve1(dest: Any) -> Any:
        if dest == "raise-me":
            raise RuntimeError("resolve failed")
        return [type("_R", (), {"objid": 1})()]

    monkeypatch.setitem(sys.modules, "pdfplumber", _Plumber())
    monkeypatch.setattr(pdftypes, "resolve1", _fake_resolve1)


def test_parse_pdf_sync_metadata_access_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising pdf.metadata accessor leaves raw_meta empty."""
    _patch_plumber(monkeypatch, meta_raises=True)
    parsed = _parse_pdf_sync(
        url=None, data_base64=base64.b64encode(b"%PDF-stub").decode()
    )
    assert parsed["metadata"]["title"] is None
    assert parsed["total_pages"] == 2
    assert [e["title"] for e in parsed["toc"]] == ["Good", "BadDest"]
    assert parsed["toc"][0]["page"] == 1
    assert parsed["toc"][1]["page"] is None


def test_parse_pdf_sync_decodes_bytes_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bytes-valued metadata is UTF-8 decoded by the inner helper."""
    _patch_plumber(monkeypatch, meta_raises=False)
    parsed = _parse_pdf_sync(
        url=None, data_base64=base64.b64encode(b"%PDF-stub").decode()
    )
    assert parsed["metadata"]["title"] == "Bytes Title"
    assert parsed["metadata"]["author"] == "Plain"
