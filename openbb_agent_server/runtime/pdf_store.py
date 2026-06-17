"""PDF ingestion store with background parsing + vector index."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import sqlite3
import threading
from typing import Any

import sqlite_vec
from langchain_community.vectorstores import SQLiteVec
from langchain_core.embeddings import Embeddings
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from openbb_agent_server.observability.logging import trace
from openbb_agent_server.persistence import models as m
from openbb_agent_server.runtime.principal import UserPrincipal

logger = logging.getLogger("openbb_agent_server.runtime.pdf_store")


_VEC_TABLE = "pdf_pages_vec"


def _url_to_file(url: str) -> str | None:
    """Return the file path for a SQLite URL, or ``None`` for in-memory."""
    import re

    match = re.match(r"^sqlite(?:\+\w+)?:///(?P<path>.*)$", url)
    if not match:
        return url
    path = match.group("path")
    if not path or path == ":memory:":
        return None
    return path


def _apply_sqlite_pragmas(engine: AsyncEngine, url: str) -> None:
    if "sqlite" not in url:
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_conn: Any, _: Any) -> None:
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.execute("PRAGMA synchronous=NORMAL")
        finally:
            cur.close()


def _build_vec_connection(db_file: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _file_key(name: str, url: str | None, data_base64: str | None) -> str:
    """Return a stable per-PDF content identity key."""
    hasher = hashlib.sha256()
    if url:
        hasher.update(b"url\x00")
        hasher.update(url.encode("utf-8"))
    if data_base64:
        hasher.update(b"b64\x00")
        hasher.update(data_base64[:16384].encode("utf-8"))
    if not url and not data_base64:
        hasher.update(b"name\x00")
        hasher.update((name or "").encode("utf-8"))
    return hasher.hexdigest()


class PdfStore:
    """SQL-backed PDF store with background parse + ANN page search."""

    def __init__(
        self,
        url: str,
        *,
        embeddings: Embeddings | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize the store and, if possible, the vector index.

        When ``embeddings`` are supplied and the database resolves to a
        real file (not in-memory), a ``sqlite-vec`` connection and an ANN
        page index are created eagerly. Otherwise search falls back to
        substring matching.

        Parameters
        ----------
        url : str
            SQLAlchemy async database URL. SQLite URLs receive WAL and
            busy-timeout pragmas unless an external ``engine`` is passed.
        embeddings : Embeddings or None, optional
            Embedding model used to build the page vector index. If
            ``None``, semantic search degrades to substring search.
        engine : AsyncEngine or None, optional
            Pre-built async engine to reuse. When provided, the store does
            not create its own engine or apply SQLite pragmas.
        """
        self._engine = engine or create_async_engine(url, future=True)
        if engine is None:
            _apply_sqlite_pragmas(self._engine, url)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        self._embeddings = embeddings
        self._vec: SQLiteVec | None = None
        self._vec_conn: sqlite3.Connection | None = None
        self._vec_lock = threading.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        # Per-process registry of file_keys whose ingestion is already
        # running / done — saves redundant background work when the
        # same PDF appears in many requests.
        self._inflight: dict[str, asyncio.Task[None]] = {}
        db_file = _url_to_file(url)
        if embeddings is not None and db_file is not None:
            self._vec_conn = _build_vec_connection(db_file)
            self._vec = SQLiteVec(
                table=_VEC_TABLE,
                connection=self._vec_conn,
                embedding=embeddings,
                db_file=db_file,
            )
            self._vec.create_table_if_not_exists()

    @property
    def engine(self) -> AsyncEngine:
        """Return the underlying SQLAlchemy async engine.

        Returns
        -------
        AsyncEngine
            The engine backing this store, whether created internally or
            injected at construction.
        """
        return self._engine

    async def status(
        self,
        *,
        principal: UserPrincipal,
        name: str,
        url: str | None = None,
        data_base64: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the stored status record for a PDF, or ``None`` if unknown.

        The PDF is identified by a content key derived from ``url``,
        ``data_base64``, or ``name`` and scoped to the principal's user.

        Parameters
        ----------
        principal : UserPrincipal
            The user whose library is queried. Records are user-scoped.
        name : str
            Display name of the PDF; also a fallback identity component.
        url : str or None, optional
            Source URL of the PDF, if it was ingested from a URL.
        data_base64 : str or None, optional
            Base64-encoded PDF bytes, if it was ingested inline.

        Returns
        -------
        dict of str to Any or None
            A record with ``id``, ``name``, ``url``, ``status``, ``error``,
            ``total_pages``, ``metadata``, and ``toc`` keys, or ``None``
            when no matching document exists for this user.
        """
        key = _file_key(name, url, data_base64)
        async with self._sessionmaker() as session:
            doc = (
                (
                    await session.execute(
                        select(m.PdfDocument).where(
                            m.PdfDocument.user_id == principal.user_id,
                            m.PdfDocument.file_key == key,
                        )
                    )
                )
                .scalars()
                .first()
            )
        if doc is None:
            return None
        return {
            "id": doc.id,
            "name": doc.name,
            "url": doc.url,
            "status": doc.status,
            "error": doc.error,
            "total_pages": doc.total_pages,
            "metadata": dict(doc.metadata_json or {}),
            "toc": list(doc.toc_json or []),
        }

    async def get_pages(
        self,
        *,
        principal: UserPrincipal,
        name: str,
        url: str | None = None,
        data_base64: str | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return parsed pages for a PDF, optionally filtered by page range.

        Returns ``None`` unless the document exists and has finished
        parsing (``status == "ready"``).

        Parameters
        ----------
        principal : UserPrincipal
            The user whose library is queried.
        name : str
            Display name of the PDF; also a fallback identity component.
        url : str or None, optional
            Source URL of the PDF, if ingested from a URL.
        data_base64 : str or None, optional
            Base64-encoded PDF bytes, if ingested inline.
        page_range : tuple of (int, int) or None, optional
            Inclusive ``(lo, hi)`` 1-based page bounds. ``None`` returns
            all pages.

        Returns
        -------
        list of dict of str to Any or None
            Page records each with ``page``, ``text``, and ``words`` keys,
            ordered by page number, or ``None`` if the PDF is not ready.
        """
        st = await self.status(
            principal=principal, name=name, url=url, data_base64=data_base64
        )
        if st is None or st["status"] != "ready":
            return None
        pdf_id = int(st["id"])
        async with self._sessionmaker() as session:
            stmt = (
                select(m.PdfPage)
                .where(m.PdfPage.pdf_id == pdf_id)
                .order_by(m.PdfPage.page.asc())
            )
            if page_range is not None:
                lo, hi = int(page_range[0]), int(page_range[1])
                stmt = stmt.where(m.PdfPage.page >= lo, m.PdfPage.page <= hi)
            rows = (await session.execute(stmt)).scalars().all()
        return [
            {"page": r.page, "text": r.text, "words": list(r.words_json or [])}
            for r in rows
        ]

    async def ingest_async(
        self,
        *,
        principal: UserPrincipal,
        name: str,
        url: str | None = None,
        data_base64: str | None = None,
        mime: str | None = None,
    ) -> str:
        """Schedule background parsing and indexing of a PDF.

        Idempotent and deduplicating: a PDF already ``ready`` or with an
        in-flight ingest task is not re-scheduled. The background task
        fetches/decodes the PDF, persists pages, and (when embeddings are
        configured) builds the page vector index.

        Parameters
        ----------
        principal : UserPrincipal
            The user the document is ingested for; rows are user-scoped.
        name : str
            Display name of the PDF; also a fallback identity component.
        url : str or None, optional
            Source URL or ``data:`` URI to fetch/decode the PDF from.
        data_base64 : str or None, optional
            Base64-encoded PDF bytes provided inline.
        mime : str or None, optional
            MIME type recorded on the document, if known.

        Returns
        -------
        str
            The stable content ``file_key`` identifying this PDF, which
            callers can pass to :meth:`~openbb_agent_server.runtime.jobs.JobRegistry.status` to poll progress.
        """
        key = _file_key(name, url, data_base64)
        existing = await self.status(
            principal=principal, name=name, url=url, data_base64=data_base64
        )
        if existing is not None and existing["status"] == "ready":
            logger.debug(
                "pdf_store: ingest skip — %r already ready (file_key=%s)",
                name,
                key[:12],
            )
            return key
        if key in self._inflight and not self._inflight[key].done():
            logger.debug(
                "pdf_store: ingest skip — %r already in flight (file_key=%s)",
                name,
                key[:12],
            )
            return key
        trace(
            logger,
            "pdf_store: scheduling ingest for %r (file_key=%s, prior_status=%s)",
            name,
            key[:12],
            (existing or {}).get("status") if existing else "absent",
        )
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._ingest_in_background(
                principal=principal,
                file_key=key,
                name=name,
                url=url,
                data_base64=data_base64,
                mime=mime,
            )
        )
        self._inflight[key] = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return key

    async def await_pending(self) -> None:
        """Block until every in-flight ingest finishes. Test-only."""
        if not self._tasks:
            return
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _ingest_in_background(
        self,
        *,
        principal: UserPrincipal,
        file_key: str,
        name: str,
        url: str | None,
        data_base64: str | None,
        mime: str | None,
    ) -> None:
        async with self._sessionmaker() as session:
            existing = (
                (
                    await session.execute(
                        select(m.PdfDocument).where(
                            m.PdfDocument.user_id == principal.user_id,
                            m.PdfDocument.file_key == file_key,
                        )
                    )
                )
                .scalars()
                .first()
            )
            if existing is None:
                doc = m.PdfDocument(
                    user_id=principal.user_id,
                    file_key=file_key,
                    name=name,
                    url=url,
                    mime=mime,
                    status="pending",
                )
                session.add(doc)
                await session.commit()
                doc_id = int(doc.id)
            elif existing.status == "ready":
                return
            else:
                existing.status = "pending"
                existing.error = None
                await session.commit()
                doc_id = int(existing.id)

        try:
            parsed = await asyncio.to_thread(
                _parse_pdf_sync, url=url, data_base64=data_base64
            )
        except Exception as exc:
            logger.warning("pdf_store: parse failed for %r (%s): %s", name, url, exc)
            async with self._sessionmaker() as session:
                doc = await session.get(m.PdfDocument, doc_id)
                if doc is not None:
                    doc.status = "error"
                    doc.error = f"{type(exc).__name__}: {exc}"
                    await session.commit()
            return

        async with self._sessionmaker() as session:
            doc = await session.get(m.PdfDocument, doc_id)
            if doc is None:  # pragma: no cover — defensive
                return
            doc.total_pages = parsed["total_pages"]
            doc.metadata_json = parsed["metadata"]
            doc.toc_json = parsed["toc"]
            doc.status = "ready"
            for p in parsed["pages"]:
                session.add(
                    m.PdfPage(
                        pdf_id=doc_id,
                        page=int(p["page"]),
                        text=str(p.get("text") or ""),
                        words_json=list(p.get("words") or []),
                    )
                )
            await session.commit()

        if self._vec is not None and parsed["pages"]:
            try:
                await asyncio.to_thread(
                    self._index_pages_sync,
                    doc_id=doc_id,
                    user_id=principal.user_id,
                    name=name,
                    pages=parsed["pages"],
                )
            except Exception:
                logger.warning(
                    "pdf_store: vector index failed for %r — semantic "
                    "search falls back to substring",
                    name,
                    exc_info=True,
                )

    def _index_pages_sync(
        self,
        *,
        doc_id: int,
        user_id: str,
        name: str,
        pages: list[dict[str, Any]],
    ) -> None:
        if self._vec is None:
            return
        batch_size = 32
        for start in range(0, len(pages), batch_size):
            chunk = pages[start : start + batch_size]
            texts = [str(p.get("text") or "") for p in chunk]
            metas = [
                {
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "name": name,
                    "page": int(p.get("page") or 0),
                }
                for p in chunk
            ]
            with self._vec_lock:
                self._vec.add_texts(texts, metadatas=metas)

    async def search(
        self,
        *,
        principal: UserPrincipal,
        query: str,
        k: int = 8,
    ) -> list[dict[str, Any]]:
        """Semantic-search PDF pages across this user's library.

        Uses the ANN vector index when available, deduplicating hits by
        ``(doc_id, page)`` and filtering to the principal's documents. When
        no index is configured, or an ANN query fails, falls back to a
        case-insensitive substring search.

        Parameters
        ----------
        principal : UserPrincipal
            The user whose pages are searched; results are user-scoped.
        query : str
            Free-text query. Blank/whitespace queries yield no results.
        k : int, default 8
            Maximum number of page hits to return.

        Returns
        -------
        list of dict of str to Any
            Up to ``k`` hits, each with ``score`` (higher is closer),
            ``name``, ``page``, and ``text`` keys. Empty if nothing matches.
        """
        if not query.strip():
            return []
        if self._vec is None:
            return await self._substring_search(principal=principal, query=query, k=k)
        vec = self._vec

        def _run() -> list[tuple[Any, float]]:
            with self._vec_lock:
                return vec.similarity_search_with_score(query, k=k * 4)

        try:
            scored = await asyncio.to_thread(_run)
        except Exception:
            logger.warning(
                "pdf_store: ANN failed; falling back to substring",
                exc_info=True,
            )
            return await self._substring_search(principal=principal, query=query, k=k)
        out: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for doc, distance in scored:
            meta = dict(doc.metadata or {})
            if meta.get("user_id") != principal.user_id:
                continue
            doc_id = int(meta.get("doc_id") or 0)
            page = int(meta.get("page") or 0)
            key = (doc_id, page)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "score": 1.0 / (1.0 + max(0.0, float(distance))),
                    "name": meta.get("name"),
                    "page": page,
                    "text": doc.page_content,
                }
            )
            if len(out) >= k:
                break
        return out

    async def _substring_search(
        self,
        *,
        principal: UserPrincipal,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        async with self._sessionmaker() as session:
            docs = (
                (
                    await session.execute(
                        select(m.PdfDocument).where(
                            m.PdfDocument.user_id == principal.user_id,
                            m.PdfDocument.status == "ready",
                        )
                    )
                )
                .scalars()
                .all()
            )
            doc_by_id = {d.id: d for d in docs}
            if not doc_by_id:
                return []
            pages_stmt = (
                select(m.PdfPage)
                .where(m.PdfPage.pdf_id.in_(doc_by_id.keys()))
                .order_by(m.PdfPage.pdf_id.asc(), m.PdfPage.page.asc())
            )
            pages = (await session.execute(pages_stmt)).scalars().all()
        q = query.lower().strip()
        out: list[dict[str, Any]] = []
        for p in pages:
            if q and q in (p.text or "").lower():
                doc = doc_by_id.get(p.pdf_id)
                if (
                    doc is None
                ):  # pragma: no cover — pages are queried with pdf_id IN doc_by_id, always present
                    continue
                out.append(
                    {
                        "score": 1.0,
                        "name": doc.name,
                        "page": p.page,
                        "text": p.text,
                    }
                )
                if len(out) >= k:
                    break
        return out


def _parse_pdf_sync(
    *,
    url: str | None,
    data_base64: str | None,
) -> dict[str, Any]:
    """Fetch + parse a PDF synchronously. Run inside ``asyncio.to_thread``."""
    import httpx
    import pdfplumber
    from pdfminer.pdftypes import resolve1

    if data_base64:
        pdf_bytes = base64.b64decode(data_base64)
    elif url:
        head = url[:64].lower()
        if head.startswith(("data:application/pdf", "data:application/octet-stream")):
            _, _, payload = url.partition("base64,")
            pdf_bytes = base64.b64decode(payload.strip())
        else:
            headers = {
                "User-Agent": "OpenBB-Agent-Server/1.0 (+pdf_store)",
                "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
            }
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                pdf_bytes = resp.content
    else:
        raise RuntimeError("PDF has no url or data_base64")

    pages_out: list[dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        raw_meta = {}
        try:
            raw_meta = dict(pdf.metadata or {})
        except Exception:
            raw_meta = {}

        def _s(v: Any) -> str:
            if isinstance(v, bytes):
                try:
                    return v.decode("utf-8", errors="replace")
                except (
                    Exception
                ):  # pragma: no cover — decode(errors="replace") cannot raise
                    return v.decode("latin-1", errors="replace")
            return "" if v is None else str(v)

        metadata = {
            "title": _s(raw_meta.get("Title")).strip() or None,
            "author": _s(raw_meta.get("Author")).strip() or None,
            "subject": _s(raw_meta.get("Subject")).strip() or None,
            "creator": _s(raw_meta.get("Creator")).strip() or None,
            "producer": _s(raw_meta.get("Producer")).strip() or None,
            "creation_date": _s(raw_meta.get("CreationDate")).strip() or None,
            "mod_date": _s(raw_meta.get("ModDate")).strip() or None,
            "total_pages": total_pages,
        }
        page_id_to_num: dict[int, int] = {}
        for idx, page in enumerate(pdf.pages, start=1):
            try:
                obj = page.page_obj
            except AttributeError:
                continue
            pid = getattr(obj, "pageid", None) or getattr(obj, "objid", None)
            if isinstance(pid, int):
                page_id_to_num[pid] = idx
        toc: list[dict[str, Any]] = []
        try:
            for entry in pdf.doc.get_outlines():
                try:
                    level, title, dest, _action, _se = entry
                except (TypeError, ValueError):
                    continue
                page_num: int | None = None
                try:
                    resolved = resolve1(dest) if dest is not None else None
                except Exception:
                    resolved = dest
                if isinstance(resolved, list) and resolved:
                    first = resolved[0]
                    objid = getattr(first, "objid", None)
                    if isinstance(objid, int):
                        page_num = page_id_to_num.get(objid)
                toc.append(
                    {
                        "level": int(level) if isinstance(level, int) else 0,
                        "title": _s(title).strip(),
                        "page": page_num,
                    }
                )
        except Exception:
            toc = []

        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            words = page.extract_words() or []
            pages_out.append(
                {
                    "page": idx,
                    "text": text,
                    "words": [
                        {
                            "text": w.get("text", ""),
                            "x0": float(w.get("x0", 0.0)),
                            "top": float(w.get("top", 0.0)),
                            "x1": float(w.get("x1", 0.0)),
                            "bottom": float(w.get("bottom", 0.0)),
                        }
                        for w in words
                    ],
                }
            )

    return {
        "total_pages": total_pages,
        "metadata": metadata,
        "toc": toc,
        "pages": pages_out,
    }
