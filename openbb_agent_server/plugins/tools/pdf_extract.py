"""``pdf_extract`` tool source."""

from __future__ import annotations

import asyncio
import base64
import io
import json as _json
import logging
from typing import Annotated, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, BeforeValidator, Field

from openbb_agent_server.observability.logging import trace
from openbb_agent_server.runtime import (
    context as run_context,
    emit,
    services,
)
from openbb_agent_server.runtime.context import FileRef, RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.pdf_extract")


_PREVIEW_PAGES = 3


def _coerce_page_range(value: Any) -> Any:
    """Coerce loose ``page_range`` shapes into an ``(int, int)`` tuple."""
    if value is None or isinstance(value, (tuple, list)):
        return value
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.startswith("[") or s.startswith("("):
            try:
                parsed = _json.loads(s.replace("(", "[").replace(")", "]"))
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return parsed
        for sep in ("-", "..", ","):
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                if len(parts) == 2:
                    try:
                        return (int(parts[0]), int(parts[1]))
                    except ValueError:
                        continue
        try:
            n = int(s)
            return (n, n)
        except ValueError:
            return value
    return value


_LoosePageRange = Annotated[
    tuple[int, int] | None,
    BeforeValidator(_coerce_page_range),
]


class _ExtractArgs(BaseModel):
    name: str = Field(
        description="Filename of the uploaded PDF (e.g. '10K-2024.pdf').",
    )
    page_range: _LoosePageRange = Field(
        default=None,
        description=(
            "1-based inclusive page range — PASS A LIST OF TWO "
            "INTEGERS like ``[137, 140]`` (or omit entirely for a "
            "preview probe). Accepted shapes: ``[a, b]``, ``(a, b)``, "
            '``"a-b"``, ``"a,b"``. Don\'t omit page_range when '
            "targeting a specific section."
        ),
    )
    include_words: bool = Field(
        default=False,
        description=(
            "Include per-word bounding boxes in the response. Only "
            "set True when you actually need bboxes for citations — "
            "they roughly 10× the response size."
        ),
    )


class _ListArgs(BaseModel):
    pass


class _SearchArgs(BaseModel):
    query: str = Field(
        description="Natural-language query to match PDF page text against.",
    )
    k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of page hits to return.",
    )


class _OutlineArgs(BaseModel):
    name: str = Field(
        description="Filename of the uploaded PDF (fuzzy-matched).",
    )


_PDF_FETCH_HEADERS = {
    "User-Agent": "OpenBB-Agent-Server/1.0 (+pdf_extract)",
    "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
}


def _coerce_str(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover — decode(errors="replace") cannot raise
            return value.decode("latin-1", errors="replace")
    if value is None:
        return ""
    return str(value)


def _build_page_map(pdf: Any) -> dict[int, int]:
    """Map pdfminer page object id to a 1-based page number."""
    out: dict[int, int] = {}
    for idx, page in enumerate(pdf.pages, start=1):
        try:
            page_obj = page.page_obj
        except AttributeError:
            continue
        objid = getattr(page_obj, "pageid", None) or getattr(page_obj, "objid", None)
        if isinstance(objid, int):
            out[objid] = idx
    return out


def _resolve_outline_page(dest: Any, page_map: dict[int, int]) -> int | None:
    """Turn a pdfminer outline destination into a page number."""
    try:
        from pdfminer.pdftypes import resolve1
    except ImportError:  # pragma: no cover — pdfminer is bundled with pdfplumber
        return None
    try:
        resolved = resolve1(dest) if dest is not None else None
    except Exception:
        resolved = dest
    if isinstance(resolved, list) and resolved:
        first = resolved[0]
        objid = getattr(first, "objid", None)
        if isinstance(objid, int):
            return page_map.get(objid)
    objid = getattr(resolved, "objid", None)
    if isinstance(objid, int):
        return page_map.get(objid)
    return None


def _extract_toc(pdf: Any) -> list[dict[str, Any]]:
    """Walk the PDF outline into a flat ``[{level, title, page}]`` list."""
    try:
        outlines_iter = list(pdf.doc.get_outlines())
    except Exception:
        return []
    page_map = _build_page_map(pdf)
    toc: list[dict[str, Any]] = []
    for entry in outlines_iter:
        try:
            level, title, dest, _action, _se = entry
        except (TypeError, ValueError):
            continue
        page_num = _resolve_outline_page(dest, page_map)
        toc.append(
            {
                "level": int(level) if isinstance(level, int) else 0,
                "title": _coerce_str(title).strip(),
                "page": page_num,
            }
        )
    return toc


def _extract_pdf_metadata(pdf: Any) -> dict[str, Any]:
    """Pull title, author, dates, and page count out of a PDF."""
    raw = {}
    try:
        raw = dict(pdf.metadata or {})
    except Exception:
        raw = {}
    return {
        "title": _coerce_str(raw.get("Title")).strip() or None,
        "author": _coerce_str(raw.get("Author")).strip() or None,
        "subject": _coerce_str(raw.get("Subject")).strip() or None,
        "creator": _coerce_str(raw.get("Creator")).strip() or None,
        "producer": _coerce_str(raw.get("Producer")).strip() or None,
        "creation_date": _coerce_str(raw.get("CreationDate")).strip() or None,
        "mod_date": _coerce_str(raw.get("ModDate")).strip() or None,
        "total_pages": len(pdf.pages),
    }


_PDF_B64_MAGIC = "JVBERi0"
_PDF_RAW_MAGIC = "%PDF-"


def _string_to_pdf_b64(s: str) -> str | None:
    """Detect a PDF byte payload in ``s``."""
    if not isinstance(s, str) or not s:
        return None
    stripped = s.lstrip()
    if stripped.startswith(_PDF_B64_MAGIC):
        return stripped
    if stripped[:64].lower().startswith("data:application/pdf"):
        _, _, payload = stripped.partition("base64,")
        payload = payload.strip()
        if payload.startswith(_PDF_B64_MAGIC):
            return payload
    if stripped.startswith(_PDF_RAW_MAGIC):
        try:
            return base64.b64encode(
                stripped.encode("latin-1", errors="replace")
            ).decode("ascii")
        except Exception:  # pragma: no cover — b64encode of bytes cannot raise
            return None
    return None


def _find_any_http_url(blob: Any, *, _depth: int = 0) -> str | None:
    """Recursively walk a value for a fetchable URL."""
    if _depth > 6:
        return None
    if isinstance(blob, str):
        s = blob.strip()
        lower = s.lower()
        if lower.startswith(("http://", "https://")):
            return s
        if lower.startswith(("data:application/pdf", "data:application/octet-stream")):
            return s
        return None
    if isinstance(blob, dict):
        for v in blob.values():
            found = _find_any_http_url(v, _depth=_depth + 1)
            if found:
                return found
    elif isinstance(blob, (list, tuple)):
        for v in blob:
            found = _find_any_http_url(v, _depth=_depth + 1)
            if found:
                return found
    return None


def _find_any_pdf_b64(blob: Any, *, _depth: int = 0) -> str | None:
    """Recursively walk a value for a PDF byte payload."""
    if _depth > 6:
        return None
    if isinstance(blob, str):
        return _string_to_pdf_b64(blob)
    if isinstance(blob, dict):
        for v in blob.values():
            found = _find_any_pdf_b64(v, _depth=_depth + 1)
            if found:
                return found
    elif isinstance(blob, (list, tuple)):
        for v in blob:
            found = _find_any_pdf_b64(v, _depth=_depth + 1)
            if found:
                return found
    return None


def _bytes_from_data_url(url: str) -> bytes | None:
    """Decode a ``data:`` PDF URL into raw bytes."""
    if not isinstance(url, str):
        return None
    head = url[:64].lower()
    if not head.startswith(("data:application/pdf", "data:application/octet-stream")):
        return None
    _, _, payload = url.partition("base64,")
    if not payload:
        return None
    try:
        return base64.b64decode(payload.strip())
    except Exception:
        return None


def _resolve_pdf_bytes(fileref: FileRef, http_client: Any) -> bytes:
    if fileref.data_base64:
        return base64.b64decode(fileref.data_base64)
    if fileref.url:
        data_bytes = _bytes_from_data_url(fileref.url)
        if data_bytes is not None:
            return data_bytes
        resp = http_client.get(fileref.url, headers=_PDF_FETCH_HEADERS)
        resp.raise_for_status()
        return resp.content
    extras = getattr(fileref, "model_extra", None) or {}
    b64 = _find_any_pdf_b64(extras)
    if b64:
        trace(
            logger,
            "pdf_extract: recovered base64 PDF for %r from FileRef.extras (keys=%r)",
            fileref.name,
            sorted(extras.keys()),
        )
        return base64.b64decode(b64)
    url = _find_any_http_url(extras)
    if url:
        trace(
            logger,
            "pdf_extract: recovered url for %r from FileRef.extras (keys=%r): %s",
            fileref.name,
            sorted(extras.keys()),
            url[:120],
        )
        data_bytes = _bytes_from_data_url(url)
        if data_bytes is not None:
            return data_bytes
        resp = http_client.get(url, headers=_PDF_FETCH_HEADERS)
        resp.raise_for_status()
        return resp.content
    raise RuntimeError(
        f"file {fileref.name!r} has no url or data_base64, and a "
        f"recursive scan of its extra fields found neither an HTTP URL "
        f"nor an inline base64 PDF payload. Extra keys present: "
        f"{sorted(extras.keys())}. Workspace's wire protocol shipped "
        f"this file without a way for the agent server to fetch its "
        f"bytes."
    )


def _is_pdf(f: FileRef) -> bool:
    if f.mime and "pdf" in f.mime.lower():
        return True
    return f.name.lower().endswith(".pdf")


def _normalise_pdf_name(name: str) -> str:
    """Lowercase, strip, and drop the .pdf suffix for fuzzy matching."""
    n = name.strip().lower()
    if n.endswith(".pdf"):
        n = n[:-4]
    return n


def _match_uploaded_pdf(uploads: tuple[FileRef, ...], requested: str) -> FileRef | None:
    """Resolve the agent's ``name`` argument to one of the uploaded PDFs."""
    pdfs = [f for f in uploads if _is_pdf(f)]
    if not pdfs:
        return None
    for f in pdfs:
        if f.name == requested:
            return f
    target = _normalise_pdf_name(requested)
    if not target:
        return None
    for f in pdfs:
        if _normalise_pdf_name(f.name) == target:
            return f
    for f in pdfs:
        if target in _normalise_pdf_name(f.name):
            return f
    for f in pdfs:
        if _normalise_pdf_name(f.name) in target:
            return f
    return None


class PdfExtractToolSource(ToolSource):
    """Expose PDF listing, outline, search, and extraction tools.

    A :class:`~openbb_agent_server.runtime.plugins.ToolSource` plugin (registered name ``"pdf_extract"``)
    whose tools read PDFs the user uploaded (or that were promoted from
    dashboard document widgets). Extracted pages emit Workspace citations
    that link back to the originating pinned widget where one can be
    resolved.
    """

    name = "pdf_extract"

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Build the PDF tools for the current run.

        Prepares an HTTP client for fetching PDF bytes, indexes the
        run's pinned widgets so extracted pages can be cited back to
        their source widget, and defines the ``list_pdfs``,
        ``get_pdf_outline``, ``search_pdf``, and ``pdf_extract`` tools.
        Where a PDF store is configured, ``get_pdf_outline`` and
        ``pdf_extract`` await background ingestion before reading; with
        no store they fall back to inline ``pdfplumber`` parsing.

        Parameters
        ----------
        ctx : RunContext
            Active run context; ``ctx.widgets`` seeds citation matching.
        config : dict
            Per-run configuration (currently unused by these tools).

        Returns
        -------
        list of Any
            The structured PDF tools registered for the run.

        Raises
        ------
        RuntimeError
            If ``pdfplumber`` or ``httpx`` are not installed (install the
            ``[pdf]`` extra).
        """
        try:
            import httpx
            import pdfplumber
        except ImportError as exc:  # pragma: no cover — install-hint path
            raise RuntimeError(
                "pdf_extract requires pdfplumber and httpx. Install the "
                "agent_server with the [pdf] extra."
            ) from exc

        http_client = httpx.Client(timeout=30.0)

        pdf_cited: set[tuple[str, int]] = set()

        dashboard_by_uuid: dict[str, Any] = {}
        dashboard_by_widget_id: dict[str, Any] = {}
        for w in ctx.widgets or []:
            if w.uuid:
                dashboard_by_uuid[w.uuid] = w
            wid = (getattr(w, "widget_id", "") or "").strip()
            if wid:
                dashboard_by_widget_id[wid] = w

        def _name_tokens(s: str) -> set[str]:
            """Tokenise a name into 3+ char lowercase tokens."""
            normalised = (
                (s or "")
                .lower()
                .replace(".pdf", "")
                .replace("_", " ")
                .replace("-", " ")
                .replace("/", " ")
            )
            return {t for t in normalised.split() if len(t) >= 3}

        def _widget_param_tokens(w: Any) -> set[str]:
            """Tokenise every string value in ``w.params`` for matching."""
            params = getattr(w, "params", None) or {}
            value_tokens: set[str] = set()
            if isinstance(params, dict):
                for v in params.values():
                    if isinstance(v, str):
                        value_tokens |= _name_tokens(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                value_tokens |= _name_tokens(item)
            elif isinstance(params, list):
                for p in params:
                    cv = getattr(p, "current_value", None)
                    if isinstance(cv, str):
                        value_tokens |= _name_tokens(cv)
            return value_tokens

        def _content_match(target_file: Any) -> Any:
            """Find the pinned widget whose params match the PDF."""
            pdf_tokens = _name_tokens(target_file.name or "")
            if not pdf_tokens:
                return None
            best: Any = None
            best_score = 0
            for w in ctx.widgets or []:
                value_tokens = _widget_param_tokens(w)
                overlap = pdf_tokens & value_tokens
                if not overlap:
                    continue
                score = len(overlap)
                if score > best_score:
                    best_score = score
                    best = w
            return best

        def _resolve_source_widget(target_file: Any) -> Any:
            """Pick the pinned widget that backs ``target_file``."""
            extras = getattr(target_file, "model_extra", None) or {}
            src_uuid = str(extras.get("source_widget_uuid") or "")
            src_widget_id = str(extras.get("source_widget_id") or "")

            content = _content_match(target_file)
            stamped = dashboard_by_uuid.get(src_uuid) if src_uuid else None

            pick = content or stamped or dashboard_by_widget_id.get(src_widget_id)

            trace(
                logger,
                "pdf_citation: file=%r stamped=%s content=%s pick=%s pinned_widgets=%s",
                target_file.name,
                (stamped.uuid if stamped else None),
                (content.uuid if content else None),
                (pick.uuid if pick else None),
                [
                    {
                        "uuid": w.uuid,
                        "widget_id": getattr(w, "widget_id", ""),
                        "param_tokens": sorted(_widget_param_tokens(w)),
                    }
                    for w in (ctx.widgets or [])
                ],
            )
            return pick

        def _resolve_widget_context(
            target_file: Any,
        ) -> tuple[str | None, str | None, str | None, str | None, dict[str, Any]]:
            """Return widget uuid, id, name, origin, and input args."""
            pinned = _resolve_source_widget(target_file)
            widget_uuid: str | None = None
            widget_id: str | None = None
            widget_name: str | None = None
            widget_origin: str | None = None
            input_args: dict[str, Any] = {}
            if pinned is not None:
                widget_uuid = pinned.uuid or None
                widget_id = (getattr(pinned, "widget_id", "") or "").strip() or None
                widget_name = (getattr(pinned, "name", "") or "").strip() or None
                widget_origin = (getattr(pinned, "origin", "") or "").strip() or None
                try:
                    params = getattr(pinned, "params", None) or {}
                    if isinstance(params, dict):
                        input_args = {k: params[k] for k in list(params)[:16]}
                    elif isinstance(params, list):
                        input_args = {
                            getattr(p, "name", str(idx)): getattr(
                                p, "current_value", None
                            )
                            for idx, p in enumerate(params)
                            if getattr(p, "name", None)
                        }
                except Exception:
                    input_args = {}
            return widget_uuid, widget_id, widget_name, widget_origin, input_args

        def _navigation_bbox(
            words: list[dict[str, Any]], page_num: int
        ) -> dict[str, Any]:
            """Build one small navigation bbox for the citation."""
            label = f"Page {page_num}"
            if words:
                first = words[0]
                return {
                    "text": label,
                    "page": page_num,
                    "x0": float(first.get("x0", 0.0)),
                    "top": float(first.get("top", 0.0)),
                    "x1": float(first.get("x1", 0.0)),
                    "bottom": float(first.get("bottom", 0.0)),
                }
            return {
                "text": label,
                "page": page_num,
                "x0": 0.0,
                "top": 0.0,
                "x1": 1.0,
                "bottom": 1.0,
            }

        def _emit_pdf_citation(
            target_file: Any,
            page_num: int,
            words: list[dict[str, Any]],
            page_text: str = "",
            fallback_text: str = "",
            reference_label: str | None = None,
            quote: str | None = None,
        ) -> None:
            """Emit one PDF citation with one small navigation bbox."""
            del fallback_text
            del quote
            key = (target_file.name, page_num)
            if key in pdf_cited:
                return
            pdf_cited.add(key)
            (
                widget_uuid,
                widget_id,
                widget_name,
                widget_origin,
                input_args,
            ) = _resolve_widget_context(target_file)

            quote_bounding_boxes = [[_navigation_bbox(words, page_num)]]
            del reference_label

            extra_details: dict[str, Any] = {
                "Name": widget_name or "",
                "Filename": target_file.name,
                "Page": page_num,
            }
            emit.cite(
                widget=widget_uuid,
                widget_id=widget_id,
                source=widget_name,
                source_url=widget_origin,
                input_arguments=input_args or None,
                extra_details=extra_details,
                quote_bounding_boxes=quote_bounding_boxes,
            )

        def _cite_pdf_pages(target_file: Any, pages: list[dict[str, Any]]) -> None:
            """Emit one PDF citation per ``pdf_extract`` call."""
            if not pages:
                return
            anchor: dict[str, Any] | None = None
            for p in pages:
                if (p.get("text") or "").strip():
                    anchor = p
                    break
            if anchor is None:
                anchor = pages[0]
            _emit_pdf_citation(
                target_file,
                page_num=int(anchor.get("page") or 0),
                words=list(anchor.get("words") or []),
                page_text=str(anchor.get("text") or ""),
                fallback_text=str(anchor.get("text") or ""),
            )

        def _cite_pdf_search_hit(
            target_file: Any,
            hit_page: int,
            hit_text: str,
            words: list[dict[str, Any]],
            page_text: str = "",
        ) -> None:
            """Emit one PDF citation for a single ``search_pdf`` hit."""
            _emit_pdf_citation(
                target_file,
                page_num=hit_page,
                words=words,
                page_text=page_text or hit_text,
                fallback_text=hit_text,
                quote=hit_text,
            )

        list_called = {"v": False}

        def list_pdfs() -> dict[str, Any]:
            current = run_context.current()
            pdfs = [f for f in current.uploaded_files if _is_pdf(f)]
            emit.reasoning_step("list_pdfs", count=len(pdfs))
            entries = [{"name": f.name, "mime": f.mime, "url": f.url} for f in pdfs]
            if list_called["v"]:
                return {
                    "count": len(entries),
                    "pdfs": entries,
                    "message": (
                        "list_pdfs was already called this turn — the "
                        "result is identical. STOP calling it. If the "
                        "list is empty, the user hasn't attached a "
                        "readable PDF this turn (document widgets only "
                        "expose their PDFs once the user selects them "
                        "or once Workspace serialises a download URL "
                        "into the request). Tell the user what's "
                        "missing instead of retrying."
                    ),
                }
            list_called["v"] = True
            if not entries:
                doc_widgets = [
                    {
                        "widget_id": getattr(w, "widget_id", None),
                        "name": getattr(w, "name", None),
                        "uuid": w.uuid,
                    }
                    for w in (current.widgets or [])
                    if any(
                        tok in (getattr(w, "widget_id", "") or "").lower()
                        for tok in ("document", "filing", "prospectus", "pdf")
                    )
                ]
                return {
                    "count": 0,
                    "pdfs": [],
                    "document_widgets": doc_widgets,
                    "message": (
                        "No PDFs are loaded yet. Document widgets show "
                        "up on the dashboard with their bytes UNLOADED "
                        "until you fetch them. If ``document_widgets`` "
                        "above lists a widget, call ``get_widget_data("
                        "widget_ids=[<that uuid>])`` to dispatch a "
                        "fetch — Workspace returns the PDF content in "
                        "the tool message on the NEXT turn, and it "
                        "auto-promotes into uploaded_files. THEN call "
                        "list_pdfs / pdf_extract. Do NOT ask the user "
                        "to re-attach a file that's already on the "
                        "widget."
                    ),
                }
            return {"count": len(entries), "pdfs": entries}

        def _shape_pages(
            pages: list[dict[str, Any]],
            include_words: bool,
        ) -> list[dict[str, Any]]:
            """Drop word bboxes unless explicitly requested."""
            if include_words:
                return pages
            return [{"page": p["page"], "text": p.get("text", "")} for p in pages]

        def _preview_range(total_pages: int) -> tuple[int, int]:
            return (1, min(_PREVIEW_PAGES, total_pages))

        async def pdf_extract(
            name: str,
            page_range: tuple[int, int] | None = None,
            include_words: bool = False,
        ) -> dict[str, Any]:
            current = run_context.current()
            target = _match_uploaded_pdf(current.uploaded_files, name)
            if target is None:
                available = [f.name for f in current.uploaded_files if _is_pdf(f)]
                return {
                    "error": (
                        f"pdf {name!r} not found in this run's uploads. "
                        f"Available: {available}. Call list_pdfs() to "
                        "see the exact filenames."
                    ),
                    "name": name,
                    "available": available,
                }
            emit.reasoning_step(
                f"pdf_extract({target.name})",
                page_range=list(page_range) if page_range else None,
            )

            async def _from_store_ready(
                st: dict[str, Any],
            ) -> dict[str, Any]:
                effective_range = page_range or _preview_range(st["total_pages"])
                pages = (
                    await pdf_store.get_pages(
                        principal=current.principal,
                        name=target.name,
                        url=target.url,
                        data_base64=target.data_base64,
                        page_range=effective_range,
                    )
                    or []
                )
                _cite_pdf_pages(target, pages)
                return {
                    "name": target.name,
                    "total_pages": st["total_pages"],
                    "page_range": list(effective_range),
                    "is_preview": page_range is None
                    and st["total_pages"] > _PREVIEW_PAGES,
                    "metadata": st["metadata"],
                    "toc": st["toc"],
                    "pages": _shape_pages(pages, include_words),
                }

            pdf_store = services.get_pdf_store()
            if pdf_store is not None:
                st0 = await pdf_store.status(
                    principal=current.principal,
                    name=target.name,
                    url=target.url,
                    data_base64=target.data_base64,
                )
                if st0 is not None and st0["status"] == "ready":
                    return await _from_store_ready(st0)
                if st0 is not None and st0["status"] == "error":
                    return {
                        "error": (
                            f"pdf {target.name!r} failed "
                            f"background ingestion: {st0['error']}"
                        ),
                        "name": target.name,
                    }
                emit.reasoning_step(
                    f"Waiting for background PDF ingestion of {target.name!r}…"
                )
                for tick in range(300):
                    st = await pdf_store.status(
                        principal=current.principal,
                        name=target.name,
                        url=target.url,
                        data_base64=target.data_base64,
                    )
                    if st is not None and st["status"] == "ready":
                        return await _from_store_ready(st)
                    if st is not None and st["status"] == "error":
                        return {
                            "error": (
                                f"pdf {target.name!r} failed "
                                f"background ingestion: {st['error']}"
                            ),
                            "name": target.name,
                        }
                    if tick > 0 and tick % 30 == 0:
                        emit.reasoning_step(
                            f"Still ingesting {target.name!r} ({tick / 10:.0f}s elapsed)…"
                        )
                    await asyncio.sleep(0.1)
            try:
                data = _resolve_pdf_bytes(target, http_client)
            except Exception as exc:
                logger.warning(
                    "pdf_extract: failed to fetch bytes for %r url=%r: %s",
                    target.name,
                    target.url,
                    exc,
                )
                return {
                    "error": (
                        f"failed to fetch pdf bytes for {target.name!r}: "
                        f"{type(exc).__name__}: {exc}. The URL may require "
                        "auth the agent server can't forward."
                    ),
                    "name": target.name,
                    "url": target.url,
                }
            try:
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    metadata = _extract_pdf_metadata(pdf)
                    toc = _extract_toc(pdf)
                    total_pages = metadata["total_pages"]
                    effective_range = page_range or _preview_range(total_pages)
                    pages_out: list[dict[str, Any]] = []
                    for idx, page in enumerate(pdf.pages, start=1):
                        if not (effective_range[0] <= idx <= effective_range[1]):
                            continue
                        text = page.extract_text() or ""
                        words = page.extract_words() or []
                        page_entry: dict[str, Any] = {
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
                        pages_out.append(page_entry)
            except Exception as exc:
                logger.warning(
                    "pdf_extract: pdfplumber failed for %r: %s", target.name, exc
                )
                return {
                    "error": (
                        f"failed to parse PDF bytes for {target.name!r}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "name": target.name,
                }
            _cite_pdf_pages(target, pages_out)
            return {
                "name": target.name,
                "total_pages": total_pages,
                "page_range": list(effective_range),
                "is_preview": page_range is None and total_pages > _PREVIEW_PAGES,
                "metadata": metadata,
                "toc": toc,
                "pages": _shape_pages(pages_out, include_words),
            }

        async def search_pdf(query: str, k: int = 8) -> list[dict[str, Any]]:
            pdf_store = services.get_pdf_store()
            if pdf_store is None:
                return []
            current = run_context.current()
            hits = await pdf_store.search(principal=current.principal, query=query, k=k)
            by_name: dict[str, list[dict[str, Any]]] = {}
            for h in hits:
                name = str(h.get("name") or "")
                if not name:
                    continue
                by_name.setdefault(name, []).append(h)
            for name, pdf_hits in by_name.items():
                target_file = _match_uploaded_pdf(current.uploaded_files, name)
                if target_file is None:
                    for h in pdf_hits:
                        page = int(h.get("page") or 0)
                        if (name, page) in pdf_cited:
                            continue
                        pdf_cited.add((name, page))
                        emit.cite(
                            source=f"{name} (p. {page})" if page else name,
                        )
                    continue
                for h in pdf_hits:
                    page = int(h.get("page") or 0)
                    if not page:
                        continue
                    page_data = await pdf_store.get_pages(
                        principal=current.principal,
                        name=name,
                        url=target_file.url,
                        data_base64=target_file.data_base64,
                        page_range=(page, page),
                    )
                    words: list[dict[str, Any]] = []
                    page_text: str = ""
                    if page_data:
                        words = list(page_data[0].get("words") or [])
                        page_text = str(page_data[0].get("text") or "")
                    _cite_pdf_search_hit(
                        target_file,
                        hit_page=page,
                        hit_text=str(h.get("text") or ""),
                        words=words,
                        page_text=page_text,
                    )
            return hits

        async def get_pdf_outline(name: str) -> dict[str, Any]:
            current = run_context.current()
            target = _match_uploaded_pdf(current.uploaded_files, name)
            if target is None:
                available = [f.name for f in current.uploaded_files if _is_pdf(f)]
                return {
                    "error": (
                        f"pdf {name!r} not found in this run's uploads. "
                        f"Available: {available}."
                    ),
                    "name": name,
                    "available": available,
                }
            pdf_store = services.get_pdf_store()
            if pdf_store is None:
                return {
                    "error": "PDF store is not configured on this server.",
                    "name": target.name,
                }
            initial_st = await pdf_store.status(
                principal=current.principal,
                name=target.name,
                url=target.url,
                data_base64=target.data_base64,
            )
            if initial_st is not None and initial_st["status"] == "ready":
                return {
                    "name": target.name,
                    "total_pages": initial_st["total_pages"],
                    "metadata": initial_st["metadata"],
                    "toc": initial_st["toc"],
                }
            if initial_st is not None and initial_st["status"] == "error":
                return {
                    "error": (
                        f"pdf {target.name!r} failed background "
                        f"ingestion: {initial_st['error']}"
                    ),
                    "name": target.name,
                }

            emit.reasoning_step(
                f"Waiting for background PDF ingestion of {target.name!r}…"
            )
            last_status: str | None = (
                initial_st["status"] if initial_st else "scheduling"
            )
            for tick in range(300):
                st = await pdf_store.status(
                    principal=current.principal,
                    name=target.name,
                    url=target.url,
                    data_base64=target.data_base64,
                )
                cur = st["status"] if st else "scheduling"
                if cur != last_status:
                    emit.reasoning_step(f"PDF ingest status for {target.name!r}: {cur}")
                last_status = cur
                if st is not None and st["status"] == "ready":
                    return {
                        "name": target.name,
                        "total_pages": st["total_pages"],
                        "metadata": st["metadata"],
                        "toc": st["toc"],
                    }
                if st is not None and st["status"] == "error":
                    return {
                        "error": (
                            f"pdf {target.name!r} failed background "
                            f"ingestion: {st['error']}"
                        ),
                        "name": target.name,
                    }
                if tick > 0 and tick % 30 == 0:
                    emit.reasoning_step(
                        f"Still ingesting {target.name!r} ({tick / 10:.0f}s elapsed)…"
                    )
                await asyncio.sleep(0.1)
            return {
                "error": (
                    f"pdf {target.name!r} is still being indexed in "
                    "the background after 30s. Call "
                    "``pdf_extract(name=..., page_range=(1, 5))`` for "
                    "an inline parse, or retry ``get_pdf_outline`` in "
                    "a few seconds."
                ),
                "name": target.name,
            }

        return [
            StructuredTool.from_function(
                list_pdfs,
                name="list_pdfs",
                description=(
                    "Index of PDFs the user has attached this turn. "
                    "Returns ``{count, pdfs: [{name, mime, url}]}``. "
                    "When empty, also includes ``document_widgets`` "
                    "(PDF-bearing widgets on the dashboard that "
                    "haven't been selected). Call AT MOST ONCE per "
                    "turn — a second call returns a STOP message."
                ),
                args_schema=_ListArgs,
            ),
            StructuredTool.from_function(
                coroutine=get_pdf_outline,
                name="get_pdf_outline",
                description=(
                    "Return ``{name, total_pages, metadata, toc}`` for "
                    "a single PDF — no page text. Cheapest navigation "
                    "primitive: use it first to plan which sections "
                    "to extract, then call ``pdf_extract`` with the "
                    "right ``page_range``."
                ),
                args_schema=_OutlineArgs,
            ),
            StructuredTool.from_function(
                coroutine=search_pdf,
                name="search_pdf",
                description=(
                    "Semantic-search the user's ingested PDFs by "
                    "natural-language ``query``. Returns up to ``k`` "
                    "page hits sorted by relevance — each entry is "
                    "``{score, name, page, text}``. Vectors are built "
                    "in the background at request ingest time, so this "
                    "is the cheapest way to locate a section without "
                    "extracting every page. Follow up with "
                    "``pdf_extract(name=..., page_range=(p, p))`` to "
                    "pull the full text + word bounding boxes for the "
                    "page you want to cite."
                ),
                args_schema=_SearchArgs,
            ),
            StructuredTool.from_function(
                coroutine=pdf_extract,
                name="pdf_extract",
                description=(
                    "Read an uploaded PDF. Two-step workflow:\n"
                    "  1. CALL FIRST with just the ``name`` (no "
                    "``page_range``) — returns ``metadata`` + ``toc`` "
                    "+ the first few pages of text (preview). Use the "
                    "``toc`` (a list of ``{level, title, page}``) to "
                    "locate the section you need.\n"
                    "  2. CALL AGAIN with the target "
                    "``page_range=(start, end)`` (1-based, inclusive) "
                    "to get the full text of that section. Add "
                    "``include_words=True`` only when you need "
                    "per-word bounding boxes for citation highlights "
                    "(they ~10× the response size).\n"
                    "\n"
                    "``name`` is fuzzy-matched (case-insensitive, "
                    "``.pdf`` suffix optional). Prefer ``search_pdf`` "
                    "for keyword/topic discovery — it returns page "
                    "hits without bytes of full extraction.\n"
                    "\n"
                    "Returns ``{name, total_pages, page_range, "
                    "is_preview, metadata, toc, pages}`` on success, "
                    "or ``{error, ...}`` on failure (read the error "
                    "message, don't retry the same call)."
                ),
                args_schema=_ExtractArgs,
            ),
        ]
