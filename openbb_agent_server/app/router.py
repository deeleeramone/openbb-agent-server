"""FastAPI router — ``/agents.json`` + ``/v1/*`` endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path as _PathlibPath
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response
from sse_starlette.sse import EventSourceResponse

from openbb_agent_server.app.settings import AgentProfile, AgentServerSettings
from openbb_agent_server.memory.ingestion import ingest_request_context
from openbb_agent_server.memory.store import MemoryStore
from openbb_agent_server.memory.translation import NvidiaTranslator
from openbb_agent_server.observability.logging import TRACE, trace
from openbb_agent_server.persistence.store import HistoryStore
from openbb_agent_server.protocol.schemas import (
    FunctionCallSSE,
    FunctionCallSSEData,
    MessageChunkSSE,
    QueryRequest,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)
from openbb_agent_server.runtime import (
    context as run_context,
    registry,
    services,
)
from openbb_agent_server.runtime.builder import run_agent
from openbb_agent_server.runtime.context import (
    FileRef,
    RunContext,
    WidgetRef,
)
from openbb_agent_server.runtime.plugins import AuthBackend
from openbb_agent_server.runtime.principal import UserPrincipal
from openbb_agent_server.runtime.widget_store import (
    WidgetDataStore,
    parse_widget_data_messages,
)

logger = logging.getLogger("openbb_agent_server.router")

CONVERSATION_HEADER = "X-Trace-ID"
SERVER_TRACE_HEADER = "X-Server-Trace-ID"

_cancellations: dict[tuple[str, str], asyncio.Event] = {}

_RESERVED_BOOLEAN_FEATURES: frozenset[str] = frozenset(
    {
        "streaming",
        "widget-dashboard-select",
        "widget-dashboard-search",
        "widget-global-search",
        "file-upload",
        "generative-ui",
    }
)
_AGENT_ID_RE = re.compile(r"^[a-z0-9-]+$")
_RESERVED_CUSTOM_FEATURE_NAMES: frozenset[str] = frozenset(
    {"web search", "web-search", "websearch"}
)


async def _safe_end_trace(
    history: Any,
    principal: UserPrincipal,
    trace_id: str,
    status: str,
) -> None:
    """Close a trace best-effort for fire-and-forget background tasks."""
    try:
        await history.end_trace(principal=principal, trace_id=trace_id, status=status)
    except Exception:
        logger.warning("background end_trace failed", exc_info=True)


def _rows_from_inline_widget(data: Any) -> list[dict[str, Any]]:
    """Normalise inline widget data to a list of row dicts."""
    if not data:
        return []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if all(isinstance(r, dict) and "items" not in r for r in data):
            return [r for r in data if isinstance(r, dict)]
        rows: list[dict[str, Any]] = []
        for entry in data:
            items = entry.get("items") if isinstance(entry, dict) else None
            if isinstance(items, list):
                rows.extend(r for r in items if isinstance(r, dict))
        return rows
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [r for r in items if isinstance(r, dict)]
    return []


_PDF_FILENAME_FIELDS: tuple[str, ...] = (
    "name",
    "filename",
    "file_name",
    "title",
    "label",
    "doc_name",
    "document_name",
    "doc_id",
    "id",
)
_PDF_URL_FIELDS: tuple[str, ...] = (
    "url",
    "href",
    "link",
    "download_url",
    "file_url",
    "pdf_url",
    "source_url",
    "document_url",
    "doc_url",
)
_PDF_MIME_FIELDS: tuple[str, ...] = (
    "mime",
    "mime_type",
    "content_type",
    "type",
    "format",
    "doc_type",
    "document_type",
)
_PDF_DATA_FIELDS: tuple[str, ...] = (
    "data_base64",
    "base64",
    "content_base64",
    "bytes",
    "content",
)
_PDF_B64_PREFIX = "JVBERi0"
_PDF_RAW_PREFIX = "%PDF-"
_PDF_DATA_URL_PREFIX = "data:application/pdf"


def _extract_pdf_b64_from_string(s: str) -> str | None:
    """Extract a base64 PDF payload from a single string value."""
    if not isinstance(s, str) or not s:
        return None
    stripped = s.lstrip()
    if stripped.startswith(_PDF_B64_PREFIX):
        return stripped
    lower = stripped[:64].lower()
    if lower.startswith(_PDF_DATA_URL_PREFIX):
        _, _, payload = stripped.partition("base64,")
        payload = payload.strip()
        if payload.startswith(_PDF_B64_PREFIX):
            return payload
    if stripped.startswith(_PDF_RAW_PREFIX):
        try:
            import base64 as _b64

            return _b64.b64encode(stripped.encode("latin-1", errors="replace")).decode(
                "ascii"
            )
        except Exception:
            return None
    return None


def _extract_pdf_url_from_string(s: str) -> str | None:
    if not isinstance(s, str) or not s:
        return None
    stripped = s.strip()
    lower = stripped.lower()
    if lower.startswith(("http://", "https://")):
        return stripped
    if lower.startswith(("data:application/pdf", "data:application/octet-stream")):
        return stripped
    return None


def _scan_for_http_url(blob: Any, *, _depth: int = 0) -> str | None:
    """Recursively scan a dict / list / value for any URL we can fetch."""
    if _depth > 6:
        return None
    if isinstance(blob, str):
        return _extract_pdf_url_from_string(blob)
    if isinstance(blob, dict):
        for v in blob.values():
            found = _scan_for_http_url(v, _depth=_depth + 1)
            if found:
                return found
    elif isinstance(blob, (list, tuple)):
        for v in blob:
            found = _scan_for_http_url(v, _depth=_depth + 1)
            if found:
                return found
    return None


def _scan_for_pdf_b64(blob: Any, *, _depth: int = 0) -> str | None:
    """Recursively scan a value for a PDF byte payload."""
    if _depth > 6:
        return None
    if isinstance(blob, str):
        return _extract_pdf_b64_from_string(blob)
    if isinstance(blob, dict):
        for v in blob.values():
            found = _scan_for_pdf_b64(v, _depth=_depth + 1)
            if found:
                return found
    elif isinstance(blob, (list, tuple)):
        for v in blob:
            found = _scan_for_pdf_b64(v, _depth=_depth + 1)
            if found:
                return found
    return None


def _string_is_pdf_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip().lower()
    if not s.startswith(("http://", "https://")):
        return False
    return ".pdf" in s


def _string_is_pdf_mime(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return "pdf" in value.lower()


def _string_is_pdf_b64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 8:
        return False
    return value.lstrip().startswith(_PDF_B64_PREFIX)


def _has_pdf_data_format_marker(entry: dict[str, Any]) -> bool:
    """Return True when the entry carries an SDK PdfDataFormat marker."""
    fmt = entry.get("data_format")
    if isinstance(fmt, str) and "pdf" in fmt.lower():
        return True
    return isinstance(fmt, dict) and any(
        isinstance(v, str) and "pdf" in v.lower() for v in fmt.values()
    )


def _looks_like_pdf_ref(entry: dict[str, Any]) -> bool:
    """Return True when a row dict references a PDF."""
    if _has_pdf_data_format_marker(entry):
        return True
    for k in _PDF_URL_FIELDS:
        if _string_is_pdf_url(entry.get(k)):
            return True
    for k in _PDF_MIME_FIELDS:
        if _string_is_pdf_mime(entry.get(k)):
            return True
    for k in _PDF_DATA_FIELDS:
        if _string_is_pdf_b64(entry.get(k)):
            return True
    for k in _PDF_FILENAME_FIELDS:
        v = entry.get(k)
        if isinstance(v, str) and v.lower().strip().endswith(".pdf"):
            return True
    for v in entry.values():
        if isinstance(v, str) and _extract_pdf_b64_from_string(v):
            return True
        if isinstance(v, str) and _extract_pdf_url_from_string(v):
            lower = v.strip().lower()
            if lower.startswith("data:") or ".pdf" in lower:
                return True
    return False


def _pdf_ref_from_dict(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce a PDF-shaped row dict into a FileRef-compatible dict."""
    if not isinstance(entry, dict) or not _looks_like_pdf_ref(entry):
        return None
    url: str | None = None
    for k in _PDF_URL_FIELDS:
        v = entry.get(k)
        if isinstance(v, str) and v.strip().lower().startswith(("http://", "https://")):
            url = v.strip()
            break
    if not url:
        url = _scan_for_http_url(entry)
    data_base64: str | None = None
    for k in _PDF_DATA_FIELDS:
        v = entry.get(k)
        if _string_is_pdf_b64(v):
            data_base64 = str(v).strip()
            break
    if not data_base64:
        data_base64 = _scan_for_pdf_b64(entry)
    if not url and not data_base64:
        logger.debug(
            "pdf_ref: skipping reference with no url/data_base64 even "
            "after recursive scan (keys=%r)",
            sorted(entry.keys()),
        )
        return None
    name: str | None = None
    for k in _PDF_FILENAME_FIELDS:
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            name = v.strip()
            break
    if not name and url:
        tail = url.rsplit("/", 1)[-1]
        name = tail.split("?", 1)[0] or url
    if not name:
        for v in entry.values():
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s or len(s) > 256:
                continue
            lower = s.lower()
            if lower.startswith(("http://", "https://", "data:")):
                continue
            if s.startswith(_PDF_B64_PREFIX) or s.startswith("%PDF-"):
                continue
            name = s
            break
    if not name and data_base64:
        import hashlib as _h

        digest = _h.sha256(data_base64[:512].encode("utf-8")).hexdigest()[:12]
        name = f"document_{digest}"
    if not name:  # pragma: no cover - unreachable: a url yields a tail name and data_base64 yields a hashed name, so ``name`` is always set by line 420.
        return None
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    mime: str | None = None
    for k in _PDF_MIME_FIELDS:
        v = entry.get(k)
        if isinstance(v, str) and _string_is_pdf_mime(v):
            mime = v if "/" in v else "application/pdf"
            break
    return {
        "name": name,
        "url": url,
        "data_base64": data_base64,
        "mime": mime or "application/pdf",
    }


async def _collect_uploaded_files_with_ingest(
    body: Any, *, principal: UserPrincipal
) -> tuple[FileRef, ...]:
    """Collect uploaded files and dispatch background PDF ingestion."""
    refs = _collect_uploaded_files(body)
    pdf_store = services.get_pdf_store()
    if pdf_store is None:
        return refs
    for ref in refs:
        if not _is_pdf_ref(ref):
            continue
        try:
            await pdf_store.ingest_async(
                principal=principal,
                name=ref.name,
                url=ref.url,
                data_base64=ref.data_base64,
                mime=ref.mime,
            )
        except Exception:
            logger.warning(
                "pdf_store: background ingest dispatch failed for %r",
                ref.name,
                exc_info=True,
            )
    return refs


def _is_pdf_ref(ref: FileRef) -> bool:
    if ref.mime and "pdf" in ref.mime.lower():
        return True
    return ref.name.lower().endswith(".pdf")


def _collect_uploaded_files(body: Any) -> tuple[FileRef, ...]:
    """Build the run's uploaded_files tuple."""
    out: list[FileRef] = []
    seen: set[str] = set()
    found_in_widgets = 0
    found_in_tool_msgs = 0

    def _key(name: str, url: str | None) -> str:
        return (url or "").strip() or name.strip().lower()

    def _push(pdf: dict[str, Any], source: str) -> bool:
        nonlocal found_in_widgets, found_in_tool_msgs
        name = str(pdf.get("name") or "")
        if not name:  # pragma: no cover - unreachable: ``_walk_pdf_refs`` only yields dicts from ``_pdf_ref_from_dict``, which never returns a blank ``name``.
            return False
        canonical = {"name", "mime", "data_base64", "url"}
        extras = {k: v for k, v in pdf.items() if k not in canonical and v is not None}
        ref = FileRef(
            name=name,
            mime=str(pdf.get("mime") or "application/pdf"),
            data_base64=pdf.get("data_base64"),
            url=pdf.get("url"),
            **extras,
        )
        k = _key(ref.name, ref.url)
        if k in seen:
            return False
        seen.add(k)
        out.append(ref)
        if source == "widget":
            found_in_widgets += 1
        elif source == "tool_msg":
            found_in_tool_msgs += 1
        return True

    for f in body.uploaded_files or []:
        extras = getattr(f, "model_extra", None) or {}
        resolved_url = f.url
        if not resolved_url:
            resolved_url = _scan_for_http_url(extras)
        resolved_b64 = f.data_base64
        if not resolved_b64:
            resolved_b64 = _scan_for_pdf_b64(extras)
        if not resolved_url and not resolved_b64:
            sketch: dict[str, Any] = {}
            for k, v in extras.items():
                if isinstance(v, str):
                    sketch[k] = {
                        "type": "str",
                        "len": len(v),
                        "head": v[:80],
                    }
                elif isinstance(v, (list, tuple)):
                    sketch[k] = {
                        "type": type(v).__name__,
                        "len": len(v),
                        "first_keys": sorted(v[0].keys())
                        if v and isinstance(v[0], dict)
                        else None,
                    }
                elif isinstance(v, dict):
                    sketch[k] = {"type": "dict", "keys": sorted(v.keys())[:20]}
                else:
                    sketch[k] = {"type": type(v).__name__}
            logger.warning(
                "uploaded_files: %r has neither url nor data_base64 "
                "after recursive scan. pdf_extract will fail on this "
                "file. extra_shape=%r",
                f.name,
                sketch,
            )
        elif resolved_url and resolved_url != f.url:
            trace(
                logger,
                "uploaded_files: recovered url for %r from extras=%r",
                f.name,
                sorted(extras.keys()),
            )
        elif resolved_b64 and resolved_b64 != f.data_base64:
            trace(
                logger,
                "uploaded_files: recovered base64 PDF for %r from extras=%r",
                f.name,
                sorted(extras.keys()),
            )
        ref = FileRef(
            name=f.name,
            mime=f.mime,
            data_base64=resolved_b64,
            url=resolved_url,
        )
        k = _key(ref.name, ref.url)
        if k in seen:
            continue
        seen.add(k)
        out.append(ref)

    for bag in (body.widgets.primary or [], body.widgets.secondary or []):
        for w in bag:
            src_uuid = getattr(w, "uuid", None) or None
            src_widget_id = (getattr(w, "widget_id", "") or "").strip() or None
            for pdf in _walk_pdf_refs(getattr(w, "data", None)):
                if src_uuid:
                    pdf.setdefault("source_widget_uuid", src_uuid)
                if src_widget_id:
                    pdf.setdefault("source_widget_id", src_widget_id)
                _push(pdf, "widget")

    last_widget_hint: str | None = None
    last_widget_input_args: dict[str, Any] = {}
    last_widget_uuid: str | None = None
    for m in body.messages or []:
        role = getattr(m, "role", None)
        if role == "ai":
            envelope = _ai_envelope_from_message_safe(m)
            if envelope is not None:
                input_args = envelope.get("input_arguments") or {}
                sources = input_args.get("data_sources") or []
                if sources and isinstance(sources[0], dict):
                    last_widget_hint = (
                        str(sources[0].get("id") or "")
                        or str(sources[0].get("widget_uuid") or "")
                        or None
                    )
                    last_widget_input_args = sources[0].get("input_args") or {}
                    last_widget_uuid = str(sources[0].get("widget_uuid") or "") or None
            continue
        if role != "tool":
            last_widget_hint = None
            last_widget_input_args = {}
            last_widget_uuid = None
            continue
        hint = last_widget_hint
        hint_args = last_widget_input_args
        hint_uuid = last_widget_uuid
        last_widget_hint = None
        last_widget_input_args = {}
        last_widget_uuid = None
        for blob in (getattr(m, "data", None), getattr(m, "content", None)):
            for pdf in _walk_pdf_refs(blob):
                if hint:
                    extracted = pdf.get("name", "")
                    extracted_stem = extracted.removesuffix(".pdf").strip()
                    if (
                        extracted_stem.startswith("document_")
                        or len(extracted_stem) <= 4
                        or extracted_stem.lower()
                        in {"pdf", "content", "data_format", "data"}
                    ):
                        pdf["name"] = _build_pdf_filename(hint, hint_args)
                if hint_uuid:
                    pdf.setdefault("source_widget_uuid", hint_uuid)
                if hint:
                    pdf.setdefault("source_widget_id", hint)
                _push(pdf, "tool_msg")

    extra_channels: list[tuple[str, Any]] = [
        ("context", getattr(body, "context", None)),
        ("workspace_state", getattr(body, "workspace_state", None)),
    ]
    body_extra = getattr(body, "model_extra", None) or {}
    for k, v in body_extra.items():
        if (
            k in {"uploaded_files", "widgets", "messages"}
        ):  # pragma: no cover - unreachable: these are declared ``QueryRequest`` fields, so they never surface on ``model_extra``.
            continue
        extra_channels.append((f"body.{k}", v))
    for label, blob in extra_channels:
        if blob is None:
            continue
        before = len(out)
        for pdf in _walk_pdf_refs(blob):
            _push(pdf, "tool_msg")
        if len(out) > before:
            trace(
                logger,
                "uploaded_files: promoted %d PDF(s) from %s",
                len(out) - before,
                label,
            )

    if found_in_widgets or found_in_tool_msgs:
        trace(
            logger,
            "uploaded_files: promoted %d PDF(s) from widget.data + %d "
            "PDF(s) from tool messages (total uploaded_files=%d)",
            found_in_widgets,
            found_in_tool_msgs,
            len(out),
        )
    if not any(_is_pdf_ref(r) for r in out):
        doc_widget_uuids = [
            getattr(w, "uuid", None)
            for bag in (
                body.widgets.primary or [],
                body.widgets.secondary or [],
            )
            for w in bag
            if any(
                tok in (getattr(w, "widget_id", "") or "").lower()
                for tok in ("document", "filing", "prospectus", "pdf")
            )
        ]
        if doc_widget_uuids:
            trace(
                logger,
                "uploaded_files: zero PDF refs resolved despite "
                "document widget(s) being attached %r. The wire payload "
                "isn't matching any of the recognised PDF shapes — "
                "inspect the raw request body for the actual field "
                "names (logged earlier as 'request body dumped').",
                doc_widget_uuids,
            )
    return tuple(out)


_PDF_FILENAME_PARAM_KEYS: tuple[str, ...] = (
    "ticker",
    "symbol",
    "fund",
    "fund_id",
    "doc_name",
    "doc_id",
    "document_name",
    "document_id",
    "filename",
    "file_name",
    "id",
    "key",
    "name",
)


def _slugify_filename_segment(value: Any) -> str:
    """Coerce an arbitrary value into a filesystem-safe slug fragment."""
    import re as _re

    if isinstance(value, (list, tuple)):
        for item in value:
            slug = _slugify_filename_segment(item)
            if slug:
                return slug
        return ""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    slug = _re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    slug = slug.strip("_")
    return slug[:64]


def _build_pdf_filename(widget_hint: str, params: dict[str, Any] | None) -> str:
    """Compose a PDF filename from the widget hint and params."""
    parts: list[str] = []
    base = _slugify_filename_segment(widget_hint)
    if base:
        parts.append(base)
    if isinstance(params, dict):
        seen: set[str] = set()
        for key in _PDF_FILENAME_PARAM_KEYS:
            seg = _slugify_filename_segment(params.get(key))
            if not seg or seg in seen:
                continue
            seen.add(seg)
            parts.append(seg)
    if not parts:
        return "document.pdf"
    return "-".join(parts) + ".pdf"


def _ai_envelope_from_message_safe(msg: Any) -> dict[str, Any] | None:
    """Detect an AI envelope on a message, swallowing errors."""
    try:
        from openbb_agent_server.runtime.widget_store import (
            _ai_envelope_from_message,
        )

        return _ai_envelope_from_message(msg)
    except Exception:
        return None


def _walk_pdf_refs(payload: Any) -> list[dict[str, Any]]:
    """Walk a widget data blob and yield FileRef dicts for every PDF."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        ref = _pdf_ref_from_dict(payload)
        if ref is not None:
            return [ref]
        out: list[dict[str, Any]] = []
        for v in payload.values():
            out.extend(_walk_pdf_refs(v))
        return out
    if isinstance(payload, list):
        out = []
        for v in payload:
            out.extend(_walk_pdf_refs(v))
        return out
    return []


def _params_to_input_args(params: Any) -> dict[str, Any]:
    """Build ``DataSourceRequest.input_args`` from a widget's params."""
    if isinstance(params, list):
        out: dict[str, Any] = {}
        for p in params:
            name = getattr(p, "name", None)
            if not name:
                continue
            value = getattr(p, "current_value", None)
            if value is None:
                value = getattr(p, "default_value", None)
            out[str(name)] = value
        return out
    if isinstance(params, dict):
        return dict(params)
    return {}


def _to_widget_ref(w: Any) -> WidgetRef:
    """Convert a wire ``WidgetSpec`` to a runtime ``WidgetRef``."""
    raw_params = getattr(w, "params", None)
    if isinstance(raw_params, list):
        params: dict[str, Any] = {}
        for p in raw_params:
            name = getattr(p, "name", None)
            if not name:
                continue
            value = getattr(p, "current_value", None)
            if value is None:
                value = getattr(p, "default_value", None)
            params[str(name)] = value
    elif isinstance(raw_params, dict):
        params = dict(raw_params)
    else:
        params = {}

    uuid = w.uuid or w.widget_id or ""
    extra: dict[str, Any] = {}
    if getattr(w, "name", None):
        extra["name"] = w.name
    if getattr(w, "description", None):
        extra["description"] = w.description
    return WidgetRef(
        uuid=uuid,
        widget_id=getattr(w, "widget_id", "") or "",
        origin=getattr(w, "origin", "") or "",
        params=params,
        data=getattr(w, "data", None),
        **extra,
    )


def _post_run_extractor(profile: AgentProfile, ctx: RunContext) -> Any | None:
    """Build a fresh extractor model for the post-run memory writer."""
    try:
        provider = registry.load(
            "openbb_agent_server.model_providers",
            profile.model_provider,
            {"model_name": profile.model_name, **profile.model_config_},
        )
    except (KeyError, ValueError, RuntimeError):
        return None
    try:
        return provider.build(ctx, {})
    except (TypeError, RuntimeError):
        return None


def _coerce_feature(value: Any) -> bool:
    """Accept either a plain bool or our internal ``{"default": bool, ...}`` form."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("default", False))
    return bool(value)


def _coerce_custom_feature(slug: str, value: Any) -> dict[str, Any] | None:
    """Coerce one entry from ``profile.features`` into the custom-feature wire shape."""
    if slug.lower() in _RESERVED_CUSTOM_FEATURE_NAMES:
        raise ValueError(
            f"custom feature name {slug!r} is reserved by Workspace; "
            'use ``search-web`` (label: "Search Web") instead'
        )
    if isinstance(value, bool):
        return None
    if not isinstance(value, dict):
        return None
    description = value.get("description")
    if not description:
        return None
    label = value.get("label") or _slug_to_label(slug)
    default = bool(value.get("default", False))
    return {
        "label": str(label),
        "default": default,
        "description": str(description),
    }


def _slug_to_label(slug: str) -> str:
    """Title-case a kebab-cased slug for the UI label fallback."""
    return " ".join(word.capitalize() for word in slug.split("-") if word)


def _require_scope(principal: UserPrincipal, scope: str) -> None:
    if not principal.has_scope(scope):
        raise HTTPException(status_code=403, detail=f"missing scope: {scope}")


def build_router(
    *,
    settings: AgentServerSettings,
    auth: AuthBackend,
    history: HistoryStore,
    memory: MemoryStore,
    translator: NvidiaTranslator | None = None,
    widget_store: WidgetDataStore | None = None,
) -> APIRouter:
    """Assemble the API router with collaborators baked in.

    Wire the supplied services into a fresh :class:`APIRouter` exposing the
    Workspace agent endpoints: ``/agents.json`` registration, ``/v1/me``,
    conversation / message / cancel routes, memory CRUD, trace and usage
    lookups, and the streaming ``/v1/query`` (and per-agent
    ``/agents/{agent_name}/v1/query``) SSE endpoints.

    Parameters
    ----------
    settings : AgentServerSettings
        Server configuration, including agent profiles and ingestion knobs.
    auth : AuthBackend
        Backend used to authenticate every request into a
        :class:`~openbb_agent_server.runtime.principal.UserPrincipal`.
    history : HistoryStore
        Store for conversations, messages, traces, and usage.
    memory : MemoryStore
        Long-term memory store backing the ``/v1/memory`` routes and
        request-context ingestion.
    translator : NvidiaTranslator or None, optional
        Translator applied during context ingestion when
        ``settings.translate_for_ingestion`` is set. ``None`` disables it.
    widget_store : WidgetDataStore or None, optional
        Store for widget result rows. When provided, on-dashboard widget
        data from tool messages and inline widgets is recorded per query.
        ``None`` skips widget ingestion.

    Returns
    -------
    APIRouter
        Router with all agent endpoints registered, ready to mount on the
        FastAPI app.
    """

    router = APIRouter()

    async def _principal(request: Request) -> UserPrincipal:
        return await auth.authenticate(request)

    def _registration(profile_name: str) -> dict[str, Any]:
        profile = settings.resolve_profile(profile_name)
        meta = profile.metadata
        query_path = (
            "/v1/query"
            if profile_name == settings.default_profile
            else f"/agents/{profile_name}/v1/query"
        )
        features: dict[str, Any] = {}
        for key, value in (profile.features or {}).items():
            if key in _RESERVED_BOOLEAN_FEATURES:
                features[key] = _coerce_feature(value)
                continue
            custom = _coerce_custom_feature(key, value)
            if custom is not None:
                features[key] = custom
        features.setdefault("streaming", True)
        entry: dict[str, Any] = {
            "name": meta.name,
            "description": meta.description,
            "endpoints": {"query": query_path},
            "features": features,
        }
        if meta.image_url:
            entry["image"] = meta.image_url
        return entry

    @router.get("/agents.json")
    async def agents_json() -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in settings.all_profile_names():
            if not _AGENT_ID_RE.match(name):
                logger.warning(
                    "agents.json: dropping profile %r — agent_id must match "
                    "%s per the Workspace spec",
                    name,
                    _AGENT_ID_RE.pattern,
                )
                continue
            try:
                out[name] = _registration(name)
            except KeyError:  # pragma: no cover - unreachable: ``all_profile_names`` only yields names ``resolve_profile`` can resolve.
                continue
            except ValueError as exc:
                logger.warning(
                    "agents.json: dropping profile %r — invalid config: %s",
                    name,
                    exc,
                )
                continue
        return out

    @router.get("/v1/me")
    async def me(principal: UserPrincipal = Depends(_principal)) -> dict[str, Any]:
        return {
            "user_id": principal.user_id,
            "display_name": principal.display_name,
            "email": principal.email,
            "scopes": list(principal.scopes),
        }

    @router.delete("/v1/me", status_code=204)
    async def delete_me(
        principal: UserPrincipal = Depends(_principal),
    ) -> Response:
        await memory.delete_all_for_user(principal)
        await history.delete_user(principal)
        return Response(status_code=204)

    @router.get("/v1/conversations")
    async def list_conversations(
        principal: UserPrincipal = Depends(_principal),
        limit: int = 50,
    ) -> dict[str, Any]:
        rows = await history.list_conversations(principal=principal, limit=limit)
        return {"conversations": rows}

    @router.get("/v1/conversations/{conversation_id}/messages")
    async def list_messages(
        conversation_id: str = Path(...),
        principal: UserPrincipal = Depends(_principal),
        limit: int = 200,
    ) -> dict[str, Any]:
        rows = await history.get_messages(
            principal=principal,
            conversation_id=conversation_id,
            limit=limit,
        )
        return {"messages": [r.model_dump() for r in rows]}

    @router.post("/v1/conversations/{conversation_id}/cancel", status_code=202)
    async def cancel_conversation(
        conversation_id: str = Path(...),
        principal: UserPrincipal = Depends(_principal),
    ) -> dict[str, Any]:
        cancelled: list[str] = []
        for (uid, rid), ev in list(_cancellations.items()):
            if uid == principal.user_id:
                ev.set()
                cancelled.append(rid)
        return {"cancelled_runs": cancelled, "conversation_id": conversation_id}

    @router.get("/v1/memory")
    async def list_memory(
        principal: UserPrincipal = Depends(_principal),
        limit: int = 100,
    ) -> dict[str, Any]:
        _require_scope(principal, "memory:read")
        rows = await memory.list_memories(principal=principal, limit=limit)
        return {"memories": [r.model_dump() for r in rows]}

    @router.patch("/v1/memory/{memory_id}")
    async def patch_memory(
        memory_id: str = Path(...),
        body: dict[str, Any] | None = None,
        principal: UserPrincipal = Depends(_principal),
    ) -> dict[str, Any]:
        _require_scope(principal, "memory:write")
        body = body or {}
        if "pinned" in body:
            row = await memory.pin(
                principal=principal,
                memory_id=memory_id,
                pinned=bool(body["pinned"]),
            )
            if row is None:
                raise HTTPException(status_code=404, detail="memory not found")
            return row.model_dump()
        raise HTTPException(status_code=400, detail="no recognised fields to update")

    @router.delete("/v1/memory/{memory_id}", status_code=204)
    async def delete_memory(
        memory_id: str = Path(...),
        principal: UserPrincipal = Depends(_principal),
    ) -> Response:
        _require_scope(principal, "memory:write")
        ok = await memory.forget(principal=principal, memory_id=memory_id)
        if not ok:
            raise HTTPException(status_code=404, detail="memory not found")
        return Response(status_code=204)

    @router.get("/v1/traces/{trace_id}")
    async def get_trace(
        trace_id: str = Path(...),
        principal: UserPrincipal = Depends(_principal),
    ) -> dict[str, Any]:
        bundle = await history.get_trace_bundle(
            principal=principal,
            trace_id=trace_id,
        )
        if bundle is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return bundle

    @router.get("/v1/usage")
    async def get_usage(
        principal: UserPrincipal = Depends(_principal),
        trace_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        return await history.usage_summary(
            principal=principal,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )

    async def _run_query(  # noqa: PLR0912 — top-level endpoint orchestration.
        body: QueryRequest,
        request: Request,
        principal: UserPrincipal,
        agent_name: str,
    ) -> EventSourceResponse:
        _require_scope(principal, "agent:query")
        try:
            profile = settings.resolve_profile(agent_name)
        except KeyError:
            raise HTTPException(status_code=404, detail="agent not found")
        await history.upsert_user(principal)

        if logger.isEnabledFor(TRACE):
            try:
                raw_body = await request.body()
                if raw_body:
                    fname = f"agent-query-{int(time.time() * 1000)}-{os.getpid()}.json"
                    dump_path = _PathlibPath(tempfile.gettempdir()) / fname
                    dump_path.write_bytes(raw_body)
                    trace(
                        logger,
                        "request body dumped | bytes=%d path=%s",
                        len(raw_body),
                        dump_path,
                    )
            except Exception as exc:
                logger.warning("RAW BODY dump failed: %s", exc)

        last_role = body.messages[-1].role if body.messages else None
        n_primary = len(body.widgets.primary) if body.widgets else 0
        n_secondary = len(body.widgets.secondary) if body.widgets else 0
        n_extra = len(body.widgets.extra) if body.widgets else 0
        n_tool_msgs = sum(1 for m in body.messages if m.role == "tool")
        n_uploaded = len(body.uploaded_files or [])
        trace(
            logger,
            "query received | agent=%s msgs=%d last_role=%s tool_msgs=%d primary=%d secondary=%d extra=%d uploaded_files=%d",
            agent_name,
            len(body.messages),
            last_role,
            n_tool_msgs,
            n_primary,
            n_secondary,
            n_extra,
            n_uploaded,
        )
        for idx, f in enumerate(body.uploaded_files or []):
            try:
                trace(
                    logger,
                    "  uploaded_files[%d] name=%r mime=%r url=%r has_data_base64=%s extra_keys=%r",
                    idx,
                    getattr(f, "name", None),
                    getattr(f, "mime", None),
                    getattr(f, "url", None),
                    bool(getattr(f, "data_base64", None)),
                    sorted(
                        k
                        for k in (f.model_extra or {})
                        if k not in {"name", "mime", "url", "data_base64"}
                    ),
                )
            except Exception:  # pragma: no cover - defensive: ``getattr`` / ``sorted`` over a validated ``UploadedFile`` cannot raise.
                logger.exception("uploaded_files log entry failed")
        for bag_name, bag in (
            ("primary", body.widgets.primary if body.widgets else []),
            ("secondary", body.widgets.secondary if body.widgets else []),
            ("extra", body.widgets.extra if body.widgets else []),
        ):
            for idx, w in enumerate(bag or []):
                wid = getattr(w, "widget_id", "") or ""
                data_attr = getattr(w, "data", None)
                wid_lower = wid.lower()
                is_file_like = (
                    "file" in wid_lower
                    or wid == "file-undefined"
                    or "document" in wid_lower
                    or "filing" in wid_lower
                    or "prospectus" in wid_lower
                    or "pdf" in wid_lower
                )
                if is_file_like:
                    data_type = type(data_attr).__name__
                    data_len = (
                        len(data_attr)
                        if isinstance(data_attr, (list, tuple, dict, str))
                        else None
                    )
                    data_preview: Any
                    try:
                        if isinstance(data_attr, (list, tuple)):
                            data_preview = list(data_attr)[:2]
                        elif isinstance(data_attr, dict):
                            data_preview = {
                                k: data_attr[k] for k in list(data_attr.keys())[:5]
                            }
                        else:
                            data_preview = repr(data_attr)[:300]
                    except Exception:  # pragma: no cover - defensive: JSON-decoded ``data`` cannot raise on slice / repr / dict-comprehension.
                        data_preview = "<unrepr-able>"
                    extra_keys: list[str] = []
                    extra_blob: dict[str, Any] = {}
                    try:
                        model_extra = getattr(w, "model_extra", None) or {}
                        extra_keys = sorted(model_extra.keys())
                        extra_blob = dict(model_extra)
                    except Exception:  # pragma: no cover - defensive: ``model_extra`` on a validated ``WidgetSpec`` is always a plain dict.
                        extra_keys = []
                    trace(
                        logger,
                        "  widgets.%s[%d] FILE-LIKE | uuid=%r widget_id=%r name=%r "
                        "origin=%r params=%r data_type=%s data_len=%s "
                        "data_preview=%r extra_keys=%r extra_blob=%r",
                        bag_name,
                        idx,
                        getattr(w, "uuid", None),
                        wid,
                        getattr(w, "name", None),
                        getattr(w, "origin", None),
                        getattr(w, "params", None),
                        data_type,
                        data_len,
                        data_preview,
                        extra_keys,
                        {
                            k: (
                                v
                                if not isinstance(v, str)
                                else (
                                    v[:200] + f"...<{len(v)} chars>"
                                    if len(v) > 200
                                    else v
                                )
                            )
                            for k, v in extra_blob.items()
                        },
                    )

        if logger.isEnabledFor(TRACE):
            full = body.model_dump(exclude_none=True)
            for top_key, top_val in full.items():
                if top_key == "messages":
                    for idx, m in enumerate(top_val):
                        content = m.get("content")
                        if isinstance(content, str) and len(content) > 400:
                            content = (
                                content[:400] + f"...<{len(m.get('content'))} chars>"
                            )
                        data_repr = (
                            json.dumps(m.get("data"), default=str)[:4000]
                            if m.get("data") is not None
                            else None
                        )
                        trace(
                            logger,
                            "  msg[%d] role=%s function=%s tool_call_id=%s content=%s data=%s",
                            idx,
                            m.get("role"),
                            m.get("function"),
                            m.get("tool_call_id"),
                            content,
                            data_repr,
                        )
                else:
                    blob = json.dumps(top_val, default=str)
                    if len(blob) > 4000:
                        blob = blob[:4000] + f"...<{len(blob)} chars>"
                    trace(logger, "  body.%s=%s", top_key, blob)

        last_msg = body.messages[-1] if body.messages else None
        last_role_pre = getattr(last_msg, "role", None) if last_msg else None
        prior_tool_results = sum(
            1 for m in body.messages if getattr(m, "role", None) == "tool"
        )
        fetchable_primary = [
            w
            for w in (body.widgets.primary if body.widgets else [])
            if not str(getattr(w, "widget_id", "") or "").startswith("file-")
        ]
        if (
            body.widgets is not None
            and fetchable_primary
            and last_role_pre == "human"
            and prior_tool_results == 0
            and not any(
                _rows_from_inline_widget(getattr(w, "data", None))
                for w in fetchable_primary
            )
        ):
            data_sources: list[dict[str, Any]] = []
            for w in fetchable_primary:
                data_sources.append(
                    {
                        "widget_uuid": str(getattr(w, "uuid", "") or ""),
                        "origin": str(getattr(w, "origin", "") or ""),
                        "id": str(getattr(w, "widget_id", "") or ""),
                        "input_args": _params_to_input_args(getattr(w, "params", None)),
                    }
                )
            call_id = str(uuid.uuid4())
            fcall_event = FunctionCallSSE(
                data=FunctionCallSSEData(
                    function="get_widget_data",
                    input_arguments={"data_sources": data_sources},
                    extra_state={"call_id": call_id},
                )
            )
            widget_names = ", ".join(
                str(getattr(w, "name", None) or getattr(w, "widget_id", "") or "widget")
                for w in body.widgets.primary
            )
            reasoning_event = StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message=f"Fetching {widget_names} data from Workspace…",
                )
            )
            tool_call_event = StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Calling tool: get_widget_data",
                )
            )
            conversation_id = request.headers.get(CONVERSATION_HEADER) or str(
                uuid.uuid4()
            )
            logger.debug(
                "router preempt: yielding reasoning + get_widget_data SSE | %d data_source(s)",
                len(data_sources),
            )

            done_event = StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="SUCCESS",
                    message="Widget data dispatched.",
                    hidden=True,
                )
            )

            async def retrieve_widget_data() -> AsyncIterator[dict[str, Any]]:
                yield {
                    "event": reasoning_event.event,
                    "data": reasoning_event.data.model_dump_json(),
                }
                yield {
                    "event": tool_call_event.event,
                    "data": tool_call_event.data.model_dump_json(),
                }
                yield {
                    "event": fcall_event.event,
                    "data": fcall_event.data.model_dump_json(),
                }
                yield {
                    "event": done_event.event,
                    "data": done_event.data.model_dump_json(),
                }

            return EventSourceResponse(
                retrieve_widget_data(),
                headers={CONVERSATION_HEADER: conversation_id},
                media_type="text/event-stream",
            )

        body_conv = getattr(body, "conversation_id", None)
        conversation_id = (
            body_conv or request.headers.get(CONVERSATION_HEADER) or str(uuid.uuid4())
        )
        trace_id = str(uuid.uuid4())
        body_run = getattr(body, "run_id", None)
        turn_idx = sum(1 for m in body.messages if getattr(m, "role", None) == "human")
        run_id = body_run or f"{conversation_id}:turn{turn_idx}"

        if widget_store is not None:
            try:
                attached_uuids: set[str] = set()
                attached_widget_ids: set[str] = set()
                for bag in (
                    body.widgets.primary or [],
                    body.widgets.secondary or [],
                ):
                    for w in bag:
                        wu = str(getattr(w, "uuid", "") or "")
                        wi = str(getattr(w, "widget_id", "") or "")
                        if wu:
                            attached_uuids.add(wu)
                        if wi:
                            attached_widget_ids.add(wi)

                def _is_on_dashboard(uuid_value: str, name_value: str) -> bool:
                    return (bool(uuid_value) and uuid_value in attached_uuids) or (
                        bool(name_value) and name_value in attached_widget_ids
                    )

                ingests = parse_widget_data_messages(body.messages)
                actually_ingested = 0
                skipped_off_dashboard = 0
                for ing in ingests:
                    rows = ing.get("rows") or []
                    first_row_preview: Any = None
                    is_error = False
                    if rows:
                        try:
                            first = rows[0]
                            first_row_preview = (
                                {k: first[k] for k in list(first.keys())[:6]}
                                if isinstance(first, dict)
                                else repr(first)[:200]
                            )
                            if (
                                isinstance(first, dict)
                                and "error_type" in first
                                and len(rows) == 1
                            ):
                                is_error = True
                        except Exception:  # pragma: no cover - defensive: rows from ``_extract_rows`` are plain dicts, so ``.keys()`` cannot raise.
                            first_row_preview = "<unrepr-able>"
                    wu = str(ing.get("widget_uuid") or "")
                    wname = str(ing.get("widget_name") or "")
                    on_dashboard = _is_on_dashboard(wu, wname)
                    trace(
                        logger,
                        "  ingest candidate | widget_uuid=%r widget_name=%r "
                        "origin=%r columns=%r row_count=%d is_error=%s "
                        "on_dashboard=%s first_row=%r",
                        ing.get("widget_uuid"),
                        ing.get("widget_name"),
                        ing.get("origin"),
                        ing.get("columns"),
                        len(rows),
                        is_error,
                        on_dashboard,
                        first_row_preview,
                    )
                    if is_error or not ing.get("widget_uuid") or not rows:
                        continue
                    if not on_dashboard:
                        skipped_off_dashboard += 1
                        continue
                    await widget_store.record(
                        principal=principal,
                        conversation_id=conversation_id,
                        widget_uuid=ing["widget_uuid"],
                        widget_name=ing.get("widget_name"),
                        origin=ing.get("origin"),
                        input_args=ing.get("input_args") or {},
                        rows=rows,
                        columns=ing.get("columns"),
                    )
                    actually_ingested += 1
                if ingests:
                    trace(
                        logger,
                        "widget_store: %d/%d tool-message widget result(s) "
                        "ingested (%d skipped: off-dashboard; rest were "
                        "error payloads / empty) for conversation_id=%s",
                        actually_ingested,
                        len(ingests),
                        skipped_off_dashboard,
                        conversation_id,
                    )
                inline_count = 0
                for bag_name, bag in (
                    ("primary", body.widgets.primary or []),
                    ("secondary", body.widgets.secondary or []),
                ):
                    for w in bag:
                        rows = _rows_from_inline_widget(getattr(w, "data", None))
                        if not rows:
                            continue
                        widget_uuid = str(getattr(w, "uuid", "") or "") or str(
                            getattr(w, "widget_id", "") or ""
                        )
                        if not widget_uuid:
                            continue
                        await widget_store.record(
                            principal=principal,
                            conversation_id=conversation_id,
                            widget_uuid=widget_uuid,
                            widget_name=getattr(w, "name", None)
                            or getattr(w, "widget_id", None),
                            origin=getattr(w, "origin", None),
                            input_args=_params_to_input_args(
                                getattr(w, "params", None)
                            ),
                            rows=rows,
                            columns=list(rows[0].keys()) if rows else None,
                        )
                        inline_count += 1
                if inline_count:
                    trace(
                        logger,
                        "widget_store: ingested %d inline widget(s) (e.g. uploads) "
                        "from bag=%s for conversation_id=%s",
                        inline_count,
                        bag_name,
                        conversation_id,
                    )
            except Exception:
                logger.warning(
                    "widget_store: ingestion failed; continuing without it",
                    exc_info=True,
                )

        ctx = RunContext(
            principal=principal,
            trace_id=trace_id,
            run_id=run_id,
            conversation_id=conversation_id,
            agent_name=profile.name,
            timezone=body.timezone,
            widgets=tuple(
                _to_widget_ref(w)
                for w in (
                    *body.widgets.primary,
                    *body.widgets.secondary,
                    *body.widgets.extra,
                )
            ),
            uploaded_files=await _collect_uploaded_files_with_ingest(
                body, principal=principal
            ),
            api_keys={
                str(k): str(v)
                for k, v in (body.api_keys or {}).items()
                if v is not None
            },
            api_urls={
                str(k): str(v)
                for k, v in (body.api_urls or {}).items()
                if v is not None
            },
            tools=tuple(dict(t) for t in (body.tools or [])),
            workspace_options=dict(body.workspace_options),
        )

        await history.begin_trace(
            principal=principal,
            trace_id=trace_id,
            conversation_id=conversation_id,
            run_id=run_id,
        )

        if body.messages and body.messages[-1].role == "human":
            last_content = body.messages[-1].content
            await history.append_message(
                principal=principal,
                conversation_id=conversation_id,
                role="human",
                content=last_content
                if isinstance(last_content, str)
                else str(last_content or ""),
                trace_id=trace_id,
            )

        try:
            await ingest_request_context(
                principal=principal,
                store=memory,
                body=body,
                trace_id=trace_id,
                char_threshold=settings.ingest_char_threshold,
                chunk_chars=settings.ingest_chunk_chars,
                chunk_overlap=settings.ingest_chunk_overlap,
                translator=translator if settings.translate_for_ingestion else None,
                translate_target_lang=settings.ingest_target_language,
            )
        except Exception:
            logger.warning("context ingestion errored", exc_info=True)

        cancel_event = asyncio.Event()
        _cancellations[(principal.user_id, run_id)] = cancel_event

        async def stream() -> AsyncIterator[dict[str, Any]]:  # noqa: PLR0912
            """Drive the agent loop to completion."""
            assembled: list[str] = []
            client_gone = False
            run_iter = run_agent(
                ctx=ctx,
                body=body,
                settings=settings,
                profile=profile,
            )
            with run_context.bind(ctx):
                try:
                    async for ev in run_iter:
                        if logger.isEnabledFor(TRACE):
                            _wire = ev.data.model_dump_json()
                            _truncate = (
                                4000 if ev.event == "copilotCitationCollection" else 400
                            )
                            trace(
                                logger,
                                "sse out | event=%s data=%r",
                                ev.event,
                                _wire[:_truncate],
                            )
                        if (
                            cancel_event.is_set()
                        ):  # pragma: no cover - cross-process cancel API
                            try:
                                await asyncio.shield(
                                    history.end_trace(
                                        principal=principal,
                                        trace_id=trace_id,
                                        status="cancelled",
                                    )
                                )
                            except Exception:
                                logger.warning(
                                    "end_trace failed on explicit cancel",
                                    exc_info=True,
                                )
                            return
                        if isinstance(ev, MessageChunkSSE):
                            assembled.append(ev.data.delta)
                        if not client_gone and await request.is_disconnected():
                            logger.debug(
                                "client disconnected; draining run %s in background",
                                run_id,
                            )
                            client_gone = True
                        if not client_gone:
                            yield {
                                "event": ev.event,
                                "data": ev.data.model_dump_json(),
                            }
                        if isinstance(ev, FunctionCallSSE):
                            logger.info(
                                "sse: closing after function call dispatch | trace_id=%s",
                                trace_id,
                            )
                            _bg = asyncio.create_task(
                                _safe_end_trace(
                                    history, principal, trace_id, "dispatched"
                                )
                            )
                            _bg.add_done_callback(lambda _t: None)
                            return
                    final_ai_text = "".join(assembled)
                    if final_ai_text:
                        await history.append_message(
                            principal=principal,
                            conversation_id=conversation_id,
                            role="ai",
                            content=final_ai_text,
                            trace_id=trace_id,
                        )
                    await history.end_trace(
                        principal=principal,
                        trace_id=trace_id,
                        status="completed",
                    )
                except asyncio.CancelledError:  # pragma: no cover - SSE teardown path
                    logger.info(
                        "agent run cancelled (client disconnect / SSE teardown)",
                        extra={"run_id": run_id, "trace_id": trace_id},
                    )
                    try:
                        await asyncio.shield(
                            history.end_trace(
                                principal=principal,
                                trace_id=trace_id,
                                status="cancelled",
                            )
                        )
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.warning(
                            "end_trace failed during cancellation",
                            exc_info=True,
                        )
                    raise
                except Exception as exc:
                    from langgraph.errors import GraphBubbleUp

                    if isinstance(exc, GraphBubbleUp):
                        logger.debug(
                            "agent paused at interrupt — closing SSE",
                        )
                        try:
                            await asyncio.shield(
                                history.end_trace(
                                    principal=principal,
                                    trace_id=trace_id,
                                    status="paused",
                                )
                            )
                        except Exception:
                            logger.warning(
                                "end_trace failed after interrupt",
                                exc_info=True,
                            )
                        return

                    logger.exception("agent run failed")
                    err = StatusUpdateSSE(
                        data=StatusUpdateSSEData(
                            eventType="ERROR",
                            message=str(exc),
                            details=[{"trace_id": trace_id}],
                        )
                    )
                    if not client_gone:
                        try:
                            yield {
                                "event": err.event,
                                "data": err.data.model_dump_json(),
                            }
                        except (
                            asyncio.CancelledError,
                            GeneratorExit,
                        ):  # pragma: no cover - disconnect race
                            logger.debug(
                                "could not deliver error frame — client already gone",
                            )
                    try:
                        await asyncio.shield(
                            history.end_trace(
                                principal=principal,
                                trace_id=trace_id,
                                status="error",
                            )
                        )
                    except Exception:
                        logger.warning(
                            "end_trace failed after error path",
                            exc_info=True,
                        )
                finally:
                    _cancellations.pop((principal.user_id, run_id), None)

        return EventSourceResponse(
            stream(),
            headers={
                CONVERSATION_HEADER: conversation_id,
                SERVER_TRACE_HEADER: trace_id,
            },
            ping=15,
        )

    @router.post("/v1/query")
    async def query_default(
        body: QueryRequest,
        request: Request,
        principal: UserPrincipal = Depends(_principal),
    ) -> EventSourceResponse:
        return await _run_query(body, request, principal, settings.default_profile)

    @router.post("/agents/{agent_name}/v1/query")
    async def query_named(
        body: QueryRequest,
        request: Request,
        agent_name: str = Path(...),
        principal: UserPrincipal = Depends(_principal),
    ) -> EventSourceResponse:
        return await _run_query(body, request, principal, agent_name)

    return router
