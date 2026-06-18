"""Tool-side helpers for emitting OpenBB Workspace SSE events."""

from __future__ import annotations

import contextvars
import logging
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("openbb_agent_server.emit")


_writer_override: contextvars.ContextVar[Callable[[dict[str, Any]], None] | None] = (
    contextvars.ContextVar("openbb_agent_server.emit_writer", default=None)
)


@contextmanager
def bind_writer(
    sink: Callable[[dict[str, Any]], None],
) -> Iterator[Callable[[dict[str, Any]], None]]:
    """Bind ``sink`` as the emit writer for the lifetime of the ``with`` block.

    Override the context-local writer so every ``emit`` helper called inside the
    block routes its events to ``sink`` instead of the LangGraph stream writer.
    The previous writer is restored on exit, even if the block raises.

    Parameters
    ----------
    sink : Callable[[dict[str, Any]], None]
        Callable invoked with each emitted event payload while the block is
        active.

    Yields
    ------
    Callable[[dict[str, Any]], None]
        The same ``sink`` that was bound, for convenient use as the ``with``
        target.
    """
    token = _writer_override.set(sink)
    try:
        yield sink
    finally:
        _writer_override.reset(token)


def _writer() -> Any:
    """Return the writer for the current call, or ``None``."""
    override = _writer_override.get()
    if override is not None:
        return override
    try:
        from langgraph.config import get_stream_writer
    except ImportError:  # pragma: no cover — langgraph is a hard dep
        return None
    try:
        return get_stream_writer()
    except (LookupError, RuntimeError):
        return None


def _new_uuid() -> str:
    return secrets.token_urlsafe(12)


def reasoning_step(
    message: str,
    *,
    event_type: str = "INFO",
    **details: Any,
) -> None:
    """Emit a reasoning step event to the active stream writer.

    Send a ``step`` event describing intermediate agent reasoning. If no stream
    writer is bound for the current call, the step is dropped and a warning is
    logged instead.

    Parameters
    ----------
    message : str
        Human-readable description of the reasoning step.
    event_type : str, default "INFO"
        Severity/category of the step; one of ``INFO``, ``SUCCESS``,
        ``WARNING``, or ``ERROR``.
    **details : Any
        Arbitrary structured fields attached to the step under ``details``.
    """
    w = _writer()
    if w is None:
        logger.warning("reasoning_step: no stream writer; would emit %s", message)
        return
    w(
        {
            "type": "step",
            "event_type": event_type,
            "message": message,
            "details": dict(details),
        }
    )


def _emit_artifact(payload: dict[str, Any]) -> None:
    w = _writer()
    if w is None:
        logger.warning("artifact: no stream writer; would emit %s", payload.get("name"))
        return
    w({"type": "artifact", "artifact": payload})


def html_artifact(
    *,
    content: str,
    name: str = "",
    description: str = "",
    uuid: str | None = None,
) -> str:
    """Emit an HTML artifact and return its uuid.

    Parameters
    ----------
    content : str
        Raw HTML markup rendered as the artifact body.
    name : str, default ""
        Display name shown for the artifact in the Workspace UI.
    description : str, default ""
        Longer description of the artifact's purpose or contents.
    uuid : str or None, default None
        Explicit artifact identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The artifact uuid, usable to reference this artifact later.
    """
    artifact_uuid = uuid or _new_uuid()
    _emit_artifact(
        {
            "type": "html",
            "uuid": artifact_uuid,
            "name": name,
            "description": description,
            "content": content,
        }
    )
    return artifact_uuid


def markdown_artifact(
    *,
    content: str,
    name: str = "",
    description: str = "",
    uuid: str | None = None,
) -> str:
    """Emit a markdown artifact and return its uuid.

    Parameters
    ----------
    content : str
        Markdown source rendered as the artifact body.
    name : str, default ""
        Display name shown for the artifact in the Workspace UI.
    description : str, default ""
        Longer description of the artifact's purpose or contents.
    uuid : str or None, default None
        Explicit artifact identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The artifact uuid, usable to reference this artifact later.
    """
    artifact_uuid = uuid or _new_uuid()
    _emit_artifact(
        {
            "type": "markdown",
            "uuid": artifact_uuid,
            "name": name,
            "description": description,
            "content": content,
        }
    )
    return artifact_uuid


def code_artifact(
    *,
    content: str,
    language: str = "",
    name: str = "",
    description: str = "",
    uuid: str | None = None,
) -> str:
    """Emit a code artifact and return its uuid.

    Parameters
    ----------
    content : str
        Source code rendered as the artifact body.
    language : str, default ""
        Programming language hint for syntax highlighting.
    name : str, default ""
        Display name shown for the artifact in the Workspace UI.
    description : str, default ""
        Longer description of the artifact's purpose or contents.
    uuid : str or None, default None
        Explicit artifact identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The artifact uuid, usable to reference this artifact later.
    """
    artifact_uuid = uuid or _new_uuid()
    _emit_artifact(
        {
            "type": "code",
            "uuid": artifact_uuid,
            "name": name,
            "description": description,
            "content": content,
            "language": language,
        }
    )
    return artifact_uuid


def table_artifact(
    *,
    columns: list[str],
    rows: list[list[Any]],
    name: str = "",
    description: str = "",
    uuid: str | None = None,
) -> str:
    """Emit a table artifact and return its uuid.

    Parameters
    ----------
    columns : list[str]
        Ordered column header labels for the table.
    rows : list[list[Any]]
        Row data, each inner list aligned positionally with ``columns``.
    name : str, default ""
        Display name shown for the artifact in the Workspace UI.
    description : str, default ""
        Longer description of the artifact's purpose or contents.
    uuid : str or None, default None
        Explicit artifact identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The artifact uuid, usable to reference this artifact later.
    """
    artifact_uuid = uuid or _new_uuid()
    _emit_artifact(
        {
            "type": "table",
            "uuid": artifact_uuid,
            "name": name,
            "description": description,
            "columns": columns,
            "rows": rows,
        }
    )
    return artifact_uuid


def chart_artifact(
    *,
    plotly: dict[str, Any],
    name: str = "",
    description: str = "",
    uuid: str | None = None,
) -> str:
    """Emit a chart artifact and return its uuid.

    Parameters
    ----------
    plotly : dict[str, Any]
        Plotly figure JSON (``data``/``layout``) rendered as the chart.
    name : str, default ""
        Display name shown for the artifact in the Workspace UI.
    description : str, default ""
        Longer description of the artifact's purpose or contents.
    uuid : str or None, default None
        Explicit artifact identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The artifact uuid, usable to reference this artifact later.
    """
    artifact_uuid = uuid or _new_uuid()
    _emit_artifact(
        {
            "type": "chart",
            "uuid": artifact_uuid,
            "name": name,
            "description": description,
            "plotly": plotly,
        }
    )
    return artifact_uuid


def image_artifact(
    *,
    name: str = "",
    description: str = "",
    mime: str = "image/png",
    data_base64: str | None = None,
    url: str | None = None,
    uuid: str | None = None,
) -> str:
    """Emit an image wrapped in an HTML artifact and return its uuid.

    HTML is the only image-capable wire type, so the image is rendered as an
    ``<img>`` tag. The source is either the provided ``url`` or a ``data:`` URI
    built from ``data_base64`` and ``mime``.

    Parameters
    ----------
    name : str, default ""
        Display name shown for the artifact; also used as ``alt`` text.
    description : str, default ""
        Longer description; used as ``alt`` text when ``name`` is empty.
    mime : str, default "image/png"
        MIME type used when building the ``data:`` URI from ``data_base64``.
    data_base64 : str or None, default None
        Base64-encoded image bytes; ignored when ``url`` is given.
    url : str or None, default None
        Direct image URL; takes precedence over ``data_base64``.
    uuid : str or None, default None
        Explicit artifact identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The artifact uuid, usable to reference this artifact later.

    Raises
    ------
    ValueError
        If neither ``data_base64`` nor ``url`` is provided.
    """
    if not (data_base64 or url):
        raise ValueError("image_artifact requires data_base64 or url")
    src = url or f"data:{mime};base64,{data_base64}"
    artifact_uuid = uuid or _new_uuid()
    alt = name or description or "image"
    _emit_artifact(
        {
            "type": "html",
            "uuid": artifact_uuid,
            "name": name,
            "description": description,
            "content": f'<img src="{src}" alt="{alt}" />',
        }
    )
    return artifact_uuid


def file_artifact(
    *,
    name: str = "",
    description: str = "",
    mime: str = "application/octet-stream",
    data_base64: str | None = None,
    url: str | None = None,
    uuid: str | None = None,
) -> str:
    """Emit a downloadable file as an HTML artifact and return its uuid.

    The file is rendered as an HTML anchor with a ``download`` attribute. The
    link target is either the provided ``url`` or a ``data:`` URI built from
    ``data_base64`` and ``mime``.

    Parameters
    ----------
    name : str, default ""
        Display name shown for the artifact and used as the download filename.
    description : str, default ""
        Longer description of the artifact's purpose or contents.
    mime : str, default "application/octet-stream"
        MIME type used when building the ``data:`` URI from ``data_base64``.
    data_base64 : str or None, default None
        Base64-encoded file bytes; ignored when ``url`` is given.
    url : str or None, default None
        Direct file URL; takes precedence over ``data_base64``.
    uuid : str or None, default None
        Explicit artifact identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The artifact uuid, usable to reference this artifact later.

    Raises
    ------
    ValueError
        If neither ``data_base64`` nor ``url`` is provided.
    """
    if not (data_base64 or url):
        raise ValueError("file_artifact requires data_base64 or url")
    href = url or f"data:{mime};base64,{data_base64}"
    artifact_uuid = uuid or _new_uuid()
    label = name or "Download"
    _emit_artifact(
        {
            "type": "html",
            "uuid": artifact_uuid,
            "name": name,
            "description": description,
            "content": f'<a href="{href}" download="{name}">{label}</a>',
        }
    )
    return artifact_uuid


def cite(
    *,
    text: str | None = None,
    source: str | None = None,
    source_url: str | None = None,
    quote_bounding_boxes: list[list[dict[str, Any]]] | None = None,
    widget: str | None = None,
    widget_id: str | None = None,
    input_arguments: dict[str, Any] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> None:
    """Emit a single citation event.

    Build a citation ``source_info`` for either a Workspace widget (when
    ``widget`` is given) or a web source, attach optional ``details`` and
    metadata, and emit it. The router buffers citations and flushes them as a
    single batch. If no stream writer is bound, the citation is dropped and a
    warning is logged.

    Parameters
    ----------
    text : str or None, default None
        Quoted or summarized text from the source; added to web ``details``.
    source : str or None, default None
        Human-readable source name/title.
    source_url : str or None, default None
        Origin URL of the source; added to web ``details`` as ``url``.
    quote_bounding_boxes : list[list[dict[str, Any]]] or None, default None
        Per-quote bounding-box coordinates highlighting cited regions.
    widget : str or None, default None
        Widget uuid; when set, the citation is emitted as a ``widget`` source
        rather than a ``web`` source.
    widget_id : str or None, default None
        Widget type identifier paired with ``widget``.
    input_arguments : dict[str, Any] or None, default None
        Input arguments recorded under source metadata as ``input_args``.
    extra_details : dict[str, Any] or None, default None
        Additional detail entry merged into the citation ``details`` list.
    """
    w = _writer()
    if w is None:
        logger.warning("cite: no stream writer; would emit %s", source or source_url)
        return

    source_info: dict[str, Any]
    if widget:
        source_info = {
            "type": "widget",
            "uuid": widget,
            "widget_id": widget_id,
            "name": source,
            "origin": source_url,
        }
    else:
        source_info = {
            "type": "web",
            "name": source,
            "origin": source_url,
        }
    if widget or input_arguments:
        metadata: dict[str, Any] = {}
        if widget:
            metadata["widget_uuid"] = widget
        if input_arguments:
            metadata["input_args"] = input_arguments
        source_info["metadata"] = metadata

    details: list[dict[str, Any]] | None
    details_entries: list[dict[str, Any]] = []
    if extra_details:
        details_entries.append(dict(extra_details))
    if not widget:
        details_entry: dict[str, Any] = {}
        if text:
            details_entry["text"] = text
        if source_url:
            details_entry["url"] = source_url
        if source:
            details_entry["title"] = source
        if details_entry:
            details_entries.append(details_entry)
    details = details_entries or None

    citation: dict[str, Any] = {
        "id": _new_uuid(),
        "source_info": source_info,
        "details": details,
    }
    if quote_bounding_boxes is not None:
        citation["quote_bounding_boxes"] = quote_bounding_boxes
    w({"type": "citations", "citations": [citation]})


def function_call(
    *,
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    server_id: str = "agent",
    call_id: str | None = None,
) -> str:
    """Emit a function-call event asking the Workspace UI to run a tool.

    Request execution of a client-side tool by emitting a ``function_call``
    event. If no stream writer is bound, the call is dropped and a warning is
    logged, but a call id is still returned.

    Parameters
    ----------
    tool_name : str
        Name of the client-side tool the Workspace UI should execute.
    parameters : dict[str, Any] or None, default None
        Arguments passed to the tool; an empty dict is sent when omitted.
    server_id : str, default "agent"
        Identifier of the MCP/tool server that owns ``tool_name``.
    call_id : str or None, default None
        Explicit call identifier; a new url-safe token is generated when
        omitted.

    Returns
    -------
    str
        The call id, used to correlate the eventual tool result.
    """
    cid = call_id or _new_uuid()
    w = _writer()
    if w is None:
        logger.warning("function_call: no stream writer; would emit %s", tool_name)
        return cid
    w(
        {
            "type": "function_call",
            "server_id": server_id,
            "tool_name": tool_name,
            "parameters": parameters or {},
            "call_id": cid,
        }
    )
    return cid
