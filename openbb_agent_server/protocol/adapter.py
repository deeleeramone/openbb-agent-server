"""DeepAgents stream → OpenBB SSE adapter."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

logger = logging.getLogger("openbb_agent_server.protocol.adapter")

from openbb_agent_server.observability.logging import trace
from openbb_agent_server.protocol.schemas import (
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    ClientArtifact,
    FunctionCallSSE,
    FunctionCallSSEData,
    MessageArtifactSSE,
    MessageChunkSSE,
    MessageChunkSSEData,
    SSEEvent,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)

_THINKING_RE = re.compile(
    r"<(?P<tag>thinking|think|reasoning)\b[^>]*>(?P<body>.*?)</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)

_THINKING_OPEN_RE = re.compile(
    r"<(thinking|think|reasoning)\b[^>]*>",
    re.IGNORECASE,
)

_THINKING_CLOSE_RE = re.compile(
    r"</(thinking|think|reasoning)\s*>",
    re.IGNORECASE,
)

_PARTIAL_TAG_HOLD = 32

_CITATION_MARKER_RE = re.compile(
    r"【(?:[^】]*(?:cursor|loc|source|ref|cite)[^】]*|[A-Za-z0-9_-]{8,})】"
    r"|<\|start_citation_id\|>[^<]*<\|end_citation_id\|>"
    r"|<\|[a-z0-9_]*citation[a-z0-9_]*\|>",
    re.IGNORECASE,
)

_HARMONY_TRIGGER_RE = re.compile(
    r"(?:【\s*assistant\s+to=functions\.|<\|(?:channel|start)\|>)",
    re.IGNORECASE,
)


class _ThinkingStreamSplitter:
    """Stateful streaming splitter for inline ``<thinking>`` markers."""

    def __init__(self) -> None:
        self._buf: str = ""
        self._in_thinking: bool = False
        self._saw_open_tag: bool = False
        self._signalled_implicit: bool = False
        self._harmony_suppress: bool = False

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """Feed a text delta. Returns zero or more ``(channel, text)`` pairs."""
        if not delta or self._harmony_suppress:
            return []
        self._buf += delta
        harmony_match = _HARMONY_TRIGGER_RE.search(self._buf)
        if harmony_match:
            self._buf = self._buf[: harmony_match.start()]
            self._harmony_suppress = True
        out: list[tuple[str, str]] = []
        while self._buf:
            channel = "thinking" if self._in_thinking else "prose"
            if self._in_thinking:
                m = _THINKING_CLOSE_RE.search(self._buf)
                toggles_state = True
            else:
                open_m = _THINKING_OPEN_RE.search(self._buf)
                close_m = _THINKING_CLOSE_RE.search(self._buf)
                if open_m and (not close_m or open_m.start() <= close_m.start()):
                    m = open_m
                    toggles_state = True
                    self._saw_open_tag = True
                elif close_m:
                    m = close_m
                    toggles_state = False
                    if not self._saw_open_tag and not self._signalled_implicit:
                        channel = "thinking"
                        out.append(("close_unmatched", ""))
                        self._signalled_implicit = True
                else:
                    m = None
                    toggles_state = False
            if m is None:
                safe_end = self._safe_emit_end(self._buf)
                if safe_end > 0:
                    out.append((channel, self._buf[:safe_end]))
                    self._buf = self._buf[safe_end:]
                break
            if m.start() > 0:
                out.append((channel, self._buf[: m.start()]))
            if toggles_state:
                self._in_thinking = not self._in_thinking
            self._buf = self._buf[m.end() :]
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit any held-back tail. Call at end of message stream."""
        if not self._buf:
            return []
        channel = "thinking" if self._in_thinking else "prose"
        out = [(channel, self._buf)]
        self._buf = ""
        return out

    @staticmethod
    def _safe_emit_end(buf: str) -> int:
        """Index up to which it's safe to flush without splitting a marker."""
        cap = len(buf)
        open_brk = buf.rfind("【")
        if open_brk >= 0 and "】" not in buf[open_brk:]:
            cap = open_brk
        cite_tok = buf.rfind("<|start_citation_id|>")
        if cite_tok >= 0 and "<|end_citation_id|>" not in buf[cite_tok:]:
            cap = min(cap, cite_tok)
        last_lt = buf.rfind("<")
        if last_lt >= 0 and len(buf) - last_lt <= _PARTIAL_TAG_HOLD:
            cap = min(cap, last_lt)
        return cap


CLIENT_SIDE_TOOL_PREFIX = "client:"


_WORKSPACE_NATIVE_FUNCTIONS: frozenset[str] = frozenset(
    {
        "get_widget_data",
        "get_extra_widget_data",
        "get_params_options",
        "add_widget_to_dashboard",
        "add_generative_widget",
        "update_widget_in_dashboard",
        "assign_tasks_to_agents",
        "execute_agent_tool",
        "manage_navigation_bar",
        "get_skill_content",
    }
)


_WORKSPACE_ARTIFACT_TYPES: frozenset[str] = frozenset(
    {"text", "table", "chart", "code", "html"}
)


_ArtifactWireType = Literal["text", "table", "chart", "code", "html"]


def _resolve_artifact_wire_type(raw_type: str) -> _ArtifactWireType:
    if raw_type == "markdown":
        return "text"
    if raw_type in _WORKSPACE_ARTIFACT_TYPES:
        return cast(_ArtifactWireType, raw_type)
    return "text"


def _resolve_artifact_table_content(
    payload: dict[str, Any], raw_content: Any
) -> str | list[dict[str, Any]]:
    cols = payload.get("columns") or []
    rows = payload.get("rows") or []
    if cols and rows:
        return [
            {col: row[i] if i < len(row) else None for i, col in enumerate(cols)}
            for row in rows
        ]
    if isinstance(raw_content, list):
        return raw_content
    return str(raw_content) if raw_content is not None else ""


def _build_artifact(payload: dict[str, Any]) -> ClientArtifact:
    """Coerce a tool's artifact dict into the wire ``ClientArtifact`` shape."""
    wire_type = _resolve_artifact_wire_type(str(payload.get("type") or "text").lower())
    name = str(payload.get("name") or "")
    description = str(payload.get("description") or "")
    uuid = str(payload.get("uuid") or "")
    raw_content = payload.get("content")

    content: str | list[dict[str, Any]]
    chart_params: dict[str, Any] | None = None

    if wire_type == "table":
        content = _resolve_artifact_table_content(payload, raw_content)
    elif wire_type == "chart":
        plotly = payload.get("plotly")
        if isinstance(plotly, dict):
            chart_params = plotly
            content = ""
        else:
            content = str(raw_content) if raw_content is not None else ""
    elif wire_type == "code":
        content = str(raw_content) if raw_content is not None else ""
    elif isinstance(raw_content, (str, list)):
        content = raw_content
    elif raw_content is None:
        content = ""
    else:
        content = str(raw_content)

    return ClientArtifact(
        type=wire_type,
        name=name,
        description=description,
        uuid=uuid,
        content=content,
        chart_params=chart_params,
    )


def _flatten_reasoning(raw: Any) -> str:
    """Normalise ``additional_kwargs.reasoning_content`` to a flat string."""
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
        )
    return str(raw)


def _split_thinking(text: str) -> tuple[list[str], str]:
    """Split text into ``(thinking_blocks, remaining_prose)``."""
    thinking_blocks: list[str] = []
    parts: list[str] = []
    last_end = 0
    for m in _THINKING_RE.finditer(text):
        parts.append(text[last_end : m.start()])
        body = m.group("body").strip()
        if body:
            thinking_blocks.append(body)
        last_end = m.end()
    parts.append(text[last_end:])
    remaining = "".join(parts)
    remaining = _CITATION_MARKER_RE.sub("", remaining)
    remaining = re.sub(r"[ \t]+([.,;:!?])", r"\1", remaining)
    return thinking_blocks, remaining


def _coerce_status_event_type(
    value: Any,
) -> Literal["INFO", "SUCCESS", "WARNING", "ERROR"]:
    """Normalise to ``{INFO, WARNING, ERROR}`` for plugin-emitted steps."""
    raw = str(value or "INFO").upper()
    if raw == "WARNING":
        return "WARNING"
    if raw == "ERROR":
        return "ERROR"
    return "INFO"


def _coerce_status_details(
    raw: Any,
) -> list[dict[str, Any] | str] | None:
    """Workspace expects ``details`` as a list, never a flat dict."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, (dict, str))]
    if isinstance(raw, dict):
        return [raw] if raw else None
    if isinstance(raw, str):
        return [raw]
    return [str(raw)]


def _build_function_call(
    *,
    server_id: str,
    function: str,
    input_arguments: dict[str, Any],
    call_id: str,
) -> FunctionCallSSE:
    """Wrap a tool call in the right ``FunctionCallSSEData`` shape."""
    if function in _WORKSPACE_NATIVE_FUNCTIONS:
        from openbb_agent_server.protocol.schemas import FunctionName

        data = FunctionCallSSEData(
            function=cast(FunctionName, function),
            input_arguments=dict(input_arguments),
            extra_state={"call_id": call_id} if call_id else None,
        )
    else:
        data = FunctionCallSSEData(
            function="execute_agent_tool",
            input_arguments={
                "server_id": server_id,
                "name": function,
                "arguments": dict(input_arguments),
            },
            extra_state={"call_id": call_id} if call_id else None,
        )
    return FunctionCallSSE(data=data)


def _extract_text(content: Any) -> str:
    """Pull user-visible text out of any LangChain ``AIMessage.content`` shape."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    raw = block.get("text") or ""
                    if isinstance(raw, str):
                        parts.append(raw)
                elif btype is None and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "".join(parts)
    return str(content) if content else ""


_CITATION_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "its",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "has",
        "new",
        "you",
        "your",
        "out",
        "now",
    }
)

_CITATION_MATCH_THRESHOLD = 0.6


def _citation_tokens(text: str) -> set[str]:
    """Significant lower-case word tokens (length >= 3, no stopwords)."""
    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= 3 and w not in _CITATION_STOPWORDS
    }


def _citation_is_relevant(citation: dict[str, Any], answer_tokens: set[str]) -> bool:
    """Report whether ``citation`` is referenced by the final answer."""
    src = citation.get("source_info") or {}
    if str(src.get("type") or "") == "widget":
        return True
    title_tokens = _citation_tokens(str(src.get("name") or ""))
    if not title_tokens:
        return True
    overlap = len(title_tokens & answer_tokens) / len(title_tokens)
    return overlap >= _CITATION_MATCH_THRESHOLD


class DeepAgentEventAdapter:
    """Translate a deepagents event stream into OpenBB Workspace SSE events.

    A single instance is stateful and consumed once per run via
    :meth:`adapt`. As events arrive it splits inline ``<thinking>`` markers
    from prose, batches reasoning into status updates, maps tool calls to
    function-call events, and buffers artifacts and citations so they can be
    emitted (de-duplicated and relevance-filtered) at end of stream.
    """

    def __init__(self, *, client_tool_names: frozenset[str] = frozenset()) -> None:
        """Initialize the adapter with the set of client-side tool names.

        Parameters
        ----------
        client_tool_names : frozenset[str], optional
            Names of tools handled on the client. Membership decides whether
            a tool call is forwarded as a function-call event and how its
            server id is resolved. Defaults to an empty set.
        """
        self._client_tool_names = client_tool_names
        self._cur_id: str | None = None
        self._splitter: _ThinkingStreamSplitter | None = None
        self._reasoning_buf: list[str] = []
        self._prose_buf: list[str] = []
        self._artifacts: list[SSEEvent] = []
        self._citations: list[dict[str, Any]] = []
        self._citation_keys: set[tuple[str, str]] = set()
        self._answer_text_parts: list[str] = []

    def _emit_splits(self, splits: list[tuple[str, str]]) -> list[SSEEvent]:
        """Convert ``(channel, text)`` pairs into pending buffers."""
        for channel, text in splits:
            if channel == "close_unmatched":
                if self._prose_buf:
                    self._reasoning_buf.extend(self._prose_buf)
                    self._prose_buf = []
                continue
            if not text:
                continue
            if channel == "thinking":
                self._reasoning_buf.append(text)
                continue
            prose = _CITATION_MARKER_RE.sub("", text)
            if not prose:
                continue
            self._prose_buf.append(prose)
        return []

    def _flush_reasoning_segment(self) -> list[SSEEvent]:
        """Emit the current reasoning segment as one ``StatusUpdateSSE`` row."""
        reasoning_text = "".join(self._reasoning_buf).strip()
        prose_text = "".join(self._prose_buf).strip()
        self._reasoning_buf = []
        self._prose_buf = []
        merged = "\n\n".join(s for s in (reasoning_text, prose_text) if s)
        if not merged:
            return []
        return [
            StatusUpdateSSE(data=StatusUpdateSSEData(eventType="INFO", message=merged))
        ]

    def _flush_final_answer(self) -> list[SSEEvent]:
        """Emit buffered text at end-of-stream by channel."""
        out: list[SSEEvent] = []
        reasoning_text = "".join(self._reasoning_buf).strip()
        prose_text = "".join(self._prose_buf).strip()
        self._reasoning_buf = []
        self._prose_buf = []
        if reasoning_text:
            out.append(
                StatusUpdateSSE(
                    data=StatusUpdateSSEData(eventType="INFO", message=reasoning_text)
                )
            )
        if prose_text:
            out.append(MessageChunkSSE(data=MessageChunkSSEData(delta=prose_text)))
        return out

    def _drain_pending(self) -> list[SSEEvent]:
        """Flush splitter + buffered prose as the final answer."""
        out: list[SSEEvent] = []
        if self._splitter is not None:
            self._emit_splits(self._splitter.flush())
            self._splitter = None
        out.extend(self._flush_final_answer())
        self._cur_id = None
        return out

    async def adapt(
        self,
        stream: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[SSEEvent]:
        """Translate a deepagents event stream into Workspace SSE events.

        Consumes the source stream to exhaustion, yielding translated events
        as they are produced. After the source ends it drains any held-back
        prose, then the buffered artifacts, then a single relevance-filtered
        citation collection.

        Parameters
        ----------
        stream : AsyncIterator[dict[str, Any]]
            Async iterator of raw deepagents events. Each event is a mapping
            with ``type`` (``"messages"``, ``"updates"``, ``"custom"`` or
            ``"error"``), an optional ``ns`` namespace tuple, and ``data``.

        Yields
        ------
        SSEEvent
            Workspace server-sent events: message chunks, status updates,
            function calls, artifacts and citation collections.
        """
        async for ev in stream:
            for translated in self._translate(ev):
                self._note_for_citations(translated)
                yield translated
        for tail in self._drain_pending():
            self._note_for_citations(tail)
            yield tail
        for art in self._drain_artifacts():
            self._note_for_citations(art)
            yield art
        for ev in self._drain_citations():
            yield ev

    def _note_for_citations(self, ev: SSEEvent) -> None:
        """Record final-answer text so unreferenced citations can be filtered."""
        if isinstance(ev, MessageChunkSSE):
            self._answer_text_parts.append(ev.data.delta)
        elif isinstance(ev, MessageArtifactSSE):
            art = ev.data
            parts = [art.name or "", art.description or ""]
            content = art.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for row in content:
                    if isinstance(row, dict):
                        parts.extend(str(v) for v in row.values())
            self._answer_text_parts.append(" ".join(parts))

    def _drain_artifacts(self) -> list[SSEEvent]:
        """Emit every buffered artifact in arrival order, then reset."""
        out = self._artifacts
        trace(logger, "adapter: draining %d buffered artifact(s)", len(out))
        self._artifacts = []
        return out

    def _drain_citations(self) -> list[SSEEvent]:
        """Emit the referenced citations as one ``CitationCollectionSSE``."""
        if not self._citations:
            return []
        answer_tokens = _citation_tokens(" ".join(self._answer_text_parts))
        relevant = [
            c for c in self._citations if _citation_is_relevant(c, answer_tokens)
        ]
        dropped = len(self._citations) - len(relevant)
        self._citations = []
        self._citation_keys = set()
        if dropped:
            trace(
                logger,
                "adapter: dropped %d citation(s) unreferenced by the final answer",
                dropped,
            )
        if not relevant:
            return []
        citations = [Citation.model_validate(c) for c in relevant]
        return [CitationCollectionSSE(data=CitationCollection(citations=citations))]

    def _absorb_citations(self, items: Any) -> None:
        """Accumulate citations across the run; de-dup by stable identity."""
        if not isinstance(items, list):
            return
        for c in items:
            if not isinstance(c, dict):
                continue
            src = c.get("source_info") or {}
            src_type = str(src.get("type") or "")
            if src_type == "widget":
                widget_uuid = str(src.get("uuid") or "") or str(
                    src.get("widget_id") or ""
                )
                meta = src.get("metadata") or {}
                page = ""
                if isinstance(meta, dict):
                    page = str(meta.get("Page") or "")
                key: tuple[str, str] = (
                    "widget",
                    f"{widget_uuid}#p{page}" if page else widget_uuid,
                )
            else:
                url = str(src.get("origin") or "")
                details = c.get("details") or []
                snippet = ""
                if isinstance(details, list) and details:
                    first = details[0]
                    if isinstance(first, dict):
                        snippet = str(first.get("text") or "")
                key = (url, snippet)
            if key in self._citation_keys:
                continue
            self._citation_keys.add(key)
            self._citations.append(c)

    def _translate(self, ev: dict[str, Any]) -> list[SSEEvent]:
        kind = ev.get("type")
        ns = tuple(ev.get("ns") or ())
        data = ev.get("data") or {}

        if kind == "messages":
            return self._translate_messages(data, ns)
        if kind == "updates":
            return []
        if kind == "custom":
            custom = data or {}
            custom_type = custom.get("type")
            if custom_type == "citations":
                self._absorb_citations(custom.get("citations") or [])
                return []
            if custom_type == "artifact":
                artifact_payload = custom.get("artifact") or {}
                try:
                    sse = MessageArtifactSSE(data=_build_artifact(artifact_payload))
                except Exception:
                    logger.exception(
                        "adapter: artifact payload failed to build "
                        "(type=%r name=%r uuid=%r)",
                        artifact_payload.get("type"),
                        artifact_payload.get("name"),
                        artifact_payload.get("uuid"),
                    )
                    return []
                self._artifacts.append(sse)
                trace(
                    logger,
                    "adapter: buffered artifact (type=%r name=%r uuid=%r) — "
                    "total buffered=%d",
                    artifact_payload.get("type"),
                    artifact_payload.get("name"),
                    artifact_payload.get("uuid"),
                    len(self._artifacts),
                )
                return []
            if self._splitter is not None:
                self._emit_splits(self._splitter.flush())
                self._splitter = _ThinkingStreamSplitter()
            out = self._flush_reasoning_segment()
            out.extend(self._translate_custom(custom))
            return out
        if kind == "error":
            out = self._drain_pending()
            out.append(
                StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="ERROR",
                        message=str(data.get("message", "agent error")),
                        details=_coerce_status_details({"ns": list(ns), **data}),
                    )
                )
            )
            return out
        return []

    def _build_function_call_from_tool_call(
        self, tc: Any, ns: tuple[str, ...]
    ) -> SSEEvent | None:
        """Map one ``message.tool_calls[i]`` entry to a ``FunctionCallSSE``."""
        if not isinstance(tc, dict):
            return None
        raw_name = str(tc.get("name") or "")
        if not raw_name:
            return None
        args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
        call_id = str(tc.get("id") or "")
        if (
            raw_name in _WORKSPACE_NATIVE_FUNCTIONS
            and raw_name not in self._client_tool_names
        ):
            return None
        if raw_name.startswith(CLIENT_SIDE_TOOL_PREFIX):
            server_id = ns[0] if ns else "agent"
            function = raw_name[len(CLIENT_SIDE_TOOL_PREFIX) :]
        elif raw_name in self._client_tool_names:
            server_id = "agent"
            function = raw_name
        else:
            return None
        return _build_function_call(
            server_id=server_id,
            function=function,
            input_arguments=dict(args),
            call_id=call_id,
        )

    def _translate_messages(
        self,
        data: dict[str, Any],
        ns: tuple[str, ...],
    ) -> list[SSEEvent]:
        out: list[SSEEvent] = []
        message = data.get("message") or {}
        message_id = str(message.get("id") or "")
        text = _extract_text(message.get("content"))
        tool_calls = message.get("tool_calls") or []

        deferred_fn_calls: list[SSEEvent] = []
        for tc in tool_calls:
            fc = self._build_function_call_from_tool_call(tc, ns)
            if fc is not None:
                deferred_fn_calls.append(fc)
        has_tool_call = bool(tool_calls)

        if message_id and message_id != self._cur_id:
            if self._splitter is not None:
                self._emit_splits(self._splitter.flush())
            self._cur_id = message_id
            self._splitter = _ThinkingStreamSplitter()
        elif not message_id and self._splitter is None:
            self._splitter = _ThinkingStreamSplitter()

        kw = message.get("additional_kwargs") or {}
        reasoning_delta = _flatten_reasoning(
            kw.get("reasoning_content") or kw.get("reasoning") or ""
        )
        if reasoning_delta and not (text and _THINKING_CLOSE_RE.search(text)):
            self._reasoning_buf.append(reasoning_delta)

        if text and self._splitter is not None:
            self._emit_splits(self._splitter.feed(text))

        if has_tool_call:
            if self._splitter is not None:
                self._emit_splits(self._splitter.flush())
                self._splitter = _ThinkingStreamSplitter()
            out.extend(self._flush_reasoning_segment())
            if deferred_fn_calls:
                out.extend(deferred_fn_calls)
        return out

    def _translate_custom(self, data: dict[str, Any]) -> list[SSEEvent]:
        kind = data.get("type")
        if kind == "chunk":
            return [
                MessageChunkSSE(
                    data=MessageChunkSSEData(delta=str(data.get("content", "")))
                )
            ]
        if kind == "step":
            return [
                StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType=_coerce_status_event_type(data.get("event_type")),
                        message=str(data.get("message", "")),
                        details=_coerce_status_details(data.get("details")),
                    )
                )
            ]
        if kind == "artifact":
            return [
                MessageArtifactSSE(data=_build_artifact(data.get("artifact") or {}))
            ]
        if kind == "function_call":
            return [
                _build_function_call(
                    server_id=str(data.get("server_id", "agent")),
                    function=str(data["tool_name"]),
                    input_arguments=data.get("parameters") or {},
                    call_id=str(data.get("call_id", "")),
                )
            ]
        return [
            StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message=str(data.get("message", "")),
                    details=_coerce_status_details(data),
                )
            )
        ]
