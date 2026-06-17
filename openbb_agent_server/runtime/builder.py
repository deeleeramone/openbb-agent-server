"""Compose plugins into a DeepAgents agent and stream its events."""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from openbb_agent_server.app.settings import AgentProfile, AgentServerSettings
from openbb_agent_server.observability.logging import TRACE, trace
from openbb_agent_server.protocol.adapter import DeepAgentEventAdapter
from openbb_agent_server.protocol.schemas import (
    ChatMessage,
    QueryRequest,
    SSEEvent,
)
from openbb_agent_server.runtime import registry, services
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import (
    Middleware,
    ModelProvider,
    ToolSource,
)

logger = logging.getLogger("openbb_agent_server.builder")


SYSTEM_PROMPT_TEMPLATE = """\
You are the OpenBB Agent — an AI assistant embedded inside the OpenBB
Workspace, a financial-analysis dashboard.

Session
-------
- timezone: {timezone}
- today's date: {today}

`{today}` is the real current date and overrides your training cutoff.
A date before `{today}` is in the past; a date after it is in the
future. When asked for the "next" / "latest" / "most recent" / "upcoming"
event, compare every candidate date against `{today}` rather than
trusting a snippet's tense.
{widget_snapshot}{file_snapshot}

You do NOT receive the user's identity (email, display name,
user_id). Authentication is the server's job; the data you see has
already been partitioned for the calling user.

═══════════════════════════════════════════════════════════════════
CORE RULES (read these once and always follow them)
═══════════════════════════════════════════════════════════════════

1. ANSWER DIRECTLY when the user's request is a simple question,
   greeting, definition, or one-line factual ask. Do NOT call any
   tool — just respond in plain text. Examples:
   - "hi" → "Hello! How can I help?"
   - "what's 2+2?" → "4"
   - "Reply with: pong" → "pong"

2. USE A TOOL only when the answer requires data, computation, or
   side-effects that the tool provides. Pick the smallest useful
   tool for the job (a single call, not a chain).

3. NEVER retry a tool call that just errored with the same input.
   If a tool fails, either pick a different tool, change the
   arguments meaningfully, or tell the user what went wrong.

4. STOP when you have the answer. Once you've got enough info to
   reply, write the final answer and end the turn. Don't keep
   calling tools "to be thorough".

═══════════════════════════════════════════════════════════════════
TOOL CATALOGUE
═══════════════════════════════════════════════════════════════════

A) Planning / scratchpad — use ONLY for multi-step research that
   needs explicit tracking:
   - ``write_todos(todos)``  — sketch a plan when the task has
     ≥3 distinct steps. Skip for short tasks.

B) Virtual filesystem — internal scratch space, NOT a real disk.
   The user does not see files you write here. Use only when you
   need to stash intermediate work for a long-running task:
   - ``ls(path)``, ``read_file(path)``, ``write_file(path, text)``,
     ``edit_file(path, ...)``, ``glob(pattern)``, ``grep(pattern)``
   DO NOT use ``write_file`` to "answer" the user — write_file is
   for caching intermediate research, not for delivering replies.

C) Sub-agent delegation:
   - ``task(subagent_name, instruction)`` — delegate to a named
     specialist (researcher, analyst, charter, pdf_reader). Use
     when one specialised step is part of a larger task.

D) OpenBB Workspace artifacts — these are how the user actually
   sees rich content:
   - ``emit_markdown_artifact(content, name, description)``
   - ``emit_table_artifact(columns, rows, name, description)``
   - ``emit_chart_artifact(plotly, name, description)`` — Plotly JSON
   - ``emit_html_artifact(content, name, description)`` — sanitized HTML
   - ``emit_reasoning_step(message, event_type)`` — surface
     "thinking out loud" to the user without inlining it in the
     final answer.
   - ``cite_source(text, source, source_url)`` — attach a citation
     for any factual claim you retrieved from outside your training.

E) Workspace data access (CAREFUL — read the rules below before fetching):

   ``list_widgets()`` returns a compact index of every pinned widget
   with a ``widget_id``, a ``params_hash`` and a ``data_hash``. The
   Session block at the top of this prompt already lists this
   snapshot — you do NOT need to call ``list_widgets`` first if the
   snapshot is present and you can see what's there.

   ``get_widget_data(widget_id)`` returns the full ``params`` and
   ``data`` for one widget along with its ``params_hash``,
   ``data_hash``, ``fetched_at``, and a ``cache_hit`` flag.

   When to call ``get_widget_data``:
   - ONLY if the user's question requires the actual values inside
     the widget (a metric, a row, a column, a chart point) to answer.
   - DO NOT call it for generic questions, definitions, or anything
     you can answer from your own knowledge.
   - DO NOT call it twice with the same widget_id in one turn — the
     second call returns the same payload with ``cache_hit=true``.
     If you already see the data in the conversation, USE THAT.
   - Across turns: compare the widget's current ``data_hash`` to the
     hash you saw on a previous turn. If hashes match, the data did
     not change — reuse what you already wrote / quoted; do not
     re-fetch and do not re-emit the same artifact.
   - If the widget_id you need isn't in the snapshot, the user
     hasn't pinned that widget — say so, don't guess.

   ``list_pdfs()`` / ``pdf_extract(name, page_range)`` — same
   discipline: read only when the user's question is grounded in
   the PDF content; cite every claim with ``cite_source``.

F) Web research:
   - ``web_search(query, k)`` — current events, news, anything
     beyond your training cutoff. Cites every result automatically.

G) Per-user memory:
   - ``recall_user_memory(query, k)`` — retrieve durable facts
     this user has accumulated (preferences, watchlists, style).
     Use when context might exist from prior conversations.

H) Client-side actions (Workspace UI):
   - ``client:open_widget(widget_id)`` — focus a widget.
   - ``client:highlight_widget(widget_id)`` — visual highlight.
   - ``client:change_dashboard(dashboard_id)`` — switch dashboard.
   - ``client:add_widget_to_dashboard(widget_type, params)``
   When you call these, the host executes them in the user's
   browser and returns a result on the next turn.

═══════════════════════════════════════════════════════════════════
WORKFLOW
═══════════════════════════════════════════════════════════════════

For trivial asks: answer directly, no tools.

For data tasks:
1. If the user mentions a widget / dashboard / pinned item, call
   ``list_widgets`` then ``get_widget_data``.
2. If the user uploaded a PDF, use ``list_pdfs`` then
   ``pdf_extract`` with the relevant page range, and ``cite_source``
   every claim you draw from it.
3. For current events / external lookups, ``web_search``.
4. Render results: tables → ``emit_table_artifact``,
   charts → ``emit_chart_artifact``, long-form → ``emit_markdown_artifact``.
5. Keep the chat reply terse — the artifact is the deliverable.

For multi-step research:
1. ``write_todos`` to outline.
2. Execute steps; ``emit_reasoning_step`` for visible status.
3. ``task('researcher', ...)`` to delegate sub-questions.
4. Synthesise + emit a final artifact + cite.

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════
- Markdown is OK in chat replies.
- Numbers in tables, plots in charts, raw HTML only when the layout
  actually requires it.
- Always cite when retrieving from PDFs or the web.
"""


def _render_widget_snapshot(
    ctx: RunContext, in_store: frozenset[str] = frozenset()
) -> str:
    """Render the attached-widgets section in the system prompt."""
    if not ctx.widgets:
        return ""
    from openbb_agent_server.plugins.tools.widget_data import (
        _short_hash as _hash,
    )

    lines = [
        "",
        "Attached widgets (from the user's pinned dashboard)",
        "-" * 51,
        "Pass ``widget_uuid`` to read_widget_data / get_widget_data; "
        "use ``widget_id`` as the SQL table name; ``name`` is a label.",
    ]
    any_file_widget = False
    any_in_store = False
    any_pdf_widget = False
    for w in ctx.widgets:
        name = getattr(w, "name", None) or "(unnamed)"
        internal_id = getattr(w, "widget_id", "") or ""
        wid_for_kind_check = internal_id or w.uuid
        is_file = isinstance(wid_for_kind_check, str) and wid_for_kind_check.startswith(
            "file-"
        )
        extra = getattr(w, "model_extra", None) or {}
        columns = extra.get("columns") if isinstance(extra, dict) else None
        is_pdf_widget = False
        if isinstance(columns, list) and any(
            isinstance(c, str) and c.lower() in {"content", "data_format"}
            for c in columns
        ):
            is_pdf_widget = True
        if not is_pdf_widget:
            lower_id = (internal_id or "").lower()
            lower_name = (name or "").lower()
            if any(
                tok in lower_id or tok in lower_name
                for tok in ("document", "filing", "prospectus", "pdf")
            ):
                is_pdf_widget = True
        if is_file:
            any_file_widget = True
            line = (
                f"- widget_uuid={w.uuid!r} widget_id={internal_id!r} "
                f"name={name!r} KIND=user-uploaded-file UNREADABLE=true "
                f"(Workspace does NOT expose this file's bytes to the "
                f"agent; do NOT call get_widget_data on it)"
            )
        else:
            cached = w.uuid in in_store
            if cached:
                any_in_store = True
                data_state = "data_in_store=true"
            elif w.data is not None:
                data_state = f"data_hash={_hash(w.data)}"
            else:
                data_state = "data=<not loaded>"
            kind_tag = " KIND=pdf-document" if is_pdf_widget else ""
            if is_pdf_widget:
                any_pdf_widget = True
            line = (
                f"- widget_uuid={w.uuid!r} widget_id={internal_id!r} "
                f"name={name!r}{kind_tag} params={dict(w.params)} "
                f"params_hash={_hash(dict(w.params))} "
                f"{data_state}"
            )
        description = getattr(w, "description", None)
        if description:
            line += f"\n  description: {description}"
        lines.append(line)
    if any_in_store:
        lines.append(
            "``data_in_store=true`` → read via read_widget_data("
            "widget_uuid=…); do NOT call get_widget_data on these."
        )
    lines.append(
        "Otherwise → get_widget_data(widget_ids=[…]) in ONE call with "
        "every needed widget_uuid; rows arrive next turn."
    )
    if any_pdf_widget:
        lines.append(
            "``KIND=pdf-document`` → these expose PDF bytes, not rows. "
            "Fetch them with ``get_widget_data(widget_ids=[<uuid>])`` "
            "EXACTLY like a normal widget — the PDF arrives in the "
            "next turn's tool message and the server auto-promotes it "
            "into uploaded_files. THEN use list_pdfs / get_pdf_outline "
            "/ pdf_extract. Do NOT skip the get_widget_data step and "
            "do NOT try read_widget_data on these."
        )
    if any_file_widget:
        lines.append(
            "``UNREADABLE=true`` widgets are user uploads — Workspace "
            "cannot serve their bytes via get_widget_data. If the "
            "question needs that content, ask the user to paste a sample."
        )
    lines.append("")
    return "\n".join(lines)


def _render_file_snapshot(ctx: RunContext) -> str:
    """Render the uploaded-files section in the system prompt."""
    if not ctx.uploaded_files:
        return ""
    lines = ["", "Uploaded files", "-" * 14]
    for f in ctx.uploaded_files:
        lines.append(f"- name={f.name!r} mime={f.mime}")
    lines.append(
        "Use pdf_extract(name, page_range) on PDFs only if the user's "
        "question is grounded in the document; cite_source every claim."
    )
    lines.append("")
    return "\n".join(lines)


async def _build_turn_addendum(
    ctx: RunContext,
    body: QueryRequest,
    has_ingested: bool,
) -> str:
    """Build a per-turn instruction block appended to the system prompt."""
    if not has_ingested:
        return ""

    from openbb_agent_server.runtime import services

    store = services.get_widget_store()
    if store is None:
        return ""

    try:
        entries = await store.schema(
            principal=ctx.principal,
            conversation_id=None,
        )
    except Exception:
        return ""

    if not entries:
        return ""

    lines: list[str] = []
    for entry in entries:
        table = str(entry.get("table") or "")
        cols = entry.get("columns") or []
        rows = entry.get("row_count") or 0
        cols_str = ", ".join(f'"{c}"' for c in cols)
        lines.append(f'  - "{table}" (rows={rows}): {cols_str}')

    return (
        "Local widget_data tables (table = widget_id slug, columns are "
        'TEXT — cast with CAST("col" AS REAL) for arithmetic):\n'
        + "\n".join(lines)
        + '\n\nRead them via ``read_widget_data(widget_uuid="<uuid>")`` '
        "or ``query_widget_data(sql=…)``. For widgets not listed, "
        "fetch with ``get_widget_data(widget_ids=[…])`` in ONE call."
    )


def _build_system_prompt(
    ctx: RunContext, in_store: frozenset[str] = frozenset()
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        timezone=ctx.timezone or "UTC",
        today=_dt.date.today().isoformat(),
        widget_snapshot=_render_widget_snapshot(ctx, in_store),
        file_snapshot=_render_file_snapshot(ctx),
    )


def _load_system_prompt(
    ctx: RunContext,
    profile: AgentProfile,
    in_store: frozenset[str] = frozenset(),
) -> str:
    """Load the prompt file (or the bundled default) and substitute context."""
    from string import Formatter

    from openbb_agent_server.prompts import default_system_prompt_path

    path_str = profile.system_prompt_file
    path = Path(path_str).expanduser() if path_str else default_system_prompt_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "system_prompt_file %s could not be read (%s); falling back to "
            "the in-memory template",
            path,
            exc,
        )
        return _build_system_prompt(ctx, in_store)

    fields = {
        "timezone": ctx.timezone or "UTC",
        "today": _dt.date.today().isoformat(),
        "widget_snapshot": _render_widget_snapshot(ctx, in_store),
        "file_snapshot": _render_file_snapshot(ctx),
    }

    out: list[str] = []
    for literal, name, spec, conversion in Formatter().parse(raw):
        out.append(literal)
        if name is None:
            continue
        if name in fields:
            value = fields[name]
            if conversion or spec:
                out.append(format(value, spec or ""))
            else:
                out.append(str(value))
        else:
            placeholder = "{" + name
            if conversion:
                placeholder += "!" + conversion
            if spec:
                placeholder += ":" + spec
            placeholder += "}"
            out.append(placeholder)
    return "".join(out)


def _json_loads(value: str) -> Any:
    import json as _json

    return _json.loads(value)


def _parse_function_call_envelope(content: Any) -> dict[str, Any] | None:
    """Return the parsed function-call envelope, or None."""
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped.startswith("{") or '"function"' not in stripped:
        return None
    try:
        parsed = _json_loads(stripped)
    except Exception:
        return None
    if (
        isinstance(parsed, dict)
        and "function" in parsed
        and "input_arguments" in parsed
    ):
        return parsed
    return None


def _tool_message_content(data: Any) -> str:
    """Flatten ``LlmClientFunctionCallResultMessage.data`` into a string."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if not isinstance(data, list):
        return str(data)
    chunks: list[str] = []
    for entry in data:
        chunks.extend(_tool_message_entry_chunks(entry))
    return "\n".join(chunks)


def _tool_message_entry_chunks(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return [str(entry)]
    items = entry.get("items")
    if isinstance(items, list):
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                body = item.get("content")
                if body is not None:
                    out.append(body if isinstance(body, str) else str(body))
            elif item is not None:
                out.append(str(item))
        return out
    inner = entry.get("content")
    if inner is not None:
        return [inner if isinstance(inner, str) else str(inner)]
    if entry.get("data") is not None:
        return [str(entry.get("data"))]
    try:
        import json as _json

        return [_json.dumps(entry, default=str)]
    except (TypeError, ValueError):
        return [str(entry)]


def _to_lc_messages(messages: list[ChatMessage]) -> list[Any]:
    """Convert wire-protocol messages to LangChain message objects."""
    import json as _json

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    def _coerce_text(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        try:
            return _json.dumps(raw, default=str)
        except (TypeError, ValueError):
            return str(raw)

    def _call_id_of(msg: ChatMessage) -> str:
        extra = getattr(msg, "extra_state", None) or {}
        return (
            msg.tool_call_id
            or (extra.get("call_id") if isinstance(extra, dict) else None)
            or ""
        )

    out: list[Any] = []
    for m in messages:
        if m.role == "human":
            out.append(HumanMessage(content=_coerce_text(m.content)))
            continue

        if m.role == "ai":
            envelope = _parse_function_call_envelope(m.content)
            if envelope is not None:
                continue
            fn_name = m.function or ""
            if fn_name:
                tool_call_id = _call_id_of(m) or f"call-{len(out)}"
                out.append(
                    AIMessage(
                        content=_coerce_text(m.content),
                        tool_calls=[
                            {
                                "name": fn_name,
                                "args": dict(m.input_arguments or {}),
                                "id": tool_call_id,
                            }
                        ],
                    )
                )
                continue
            text = _coerce_text(m.content)
            if text.strip():
                out.append(AIMessage(content=text))
            continue

        if m.role == "tool":
            tool_call_id = _call_id_of(m) or f"call-{len(out)}"
            tool_name = m.function or "tool"
            if tool_name == "get_widget_data":
                fetched_ids: list[str] = []
                pdf_ids: list[str] = []
                tabular_ids: list[str] = []
                for src in (m.input_arguments or {}).get("data_sources") or []:
                    if not isinstance(src, dict):
                        continue
                    wid = str(src.get("widget_uuid") or src.get("id") or "")
                    if not wid:
                        continue
                    fetched_ids.append(wid)
                    src_id = str(src.get("id") or "").lower()
                    if any(
                        tok in src_id
                        for tok in ("document", "filing", "prospectus", "pdf")
                    ):
                        pdf_ids.append(wid)
                    else:
                        tabular_ids.append(wid)
                msg_lines: list[str] = [
                    "Widget data has been fetched and stored locally "
                    f"(widget_ids={fetched_ids}). Do NOT re-fetch via "
                    "get_widget_data."
                ]
                if tabular_ids:
                    msg_lines.append(
                        f"Tabular widgets {tabular_ids}: read with "
                        "``read_widget_data(widget_uuid=...)`` or "
                        "``query_widget_data(sql=...)``."
                    )
                if pdf_ids:
                    msg_lines.append(
                        f"PDF / document widgets {pdf_ids}: their PDFs "
                        "are now in ``uploaded_files``. Use "
                        "``list_pdfs()`` → ``get_pdf_outline(name=...)`` "
                        "→ ``pdf_extract(name=..., page_range=...)`` or "
                        "``search_pdf(query=...)``. "
                        "``read_widget_data`` will return None for "
                        "these — DON'T call it."
                    )
                content = "\n".join(msg_lines)
            else:
                payload: Any = m.data if m.data is not None else m.content
                content = payload if isinstance(payload, str) else _coerce_text(payload)
            need_synth = not any(
                isinstance(prev, AIMessage)
                and any(tc.get("id") == tool_call_id for tc in (prev.tool_calls or []))
                for prev in out
            )
            if need_synth:
                out.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": tool_name,
                                "args": dict(m.input_arguments or {}),
                                "id": tool_call_id,
                            }
                        ],
                    )
                )
            out.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
            continue
    return out


async def _resolve_tools(
    ctx: RunContext,
    profile: AgentProfile,
) -> tuple[list[Any], frozenset[str]]:
    """Return ``(tools, client_tool_names)`` for the run."""
    from openbb_agent_server.protocol.adapter import (
        CLIENT_SIDE_TOOL_PREFIX,
        WORKSPACE_MCP_TOOL_PREFIX,
    )

    tools: list[Any] = []
    client_tools: set[str] = set()
    for name in profile.tool_sources:
        source_cfg = profile.tool_source_config.get(name, {})
        source: ToolSource = registry.load(
            "openbb_agent_server.tools", name, source_cfg
        )
        for t in await source.tools(ctx, source_cfg):
            tools.append(t)
            tool_name = getattr(t, "name", "")
            if tool_name.startswith(CLIENT_SIDE_TOOL_PREFIX) or tool_name.startswith(
                WORKSPACE_MCP_TOOL_PREFIX
            ):
                client_tools.add(tool_name)
    return tools, frozenset(client_tools)


def _resolve_subagents(
    profile: AgentProfile,
    main_tools: list[Any],
) -> list[dict[str, Any]]:
    """Resolve subagent specs against the chosen profile."""
    by_name = {getattr(t, "name", ""): t for t in main_tools}
    out: list[dict[str, Any]] = []
    for name in profile.subagents:
        spec = registry.load("openbb_agent_server.subagents", name)
        entry: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "system_prompt": spec.system_prompt,
        }
        wanted = tuple(spec.tools or ())
        if wanted:
            resolved = [by_name[n] for n in wanted if n in by_name]
            if resolved:
                entry["tools"] = resolved
        if spec.model:
            entry["model"] = spec.model
        out.append(entry)
    return out


def _resolve_middleware(
    ctx: RunContext,
    profile: AgentProfile,
    model: Any,
) -> list[Any]:
    out: list[Any] = []
    for name in profile.middleware:
        mw: Middleware = registry.load("openbb_agent_server.middleware", name)
        out.append(mw.build(ctx, {"model": model}))
    return out


async def run_agent(
    *,
    ctx: RunContext,
    body: QueryRequest,
    settings: AgentServerSettings,
    profile: AgentProfile | None = None,
) -> AsyncIterator[SSEEvent]:
    """Run the DeepAgents loop and yield OpenBB SSE events.

    Resolves the agent profile, then assembles its model provider, tools,
    subagents, and middleware from the plugin registry. Inspects any
    stored or in-turn widget data: when data has been ingested the tool
    set is narrowed to data-reading and artifact-emitting tools so the
    agent works from local tables instead of re-fetching. Builds the
    system prompt (optionally prepending a per-turn table addendum),
    creates the deep agent with a checkpointer keyed by user, profile, and
    trace, converts the wire-protocol messages to LangChain messages, and
    streams the agent's output through :class:`~openbb_agent_server.protocol.adapter.DeepAgentEventAdapter` as
    OpenBB SSE events.

    Parameters
    ----------
    ctx : RunContext
        Per-run context (principal, trace/run/conversation ids, timezone,
        attached widgets, and uploaded files).
    body : QueryRequest
        Incoming request whose ``messages`` drive the turn.
    settings : AgentServerSettings
        Server settings used to resolve the profile when one is not given.
    profile : AgentProfile or None, optional
        Pre-resolved profile. When ``None`` it is resolved from
        ``ctx.agent_name`` via ``settings``.

    Yields
    ------
    SSEEvent
        OpenBB server-sent events adapted from the DeepAgents stream.
    """
    if profile is None:
        profile = settings.resolve_profile(ctx.agent_name)

    logger.debug(
        "agent run started",
        extra={
            "trace_id": ctx.trace_id,
            "run_id": ctx.run_id,
            "conversation_id": ctx.conversation_id,
            "agent_name": profile.name,
        },
    )

    from deepagents import create_deep_agent

    model_provider: ModelProvider = registry.load(
        "openbb_agent_server.models",
        profile.model_provider,
        {"model_name": profile.model_name, **profile.model_config_},
    )
    model = model_provider.build(ctx, {})

    tools, client_tool_names = await _resolve_tools(ctx, profile)

    in_store: frozenset[str] = frozenset()
    has_any_user_widget_data = False
    widget_store = services.get_widget_store()
    if widget_store is not None:
        try:
            entries = await asyncio.wait_for(
                widget_store.list_entries(
                    principal=ctx.principal,
                    conversation_id=None,
                ),
                timeout=2.0,
            )
            has_any_user_widget_data = bool(entries)
            stored_uuids = {
                str(e.get("widget_uuid") or "") for e in entries if e.get("widget_uuid")
            }
            attached_uuids = {w.uuid for w in ctx.widgets if w.uuid}
            in_store = frozenset(stored_uuids & attached_uuids)
        except asyncio.TimeoutError:
            logger.warning(
                "in_store lookup timed out after 2s — agent will run "
                "without the data-in-store hint",
            )
        except Exception:
            logger.warning(
                "in_store lookup failed; continuing without it",
                exc_info=True,
            )

    has_tool_msg_this_turn = any(
        getattr(m, "role", None) == "tool" for m in body.messages
    )
    has_ingested = has_tool_msg_this_turn or has_any_user_widget_data
    if has_ingested:
        keep = {
            "list_widget_data",
            "read_widget_data",
            "search_widget_data",
            "describe_widget_data",
            "query_widget_data",
            "get_widget_data",
            "list_widgets",
            "list_pdfs",
            "search_pdf",
            "get_pdf_outline",
            "pdf_extract",
            "list_images",
            "understand_image",
            "caption_image",
            "read_image_text",
            "ask_about_image",
            "list_audio",
            "transcribe_audio",
            "web_search",
            "fetch_url",
            "recall_user_memory",
            "task",
            "emit_table_artifact",
            "emit_chart_artifact",
            "emit_markdown_artifact",
            "emit_html_artifact",
            "emit_reasoning_step",
            "cite_source",
        }
        before = [getattr(t, "name", None) for t in tools]
        tools = [t for t in tools if getattr(t, "name", None) in keep]
        after = [getattr(t, "name", None) for t in tools]
        removed = [n for n in before if n not in after]
        logger.debug(
            "tools: narrowed to data-reading + emitter tools | kept=%s removed=%s",
            after,
            removed,
        )

    subagents = _resolve_subagents(profile, tools)
    middleware = _resolve_middleware(ctx, profile, model)

    from langchain_core.runnables import RunnableConfig

    checkpointer = services.get_checkpointer()
    thread_id = f"{ctx.principal.user_id}:{profile.name}:{ctx.trace_id}"
    run_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": f"{ctx.principal.user_id}:{profile.name}",
            "user_id": ctx.principal.user_id,
            "agent": profile.name,
            "trace_id": ctx.trace_id,
            "run_id": ctx.run_id,
            "conversation_id": ctx.conversation_id,
        }
    }

    system_prompt = _load_system_prompt(ctx, profile, in_store)
    addendum = await _build_turn_addendum(ctx, body, has_ingested)
    if addendum:
        system_prompt = f"{addendum}\n\n---\n\n{system_prompt}"
        logger.debug(
            "system prompt: prepended %d-char turn addendum",
            len(addendum),
        )
    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": system_prompt,
        "subagents": subagents,
        "middleware": middleware,
        "checkpointer": checkpointer,
    }
    if profile.skills:
        agent_kwargs["skills"] = list(profile.skills)
    agent = create_deep_agent(**agent_kwargs)

    adapter = DeepAgentEventAdapter(client_tool_names=client_tool_names)
    lc_messages = _to_lc_messages(body.messages)

    last_role = body.messages[-1].role if body.messages else None
    logger.debug(
        "agent.astream entry | thread_id=%s last_role=%s msgs=%d lc_msgs=%d",
        thread_id,
        last_role,
        len(body.messages),
        len(lc_messages),
    )

    async def deepagents_events() -> AsyncIterator[dict[str, Any]]:
        n_raw = 0
        n_emit = 0
        trace(logger, "astream: iterator entry | thread_id=%s", thread_id)
        try:
            async for raw in agent.astream(
                {"messages": lc_messages},
                config=run_config,
                stream_mode=["messages", "custom"],
                subgraphs=True,
            ):
                n_raw += 1
                if logger.isEnabledFor(TRACE):
                    try:
                        if isinstance(raw, tuple) and len(raw) == 3:
                            mode, ns, payload = raw
                            if mode == "messages":
                                msg, _meta = (
                                    payload
                                    if isinstance(payload, tuple)
                                    else (payload, {})
                                )
                                content_preview = getattr(msg, "content", "")
                                kw = getattr(msg, "additional_kwargs", {}) or {}
                                rc = kw.get("reasoning_content") or ""
                                tc = (
                                    getattr(msg, "tool_call_chunks", None)
                                    or getattr(msg, "tool_calls", None)
                                    or []
                                )
                                trace(
                                    logger,
                                    "llm chunk #%d | content=%r reasoning=%r tool_calls=%r",
                                    n_raw,
                                    content_preview
                                    if isinstance(content_preview, str)
                                    else str(content_preview)[:200],
                                    rc if isinstance(rc, str) else str(rc)[:200],
                                    tc,
                                )
                            else:
                                trace(
                                    logger,
                                    "astream raw #%d | mode=%s payload=%r",
                                    n_raw,
                                    mode,
                                    repr(payload)[:300],
                                )
                        else:
                            trace(
                                logger,
                                "astream raw #%d | %r",
                                n_raw,
                                repr(raw)[:300],
                            )
                    except Exception:
                        logger.exception("astream chunk log failed (non-fatal)")
                normalised = _normalise_stream_event(raw)
                if normalised is not None:
                    n_emit += 1
                    yield normalised
        except Exception:
            logger.exception("astream: iterator raised")
            raise
        finally:
            trace(
                logger,
                "astream: iterator exit (raw=%d emit=%d)",
                n_raw,
                n_emit,
            )

    async for sse in adapter.adapt(deepagents_events()):
        yield sse


def _normalise_message_payload(
    ns: tuple[str, ...] | list[str], payload: Any
) -> dict[str, Any] | None:
    """Normalise a ``messages``-mode payload from LangGraph astream."""
    if not (isinstance(payload, tuple) and len(payload) >= 1):
        return None
    message = payload[0]
    from langchain_core.messages import AIMessage, AIMessageChunk

    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return None
    raw_content = getattr(message, "content", None)
    if raw_content is None:
        raw_content = ""
    extra_kw = dict(getattr(message, "additional_kwargs", {}) or {})
    return {
        "type": "messages",
        "ns": list(ns),
        "data": {
            "message": {
                "id": str(getattr(message, "id", "") or ""),
                "content": raw_content,
                "tool_calls": list(getattr(message, "tool_calls", []) or []),
                "tool_call_chunks": list(
                    getattr(message, "tool_call_chunks", []) or []
                ),
                "additional_kwargs": extra_kw,
            }
        },
    }


def _normalise_stream_event(raw: Any) -> dict[str, Any] | None:
    """Coerce one LangGraph ``astream`` item into our ``{type, ns, data}`` shape."""
    if not isinstance(raw, tuple) or len(raw) != 3:
        return None
    ns, mode, payload = raw

    if mode == "messages":
        return _normalise_message_payload(ns, payload)

    if mode == "updates":
        return {"type": "updates", "ns": list(ns), "data": payload or {}}

    if mode == "custom":
        if isinstance(payload, dict):
            return {"type": "custom", "ns": list(ns), "data": payload}
        return {"type": "custom", "ns": list(ns), "data": {"message": str(payload)}}

    return None
