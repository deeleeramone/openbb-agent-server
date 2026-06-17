"""``artifacts`` tool source — server-side artifact emission."""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, BeforeValidator, Field

from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.artifacts")


_SCRATCHPAD_HEADERS = (
    "session intent",
    "tool activity",
    "tools used",
    "tools called",
    "steps taken",
    "actions taken",
    "methodology",
    "method used",
    "next steps",
    "what i did",
    "what we did",
    "process followed",
    "tool activity performed so far",
    "sources",
    "citations",
    "references",
    "source list",
)

_HEADING_RE = re.compile(
    r"""^(?:
        \#{1,6}\s+(?P<atx>.+?)\s*$
        | \*{1,3}(?P<bold>[^*\n]+?)\*{1,3}\s*:?\s*$
        | (?P<plain>[A-Z][A-Za-z0-9 /&\-]{2,40}):\s*$
    )""",
    re.VERBOSE | re.MULTILINE,
)

_TOOL_CALL_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+\.)\s+`?\\?(?:list_pdfs|get_pdf_outline|pdf_extract|search_pdf|"
    r"read_widget_data|get_widget_data|search_widget_data|describe_widget_data|"
    r"query_widget_data|web_search|read_file|write_file|understand_image|"
    r"caption_image|read_image_text|ask_about_image|transcribe_audio|"
    r"recall_user_memory|cite_source|emit_\w+)\s*\([^)]*\)`?.*$",
    re.MULTILINE,
)


def _sanitise_markdown_body(content: str) -> tuple[str, list[str]]:
    """Strip scratchpad sections from a markdown body."""
    notes: list[str] = []
    matches = list(_HEADING_RE.finditer(content))
    if matches:
        keep: list[tuple[int, int]] = []
        cursor = 0
        for idx, m in enumerate(matches):
            label = (
                m.group("atx") or m.group("bold") or m.group("plain") or ""
            ).strip()
            label_norm = re.sub(r"[^a-z ]", "", label.lower()).strip()
            if any(h in label_norm for h in _SCRATCHPAD_HEADERS):
                if cursor < m.start():
                    keep.append((cursor, m.start()))
                next_start = (
                    matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
                )
                cursor = next_start
                notes.append(f"removed `{label}` section")
        if cursor < len(content):
            keep.append((cursor, len(content)))
        if notes:
            content = "".join(content[a:b] for a, b in keep)

    cleaned, n_lines = _TOOL_CALL_LINE_RE.subn("", content)
    if n_lines:
        notes.append(f"removed {n_lines} inline tool-call line(s)")
        content = cleaned

    if notes:
        content = re.sub(r"\n{3,}", "\n\n", content).strip() + "\n"
    return content, notes


def _decode_if_string(value: Any) -> Any:
    """Decode a JSON-encoded string into a Python list/dict, else passthrough."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


_LooseList = Annotated[list, BeforeValidator(_decode_if_string)]
_LooseListOfList = Annotated[list[list[Any]], BeforeValidator(_decode_if_string)]
_LooseDict = Annotated[dict[str, Any], BeforeValidator(_decode_if_string)]


class _HtmlArgs(BaseModel):
    content: str = Field(description="Sanitised HTML markup to render.")
    name: str = Field(default="", description="Short artifact title.")
    description: str = Field(default="", description="Brief description.")


class _MarkdownArgs(BaseModel):
    content: str = Field(description="Markdown-formatted body.")
    name: str = Field(default="", description="Short artifact title.")
    description: str = Field(default="", description="Brief description.")


class _TableArgs(BaseModel):
    columns: _LooseList = Field(description="Column headers, in order.")
    rows: _LooseListOfList = Field(description="Rows of cell values.")
    name: str = Field(default="", description="Short table title.")
    description: str = Field(default="", description="Brief description.")


class _ChartArgs(BaseModel):
    plotly: _LooseDict = Field(description="Plotly figure JSON.")
    name: str = Field(default="", description="Short chart title.")
    description: str = Field(default="", description="Brief description.")


class _ReasoningArgs(BaseModel):
    message: str = Field(description="Status message shown to the user.")
    event_type: str = Field(
        default="INFO", description="One of INFO, SUCCESS, WARNING, ERROR."
    )


class _CiteArgs(BaseModel):
    text: str | None = None
    source: str | None = None
    source_url: str | None = None


def _sanitise_warning(notes: list[str]) -> str:
    """Build the warning string returned alongside a successful emit."""
    joined = "; ".join(notes)
    return (
        f"WARNING: {joined}. Markdown/HTML artifact bodies are USER-"
        "FACING analysis — they must not contain 'Tool activity', "
        "'Steps taken', 'Next steps', 'Session intent', or any list of "
        "tool calls with arguments. Rewrite future artifacts as polished "
        "prose without those sections."
    )


class ArtifactsToolSource(ToolSource):
    """Bundle the artifact, citation, and reasoning emission tools.

    Expose the ``emit_*_artifact``, ``emit_reasoning_step``, and
    ``cite_source`` tools that let the agent render user-facing cards and
    queue citations on the active run.
    """

    name = "artifacts"

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Return the artifact-emission tools for this run.

        Build closures over :mod:`openbb_agent_server.runtime.emit` for
        HTML, markdown, table, chart, reasoning-step, and citation
        emission, wrap each as a ``StructuredTool`` with its argument
        schema and guidance description, and return them as a list. The
        markdown and HTML bodies are sanitised of scratchpad sections and
        inline tool-call lines before emission.

        Parameters
        ----------
        ctx : RunContext
            Active run context (unused here but part of the tool-source
            interface).
        config : dict of str to Any
            Per-run configuration (unused here but part of the interface).

        Returns
        -------
        list of Any
            The ``StructuredTool`` instances exposed to the agent.
        """

        def _success_message(kind_label: str) -> str:
            """Build the tool-result text shown after a successful emit."""
            return (
                f"OK. {kind_label} artifact rendered. "
                "The user can see it now. Your next response is the "
                "FINAL textual answer: a concise 1–2 sentence "
                "takeaway pointing at this artifact with the key "
                'insight. Refer to it in plain words ("the table '
                'below") — never by any id. STOP. Do NOT call any '
                "more artifact emitters unless the user EXPLICITLY "
                "asked for one artifact per dimension / category — "
                "one summary artifact answers a summary question."
            )

        def emit_html(content: str, name: str = "", description: str = "") -> str:
            cleaned, notes = _sanitise_markdown_body(content)
            emit.html_artifact(content=cleaned, name=name, description=description)
            msg = _success_message("HTML")
            if notes:
                msg += " " + _sanitise_warning(notes)
            return msg

        def emit_markdown(content: str, name: str = "", description: str = "") -> str:
            cleaned, notes = _sanitise_markdown_body(content)
            emit.markdown_artifact(content=cleaned, name=name, description=description)
            msg = _success_message("Markdown")
            if notes:
                msg += " " + _sanitise_warning(notes)
            return msg

        def emit_table(
            columns: list[str],
            rows: list[list[Any]],
            name: str = "",
            description: str = "",
        ) -> str:
            emit.table_artifact(
                columns=columns, rows=rows, name=name, description=description
            )
            return _success_message("Table")

        def emit_chart(
            plotly: dict[str, Any], name: str = "", description: str = ""
        ) -> str:
            emit.chart_artifact(plotly=plotly, name=name, description=description)
            return _success_message("Chart")

        def emit_reasoning_step(message: str, event_type: str = "INFO") -> str:
            """Emit one reasoning step."""
            emit.reasoning_step(message, event_type=event_type)
            return f"OK. Reasoning step '{message}' shown to the user."

        def cite_source(
            text: str | None = None,
            source: str | None = None,
            source_url: str | None = None,
        ) -> str:
            """Attach one citation to the run."""
            emit.cite(text=text, source=source, source_url=source_url)
            return "OK. Citation queued for the end-of-turn flush."

        return [
            StructuredTool.from_function(
                emit_html,
                name="emit_html_artifact",
                description=(
                    "Emit an HTML artifact for the user to view. Use sparingly — "
                    "raw HTML is sanitised by the Workspace UI (no scripts, no "
                    "iframes). The card renders automatically; refer to it in "
                    "plain words, never by id."
                ),
                args_schema=_HtmlArgs,
            ),
            StructuredTool.from_function(
                emit_markdown,
                name="emit_markdown_artifact",
                description=(
                    "Emit a polished, user-facing markdown analysis. "
                    "The body must read like a deliverable, not a "
                    "scratchpad. FORBIDDEN content (the server strips "
                    "these on emit): 'Session Intent' / 'Tool activity' "
                    "/ 'Tools used' / 'Steps taken' / 'Methodology' / "
                    "'Next steps' headings; 'Sources' / 'Citations' / "
                    "'References' sections — Workspace renders real "
                    "citations as clickable chips on every "
                    "``cite_source`` / PDF auto-citation, so inline "
                    "source lists are duplicate noise and lose the "
                    "click-to-navigate affordance; numbered/bulleted "
                    "lists of tool calls with arguments (e.g. "
                    "`pdf_extract(name=..., page_range=[..])`); "
                    "backticked internal filenames like "
                    "`blk_drill_fund_documents-...pdf` — say 'the "
                    "prospectus' instead. Write only the analysis the "
                    "user asked for."
                ),
                args_schema=_MarkdownArgs,
            ),
            StructuredTool.from_function(
                emit_table,
                name="emit_table_artifact",
                description=(
                    "Emit a tabular artifact. Prefer this over inlining large "
                    "tables in the chat reply."
                ),
                args_schema=_TableArgs,
            ),
            StructuredTool.from_function(
                emit_chart,
                name="emit_chart_artifact",
                description=(
                    "Emit a chart artifact (Plotly figure JSON). Pick line for "
                    "time series, bar for categorical, scatter for two numerics."
                ),
                args_schema=_ChartArgs,
            ),
            StructuredTool.from_function(
                emit_reasoning_step,
                name="emit_reasoning_step",
                description=(
                    "Show one progress / reasoning step to the user in the "
                    "step-by-step lane. ``event_type`` ∈ INFO / WARNING / "
                    "ERROR (SUCCESS is reserved for the run terminator)."
                ),
                args_schema=_ReasoningArgs,
            ),
            StructuredTool.from_function(
                cite_source,
                name="cite_source",
                description=(
                    "Attach a citation (text snippet + source + URL) to "
                    "the run. Buffered and emitted as a single batch at "
                    "end-of-turn so they decorate the final-answer bubble."
                ),
                args_schema=_CiteArgs,
            ),
        ]
