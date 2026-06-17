"""``widget_data`` tool source."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from openbb_agent_server.runtime import (
    context as run_context,
    emit,
)
from openbb_agent_server.runtime.context import RunContext, WidgetRef
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.widget_data")


class _ListArgs(BaseModel):
    pass


class _GetWidgetArgs(BaseModel):
    widget_ids: list[str] = Field(
        description=(
            "List of per-instance widget uuids (the ``widget_id`` "
            "values from the attached-widgets snapshot). Pass every "
            "widget you need in one call — the runtime emits a single "
            "batched fetch request to Workspace."
        ),
        min_length=1,
    )

    @field_validator("widget_ids", mode="before")
    @classmethod
    def _coerce_widget_ids(cls, value: Any) -> Any:
        """Accept a JSON-encoded list or comma-string and normalise to a list."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
            return [p.strip() for p in stripped.split(",") if p.strip()]
        if isinstance(value, (list, tuple)):
            return list(value)
        return value


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _short_hash(value: Any) -> str:
    return hashlib.blake2s(
        _stable_json(value).encode("utf-8"), digest_size=8
    ).hexdigest()


def _summarise(w: WidgetRef) -> dict[str, Any]:
    out: dict[str, Any] = {
        "widget_id": w.uuid,
        "params_hash": _short_hash(dict(w.params)),
        "data_hash": _short_hash(w.data),
    }
    name = getattr(w, "name", None)
    if name:
        out["name"] = name
    description = getattr(w, "description", None)
    if description:
        out["description"] = description
    return out


class WidgetDataToolSource(ToolSource):
    """Provide the agent with widget listing and data-fetch tools.

    A :class:`~openbb_agent_server.runtime.plugins.ToolSource` plugin that surfaces two LangChain tools backed by
    the widgets attached to the current run: ``list_widgets`` for
    discovery and ``get_widget_data`` for dispatching a batched fetch to
    Workspace.
    """

    name = "widget_data"

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Build the ``list_widgets`` and ``get_widget_data`` tools.

        Parameters
        ----------
        ctx : RunContext
            Run context for the active session. The tool callables read the
            live attached widgets via the runtime context rather than from
            this argument.
        config : dict[str, Any]
            Per-source configuration block (unused by this source).

        Returns
        -------
        list[Any]
            Two :class:`StructuredTool` instances:
            ``list_widgets``, which returns a hash-only summary of every
            attached widget, and ``get_widget_data``, which emits a single
            batched fetch request to Workspace and returns directly.
        """

        def list_widgets() -> dict[str, Any]:
            current = run_context.current()
            widgets = [_summarise(w) for w in current.widgets]
            emit.reasoning_step("list_widgets", count=len(widgets))
            return {"count": len(widgets), "widgets": widgets}

        def get_widget_data(widget_ids: list[str]) -> str:
            """Dispatch a batched widget-data fetch to Workspace."""
            current = run_context.current()
            by_id: dict[str, WidgetRef] = {}
            for w in current.widgets:
                by_id[w.uuid] = w
                if w.widget_id:
                    by_id.setdefault(w.widget_id, w)

            unique_ids: list[str] = []
            seen: set[str] = set()
            for wid in widget_ids:
                if wid in seen:
                    continue
                seen.add(wid)
                unique_ids.append(wid)

            missing = [wid for wid in unique_ids if wid not in by_id]
            if missing:
                available = [
                    {"uuid": w.uuid, "widget_id": w.widget_id} for w in current.widgets
                ]
                raise ValueError(
                    f"widget(s) {missing!r} not found (available: {available})"
                )

            data_sources = [
                {
                    "widget_uuid": by_id[wid].uuid,
                    "origin": by_id[wid].origin or "",
                    "id": by_id[wid].widget_id or by_id[wid].uuid,
                    "input_args": dict(by_id[wid].params),
                }
                for wid in unique_ids
            ]
            emit.function_call(
                tool_name="get_widget_data",
                parameters={"data_sources": data_sources},
            )
            return (
                f"Dispatched fetch for {len(data_sources)} widget(s) to "
                "Workspace. End this turn — no further tool calls, no "
                "reply text. The rows arrive on the next turn."
            )

        return [
            StructuredTool.from_function(
                list_widgets,
                name="list_widgets",
                description=(
                    "List every attached widget (uuid + hashes only — "
                    "no full data). Use to disambiguate by name when "
                    "the user mentions a widget without exact id."
                ),
                args_schema=_ListArgs,
            ),
            StructuredTool.from_function(
                get_widget_data,
                name="get_widget_data",
                description=(
                    "Fetch data for one or more attached widgets in a "
                    "single batched call. Pass ``widget_ids`` as a list "
                    "of the per-instance uuids shown in the "
                    "attached-widgets snapshot. Workspace executes the "
                    "fetch and returns the rows as the tool result on "
                    "the next turn. CALL THIS ONCE PER TURN with every "
                    "widget you need — do not loop calling it per id."
                ),
                args_schema=_GetWidgetArgs,
                return_direct=True,
            ),
        ]
