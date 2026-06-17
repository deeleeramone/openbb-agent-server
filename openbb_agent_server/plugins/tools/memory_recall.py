"""``recall_user_memory`` tool source."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.memory.store import MemoryStore
from openbb_agent_server.runtime import context as run_context
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource


class _RecallArgs(BaseModel):
    query: str = Field(description="What to recall about the user.")
    k: int = Field(default=8, ge=1, le=32, description="How many memories to return.")


class MemoryRecallToolSource(ToolSource):
    """Expose a :class:`~openbb_agent_server.memory.store.MemoryStore` as the ``recall_user_memory`` tool.

    The bound store may be supplied at construction or injected later via
    ``_bind_store``. When no store is bound, :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools` yields nothing,
    so the tool is simply absent from the agent.
    """

    name = "recall_user_memory"

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        """Initialize the tool source, optionally binding a memory store.

        Parameters
        ----------
        store : MemoryStore or None, optional
            The store to recall from. If ``None``, the store can be bound
            later via ``_bind_store``; until then no tool is produced.
        """
        self._store = store

    def _bind_store(self, store: MemoryStore) -> None:
        self._store = store

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Return the ``recall_user_memory`` tool for the current run.

        The returned coroutine tool queries the bound store, scoping every
        recall to the principal of the active run context so memories
        cannot leak across users.

        Parameters
        ----------
        ctx : RunContext
            The active run context. Unused directly here; the principal is
            read from the ambient context at call time.
        config : dict of str to Any
            Per-run tool configuration. Unused by this source.

        Returns
        -------
        list of Any
            A single-element list holding the recall ``StructuredTool``, or
            an empty list when no memory store is bound.
        """
        store = self._store
        if store is None:
            return []

        async def _recall(query: str, k: int = 8) -> list[dict[str, Any]]:
            principal = run_context.current().principal
            results = await store.recall(principal=principal, query=query, k=k)
            return [
                {
                    "id": m.memory_id,
                    "text": m.text,
                    "kind": m.kind,
                    "pinned": m.pinned,
                    "score": m.score,
                }
                for m in results
            ]

        return [
            StructuredTool.from_function(
                coroutine=_recall,
                name="recall_user_memory",
                description=(
                    "Recall durable facts/preferences this user has accumulated "
                    "across prior conversations. Always scoped to the current "
                    "user; cannot leak data across users."
                ),
                args_schema=_RecallArgs,
            )
        ]
