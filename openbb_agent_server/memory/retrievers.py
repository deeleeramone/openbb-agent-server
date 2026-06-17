"""LangChain ``BaseRetriever`` adapters for memory + widget stores."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from openbb_agent_server.memory.store import MemoryStore
from openbb_agent_server.runtime.principal import UserPrincipal
from openbb_agent_server.runtime.widget_store import WidgetDataStore


class MemoryStoreRetriever(BaseRetriever):
    """Retriever-shaped facade over :meth:`~openbb_agent_server.memory.store.MemoryStore.recall`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _store: MemoryStore = PrivateAttr()
    _principal: UserPrincipal = PrivateAttr()
    _k: int = PrivateAttr(default=8)

    def __init__(
        self,
        *,
        store: MemoryStore,
        principal: UserPrincipal,
        k: int = 8,
    ) -> None:
        """Bind the retriever to a memory store and a principal.

        Parameters
        ----------
        store : MemoryStore
            Backing store whose ``recall`` method supplies the memories.
        principal : UserPrincipal
            User whose memories are searched; scopes recall to that user.
        k : int, optional
            Maximum number of memories to return per query. Coerced to an
            integer and clamped to a minimum of ``1``. Defaults to ``8``.
        """
        super().__init__()
        self._store = store
        self._principal = principal
        self._k = max(1, int(k))

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return asyncio.run(self._arun(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return await self._arun(query)

    async def _arun(self, query: str) -> list[Document]:
        memories = await self._store.recall(
            principal=self._principal, query=query, k=self._k
        )
        return [
            Document(
                page_content=mem.text,
                metadata={
                    "memory_id": mem.memory_id,
                    "user_id": mem.user_id,
                    "kind": mem.kind,
                    "pinned": mem.pinned,
                    "source_trace_id": mem.source_trace_id,
                    "score": mem.score,
                },
            )
            for mem in memories
        ]


class WidgetDataRetriever(BaseRetriever):
    """Retriever-shaped facade over :meth:`~openbb_agent_server.runtime.widget_store.WidgetDataStore.search`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _store: WidgetDataStore = PrivateAttr()
    _principal: UserPrincipal = PrivateAttr()
    _conversation_id: str = PrivateAttr()
    _k: int = PrivateAttr(default=8)
    _widget_uuid: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        store: WidgetDataStore,
        principal: UserPrincipal,
        conversation_id: str,
        k: int = 8,
        widget_uuid: str | None = None,
    ) -> None:
        """Bind the retriever to a widget data store and a conversation.

        Parameters
        ----------
        store : WidgetDataStore
            Backing store whose ``search`` method supplies widget rows.
        principal : UserPrincipal
            User whose widget data is searched; scopes the search to that
            user.
        conversation_id : str
            Identifier of the conversation whose widget data is searched.
        k : int, optional
            Maximum number of rows to return per query. Coerced to an
            integer and clamped to a minimum of ``1``. Defaults to ``8``.
        widget_uuid : str or None, optional
            When provided, restricts the search to a single widget; when
            ``None`` (the default), all widgets in the conversation are
            searched.
        """
        super().__init__()
        self._store = store
        self._principal = principal
        self._conversation_id = conversation_id
        self._k = max(1, int(k))
        self._widget_uuid = widget_uuid

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return asyncio.run(self._arun(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return await self._arun(query)

    async def _arun(self, query: str) -> list[Document]:
        results = await self._store.search(
            principal=self._principal,
            conversation_id=self._conversation_id,
            query=query,
            k=self._k,
            widget_uuid=self._widget_uuid,
        )
        return [_widget_hit_to_doc(hit) for hit in results]


def _widget_hit_to_doc(hit: dict[str, Any]) -> Document:
    row = hit.get("row") or {}
    content = " | ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
    return Document(
        page_content=content,
        metadata={
            "widget_uuid": hit.get("widget_uuid"),
            "widget_name": hit.get("widget_name"),
            "score": hit.get("score"),
            "row": row,
        },
    )


__all__ = ["MemoryStoreRetriever", "WidgetDataRetriever"]
