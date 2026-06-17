"""MemoryStore ABC — per-user vector memory for cross-thread recall."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from openbb_agent_server.runtime.principal import UserPrincipal


class Memory(BaseModel):
    """One stored memory belonging to a single user.

    Attributes
    ----------
    memory_id : str
        Store-assigned unique identifier for this memory.
    user_id : str
        Identifier of the user the memory is scoped to.
    text : str
        The remembered content used for both display and recall.
    kind : str
        Category of the memory, e.g. ``"fact"`` (the default).
    pinned : bool
        Whether the memory is pinned so it is always surfaced
        regardless of relevance to a query.
    source_trace_id : str or None
        Trace identifier of the exchange that produced the memory,
        or ``None`` when it was not written from a traced run.
    score : float or None
        Relevance score assigned during recall; ``None`` outside of
        a similarity-search result.
    """

    memory_id: str
    user_id: str
    text: str
    kind: str = "fact"
    pinned: bool = False
    source_trace_id: str | None = None
    score: float | None = None


class MemoryStore(ABC):
    """Abstract per-user vector memory store for cross-thread recall.

    Implementations persist :class:`Memory` records and retrieve them by
    similarity. Every method is principal-scoped: an implementation must
    only ever read or mutate memories owned by the supplied principal and
    must never expose one user's memories to another.
    """

    @abstractmethod
    async def write(
        self,
        *,
        principal: UserPrincipal,
        text: str,
        kind: str = "fact",
        source_trace_id: str | None = None,
    ) -> Memory:
        """Persist a new memory for the principal and return it.

        Parameters
        ----------
        principal : UserPrincipal
            Owner the memory is scoped to.
        text : str
            Content to remember.
        kind : str
            Category label for the memory; defaults to ``"fact"``.
        source_trace_id : str or None
            Trace identifier of the originating exchange, if any.

        Returns
        -------
        Memory
            The stored memory, including its assigned ``memory_id``.
        """
        ...

    @abstractmethod
    async def recall(
        self,
        *,
        principal: UserPrincipal,
        query: str,
        k: int = 8,
    ) -> list[Memory]:
        """Return the principal's memories most relevant to a query.

        Parameters
        ----------
        principal : UserPrincipal
            Owner whose memories are searched.
        query : str
            Natural-language text matched against stored memories.
        k : int
            Maximum number of memories to return; defaults to 8.

        Returns
        -------
        list of Memory
            Up to ``k`` matching memories, each carrying a relevance
            ``score``, ordered most relevant first.
        """
        ...

    @abstractmethod
    async def list_memories(
        self,
        *,
        principal: UserPrincipal,
        limit: int = 100,
    ) -> list[Memory]:
        """Return the principal's memories without similarity ranking.

        Parameters
        ----------
        principal : UserPrincipal
            Owner whose memories are listed.
        limit : int
            Maximum number of memories to return; defaults to 100.

        Returns
        -------
        list of Memory
            Up to ``limit`` of the principal's stored memories.
        """
        ...

    @abstractmethod
    async def pin(
        self,
        *,
        principal: UserPrincipal,
        memory_id: str,
        pinned: bool,
    ) -> Memory | None:
        """Set the pinned flag on one of the principal's memories.

        Parameters
        ----------
        principal : UserPrincipal
            Owner the memory must belong to.
        memory_id : str
            Identifier of the memory to update.
        pinned : bool
            New pinned state to apply.

        Returns
        -------
        Memory or None
            The updated memory, or ``None`` if no memory with that id
            exists for the principal.
        """
        ...

    @abstractmethod
    async def forget(
        self,
        *,
        principal: UserPrincipal,
        memory_id: str,
    ) -> bool:
        """Delete one of the principal's memories.

        Parameters
        ----------
        principal : UserPrincipal
            Owner the memory must belong to.
        memory_id : str
            Identifier of the memory to delete.

        Returns
        -------
        bool
            ``True`` if a memory was deleted, ``False`` if none matched.
        """
        ...

    @abstractmethod
    async def delete_all_for_user(self, principal: UserPrincipal) -> int:
        """Drop every memory for the principal.

        Parameters
        ----------
        principal : UserPrincipal
            Owner whose memories are all removed.

        Returns
        -------
        int
            Number of memories deleted.
        """
