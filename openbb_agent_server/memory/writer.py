"""Post-run memory writer."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from openbb_agent_server.memory.store import MemoryStore
from openbb_agent_server.runtime.principal import UserPrincipal

logger = logging.getLogger("openbb_agent_server.memory.writer")


EXTRACTOR_SYSTEM_PROMPT = """\
Extract durable, user-scoped facts and preferences from the conversation
below — things worth remembering across future sessions. Skip volatile
chatter (today's stock prices, transient task state, model errors).

Output one line per memory, plain text, no numbering. If nothing is
worth remembering, output exactly: NONE
"""


def _split(raw: str) -> list[str]:
    """Parse the extractor's reply into a list of memory lines."""
    if not raw or "NONE" in raw.upper().split():
        return []
    lines = [line.strip(" -*•\t") for line in raw.splitlines()]
    return [line for line in lines if line and len(line) > 6]


def _render_transcript(human_text: str, ai_text: str) -> str:
    parts: list[str] = []
    if human_text:
        parts.append(f"human: {human_text}")
    if ai_text:
        parts.append(f"ai: {ai_text}")
    return "\n".join(parts)


async def write_memories(
    *,
    principal: UserPrincipal,
    store: MemoryStore,
    extractor: BaseChatModel,
    human_text: str,
    ai_text: str,
    trace_id: str,
) -> int:
    """Extract durable facts from a turn and persist them to the store.

    Render the human / AI turn into a transcript, ask ``extractor`` to
    distil durable user-scoped facts (using
    :data:`EXTRACTOR_SYSTEM_PROMPT`), split the reply into individual
    memory lines, and write each as a ``"fact"`` to ``store``.

    Returns ``0`` without invoking the model when the principal lacks the
    ``memory:write`` scope, the transcript is empty, or the extraction
    call fails (the error is logged, not raised).

    Parameters
    ----------
    principal : UserPrincipal
        Authenticated user the memories belong to; must hold the
        ``memory:write`` scope.
    store : MemoryStore
        Destination store each extracted fact is written to.
    extractor : BaseChatModel
        Chat model invoked to produce the candidate memory lines.
    human_text : str
        The user's message for this turn.
    ai_text : str
        The assistant's reply for this turn.
    trace_id : str
        Trace identifier recorded on each write as ``source_trace_id``.

    Returns
    -------
    int
        The number of memory lines successfully written.
    """
    if not principal.has_scope("memory:write"):
        return 0
    transcript = _render_transcript(human_text, ai_text)
    if not transcript:
        return 0
    prompt = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=transcript),
    ]
    try:
        result = await extractor.ainvoke(prompt)
    except Exception as exc:
        logger.warning("memory extraction failed: %s", exc)
        return 0
    text = result.content if isinstance(result, AIMessage) else str(result)
    written = 0
    for line in _split(str(text)):
        await store.write(
            principal=principal,
            text=line,
            kind="fact",
            source_trace_id=trace_id,
        )
        written += 1
    return written


def schedule(
    *,
    principal: UserPrincipal,
    store: MemoryStore | None,
    extractor: BaseChatModel | None,
    human_text: str,
    ai_text: str,
    trace_id: str,
) -> asyncio.Task[int] | None:
    """Fire-and-forget the memory write so the SSE response can close.

    Schedule :func:`write_memories` on the event loop without awaiting it,
    so the streaming response can finish while extraction runs in the
    background. Return ``None`` (scheduling nothing) when memory is not
    configured (``store`` or ``extractor`` is ``None``) or the principal
    lacks the ``memory:write`` scope.

    Parameters
    ----------
    principal : UserPrincipal
        Authenticated user the memories belong to; must hold the
        ``memory:write`` scope.
    store : MemoryStore or None
        Destination store; ``None`` disables memory writing.
    extractor : BaseChatModel or None
        Extraction chat model; ``None`` disables memory writing.
    human_text : str
        The user's message for this turn.
    ai_text : str
        The assistant's reply for this turn.
    trace_id : str
        Trace identifier; also used to name the background task.

    Returns
    -------
    asyncio.Task of int or None
        The scheduled task (resolving to the count written), or ``None``
        when nothing was scheduled.
    """
    if store is None or extractor is None:
        return None
    if not principal.has_scope("memory:write"):
        return None
    coro = write_memories(
        principal=principal,
        store=store,
        extractor=extractor,
        human_text=human_text,
        ai_text=ai_text,
        trace_id=trace_id,
    )
    return asyncio.create_task(coro, name=f"memory-write:{trace_id}")
