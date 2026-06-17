"""SSE encoder."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from openbb_agent_server.protocol.schemas import SSEEvent


def encode_event(event: SSEEvent) -> bytes:
    """Encode one event as a UTF-8 SSE frame.

    Serialises the event's ``data`` model to JSON and wraps it in the
    text/event-stream framing (``event:`` and ``data:`` lines terminated by
    a blank line).

    Parameters
    ----------
    event : SSEEvent
        The event to encode; its ``event`` field names the SSE event type
        and its ``data`` field is dumped to JSON as the payload.

    Returns
    -------
    bytes
        The UTF-8-encoded SSE frame.
    """
    payload = event.data.model_dump_json()
    frame = f"event: {event.event}\ndata: {payload}\n\n"
    return frame.encode("utf-8")


async def encode_stream(
    events: Iterable[SSEEvent] | AsyncIterator[SSEEvent],
) -> AsyncIterator[bytes]:
    """Encode a sync or async stream of events into SSE frames.

    Detects whether ``events`` is async-iterable (``__aiter__``) and
    consumes it accordingly, yielding each event encoded by
    :func:`encode_event`.

    Parameters
    ----------
    events : Iterable[SSEEvent] | AsyncIterator[SSEEvent]
        The source of events, either a synchronous iterable or an async
        iterator.

    Yields
    ------
    bytes
        One UTF-8-encoded SSE frame per event, in order.
    """
    from typing import cast

    if hasattr(events, "__aiter__"):
        async for ev in cast(AsyncIterator[SSEEvent], events):
            yield encode_event(ev)
    else:
        for ev in cast(Iterable[SSEEvent], events):
            yield encode_event(ev)
