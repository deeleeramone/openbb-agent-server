"""SSE encoder tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from openbb_agent_server.protocol.schemas import (
    MessageChunkSSE,
    MessageChunkSSEData,
    SSEEvent,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)
from openbb_agent_server.protocol.sse import encode_event, encode_stream


def test_encode_chunk_event_emits_only_inner_data() -> None:
    """The chunk event emits only the inner data payload."""
    frame = encode_event(MessageChunkSSE(data=MessageChunkSSEData(delta="hello")))
    text = frame.decode()
    assert text.startswith("event: copilotMessageChunk\n")
    assert "data: " in text
    payload = json.loads(text.split("data: ", 1)[1].rstrip())
    assert payload == {"delta": "hello"}
    assert text.endswith("\n\n")


def test_encode_status_update_uses_camelcase_eventtype_field() -> None:
    frame = encode_event(
        StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="ERROR", message="boom", details=[{"k": 1}]
            )
        )
    ).decode()
    assert "event: copilotStatusUpdate\n" in frame
    payload = json.loads(frame.split("data: ", 1)[1].rstrip())
    assert payload["eventType"] == "ERROR"
    assert payload["details"] == [{"k": 1}]


@pytest.mark.asyncio
async def test_encode_stream_handles_sync_iterable() -> None:
    events = [
        MessageChunkSSE(data=MessageChunkSSEData(delta="a")),
        MessageChunkSSE(data=MessageChunkSSEData(delta="b")),
    ]
    out: list[bytes] = []
    async for frame in encode_stream(events):
        out.append(frame)
    assert len(out) == 2
    assert b'"delta":"a"' in out[0]
    assert b'"delta":"b"' in out[1]


@pytest.mark.asyncio
async def test_encode_stream_handles_async_iterable() -> None:
    async def gen() -> AsyncIterator[SSEEvent]:
        yield MessageChunkSSE(data=MessageChunkSSEData(delta="x"))
        yield StatusUpdateSSE(data=StatusUpdateSSEData(eventType="INFO", message="y"))

    out: list[bytes] = []
    async for frame in encode_stream(gen()):
        out.append(frame)
    assert len(out) == 2
    assert b"event: copilotMessageChunk" in out[0]
    assert b"event: copilotStatusUpdate" in out[1]
