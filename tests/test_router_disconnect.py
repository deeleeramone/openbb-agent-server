"""Disconnect-during-stream test."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import services


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_CHECKPOINTER_PROVIDER", "sqlite")
    monkeypatch.setenv(
        "OPENBB_AGENT_FAKE_RESPONSES", json.dumps(["the full final answer"])
    )
    with TestClient(create_app(AgentServerSettings())) as client:
        yield client


def test_run_drains_to_natural_end_when_client_disconnects(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run drains to natural end after the client disconnects."""
    from starlette.requests import Request

    flipped = {"value": False}
    original = Request.is_disconnected

    async def patched_is_disconnected(self):  # type: ignore[no-untyped-def]
        if not flipped["value"]:
            flipped["value"] = True
            return False
        return True

    monkeypatch.setattr(Request, "is_disconnected", patched_is_disconnected)

    resp = client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "say hi"}],
            "conversation_id": "conv-disconnect-1",
        },
    )
    assert resp.status_code == 200

    msgs = client.get("/v1/conversations/conv-disconnect-1/messages").json()["messages"]
    ai_turns = [m for m in msgs if m["role"] == "ai"]
    assert ai_turns
    assert ai_turns[-1]["content"] == "the full final answer"


def test_run_drains_does_not_record_cancelled_status(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnect does not leave the trace marked cancelled."""
    from starlette.requests import Request

    async def always_disconnected(self):  # type: ignore[no-untyped-def]
        return True

    monkeypatch.setattr(Request, "is_disconnected", always_disconnected)

    resp = client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "say hi"}],
            "conversation_id": "conv-disconnect-2",
        },
    )
    assert resp.status_code == 200
    server_trace_id = resp.headers["X-Server-Trace-ID"]
    bundle = client.get(f"/v1/traces/{server_trace_id}").json()
    assert bundle["trace"]["status"] == "completed"


def test_explicit_cancel_endpoint_still_marks_cancelled(
    client: TestClient,
) -> None:
    """The opt-in cancel POST is still honoured."""
    resp = client.post("/v1/conversations/whatever/cancel")
    assert resp.status_code == 202
    assert "cancelled_runs" in resp.json()


def test_router_cancel_matches_user_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Match the cancel event by user id."""
    import asyncio as _asyncio

    from openbb_agent_server.app import router as router_mod

    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")

    settings = AgentServerSettings()
    app = create_app(settings)

    ev = _asyncio.Event()
    router_mod._cancellations[("anonymous", "preset-run")] = ev

    with TestClient(app) as client:
        resp = client.post("/v1/conversations/cv-1/cancel")
    assert resp.status_code == 202
    assert ev.is_set()
    router_mod._cancellations.clear()


def test_router_streams_error_status_when_run_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stream an error status when run_agent raises."""
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")

    async def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("simulated agent failure")
        yield  # noqa: F811, ASYNC131  — required so this stays an async generator

    monkeypatch.setattr("openbb_agent_server.app.router.run_agent", _boom)

    settings = AgentServerSettings()
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/query",
            json={
                "messages": [{"role": "human", "content": "hi"}],
                "conversation_id": "cv-err",
            },
        )
    body = resp.text
    assert "simulated agent failure" in body
    assert "ERROR" in body


def test_router_cancel_during_stream_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Short-circuit the stream when a cancel event fires mid-stream."""
    from openbb_agent_server.app import router as router_mod
    from openbb_agent_server.protocol.schemas import StatusUpdateSSE

    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")

    async def _slow_stream(*args: Any, **kwargs: Any):
        yield StatusUpdateSSE(event_type="INFO", message="step", details={})
        for ev in list(router_mod._cancellations.values()):
            ev.set()
        yield StatusUpdateSSE(event_type="INFO", message="should-not-emit", details={})

    monkeypatch.setattr("openbb_agent_server.app.router.run_agent", _slow_stream)

    settings = AgentServerSettings()
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/query",
            json={
                "messages": [{"role": "human", "content": "hi"}],
                "conversation_id": "cv-cancel",
            },
        )
    assert resp.status_code == 200
