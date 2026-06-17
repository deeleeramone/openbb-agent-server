"""Multi-tenant isolation through the HTTP surface."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import services


@pytest.fixture
def alice_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[TestClient]:
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "bearer_static")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BEARER", "alice-token")
    monkeypatch.setenv(
        "OPENBB_AGENT_AUTH_CONFIG",
        '{"token": "alice-token", "user_id": "alice", "scopes": ["agent:query","memory:read","memory:write"]}',
    )
    app = create_app(AgentServerSettings())
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer alice-token"
        yield client


def test_bearer_static_round_trip_with_real_principal(
    alice_client: TestClient,
) -> None:
    me = alice_client.get("/v1/me")
    assert me.status_code == 200
    assert me.json()["user_id"] == "alice"

    resp = alice_client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "hi"}],
            "conversation_id": "alice-conv-1",
        },
    )
    assert resp.status_code == 200

    convs = alice_client.get("/v1/conversations").json()["conversations"]
    assert len(convs) == 1
    assert convs[0]["conversation_id"] == "alice-conv-1"


def test_unauthenticated_request_is_401(
    alice_client: TestClient,
) -> None:
    raw = TestClient(alice_client.app)
    with raw:
        resp = raw.get("/v1/me")
    assert resp.status_code == 401
