"""Tests for the memory, cancel, and scope-error router endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import services


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "bearer_static")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BEARER", "tk")
    monkeypatch.setenv(
        "OPENBB_AGENT_AUTH_CONFIG",
        '{"token": "tk", "user_id": "u1", '
        '"scopes": ["agent:query", "memory:read", "memory:write"]}',
    )
    with TestClient(create_app(AgentServerSettings())) as client:
        client.headers["Authorization"] = "Bearer tk"
        yield client


@pytest.fixture
def readonly_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[TestClient]:
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "bearer_static")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'ro.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BEARER", "tk")
    monkeypatch.setenv(
        "OPENBB_AGENT_AUTH_CONFIG",
        '{"token": "tk", "user_id": "ro", "scopes": ["memory:read"]}',
    )
    with TestClient(create_app(AgentServerSettings())) as client:
        client.headers["Authorization"] = "Bearer tk"
        yield client


def _seed_memory(client: TestClient) -> str:
    """Populate one memory directly via the memory store."""
    history = client.app.state.history
    memory = client.app.state.memory
    import asyncio

    async def _seed() -> str:
        from openbb_agent_server.runtime.principal import UserPrincipal

        p = UserPrincipal(user_id="u1", scopes=("memory:read", "memory:write"))
        await history.upsert_user(p)
        m = await memory.write(principal=p, text="seeded memory")
        return m.memory_id

    return asyncio.run(_seed())


def test_memory_endpoint_lists_memories(client: TestClient) -> None:
    _seed_memory(client)
    resp = client.get("/v1/memory")
    assert resp.status_code == 200
    rows = resp.json()["memories"]
    assert len(rows) == 1
    assert rows[0]["text"] == "seeded memory"


def test_memory_pin_endpoint(client: TestClient) -> None:
    memory_id = _seed_memory(client)
    resp = client.patch(
        f"/v1/memory/{memory_id}",
        json={"pinned": True},
    )
    assert resp.status_code == 200
    assert resp.json()["pinned"] is True


def test_memory_pin_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.patch("/v1/memory/does-not-exist", json={"pinned": True})
    assert resp.status_code == 404


def test_memory_patch_with_no_recognised_fields_returns_400(client: TestClient) -> None:
    memory_id = _seed_memory(client)
    resp = client.patch(f"/v1/memory/{memory_id}", json={})
    assert resp.status_code == 400


def test_memory_delete_endpoint(client: TestClient) -> None:
    memory_id = _seed_memory(client)
    resp = client.delete(f"/v1/memory/{memory_id}")
    assert resp.status_code == 204
    rows = client.get("/v1/memory").json()["memories"]
    assert all(r["memory_id"] != memory_id for r in rows)


def test_memory_delete_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.delete("/v1/memory/does-not-exist")
    assert resp.status_code == 404


def test_memory_write_requires_memory_write_scope(
    readonly_client: TestClient,
) -> None:
    resp = readonly_client.delete("/v1/memory/anything")
    assert resp.status_code == 403


def test_query_requires_agent_query_scope(
    readonly_client: TestClient,
) -> None:
    resp = readonly_client.post(
        "/v1/query",
        json={"messages": [{"role": "human", "content": "hi"}]},
    )
    assert resp.status_code == 403


def test_cancel_returns_run_ids_for_caller(client: TestClient) -> None:
    resp = client.post("/v1/conversations/conv-cancel-1/cancel")
    assert resp.status_code == 202
    body = resp.json()
    assert body["conversation_id"] == "conv-cancel-1"
    assert body["cancelled_runs"] == []


def test_messages_endpoint_returns_persisted_history(client: TestClient) -> None:
    client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "first"}],
            "conversation_id": "conv-msg-1",
        },
    )
    client.post(
        "/v1/query",
        json={
            "messages": [
                {"role": "human", "content": "first"},
                {"role": "ai", "content": "OK."},
                {"role": "human", "content": "second"},
            ],
            "conversation_id": "conv-msg-1",
        },
    )
    resp = client.get("/v1/conversations/conv-msg-1/messages")
    assert resp.status_code == 200
    msgs = resp.json()["messages"]
    contents = [m["content"] for m in msgs if m["role"] == "human"]
    assert "first" in contents
    assert "second" in contents


def test_list_messages_endpoint_returns_empty_for_unknown_conversation(
    client: TestClient,
) -> None:
    """The messages endpoint returns empty for an unknown conversation."""
    resp = client.get("/v1/conversations/no-such-conv/messages")
    assert resp.status_code == 200
    assert resp.json() == {"messages": []}


def test_create_app_resolves_settings_from_toml_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENBB_AGENT_BOOTSTRAP_TOML", raising=False)
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    app = create_app()
    assert app is not None
