"""Multi-agent / per-profile tests."""

from __future__ import annotations

import json
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
        "OPENBB_AGENT_FAKE_RESPONSES",
        json.dumps(["one", "two", "three", "four"]),
    )
    monkeypatch.setenv(
        "OPENBB_AGENT_PROFILES",
        json.dumps(
            {
                "equity": {
                    "metadata": {
                        "name": "Equity Analyst",
                        "description": "Specialised in equities & fundamentals.",
                    },
                    "tool_sources": [],
                    "subagents": [],
                    "middleware": [],
                    "system_prompt_file": "/tmp/equity-prompt.md",
                },
                "sentiment": {
                    "metadata": {
                        "name": "Sentiment Tracker",
                        "description": "Reads sentiment from text.",
                    },
                    "tool_sources": [],
                    "subagents": [],
                    "middleware": [],
                    "system_prompt_file": "/tmp/sentiment-prompt.md",
                },
            }
        ),
    )
    with TestClient(create_app(AgentServerSettings())) as client:
        yield client


def test_agents_json_lists_every_profile_in_one_payload(client: TestClient) -> None:
    body = client.get("/agents.json").json()
    assert set(body.keys()) == {"default", "equity", "sentiment"}

    assert body["default"]["endpoints"]["query"] == "/v1/query"

    equity = body["equity"]
    assert equity["name"] == "Equity Analyst"
    assert equity["description"] == "Specialised in equities & fundamentals."
    assert equity["endpoints"]["query"] == "/agents/equity/v1/query"

    sentiment = body["sentiment"]
    assert sentiment["name"] == "Sentiment Tracker"
    assert sentiment["endpoints"]["query"] == "/agents/sentiment/v1/query"


def test_named_profile_query_runs_to_completion(client: TestClient) -> None:
    resp = client.post(
        "/agents/equity/v1/query",
        json={
            "messages": [{"role": "human", "content": "hi"}],
            "conversation_id": "conv-equity-1",
        },
    )
    assert resp.status_code == 200


def test_unknown_profile_query_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/agents/missing/v1/query",
        json={"messages": [{"role": "human", "content": "hi"}]},
    )
    assert resp.status_code == 404


def test_same_conversation_id_under_different_profiles_uses_distinct_threads(
    client: TestClient,
) -> None:
    """The agent name namespaces state so profiles never share it."""
    convo = "shared-conversation-id"
    client.post(
        "/agents/equity/v1/query",
        json={
            "messages": [{"role": "human", "content": "hi from equity"}],
            "conversation_id": convo,
        },
    )
    client.post(
        "/agents/sentiment/v1/query",
        json={
            "messages": [{"role": "human", "content": "hi from sentiment"}],
            "conversation_id": convo,
        },
    )

    saver = client.app.state.checkpointer
    import asyncio

    async def _all() -> list:
        out = []
        async for it in saver.alist(None):
            out.append(it)
        return out

    items = asyncio.run(_all())
    threads = {it.config["configurable"]["thread_id"] for it in items}
    assert any(t.startswith("anonymous:equity:") for t in threads)
    assert any(t.startswith("anonymous:sentiment:") for t in threads)
    assert not any(t.startswith("anonymous:default:") for t in threads)


def test_default_profile_thread_uses_default_in_thread_id(
    client: TestClient,
) -> None:
    """An unprefixed query writes its checkpoint under the default namespace."""
    client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "hi"}],
            "conversation_id": "default-flow",
        },
    )
    saver = client.app.state.checkpointer
    import asyncio

    async def _all() -> list:
        out = []
        async for it in saver.alist(None):
            out.append(it)
        return out

    items = asyncio.run(_all())
    threads = {it.config["configurable"]["thread_id"] for it in items}
    assert any(t.startswith("anonymous:default:") for t in threads)
