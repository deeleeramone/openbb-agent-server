"""Multi-turn + tool-loop integration tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import services


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    out: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        name = ""
        payload = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payload = line[len("data:") :].strip()
        if name and payload:
            out.append((name, json.loads(payload)))
    return out


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
        json.dumps(
            [
                "First answer.",
                "Second answer that builds on the first.",
                "Third answer continuing the thread.",
            ]
        ),
    )
    with TestClient(create_app(AgentServerSettings())) as client:
        yield client


def test_two_turns_share_conversation_via_message_list(client: TestClient) -> None:
    """Turn 2 of the same conversation_id picks up where turn 1 left off."""
    r1 = client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "what is 2 + 2"}],
            "conversation_id": "conv-multiturn-1",
        },
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/v1/query",
        json={
            "messages": [
                {"role": "human", "content": "what is 2 + 2"},
                {"role": "ai", "content": "4"},
                {"role": "human", "content": "and 3 + 3?"},
            ],
            "conversation_id": "conv-multiturn-1",
        },
    )
    assert r2.status_code == 200
    assert r1.headers["X-Trace-ID"] == r2.headers["X-Trace-ID"] == "conv-multiturn-1"
    assert r1.headers["X-Server-Trace-ID"] != r2.headers["X-Server-Trace-ID"]

    saver = client.app.state.checkpointer
    import asyncio

    async def _collect():
        out = []
        async for it in saver.alist(None):
            out.append(it)
        return out

    items = asyncio.run(_collect())
    threads = {it.config["configurable"]["thread_id"] for it in items}
    matching = [t for t in threads if t.startswith("anonymous:default:")]
    assert len(matching) >= 2

    msgs = client.get("/v1/conversations/conv-multiturn-1/messages").json()["messages"]
    human_contents = [m["content"] for m in msgs if m["role"] == "human"]
    assert "what is 2 + 2" in human_contents
    assert "and 3 + 3?" in human_contents


def test_three_turns_keep_main_thread_alive(client: TestClient) -> None:
    """The same conversation_id can host arbitrarily many turns end-to-end."""
    convo = "conv-three-turn-1"
    history: list[dict[str, str]] = []
    for prompt in ("hello", "how are you", "what's the weather"):
        history.append({"role": "human", "content": prompt})
        resp = client.post(
            "/v1/query",
            json={"messages": list(history), "conversation_id": convo},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        names = [n for n, _ in events]
        assert "copilotMessageChunk" in names
        history.append({"role": "ai", "content": "ok"})

    msgs = client.get(f"/v1/conversations/{convo}/messages").json()["messages"]
    human = [m for m in msgs if m["role"] == "human"]
    assert len(human) == 3


def test_run_completes_to_natural_end_with_full_chunk_assembly(
    client: TestClient,
) -> None:
    """The full canned response is assembled and saved as the AI turn."""
    resp = client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "hi"}],
            "conversation_id": "conv-natural-end-1",
        },
    )
    assert resp.status_code == 200

    msgs = client.get("/v1/conversations/conv-natural-end-1/messages").json()[
        "messages"
    ]
    ai_turns = [m for m in msgs if m["role"] == "ai"]
    assert len(ai_turns) == 1
    assert ai_turns[0]["content"] == "First answer."
