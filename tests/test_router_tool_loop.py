"""Tool-loop drain test."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import services

_tool_calls: list[dict] = []


@tool
def lookup(symbol: str) -> str:
    """Look up a fixture symbol and record the invocation."""
    _tool_calls.append({"symbol": symbol})
    return f"PRICE({symbol})=42"


_TOOLS = [lookup]


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
    _tool_calls.clear()
    services.reset()

    from openbb_agent_server.plugins.models import fake_provider

    def fake_build(self, ctx, config):  # type: ignore[no-untyped-def]

        return fake_provider._ToolAwareFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "lookup",
                                "args": {"symbol": "AAPL"},
                                "id": str(uuid.uuid4()),
                            }
                        ],
                    ),
                    AIMessage(content="The price of AAPL is 42."),
                ]
            )
        )

    monkeypatch.setattr(fake_provider.FakeProvider, "build", fake_build, raising=True)

    from openbb_agent_server.plugins.tools import python_module

    monkeypatch.setattr(
        python_module.PythonModuleToolSource,
        "__init__",
        lambda self, **kw: setattr(
            self, "_specs", ("tests.test_router_tool_loop:_TOOLS",)
        ),
    )

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", '["python_module"]')
    monkeypatch.setenv("OPENBB_AGENT_CHECKPOINTER_PROVIDER", "sqlite")
    with TestClient(create_app(AgentServerSettings())) as client:
        yield client


def test_tool_loop_runs_to_final_answer(client: TestClient) -> None:
    """The agent invokes the tool, then returns a final-answer chunk."""
    resp = client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "what's AAPL?"}],
            "conversation_id": "conv-tool-loop-1",
        },
    )
    assert resp.status_code == 200

    assert _tool_calls == [{"symbol": "AAPL"}]

    events = _parse_sse(resp.text)
    chunks = [b["delta"] for n, b in events if n == "copilotMessageChunk"]
    assert any("AAPL" in c for c in chunks)
    assert any("42" in c for c in chunks)

    msgs = client.get("/v1/conversations/conv-tool-loop-1/messages").json()["messages"]
    ai_turns = [m for m in msgs if m["role"] == "ai"]
    assert ai_turns
    assert "42" in ai_turns[-1]["content"]


def test_tool_loop_persists_full_trace(client: TestClient) -> None:
    """The trace bundle reflects a completed run, not a cancelled one."""
    resp = client.post(
        "/v1/query",
        json={
            "messages": [{"role": "human", "content": "what's AAPL?"}],
            "conversation_id": "conv-tool-loop-2",
        },
    )
    assert resp.status_code == 200
    server_trace_id = resp.headers["X-Server-Trace-ID"]
    bundle = client.get(f"/v1/traces/{server_trace_id}").json()
    assert bundle["trace"]["status"] == "completed"
