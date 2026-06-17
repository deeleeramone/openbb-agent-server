"""End-to-end test for a tool emitting an HTML artifact."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import tool

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import emit, services


@tool
def emit_test_html() -> str:
    """Emit a small HTML artifact for the smoke test."""
    emit.html_artifact(
        content="<h1>End-to-end works</h1>",
        name="E2E artifact",
        description="proof of life",
    )
    return "ok"


_E2E_TOOLS = [emit_test_html]


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
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", '["python_module"]')
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", json.dumps(["short reply"]))
    from openbb_agent_server.plugins.tools import python_module

    monkeypatch.setattr(
        python_module.PythonModuleToolSource,
        "__init__",
        lambda self, **kw: setattr(
            self,
            "_specs",
            ("tests.test_router_html_artifact_e2e:_E2E_TOOLS",),
        ),
    )
    with TestClient(create_app(AgentServerSettings())) as client:
        yield client


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


def test_python_module_tool_loads_and_runs(client: TestClient) -> None:
    """The python_module tool source picks up the @tool callable."""
    resp = client.post(
        "/v1/query",
        json={"messages": [{"role": "human", "content": "hi"}]},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert any(name == "copilotMessageChunk" for name, _ in events)
