"""End-to-end smoke tests for /agents.json and /v1/me."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.runtime import services


@pytest.fixture
def client(settings_env: AgentServerSettings) -> Iterator[TestClient]:
    services.reset()
    app = create_app(settings_env)
    with TestClient(app) as client:
        yield client


def test_agents_json_returns_workspace_metadata(client: TestClient) -> None:
    resp = client.get("/agents.json")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    assert "default" in body
    entry = body["default"]
    assert "name" in entry
    assert "description" in entry
    assert entry["endpoints"]["query"] == "/v1/query"
    assert entry["features"]["streaming"] is True


def test_v1_query_accepts_real_workspace_body_shape(client: TestClient) -> None:
    """Post a body shaped like Workspace's actual query payload."""
    body = {
        "messages": [
            {"role": "human", "content": "Hi there."},
        ],
        "widgets": {
            "primary": [
                {
                    "uuid": "12345678-1234-1234-1234-123456789abc",
                    "origin": "openbb",
                    "widget_id": "fundamentals",
                    "name": "Fundamentals",
                    "description": "Company fundamentals data",
                    "params": [
                        {
                            "name": "ticker",
                            "type": "ticker",
                            "description": "Equity ticker",
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "uploaded_files": [],
        "api_keys": {"openai_api_key": None},
        "api_urls": {},
        "workspace_options": ["deep-research", "web-search"],
        "timezone": "UTC",
        "urls": None,
        "force_web_search": False,
        "workspace_state": None,
        "context": None,
        "tools": None,
        "run_id": "00000000-0000-0000-0000-000000000001",
        "conversation_id": "11111111-1111-1111-1111-111111111111",
    }
    resp = client.post("/v1/query", json=body)
    assert resp.status_code != 422, (
        f"Workspace body must not trip pydantic validation; got {resp.status_code} "
        f"with body: {resp.text[:500]}"
    )


def test_agents_json_conforms_to_workspace_spec(client: TestClient) -> None:
    """Validate the response against the OpenBB Workspace agents.json schema."""
    import re

    body = client.get("/agents.json").json()
    assert isinstance(body, dict)

    agent_id_re = re.compile(r"^[a-z0-9-]+$")
    allowed_per_agent = {"name", "description", "image", "endpoints", "features"}
    required_per_agent = {"name", "description", "endpoints", "features"}

    for agent_id, entry in body.items():
        assert agent_id_re.match(agent_id), f"agent_id {agent_id!r} fails regex"

        assert isinstance(entry, dict)
        assert required_per_agent.issubset(entry.keys()), (
            f"missing required keys: {required_per_agent - entry.keys()}"
        )
        assert set(entry.keys()).issubset(allowed_per_agent), (
            f"unknown keys at agents[{agent_id!r}]: "
            f"{set(entry.keys()) - allowed_per_agent}"
        )

        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["description"], str) and entry["description"]

        if "image" in entry:
            assert entry["image"] is None or isinstance(entry["image"], str)

        endpoints = entry["endpoints"]
        assert isinstance(endpoints, dict)
        assert set(endpoints.keys()) == {"query"}
        assert isinstance(endpoints["query"], str) and endpoints["query"]

        features = entry["features"]
        assert isinstance(features, dict)
        for known_key in (
            "streaming",
            "widget-dashboard-select",
            "widget-dashboard-search",
        ):
            if known_key in features:
                assert isinstance(features[known_key], bool), (
                    f"features[{known_key!r}] must be bool, got "
                    f"{type(features[known_key]).__name__}"
                )


def test_v1_me_with_none_auth_returns_anonymous(client: TestClient) -> None:
    resp = client.get("/v1/me")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "anonymous"
    assert "agent:query" in body["scopes"]


def test_list_conversations_empty_for_fresh_user(client: TestClient) -> None:
    resp = client.get("/v1/conversations")
    assert resp.status_code == 200
    assert resp.json() == {"conversations": []}


def test_memory_endpoint_requires_memory_read_scope(client: TestClient) -> None:
    resp = client.get("/v1/memory")
    assert resp.status_code == 200
    assert resp.json() == {"memories": []}


def test_create_app_with_code_embeddings_provider_boots(
    monkeypatch: pytest.MonkeyPatch, settings_env: AgentServerSettings
) -> None:
    """Setting ``embeddings_code_provider`` builds the code-embedding store."""
    monkeypatch.setenv("OPENBB_AGENT_EMBEDDINGS_CODE_PROVIDER", "hash")
    services.reset()
    settings = AgentServerSettings()
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/agents.json").status_code == 200


def test_agents_json_emits_image_when_metadata_image_url_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A profile whose metadata carries ``image_url`` emits an ``image`` key."""
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_META_IMAGE_URL", "https://x/logo.png")
    with TestClient(create_app(AgentServerSettings())) as c:
        body = c.get("/agents.json").json()
    assert body["default"]["image"] == "https://x/logo.png"


def test_agents_json_drops_profile_with_illegal_agent_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A profile name that fails the ``[a-z0-9-]+`` regex is dropped."""

    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv(
        "OPENBB_AGENT_PROFILES",
        json.dumps(
            {"Bad_Name": {"metadata": {"name": "Bad", "description": "illegal id"}}}
        ),
    )
    with (
        caplog.at_level(logging.WARNING),
        TestClient(create_app(AgentServerSettings())) as c,
    ):
        body = c.get("/agents.json").json()
    assert "Bad_Name" not in body
    assert any("agent_id must match" in r.message for r in caplog.records)


def test_router_coerce_feature_handles_dict_and_other_shapes() -> None:
    """Coerce features from bools and dict shapes."""
    from openbb_agent_server.app.router import _coerce_feature

    assert _coerce_feature(True) is True
    assert _coerce_feature(False) is False
    assert _coerce_feature({"default": True}) is True
    assert _coerce_feature({"default": False}) is False
    assert _coerce_feature({}) is False
    assert _coerce_feature(1) is True
    assert _coerce_feature(0) is False
    assert _coerce_feature(None) is False


def test_router_agents_json_emits_image_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Include the image in the agents.json entry when image_url is set."""
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_META_IMAGE_URL", "https://x.test/logo.png")

    with TestClient(create_app(AgentServerSettings())) as client:
        body = client.get("/agents.json").json()
    assert body["default"]["image"] == "https://x.test/logo.png"


def test_router_agents_json_drops_profiles_with_invalid_agent_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drop profiles whose agent id violates the Workspace regex."""
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv(
        "OPENBB_AGENT_PROFILES",
        json.dumps({"Bad_ID": {}, "fine-id": {}}),
    )

    app = create_app(AgentServerSettings())
    with TestClient(app) as client:
        body = client.get("/agents.json").json()
    assert "default" in body
    assert "fine-id" in body
    assert "Bad_ID" not in body


def test_router_agents_json_skips_unresolvable_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skip a profile that resolve_profile cannot resolve."""
    services.reset()
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_PROFILES", json.dumps({"alt": {}}))
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')

    settings = AgentServerSettings()
    app = create_app(settings)

    real_resolve = AgentServerSettings.resolve_profile

    def _resolve(self: Any, name: str | None = None):
        if name == "alt":
            raise KeyError("simulated")
        return real_resolve(self, name)

    monkeypatch.setattr(AgentServerSettings, "resolve_profile", _resolve)
    with TestClient(app) as client:
        body = client.get("/agents.json").json()
    assert "default" in body
    assert "alt" not in body
