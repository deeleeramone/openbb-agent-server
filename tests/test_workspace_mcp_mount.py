"""Tests for the optional workspace-mcp sub-app mount."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest
from fastapi import FastAPI

from openbb_agent_server.app.app import create_app
from openbb_agent_server.app.settings import AgentServerSettings


def _find_mount(app: FastAPI, path: str) -> Any | None:
    """Return the Mount route matching ``path`` if present, else ``None``."""
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route
    return None


def test_mount_skipped_by_default(
    settings_env: AgentServerSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mount_workspace_mcp`` defaults to ``False`` — installing the extra alone is not enough.

    Even with workspace_mcp importable (stubbed), the mount stays off until
    the operator explicitly opts in. This is the safe default.
    """
    stub_module = _install_workspace_mcp_stub(monkeypatch)

    app = create_app(settings_env)

    assert _find_mount(app, "/mcp/workspace") is None
    # Import should never have been attempted, so the factory must not have run.
    assert stub_module["create_app_called"] is False


def test_mount_skipped_when_workspace_mcp_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    settings_env: AgentServerSettings,
) -> None:
    """Opt-in but extra missing: soft-skip with an info log, no crash."""
    monkeypatch.setenv("OPENBB_AGENT_MOUNT_WORKSPACE_MCP", "true")
    # Force ImportError on `from workspace_mcp.app import create_app` and friends
    # by poisoning the parent package entry in sys.modules.
    monkeypatch.setitem(sys.modules, "workspace_mcp", None)

    with caplog.at_level(logging.INFO, logger="openbb_agent_server.app"):
        app = create_app(AgentServerSettings())

    assert _find_mount(app, "/mcp/workspace") is None
    assert any(
        "workspace-mcp extra not installed" in rec.message for rec in caplog.records
    )


def test_mount_added_when_opted_in_and_importable(
    monkeypatch: pytest.MonkeyPatch,
    settings_env: AgentServerSettings,
) -> None:
    """Opt-in + importable: the sub-app is mounted at /mcp/workspace."""
    monkeypatch.setenv("OPENBB_AGENT_MOUNT_WORKSPACE_MCP", "true")
    stub_module = _install_workspace_mcp_stub(monkeypatch)

    app = create_app(AgentServerSettings())

    mount = _find_mount(app, "/mcp/workspace")
    assert mount is not None, "expected a Mount at /mcp/workspace"
    assert stub_module["create_app_called"] is True


def test_mount_failure_does_not_break_parent_app(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    settings_env: AgentServerSettings,
) -> None:
    """If the workspace-mcp factory raises, the parent app still constructs."""
    monkeypatch.setenv("OPENBB_AGENT_MOUNT_WORKSPACE_MCP", "true")

    def _boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("simulated workspace-mcp init failure")

    _install_workspace_mcp_stub(monkeypatch, factory=_boom)

    with caplog.at_level(logging.WARNING, logger="openbb_agent_server.app"):
        app = create_app(AgentServerSettings())

    assert _find_mount(app, "/mcp/workspace") is None
    assert any("workspace-mcp mount failed" in rec.message for rec in caplog.records)


def _install_workspace_mcp_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    factory: Any | None = None,
) -> dict[str, Any]:
    """Register a fake ``workspace_mcp`` package in sys.modules.

    Returns a sentinel dict recording whether ``create_app`` was invoked.
    """
    state: dict[str, Any] = {"create_app_called": False}

    pkg = types.ModuleType("workspace_mcp")
    app_mod = types.ModuleType("workspace_mcp.app")
    config_mod = types.ModuleType("workspace_mcp.config")

    class _Settings:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def _default_factory(_settings: Any) -> FastAPI:
        state["create_app_called"] = True
        return FastAPI()

    app_mod.create_app = factory or _default_factory  # type: ignore[attr-defined]
    config_mod.Settings = _Settings  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "workspace_mcp", pkg)
    monkeypatch.setitem(sys.modules, "workspace_mcp.app", app_mod)
    monkeypatch.setitem(sys.modules, "workspace_mcp.config", config_mod)
    return state
