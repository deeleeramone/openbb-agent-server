"""Plugin registry tests — entry-point loading + error paths."""

from __future__ import annotations

import pytest

from openbb_agent_server.runtime import registry
from openbb_agent_server.runtime.plugins import AuthBackend


def test_available_lists_registered_auth_backends() -> None:
    names = registry.available("openbb_agent_server.auth")
    assert {
        "none",
        "bearer_static",
        "api_key_table",
        "oidc_jwt",
        "openbb_workspace",
    }.issubset(names)


def test_load_resolves_to_real_class() -> None:
    backend = registry.load("openbb_agent_server.auth", "none")
    assert isinstance(backend, AuthBackend)
    assert backend.name == "none"


def test_load_unknown_plugin_raises_keyerror() -> None:
    with pytest.raises(KeyError) as exc:
        registry.load("openbb_agent_server.auth", "does-not-exist")
    assert "does-not-exist" in str(exc.value)


def test_load_passes_config_kwargs_to_constructor() -> None:
    registry.load(
        "openbb_agent_server.auth",
        "none",
        {"unused_key": "ignored"},
    )


def test_model_provider_group_registers_all_providers() -> None:
    names = registry.available("openbb_agent_server.model_providers")
    assert {
        "anthropic",
        "openai",
        "openai_compat",
        "nvidia",
        "bedrock",
        "vertex",
        "google_genai",
        "groq",
        "fake",
    }.issubset(names)


def test_tool_source_group_registers_all_sources() -> None:
    names = registry.available("openbb_agent_server.tools")
    expected = {
        "python_module",
        "client_side",
        "mcp_local",
        "mcp_http",
        "recall_user_memory",
        "groq_audio",
        "gemini_image",
        "gemini_embeddings",
    }
    assert expected.issubset(names)


def test_subagent_group_registers_all_subagents() -> None:
    names = registry.available("openbb_agent_server.subagents")
    assert {"researcher", "charter", "analyst", "pdf_reader"}.issubset(names)


def test_middleware_group_registers_all_middleware() -> None:
    names = registry.available("openbb_agent_server.middleware")
    assert {
        "usage_recorder",
        "tool_call_ledger",
        "tool_call_announcer",
        "call_limit",
        "tool_call_limit",
    }.issubset(names)
    assert "memory_writer" not in names


def test_load_drops_unknown_config_keys_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A strict-signature plugin keeps only kwargs its __init__ accepts."""
    import logging

    with caplog.at_level(logging.WARNING):
        provider = registry.load(
            "openbb_agent_server.model_providers",
            "openai_compat",
            {
                "base_url": "http://localhost:8000/v1",
                "model_name": "llama3.1",
                "misplaced_key": "oops",
            },
        )
    assert provider is not None
    assert any("misplaced_key" in r.message for r in caplog.records)


def test_load_tolerates_uninspectable_init_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When inspect.signature cannot read __init__, all config keys pass."""

    def _raising_signature(obj: object, *args: object, **kw: object) -> object:
        raise ValueError("synthetic signature failure")

    monkeypatch.setattr(registry.inspect, "signature", _raising_signature)
    backend = registry.load(
        "openbb_agent_server.auth",
        "none",
        {"anything_at_all": "kept"},
    )
    assert backend.name == "none"


def test_accepted_kwargs_ignores_var_positional() -> None:
    """Skip a *args parameter and collect only named kwargs."""
    from openbb_agent_server.runtime.registry import _accepted_kwargs

    class _HasVarPositional:
        def __init__(self, alpha: int, *args: object, beta: int = 1) -> None:
            self.alpha = alpha
            self.beta = beta

    accepted = _accepted_kwargs(_HasVarPositional)
    assert accepted == {"alpha", "beta"}
