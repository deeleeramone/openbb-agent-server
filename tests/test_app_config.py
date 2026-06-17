"""Tests for the layered TOML config bootstrap."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from openbb_agent_server.app import config as cfg_mod
from openbb_agent_server.app.config import (
    EXPLICIT_CONFIG_ENVS,
    agent_section,
    apply_launcher_env,
    bootstrap_launcher_config,
    expand_env_refs,
    expand_in_dict,
    explicit_config_path,
    extract_config_file_from_argv,
    merge_launcher_kwargs,
)
from openbb_agent_server.app.settings import AgentServerSettings


def test_extract_config_file_from_argv_space_separated() -> None:
    out = extract_config_file_from_argv(["--port", "8000", "--config-file", "/x.toml"])
    assert out == "/x.toml"


def test_extract_config_file_from_argv_equals_form() -> None:
    out = extract_config_file_from_argv(["--config-file=/y.toml"])
    assert out == "/y.toml"


def test_extract_config_file_from_argv_missing_returns_none() -> None:
    assert extract_config_file_from_argv(["--port", "9001"]) is None


def test_extract_config_file_from_argv_followed_by_flag_is_ignored() -> None:
    out = extract_config_file_from_argv(["--config-file", "--reload"])
    assert out is None


def test_explicit_config_path_prefers_cli_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBB_AGENT_CONFIG", "/from-env.toml")
    out = explicit_config_path(["--config-file", "/from-cli.toml"])
    assert out == "/from-cli.toml"


def test_explicit_config_path_falls_through_env_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for v in EXPLICIT_CONFIG_ENVS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPENBB_CONFIG", "/c.toml")
    assert explicit_config_path([]) == "/c.toml"

    monkeypatch.setenv("OPENBB_AGENT_CONFIG", "/agent.toml")
    assert explicit_config_path([]) == "/agent.toml"


def test_explicit_config_path_returns_none_when_unset() -> None:
    assert explicit_config_path([]) is None


def test_expand_env_refs_resolves_braced_and_bare() -> None:
    env = {"FOO": "bar", "X": "y"}
    out, missing = expand_env_refs("a/${FOO}/$X", env=env)
    assert out == "a/bar/y"
    assert missing == []


def test_expand_env_refs_preserves_literal_dollar_for_non_idents() -> None:
    out, missing = expand_env_refs("$5 and $@ and $", env={})
    assert out == "$5 and $@ and $"
    assert missing == []


def test_expand_env_refs_reports_missing_once() -> None:
    out, missing = expand_env_refs("${A}-${A}-${B}", env={})
    assert out == "${A}-${A}-${B}"
    assert missing == ["A", "B"]


def test_expand_env_refs_uses_os_environ_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("__TEST_EXPAND__", "ok")
    out, _ = expand_env_refs("${__TEST_EXPAND__}")
    assert out == "ok"


def test_expand_in_dict_walks_lists_and_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("X", "42")
    inp = {"a": "${X}", "b": ["${X}", 1], "c": {"d": "${X}"}, "e": ("${X}",)}
    out = expand_in_dict(inp)
    assert out == {"a": "42", "b": ["42", 1], "c": {"d": "42"}, "e": ("42",)}


def test_expand_in_dict_preserves_non_strings() -> None:
    assert expand_in_dict(42) == 42
    assert expand_in_dict(None) is None
    assert expand_in_dict(True) is True


def test_apply_launcher_env_sets_unset_keys() -> None:
    target: dict[str, str] = {}
    applied = apply_launcher_env(
        {"GITHUB_TOKEN": "ghp_xxxx"},
        env=target,
    )
    assert applied == ["GITHUB_TOKEN"]
    assert target == {"GITHUB_TOKEN": "ghp_xxxx"}


def test_apply_launcher_env_no_clobber_real_env() -> None:
    target = {"GITHUB_TOKEN": "from-shell"}
    applied = apply_launcher_env(
        {"GITHUB_TOKEN": "from-toml"},
        env=target,
    )
    assert applied == []
    assert target["GITHUB_TOKEN"] == "from-shell"


def test_apply_launcher_env_expands_refs() -> None:
    target = {"HOST_TOKEN": "ghp_xxxx"}
    applied = apply_launcher_env(
        {"GITHUB_TOKEN": "${HOST_TOKEN}"},
        env=target,
    )
    assert applied == ["GITHUB_TOKEN"]
    assert target["GITHUB_TOKEN"] == "ghp_xxxx"


def test_apply_launcher_env_skips_missing_refs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target: dict[str, str] = {}
    with caplog.at_level("WARNING"):
        applied = apply_launcher_env(
            {"GITHUB_TOKEN": "${HOST_TOKEN}"},
            env=target,
        )
    assert applied == []
    assert "GITHUB_TOKEN" not in target
    assert any("HOST_TOKEN" in r.message for r in caplog.records)


def test_apply_user_settings_credentials_seeds_env(tmp_path: Path) -> None:
    """Map user_settings.json credentials to upper-cased env vars."""
    from openbb_agent_server.app import config as cfg_mod

    settings = tmp_path / "user_settings.json"
    settings.write_text(
        json.dumps(
            {
                "credentials": {
                    "groq_api_key": "gsk-from-file",
                    "nvidia_api_key": "nvapi-from-file",
                    "google_api_key": None,
                    "anthropic_api_key": "",
                    "openai_api_key": 12345,
                },
                "preferences": {},
            }
        )
    )
    env: dict[str, str] = {}
    applied = cfg_mod.apply_user_settings_credentials(
        settings_path=str(settings), env=env
    )
    assert "GROQ_API_KEY" in applied
    assert "NVIDIA_API_KEY" in applied
    assert env["GROQ_API_KEY"] == "gsk-from-file"
    assert env["NVIDIA_API_KEY"] == "nvapi-from-file"
    assert "GOOGLE_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env


def test_apply_user_settings_credentials_does_not_clobber_real_env(
    tmp_path: Path,
) -> None:
    from openbb_agent_server.app import config as cfg_mod

    settings = tmp_path / "user_settings.json"
    settings.write_text(json.dumps({"credentials": {"groq_api_key": "FROM_FILE"}}))
    env = {"GROQ_API_KEY": "FROM_SHELL"}
    applied = cfg_mod.apply_user_settings_credentials(
        settings_path=str(settings), env=env
    )
    assert applied == []
    assert env["GROQ_API_KEY"] == "FROM_SHELL"


def test_apply_user_settings_credentials_handles_missing_file(
    tmp_path: Path,
) -> None:
    from openbb_agent_server.app import config as cfg_mod

    env: dict[str, str] = {}
    applied = cfg_mod.apply_user_settings_credentials(
        settings_path=str(tmp_path / "nope.json"), env=env
    )
    assert applied == []
    assert env == {}


def test_apply_user_settings_credentials_handles_malformed_json(
    tmp_path: Path,
) -> None:
    from openbb_agent_server.app import config as cfg_mod

    settings = tmp_path / "user_settings.json"
    settings.write_text("not-json")
    env: dict[str, str] = {}
    applied = cfg_mod.apply_user_settings_credentials(
        settings_path=str(settings), env=env
    )
    assert applied == []
    assert env == {}


def test_apply_user_settings_credentials_skips_when_credentials_missing(
    tmp_path: Path,
) -> None:
    from openbb_agent_server.app import config as cfg_mod

    settings = tmp_path / "user_settings.json"
    settings.write_text(json.dumps({"preferences": {"theme": "dark"}}))
    env: dict[str, str] = {}
    applied = cfg_mod.apply_user_settings_credentials(
        settings_path=str(settings), env=env
    )
    assert applied == []
    assert env == {}


def test_apply_launcher_env_handles_none() -> None:
    assert apply_launcher_env(None) == []
    assert apply_launcher_env({}) == []


def test_apply_launcher_env_skips_non_string_keys() -> None:
    target: dict[str, str] = {}
    applied = apply_launcher_env({1: "x"}, env=target)  # type: ignore[dict-item]
    assert applied == []
    assert target == {}


def test_merge_launcher_kwargs_cli_wins() -> None:
    out = merge_launcher_kwargs(
        {"port": 1111},
        {"port": 9999, "host": "0.0.0.0"},
    )
    assert out == {"port": 1111, "host": "0.0.0.0"}


def test_merge_launcher_kwargs_handles_no_launcher() -> None:
    assert merge_launcher_kwargs({"x": 1}, None) == {"x": 1}


def test_merge_launcher_kwargs_handles_empty_launcher() -> None:
    assert merge_launcher_kwargs({"x": 1}, {}) == {"x": 1}


def _write_toml(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).strip() + "\n")


def test_bootstrap_loads_toml_and_pushes_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "openbb.toml"
    _write_toml(
        cfg_path,
        """
        [agent]
        host = "0.0.0.0"
        port = 6900
        model_provider = "fake"

        [env]
        OPENBB_TEST_BOOTSTRAP = "value-from-toml"
        """,
    )
    monkeypatch.delenv("OPENBB_TEST_BOOTSTRAP", raising=False)
    cfg = bootstrap_launcher_config(explicit_path=str(cfg_path))
    assert cfg.get("agent", {}).get("host") == "0.0.0.0"
    import os

    assert os.environ["OPENBB_TEST_BOOTSTRAP"] == "value-from-toml"


def test_bootstrap_substitutes_env_refs_in_agent_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOST_DB_URL", "postgresql+psycopg://prod-db:5432/x")
    cfg_path = tmp_path / "openbb.toml"
    _write_toml(
        cfg_path,
        """
        [agent]
        db_url = "${HOST_DB_URL}"
        model_provider = "fake"
        """,
    )
    merged = bootstrap_launcher_config(explicit_path=str(cfg_path))
    section = agent_section(merged)
    assert section["db_url"] == "postgresql+psycopg://prod-db:5432/x"


def test_bootstrap_real_shell_env_wins_over_env_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENBB_TEST_KEY", "from-shell")
    cfg_path = tmp_path / "openbb.toml"
    _write_toml(
        cfg_path,
        """
        [env]
        OPENBB_TEST_KEY = "from-toml"
        """,
    )
    bootstrap_launcher_config(explicit_path=str(cfg_path))
    import os

    assert os.environ["OPENBB_TEST_KEY"] == "from-shell"


def test_bootstrap_rejects_malformed_toml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "openbb.toml"
    cfg_path.write_text("not valid = = toml\n")
    with pytest.raises(ValueError):
        bootstrap_launcher_config(explicit_path=str(cfg_path))


def test_agent_section_returns_empty_dict_when_missing() -> None:
    assert agent_section({}) == {}
    assert agent_section({"agent": "not-a-dict"}) == {}


def test_agent_section_returns_copy_not_reference() -> None:
    cfg = {"agent": {"host": "x"}}
    section = agent_section(cfg)
    section["host"] = "y"
    assert cfg["agent"]["host"] == "x"


def test_from_toml_with_empty_section_returns_defaults() -> None:
    s = AgentServerSettings.from_toml({})
    assert s.host == "127.0.0.1"
    assert s.model_provider == "nvidia"


def test_from_toml_top_level_fields_apply() -> None:
    s = AgentServerSettings.from_toml(
        {
            "host": "0.0.0.0",
            "port": 8080,
            "model_provider": "anthropic",
            "model_name": "claude-opus-4-7",
            "tool_sources": ["artifacts", "web_search"],
            "subagents": ["researcher"],
            "middleware": ["usage_recorder"],
            "skills": ["/skills/finance"],
        }
    )
    assert s.host == "0.0.0.0"
    assert s.port == 8080
    assert s.model_provider == "anthropic"
    assert s.model_name == "claude-opus-4-7"
    assert s.tool_sources == ("artifacts", "web_search")
    assert s.subagents == ("researcher",)
    assert s.middleware == ("usage_recorder",)
    assert s.skills == ("/skills/finance",)


def test_from_toml_auth_subtable_applies() -> None:
    s = AgentServerSettings.from_toml(
        {
            "auth": {
                "backend": "oidc_jwt",
                "config": {
                    "jwks_url": "https://idp.example.com/.well-known/jwks.json",
                    "audience": "agent-server",
                },
            }
        }
    )
    assert s.auth_backend == "oidc_jwt"
    assert s.auth_config["jwks_url"].startswith("https://idp.example.com")
    assert s.auth_config["audience"] == "agent-server"


def test_from_toml_model_subtable_applies() -> None:
    s = AgentServerSettings.from_toml(
        {
            "model": {
                "provider": "nvidia",
                "name": "meta/llama-3.1-405b-instruct",
                "config": {"base_url": "https://integrate.api.nvidia.com/v1"},
            }
        }
    )
    assert s.model_provider == "nvidia"
    assert s.model_name == "meta/llama-3.1-405b-instruct"
    assert s.model_config_["base_url"].startswith("https://integrate")


def test_from_toml_metadata_subtable_applies() -> None:
    s = AgentServerSettings.from_toml(
        {
            "metadata": {
                "name": "My Custom Agent",
                "description": "Tailored for finance.",
                "image_url": "https://example.com/logo.png",
            }
        }
    )
    assert s.metadata.name == "My Custom Agent"
    assert s.metadata.description == "Tailored for finance."
    assert s.metadata.image_url == "https://example.com/logo.png"


def test_from_toml_features_merges_over_defaults() -> None:
    s = AgentServerSettings.from_toml(
        {
            "features": {
                "deep-research": {
                    "label": "Deep Research",
                    "default": True,
                    "description": "Multi-step research subagent.",
                }
            }
        }
    )
    assert s.features["deep-research"]["default"] is True
    assert "search-web" in s.features
    assert "fetch-url" in s.features
    assert s.features["streaming"] is True


def test_from_toml_data_dir_expanduser(tmp_path: Path) -> None:
    s = AgentServerSettings.from_toml({"data_dir": str(tmp_path / "agent")})
    assert s.data_dir == tmp_path / "agent"


def test_from_toml_env_var_still_wins_over_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBB_AGENT_HOST", "from-env-host")
    s = AgentServerSettings.from_toml({"host": "from-toml-host"})
    assert s.host == "from-env-host"


def test_container_style_env_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map a host env var to a container env var via the TOML env table."""
    import os

    monkeypatch.setenv("HOST_GITHUB_TOKEN", "ghp_secret_value")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    cfg_path = tmp_path / "openbb.toml"
    _write_toml(
        cfg_path,
        """
        [env]
        GITHUB_TOKEN = "${HOST_GITHUB_TOKEN}"

        [agent]
        model_provider = "fake"
        """,
    )

    bootstrap_launcher_config(explicit_path=str(cfg_path))
    assert os.environ["GITHUB_TOKEN"] == "ghp_secret_value"


def test_module_constants_are_immutable() -> None:
    assert "OPENBB_AGENT_CONFIG" in cfg_mod.EXPLICIT_CONFIG_ENVS
    assert "OPENBB_API_CONFIG" in cfg_mod.EXPLICIT_CONFIG_ENVS
    assert "OPENBB_CONFIG" in cfg_mod.EXPLICIT_CONFIG_ENVS
    assert cfg_mod.CONFIG_FILE_FLAG == "--config-file"


def test_from_toml_loads_profiles_from_subtables() -> None:
    s = AgentServerSettings.from_toml(
        {
            "model_provider": "fake",
            "profiles": {
                "equity": {
                    "metadata": {"name": "Equity Analyst"},
                    "tool_sources": ["artifacts"],
                    "system_prompt_file": "/etc/openbb/equity-prompt.md",
                },
                "research": {
                    "metadata": {"name": "Researcher"},
                    "subagents": ["researcher"],
                },
            },
        }
    )
    assert set(s.profiles) == {"equity", "research"}
    eq = s.resolve_profile("equity")
    assert eq.metadata.name == "Equity Analyst"
    assert eq.tool_sources == ("artifacts",)
    assert eq.system_prompt_file == "/etc/openbb/equity-prompt.md"


def test_resolve_profile_rejects_inline_system_prompt() -> None:
    """Reject an inline system_prompt."""
    s = AgentServerSettings(profiles={"alt": {"system_prompt": "you are an analyst"}})
    with pytest.raises(ValueError, match="system_prompt_file"):
        s.resolve_profile("alt")


def test_from_toml_rejects_inline_system_prompt_at_top_level() -> None:
    with pytest.raises(ValueError, match="system_prompt_file"):
        AgentServerSettings.from_toml(
            {"model_provider": "fake", "system_prompt": "inline forbidden"}
        )


def test_resolve_profile_falls_back_to_base_settings_for_unset_fields() -> None:
    s = AgentServerSettings(
        model_provider="fake",
        tool_sources=("artifacts", "web_search"),
        profiles={"only-name": {"metadata": {"name": "Just A Name"}}},
    )
    p = s.resolve_profile("only-name")
    assert p.tool_sources == ("artifacts", "web_search")
    assert p.metadata.name == "Just A Name"


def test_resolve_profile_default_returns_global_settings() -> None:
    s = AgentServerSettings(
        model_provider="fake",
        model_name="llama3",
        tool_sources=("artifacts",),
    )
    p = s.resolve_profile()
    assert p.name == "default"
    assert p.model_provider == "fake"
    assert p.model_name == "llama3"
    assert p.tool_sources == ("artifacts",)


def test_resolve_profile_unknown_name_raises_keyerror() -> None:
    s = AgentServerSettings(model_provider="fake")
    with pytest.raises(KeyError):
        s.resolve_profile("nope")


def test_all_profile_names_includes_default() -> None:
    s = AgentServerSettings(
        model_provider="fake",
        profiles={"a": {}, "b": {}},
    )
    names = s.all_profile_names()
    assert "default" in names
    assert "a" in names
    assert "b" in names


def test_from_toml_loads_tool_source_config() -> None:
    s = AgentServerSettings.from_toml(
        {
            "model_provider": "fake",
            "tool_source_config": {
                "mcp_local": {"config_file": "/etc/openbb/mcp.toml"},
                "web_search": {"provider": "tavily"},
            },
        }
    )
    assert s.tool_source_config["mcp_local"]["config_file"] == "/etc/openbb/mcp.toml"
    assert s.tool_source_config["web_search"]["provider"] == "tavily"


def test_resolve_profile_inherits_base_tool_source_config() -> None:
    s = AgentServerSettings(
        model_provider="fake",
        tool_source_config={"mcp_local": {"config_file": "/base.toml"}},
        profiles={"equity": {}},
    )
    p = s.resolve_profile("equity")
    assert p.tool_source_config["mcp_local"]["config_file"] == "/base.toml"


def test_resolve_profile_overlay_merges_per_tool_source() -> None:
    s = AgentServerSettings(
        model_provider="fake",
        tool_source_config={
            "mcp_local": {"config_file": "/base.toml", "command": "openbb-mcp"}
        },
        profiles={
            "equity": {
                "tool_source_config": {"mcp_local": {"config_file": "/equity.toml"}}
            }
        },
    )
    p = s.resolve_profile("equity")
    assert p.tool_source_config["mcp_local"]["config_file"] == "/equity.toml"
    assert p.tool_source_config["mcp_local"]["command"] == "openbb-mcp"


def test_default_profile_resolves_with_empty_tool_source_config() -> None:
    s = AgentServerSettings(model_provider="fake")
    p = s.resolve_profile()
    assert p.tool_source_config == {}


def test_load_preset_default_returns_a_dict() -> None:
    """Parse the bundled default preset."""
    from openbb_agent_server.app.config import load_preset

    out = load_preset("default")
    assert isinstance(out, dict)
    assert out


def test_load_preset_unknown_name_raises() -> None:
    from openbb_agent_server.app.config import load_preset

    with pytest.raises(ValueError, match="unknown preset"):
        load_preset("totally-fake-preset")


# ---------------------------------------------------------------------------
# Vendored layered TOML cascade (formerly delegated to openbb-core).
# ---------------------------------------------------------------------------


def _isolate_user_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the user-global lookup away from the real ~/.openbb_platform."""
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", tmp_path / "no_user_dir")


def test_load_config_returns_empty_when_no_layers_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    _isolate_user_dir(monkeypatch, tmp_path)
    assert cfg_mod.load_config() == {}


def test_load_config_reads_project_local_openbb_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_toml(
        tmp_path / "openbb.toml",
        """
        [agent]
        model_provider = "fake"

        [system]
        debug_mode = true
        """,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    _isolate_user_dir(monkeypatch, tmp_path)
    out = cfg_mod.load_config()
    assert out["agent"]["model_provider"] == "fake"
    assert out["system"]["debug_mode"] is True


def test_load_config_reads_pyproject_tool_openbb_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_toml(
        tmp_path / "pyproject.toml",
        """
        [tool.openbb.agent]
        port = 6900
        """,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    _isolate_user_dir(monkeypatch, tmp_path)
    out = cfg_mod.load_config()
    assert out["agent"]["port"] == 6900


def test_load_config_explicit_path_wins_over_project_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_toml(tmp_path / "openbb.toml", '[agent]\nhost = "from-project"')
    explicit = tmp_path / "explicit.toml"
    _write_toml(explicit, '[agent]\nhost = "from-explicit"')
    monkeypatch.chdir(tmp_path)
    _isolate_user_dir(monkeypatch, tmp_path)
    out = cfg_mod.load_config(explicit_path=str(explicit))
    assert out["agent"]["host"] == "from-explicit"


def test_load_config_env_var_acts_as_explicit_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "from_env.toml"
    _write_toml(explicit, "[agent]\nport = 7777")
    monkeypatch.setenv("OPENBB_CONFIG", str(explicit))
    monkeypatch.chdir(tmp_path)
    _isolate_user_dir(monkeypatch, tmp_path)
    out = cfg_mod.load_config()
    assert out["agent"]["port"] == 7777


def test_load_config_user_global_layer_picked_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    user_dir = tmp_path / "user_openbb"
    user_dir.mkdir()
    _write_toml(user_dir / "openbb.toml", '[agent]\nhost = "from-user-global"')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", user_dir)
    out = cfg_mod.load_config()
    assert out["agent"]["host"] == "from-user-global"


def test_load_config_project_overrides_user_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    user_dir = tmp_path / "user_openbb"
    user_dir.mkdir()
    _write_toml(user_dir / "openbb.toml", '[agent]\nhost = "from-user-global"')
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_toml(project_dir / "openbb.toml", '[agent]\nhost = "from-project"')
    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", user_dir)
    out = cfg_mod.load_config()
    assert out["agent"]["host"] == "from-project"


def test_load_config_layers_deep_merge_nested_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A higher layer overrides one nested key without dropping siblings."""
    _write_toml(
        tmp_path / "openbb.toml",
        """
        [agent]
        host = "from-project"
        port = 6900
        """,
    )
    explicit = tmp_path / "explicit.toml"
    _write_toml(explicit, "[agent]\nport = 7000")
    monkeypatch.chdir(tmp_path)
    _isolate_user_dir(monkeypatch, tmp_path)
    out = cfg_mod.load_config(explicit_path=str(explicit))
    assert out["agent"]["host"] == "from-project"
    assert out["agent"]["port"] == 7000


def test_load_config_normalizes_kebab_case_top_level_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_toml(tmp_path / "openbb.toml", "debug-mode = true")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    _isolate_user_dir(monkeypatch, tmp_path)
    out = cfg_mod.load_config()
    assert out["debug_mode"] is True


def test_load_config_malformed_cascade_file_is_empty_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken TOML discovered in the cascade is skipped, not fatal.

    Contrast with an explicit ``--config-file`` path, which
    ``_validate_explicit_toml`` rejects loudly at bootstrap.
    """
    (tmp_path / "openbb.toml").write_text("not [valid toml at all")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    _isolate_user_dir(monkeypatch, tmp_path)
    assert cfg_mod.load_config() == {}


def test_load_config_explicit_path_missing_is_no_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _isolate_user_dir(monkeypatch, tmp_path)
    out = cfg_mod.load_config(explicit_path=str(tmp_path / "missing.toml"))
    assert out == {}


def test_load_config_pyproject_without_tool_openbb_contributes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_toml(tmp_path / "pyproject.toml", '[tool.poetry]\nname = "x"')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENBB_CONFIG", raising=False)
    _isolate_user_dir(monkeypatch, tmp_path)
    assert cfg_mod.load_config() == {}


def test_load_env_files_loads_user_global_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    user_dir = tmp_path / "user_openbb"
    user_dir.mkdir()
    (user_dir / ".env").write_text("FROM_USER_GLOBAL_DOTENV=value-1\n")
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", user_dir)
    monkeypatch.delenv("OPENBB_ENV_FILE", raising=False)
    monkeypatch.delenv("FROM_USER_GLOBAL_DOTENV", raising=False)
    try:
        loaded = cfg_mod.load_env_files()
        assert any(p.name == ".env" for p in loaded)
        assert os.environ["FROM_USER_GLOBAL_DOTENV"] == "value-1"
    finally:
        os.environ.pop("FROM_USER_GLOBAL_DOTENV", None)


def test_load_env_files_skips_bare_keys_without_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    user_dir = tmp_path / "user_openbb"
    user_dir.mkdir()
    (user_dir / ".env").write_text("BARE_KEY_NO_VALUE\nREAL_DOTENV_KEY=v\n")
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", user_dir)
    monkeypatch.delenv("BARE_KEY_NO_VALUE", raising=False)
    monkeypatch.delenv("REAL_DOTENV_KEY", raising=False)
    try:
        cfg_mod.load_env_files()
        assert "BARE_KEY_NO_VALUE" not in os.environ
        assert os.environ["REAL_DOTENV_KEY"] == "v"
    finally:
        os.environ.pop("REAL_DOTENV_KEY", None)


def test_load_env_files_no_files_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_user_dir(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENBB_ENV_FILE", raising=False)
    assert cfg_mod.load_env_files() == []


def test_apply_settings_to_env_seeds_promoted_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.delenv("OPENBB_DEBUG_MODE", raising=False)
    try:
        applied = cfg_mod.apply_settings_to_env({"debug_mode": True})
        assert "OPENBB_DEBUG_MODE" in applied
        assert os.environ["OPENBB_DEBUG_MODE"] == "True"
    finally:
        os.environ.pop("OPENBB_DEBUG_MODE", None)


def test_apply_settings_to_env_existing_env_var_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.setenv("OPENBB_DEBUG_MODE", "False")
    applied = cfg_mod.apply_settings_to_env({"debug_mode": True})
    assert "OPENBB_DEBUG_MODE" not in applied
    assert os.environ["OPENBB_DEBUG_MODE"] == "False"


def test_apply_settings_to_env_skips_non_promoted_section_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.delenv("OPENBB_LOGGING_VERBOSITY", raising=False)
    monkeypatch.delenv("OPENBB_DEBUG_MODE", raising=False)
    try:
        applied = cfg_mod.apply_settings_to_env(
            {"system": {"logging_verbosity": 30, "debug_mode": True}}
        )
        assert applied == ["OPENBB_DEBUG_MODE"]
        assert "OPENBB_LOGGING_VERBOSITY" not in os.environ
    finally:
        os.environ.pop("OPENBB_DEBUG_MODE", None)


def test_apply_settings_to_env_handles_empty_and_malformed() -> None:
    assert cfg_mod.apply_settings_to_env(None) == []
    assert cfg_mod.apply_settings_to_env({}) == []
    assert cfg_mod.apply_settings_to_env({"system": "not-a-table"}) == []


def test_apply_settings_to_env_renders_non_bool_values_as_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.delenv("OPENBB_LOGGING_SUPPRESS", raising=False)
    try:
        applied = cfg_mod.apply_settings_to_env({"logging_suppress": "verbose"})
        assert applied == ["OPENBB_LOGGING_SUPPRESS"]
        assert os.environ["OPENBB_LOGGING_SUPPRESS"] == "verbose"
    finally:
        os.environ.pop("OPENBB_LOGGING_SUPPRESS", None)


def test_promote_top_level_keys_empty_config_short_circuits() -> None:
    assert cfg_mod._promote_top_level_keys({}) == {}


def test_load_env_files_explicit_path_layered_after_user_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    _isolate_user_dir(monkeypatch, tmp_path)
    explicit = tmp_path / "explicit.env"
    explicit.write_text("FROM_EXPLICIT_DOTENV=yes\n")
    monkeypatch.delenv("OPENBB_ENV_FILE", raising=False)
    monkeypatch.delenv("FROM_EXPLICIT_DOTENV", raising=False)
    try:
        loaded = cfg_mod.load_env_files(explicit_path=str(explicit))
        assert [p.name for p in loaded] == ["explicit.env"]
        assert os.environ["FROM_EXPLICIT_DOTENV"] == "yes"
    finally:
        os.environ.pop("FROM_EXPLICIT_DOTENV", None)


def test_bootstrap_does_not_require_openbb_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The full bootstrap runs with openbb-core absent from the env.

    ``[system]`` / ``[user]`` tables still parse and merge; the push
    onto openbb-core's services is silently skipped.
    """
    import importlib.util
    import os

    if importlib.util.find_spec("openbb_core") is not None:
        pytest.skip("openbb-core installed in this environment")

    # Keep the real ~/.openbb_platform/.env out of the assertion below.
    _isolate_user_dir(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENBB_ENV_FILE", raising=False)

    cfg_path = tmp_path / "openbb.toml"
    _write_toml(
        cfg_path,
        """
        [agent]
        model_provider = "fake"

        [system]
        debug_mode = true

        [user.credentials]
        fmp_api_key = "abc"
        """,
    )
    monkeypatch.delenv("OPENBB_DEBUG_MODE", raising=False)
    try:
        cfg = bootstrap_launcher_config(explicit_path=str(cfg_path))
        assert cfg["agent"]["model_provider"] == "fake"
        assert cfg["system"]["debug_mode"] is True
        assert cfg["user"]["credentials"]["fmp_api_key"] == "abc"
        # Promoted system flags still seed OPENBB_* env vars.
        assert os.environ["OPENBB_DEBUG_MODE"] == "True"
    finally:
        os.environ.pop("OPENBB_DEBUG_MODE", None)


def test_service_push_delegates_to_openbb_core_when_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When openbb-core IS importable, [system]/[user] are handed to it."""
    import sys
    import types

    calls: list[dict] = []
    loader_mod = types.ModuleType("openbb_core.app.config.loader")
    loader_mod.apply_config_to_services = calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openbb_core", types.ModuleType("openbb_core"))
    monkeypatch.setitem(
        sys.modules, "openbb_core.app", types.ModuleType("openbb_core.app")
    )
    monkeypatch.setitem(
        sys.modules,
        "openbb_core.app.config",
        types.ModuleType("openbb_core.app.config"),
    )
    monkeypatch.setitem(sys.modules, "openbb_core.app.config.loader", loader_mod)

    cfg = {"system": {"debug_mode": True}}
    cfg_mod._apply_config_to_openbb_services(cfg)
    assert calls == [cfg]


def test_service_push_skips_empty_config() -> None:
    cfg_mod._apply_config_to_openbb_services(None)
    cfg_mod._apply_config_to_openbb_services({})
