"""mcp_local + mcp_http tool source tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from openbb_agent_server.plugins.tools.mcp_http import HttpMcpToolSource
from openbb_agent_server.plugins.tools.mcp_local import LocalMcpToolSource
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(api_keys: dict[str, str] | None = None) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=api_keys or {},
    )


def test_mcp_local_default_command_is_openbb_mcp() -> None:
    src = LocalMcpToolSource()
    assert src._command == "openbb-mcp"  # noqa: SLF001 — internal contract test
    assert src._args == ["--transport", "stdio"]  # noqa: SLF001


def test_mcp_local_explicit_command_args_env_persist() -> None:
    src = LocalMcpToolSource(
        command="/opt/custom-mcp",
        args=["--allowed-categories", "equity"],
        env={"FOO": "bar"},
    )
    assert src._command == "/opt/custom-mcp"  # noqa: SLF001
    assert src._args == ["--allowed-categories", "equity"]  # noqa: SLF001
    assert src._env == {"FOO": "bar"}  # noqa: SLF001


def test_mcp_local_explicit_empty_args_is_respected() -> None:
    """Respect an explicit args=[] opting out of the stdio default."""
    src = LocalMcpToolSource(args=[])
    assert src._args == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_mcp_local_raises_when_command_missing() -> None:
    src = LocalMcpToolSource(command="/this/binary/does/not/exist")
    with pytest.raises(RuntimeError):
        await src.tools(_ctx(), {})


def test_mcp_local_resolves_command_from_venv_bin_when_path_lacks_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Resolve the command from the venv bin when PATH lacks it."""
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_interpreter = bin_dir / "python"
    fake_interpreter.write_text("#!/bin/sh\nexit 0\n")
    fake_interpreter.chmod(0o755)
    fake_mcp = bin_dir / "openbb-mcp"
    fake_mcp.write_text("#!/bin/sh\nexit 0\n")
    fake_mcp.chmod(0o755)

    monkeypatch.setattr(sys, "executable", str(fake_interpreter))
    monkeypatch.setenv("PATH", "/nowhere")

    from openbb_agent_server.plugins.tools.mcp_local import _resolve_command

    assert _resolve_command("openbb-mcp") == str(fake_mcp)


def test_mcp_local_returns_none_when_command_unfindable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/nowhere")
    from openbb_agent_server.plugins.tools.mcp_local import _resolve_command

    assert _resolve_command("definitely-not-installed-cli") is None


def test_mcp_local_absolute_path_must_exist() -> None:
    from openbb_agent_server.plugins.tools.mcp_local import _resolve_command

    assert _resolve_command("/this/binary/does/not/exist") is None


def test_resolve_config_file_uses_explicit_kwarg() -> None:
    from openbb_agent_server.plugins.tools.mcp_local import _resolve_config_file

    assert _resolve_config_file("/some/explicit.toml") == "/some/explicit.toml"


def test_resolve_config_file_env_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    from openbb_agent_server.plugins.tools.mcp_local import _resolve_config_file

    monkeypatch.setenv("OPENBB_CONFIG", "/c.toml")
    assert _resolve_config_file(None) == "/c.toml"
    monkeypatch.setenv("OPENBB_AGENT_CONFIG", "/agent.toml")
    assert _resolve_config_file(None) == "/agent.toml"
    monkeypatch.setenv("OPENBB_AGENT_MCP_CONFIG", "/mcp.toml")
    assert _resolve_config_file(None) == "/mcp.toml"


def test_resolve_config_file_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.mcp_local import (
        _CONFIG_FILE_ENVS,
        _resolve_config_file,
    )

    for var in _CONFIG_FILE_ENVS:
        monkeypatch.delenv(var, raising=False)
    assert _resolve_config_file(None) is None


def test_read_mcp_table_picks_up_section_from_toml(tmp_path) -> None:
    import textwrap

    from openbb_agent_server.plugins.tools.mcp_local import _read_mcp_table

    cfg = tmp_path / "openbb.toml"
    cfg.write_text(
        textwrap.dedent(
            """
            [mcp]
            transport = "streamable-http"
            allowed-categories = ["equity", "economy"]

            [mcp.spec]
            path = "/var/spec.yaml"
            """
        ).strip()
        + "\n"
    )
    section = _read_mcp_table(str(cfg))
    assert section["transport"] == "streamable-http"
    assert section["allowed-categories"] == ["equity", "economy"]
    assert section["spec"]["path"] == "/var/spec.yaml"


def test_read_mcp_table_returns_empty_when_no_mcp_section(tmp_path) -> None:
    from openbb_agent_server.plugins.tools.mcp_local import _read_mcp_table

    cfg = tmp_path / "openbb.toml"
    cfg.write_text("[agent]\nport = 6900\n")
    assert _read_mcp_table(str(cfg)) == {}


def test_read_mcp_table_swallows_invalid_path() -> None:
    from openbb_agent_server.plugins.tools.mcp_local import _read_mcp_table

    assert _read_mcp_table("/no/such/file.toml") == {}


@pytest.mark.asyncio
async def test_subprocess_args_force_stdio_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Include --transport stdio in the spawn args even with no [mcp] config."""
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local.MultiServerMCPClient",
        _FakeClient,
    )
    fake_bin = tmp_path / "openbb-mcp"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local._resolve_command",
        lambda c: str(fake_bin),
    )

    src = LocalMcpToolSource()
    await src.tools(_ctx(), {})
    args = captured["connections"]["openbb"]["args"]
    assert "--transport" in args
    assert args[args.index("--transport") + 1] == "stdio"


@pytest.mark.asyncio
async def test_subprocess_args_forward_config_file_from_constructor(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    fake_bin = tmp_path / "openbb-mcp"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    cfg = tmp_path / "openbb.toml"
    cfg.write_text("[mcp]\ntransport = 'stdio'\n")

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local.MultiServerMCPClient",
        _FakeClient,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local._resolve_command",
        lambda c: str(fake_bin),
    )

    src = LocalMcpToolSource(config_file=str(cfg))
    await src.tools(_ctx(), {})
    args = captured["connections"]["openbb"]["args"]
    assert "--config-file" in args
    assert args[args.index("--config-file") + 1] == str(cfg)


@pytest.mark.asyncio
async def test_subprocess_args_forward_config_file_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    fake_bin = tmp_path / "openbb-mcp"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    cfg = tmp_path / "from-env.toml"
    cfg.write_text("[mcp]\ntransport = 'stdio'\n")

    monkeypatch.setenv("OPENBB_AGENT_MCP_CONFIG", str(cfg))
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local.MultiServerMCPClient",
        _FakeClient,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local._resolve_command",
        lambda c: str(fake_bin),
    )

    src = LocalMcpToolSource()
    await src.tools(_ctx(), {})
    args = captured["connections"]["openbb"]["args"]
    assert args[args.index("--config-file") + 1] == str(cfg)


@pytest.mark.asyncio
async def test_subprocess_args_per_call_config_overrides_constructor(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    fake_bin = tmp_path / "openbb-mcp"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    constructor_cfg = tmp_path / "ctor.toml"
    constructor_cfg.write_text("[mcp]\ntransport = 'stdio'\n")
    runtime_cfg = tmp_path / "runtime.toml"
    runtime_cfg.write_text("[mcp]\ntransport = 'stdio'\n")

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local.MultiServerMCPClient",
        _FakeClient,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local._resolve_command",
        lambda c: str(fake_bin),
    )

    src = LocalMcpToolSource(config_file=str(constructor_cfg))
    await src.tools(_ctx(), {"config_file": str(runtime_cfg)})
    args = captured["connections"]["openbb"]["args"]
    assert args[args.index("--config-file") + 1] == str(runtime_cfg)


def test_mcp_http_rejects_unsupported_transport() -> None:
    with pytest.raises(ValueError):
        HttpMcpToolSource(url="http://x", transport="carrier-pigeon")


def test_mcp_http_accepts_supported_transports() -> None:
    for t in ("streamable_http", "sse", "websocket"):
        HttpMcpToolSource(url="http://x", transport=t)


def test_mcp_http_accepts_dash_form_transport() -> None:
    """Translate the dash form 'streamable-http' from openbb-mcp's TOML."""
    src = HttpMcpToolSource(url="http://x", transport="streamable-http")
    assert src._transport == "streamable_http"  # noqa: SLF001


def test_mcp_http_constructor_stashes_headers() -> None:
    src = HttpMcpToolSource(
        url="http://x",
        headers={"X-Static": "v"},
    )
    assert src._headers == {"X-Static": "v"}  # noqa: SLF001


def test_mcp_http_build_url_streamable_http() -> None:
    from openbb_agent_server.plugins.tools.mcp_http import _build_url

    assert _build_url("openbb-mcp.local", 8001, "streamable_http") == (
        "http://openbb-mcp.local:8001/mcp"
    )


def test_mcp_http_build_url_sse() -> None:
    from openbb_agent_server.plugins.tools.mcp_http import _build_url

    assert _build_url("h", 9000, "sse") == "http://h:9000/sse"


def test_mcp_http_build_url_with_explicit_scheme() -> None:
    from openbb_agent_server.plugins.tools.mcp_http import _build_url

    assert _build_url("https://mcp.example.com", 443, "streamable_http") == (
        "https://mcp.example.com:443/mcp"
    )


def test_mcp_http_build_url_scheme_without_double_port() -> None:
    """Skip appending a port when the URL already has one."""
    from openbb_agent_server.plugins.tools.mcp_http import _build_url

    assert _build_url("https://mcp.example.com:8443", 9000, "streamable_http") == (
        "https://mcp.example.com:8443/mcp"
    )


@pytest.mark.asyncio
async def test_mcp_http_composes_url_from_mcp_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Build a URL from [mcp].host/port when url= isn't given."""
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    cfg = tmp_path / "openbb.toml"
    cfg.write_text(
        '[mcp]\nhost = "openbb.internal"\nport = 8123\ntransport = "streamable-http"\n'
    )
    monkeypatch.setenv("OPENBB_AGENT_MCP_CONFIG", str(cfg))
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_http.MultiServerMCPClient",
        _FakeClient,
    )

    src = HttpMcpToolSource()
    await src.tools(_ctx(), {})
    conn = captured["connections"]["openbb"]
    assert conn["url"] == "http://openbb.internal:8123/mcp"
    assert conn["transport"] == "streamable_http"


@pytest.mark.asyncio
async def test_mcp_http_explicit_url_wins_over_mcp_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    cfg = tmp_path / "openbb.toml"
    cfg.write_text('[mcp]\nhost = "openbb.internal"\nport = 8123\n')
    monkeypatch.setenv("OPENBB_AGENT_MCP_CONFIG", str(cfg))
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_http.MultiServerMCPClient",
        _FakeClient,
    )

    src = HttpMcpToolSource(url="https://override.example.com/mcp")
    await src.tools(_ctx(), {})
    assert captured["connections"]["openbb"]["url"] == (
        "https://override.example.com/mcp"
    )


@pytest.mark.asyncio
async def test_mcp_http_forwards_spec_headers_from_mcp_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    monkeypatch.setenv("HOST_MCP_TOKEN", "ghp_fake_token")
    cfg = tmp_path / "openbb.toml"
    cfg.write_text(
        '[mcp]\nhost = "h"\nport = 8001\n\n'
        "[mcp.spec.headers]\n"
        'Authorization = "Bearer ${HOST_MCP_TOKEN}"\n'
    )
    monkeypatch.setenv("OPENBB_AGENT_MCP_CONFIG", str(cfg))
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_http.MultiServerMCPClient",
        _FakeClient,
    )

    src = HttpMcpToolSource()
    await src.tools(_ctx(), {})
    headers = captured["connections"]["openbb"]["headers"]
    assert headers["Authorization"] == "Bearer ghp_fake_token"


@pytest.mark.asyncio
async def test_mcp_http_forwards_ctx_api_keys_as_x_openbb_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, connections: dict) -> None:
            captured["connections"] = connections

        async def get_tools(self) -> list:
            return []

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_http.MultiServerMCPClient",
        _FakeClient,
    )

    src = HttpMcpToolSource(url="http://x/mcp")
    await src.tools(_ctx(api_keys={"FMP_API_KEY": "abc", "POLYGON_API_KEY": "xyz"}), {})
    headers = captured["connections"]["openbb"]["headers"]
    assert headers["X-OPENBB-FMP_API_KEY"] == "abc"
    assert headers["X-OPENBB-POLYGON_API_KEY"] == "xyz"


@pytest.mark.asyncio
async def test_mcp_http_raises_when_no_url_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear error when no url= and no [mcp] in the cascade."""
    from openbb_agent_server.plugins.tools.mcp_http import _CONFIG_FILE_ENVS

    for var in _CONFIG_FILE_ENVS:
        monkeypatch.delenv(var, raising=False)

    src = HttpMcpToolSource()
    with pytest.raises(RuntimeError, match="no URL configured"):
        await src.tools(_ctx(), {})


def test_mcp_http_resolve_config_file_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.mcp_http import (
        _CONFIG_FILE_ENVS,
        _resolve_config_file,
    )

    for var in _CONFIG_FILE_ENVS:
        monkeypatch.delenv(var, raising=False)
    assert _resolve_config_file(None) is None
    monkeypatch.setenv("OPENBB_CONFIG", "/c.toml")
    assert _resolve_config_file(None) == "/c.toml"
    monkeypatch.setenv("OPENBB_AGENT_MCP_CONFIG", "/mcp.toml")
    assert _resolve_config_file(None) == "/mcp.toml"
    assert _resolve_config_file("/explicit.toml") == "/explicit.toml"


@pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION") != "1",
    reason="set RUN_MCP_INTEGRATION=1 with a real openbb-mcp on PATH",
)
@pytest.mark.asyncio
async def test_mcp_local_real_round_trip() -> None:  # pragma: no cover — gated
    src = LocalMcpToolSource()
    tools = await src.tools(_ctx(), {})
    assert len(tools) > 0


def test_mcp_http_read_mcp_table_swallows_bootstrap_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import mcp_http

    def raiser(*_a, **_k):
        raise RuntimeError("simulated cascade failure")

    monkeypatch.setattr(
        "openbb_agent_server.app.config.bootstrap_launcher_config", raiser
    )
    assert mcp_http._read_mcp_table("anywhere.toml") == {}


def test_mcp_local_resolve_command_via_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openbb_agent_server.plugins.tools import mcp_local

    fake = tmp_path / "openbb-mcp"
    fake.write_text("#!/bin/sh\necho hi", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local.shutil.which",
        lambda name: str(fake),
    )
    assert mcp_local._resolve_command("openbb-mcp") == str(fake)


def test_mcp_local_ensure_arg_skips_when_already_present() -> None:
    from openbb_agent_server.plugins.tools.mcp_local import _ensure_arg

    args = ["--config-file", "/x.toml"]
    assert _ensure_arg(args, "--config-file", "/y.toml") == args


def test_mcp_local_read_mcp_table_swallows_bootstrap_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import mcp_local

    def raiser(*_a, **_k):
        raise RuntimeError("simulated cascade failure")

    monkeypatch.setattr(
        "openbb_agent_server.app.config.bootstrap_launcher_config", raiser
    )
    assert mcp_local._read_mcp_table("anywhere.toml") == {}


@pytest.mark.asyncio
async def test_mcp_local_skips_args_extension_when_transport_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip args extension when transport is already set."""
    from openbb_agent_server.plugins.tools import mcp_local

    captured: dict[str, Any] = {}

    class _FakeMCP:
        def __init__(self, *, connections: dict[str, Any]) -> None:
            captured.update(connections)

        async def get_tools(self) -> list[Any]:
            return []

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local.MultiServerMCPClient",
        _FakeMCP,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local._resolve_command",
        lambda cmd: "/usr/bin/openbb-mcp",
    )

    src = mcp_local.LocalMcpToolSource(args=["--transport", "stdio", "--quiet"])
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )
    await src.tools(ctx, {})
    args = captured["openbb"]["args"]
    assert args.count("--transport") == 1


@pytest.mark.asyncio
async def test_mcp_http_per_call_config_headers_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import mcp_http

    captured: dict[str, Any] = {}

    class _FakeMCP:
        def __init__(self, *, connections: dict[str, Any]) -> None:
            captured.update(connections)

        async def get_tools(self) -> list[Any]:
            return []

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_http.MultiServerMCPClient",
        _FakeMCP,
    )

    src = mcp_http.HttpMcpToolSource(url="http://x/mcp")
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )
    await src.tools(ctx, {"headers": {"X-Tenant": "abc"}})
    assert captured["openbb"]["headers"]["X-Tenant"] == "abc"


@pytest.mark.asyncio
async def test_mcp_local_appends_transport_when_caller_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import mcp_local

    captured: dict[str, Any] = {}

    class _FakeMCP:
        def __init__(self, *, connections: dict[str, Any]) -> None:
            captured.update(connections)

        async def get_tools(self) -> list[Any]:
            return []

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local.MultiServerMCPClient",
        _FakeMCP,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.mcp_local._resolve_command",
        lambda cmd: "/usr/bin/openbb-mcp",
    )

    src = mcp_local.LocalMcpToolSource(args=["--quiet"])
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )
    await src.tools(ctx, {})
    assert captured["openbb"]["args"][-2:] == ["--transport", "stdio"]
