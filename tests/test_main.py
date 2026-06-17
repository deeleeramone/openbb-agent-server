"""CLI entry-point tests."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from openbb_agent_server.main import _build_parser, main
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore


def test_help_short_circuits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_parser_accepts_host_port_auth_reload() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--auth",
            "bearer_static",
            "--reload",
            "--log-level",
            "debug",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.auth == "bearer_static"
    assert args.reload is True
    assert args.log_level == "debug"


def test_main_propagates_auth_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        captured["called"] = True

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BEARER", "x")
    monkeypatch.setattr("uvicorn.run", fake_run)
    main(["--auth", "bearer_static", "--port", "0"])
    assert captured.get("called") is True


def test_main_reload_path_invokes_uvicorn_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        seen_kwargs.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    main(["--reload", "--port", "0"])
    assert seen_kwargs.get("reload") is True
    assert seen_kwargs.get("factory") is True


def _setup_keys_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import json

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "api_key_table")
    monkeypatch.setenv(
        "OPENBB_AGENT_AUTH_CONFIG",
        json.dumps({"db_url": f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}"}),
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    import asyncio

    from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore

    async def _init():
        store = SqliteHistoryStore(f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
        await store.init_schema()
        await store.aclose()

    asyncio.run(_init())


def test_keys_issue_prints_human_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    _setup_keys_env(monkeypatch, tmp_path)
    main(["keys", "issue", "--user-id", "alice", "--label", "laptop"])
    out = capsys.readouterr().out
    assert "key_id :" in out
    assert "alice" in out
    assert "oba_" in out
    assert "laptop" in out


def test_keys_issue_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    import json

    _setup_keys_env(monkeypatch, tmp_path)
    main(["keys", "issue", "--user-id", "bob", "--json"])
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["user_id"] == "bob"
    assert payload["key_id"]
    assert payload["key"].startswith("oba_")


def test_keys_issue_then_list(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    _setup_keys_env(monkeypatch, tmp_path)
    main(["keys", "issue", "--user-id", "alice", "--json"])
    capsys.readouterr()
    main(["keys", "issue", "--user-id", "bob", "--json"])
    capsys.readouterr()

    main(["keys", "list"])
    listed = capsys.readouterr().out
    assert "alice" in listed
    assert "bob" in listed
    assert "oba_" not in listed


def test_keys_revoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    import json

    _setup_keys_env(monkeypatch, tmp_path)
    main(["keys", "issue", "--user-id", "alice", "--json"])
    issued = json.loads(capsys.readouterr().out.strip())

    main(["keys", "revoke", "--key-id", issued["key_id"]])
    out = capsys.readouterr().out
    assert "revoked" in out

    main(["keys", "list", "--json"])
    rows = json.loads(capsys.readouterr().out.strip())
    [row] = [r for r in rows if r["key_id"] == issued["key_id"]]
    assert row["revoked_at"] is not None


def test_keys_revoke_unknown_id_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _setup_keys_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["keys", "revoke", "--key-id", "does-not-exist"])
    assert exc.value.code == 1


def test_keys_list_filtered_by_user(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    _setup_keys_env(monkeypatch, tmp_path)
    main(["keys", "issue", "--user-id", "alice", "--json"])
    capsys.readouterr()
    main(["keys", "issue", "--user-id", "bob", "--json"])
    capsys.readouterr()

    main(["keys", "list", "--user-id", "alice"])
    out = capsys.readouterr().out
    assert "alice" in out
    assert "bob" not in out


def test_generate_config_prints_template_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--generate-config"])
    out = capsys.readouterr().out
    assert "openbb.toml" in out
    assert "[agent]" in out


def test_read_config_template_unknown_preset_raises() -> None:
    """Bail with a helpful message for an unknown preset."""
    from openbb_agent_server.main import _read_config_template

    with pytest.raises(SystemExit, match="unknown preset"):
        _read_config_template("does-not-exist")


def test_generate_config_writes_file(tmp_path) -> None:
    target = tmp_path / "subdir" / "openbb.toml"
    main(["--generate-config", str(target)])
    body = target.read_text(encoding="utf-8")
    assert "[agent]" in body
    assert "[agent.model]" in body


def test_generate_config_refuses_to_overwrite(tmp_path) -> None:
    target = tmp_path / "openbb.toml"
    target.write_text("# existing", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["--generate-config", str(target)])
    assert "refusing to overwrite" in str(exc.value)
    assert target.read_text(encoding="utf-8") == "# existing"


def test_prune_command_with_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    main(
        ["prune", "--keep-last", "2", "--checkpoint-days", "7", "--history-days", "30"]
    )
    out = capsys.readouterr().out
    assert "prune complete" in out


def test_prune_command_uses_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    main(["prune", "--no-vacuum"])
    out = capsys.readouterr().out
    assert "prune complete" in out


def test_main_propagates_model_provider_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        captured["called"] = True
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    main(
        [
            "--host",
            "127.0.0.1",
            "--model-provider",
            "fake",
            "--model-name",
            "fake-model",
            "--port",
            "0",
        ]
    )
    assert os.environ.get("OPENBB_AGENT_MODEL_PROVIDER") == "fake"
    assert os.environ.get("OPENBB_AGENT_MODEL_NAME") == "fake-model"


def test_main_keys_list_prints_no_keys_message_when_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "api_key_table")
    monkeypatch.setenv(
        "OPENBB_AGENT_AUTH_CONFIG",
        json.dumps({"db_url": f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}"}),
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))

    async def _init() -> None:
        store = SqliteHistoryStore(f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
        await store.init_schema()
        await store.aclose()

    asyncio.run(_init())

    main(["keys", "list"])
    out = capsys.readouterr().out
    assert "(no keys)" in out
