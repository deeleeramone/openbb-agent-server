"""``openbb-agent-server`` CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from openbb_agent_server.app.config import (
    CONFIG_FILE_FLAG,
    agent_section,
    bootstrap_launcher_config,
    extract_config_file_from_argv,
    merge_launcher_kwargs,
)

CONFIG_TEMPLATE_RESOURCE = "openbb.toml.example"

_PRESETS: dict[str, str] = {
    "default": "openbb.toml.example",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openbb-agent-server",
        description=(
            "Pluggable, multi-tenant agent backend that speaks the OpenBB "
            "Workspace custom-agent protocol over a DeepAgents harness."
        ),
    )
    p.add_argument(
        CONFIG_FILE_FLAG,
        dest="config_file",
        default=None,
        help=(
            "Path to an explicit openbb.toml. Overrides the user-global / "
            "project layers. Env: OPENBB_AGENT_CONFIG / OPENBB_API_CONFIG / "
            "OPENBB_CONFIG."
        ),
    )
    p.add_argument(
        "--generate-config",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help=(
            "Write the openbb.toml template to PATH and exit. With no "
            "PATH, prints to stdout. Use ``--preset`` to pick which "
            "template body."
        ),
    )
    p.add_argument(
        "--preset",
        choices=sorted(_PRESETS.keys()),
        default="default",
        help=(
            "Which template ``--generate-config`` writes. The bundled "
            "``default`` preset is the end-to-end NVIDIA NIM stack: "
            "seven agents (Nemotron / Mistral / Qwen3-Coder / Seed-OSS "
            "/ Step-Flash / MiniMax / Gemma-3n transcribe) on a shared "
            "NIM-only toolset (embeddings, reranker, translator, "
            "vision, audio)."
        ),
    )

    sub = p.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the agent server (default).")
    serve.add_argument("--host", default=None, help="Bind host.")
    serve.add_argument("--port", type=int, default=None, help="Bind port.")
    serve.add_argument("--auth", default=None, help="Auth backend name.")
    serve.add_argument("--model-provider", default=None, help="Model provider.")
    serve.add_argument("--model-name", default=None, help="Model name.")
    serve.add_argument("--reload", action="store_true", help="Hot-reload (dev).")
    serve.add_argument("--log-level", default="info")

    keys = sub.add_parser("keys", help="Manage api_key_table API keys.")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)

    issue = keys_sub.add_parser("issue", help="Mint a new API key.")
    issue.add_argument("--user-id", required=True)
    issue.add_argument(
        "--scope",
        action="append",
        default=None,
        help="Repeatable. Defaults to agent:query + memory:read.",
    )
    issue.add_argument("--label", default=None)
    issue.add_argument("--display-name", default=None)
    issue.add_argument("--email", default=None)
    issue.add_argument(
        "--json",
        action="store_true",
        help="Print one-line JSON instead of human format.",
    )

    revoke = keys_sub.add_parser("revoke", help="Revoke an API key by key_id.")
    revoke.add_argument("--key-id", required=True)

    listc = keys_sub.add_parser("list", help="List API keys (does NOT print secrets).")
    listc.add_argument("--user-id", default=None, help="Filter to one user.")
    listc.add_argument("--json", action="store_true")

    prune = sub.add_parser("prune", help="Delete stale checkpoints + history rows.")
    prune.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="Checkpoints to keep per conversation thread. Default: config.",
    )
    prune.add_argument(
        "--checkpoint-days",
        type=int,
        default=None,
        help="Drop checkpoint threads idle longer than this. Default: config.",
    )
    prune.add_argument(
        "--history-days",
        type=int,
        default=None,
        help="Drop history older than this many days. Default: config.",
    )
    prune.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Skip the post-delete VACUUM (faster, file does not shrink).",
    )

    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--auth", default=None)
    p.add_argument("--model-provider", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    return p


def _read_config_template(preset: str = "default") -> str:
    """Read one of the bundled preset templates."""
    resource = _PRESETS.get(preset)
    if resource is None:
        choices = ", ".join(sorted(_PRESETS.keys()))
        raise SystemExit(f"unknown preset {preset!r}; choose from: {choices}")
    return (
        resources.files("openbb_agent_server")
        .joinpath(resource)
        .read_text(encoding="utf-8")
    )


def _generate_config(target: str, preset: str = "default") -> None:
    body = _read_config_template(preset)
    if target == "-":
        sys.stdout.write(body)
        return
    out = Path(target).expanduser()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing file: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out} (preset={preset})")  # noqa: T201


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand.

    Handles ``--generate-config`` as an early exit, then bootstraps the
    layered launcher configuration before routing to the ``serve`` (default),
    ``keys``, or ``prune`` subcommand.

    Parameters
    ----------
    argv : list[str] or None, optional
        Argument vector to parse. When ``None``, ``sys.argv[1:]`` is used.

    Raises
    ------
    SystemExit
        If an unknown command is encountered (argparse normally rejects
        these first) or a subcommand requests an explicit exit.
    """
    raw_argv = argv if argv is not None else sys.argv[1:]

    if "--generate-config" in raw_argv:
        args = _build_parser().parse_args(raw_argv)
        _generate_config(args.generate_config, preset=args.preset)
        return

    explicit_path = extract_config_file_from_argv(raw_argv)
    cfg = bootstrap_launcher_config(explicit_path=explicit_path, argv=raw_argv)
    agent_cfg = agent_section(cfg)

    args = _build_parser().parse_args(raw_argv)
    command = args.command or "serve"

    if command == "serve":
        _serve(args=args, agent_cfg=agent_cfg, explicit_path=explicit_path)
    elif command == "keys":
        _keys(args=args, agent_cfg=agent_cfg)
    elif command == "prune":
        _prune(args=args, agent_cfg=agent_cfg)
    else:  # pragma: no cover — argparse rejects unknown
        raise SystemExit(f"unknown command: {command}")


def _serve(
    *, args: argparse.Namespace, agent_cfg: dict[str, Any], explicit_path: str | None
) -> None:
    cli_kwargs: dict[str, Any] = {}
    if args.host is not None:
        cli_kwargs["host"] = args.host
    if args.port is not None:
        cli_kwargs["port"] = args.port
    if args.auth is not None:
        cli_kwargs["auth_backend"] = args.auth
    if args.model_provider is not None:
        cli_kwargs["model_provider"] = args.model_provider
    if args.model_name is not None:
        cli_kwargs["model_name"] = args.model_name

    merged_kwargs = merge_launcher_kwargs(cli_kwargs, agent_cfg)

    if args.auth:
        os.environ["OPENBB_AGENT_AUTH_BACKEND"] = args.auth
    if args.model_provider:
        os.environ["OPENBB_AGENT_MODEL_PROVIDER"] = args.model_provider
    if args.model_name:
        os.environ["OPENBB_AGENT_MODEL_NAME"] = args.model_name

    from openbb_agent_server.observability.logging import (
        LOG_LEVEL_ENV,
        install_trace_logging,
    )

    os.environ[LOG_LEVEL_ENV] = args.log_level
    install_trace_logging()

    import uvicorn

    from openbb_agent_server.app.app import create_app
    from openbb_agent_server.app.settings import AgentServerSettings

    settings = AgentServerSettings.from_toml(agent_cfg)
    host = merged_kwargs.get("host", settings.host)
    port = int(merged_kwargs.get("port", settings.port))

    if args.reload:
        os.environ.setdefault("OPENBB_AGENT_BOOTSTRAP_TOML", explicit_path or "")
        uvicorn.run(
            "openbb_agent_server.app.app:create_app",
            host=host,
            port=port,
            reload=True,
            factory=True,
            log_level=args.log_level,
        )
    else:
        uvicorn.run(
            create_app(settings),
            host=host,
            port=port,
            log_level=args.log_level,
        )


def _keys(*, args: argparse.Namespace, agent_cfg: dict[str, Any]) -> None:
    from openbb_agent_server.app.settings import AgentServerSettings
    from openbb_agent_server.plugins.auth.api_key_table import (
        ApiKeyTableAuthBackend,
    )

    settings = AgentServerSettings.from_toml(agent_cfg)
    db_url = settings.auth_config.get("db_url") or settings.resolved_db_url()
    backend = ApiKeyTableAuthBackend(db_url=db_url)

    sub = args.keys_command
    if sub == "issue":
        scopes = tuple(args.scope) if args.scope else ("agent:query", "memory:read")

        async def _do() -> None:
            issued = await backend.issue(
                user_id=args.user_id,
                scopes=scopes,
                label=args.label,
                display_name=args.display_name,
                email=args.email,
            )
            await backend.aclose()
            payload = {
                "key_id": issued.key_id,
                "user_id": issued.user_id,
                "label": issued.label,
                "scopes": list(issued.scopes),
                "key": issued.plaintext,
            }
            if getattr(args, "json", False):
                print(json.dumps(payload))  # noqa: T201
            else:
                print(  # noqa: T201
                    f"key_id : {issued.key_id}\n"
                    f"user_id: {issued.user_id}\n"
                    f"scopes : {' '.join(issued.scopes)}\n"
                    f"label  : {issued.label or ''}\n\n"
                    f"key (shown ONCE — store it now):\n"
                    f"  {issued.plaintext}"
                )

        asyncio.run(_do())

    elif sub == "revoke":

        async def _do() -> None:
            ok = await backend.revoke(key_id=args.key_id)
            await backend.aclose()
            print(  # noqa: T201
                f"revoked: {args.key_id}" if ok else f"not found: {args.key_id}"
            )
            if not ok:
                raise SystemExit(1)

        asyncio.run(_do())

    elif sub == "list":

        async def _do() -> None:
            rows = await backend.list_keys(user_id=args.user_id)
            await backend.aclose()
            if getattr(args, "json", False):
                print(json.dumps(rows, default=str))  # noqa: T201
                return
            if not rows:
                print("(no keys)")  # noqa: T201
                return
            for r in rows:
                state = "REVOKED" if r["revoked_at"] else "active"
                print(  # noqa: T201
                    f"{r['key_id']:<14} {r['user_id']:<24} {state:<8} "
                    f"scopes={','.join(r['scopes'] or [])} "
                    f"label={r['label'] or ''}"
                )

        asyncio.run(_do())


def _prune(*, args: argparse.Namespace, agent_cfg: dict[str, Any]) -> None:
    """Run a one-shot retention prune of the checkpoint + history databases."""
    from openbb_agent_server.app.settings import AgentServerSettings
    from openbb_agent_server.persistence.prune import run_prune
    from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore

    settings = AgentServerSettings.from_toml(agent_cfg)
    keep_last = (
        args.keep_last if args.keep_last is not None else settings.checkpoint_keep_last
    )
    checkpoint_days = (
        args.checkpoint_days
        if args.checkpoint_days is not None
        else settings.checkpoint_retention_days
    )
    history_days = (
        args.history_days
        if args.history_days is not None
        else settings.history_retention_days
    )

    async def _do() -> None:
        history = SqliteHistoryStore(settings.resolved_db_url())
        await history.init_schema()
        try:
            stats = await run_prune(
                history=history,
                checkpoint_path=settings.resolved_checkpoint_path(),
                history_retention_days=history_days,
                checkpoint_retention_days=checkpoint_days,
                checkpoint_keep_last=keep_last,
                vacuum=not args.no_vacuum,
            )
        finally:
            await history.aclose()
        for table, count in sorted({**stats.history, **stats.checkpoints}.items()):
            print(f"  {table:<16} {count:>12,} deleted")  # noqa: T201
        print(f"prune complete — {stats.total():,} row(s) removed")  # noqa: T201

    asyncio.run(_do())


if __name__ == "__main__":  # pragma: no cover — script entry
    main()
