"""Layered TOML config bootstrap for ``openbb-agent-server``."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):  # pragma: no cover — version-conditional import
    import tomllib
else:  # pragma: no cover — exercised only on the 3.10 backport path
    import tomli as tomllib  # ty: ignore[unresolved-import]

logger = logging.getLogger("openbb_agent_server.config")

_ENV_REF_PATTERN = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}"
    r"|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)

EXPLICIT_CONFIG_ENVS: tuple[str, ...] = (
    "OPENBB_AGENT_CONFIG",
    "OPENBB_API_CONFIG",
    "OPENBB_CONFIG",
)

CONFIG_FILE_FLAG = "--config-file"


def extract_config_file_from_argv(argv: list[str] | None = None) -> str | None:
    """Sniff ``--config-file <path>`` out of argv WITHOUT importing args.

    Scan the argument list for ``--config-file <path>`` (space-separated)
    or ``--config-file=<path>`` (equals form) and return the path. The
    space form is ignored when the following token is missing or itself
    looks like a flag (starts with ``--``).

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to scan. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    str | None
        The config-file path if the flag is present and carries a value,
        otherwise ``None``.
    """
    args = (argv if argv is not None else sys.argv[1:]).copy()
    for i, arg in enumerate(args):
        if arg == CONFIG_FILE_FLAG and i + 1 < len(args):
            value = args[i + 1]
            if value and not value.startswith("--"):
                return value
        elif arg.startswith(f"{CONFIG_FILE_FLAG}="):
            value = arg.split("=", 1)[1]
            if value:
                return value
    return None


def explicit_config_path(argv: list[str] | None = None) -> str | None:
    """Return the explicit config path from argv or env (in priority order).

    Prefer a ``--config-file`` value found in ``argv``; falling back to
    the first non-empty value among the env vars listed in
    ``EXPLICIT_CONFIG_ENVS`` (``OPENBB_AGENT_CONFIG``,
    ``OPENBB_API_CONFIG``, ``OPENBB_CONFIG``, in that order).

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to scan. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    str | None
        The resolved config path, or ``None`` when neither argv nor any
        of the env vars supply one.
    """
    cli = extract_config_file_from_argv(argv)
    if cli:
        return cli
    for var in EXPLICIT_CONFIG_ENVS:
        v = os.environ.get(var)
        if v:
            return v
    return None


def _validate_explicit_toml(explicit_path: str) -> None:
    p = Path(explicit_path)
    if not p.is_file():
        return
    try:
        with p.open("rb") as fh:
            tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"Malformed TOML at explicit config path '{explicit_path}': {exc}"
        ) from exc


def expand_env_refs(
    value: str, env: Mapping[str, str] | None = None
) -> tuple[str, list[str]]:
    """Substitute ``$VAR`` / ``${VAR}`` references against ``env``.

    Replace each ``$VAR`` and ``${VAR}`` reference with its value from
    ``env``. References whose name is absent from ``env`` are left
    verbatim in the returned string and reported as missing.

    Parameters
    ----------
    value : str
        Source string possibly containing variable references.
    env : Mapping[str, str] | None, optional
        Lookup table for variable values. Defaults to ``os.environ``
        when ``None``.

    Returns
    -------
    tuple[str, list[str]]
        The expanded string and the list of unique referenced names that
        had no value in ``env`` (in first-seen order).
    """
    target_env = env if env is not None else os.environ
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        if name and name in target_env:
            return target_env[name]
        if name and name not in missing:
            missing.append(name)
        return match.group(0)

    return _ENV_REF_PATTERN.sub(replace, value), missing


USER_SETTINGS_PATH = "~/.openbb_platform/user_settings.json"


def apply_user_settings_credentials(
    *,
    settings_path: str = USER_SETTINGS_PATH,
    env: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Push ``credentials`` from ``user_settings.json`` into ``os.environ``.

    Read the OpenBB Platform ``user_settings.json`` file, take its
    ``credentials`` object, and upper-case each key into an env var. Only
    string-valued, non-empty entries are applied, and existing env vars
    are never clobbered. A missing or unparseable file is treated as a
    no-op (the parse failure is logged at WARNING).

    Parameters
    ----------
    settings_path : str, optional
        Path to ``user_settings.json``; ``~`` is expanded. Defaults to
        ``USER_SETTINGS_PATH``.
    env : MutableMapping[str, str] | None, optional
        Target mapping to mutate. Defaults to ``os.environ`` when
        ``None``.

    Returns
    -------
    list[str]
        The upper-cased env var names that were newly set, for logging.
    """
    import json

    target: MutableMapping[str, str] = env if env is not None else os.environ
    path = Path(settings_path).expanduser()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not parse %s — skipping credential injection: %s",
            path,
            exc,
        )
        return []
    creds = data.get("credentials")
    if not isinstance(creds, dict):
        return []

    applied: list[str] = []
    for raw_key, raw_value in creds.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            continue
        if not raw_value:
            continue
        name = raw_key.upper()
        if name in target:
            continue
        target[name] = raw_value
        applied.append(name)
    if applied:
        logger.debug(
            "user_settings.json: applied %d env var(s) from credentials: %s",
            len(applied),
            ", ".join(sorted(applied)),
        )
    return applied


def apply_launcher_env(
    env_section: dict[str, Any] | None,
    *,
    env: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Push ``[env]`` table entries into ``os.environ`` (no clobber).

    Apply each ``[env]`` key/value pair as an env var, expanding any
    ``$VAR`` / ``${VAR}`` references in the value first. Existing env
    vars are never overwritten, and an entry referencing an unresolved
    variable is skipped with a WARNING rather than applied half-expanded.

    Parameters
    ----------
    env_section : dict[str, Any] | None
        The parsed ``[env]`` table. A falsy value yields an empty result.
    env : MutableMapping[str, str] | None, optional
        Target mapping to mutate and to resolve references against.
        Defaults to ``os.environ`` when ``None``.

    Returns
    -------
    list[str]
        The env var names that were newly set, for logging.
    """
    if not env_section:
        return []
    target: MutableMapping[str, str] = env if env is not None else os.environ
    applied: list[str] = []
    for key, value in env_section.items():
        if not isinstance(key, str):
            continue
        if key in target:
            continue
        expanded, missing = expand_env_refs(str(value), target)
        if missing:
            logger.warning(
                "Skipping [env] entry %s: references unresolved variable(s) %s",
                key,
                ", ".join(missing),
            )
            continue
        target[key] = expanded
        applied.append(key)
    return applied


def expand_in_dict(
    value: Any,
    *,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Recursively expand ``$VAR`` / ``${VAR}`` in every string in ``value``.

    Walk dicts, lists, and tuples, expanding variable references in every
    nested string. Container types are preserved (a tuple stays a tuple).
    Missing references are left verbatim; unlike ``expand_env_refs`` the
    list of missing names is discarded.

    Parameters
    ----------
    value : Any
        The value to expand. Non-string, non-container values are
        returned unchanged.
    env : Mapping[str, str] | None, optional
        Lookup table for variable values. Defaults to ``os.environ``
        when ``None``.

    Returns
    -------
    Any
        A structurally identical value with all nested strings expanded.
    """
    if isinstance(value, str):
        expanded, _ = expand_env_refs(value, env)
        return expanded
    if isinstance(value, dict):
        return {k: expand_in_dict(v, env=env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_in_dict(v, env=env) for v in value]
    if isinstance(value, tuple):
        return tuple(expand_in_dict(v, env=env) for v in value)
    return value


# ---------------------------------------------------------------------------
# Layered TOML cascade.
#
# Vendored from openbb-core's ``app.config.loader`` so the server runs
# without openbb-core installed. Discovery order, merge semantics, and
# key normalization are kept identical so an agent server and a
# co-installed OpenBB Platform resolve the same files the same way.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_NAMES: tuple[str, ...] = ("openbb.toml", ".openbb.toml")
PYPROJECT_NAME = "pyproject.toml"
PYPROJECT_TABLE: tuple[str, ...] = ("tool", "openbb")
EXPLICIT_CONFIG_ENV = "OPENBB_CONFIG"
EXPLICIT_ENV_FILE_ENV = "OPENBB_ENV_FILE"

USER_OPENBB_DIR = Path.home() / ".openbb_platform"
USER_OPENBB_TOML_NAMES: tuple[str, ...] = ("openbb.toml", ".openbb.toml")
USER_OPENBB_ENV_NAME = ".env"

# Top-level convenience keys that fold onto the matching ``[system]``
# field and seed the ``OPENBB_*`` env vars openbb-core's ``Env`` reads.
# The agent server itself reads none of them — they exist so a TOML
# shared with an in-process OpenBB Platform behaves the same here.
_TOP_LEVEL_SYSTEM_PROMOTIONS: tuple[str, ...] = (
    "debug_mode",
    "test_mode",
    "headless",
    "logging_suppress",
    "allow_mutable_extensions",
    "allow_on_command_output",
)


def _walk_up(start: Path | None = None):
    """Yield ``start`` and every parent up to the filesystem root."""
    cur = (start or Path.cwd()).resolve()
    yield cur
    yield from cur.parents


def _read_toml(path: Path) -> dict[str, Any]:
    """Safe-load a TOML file. Missing / unreadable files yield ``{}``."""
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _find_first(start: Path | None, names: tuple[str, ...]) -> Path | None:
    """Walk up from ``start`` looking for any filename in ``names``."""
    for parent in _walk_up(start):
        for name in names:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


def _find_pyproject_section(start: Path | None = None) -> dict[str, Any]:
    """Return ``[tool.openbb]`` from the nearest ancestor ``pyproject.toml``."""
    py = _find_first(start, (PYPROJECT_NAME,))
    if py is None:
        return {}
    data = _read_toml(py)
    node: Any = data
    for key in PYPROJECT_TABLE:
        if not isinstance(node, dict) or key not in node:
            return {}
        node = node[key]
    return node if isinstance(node, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """In-place deep merge: nested dicts merge, scalars / lists overwrite."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _normalize_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize top-level kebab-case keys to snake_case.

    Nested tables keep their original keys — section names like
    ``[agent]`` / ``[system]`` map directly onto settings fields and
    must round-trip exactly. Only the top-level convenience keys
    (``debug-mode``, ``test-mode``, ...) accept either spelling.
    """
    return {k.replace("-", "_"): v for k, v in d.items()}


def _user_global_toml() -> Path | None:
    """Locate ``~/.openbb_platform/openbb.toml`` (or the dotted variant)."""
    if not USER_OPENBB_DIR.is_dir():
        return None
    for name in USER_OPENBB_TOML_NAMES:
        candidate = USER_OPENBB_DIR / name
        if candidate.is_file():
            return candidate
    return None


def load_config(
    explicit_path: str | os.PathLike[str] | None = None,
    *,
    start: Path | None = None,
) -> dict[str, Any]:
    """Resolve the layered TOML cascade into a single merged dict.

    Layers, lowest to highest priority:

    * ``[tool.openbb]`` from the nearest ancestor ``pyproject.toml``
    * ``~/.openbb_platform/openbb.toml`` (or ``.openbb.toml``)
    * ``openbb.toml`` (or ``.openbb.toml``) walking up from ``start``
      (defaults to CWD)
    * ``explicit_path`` if provided, otherwise ``$OPENBB_CONFIG``

    Top-level keys are normalized kebab-case → snake_case; nested
    tables deep-merge across layers. Returns ``{}`` when no layer is
    found.

    Parameters
    ----------
    explicit_path : str or os.PathLike[str] or None, optional
        Highest-priority config file. Falls back to ``$OPENBB_CONFIG``
        when ``None``.
    start : pathlib.Path or None, optional
        Directory the upward walk for project ``openbb.toml`` /
        ``pyproject.toml`` starts from. Defaults to the current working
        directory.

    Returns
    -------
    dict[str, Any]
        The merged config dict, or ``{}`` when no layer is found.
    """
    layers: list[dict[str, Any]] = [_find_pyproject_section(start)]
    user_toml = _user_global_toml()
    if user_toml is not None:
        layers.append(_read_toml(user_toml))
    project_toml = _find_first(start, DEFAULT_CONFIG_NAMES)
    if project_toml is not None:
        layers.append(_read_toml(project_toml))
    explicit = explicit_path or os.environ.get(EXPLICIT_CONFIG_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            layers.append(_read_toml(path))
    merged: dict[str, Any] = {}
    for layer in layers:
        _deep_merge(merged, _normalize_keys(layer))
    return merged


def load_env_files(
    explicit_path: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """Apply ``.env`` files into ``os.environ`` for subsequent lookups.

    ``~/.openbb_platform/.env`` first, then ``explicit_path`` (or
    ``$OPENBB_ENV_FILE``). Values go in via ``setdefault`` so real
    shell exports always win. Returns the files applied, for logging.

    Parameters
    ----------
    explicit_path : str or os.PathLike[str] or None, optional
        Path to an additional ``.env`` file applied after the user
        global one. Falls back to ``$OPENBB_ENV_FILE`` when ``None``.

    Returns
    -------
    list[pathlib.Path]
        The ``.env`` files that were applied, in load order.
    """
    from dotenv import dotenv_values

    candidates: list[Path] = []
    user_env = USER_OPENBB_DIR / USER_OPENBB_ENV_NAME
    if user_env.is_file():
        candidates.append(user_env)
    explicit = explicit_path or os.environ.get(EXPLICIT_ENV_FILE_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            candidates.append(path)

    loaded: list[Path] = []
    for path in candidates:
        for k, v in dotenv_values(path).items():
            if v is None:
                continue
            os.environ.setdefault(k, v)
        loaded.append(path)
    return loaded


def _promote_top_level_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Fold top-level convenience keys into ``[system]``.

    Returns a new dict — caller's input untouched. Top-level wins on
    conflict because users who set both clearly meant the explicit
    short form.
    """
    if not config:
        return {}
    out = {k: v for k, v in config.items() if k != "system" or isinstance(v, dict)}
    system_section: dict[str, Any] = dict(out.get("system") or {})
    for key in _TOP_LEVEL_SYSTEM_PROMOTIONS:
        if key in out:
            system_section[key] = out.pop(key)
    if system_section:
        out["system"] = system_section
    return out


def apply_settings_to_env(config: dict[str, Any] | None) -> list[str]:
    """Seed ``OPENBB_*`` env vars from promoted top-level system flags.

    Fold top-level convenience keys into ``[system]`` then export each
    recognized flag as ``OPENBB_<KEY>``. Booleans are rendered as
    ``"True"`` / ``"False"``; other scalars via ``str``. Uses
    ``setdefault`` semantics so real shell exports stay authoritative.

    Parameters
    ----------
    config : dict[str, Any] | None
        The merged config dict. A falsy value yields an empty result.

    Returns
    -------
    list[str]
        The ``OPENBB_*`` env var names that were newly set, for logging.
    """
    if not config:
        return []
    promoted = _promote_top_level_keys(config)
    applied: list[str] = []
    system_section = promoted.get("system") or {}
    if not isinstance(system_section, dict):  # pragma: no cover — defensive
        # ``_promote_top_level_keys`` already drops non-dict ``system``
        # values, so this guard is unreachable via the normal load path.
        return []
    for key, value in system_section.items():
        if key not in _TOP_LEVEL_SYSTEM_PROMOTIONS:
            continue
        env_key = f"OPENBB_{key.upper()}"
        if env_key in os.environ:
            continue
        if isinstance(value, bool):
            os.environ[env_key] = "True" if value else "False"
        else:
            os.environ[env_key] = str(value)
        applied.append(env_key)
    return applied


def _apply_config_to_openbb_services(config: dict[str, Any] | None) -> None:
    """Push ``[system]`` / ``[user]`` onto openbb-core's services, if present.

    The agent server reads only ``[agent]`` / ``[env]``; the
    ``[system]`` / ``[user]`` tables exist for an OpenBB Platform
    running in the same process (e.g. a ``python_module`` tool source
    importing ``openbb``). openbb-core is not a dependency of this
    package — when it isn't importable there are no services to
    configure and the push is skipped.
    """
    if not config:
        return
    try:
        from openbb_core.app.config.loader import (  # ty: ignore[unresolved-import]
            apply_config_to_services,
        )
    except ImportError:
        return
    apply_config_to_services(config)


def load_launcher_config(
    explicit_path: str | None = None,
    *,
    apply_to_services: bool = True,
    apply_to_env: bool = True,
) -> dict[str, Any]:
    """Run the layered TOML cascade and return the merged config dict.

    Validate that an explicit path parses as TOML first (so a malformed
    config file fails loudly at startup rather than silently dropping
    settings), load ``.env`` files, resolve the cascade, then optionally
    seed ``OPENBB_*`` env vars and push ``[system]`` / ``[user]`` onto
    openbb-core's services when that package happens to be installed.

    Parameters
    ----------
    explicit_path : str | None, optional
        Highest-priority config file. When set it is also validated as
        parseable TOML before the cascade runs.
    apply_to_services : bool, optional
        When ``True`` (default), push ``[system]`` / ``[user]`` onto
        openbb-core's services if that package is importable.
    apply_to_env : bool, optional
        When ``True`` (default), seed ``OPENBB_*`` env vars from promoted
        top-level system flags.

    Returns
    -------
    dict[str, Any]
        The merged config dict from the layered cascade.

    Raises
    ------
    ValueError
        If ``explicit_path`` points at an existing but malformed TOML
        file.
    """
    if explicit_path:
        _validate_explicit_toml(explicit_path)

    load_env_files()
    cfg = load_config(explicit_path)
    if apply_to_env:
        apply_settings_to_env(cfg)
    if apply_to_services:
        _apply_config_to_openbb_services(cfg)
    return cfg


def merge_launcher_kwargs(
    cli_kwargs: dict[str, Any],
    launcher_section: dict[str, Any] | None,
) -> dict[str, Any]:
    """Overlay ``[agent]`` section under CLI kwargs (CLI wins).

    Merge the config-file ``[agent]`` values with command-line kwargs so
    that any key supplied on the CLI overrides the same key from the
    config file.

    Parameters
    ----------
    cli_kwargs : dict[str, Any]
        Keyword arguments collected from the command line.
    launcher_section : dict[str, Any] | None
        The ``[agent]`` section. When falsy, ``cli_kwargs`` is returned
        unchanged.

    Returns
    -------
    dict[str, Any]
        A new dict with ``launcher_section`` as the base and
        ``cli_kwargs`` applied on top.
    """
    if not launcher_section:
        return cli_kwargs
    merged = dict(launcher_section)
    merged.update(cli_kwargs)
    return merged


def bootstrap_launcher_config(
    explicit_path: str | None = None,
    *,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Two-phase bootstrap entry point. Run BEFORE any heavy import.

    Resolve the explicit config path (when not given), run the launcher
    cascade, inject ``user_settings.json`` credentials, apply the
    ``[env]`` table, and finally expand ``$VAR`` references throughout the
    returned config. Intended to run early at startup so env vars are in
    place before modules that read them are imported.

    Parameters
    ----------
    explicit_path : str | None, optional
        Highest-priority config file. When ``None`` it is resolved from
        argv / env via ``explicit_config_path``.
    argv : list[str] | None, optional
        Argument list used to discover ``--config-file`` when
        ``explicit_path`` is ``None``. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    dict[str, Any]
        The merged, env-expanded config dict.
    """
    if explicit_path is None:
        explicit_path = explicit_config_path(argv)
    cfg = load_launcher_config(explicit_path)
    apply_user_settings_credentials()
    apply_launcher_env(cfg.get("env"))
    cfg = expand_in_dict(cfg)
    return cfg


def agent_section(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the ``[agent]`` sub-table out of the merged config.

    Parameters
    ----------
    cfg : dict[str, Any]
        The merged config dict.

    Returns
    -------
    dict[str, Any]
        A shallow copy of the ``[agent]`` table, or an empty dict when it
        is absent or not a mapping.
    """
    section = cfg.get("agent")
    if not isinstance(section, dict):
        return {}
    return dict(section)


def load_preset(preset: str) -> dict[str, Any]:
    """Parse one of the bundled preset TOMLs into a config dict.

    Look the preset name up in ``main._PRESETS``, read the packaged TOML
    resource, parse it, apply its ``[env]`` table, and expand ``$VAR``
    references throughout before returning.

    Parameters
    ----------
    preset : str
        Name of a bundled preset, as keyed in ``_PRESETS``.

    Returns
    -------
    dict[str, Any]
        The parsed, env-expanded preset config dict.

    Raises
    ------
    ValueError
        If ``preset`` is not a known preset name; the message lists the
        available choices.
    """
    from importlib import resources

    from openbb_agent_server.main import _PRESETS  # noqa: PLC0415

    resource = _PRESETS.get(preset)
    if resource is None:
        choices = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(f"unknown preset {preset!r}; choose from: {choices}")

    body = (
        resources.files("openbb_agent_server")
        .joinpath(resource)
        .read_text(encoding="utf-8")
    )
    cfg = tomllib.loads(body)
    apply_launcher_env(cfg)
    cfg = expand_in_dict(cfg)
    return cfg
