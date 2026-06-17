"""Server-level settings (env-prefixed ``OPENBB_AGENT_…`` + ``openbb.toml``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureSpec(BaseSettings):
    """Describe one toggleable feature row in the ``agents.json`` map.

    Attributes
    ----------
    label : str
        Human-readable name shown in the client's feature toggle UI.
    description : str
        Longer explanation of what enabling the feature does.
    default : bool
        Whether the feature is on by default for a new conversation.
    """

    model_config = SettingsConfigDict(extra="allow")

    label: str
    description: str
    default: bool = False


class AgentMetadata(BaseSettings):
    """Hold the static identity fields rendered in ``GET /agents.json``.

    Populated from ``OPENBB_AGENT_META_*`` environment variables or a
    profile's ``metadata`` overlay. The instance is frozen once built.

    Attributes
    ----------
    name : str
        Display name of the agent shown in the client.
    description : str
        Marketing/capability blurb describing the agent stack.
    image_url : str or None
        Optional avatar/logo URL; ``None`` when unset.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENBB_AGENT_META_",
        extra="ignore",
        frozen=True,
    )

    name: str = "OpenBB · NVIDIA Stack"
    description: str = (
        "DeepAgents harness over the OpenBB Platform with NVIDIA NIM "
        "end-to-end: nemotron-3-super-120b-a12b + nv-embed-v1 + "
        "nv-embedcode-7b-v1 + nv-rerank-qa-mistral-4b + riva-translate "
        "+ nemotron-nano-vl-8b vision + gemma-3n audio + paligemma."
    )
    image_url: str | None = None


class AgentProfile(BaseSettings):
    """Hold one fully-resolved agent profile used by the runtime.

    Produced by :meth:`AgentServerSettings.resolve_profile`, which merges
    a profile overlay over the server defaults. The instance is frozen.

    Attributes
    ----------
    name : str
        Profile key (``"default"`` or a key from ``profiles``).
    metadata : AgentMetadata
        Resolved display metadata for this profile.
    model_provider : str
        Registered model-provider plugin name (e.g. ``"nvidia"``).
    model_name : str
        Model identifier passed to the provider.
    model_config_ : dict
        Provider keyword arguments (temperature, token caps, etc.);
        aliased from ``model_config`` on input.
    tool_sources : tuple of str
        Ordered tool-source plugin names enabled for this profile.
    tool_source_config : dict
        Per-tool-source keyword overrides keyed by tool-source name.
    subagents : tuple of str
        Subagent plugin names available to the agent.
    middleware : tuple of str
        Middleware plugin names applied to the agent loop.
    skills : tuple of str
        Skill plugin names enabled for this profile.
    features : dict
        Feature-flag map merged from defaults and overlay.
    system_prompt_file : str or None
        Path to a system-prompt file, or ``None`` to use the default.
    """

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    name: str
    metadata: AgentMetadata
    model_provider: str
    model_name: str
    model_config_: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    tool_sources: tuple[str, ...] = ()
    tool_source_config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    subagents: tuple[str, ...] = ()
    middleware: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    features: dict[str, Any] = Field(default_factory=dict)
    system_prompt_file: str | None = None


def _resolve_system_prompt_file(
    overlay: dict[str, Any], base: str | None
) -> str | None:
    """Pick the profile's ``system_prompt_file`` and reject inline strings."""
    if "system_prompt" in overlay:
        raise ValueError(
            "Inline ``system_prompt`` is not supported. Move the prompt to "
            'a file and reference it via ``system_prompt_file = "<path>"``.'
        )
    return overlay.get("system_prompt_file") or base


SEARCH_WEB_FEATURE: str = "search-web"
FETCH_URL_FEATURE: str = "fetch-url"


DEFAULT_FEATURES: dict[str, Any] = {
    "streaming": True,
    "widget-dashboard-select": True,
    "widget-dashboard-search": True,
    "widget-global-search": True,
    "mcp-tools": True,
    "file-upload": True,
    "generative-ui": True,
    SEARCH_WEB_FEATURE: {
        "label": "Search Web",
        "description": (
            "Allow the agent to search the public web when answering. "
            "Each result attaches a citation card with the source URL. "
            "Off by default — turn on for queries about current events "
            "or anything outside the model's training data."
        ),
        "default": False,
    },
    FETCH_URL_FEATURE: {
        "label": "Fetch URL",
        "description": (
            "Allow the agent to fetch and read the full text of a web page "
            "from a URL. SSRF-guarded: private, loopback, link-local and "
            "cloud-metadata hosts are refused. Off by default — turn on to "
            "let the agent read the article behind a link."
        ),
        "default": False,
    },
}


class AgentServerSettings(BaseSettings):
    """Top-level server configuration for the OpenBB agent server.

    Fields are populated from ``OPENBB_AGENT_*`` environment variables
    (nested via ``__``) and/or the ``[agent]`` section of ``openbb.toml``
    via :meth:`~openbb_agent_server.acp.provider.OpenBBAgentProvider.from_toml`. Environment variables take precedence over TOML.
    Holds the default model/tool/middleware stack plus a ``profiles`` map
    of named overlays that :meth:`resolve_profile` flattens on demand. The
    instance is frozen once constructed.

    Attributes
    ----------
    host : str
        Address the HTTP server binds to.
    port : int
        TCP port the HTTP server listens on.
    mount_workspace_mcp : bool
        Whether to mount the bundled Workspace MCP server in-process.
    workspace_mcp_config : dict[str, Any]
        Keyword configuration for the mounted Workspace MCP server.
    auth_backend : str
        Name of the :class:`~openbb_agent_server.runtime.plugins.AuthBackend`
        plugin used to authenticate requests.
    auth_config : dict[str, Any]
        Keyword arguments passed to the auth backend.
    model_provider : str
        Default :class:`~openbb_agent_server.runtime.plugins.ModelProvider`
        plugin name.
    model_config_ : dict[str, Any]
        Default provider keyword arguments (temperature, token caps,
        ...); aliased from ``model_config`` on input.
    model_name : str
        Default model identifier passed to the provider.
    tool_sources : tuple[str, ...]
        Ordered default
        :class:`~openbb_agent_server.runtime.plugins.ToolSource` plugin
        names.
    tool_source_config : dict[str, dict[str, Any]]
        Per-tool-source keyword overrides, keyed by tool-source name.
    subagents : tuple[str, ...]
        Default sub-agent plugin names available to the agent.
    middleware : tuple[str, ...]
        Default
        :class:`~openbb_agent_server.runtime.plugins.Middleware` plugin
        names applied to the agent loop.
    skills : tuple[str, ...]
        Default skill plugin names enabled.
    system_prompt_file : str or None
        Path to a default system-prompt file, or ``None`` for the
        built-in prompt.
    checkpointer_provider : str
        Name of the
        :class:`~openbb_agent_server.runtime.plugins.CheckpointerProvider`
        plugin.
    checkpointer_config : dict[str, Any]
        Keyword arguments passed to the checkpointer provider.
    embeddings_provider : str
        Embeddings plugin name used for vector memory.
    embeddings_model : str or None
        Text-embeddings model identifier.
    embeddings_config : dict[str, Any]
        Keyword arguments for the embeddings provider.
    embeddings_code_provider : str or None
        Optional separate provider for code embeddings.
    embeddings_code_model : str or None
        Code-embeddings model identifier.
    embeddings_code_config : dict[str, Any]
        Keyword arguments for the code-embeddings provider.
    ingest_char_threshold : int
        Minimum context length, in characters, before it is ingested
        into memory.
    ingest_chunk_chars : int
        Target chunk size, in characters, for memory ingestion.
    ingest_chunk_overlap : int
        Overlap, in characters, between adjacent ingestion chunks.
    reranker_provider : str or None
        Optional reranker plugin name for memory recall.
    reranker_model : str or None
        Reranker model identifier.
    reranker_config : dict[str, Any]
        Keyword arguments for the reranker provider.
    rerank_fanout : int
        Number of candidates fetched before reranking trims them.
    translation_provider : str or None
        Optional translation plugin name.
    translation_model : str or None
        Translation model identifier.
    translation_config : dict[str, Any]
        Keyword arguments for the translation provider.
    translate_for_ingestion : bool
        Whether to translate ingested context to
        ``ingest_target_language`` before embedding.
    ingest_target_language : str
        Target language for ingestion-time translation.
    db_url : str or None
        Explicit history-store database URL; falls back to a SQLite file
        under ``data_dir`` when unset.
    data_dir : pathlib.Path
        Base directory for the database, checkpoints, and other state.
    checkpoint_keep_last : int
        Number of most-recent checkpoints to retain per thread.
    checkpoint_retention_days : int or None
        Age, in days, after which checkpoints are pruned; ``None``
        disables age-based pruning.
    history_retention_days : int or None
        Age, in days, after which history rows are pruned; ``None``
        disables it.
    prune_interval_hours : int
        Interval, in hours, between background retention sweeps.
    features : dict[str, Any]
        Feature-flag map advertised to clients; seeded from
        :data:`DEFAULT_FEATURES`.
    metadata : AgentMetadata
        Default agent identity rendered in ``GET /agents.json``.
    profiles : dict[str, dict[str, Any]]
        Named profile overlays flattened by :meth:`resolve_profile`.
    default_profile : str
        Profile key used when a request names none.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENBB_AGENT_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    host: str = "127.0.0.1"
    port: int = 6900

    mount_workspace_mcp: bool = False
    workspace_mcp_config: dict[str, Any] = Field(default_factory=dict)

    auth_backend: str = "none"
    auth_config: dict[str, Any] = Field(default_factory=dict)

    model_provider: str = "nvidia"
    model_config_: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.4,
            "max_completion_tokens": 8192,
            "top_p": 0.95,
        },
        alias="model_config",
    )
    model_name: str = "nvidia/nemotron-3-super-120b-a12b"

    tool_sources: tuple[str, ...] = (
        "artifacts",
        "web_search",
        "fetch_url",
        "widget_data",
        "inspect_widget_data",
        "pdf_extract",
        "dashboard",
        "recall_user_memory",
        "translate",
        "rerank",
        "vision_qa",
        "workspace_mcp",
    )

    tool_source_config: dict[str, dict[str, Any]] = Field(default_factory=dict)

    subagents: tuple[str, ...] = (
        "researcher",
        "charter",
        "analyst",
        "pdf_reader",
    )
    middleware: tuple[str, ...] = (
        "tool_message_normaliser",
        "tool_filter",
        "tool_call_announcer",
        "usage_recorder",
        "tool_call_ledger",
        "loop_guard",
        "call_limit",
        "tool_call_limit",
    )

    skills: tuple[str, ...] = ()

    system_prompt_file: str | None = None

    checkpointer_provider: str = "sqlite"
    checkpointer_config: dict[str, Any] = Field(default_factory=dict)

    embeddings_provider: str = "nvidia"
    embeddings_model: str | None = "nvidia/nv-embed-v1"
    embeddings_config: dict[str, Any] = Field(default_factory=dict)

    embeddings_code_provider: str | None = "nvidia-code"
    embeddings_code_model: str | None = "nvidia/nv-embedcode-7b-v1"
    embeddings_code_config: dict[str, Any] = Field(default_factory=dict)

    ingest_char_threshold: int = 2000
    ingest_chunk_chars: int = 1500
    ingest_chunk_overlap: int = 200

    reranker_provider: str | None = "nvidia"
    reranker_model: str | None = "nv-rerank-qa-mistral-4b:1"
    reranker_config: dict[str, Any] = Field(default_factory=dict)
    rerank_fanout: int = 32

    translation_provider: str | None = "nvidia"
    translation_model: str | None = "nvidia/riva-translate-4b-instruct-v1_1"
    translation_config: dict[str, Any] = Field(default_factory=dict)
    translate_for_ingestion: bool = True
    ingest_target_language: str = "English"

    db_url: str | None = None
    data_dir: Path = Path.home() / ".openbb_platform" / "agent"

    checkpoint_keep_last: int = 1
    checkpoint_retention_days: int | None = 14
    history_retention_days: int | None = 90
    prune_interval_hours: int = 24

    features: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_FEATURES))

    metadata: AgentMetadata = Field(default_factory=AgentMetadata)

    profiles: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "mistral-large-3": {
                "metadata": {
                    "name": "OpenBB · Mistral Large 3 (675B)",
                    "description": (
                        "Long-context Mistral-Large-3 (675B Instruct) on "
                        "NVIDIA NIM, with native vision over uploaded "
                        "images. Larger reasoning headroom for multi-"
                        "document synthesis, quantitative analysis, and "
                        "chart / table OCR. Best with images cropped to "
                        "a near-1:1 aspect ratio."
                    ),
                },
                "model": {
                    "provider": "nvidia",
                    "name": "mistralai/mistral-large-3-675b-instruct-2512",
                    "config": {
                        "temperature": 0.05,
                        "max_completion_tokens": 16384,
                        "top_p": 0.9,
                    },
                },
                "tool_source_config": {
                    "vision_qa": {
                        "model": "mistralai/mistral-large-3-675b-instruct-2512",
                    },
                },
            },
            "transcribe": {
                "metadata": {
                    "name": "OpenBB · Transcribe (Gemma-3n)",
                    "description": (
                        "Audio / video transcription specialist on "
                        "``google/gemma-3n-e4b-it``. Accepts text + image + "
                        "audio in a single turn, returns text. Use when "
                        "the user attaches an audio or video clip and "
                        "wants a transcript, summary, or per-speaker "
                        "breakdown. 32K-token context, single-channel "
                        "audio."
                    ),
                },
                "model": {
                    "provider": "nvidia",
                    "name": "google/gemma-3n-e4b-it",
                    "config": {
                        "temperature": 0.05,
                        "max_completion_tokens": 16384,
                        "top_p": 0.9,
                    },
                },
                "tool_sources": [
                    "artifacts",
                    "pdf_extract",
                    "gemma_audio",
                    "paligemma_vision",
                    "inspect_widget_data",
                ],
                "subagents": [],
            },
            "qwen3-coder": {
                "metadata": {
                    "name": "OpenBB · Qwen3 Coder (480B)",
                    "description": (
                        "Code-generation specialist on "
                        "``qwen/qwen3-coder-480b-a35b-instruct`` (480B "
                        "MoE, 35B active). Tuned for OpenBB Platform "
                        "scripting, SQL drafting, and quantitative "
                        "snippets — pair with the ``snowflake`` or "
                        "``mcp_local`` tool sources for end-to-end "
                        "execution."
                    ),
                },
                "model": {
                    "provider": "nvidia",
                    "name": "qwen/qwen3-coder-480b-a35b-instruct",
                    "config": {
                        "temperature": 0.2,
                        "max_completion_tokens": 16384,
                        "top_p": 0.95,
                    },
                },
                "tool_sources": [
                    "artifacts",
                    "widget_data",
                    "inspect_widget_data",
                    "pdf_extract",
                    "recall_user_memory",
                    "web_search",
                    "workspace_mcp",
                ],
                "subagents": ["analyst"],
            },
            "seed-oss": {
                "metadata": {
                    "name": "OpenBB · Seed-OSS 36B (Thinking Budget)",
                    "description": (
                        "ByteDance Seed-OSS 36B Instruct on NVIDIA NIM "
                        "with a per-turn ``thinking_budget`` cap (1024 "
                        "tokens by default). Lower the budget for "
                        "latency-bound chat; raise it for multi-hop "
                        "synthesis. Same tool surface as the default "
                        "agent."
                    ),
                },
                "model": {
                    "provider": "nvidia",
                    "name": "bytedance/seed-oss-36b-instruct",
                    "config": {
                        "temperature": 0.3,
                        "max_completion_tokens": 8192,
                        "top_p": 0.9,
                        "extra_body": {"thinking_budget": 1024},
                    },
                },
            },
            "step-flash": {
                "metadata": {
                    "name": "OpenBB · Step 3.5 Flash (Reasoning)",
                    "description": (
                        "StepFun Step-3.5-Flash on NVIDIA NIM — fast "
                        "reasoning model with a configurable "
                        "``reasoning_effort`` enum. Reasoning tokens "
                        "stream live as step-by-step entries in the UI. "
                        "Default effort ``medium``."
                    ),
                },
                "model": {
                    "provider": "nvidia",
                    "name": "stepfun-ai/step-3.5-flash",
                    "config": {
                        "temperature": 0.4,
                        "max_completion_tokens": 8192,
                        "top_p": 0.9,
                        "extra_body": {"reasoning_effort": "medium"},
                    },
                },
            },
            "minimax-m2": {
                "metadata": {
                    "name": "OpenBB · MiniMax M2.7",
                    "description": (
                        "MiniMax M2.7 on NVIDIA NIM — long-context "
                        "generalist with strong instruction following. "
                        "A third opinion alongside Nemotron and Mistral "
                        "for multi-document synthesis."
                    ),
                },
                "model": {
                    "provider": "nvidia",
                    "name": "minimaxai/minimax-m2.7",
                    "config": {
                        "temperature": 0.4,
                        "max_completion_tokens": 8192,
                        "top_p": 0.9,
                    },
                },
            },
        }
    )
    default_profile: str = "default"

    def resolved_db_url(self) -> str:
        """Resolve the history persistence DB URL.

        Returns
        -------
        str
            ``db_url`` verbatim when set, otherwise a
            ``sqlite+aiosqlite`` URL pointing at ``data_dir/history.db``.
        """
        if self.db_url:
            return self.db_url
        path = self.data_dir / "history.db"
        return f"sqlite+aiosqlite:///{path}"

    def resolved_checkpoint_path(self) -> str | None:
        """Resolve the sqlite checkpointer file path.

        Resolution order when the provider is ``"sqlite"``:
        ``checkpointer_config["path"]``, then the
        ``OPENBB_AGENT_CHECKPOINTER_PATH`` environment variable, then
        ``data_dir/checkpoints.db``.

        Returns
        -------
        str or None
            The checkpoint file path, or ``None`` when the checkpointer
            provider is not ``"sqlite"``.
        """
        if self.checkpointer_provider != "sqlite":
            return None
        explicit = self.checkpointer_config.get("path") or os.environ.get(
            "OPENBB_AGENT_CHECKPOINTER_PATH"
        )
        if explicit:
            return str(explicit)
        return str(self.data_dir / "checkpoints.db")

    def all_profile_names(self) -> tuple[str, ...]:
        """Return every profile name this server hosts.

        Returns
        -------
        tuple of str
            The configured profile keys, with ``default_profile``
            prepended when it is not already present.
        """
        names = list(self.profiles.keys())
        if self.default_profile not in names:
            names.insert(0, self.default_profile)
        return tuple(names)

    def resolve_profile(self, name: str | None = None) -> AgentProfile:
        """Resolve a profile name into a fully-populated profile.

        Merges the named profile's overlay over the server defaults:
        metadata fields fall back to server metadata, per-tool-source
        config is deep-merged, and model provider/name/config are taken
        from either flat ``model_*`` overlay keys or the nested
        ``model`` table.

        Parameters
        ----------
        name : str or None
            Profile key to resolve; ``None`` selects ``default_profile``.

        Returns
        -------
        AgentProfile
            The flattened, frozen profile for the runtime.

        Raises
        ------
        KeyError
            If ``name`` is neither the default profile nor a configured
            profile key.
        ValueError
            If the overlay contains an inline ``system_prompt`` (only
            ``system_prompt_file`` is supported).
        """
        target = name or self.default_profile
        if target != self.default_profile and target not in self.profiles:
            raise KeyError(f"agent profile {target!r} not configured")
        overlay = self.profiles.get(target) or {}

        meta_overlay = overlay.get("metadata") or {}
        if isinstance(meta_overlay, dict):
            meta = AgentMetadata(
                **{
                    "name": meta_overlay.get("name", self.metadata.name),
                    "description": meta_overlay.get(
                        "description", self.metadata.description
                    ),
                    "image_url": meta_overlay.get("image_url", self.metadata.image_url),
                }
            )
        else:
            meta = self.metadata

        # Per-tool-source kwargs: profile overlay merges over base.
        merged_tool_cfg: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in self.tool_source_config.items()
        }
        for k, v in (overlay.get("tool_source_config") or {}).items():
            if isinstance(v, dict):
                merged_tool_cfg[k] = {**merged_tool_cfg.get(k, {}), **v}

        model_overlay = overlay.get("model") or {}
        if not isinstance(model_overlay, dict):
            model_overlay = {}
        provider = (
            overlay.get("model_provider")
            or model_overlay.get("provider")
            or self.model_provider
        )
        model_name = (
            overlay.get("model_name")
            if "model_name" in overlay
            else model_overlay.get("name", self.model_name)
        )
        model_cfg_overlay = overlay.get("model_config")
        if model_cfg_overlay is None:
            model_cfg_overlay = model_overlay.get("config")
        if not isinstance(model_cfg_overlay, dict):
            model_cfg_overlay = self.model_config_

        profile_kwargs: dict[str, Any] = {
            "name": target,
            "metadata": meta,
            "model_provider": str(provider),
            "model_name": str(model_name),
            "model_config": dict(model_cfg_overlay),
            "tool_sources": tuple(overlay.get("tool_sources", self.tool_sources)),
            "tool_source_config": merged_tool_cfg,
            "subagents": tuple(overlay.get("subagents", self.subagents)),
            "middleware": tuple(overlay.get("middleware", self.middleware)),
            "skills": tuple(overlay.get("skills", self.skills)),
            "features": {**self.features, **dict(overlay.get("features", {}))},
            "system_prompt_file": _resolve_system_prompt_file(
                overlay, self.system_prompt_file
            ),
        }
        return AgentProfile(**profile_kwargs)

    @classmethod
    def from_toml(  # noqa: PLR0912 — orchestration: walks every promoted key once.
        cls, agent_section: dict[str, Any]
    ) -> AgentServerSettings:
        """Build settings from an ``[agent]`` TOML section.

        Flattens the nested ``auth``, ``model``, ``metadata``,
        ``features``, and ``profiles`` tables into the model's flat
        fields, expands ``data_dir``, and merges ``features`` over
        :data:`DEFAULT_FEATURES`. Any key with a matching
        ``OPENBB_AGENT_<KEY>`` environment variable is dropped from the
        TOML payload so the environment variable wins.

        Parameters
        ----------
        agent_section : dict
            The parsed ``[agent]`` table from ``openbb.toml``; an empty
            mapping yields default settings.

        Returns
        -------
        AgentServerSettings
            Settings built from the TOML overlay and environment.

        Raises
        ------
        ValueError
            If the section contains an inline ``system_prompt`` (only
            ``system_prompt_file`` is supported).
        """
        import os as _os

        if not agent_section:
            return cls()

        if "system_prompt" in agent_section:
            raise ValueError(
                "Inline ``system_prompt`` in [agent] is not supported. "
                "Move the prompt to a file and reference it via "
                '``system_prompt_file = "<path>"``.'
            )

        flat: dict[str, Any] = {}
        for k, v in agent_section.items():
            if k in {"auth", "model", "metadata", "features", "profiles"}:
                continue
            flat[k] = v

        if "profiles" in agent_section and isinstance(agent_section["profiles"], dict):
            flat["profiles"] = {
                name: dict(spec)
                for name, spec in agent_section["profiles"].items()
                if isinstance(spec, dict)
            }

        auth = agent_section.get("auth") or {}
        if isinstance(auth, dict):
            if "backend" in auth:
                flat["auth_backend"] = auth["backend"]
            if "config" in auth and isinstance(auth["config"], dict):
                flat["auth_config"] = auth["config"]

        model = agent_section.get("model") or {}
        if isinstance(model, dict):
            if "provider" in model:
                flat["model_provider"] = model["provider"]
            if "name" in model:
                flat["model_name"] = model["name"]
            if "config" in model and isinstance(model["config"], dict):
                flat["model_config"] = model["config"]

        if "features" in agent_section and isinstance(agent_section["features"], dict):
            flat["features"] = {
                **DEFAULT_FEATURES,
                **agent_section["features"],
            }

        if "metadata" in agent_section and isinstance(agent_section["metadata"], dict):
            flat["metadata"] = AgentMetadata(**agent_section["metadata"])

        if "data_dir" in flat:
            flat["data_dir"] = Path(flat["data_dir"]).expanduser()

        env_winning = {
            k for k in list(flat) if f"OPENBB_AGENT_{k.upper()}" in _os.environ
        }
        for k in env_winning:
            flat.pop(k, None)

        return cls(**flat)
