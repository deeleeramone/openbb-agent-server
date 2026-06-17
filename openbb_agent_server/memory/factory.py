"""Embeddings / reranker / translator factories."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.embeddings import Embeddings

from openbb_agent_server.memory.embeddings import HashEmbeddings
from openbb_agent_server.memory.reranker import NvidiaReranker
from openbb_agent_server.memory.translation import NvidiaTranslator

logger = logging.getLogger("openbb_agent_server.memory.factory")


def make_embeddings(
    provider: str | None = None,
    *,
    model: str | None = None,
    config: dict[str, Any] | None = None,
) -> Embeddings:
    """Construct an embeddings backend for the given provider.

    Parameters
    ----------
    provider : str or None
        Provider key (case-insensitive). ``""``, ``"hash"`` or
        ``"default"`` select the deterministic ``HashEmbeddings``
        fallback; ``"nvidia"`` selects NVIDIA text embeddings
        (default model ``nvidia/nv-embed-v1``); ``"nvidia-code"``
        selects NVIDIA code embeddings (default model
        ``nvidia/nv-embedcode-7b-v1``).
    model : str or None, optional
        Override the provider's default model. Falls back to
        ``config["model"]`` when not given.
    config : dict[str, Any] or None, optional
        Extra provider options. Recognized keys include ``dim`` (hash
        fallback), ``api_key``, ``truncate``, ``base_url`` and
        ``dimensions`` (NVIDIA backends).

    Returns
    -------
    Embeddings
        A LangChain ``Embeddings`` instance for the chosen provider.

    Raises
    ------
    ValueError
        If ``provider`` is not one of the supported keys.
    RuntimeError
        For NVIDIA providers when the
        ``langchain-nvidia-ai-endpoints`` package is missing or no
        ``NVIDIA_API_KEY`` is available.
    """
    cfg = dict(config or {})
    name = (provider or "").strip().lower()

    if name in ("", "hash", "default"):
        logger.warning(
            "embeddings: using HashEmbeddings fallback. Semantic recall "
            "quality will be poor. Set EMBEDDINGS_PROVIDER=nvidia + "
            "NVIDIA_API_KEY for production-grade recall."
        )
        return HashEmbeddings(**{k: v for k, v in cfg.items() if k == "dim"})

    if name == "nvidia":
        return _nvidia(
            default_model="nvidia/nv-embed-v1",
            model=model or cfg.get("model"),
            cfg=cfg,
        )

    if name == "nvidia-code":
        return _nvidia(
            default_model="nvidia/nv-embedcode-7b-v1",
            model=model or cfg.get("model"),
            cfg=cfg,
        )

    raise ValueError(
        f"unknown embeddings provider {provider!r}; supported: "
        "'nvidia' (text, default model nv-embed-v1), "
        "'nvidia-code' (code, default model nv-embedcode-7b-v1), "
        "'hash' (deterministic fallback)"
    )


def make_reranker(
    provider: str | None,
    *,
    model: str | None = None,
    config: dict[str, Any] | None = None,
) -> NvidiaReranker | None:
    """Construct a reranker, or ``None`` when disabled.

    Parameters
    ----------
    provider : str or None
        Provider key (case-insensitive). An empty or ``None`` value
        disables reranking and returns ``None``. Only ``"nvidia"`` is
        supported.
    model : str or None, optional
        Override the reranker model. Falls back to ``config["model"]``
        and then to ``"nv-rerank-qa-mistral-4b:1"``.
    config : dict[str, Any] or None, optional
        Extra options forwarded to ``NvidiaReranker``: ``api_key``
        (defaults to ``NVIDIA_API_KEY``), ``base_url``, ``truncate``
        (default ``"END"``) and ``top_n``. Keys with ``None`` values
        are dropped before construction.

    Returns
    -------
    NvidiaReranker or None
        A configured reranker, or ``None`` when reranking is disabled.

    Raises
    ------
    ValueError
        If ``provider`` is given but is not ``"nvidia"``.
    """
    name = (provider or "").strip().lower()
    if not name:
        return None
    if name != "nvidia":
        raise ValueError(f"unknown reranker provider {provider!r}; supported: 'nvidia'")
    cfg = dict(config or {})
    kwargs: dict[str, Any] = {
        "model": model or cfg.get("model") or "nv-rerank-qa-mistral-4b:1",
        "api_key": cfg.get("api_key") or os.environ.get("NVIDIA_API_KEY"),
        "base_url": cfg.get("base_url"),
        "truncate": cfg.get("truncate", "END"),
        "top_n": cfg.get("top_n"),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug("reranker: configured NvidiaReranker (model=%s)", kwargs["model"])
    return NvidiaReranker(**kwargs)


def make_translator(
    provider: str | None,
    *,
    model: str | None = None,
    config: dict[str, Any] | None = None,
) -> NvidiaTranslator | None:
    """Construct a translator client, or ``None`` when disabled.

    Parameters
    ----------
    provider : str or None
        Provider key (case-insensitive). An empty or ``None`` value
        disables translation and returns ``None``. Only ``"nvidia"``
        is supported.
    model : str or None, optional
        Override the translation model. Falls back to
        ``config["model"]`` and then to
        ``"nvidia/riva-translate-4b-instruct-v1_1"``.
    config : dict[str, Any] or None, optional
        Extra options forwarded to ``NvidiaTranslator``: ``api_key``
        (defaults to ``NVIDIA_API_KEY``), ``base_url``, ``temperature``
        (default ``0.0``) and ``max_tokens`` (default ``2048``). Keys
        with ``None`` values are dropped before construction.

    Returns
    -------
    NvidiaTranslator or None
        A configured translator, or ``None`` when translation is
        disabled.

    Raises
    ------
    ValueError
        If ``provider`` is given but is not ``"nvidia"``.
    """
    name = (provider or "").strip().lower()
    if not name:
        return None
    if name != "nvidia":
        raise ValueError(
            f"unknown translation provider {provider!r}; supported: 'nvidia'"
        )
    cfg = dict(config or {})
    kwargs: dict[str, Any] = {
        "model": model or cfg.get("model") or "nvidia/riva-translate-4b-instruct-v1_1",
        "api_key": cfg.get("api_key") or os.environ.get("NVIDIA_API_KEY"),
        "base_url": cfg.get("base_url"),
        "temperature": float(cfg.get("temperature", 0.0)),
        "max_tokens": cfg.get("max_tokens", 2048),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug("translator: configured NvidiaTranslator (model=%s)", kwargs["model"])
    return NvidiaTranslator(**kwargs)


def _nvidia(
    *,
    default_model: str,
    model: str | None,
    cfg: dict[str, Any],
) -> Embeddings:
    try:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    except ImportError as exc:  # pragma: no cover - install hint
        raise RuntimeError(
            "nvidia embeddings require langchain-nvidia-ai-endpoints. "
            "Install the agent_server with the [nvidia] extra."
        ) from exc

    api_key = cfg.get("api_key") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "nvidia embeddings: NVIDIA_API_KEY is not set. Provide it via "
            "the environment, user_settings.json, or the factory config."
        )
    kwargs: dict[str, Any] = {
        "model": model or default_model,
        "api_key": api_key,
        "truncate": cfg.get("truncate", "END"),
    }
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    if cfg.get("dimensions"):
        kwargs["dimensions"] = cfg["dimensions"]
    logger.debug("embeddings: configured NVIDIAEmbeddings (model=%s)", kwargs["model"])
    return NVIDIAEmbeddings(**kwargs)
