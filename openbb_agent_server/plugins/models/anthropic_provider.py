"""Anthropic Claude model provider."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from openbb_agent_server.plugins.models._validation import check_min, check_range
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider


class AnthropicProvider(ModelProvider):
    """Build LangChain ``ChatAnthropic`` models for Claude.

    A :class:`~openbb_agent_server.runtime.plugins.ModelProvider` plugin (registered name ``"anthropic"``)
    that captures generation defaults at construction time and
    materialises a chat model per run via :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        model_name: str = "claude-opus-4-7",
        api_key: str | None = None,
        api_url: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        thinking: dict[str, Any] | None = None,
        betas: list[str] | None = None,
        streaming: bool = True,
    ) -> None:
        """Store and validate Claude generation defaults.

        Parameters
        ----------
        model_name : str, optional
            Default Claude model id; overridable per run via the
            ``model_name`` config key in :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`.
        api_key : str or None, optional
            Anthropic API key; a run-time ``ANTHROPIC_API_KEY`` takes
            precedence over this value.
        api_url : str or None, optional
            Override base URL for the Anthropic API.
        max_tokens : int, optional
            Maximum tokens to generate per completion. Must be >= 1.
        temperature : float, optional
            Sampling temperature in ``[0.0, 1.0]``.
        top_p : float or None, optional
            Nucleus sampling threshold in ``[0.0, 1.0]``; omitted when
            ``None``.
        top_k : int or None, optional
            Top-k sampling cutoff (>= 1); omitted when ``None``.
        stop_sequences : list of str or None, optional
            Sequences that halt generation; omitted when ``None``.
        timeout : float or None, optional
            Per-request timeout in seconds; omitted when ``None``.
        max_retries : int, optional
            Number of automatic retries on transient errors. Must be
            >= 0.
        default_headers : dict or None, optional
            Extra HTTP headers attached to each request.
        thinking : dict or None, optional
            Anthropic extended-thinking configuration block.
        betas : list of str or None, optional
            Anthropic beta feature flags to opt into.
        streaming : bool, optional
            Whether the built model streams tokens.

        Raises
        ------
        ValueError
            If ``temperature``/``top_p`` fall outside ``[0.0, 1.0]`` or
            ``top_k``/``max_tokens``/``max_retries`` fall below their
            minimums.
        """
        check_range("temperature", temperature, 0.0, 1.0)
        check_range("top_p", top_p, 0.0, 1.0)
        check_min("top_k", top_k, 1)
        check_min("max_tokens", max_tokens, 1)
        check_min("max_retries", max_retries, 0)

        self._model_name = model_name
        self._api_key = api_key
        self._api_url = api_url
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._stop_sequences = stop_sequences
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers = default_headers
        self._thinking = thinking
        self._betas = betas
        self._streaming = streaming

    def build(self, ctx: RunContext, config: dict[str, Any]) -> BaseChatModel:
        """Build a ``ChatAnthropic`` model for the current run.

        Merges the stored defaults with the run context and per-call
        ``config``. The API key is read from ``ctx.api_keys`` first,
        falling back to the configured key. Optional parameters
        (``top_p``, ``top_k``, ``stop_sequences``, ``timeout``,
        ``default_headers``, ``thinking``, ``betas``) are only forwarded
        when they were set.

        Parameters
        ----------
        ctx : RunContext
            Active run context; supplies ``api_keys`` for credentials.
        config : dict
            Per-run overrides. Only ``model_name`` is consulted (falling
            back to the constructor default).

        Returns
        -------
        BaseChatModel
            A configured ``ChatAnthropic`` chat model instance.
        """
        api_key = ctx.api_keys.get("ANTHROPIC_API_KEY") or self._api_key

        kwargs: dict[str, Any] = {
            "model": config.get("model_name", self._model_name),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "max_retries": self._max_retries,
            "streaming": self._streaming,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self._api_url is not None:
            kwargs["anthropic_api_url"] = self._api_url
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._top_k is not None:
            kwargs["top_k"] = self._top_k
        if self._stop_sequences is not None:
            kwargs["stop_sequences"] = list(self._stop_sequences)
        if self._timeout is not None:
            kwargs["default_request_timeout"] = self._timeout
        if self._default_headers is not None:
            kwargs["default_headers"] = dict(self._default_headers)
        if self._thinking is not None:
            kwargs["thinking"] = self._thinking
        if self._betas is not None:
            kwargs["betas"] = list(self._betas)
        return ChatAnthropic(**kwargs)
