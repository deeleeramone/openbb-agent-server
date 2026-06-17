"""OpenAI Chat model provider."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from openbb_agent_server.plugins.models._validation import check_min, check_range
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider


class OpenAIProvider(ModelProvider):
    """Construct LangChain ``ChatOpenAI`` chat models from config.

    Validate the sampling and retry parameters at construction time and
    defer the actual ``ChatOpenAI`` import and instantiation to
    :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`, so the optional ``langchain-openai`` dependency is
    only required when a model is requested.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        n: int = 1,
        stop: list[str] | str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        reasoning_effort: str | None = None,
        default_headers: dict[str, str] | None = None,
        streaming: bool = True,
    ) -> None:
        """Validate and store the OpenAI chat model parameters.

        Parameters
        ----------
        model_name : str, default "gpt-4o"
            Default OpenAI model id; may be overridden per call via the
            ``model_name`` key in :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`'s ``config``.
        api_key : str or None, optional
            API key used when the run context does not supply
            ``OPENAI_API_KEY``.
        base_url : str or None, optional
            Override base URL for an OpenAI-compatible endpoint.
        organization : str or None, optional
            OpenAI organization id passed as ``openai_organization``.
        temperature : float, default 0.0
            Sampling temperature; must be within ``[0.0, 2.0]``.
        max_tokens : int or None, optional
            Maximum tokens to generate; must be at least 1 if set.
        top_p : float or None, optional
            Nucleus sampling probability; must be within ``[0.0, 1.0]``.
        presence_penalty : float or None, optional
            Presence penalty; must be within ``[-2.0, 2.0]``.
        frequency_penalty : float or None, optional
            Frequency penalty; must be within ``[-2.0, 2.0]``.
        seed : int or None, optional
            Sampling seed for reproducible outputs.
        n : int, default 1
            Number of completions to request; must be at least 1.
        stop : list of str or str or None, optional
            Stop sequence(s) that halt generation.
        timeout : float or None, optional
            Per-request timeout in seconds, passed as ``request_timeout``.
        max_retries : int, default 2
            Retry attempts on transient errors; must be at least 0.
        reasoning_effort : str or None, optional
            Reasoning effort for reasoning models; one of ``"minimal"``,
            ``"low"``, ``"medium"``, or ``"high"`` when set.
        default_headers : dict of str to str or None, optional
            Extra HTTP headers sent with each request.
        streaming : bool, default True
            Whether to stream tokens from the API.

        Raises
        ------
        ValueError
            If a numeric parameter falls outside its allowed range or
            ``reasoning_effort`` is not one of the accepted values.
        """
        check_range("temperature", temperature, 0.0, 2.0)
        check_range("top_p", top_p, 0.0, 1.0)
        check_range("presence_penalty", presence_penalty, -2.0, 2.0)
        check_range("frequency_penalty", frequency_penalty, -2.0, 2.0)
        check_min("max_tokens", max_tokens, 1)
        check_min("n", n, 1)
        check_min("max_retries", max_retries, 0)
        if reasoning_effort is not None and reasoning_effort not in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "reasoning_effort must be one of 'minimal', 'low', 'medium', 'high'"
            )

        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty
        self._seed = seed
        self._n = n
        self._stop = stop
        self._timeout = timeout
        self._max_retries = max_retries
        self._reasoning_effort = reasoning_effort
        self._default_headers = default_headers
        self._streaming = streaming

    def build(  # noqa: PLR0912 — orchestrates many independent kwargs.
        self, ctx: RunContext, config: dict[str, Any]
    ) -> BaseChatModel:
        """Build a configured ``ChatOpenAI`` instance.

        Assemble the constructor keyword arguments from the stored
        parameters, resolving the API key from the run context first and
        the per-call ``model_name`` override, then only forwarding
        optional kwargs that were explicitly set.

        Parameters
        ----------
        ctx : RunContext
            Run context whose ``api_keys`` mapping may supply
            ``OPENAI_API_KEY``, taking precedence over the configured key.
        config : dict of str to Any
            Per-call overrides; only ``model_name`` is read here.

        Returns
        -------
        BaseChatModel
            A ``ChatOpenAI`` model ready for invocation.

        Raises
        ------
        RuntimeError
            If ``langchain-openai`` is not installed.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover — install-hint path
            raise RuntimeError(
                "OpenAIProvider requires langchain-openai. "
                "Install the agent_server with the [openai] extra."
            ) from exc

        api_key = ctx.api_keys.get("OPENAI_API_KEY") or self._api_key

        kwargs: dict[str, Any] = {
            "model": config.get("model_name", self._model_name),
            "temperature": self._temperature,
            "n": self._n,
            "max_retries": self._max_retries,
            "streaming": self._streaming,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._organization is not None:
            kwargs["openai_organization"] = self._organization
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._presence_penalty is not None:
            kwargs["presence_penalty"] = self._presence_penalty
        if self._frequency_penalty is not None:
            kwargs["frequency_penalty"] = self._frequency_penalty
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if self._stop is not None:
            kwargs["stop"] = self._stop
        if self._timeout is not None:
            kwargs["request_timeout"] = self._timeout
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self._default_headers is not None:
            kwargs["default_headers"] = dict(self._default_headers)
        return ChatOpenAI(**kwargs)
