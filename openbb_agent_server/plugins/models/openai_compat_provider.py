"""OpenAI-compatible-endpoint provider."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from openbb_agent_server.plugins.models._validation import check_min, check_range
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider

_LOCAL_API_KEY_PLACEHOLDER = "EMPTY"
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}


class OpenAICompatProvider(ModelProvider):
    """Wrap ChatOpenAI against an arbitrary OpenAI-compatible endpoint.

    Construct a LangChain ``ChatOpenAI`` model pointed at any server that
    speaks the OpenAI chat-completions API (vLLM, llama.cpp, Ollama's
    OpenAI shim, hosted gateways, etc.). Sampling and reasoning parameters
    are validated at construction time and applied lazily in :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`.

    Attributes
    ----------
    name : str
        Plugin registry key (``"openai_compat"``).
    """

    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
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
        reasoning_budget: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, Any] | None = None,
        streaming: bool = True,
        extra_body: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Validate and store the model configuration.

        Numeric parameters are range- or minimum-checked immediately so
        misconfiguration fails fast; the actual client is built later in
        :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`.

        Parameters
        ----------
        base_url : str
            Base URL of the OpenAI-compatible endpoint. Required; there is
            no default.
        model_name : str
            Model identifier to request from the endpoint. Required.
        api_key : str or None, optional
            Static API key fallback. Per-run keys from the run context
            take precedence; a placeholder is used for local servers that
            do not require auth.
        organization : str or None, optional
            OpenAI organization id, forwarded as ``openai_organization``.
        temperature : float, optional
            Sampling temperature in ``[0.0, 2.0]``; defaults to ``0.0``.
        max_tokens : int or None, optional
            Maximum completion tokens; must be ``>= 1`` when set.
        top_p : float or None, optional
            Nucleus sampling probability in ``[0.0, 1.0]``.
        presence_penalty : float or None, optional
            Presence penalty in ``[-2.0, 2.0]``.
        frequency_penalty : float or None, optional
            Frequency penalty in ``[-2.0, 2.0]``.
        seed : int or None, optional
            Deterministic sampling seed forwarded to the endpoint.
        n : int, optional
            Number of completions to request; must be ``>= 1``.
        stop : list of str or str or None, optional
            Stop sequence(s) that halt generation.
        timeout : float or None, optional
            Per-request timeout in seconds, forwarded as
            ``request_timeout``.
        max_retries : int, optional
            Number of retries on transient errors; must be ``>= 0``.
        reasoning_effort : str or None, optional
            Reasoning effort level; one of ``none``, ``minimal``, ``low``,
            ``medium``, or ``high``.
        reasoning_budget : int or None, optional
            Token budget for reasoning. Must be ``-1`` (disabled) or
            ``>= 0``. Passed through ``model_kwargs``.
        chat_template_kwargs : dict or None, optional
            Extra chat-template arguments passed through ``model_kwargs``
            (e.g. for servers that expose template toggles).
        default_headers : dict of str to str or None, optional
            Default HTTP headers attached to every request.
        default_query : dict or None, optional
            Default query parameters attached to every request.
        streaming : bool, optional
            Whether to stream tokens from the endpoint; defaults to
            ``True``.
        extra_body : dict or None, optional
            Additional fields merged into the request body via
            ``model_kwargs``.
        model_kwargs : dict or None, optional
            Arbitrary extra keyword arguments merged last into
            ``model_kwargs``, overriding earlier entries on conflict.

        Raises
        ------
        ValueError
            If ``base_url`` or ``model_name`` is empty, if a numeric
            parameter is out of range, if ``reasoning_effort`` is not a
            recognised level, or if ``reasoning_budget`` is below ``-1``.
        """
        if not base_url:
            raise ValueError(
                "OpenAICompatProvider requires base_url (no default). Set "
                "[agent.model.config].base_url to your OpenAI-compatible server."
            )
        if not model_name:
            raise ValueError("OpenAICompatProvider requires model_name.")
        check_range("temperature", temperature, 0.0, 2.0)
        check_range("top_p", top_p, 0.0, 1.0)
        check_range("presence_penalty", presence_penalty, -2.0, 2.0)
        check_range("frequency_penalty", frequency_penalty, -2.0, 2.0)
        check_min("max_tokens", max_tokens, 1)
        check_min("n", n, 1)
        check_min("max_retries", max_retries, 0)
        if reasoning_effort is not None and reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(_REASONING_EFFORTS)}"
            )
        if reasoning_budget is not None and reasoning_budget < -1:
            raise ValueError(
                "reasoning_budget must be -1 (disabled) or >= 0 "
                f"(got {reasoning_budget})"
            )

        self._base_url = base_url
        self._model_name = model_name
        self._api_key = api_key
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
        self._reasoning_budget = reasoning_budget
        self._chat_template_kwargs = chat_template_kwargs
        self._default_headers = default_headers
        self._default_query = default_query
        self._streaming = streaming
        self._extra_body = extra_body
        self._model_kwargs = model_kwargs

    def build(  # noqa: PLR0912 — orchestrates many independent kwargs.
        self, ctx: RunContext, config: dict[str, Any]
    ) -> BaseChatModel:
        """Return a configured ``ChatOpenAI`` for the OpenAI-compat endpoint.

        Resolve the API key from the run context (``OPENAI_COMPAT_API_KEY``
        then ``OPENAI_API_KEY``), falling back to the constructor key and
        finally a local placeholder. Assemble only the keyword arguments
        that were explicitly configured, merging reasoning/template/extra
        options into ``model_kwargs``. ``config`` may override ``model_name``
        and ``base_url`` per call.

        Parameters
        ----------
        ctx : RunContext
            Run context supplying per-run ``api_keys``.
        config : dict of str to Any
            Per-call overrides; honoured keys are ``model_name`` and
            ``base_url``.

        Returns
        -------
        BaseChatModel
            A ``ChatOpenAI`` instance bound to the endpoint.

        Raises
        ------
        RuntimeError
            If ``langchain-openai`` is not installed.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover — install-hint path
            raise RuntimeError(
                "OpenAICompatProvider requires langchain-openai. "
                "Install the agent_server with the [openai] extra."
            ) from exc

        api_key = (
            ctx.api_keys.get("OPENAI_COMPAT_API_KEY")
            or ctx.api_keys.get("OPENAI_API_KEY")
            or self._api_key
            or _LOCAL_API_KEY_PLACEHOLDER
        )

        kwargs: dict[str, Any] = {
            "model": config.get("model_name", self._model_name),
            "base_url": config.get("base_url", self._base_url),
            "api_key": api_key,
            "temperature": self._temperature,
            "n": self._n,
            "max_retries": self._max_retries,
            "streaming": self._streaming,
        }
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
        if self._default_query is not None:
            kwargs["default_query"] = dict(self._default_query)

        merged_model_kwargs: dict[str, Any] = {}
        if self._extra_body:
            merged_model_kwargs.update(self._extra_body)
        if self._reasoning_budget is not None:
            merged_model_kwargs["reasoning_budget"] = self._reasoning_budget
        if self._chat_template_kwargs is not None:
            merged_model_kwargs["chat_template_kwargs"] = dict(
                self._chat_template_kwargs
            )
        if self._model_kwargs:
            merged_model_kwargs.update(self._model_kwargs)
        if merged_model_kwargs:
            kwargs["model_kwargs"] = merged_model_kwargs
        return ChatOpenAI(**kwargs)
