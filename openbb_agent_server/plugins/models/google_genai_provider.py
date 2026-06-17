"""Google Gemini model provider."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from openbb_agent_server.plugins.models._validation import check_min, check_range
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider


class GoogleGenAIProvider(ModelProvider):
    """Build a ``ChatGoogleGenerativeAI`` (Google Gemini) chat model.

    A :class:`~openbb_agent_server.runtime.plugins.ModelProvider` that captures Gemini configuration at
    construction time and instantiates the LangChain
    ``ChatGoogleGenerativeAI`` client on demand in :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`.
    """

    name = "google_genai"

    def __init__(
        self,
        *,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        timeout: float | None = None,
        max_retries: int = 6,
        seed: int | None = None,
        stop: list[str] | None = None,
        safety_settings: dict[Any, Any] | None = None,
        base_url: str | None = None,
        additional_headers: dict[str, str] | None = None,
        cached_content: str | None = None,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        include_thoughts: bool | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Capture and validate the Gemini model configuration.

        Numeric arguments are range-checked eagerly so misconfiguration
        fails at construction rather than at run time.

        Parameters
        ----------
        model_name : str, default "gemini-2.5-flash"
            Default Gemini model id; may be overridden per run via the
            ``model_name`` key in :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`'s ``config``.
        api_key : str | None, optional
            Fallback API key used only when neither ``GOOGLE_API_KEY`` nor
            ``GEMINI_API_KEY`` is present in the run context.
        temperature : float, default 0.0
            Sampling temperature; must be in ``[0.0, 2.0]``.
        max_output_tokens : int | None, optional
            Maximum tokens to generate; must be ``>= 1`` when set.
        top_p : float | None, optional
            Nucleus-sampling probability mass; must be in ``[0.0, 1.0]``.
        top_k : int | None, optional
            Top-k sampling cutoff; must be ``>= 1`` when set.
        timeout : float | None, optional
            Per-request timeout in seconds.
        max_retries : int, default 6
            Number of automatic retries on transient errors; must be
            ``>= 0``.
        seed : int | None, optional
            Seed for reproducible sampling.
        stop : list[str] | None, optional
            Stop sequences that halt generation.
        safety_settings : dict[Any, Any] | None, optional
            Gemini safety-filter configuration.
        base_url : str | None, optional
            Override for the Generative Language API endpoint.
        additional_headers : dict[str, str] | None, optional
            Extra HTTP headers attached to every request.
        cached_content : str | None, optional
            Identifier of pre-cached content to reuse across requests.
        response_mime_type : str | None, optional
            Forced response MIME type (e.g. ``"application/json"``).
        response_schema : dict[str, Any] | None, optional
            Structured-output schema constraining the response.
        thinking_budget : int | None, optional
            Token budget allotted to reasoning; must be ``>= 0`` when set.
        thinking_level : str | None, optional
            Reasoning effort; one of ``"minimal"``, ``"low"``,
            ``"medium"``, ``"high"``.
        include_thoughts : bool | None, optional
            Whether to surface the model's reasoning trace in responses.
        labels : dict[str, str] | None, optional
            Billing/telemetry labels attached to requests.

        Raises
        ------
        ValueError
            If any numeric argument is out of range, or ``thinking_level``
            is not one of the four accepted values.
        """
        check_range("temperature", temperature, 0.0, 2.0)
        check_range("top_p", top_p, 0.0, 1.0)
        check_min("top_k", top_k, 1)
        check_min("max_output_tokens", max_output_tokens, 1)
        check_min("max_retries", max_retries, 0)
        check_min("thinking_budget", thinking_budget, 0)
        if thinking_level is not None and thinking_level not in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "thinking_level must be one of 'minimal', 'low', 'medium', 'high'"
            )

        self._model_name = model_name
        self._api_key = api_key
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._timeout = timeout
        self._max_retries = max_retries
        self._seed = seed
        self._stop = stop
        self._safety_settings = safety_settings
        self._base_url = base_url
        self._additional_headers = additional_headers
        self._cached_content = cached_content
        self._response_mime_type = response_mime_type
        self._response_schema = response_schema
        self._thinking_budget = thinking_budget
        self._thinking_level = thinking_level
        self._include_thoughts = include_thoughts
        self._labels = labels

    def build(  # noqa: PLR0912 — orchestrates many independent kwargs.
        self, ctx: RunContext, config: dict[str, Any]
    ) -> BaseChatModel:
        """Construct a configured ``ChatGoogleGenerativeAI`` for the run.

        Resolves the API key by preferring ``GOOGLE_API_KEY`` then
        ``GEMINI_API_KEY`` from the run context, falling back to the key
        supplied at construction. Only options explicitly set on this
        provider are forwarded to the client; unset options keep the
        library defaults.

        Parameters
        ----------
        ctx : RunContext
            The active run's context; its ``api_keys`` mapping supplies the
            Google credential.
        config : dict[str, Any]
            Per-run overrides. A ``model_name`` key overrides the default
            model; other keys are ignored.

        Returns
        -------
        BaseChatModel
            A ready-to-use ``ChatGoogleGenerativeAI`` instance.

        Raises
        ------
        RuntimeError
            If ``langchain-google-genai`` is not installed.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover — install-hint path
            raise RuntimeError(
                "GoogleGenAIProvider requires langchain-google-genai. "
                "Install the agent_server with the [google_genai] extra."
            ) from exc

        api_key = (
            ctx.api_keys.get("GOOGLE_API_KEY")
            or ctx.api_keys.get("GEMINI_API_KEY")
            or self._api_key
        )

        kwargs: dict[str, Any] = {
            "model": config.get("model_name", self._model_name),
            "temperature": self._temperature,
            "max_retries": self._max_retries,
        }
        if api_key:
            kwargs["google_api_key"] = api_key
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._top_k is not None:
            kwargs["top_k"] = self._top_k
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if self._stop is not None:
            kwargs["stop"] = list(self._stop)
        if self._safety_settings is not None:
            kwargs["safety_settings"] = self._safety_settings
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._additional_headers is not None:
            kwargs["additional_headers"] = dict(self._additional_headers)
        if self._cached_content is not None:
            kwargs["cached_content"] = self._cached_content
        if self._response_mime_type is not None:
            kwargs["response_mime_type"] = self._response_mime_type
        if self._response_schema is not None:
            kwargs["response_schema"] = self._response_schema
        if self._thinking_budget is not None:
            kwargs["thinking_budget"] = self._thinking_budget
        if self._thinking_level is not None:
            kwargs["thinking_level"] = self._thinking_level
        if self._include_thoughts is not None:
            kwargs["include_thoughts"] = self._include_thoughts
        if self._labels is not None:
            kwargs["labels"] = dict(self._labels)
        return ChatGoogleGenerativeAI(**kwargs)
