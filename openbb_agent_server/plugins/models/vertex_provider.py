"""Google Vertex AI model provider."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from openbb_agent_server.plugins.models._validation import check_min, check_range
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider


class VertexProvider(ModelProvider):
    """Provide Gemini chat models served through Google Vertex AI.

    Wrap ``langchain_google_genai.ChatGoogleGenerativeAI`` with
    ``vertexai=True`` so requests are routed to a Google Cloud project /
    region rather than the public Generative Language API.
    """

    name = "vertex"

    def __init__(
        self,
        *,
        model_name: str = "gemini-2.0-flash-001",
        project: str | None = None,
        location: str = "us-central1",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        max_retries: int = 6,
        timeout: float | None = None,
        safety_settings: dict[Any, Any] | None = None,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        include_thoughts: bool | None = None,
        cached_content: str | None = None,
        labels: dict[str, str] | None = None,
        additional_headers: dict[str, str] | None = None,
        credentials: Any = None,
    ) -> None:
        """Validate and store the Vertex AI generation settings.

        Parameters
        ----------
        model_name : str
            Default Gemini model id; overridable per run via ``config``.
        project : str or None
            Google Cloud project id; falls back to the ambient
            credentials' project when ``None``.
        location : str
            Vertex AI region, e.g. ``"us-central1"``.
        temperature : float
            Sampling temperature; must be in ``[0.0, 2.0]``.
        max_output_tokens : int or None
            Maximum tokens to generate; must be ``>= 1`` when set.
        top_p : float or None
            Nucleus-sampling cutoff; must be in ``[0.0, 1.0]`` when set.
        top_k : int or None
            Top-k sampling cutoff; must be ``>= 1`` when set.
        seed : int or None
            Deterministic sampling seed.
        stop : list of str or None
            Stop sequences that halt generation.
        max_retries : int
            Maximum transport retry attempts; must be ``>= 0``.
        timeout : float or None
            Per-request timeout in seconds.
        safety_settings : dict or None
            Vertex AI harm-category safety thresholds.
        response_mime_type : str or None
            Requested response MIME type, e.g. ``"application/json"``.
        response_schema : dict of str to Any or None
            JSON schema constraining structured responses.
        thinking_budget : int or None
            Token budget for model thinking; must be ``>= 0`` when set.
        thinking_level : str or None
            Thinking effort; one of ``"minimal"``, ``"low"``,
            ``"medium"``, ``"high"``.
        include_thoughts : bool or None
            Whether to return the model's thought summaries.
        cached_content : str or None
            Resource name of cached content to reuse across requests.
        labels : dict of str to str or None
            Billing / tracking labels attached to requests.
        additional_headers : dict of str to str or None
            Extra HTTP headers sent with each request.
        credentials : Any
            Explicit Google credentials object; defaults to ambient
            application-default credentials when ``None``.

        Raises
        ------
        ValueError
            If a numeric setting is out of range or ``thinking_level`` is
            not one of the accepted values.
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
        self._project = project
        self._location = location
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._seed = seed
        self._stop = stop
        self._max_retries = max_retries
        self._timeout = timeout
        self._safety_settings = safety_settings
        self._response_mime_type = response_mime_type
        self._response_schema = response_schema
        self._thinking_budget = thinking_budget
        self._thinking_level = thinking_level
        self._include_thoughts = include_thoughts
        self._cached_content = cached_content
        self._labels = labels
        self._additional_headers = additional_headers
        self._credentials = credentials

    def build(  # noqa: PLR0912 — orchestrates many independent kwargs.
        self, ctx: RunContext, config: dict[str, Any]
    ) -> BaseChatModel:
        """Build a Vertex-mode ``ChatGoogleGenerativeAI`` model.

        Assemble the kwargs from the stored settings, including only the
        optional ones that were explicitly configured, and instantiate the
        chat model in Vertex AI mode.

        Parameters
        ----------
        ctx : RunContext
            Active run context (unused here but part of the provider
            interface).
        config : dict of str to Any
            Per-run configuration; ``"model_name"`` overrides the default
            Gemini model when present.

        Returns
        -------
        BaseChatModel
            A configured ``ChatGoogleGenerativeAI`` instance.

        Raises
        ------
        RuntimeError
            If ``langchain-google-genai`` is not installed (install the
            ``[vertex]`` extra).
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover — install-hint path
            raise RuntimeError(
                "VertexProvider requires langchain-google-genai. "
                "Install the agent_server with the [vertex] extra."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": config.get("model_name", self._model_name),
            "vertexai": True,
            "location": self._location,
            "temperature": self._temperature,
            "max_retries": self._max_retries,
        }
        if self._project is not None:
            kwargs["project"] = self._project
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._top_k is not None:
            kwargs["top_k"] = self._top_k
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if self._stop is not None:
            kwargs["stop"] = list(self._stop)
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        if self._safety_settings is not None:
            kwargs["safety_settings"] = self._safety_settings
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
        if self._cached_content is not None:
            kwargs["cached_content"] = self._cached_content
        if self._labels is not None:
            kwargs["labels"] = dict(self._labels)
        if self._additional_headers is not None:
            kwargs["additional_headers"] = dict(self._additional_headers)
        if self._credentials is not None:
            kwargs["credentials"] = self._credentials
        return ChatGoogleGenerativeAI(**kwargs)
