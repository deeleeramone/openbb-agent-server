"""NVIDIA NIM / NeMo / Foundation-Models provider."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from openbb_agent_server.plugins.models._validation import check_min, check_range
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider

_REASONING_EFFORTS = {"none", "low", "medium", "high"}


def _silence_unknown_tools_warning(model: Any) -> None:
    """Flip the per-model supports_tools flag inside ChatNVIDIA."""
    try:
        client = getattr(model, "_client", None)
        registry_entry = getattr(client, "model", None) if client else None
        if registry_entry is not None and hasattr(registry_entry, "supports_tools"):
            registry_entry.supports_tools = True
    except Exception:  # noqa: S110 — deliberate best-effort swallow
        pass


class NvidiaProvider(ModelProvider):
    """Build LangChain ``ChatNVIDIA`` chat models for NVIDIA endpoints.

    Targets NVIDIA NIM, NeMo, and the hosted Foundation Models API. The
    constructor validates and stores generation parameters; :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`
    instantiates the underlying ``ChatNVIDIA`` client per run.
    """

    name = "nvidia"

    def __init__(
        self,
        *,
        model_name: str = "meta/llama-3.3-70b-instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: list[str] | str | None = None,
        streaming: bool = True,
        reasoning_effort: str | None = None,
        reasoning_budget: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Validate and store ``ChatNVIDIA`` configuration parameters.

        Parameters
        ----------
        model_name : str, optional
            NVIDIA model identifier to call (for example
            ``"meta/llama-3.3-70b-instruct"``). May be overridden per run
            via the ``model_name`` config key in :meth:`~openbb_agent_server.runtime.plugins.ModelProvider.build`.
        api_key : str or None, optional
            NVIDIA API key. When ``None``, the key is taken from the run
            context's ``NVIDIA_API_KEY`` at build time.
        base_url : str or None, optional
            Base URL of the endpoint, e.g. a self-hosted NIM. When
            ``None``, the ``ChatNVIDIA`` default is used.
        temperature : float, optional
            Sampling temperature in ``[0.0, 2.0]``. Defaults to ``0.0``.
        max_tokens : int or None, optional
            Maximum tokens to generate (must be ``>= 1``). Takes
            precedence over ``max_completion_tokens`` and is passed as
            ``max_completion_tokens`` to ``ChatNVIDIA``.
        max_completion_tokens : int or None, optional
            Fallback completion-token limit (must be ``>= 1``) used only
            when ``max_tokens`` is ``None``.
        top_p : float or None, optional
            Nucleus-sampling probability mass in ``[0.0, 1.0]``.
        frequency_penalty : float or None, optional
            Frequency penalty in ``[-2.0, 2.0]``; forwarded via
            ``model_kwargs``.
        presence_penalty : float or None, optional
            Presence penalty in ``[-2.0, 2.0]``; forwarded via
            ``model_kwargs``.
        seed : int or None, optional
            Seed for deterministic sampling, when supported.
        stop : list of str, str, or None, optional
            One or more stop sequences that halt generation.
        streaming : bool, optional
            Whether to stream tokens. Defaults to ``True``; stored as the
            inverse ``disable_streaming`` flag passed to ``ChatNVIDIA``.
        reasoning_effort : str or None, optional
            Reasoning effort level; one of ``"none"``, ``"low"``,
            ``"medium"``, or ``"high"``. Forwarded via ``model_kwargs``.
        reasoning_budget : int or None, optional
            Token budget for reasoning. Must be ``-1`` (disabled) or
            ``>= 0``. Forwarded via ``model_kwargs``.
        chat_template_kwargs : dict or None, optional
            Extra arguments for the model's chat template; forwarded via
            ``model_kwargs``.
        default_headers : dict of str to str, or None, optional
            Additional HTTP headers sent with every request.
        extra_body : dict or None, optional
            Extra request-body fields merged into ``model_kwargs`` before
            the explicit parameters above.
        model_kwargs : dict or None, optional
            Arbitrary keyword arguments merged last into ``model_kwargs``,
            overriding any colliding keys.

        Raises
        ------
        ValueError
            If a numeric parameter falls outside its allowed range, if
            ``reasoning_effort`` is not a recognized level, or if
            ``reasoning_budget`` is less than ``-1``.
        """
        check_range("temperature", temperature, 0.0, 2.0)
        check_range("top_p", top_p, 0.0, 1.0)
        check_range("frequency_penalty", frequency_penalty, -2.0, 2.0)
        check_range("presence_penalty", presence_penalty, -2.0, 2.0)
        check_min("max_tokens", max_tokens, 1)
        check_min("max_completion_tokens", max_completion_tokens, 1)
        if reasoning_effort is not None and reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(_REASONING_EFFORTS)}"
            )
        if reasoning_budget is not None and reasoning_budget < -1:
            raise ValueError(
                "reasoning_budget must be -1 (disabled) or >= 0 "
                f"(got {reasoning_budget})"
            )

        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = (
            max_tokens if max_tokens is not None else max_completion_tokens
        )
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._seed = seed
        self._stop = stop
        self._streaming = streaming
        self._reasoning_effort = reasoning_effort
        self._reasoning_budget = reasoning_budget
        self._chat_template_kwargs = chat_template_kwargs
        self._default_headers = default_headers
        self._extra_body = extra_body
        self._model_kwargs = model_kwargs

    def build(  # noqa: PLR0912 — orchestrates many independent kwargs.
        self, ctx: RunContext, config: dict[str, Any]
    ) -> BaseChatModel:
        """Construct a configured ``ChatNVIDIA`` chat model.

        Resolves the API key from the run context (falling back to the
        constructor value), assembles the ``ChatNVIDIA`` keyword
        arguments from the stored configuration, merges penalty,
        reasoning, and pass-through options into ``model_kwargs``, and
        suppresses the spurious "unknown tools" warning emitted by some
        model registry entries.

        Parameters
        ----------
        ctx : RunContext
            Per-run context; its ``api_keys`` mapping supplies
            ``NVIDIA_API_KEY`` when no constructor key was given.
        config : dict
            Per-run overrides. A ``model_name`` key overrides the model
            chosen at construction time.

        Returns
        -------
        BaseChatModel
            A ready-to-use ``ChatNVIDIA`` instance.

        Raises
        ------
        RuntimeError
            If ``langchain-nvidia-ai-endpoints`` is not installed (the
            ``[nvidia]`` extra is required).
        """
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as exc:  # pragma: no cover — install-hint path
            raise RuntimeError(
                "NvidiaProvider requires langchain-nvidia-ai-endpoints. "
                "Install the agent_server with the [nvidia] extra."
            ) from exc

        api_key = ctx.api_keys.get("NVIDIA_API_KEY") or self._api_key

        kwargs: dict[str, Any] = {
            "model": config.get("model_name", self._model_name),
            "temperature": self._temperature,
            "disable_streaming": not self._streaming,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._max_tokens is not None:
            kwargs["max_completion_tokens"] = self._max_tokens
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if self._stop is not None:
            kwargs["stop"] = self._stop
        if self._default_headers is not None:
            kwargs["default_headers"] = dict(self._default_headers)

        merged_model_kwargs: dict[str, Any] = {}
        if self._extra_body:
            merged_model_kwargs.update(self._extra_body)
        if self._frequency_penalty is not None:
            merged_model_kwargs["frequency_penalty"] = self._frequency_penalty
        if self._presence_penalty is not None:
            merged_model_kwargs["presence_penalty"] = self._presence_penalty
        if self._reasoning_effort is not None:
            merged_model_kwargs["reasoning_effort"] = self._reasoning_effort
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
        model = ChatNVIDIA(**kwargs)
        _silence_unknown_tools_warning(model)
        return model
