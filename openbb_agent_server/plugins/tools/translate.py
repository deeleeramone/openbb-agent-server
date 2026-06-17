"""``translate`` tool source."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.memory.translation import NvidiaTranslator
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.translate")


class _TranslateArgs(BaseModel):
    text: str = Field(description="Text to translate.")
    source_language: str = Field(
        default="auto",
        description=(
            "Source language name (English label, e.g. 'French', "
            "'Mandarin', 'Spanish'). 'auto' lets the model detect."
        ),
    )
    target_language: str = Field(
        default="English",
        description="Target language name (English label).",
    )


class NvidiaTranslateToolSource(ToolSource):
    """Bind one NvidiaTranslator per agent run."""

    name = "translate"

    def __init__(
        self,
        *,
        model: str = "nvidia/riva-translate-4b-instruct-v1_1",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> None:
        """Store default NVIDIA translator settings for later tool binding.

        These values act as fallback defaults; each call to :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools` may
        override them via the per-run ``config`` mapping or the run context.

        Parameters
        ----------
        model : str, optional
            NVIDIA model id used for translation. Defaults to
            ``"nvidia/riva-translate-4b-instruct-v1_1"``.
        api_key : str or None, optional
            NVIDIA API key. Used only when neither the run context nor the
            per-run config supplies one.
        base_url : str or None, optional
            Override for the NVIDIA API base URL.
        temperature : float, optional
            Sampling temperature passed to the translator. Defaults to ``0.0``.
        max_tokens : int or None, optional
            Maximum tokens for the translation response. Defaults to ``2048``.
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Build a single ``translate`` tool bound to an NvidiaTranslator.

        Resolve the API key (run context, then per-run config, then the
        instance default), construct an :class:`~openbb_agent_server.memory.translation.NvidiaTranslator` whose model,
        base URL, temperature, and max-tokens may be overridden via ``config``,
        and wrap its async ``translate`` coroutine in a LangChain
        :class:`StructuredTool`. The wrapped tool catches any error and returns
        a short ``"translation failed: ..."`` string instead of raising.

        Parameters
        ----------
        ctx : RunContext
            Per-run context; its ``api_keys`` mapping is checked for
            ``"NVIDIA_API_KEY"`` first.
        config : dict[str, Any]
            Per-run overrides. Recognized keys: ``api_key``, ``model``,
            ``base_url``, ``temperature``, and ``max_tokens``.

        Returns
        -------
        list[BaseTool]
            A single-element list holding the ``translate`` structured tool.
        """
        api_key = (
            ctx.api_keys.get("NVIDIA_API_KEY") or config.get("api_key") or self._api_key
        )
        translator = NvidiaTranslator(
            model=config.get("model", self._model),
            api_key=api_key,
            base_url=config.get("base_url", self._base_url),
            temperature=float(config.get("temperature", self._temperature)),
            max_tokens=config.get("max_tokens", self._max_tokens),
        )

        async def translate(
            text: str,
            source_language: str = "auto",
            target_language: str = "English",
        ) -> str:
            try:
                return await translator.translate(
                    text,
                    source_language=source_language,
                    target_language=target_language,
                )
            except Exception as exc:
                logger.warning("translate tool failed: %s", exc)
                return f"translation failed: {exc}"

        return [
            StructuredTool.from_function(
                coroutine=translate,
                name="translate",
                description=(
                    "Translate a piece of text from one language to another "
                    "using NVIDIA's Riva translate model. Inputs: ``text`` "
                    "(string to translate), ``source_language`` (default "
                    "'auto'), ``target_language`` (default 'English'). "
                    "Markdown, code fences, and numbers are preserved. "
                    "Returns the translated string; on failure, returns a "
                    "short error message starting with 'translation failed:'."
                ),
                args_schema=_TranslateArgs,
            )
        ]
