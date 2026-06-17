"""NVIDIA NIM translation adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger("openbb_agent_server.memory.translation")


class NvidiaTranslator:
    """Async translation client backed by an NVIDIA-hosted instruct model."""

    def __init__(
        self,
        *,
        model: str = "nvidia/riva-translate-4b-instruct-v1_1",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> None:
        """Configure the translator without building the underlying client.

        The chat client is created lazily on the first :meth:`translate`
        call. If ``api_key`` is omitted, the ``NVIDIA_API_KEY`` environment
        variable is used.

        Parameters
        ----------
        model : str, optional
            NVIDIA-hosted instruct model identifier used for translation.
        api_key : str or None, optional
            NVIDIA API key. Falls back to the ``NVIDIA_API_KEY``
            environment variable when not provided.
        base_url : str or None, optional
            Override for the NVIDIA endpoint base URL. When ``None`` the
            client's default endpoint is used.
        temperature : float, optional
            Sampling temperature passed to the model; defaults to ``0.0``
            for deterministic translations.
        max_tokens : int or None, optional
            Maximum number of output tokens. When ``None`` the limit is
            left to the model's default.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: Any | None = None

    def _build_client(self) -> Any:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as exc:  # pragma: no cover — install hint
            raise RuntimeError(
                "NvidiaTranslator requires langchain-nvidia-ai-endpoints. "
                "Install the agent_server with the [nvidia] extra."
            ) from exc

        if not self._api_key:
            raise RuntimeError(
                "NvidiaTranslator: NVIDIA_API_KEY is not set. Provide it "
                "via the environment, user_settings.json, or the "
                "constructor."
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "api_key": self._api_key,
            "temperature": self._temperature,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        return ChatNVIDIA(**kwargs)

    @staticmethod
    def _build_messages(
        text: str,
        source_language: str,
        target_language: str,
    ) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = (
            "You are a precise translation engine. Translate the user's "
            "text into the target language. Output ONLY the translation, "
            "with no preface, commentary, or quoted source. Preserve "
            "Markdown, code fences, bullet structure, numbers, and "
            "proper nouns exactly. If the text is already in the target "
            "language, return it unchanged."
        )
        src = source_language.strip() or "auto-detect"
        tgt = target_language.strip() or "English"
        user = f"Translate from {src} to {tgt}:\n\n{text}"
        return [SystemMessage(content=system), HumanMessage(content=user)]

    async def translate(
        self,
        text: str,
        *,
        source_language: str = "auto",
        target_language: str = "English",
    ) -> str:
        """Return ``text`` translated into ``target_language``.

        Build the chat client on first use, send a system/user prompt pair
        instructing the model to emit only the translation, and normalise
        the response into a plain string. Async invocation is preferred;
        a synchronous client is dispatched to a worker thread.

        Parameters
        ----------
        text : str
            Source text to translate. Empty or whitespace-only input
            returns ``""`` without contacting the model.
        source_language : str, optional
            Source language name. ``"auto"`` (the default) lets the model
            auto-detect the language.
        target_language : str, optional
            Target language name; defaults to ``"English"``.

        Returns
        -------
        str
            The translated text, stripped of surrounding whitespace.
            Returns ``""`` for empty input.

        Raises
        ------
        RuntimeError
            If the NVIDIA client cannot be built (missing
            ``langchain-nvidia-ai-endpoints`` or unset API key).
        """
        if not text or not text.strip():
            return ""
        if self._client is None:
            self._client = self._build_client()
        messages = self._build_messages(text, source_language, target_language)

        ainvoke = getattr(self._client, "ainvoke", None)
        if ainvoke is not None:
            response = await ainvoke(messages)
        else:
            response = await asyncio.to_thread(self._client.invoke, messages)

        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts).strip()
        return (str(content) if content is not None else "").strip()
