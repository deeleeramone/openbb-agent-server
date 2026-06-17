"""``groq_audio`` tool source — speech-to-text via Groq Whisper."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.plugins.models.groq_rate_limiter import (
    get_limiter,
)
from openbb_agent_server.runtime.context import FileRef, RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.groq_audio")

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "whisper-large-v3-turbo"
_VALID_RESPONSE_FORMATS = frozenset({"json", "verbose_json", "text", "srt", "vtt"})
_VALID_TIMESTAMP_GRANULARITIES = frozenset({"word", "segment"})


class _TranscribeArgs(BaseModel):
    audio_name: str | None = Field(
        default=None,
        description=(
            "Name of an uploaded audio file (matches QueryRequest.uploaded_files[].name). "
            "Either ``audio_name`` or ``audio_url`` must be set."
        ),
    )
    audio_url: str | None = Field(
        default=None,
        description="Public URL to an audio file. Mutually exclusive with audio_name.",
    )
    model: str = Field(
        default=_DEFAULT_MODEL,
        description="Whisper model — 'whisper-large-v3' or 'whisper-large-v3-turbo'.",
    )
    language: str | None = Field(
        default=None,
        description="ISO-639-1 language code of the audio (e.g. 'en'). Auto-detected if blank.",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional context to bias the transcript (vocab, names, style).",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    response_format: str = Field(
        default="verbose_json",
        description="json | verbose_json | text | srt | vtt",
    )
    timestamp_granularities: list[str] | None = Field(
        default=None,
        description="['word'] / ['segment'] / ['word','segment']. Forces verbose_json.",
    )


class GroqAudioToolSource(ToolSource):
    """Expose Groq Whisper transcription / translation as agent tools."""

    name = "groq_audio"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        default_model: str = _DEFAULT_MODEL,
        default_language: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        """Configure the default Groq Whisper connection and retry policy.

        Values set here act as fallbacks; the per-run ``config`` passed to
        :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools` can override every one of them.

        Parameters
        ----------
        api_key : str or None, optional
            Default Groq API key. Overridden at run time by
            ``ctx.api_keys['GROQ_API_KEY']`` or ``config['api_key']``.
        base_url : str, optional
            Base URL of the Groq OpenAI-compatible API. Trailing slashes
            are stripped.
        default_model : str, optional
            Whisper model used when a request omits one. Must be
            ``'whisper-large-v3'`` or ``'whisper-large-v3-turbo'``.
        default_language : str or None, optional
            ISO-639-1 language hint applied to transcriptions when a
            request leaves ``language`` unset.
        timeout : float, optional
            Per-request HTTP timeout in seconds for the Groq call.
        max_retries : int, optional
            Maximum retry attempts for 429/5xx responses and network
            errors. Must be non-negative.

        Raises
        ------
        ValueError
            If ``max_retries`` is negative, or ``default_model`` is not one
            of the two supported Whisper models.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if default_model not in {"whisper-large-v3", "whisper-large-v3-turbo"}:
            raise ValueError(
                "default_model must be 'whisper-large-v3' or 'whisper-large-v3-turbo'"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._default_language = default_language
        self._timeout = timeout
        self._max_retries = max_retries

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Build the Groq audio tools bound to this run's key and settings.

        Resolves the API key and connection settings from ``ctx``,
        ``config``, and the constructor defaults (in that precedence), then
        returns two LangChain tools that call Groq Whisper.

        Parameters
        ----------
        ctx : RunContext
            The active run context; supplies ``api_keys`` and the uploaded
            files that ``audio_name`` resolves against.
        config : dict[str, Any]
            Per-run overrides — recognised keys are ``api_key``,
            ``base_url``, ``default_model``, ``default_language``,
            ``timeout``, and ``max_retries``.

        Returns
        -------
        list[BaseTool]
            Two structured tools: ``transcribe_audio`` (speech-to-text in
            the source language) and ``translate_audio`` (speech-to-text
            translated to English).

        Raises
        ------
        RuntimeError
            If no Groq API key can be resolved from the context, config, or
            constructor default.
        """
        api_key = (
            ctx.api_keys.get("GROQ_API_KEY") or config.get("api_key") or self._api_key
        )
        if not api_key:
            raise RuntimeError(
                "groq_audio: no GROQ_API_KEY available. Forward it via "
                "QueryRequest.api_keys, set [agent.tool_source_config.groq_audio].api_key, "
                "or pass api_key= to the constructor."
            )

        base_url = config.get("base_url", self._base_url).rstrip("/")
        default_model = config.get("default_model", self._default_model)
        default_language = config.get("default_language", self._default_language)
        timeout = float(config.get("timeout", self._timeout))
        max_retries = int(config.get("max_retries", self._max_retries))

        async def _post_audio(
            *,
            endpoint: str,
            args: _TranscribeArgs,
        ) -> dict[str, Any]:
            return await _transcribe(
                ctx=ctx,
                endpoint=endpoint,
                base_url=base_url,
                api_key=api_key,
                args=args,
                default_model=default_model,
                default_language=default_language,
                timeout=timeout,
                max_retries=max_retries,
            )

        async def transcribe_audio(**kwargs: Any) -> dict[str, Any]:
            return await _post_audio(
                endpoint="/audio/transcriptions",
                args=_TranscribeArgs(**kwargs),
            )

        async def translate_audio(**kwargs: Any) -> dict[str, Any]:
            args = _TranscribeArgs(**kwargs)
            return await _post_audio(endpoint="/audio/translations", args=args)

        return [
            StructuredTool.from_function(
                coroutine=transcribe_audio,
                name="transcribe_audio",
                description=(
                    "Transcribe an audio file to text using Groq Whisper. "
                    "Accepts either ``audio_name`` (a file uploaded to the "
                    "current chat) or ``audio_url``. Returns text, optional "
                    "segments / words with timestamps, and the audio "
                    "duration in seconds. Auto-detects the language unless "
                    "``language`` is set. Default model is "
                    f"{_DEFAULT_MODEL!r} (faster) — switch to "
                    "'whisper-large-v3' for the highest accuracy."
                ),
                args_schema=_TranscribeArgs,
            ),
            StructuredTool.from_function(
                coroutine=translate_audio,
                name="translate_audio",
                description=(
                    "Same as transcribe_audio but the output is "
                    "English-translated regardless of source language."
                ),
                args_schema=_TranscribeArgs,
            ),
        ]


def _resolve_audio(ctx: RunContext, args: _TranscribeArgs) -> tuple[str, bytes, str]:
    """Return ``(filename, bytes, mime)`` for the requested audio."""
    if args.audio_url:
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(args.audio_url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"groq_audio: could not fetch audio_url {args.audio_url!r}: {exc}"
            ) from exc
        name = args.audio_url.rsplit("/", 1)[-1] or "audio.mp3"
        mime = resp.headers.get("content-type") or "application/octet-stream"
        return name, resp.content, mime

    if args.audio_name:
        match = _find_uploaded(ctx.uploaded_files, args.audio_name)
        if match is None:
            raise RuntimeError(
                f"groq_audio: no uploaded file named {args.audio_name!r}; "
                "available: " + ", ".join(f.name for f in ctx.uploaded_files)
            )
        return match

    raise RuntimeError("groq_audio: must supply either audio_name or audio_url")


def _find_uploaded(
    uploaded: tuple[FileRef, ...], name: str
) -> tuple[str, bytes, str] | None:
    for ref in uploaded:
        if ref.name != name:
            continue
        if ref.data_base64:
            return (
                ref.name,
                base64.b64decode(ref.data_base64),
                (ref.mime or "application/octet-stream"),
            )
        if ref.url:
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.get(ref.url)
                    resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"groq_audio: could not fetch {ref.url!r}: {exc}"
                ) from exc
            return (
                ref.name,
                resp.content,
                ref.mime
                or (resp.headers.get("content-type") or "application/octet-stream"),
            )
    return None


async def _transcribe(  # noqa: PLR0912
    *,
    ctx: RunContext,
    endpoint: str,
    base_url: str,
    api_key: str,
    args: _TranscribeArgs,
    default_model: str,
    default_language: str | None,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    if args.response_format not in _VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"response_format must be one of {sorted(_VALID_RESPONSE_FORMATS)}"
        )
    if args.timestamp_granularities:
        for g in args.timestamp_granularities:
            if g not in _VALID_TIMESTAMP_GRANULARITIES:
                raise ValueError(
                    "timestamp_granularities entries must be 'word' or 'segment'"
                )

    model = args.model or default_model
    name, audio_bytes, mime = _resolve_audio(ctx, args)
    limiter = get_limiter(api_key=api_key, model_name=model)

    await limiter.aacquire()

    data: dict[str, Any] = {
        "model": model,
        "response_format": args.response_format,
        "temperature": str(args.temperature),
    }
    language = args.language or default_language
    if language and endpoint.endswith("/transcriptions"):
        data["language"] = language
    if args.prompt:
        data["prompt"] = args.prompt
    if args.timestamp_granularities:
        data["timestamp_granularities[]"] = list(args.timestamp_granularities)

    files = {"file": (name, io.BytesIO(audio_bytes), mime)}

    async with httpx.AsyncClient(timeout=timeout) as client:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= max_retries:
            try:
                resp = await client.post(
                    f"{base_url}{endpoint}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                    files=files,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"groq_audio: network error after {max_retries} retries: {exc}"
                    ) from exc
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"groq_audio: HTTP {resp.status_code} after {max_retries} retries: "
                        f"{resp.text[:300]}"
                    )
                wait = _retry_after(resp) or _backoff(attempt)
                logger.warning(
                    "groq_audio: HTTP %s, retrying in %.2fs (attempt %d/%d)",
                    resp.status_code,
                    wait,
                    attempt,
                    max_retries,
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"groq_audio: HTTP {resp.status_code}: {resp.text[:300]}"
                )

            payload: dict[str, Any]
            if args.response_format in {"json", "verbose_json"}:
                payload = resp.json()
            else:
                payload = {"text": resp.text}

            duration = float(payload.get("duration") or 0.0)
            if duration > 0:
                limiter.record_audio_seconds(duration)

            payload["_groq"] = {
                "model": model,
                "endpoint": endpoint,
                "duration_seconds": duration,
                "rate_limiter_snapshot": limiter.snapshot(),
            }
            return payload

        if last_exc is None:  # pragma: no cover
            raise RuntimeError("groq_audio retry loop exited without an exception")
        raise last_exc  # pragma: no cover


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    """Return an exponential backoff delay capped at 30 seconds."""
    return min(30.0, 0.5 * (2 ** (attempt - 1)))
