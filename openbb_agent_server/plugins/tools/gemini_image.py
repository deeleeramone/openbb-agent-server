"""``gemini_image`` tool source — text-to-image and image-edit via Gemini."""

from __future__ import annotations

import asyncio
import base64
import logging
import secrets
from typing import Any

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import FileRef, RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.gemini_image")

_DEFAULT_GENERATE_MODEL = "gemini-2.5-flash-image"
_DEFAULT_EDIT_MODEL = "gemini-2.5-flash-image"
_DEFAULT_IMAGEN_MODEL = "imagen-4.0-generate-001"
_VALID_BACKENDS = frozenset({"gemini", "imagen"})
_VALID_ASPECT_RATIOS = frozenset({"1:1", "3:4", "4:3", "9:16", "16:9"})


class _GenerateArgs(BaseModel):
    prompt: str = Field(description="Text prompt describing the desired image.")
    model: str | None = Field(
        default=None,
        description=(
            "Override the configured default. For ``backend='gemini'`` use a "
            "Gemini image model (e.g. 'gemini-2.5-flash-image'); for "
            "``backend='imagen'`` use an Imagen model (e.g. "
            "'imagen-4.0-generate-001')."
        ),
    )
    backend: str = Field(
        default="gemini",
        description="``gemini`` (multimodal) or ``imagen`` (dedicated image API).",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description="One of '1:1', '3:4', '4:3', '9:16', '16:9'. Imagen only.",
    )
    number_of_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="How many images to return (Imagen up to 4).",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="What to avoid in the image. Imagen only.",
    )
    seed: int | None = Field(default=None)


class _EditArgs(BaseModel):
    prompt: str = Field(description="Instruction for the edit (e.g. 'add a hat').")
    base_image_name: str | None = Field(
        default=None,
        description=(
            "Name of an uploaded image file (matches "
            "QueryRequest.uploaded_files[].name). Either ``base_image_name`` "
            "or ``base_image_url`` must be set."
        ),
    )
    base_image_url: str | None = Field(
        default=None,
        description="Public URL of the source image.",
    )
    model: str | None = Field(default=None)
    seed: int | None = Field(default=None)


class GeminiImageToolSource(ToolSource):
    """Expose Gemini image generation + editing as agent tools.

    A :class:`~openbb_agent_server.runtime.plugins.ToolSource` that, when bound to a run, yields two
    LangChain tools: ``generate_image`` (text-to-image via the Gemini or
    Imagen backend) and ``edit_image`` (instruction-driven edit of an
    uploaded or URL-addressed image). Generated images are emitted as
    Workspace image artifacts and the tools return their UUIDs.
    """

    name = "gemini_image"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_generate_model: str = _DEFAULT_GENERATE_MODEL,
        default_edit_model: str = _DEFAULT_EDIT_MODEL,
        default_imagen_model: str = _DEFAULT_IMAGEN_MODEL,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        """Initialize the tool source with default models and limits.

        These values act as fallbacks: per-run ``config`` and (for the
        API key) the run context may override them in :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools`.

        Parameters
        ----------
        api_key : str or None, optional
            Google / Gemini API key used when neither the run context
            nor per-run config supplies one.
        default_generate_model : str, optional
            Default model for text-to-image generation on the
            ``"gemini"`` backend.
        default_edit_model : str, optional
            Default model used by ``edit_image``.
        default_imagen_model : str, optional
            Default model used for the ``"imagen"`` generation backend.
        timeout : float, optional
            HTTP timeout in seconds for image fetches. Must be greater
            than zero.
        max_retries : int, optional
            Maximum number of retries on transient API failures. Must be
            non-negative.

        Raises
        ------
        ValueError
            If ``max_retries`` is negative or ``timeout`` is not greater
            than zero.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self._api_key = api_key
        self._default_generate_model = default_generate_model
        self._default_edit_model = default_edit_model
        self._default_imagen_model = default_imagen_model
        self._timeout = timeout
        self._max_retries = max_retries

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Build the ``generate_image`` and ``edit_image`` tools for a run.

        Resolve the API key (preferring ``GOOGLE_API_KEY`` then
        ``GEMINI_API_KEY`` from the context, then per-run config, then
        the constructor value) and the effective model names and limits,
        then return the two structured tools bound to this context.

        Parameters
        ----------
        ctx : RunContext
            The active run context, providing forwarded API keys and any
            uploaded files referenced by ``edit_image``.
        config : dict[str, Any]
            Per-run tool-source configuration. Recognized keys:
            ``api_key``, ``default_generate_model``,
            ``default_edit_model``, ``default_imagen_model``,
            ``timeout``, ``max_retries``.

        Returns
        -------
        list of BaseTool
            The ``generate_image`` and ``edit_image`` tools.

        Raises
        ------
        RuntimeError
            If no Google / Gemini API key can be resolved.
        """
        api_key = (
            ctx.api_keys.get("GOOGLE_API_KEY")
            or ctx.api_keys.get("GEMINI_API_KEY")
            or config.get("api_key")
            or self._api_key
        )
        if not api_key:
            raise RuntimeError(
                "gemini_image: no GOOGLE_API_KEY / GEMINI_API_KEY available. "
                "Forward via QueryRequest.api_keys, set "
                "[agent.tool_source_config.gemini_image].api_key, or pass "
                "api_key= to the constructor."
            )

        generate_model = config.get(
            "default_generate_model", self._default_generate_model
        )
        edit_model = config.get("default_edit_model", self._default_edit_model)
        imagen_model = config.get("default_imagen_model", self._default_imagen_model)
        timeout = float(config.get("timeout", self._timeout))
        max_retries = int(config.get("max_retries", self._max_retries))

        async def generate_image(**kwargs: Any) -> dict[str, Any]:
            args = _GenerateArgs(**kwargs)
            return await _generate(
                ctx=ctx,
                args=args,
                api_key=api_key,
                generate_model=generate_model,
                imagen_model=imagen_model,
                timeout=timeout,
                max_retries=max_retries,
            )

        async def edit_image(**kwargs: Any) -> dict[str, Any]:
            args = _EditArgs(**kwargs)
            return await _edit(
                ctx=ctx,
                args=args,
                api_key=api_key,
                edit_model=edit_model,
                timeout=timeout,
                max_retries=max_retries,
            )

        return [
            StructuredTool.from_function(
                coroutine=generate_image,
                name="generate_image",
                description=(
                    "Generate one or more images from a text prompt using "
                    "Gemini. Returns the Workspace artifact UUIDs (the "
                    "images render automatically in the UI). Use "
                    "``backend='imagen'`` for the higher-fidelity Imagen "
                    "models when the operator's plan permits. Honour the "
                    "user's stylistic and content guidance closely."
                ),
                args_schema=_GenerateArgs,
            ),
            StructuredTool.from_function(
                coroutine=edit_image,
                name="edit_image",
                description=(
                    "Edit an existing image (uploaded or addressed by URL) "
                    "with a natural-language instruction (e.g. 'add a red "
                    "scarf', 'change background to night sky'). Returns "
                    "artifact UUIDs for the edited image."
                ),
                args_schema=_EditArgs,
            ),
        ]


def _resolve_uploaded_image(
    ctx: RunContext, name: str | None, url: str | None, timeout: float
) -> tuple[bytes, str]:
    if url:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"gemini_image: could not fetch base_image_url {url!r}: {exc}"
            ) from exc
        return resp.content, resp.headers.get("content-type") or "image/png"
    if name:
        match = _find_uploaded(ctx.uploaded_files, name, timeout)
        if match is None:
            raise RuntimeError(
                f"gemini_image: no uploaded file named {name!r}; available: "
                + ", ".join(f.name for f in ctx.uploaded_files)
            )
        return match
    raise RuntimeError(
        "gemini_image: must supply either base_image_name or base_image_url"
    )


def _find_uploaded(
    uploaded: tuple[FileRef, ...], name: str, timeout: float
) -> tuple[bytes, str] | None:
    for ref in uploaded:
        if ref.name != name:
            continue
        if ref.data_base64:
            return base64.b64decode(ref.data_base64), (ref.mime or "image/png")
        if ref.url:
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.get(ref.url)
                    resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"gemini_image: could not fetch {ref.url!r}: {exc}"
                ) from exc
            return resp.content, ref.mime or (
                resp.headers.get("content-type") or "image/png"
            )
    return None


async def _generate(
    *,
    ctx: RunContext,
    args: _GenerateArgs,
    api_key: str,
    generate_model: str,
    imagen_model: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    if args.backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {sorted(_VALID_BACKENDS)} (got {args.backend!r})"
        )
    if args.aspect_ratio is not None and args.aspect_ratio not in _VALID_ASPECT_RATIOS:
        raise ValueError(f"aspect_ratio must be one of {sorted(_VALID_ASPECT_RATIOS)}")

    backend = args.backend
    model = args.model or (imagen_model if backend == "imagen" else generate_model)

    return await asyncio.to_thread(
        _generate_sync,
        backend=backend,
        api_key=api_key,
        model=model,
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        number_of_images=args.number_of_images,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        max_retries=max_retries,
        prompt_label=args.prompt[:80],
    )


async def _edit(
    *,
    ctx: RunContext,
    args: _EditArgs,
    api_key: str,
    edit_model: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    base_bytes, base_mime = _resolve_uploaded_image(
        ctx,
        name=args.base_image_name,
        url=args.base_image_url,
        timeout=timeout,
    )

    return await asyncio.to_thread(
        _edit_sync,
        api_key=api_key,
        model=args.model or edit_model,
        prompt=args.prompt,
        base_bytes=base_bytes,
        base_mime=base_mime,
        seed=args.seed,
        max_retries=max_retries,
    )


def _generate_sync(
    *,
    backend: str,
    api_key: str,
    model: str,
    prompt: str,
    aspect_ratio: str | None,
    number_of_images: int,
    negative_prompt: str | None,
    seed: int | None,
    max_retries: int,
    prompt_label: str,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover — install-hint path
        raise RuntimeError(
            "gemini_image requires google-genai. Install the agent_server "
            "with the [google_genai] extra."
        ) from exc

    client = genai.Client(api_key=api_key)
    emit.reasoning_step(
        f"gemini_image generate ({backend})",
        model=model,
        prompt=prompt_label,
        n=number_of_images,
    )

    images = _retry(
        max_retries,
        lambda: _do_generate(
            client=client,
            types_mod=types,
            backend=backend,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            number_of_images=number_of_images,
            negative_prompt=negative_prompt,
            seed=seed,
        ),
    )

    artifact_uuids = []
    for img_bytes, mime in images:
        uuid = emit.image_artifact(
            name=f"gemini_image:{secrets.token_hex(4)}",
            description=prompt[:240],
            mime=mime,
            data_base64=base64.b64encode(img_bytes).decode("ascii"),
        )
        artifact_uuids.append(uuid)

    return {
        "model": model,
        "backend": backend,
        "prompt": prompt,
        "image_count": len(images),
        "artifact_uuids": artifact_uuids,
        "byte_sizes": [len(b) for b, _ in images],
    }


def _edit_sync(
    *,
    api_key: str,
    model: str,
    prompt: str,
    base_bytes: bytes,
    base_mime: str,
    seed: int | None,
    max_retries: int,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover — install-hint path
        raise RuntimeError(
            "gemini_image requires google-genai. Install the agent_server "
            "with the [google_genai] extra."
        ) from exc

    client = genai.Client(api_key=api_key)
    emit.reasoning_step(
        "gemini_image edit",
        model=model,
        prompt=prompt[:80],
        base_mime=base_mime,
        base_bytes=len(base_bytes),
    )

    images = _retry(
        max_retries,
        lambda: _do_edit(
            client=client,
            types_mod=types,
            model=model,
            prompt=prompt,
            base_bytes=base_bytes,
            base_mime=base_mime,
            seed=seed,
        ),
    )

    artifact_uuids = []
    for img_bytes, mime in images:
        uuid = emit.image_artifact(
            name=f"gemini_image_edit:{secrets.token_hex(4)}",
            description=prompt[:240],
            mime=mime,
            data_base64=base64.b64encode(img_bytes).decode("ascii"),
        )
        artifact_uuids.append(uuid)

    return {
        "model": model,
        "prompt": prompt,
        "image_count": len(images),
        "artifact_uuids": artifact_uuids,
        "byte_sizes": [len(b) for b, _ in images],
    }


def _do_generate(
    *,
    client: Any,
    types_mod: Any,
    backend: str,
    model: str,
    prompt: str,
    aspect_ratio: str | None,
    number_of_images: int,
    negative_prompt: str | None,
    seed: int | None,
) -> list[tuple[bytes, str]]:
    if backend == "imagen":
        cfg_kwargs: dict[str, Any] = {"number_of_images": number_of_images}
        if aspect_ratio:
            cfg_kwargs["aspect_ratio"] = aspect_ratio
        if negative_prompt:
            cfg_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            cfg_kwargs["seed"] = seed
        resp = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types_mod.GenerateImagesConfig(**cfg_kwargs),
        )
        out: list[tuple[bytes, str]] = []
        for g in resp.generated_images or []:
            data = getattr(getattr(g, "image", None), "image_bytes", None)
            if not data:
                continue
            mime = getattr(g.image, "mime_type", None) or "image/png"
            out.append((data, mime))
        if not out:
            raise RuntimeError(
                "gemini_image: imagen returned zero images (possibly blocked by safety)"
            )
        return out

    resp = client.models.generate_content(model=model, contents=prompt)
    return _extract_inline_images(resp)


def _do_edit(
    *,
    client: Any,
    types_mod: Any,
    model: str,
    prompt: str,
    base_bytes: bytes,
    base_mime: str,
    seed: int | None,
) -> list[tuple[bytes, str]]:
    inline = types_mod.Part.from_bytes(data=base_bytes, mime_type=base_mime)
    parts = [types_mod.Part.from_text(text=prompt), inline]
    contents = [types_mod.Content(role="user", parts=parts)]
    cfg = None
    if seed is not None:
        cfg = types_mod.GenerateContentConfig(seed=seed)
    resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    return _extract_inline_images(resp)


def _extract_inline_images(resp: Any) -> list[tuple[bytes, str]]:
    out: list[tuple[bytes, str]] = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if not data:
                continue
            mime = getattr(inline, "mime_type", None) or "image/png"
            out.append((data, mime))
    if not out:
        raise RuntimeError(
            "gemini_image: response carried no image parts (possibly blocked "
            "by safety or model returned text-only)"
        )
    return out


def _retry(max_retries: int, fn: Any) -> Any:
    """Run fn, retrying on 429, 5xx, and connection errors with backoff."""
    attempt = 0
    last_exc: BaseException | None = None
    while attempt <= max_retries:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _retryable(exc):
                raise
            attempt += 1
            if attempt > max_retries:
                break
            wait = _retry_delay(exc) or _backoff(attempt)
            logger.warning(
                "gemini_image: retryable error %s; sleeping %.2fs (attempt %d/%d)",
                type(exc).__name__,
                wait,
                attempt,
                max_retries,
            )
            import time as _time

            _time.sleep(wait)
    raise RuntimeError(
        f"gemini_image: exhausted {max_retries} retries — last error: {last_exc}"
    ) from last_exc


def _retryable(exc: BaseException) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code < 600):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    name = type(exc).__name__
    return name in {"ServerError", "ResourceExhausted", "DeadlineExceeded"}


def _retry_delay(exc: BaseException) -> float | None:
    """Pull a retry delay from a Gemini ClientError if present."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    err = details.get("error") or {}
    for d in err.get("details") or []:
        if not isinstance(d, dict):
            continue
        if d.get("@type", "").endswith("RetryInfo"):
            raw = d.get("retryDelay") or d.get("retry_delay")
            if isinstance(raw, str) and raw.endswith("s"):
                try:
                    return float(raw[:-1])
                except ValueError:
                    return None
    return None


def _backoff(attempt: int) -> float:
    return min(30.0, 0.5 * (2 ** (attempt - 1)))
