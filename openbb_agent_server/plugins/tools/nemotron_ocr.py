"""``nemotron_ocr`` tool source — document OCR via Nemotron OCR."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.plugins.tools._media import (
    MediaError,
    fetch_url,
    flatten_message_content,
    to_data_url,
)
from openbb_agent_server.runtime import (
    context as run_context,
    emit,
)
from openbb_agent_server.runtime.context import FileRef, RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.nemotron_ocr")

_DEFAULT_MODEL = "nvidia/nemotron-ocr-v2"
_DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
)


def _is_image(f: FileRef) -> bool:
    if f.mime and f.mime.lower().startswith("image/"):
        return True
    name = (f.name or "").lower()
    return any(name.endswith(ext) for ext in _IMAGE_EXTENSIONS)


async def _resolve_data_url(
    fileref: FileRef, *, max_bytes: int, timeout_s: float
) -> str:
    """Return a data URL for ``fileref``."""
    import mimetypes

    mime = fileref.mime
    if not mime:
        guessed, _ = mimetypes.guess_type(fileref.name or "")
        mime = guessed or "image/png"

    if fileref.data_base64:
        b64 = fileref.data_base64
        if b64.startswith("data:"):
            _, _, b64 = b64.partition(",")
        return f"data:{mime};base64,{b64}"

    if fileref.url:
        fetched = await fetch_url(
            fileref.url,
            max_bytes=max_bytes,
            timeout_s=timeout_s,
            fallback_mime=mime,
        )
        return await to_data_url(fetched.data, mime=fetched.mime)

    raise RuntimeError(f"image {fileref.name!r} has no data_base64 or url to resolve")


class _ListArgs(BaseModel):
    pass


class _OcrArgs(BaseModel):
    name: str | None = Field(
        default=None,
        description=(
            "Name of an uploaded image (matches the output of "
            "``list_ocr_images``). Either ``name`` or ``url`` must be set."
        ),
    )
    url: str | None = Field(
        default=None,
        description=(
            "Direct image URL (https). Either ``name`` or ``url`` must be set."
        ),
    )
    instruction: str = Field(
        default=(
            "Extract all text from this image. Preserve layout, paragraphs, "
            "and tables as accurately as possible."
        ),
        description=(
            "What to extract from the image. Examples: 'Transcribe all text.', "
            "'Extract the balance sheet table as rows.', "
            "'Return the chart data as JSON.'"
        ),
    )
    return_table: bool = Field(
        default=False,
        description=(
            "If True and the image contains a table, return the full table "
            "as an artifact object with {columns, rows, name, description}."
        ),
    )
    max_output_tokens: int = Field(
        default=4096,
        ge=64,
        le=16384,
        description="Token cap on the model's reply.",
    )


class NemotronOcrToolSource(ToolSource):
    """Expose document/image OCR tools backed by a Nemotron OCR model.

    Registers ``list_ocr_images``, ``extract_text_from_image``, and
    ``submit_extract_text_from_image``. The default target is
    ``nvidia/nemotron-ocr-v2`` on NVIDIA NIM; any compatible
    image-to-text endpoint can be configured via ``model``.
    """

    name = "nemotron_ocr"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_fetch_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
        fetch_timeout_s: float = 60.0,
    ) -> None:
        """Store default configuration for the OCR tools.

        Parameters
        ----------
        model : str, optional
            Nemotron OCR model identifier. Defaults to
            ``nvidia/nemotron-ocr-v2``.
        api_key : str or None, optional
            Fallback NVIDIA API key used when none is found in the run
            context, config, or environment.
        base_url : str or None, optional
            Override base URL for the NIM endpoint.
        temperature : float, optional
            Sampling temperature for the chat model. Defaults to 0.0.
        max_fetch_bytes : int, optional
            Maximum size, in bytes, of any fetched image.
        fetch_timeout_s : float, optional
            Timeout in seconds for fetching a remote image.
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_fetch_bytes = int(max_fetch_bytes)
        self._fetch_timeout_s = float(fetch_timeout_s)

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Build the OCR tools for one run, or none if no API key is set.

        Parameters
        ----------
        ctx : RunContext
            Active run context supplying per-run API keys.
        config : dict[str, Any]
            Per-run overrides for the instance defaults.

        Returns
        -------
        list of BaseTool
            ``[list_ocr_images, extract_text_from_image,
            submit_extract_text_from_image]``, or an empty list when no
            NVIDIA API key is available.
        """
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "nemotron_ocr requires langchain-nvidia-ai-endpoints. "
                "Install the agent_server with the [nvidia] extra."
            ) from exc

        api_key = (
            ctx.api_keys.get("NVIDIA_API_KEY")
            or config.get("api_key")
            or self._api_key
            or os.environ.get("NVIDIA_API_KEY")
        )
        if not api_key:
            logger.warning(
                "nemotron_ocr: NVIDIA_API_KEY is not set; skipping tool "
                "registration. Set it to enable extract_text_from_image."
            )
            return []

        model_name = config.get("model", self._model)
        base_url = config.get("base_url", self._base_url)
        temperature = float(config.get("temperature", self._temperature))
        max_fetch_bytes = int(config.get("max_fetch_bytes", self._max_fetch_bytes))
        fetch_timeout_s = float(config.get("fetch_timeout_s", self._fetch_timeout_s))

        def _build_client(max_tokens: int) -> Any:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": api_key,
                "temperature": temperature,
                "max_tokens": int(max_tokens),
            }
            if base_url:
                kwargs["base_url"] = base_url
            return ChatNVIDIA(**kwargs)

        def list_ocr_images() -> list[dict[str, Any]]:
            current = run_context.current()
            imgs = [f for f in current.uploaded_files if _is_image(f)]
            emit.reasoning_step("list_ocr_images", count=len(imgs))
            return [{"name": f.name, "mime": f.mime, "url": f.url} for f in imgs]

        async def extract_text_from_image(
            name: str | None = None,
            url: str | None = None,
            instruction: str = (
                "Extract all text from this image. Preserve layout, paragraphs, "
                "and tables as accurately as possible."
            ),
            return_table: bool = False,
            max_output_tokens: int = 4096,
        ) -> dict[str, Any] | str:
            if not name and not url:
                raise ValueError(
                    "extract_text_from_image: provide either ``name`` (uploaded "
                    "file) or ``url`` (https image URL)."
                )

            current = run_context.current()
            if name:
                target = next(
                    (
                        f
                        for f in current.uploaded_files
                        if _is_image(f) and f.name == name
                    ),
                    None,
                )
                if target is None:
                    raise ValueError(f"image {name!r} is not among this run's uploads")
                try:
                    data_url = await _resolve_data_url(
                        target,
                        max_bytes=max_fetch_bytes,
                        timeout_s=fetch_timeout_s,
                    )
                except MediaError as exc:
                    raise RuntimeError(f"nemotron_ocr: {exc}") from exc
                source = name
            else:
                if url is None:  # pragma: no cover
                    raise ValueError("nemotron_ocr: image source must carry a 'url' or 'name'")
                try:
                    fetched = await fetch_url(
                        url,
                        max_bytes=max_fetch_bytes,
                        timeout_s=fetch_timeout_s,
                        fallback_mime="image/png",
                    )
                    data_url = await to_data_url(fetched.data, mime=fetched.mime)
                except MediaError as exc:
                    raise RuntimeError(f"nemotron_ocr: {exc}") from exc
                source = url

            emit.reasoning_step(
                "extract_text_from_image",
                source=source,
                model=model_name,
            )

            from langchain_core.messages import HumanMessage, SystemMessage

            system = (
                "You are a document OCR assistant. Extract text from the "
                "provided image exactly as it appears. Preserve headings, "
                "lists, paragraphs, and table structure. When a table is "
                "present and the user asks for table output, return a JSON "
                "object with 'columns' and 'rows'. Otherwise return plain "
                "text or Markdown. Do not invent content."
            )
            prompt = instruction
            if return_table:
                prompt = (
                    f"{instruction}\n\n"
                    "If the image contains a table, return ONLY a JSON object "
                    "with keys 'columns' (list of strings) and 'rows' (list of "
                    "lists). Otherwise return the extracted text."
                )
            content: list[str | dict[Any, Any]] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            client = _build_client(max_output_tokens)
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=content),
            ]

            chunks: list[str] = []
            astream = getattr(client, "astream", None)
            if astream is not None:
                async for chunk in astream(messages):
                    chunks.append(
                        flatten_message_content(getattr(chunk, "content", ""))
                    )
            else:  # pragma: no cover
                import asyncio

                response = await asyncio.to_thread(client.invoke, messages)
                chunks.append(flatten_message_content(getattr(response, "content", "")))

            text = "".join(chunks).strip()
            if return_table:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and "columns" in parsed and "rows" in parsed:
                        return {
                            "model": model_name,
                            "source": source,
                            "table": parsed,
                        }
                except json.JSONDecodeError:
                    pass
            return {
                "model": model_name,
                "source": source,
                "text": text,
            }

        async def submit_extract_text_from_image(
            name: str | None = None,
            url: str | None = None,
            instruction: str = (
                "Extract all text from this image. Preserve layout, paragraphs, "
                "and tables as accurately as possible."
            ),
            return_table: bool = False,
            max_output_tokens: int = 4096,
        ) -> dict[str, Any]:
            from openbb_agent_server.runtime.jobs import get_registry

            label = f"extract_text_from_image({name or url or '<unspecified>'})"
            job_id = get_registry().submit(
                lambda: extract_text_from_image(
                    name=name,
                    url=url,
                    instruction=instruction,
                    return_table=return_table,
                    max_output_tokens=max_output_tokens,
                ),
                label=label,
                metadata={"tool": "extract_text_from_image", "source": name or url},
            )
            emit.reasoning_step(
                "submit_extract_text_from_image",
                job_id=job_id,
                source=name or url,
            )
            return {"job_id": job_id, "label": label}

        return [
            StructuredTool.from_function(
                list_ocr_images,
                name="list_ocr_images",
                description=(
                    "List image files (PNG/JPG/WEBP/…) available for OCR. "
                    "Returns ``[{name, mime, url}]``. Call this first to "
                    "discover images before extracting text."
                ),
                args_schema=_ListArgs,
            ),
            StructuredTool.from_function(
                coroutine=extract_text_from_image,
                name="extract_text_from_image",
                description=(
                    "Run OCR on one image using a Nemotron OCR model "
                    "(default ``nvidia/nemotron-ocr-v2``). Provide EITHER "
                    "``name`` (an uploaded file from ``list_ocr_images``) OR "
                    "``url`` (direct https). Pass ``return_table=True`` when "
                    "the goal is to extract a table so the result can be "
                    "rendered as an ``emit_table_artifact``. For batches of "
                    "images, prefer ``submit_extract_text_from_image``."
                ),
                args_schema=_OcrArgs,
            ),
            StructuredTool.from_function(
                coroutine=submit_extract_text_from_image,
                name="submit_extract_text_from_image",
                description=(
                    "Background variant of ``extract_text_from_image``. "
                    "Returns ``{job_id, label}`` immediately; collect with "
                    "``background_jobs`` tools."
                ),
                args_schema=_OcrArgs,
            ),
        ]
