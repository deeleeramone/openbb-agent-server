"""``vision_qa`` tool source."""

from __future__ import annotations

import logging
import mimetypes
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

logger = logging.getLogger("openbb_agent_server.tools.vision_qa")

_DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
_DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".svg"}
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


class _UnderstandArgs(BaseModel):
    instruction: str = Field(
        description=(
            "What to do with the image. Examples: 'Read this chart and "
            "list the YoY growth rates.', 'Extract the table as rows.', "
            "'Run OCR and return all visible text.'"
        )
    )
    name: str | None = Field(
        default=None,
        description=(
            "Name of an uploaded image (matches the output of "
            "``list_images``). Either ``name`` or ``url`` must be set."
        ),
    )
    url: str | None = Field(
        default=None,
        description=(
            "Direct image URL (https). Either ``name`` or ``url`` must be set."
        ),
    )
    max_output_tokens: int = Field(
        default=2048,
        ge=64,
        le=8192,
        description="Token cap on the model's reply.",
    )


class VisionQaToolSource(ToolSource):
    """Expose image understanding tools over a vision-capable NIM model."""

    name = "vision_qa"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_fetch_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
        fetch_timeout_s: float = 60.0,
    ) -> None:
        """Configure default settings for the vision-QA tools.

        These values act as fallbacks; per-run ``config`` keys passed to
        :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools` override them.

        Parameters
        ----------
        model : str, optional
            NIM vision model id used for image understanding. Defaults to
            ``nvidia/llama-3.1-nemotron-nano-vl-8b-v1``.
        api_key : str or None, optional
            NVIDIA API key fallback used when none is found on the run context,
            in ``config``, or in the ``NVIDIA_API_KEY`` environment variable.
        base_url : str or None, optional
            Override base URL for the NIM endpoint; passed through to the
            client only when set.
        temperature : float, optional
            Sampling temperature for the chat model. Defaults to 0.2.
        max_fetch_bytes : int, optional
            Maximum number of bytes to download when resolving an image URL.
            Defaults to 32 MiB.
        fetch_timeout_s : float, optional
            Timeout in seconds for fetching a remote image. Defaults to 60.
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_fetch_bytes = int(max_fetch_bytes)
        self._fetch_timeout_s = float(fetch_timeout_s)

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Build the vision-QA tools for a run, if an API key is available.

        Resolves the NVIDIA API key from the run context, ``config``, the
        configured fallback, then the ``NVIDIA_API_KEY`` environment variable.
        When no key is found, logs a warning and returns no tools. Otherwise
        returns ``list_images``, ``understand_image`` and
        ``submit_understand_image`` bound to a per-run NIM client.

        Parameters
        ----------
        ctx : RunContext
            Active run context; its ``api_keys`` are checked for
            ``NVIDIA_API_KEY``.
        config : dict
            Per-run overrides. Recognized keys: ``api_key``, ``model``,
            ``base_url``, ``temperature``, ``max_fetch_bytes`` and
            ``fetch_timeout_s``.

        Returns
        -------
        list of BaseTool
            The registered vision tools, or an empty list when no API key is
            available.

        Raises
        ------
        RuntimeError
            If ``langchain-nvidia-ai-endpoints`` is not installed.
        """
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "vision_qa requires langchain-nvidia-ai-endpoints. "
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
                "vision_qa: NVIDIA_API_KEY is not set; skipping "
                "tool registration. Set it to enable understand_image / "
                "submit_understand_image."
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

        def list_images() -> list[dict[str, Any]]:
            current = run_context.current()
            imgs = [f for f in current.uploaded_files if _is_image(f)]
            emit.reasoning_step("list_images", count=len(imgs))
            return [{"name": f.name, "mime": f.mime, "url": f.url} for f in imgs]

        async def understand_image(
            instruction: str,
            name: str | None = None,
            url: str | None = None,
            max_output_tokens: int = 2048,
        ) -> str:
            if not name and not url:
                raise ValueError(
                    "understand_image: provide either ``name`` (uploaded "
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
                    raise RuntimeError(f"vision_qa: {exc}") from exc
            else:
                if url is None:  # pragma: no cover - first guard rejects both-None
                    raise ValueError(
                        "vision_qa: image source must carry a 'url' or 'name'"
                    )
                try:
                    fetched = await fetch_url(
                        url,
                        max_bytes=max_fetch_bytes,
                        timeout_s=fetch_timeout_s,
                        fallback_mime="image/png",
                    )
                    data_url = await to_data_url(fetched.data, mime=fetched.mime)
                except MediaError as exc:
                    raise RuntimeError(f"vision_qa: {exc}") from exc

            emit.reasoning_step(
                "understand_image",
                source=name or url,
                model=model_name,
            )

            from langchain_core.messages import HumanMessage, SystemMessage

            system = (
                "You are a multimodal vision assistant. Given one image "
                "and a user instruction, respond with the most useful "
                "answer. For charts, give exact numeric values when "
                "they are legible. For tables, return rows as Markdown "
                "or JSON. For documents / receipts, transcribe text "
                "verbatim (OCR). Never invent values you can't see."
            )
            content: list[str | dict[Any, Any]] = [
                {"type": "text", "text": instruction},
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

            return "".join(chunks).strip()

        async def submit_understand_image(
            instruction: str,
            name: str | None = None,
            url: str | None = None,
            max_output_tokens: int = 2048,
        ) -> dict[str, Any]:
            from openbb_agent_server.runtime.jobs import get_registry

            label = f"understand_image({name or url or '<unspecified>'})"
            job_id = get_registry().submit(
                lambda: understand_image(
                    instruction=instruction,
                    name=name,
                    url=url,
                    max_output_tokens=max_output_tokens,
                ),
                label=label,
                metadata={"tool": "understand_image", "source": name or url},
            )
            emit.reasoning_step(
                "submit_understand_image",
                job_id=job_id,
                source=name or url,
            )
            return {"job_id": job_id, "label": label}

        return [
            StructuredTool.from_function(
                list_images,
                name="list_images",
                description=(
                    "List image files (PNG/JPG/WEBP/SVG/…) the user "
                    "uploaded for this run. Returns ``[{name, mime, url}]``."
                ),
                args_schema=_ListArgs,
            ),
            StructuredTool.from_function(
                coroutine=understand_image,
                name="understand_image",
                description=(
                    "Answer a natural-language instruction about one "
                    "image using a vision-capable NIM chat model "
                    "(default ``nvidia/llama-3.1-nemotron-nano-vl-8b-v1``). "
                    "Strong at: chart reading (returning "
                    "exact numeric values), table extraction (rows as "
                    "markdown or JSON), OCR of receipts / scans, and "
                    "general image Q&A. Provide EITHER ``name`` (an "
                    "uploaded file from ``list_images``) OR ``url`` "
                    "(direct https). ``instruction`` is the question / "
                    "task you want answered about the image. For batches "
                    "of images where you want to interleave other tool "
                    "calls, prefer ``submit_understand_image`` and "
                    "collect each result via ``wait_for_job``."
                ),
                args_schema=_UnderstandArgs,
            ),
            StructuredTool.from_function(
                coroutine=submit_understand_image,
                name="submit_understand_image",
                description=(
                    "Background variant of ``understand_image``. Returns "
                    "``{job_id, label}`` immediately and runs the model "
                    "call off the agent's main loop. Use the "
                    "``background_jobs`` tools (``check_job``, "
                    "``wait_for_job``, ``cancel_job``) to track progress."
                ),
                args_schema=_UnderstandArgs,
            ),
        ]
