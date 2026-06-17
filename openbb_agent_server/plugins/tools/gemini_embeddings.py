"""``gemini_embeddings`` tool source — text-vector embeddings via Gemini."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.gemini_embeddings")

_DEFAULT_MODEL = "gemini-embedding-001"
_VALID_TASK_TYPES: frozenset[str] = frozenset(
    {
        "RETRIEVAL_QUERY",
        "RETRIEVAL_DOCUMENT",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CODE_RETRIEVAL_QUERY",
    }
)


class _EmbedArgs(BaseModel):
    texts: list[str] = Field(
        description="Strings to embed. Returns one vector per element.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Override the default Gemini embedding model "
            f"(constructor default: {_DEFAULT_MODEL!r})."
        ),
    )
    task_type: str | None = Field(
        default=None,
        description=(
            "Embedding intent: one of "
            "RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT / SEMANTIC_SIMILARITY / "
            "CLASSIFICATION / CLUSTERING / QUESTION_ANSWERING / "
            "FACT_VERIFICATION / CODE_RETRIEVAL_QUERY."
        ),
    )
    output_dimensionality: int | None = Field(
        default=None,
        ge=1,
        description="Truncate vectors to this many dims. Model-dependent.",
    )


class GeminiEmbeddingsToolSource(ToolSource):
    """Expose Gemini text embeddings as an agent tool."""

    name = "gemini_embeddings"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = _DEFAULT_MODEL,
        default_task_type: str | None = None,
        default_output_dimensionality: int | None = None,
    ) -> None:
        """Configure default embedding parameters for the tool source.

        Parameters
        ----------
        api_key : str | None
            Fallback Gemini / Google API key, used only when no key is
            supplied via the run context or per-call config.
        default_model : str
            Gemini embedding model used when a call omits ``model``.
        default_task_type : str | None
            Default embedding intent applied when a call omits
            ``task_type``. Must be one of the supported task types.
        default_output_dimensionality : int | None
            Default truncation length for output vectors. Must be >= 1.

        Raises
        ------
        ValueError
            If ``default_task_type`` is not a recognized task type, or
            ``default_output_dimensionality`` is less than 1.
        """
        if default_task_type is not None and default_task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"default_task_type must be one of {sorted(_VALID_TASK_TYPES)}"
            )
        if (
            default_output_dimensionality is not None
            and default_output_dimensionality < 1
        ):
            raise ValueError("default_output_dimensionality must be >= 1")

        self._api_key = api_key
        self._default_model = default_model
        self._default_task_type = default_task_type
        self._default_output_dimensionality = default_output_dimensionality

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Build the ``embed_text`` tool bound to a resolved API key.

        Resolves the Gemini API key in priority order — the run
        context's ``GOOGLE_API_KEY`` then ``GEMINI_API_KEY``, the
        per-call ``config["api_key"]``, then the constructor fallback —
        and overlays ``config`` defaults over the constructor defaults
        for model, task type, and output dimensionality.

        Parameters
        ----------
        ctx : RunContext
            Current run context; supplies caller API keys.
        config : dict[str, Any]
            Per-invocation overrides: ``api_key``, ``default_model``,
            ``default_task_type``, ``default_output_dimensionality``.

        Returns
        -------
        list[BaseTool]
            A single ``embed_text`` structured tool.

        Raises
        ------
        RuntimeError
            If no Gemini / Google API key can be resolved.
        """
        api_key = (
            ctx.api_keys.get("GOOGLE_API_KEY")
            or ctx.api_keys.get("GEMINI_API_KEY")
            or config.get("api_key")
            or self._api_key
        )
        if not api_key:
            raise RuntimeError(
                "gemini_embeddings: no GOOGLE_API_KEY / GEMINI_API_KEY available."
            )

        default_model = config.get("default_model", self._default_model)
        default_task_type = config.get("default_task_type", self._default_task_type)
        default_dim = config.get(
            "default_output_dimensionality",
            self._default_output_dimensionality,
        )

        async def embed_text(**kwargs: Any) -> dict[str, Any]:
            args = _EmbedArgs(**kwargs)
            return await _embed(
                args=args,
                api_key=api_key,
                default_model=default_model,
                default_task_type=default_task_type,
                default_dim=default_dim,
            )

        return [
            StructuredTool.from_function(
                coroutine=embed_text,
                name="embed_text",
                description=(
                    "Compute Gemini text embeddings for one or more strings. "
                    "Returns a list of float vectors matching the input order. "
                    "Use ``task_type`` to match the use case (e.g. "
                    "RETRIEVAL_QUERY for search inputs, RETRIEVAL_DOCUMENT for "
                    "indexed documents) — the embedding geometry is tuned per "
                    "intent and mismatched task_types degrade similarity."
                ),
                args_schema=_EmbedArgs,
            )
        ]


async def _embed(
    *,
    args: _EmbedArgs,
    api_key: str,
    default_model: str,
    default_task_type: str | None,
    default_dim: int | None,
) -> dict[str, Any]:
    if not args.texts:
        return {"vectors": [], "model": default_model, "dimensions": 0}

    if args.task_type is not None and args.task_type not in _VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {sorted(_VALID_TASK_TYPES)}")
    task_type = args.task_type or default_task_type
    output_dim = args.output_dimensionality or default_dim
    model = args.model or default_model

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:  # pragma: no cover — install-hint path
        raise RuntimeError(
            "gemini_embeddings requires langchain-google-genai. Install the "
            "agent_server with the [google_genai] extra."
        ) from exc

    kwargs: dict[str, Any] = {
        "model": model if model.startswith("models/") else f"models/{model}",
        "google_api_key": api_key,
    }
    if task_type:
        kwargs["task_type"] = task_type
    if output_dim:
        kwargs["output_dimensionality"] = output_dim

    embedder = GoogleGenerativeAIEmbeddings(**kwargs)

    emit.reasoning_step(
        "gemini_embeddings",
        model=model,
        task_type=task_type,
        n=len(args.texts),
    )
    vectors = await embedder.aembed_documents(args.texts)

    return {
        "vectors": vectors,
        "model": model,
        "task_type": task_type,
        "dimensions": len(vectors[0]) if vectors else 0,
        "count": len(vectors),
    }
