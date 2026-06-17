"""NVIDIA NIM reranker adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger("openbb_agent_server.memory.reranker")


class NvidiaReranker:
    """Cross-encoder reranker backed by an NVIDIA NIM model."""

    def __init__(
        self,
        *,
        model: str = "nv-rerank-qa-mistral-4b:1",
        api_key: str | None = None,
        base_url: str | None = None,
        truncate: str = "END",
        top_n: int | None = None,
    ) -> None:
        """Store reranker configuration without opening a client.

        The underlying NVIDIA NIM client is built lazily on the first
        :meth:`rerank` call, so construction never performs network or
        import work.

        Parameters
        ----------
        model : str, optional
            NVIDIA NIM rerank model identifier. Defaults to
            ``"nv-rerank-qa-mistral-4b:1"``.
        api_key : str or None, optional
            API key for the NIM endpoint. Falls back to the
            ``NVIDIA_API_KEY`` environment variable when not given.
        base_url : str or None, optional
            Override the endpoint base URL (e.g. a self-hosted NIM).
        truncate : str, optional
            Truncation strategy passed to the model for inputs that
            exceed the context window. Defaults to ``"END"``.
        top_n : int or None, optional
            Server-side cap on how many ranked results the endpoint
            returns. Independent of the per-call ``top_k`` slice.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._base_url = base_url
        self._truncate = truncate
        self._top_n = top_n
        self._client: Any | None = None

    def _build_client(self) -> Any:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIARerank
        except ImportError as exc:  # pragma: no cover — install hint
            raise RuntimeError(
                "NvidiaReranker requires langchain-nvidia-ai-endpoints. "
                "Install the agent_server with the [nvidia] extra."
            ) from exc

        if not self._api_key:
            raise RuntimeError(
                "NvidiaReranker: NVIDIA_API_KEY is not set. Provide it "
                "via the environment, user_settings.json, or the "
                "constructor."
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "api_key": self._api_key,
            "truncate": self._truncate,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._top_n is not None:
            kwargs["top_n"] = int(self._top_n)
        return NVIDIARerank(**kwargs)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Re-rank ``candidates`` against ``query`` by relevance.

        Builds the NIM client on first use, then submits each candidate
        as a document and reads the model's relevance scores back. Async
        compression is used when the client exposes it, otherwise the
        synchronous call is offloaded to a worker thread. Candidates
        whose rerank id is lost are dropped, and unparsable scores
        default to ``0.0``.

        Parameters
        ----------
        query : str
            The query text candidates are scored against. An empty or
            whitespace-only query short-circuits the model and returns
            the first ``top_k`` candidates with ``0.0`` scores.
        candidates : Sequence[tuple[str, str]]
            ``(candidate_id, text)`` pairs to score. An empty sequence
            returns an empty list without calling the model.
        top_k : int or None, optional
            Keep only the top ``top_k`` results after ranking. ``None``
            keeps all returned results.

        Returns
        -------
        list[tuple[str, float]]
            ``(candidate_id, relevance_score)`` pairs ordered by the
            model's ranking, highest relevance first.

        Raises
        ------
        RuntimeError
            If ``langchain-nvidia-ai-endpoints`` is not installed or no
            API key is available when the client is first built.
        """
        if not candidates:
            return []
        if not query or not query.strip():
            return [(cid, 0.0) for cid, _ in candidates[: top_k or len(candidates)]]

        if self._client is None:
            self._client = self._build_client()

        from langchain_core.documents import Document

        docs = [
            Document(page_content=text, metadata={"_rerank_id": cid})
            for cid, text in candidates
        ]

        acompress = getattr(self._client, "acompress_documents", None)
        if acompress is not None:
            ranked = await acompress(documents=docs, query=query)
        else:
            ranked = await asyncio.to_thread(
                self._client.compress_documents,
                documents=docs,
                query=query,
            )

        out: list[tuple[str, float]] = []
        for d in ranked:
            md = getattr(d, "metadata", {}) or {}
            cid = md.get("_rerank_id")
            if cid is None:
                continue
            score = md.get("relevance_score")
            if score is None:
                score = md.get("score", 0.0)
            try:
                out.append((str(cid), float(score)))
            except (TypeError, ValueError):
                out.append((str(cid), 0.0))

        if top_k is not None:
            out = out[: int(top_k)]
        return out
