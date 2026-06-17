"""Deterministic feature-hashed embeddings + cosine helper."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Zero-config feature-hashed embeddings for tests and dev fallback.

    Produce deterministic, L2-normalised vectors without any model
    download or API call. Each text is tokenised into unigrams plus
    adjacent-word bigrams; every feature is hashed (blake2s) into a
    bucket and accumulated with a sign derived from the hash, giving a
    signed feature-hashing (a.k.a. hashing-trick) embedding. Identical
    input always yields the identical vector, which makes it suitable
    for unit tests and offline development where a real embedding model
    is unavailable.
    """

    def __init__(self, *, dim: int = 256) -> None:
        """Initialise the embedder with a fixed output dimensionality.

        Parameters
        ----------
        dim : int, optional
            Length of every produced embedding vector and the number of
            hash buckets. Must be positive. Defaults to ``256``.

        Raises
        ------
        ValueError
            If ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents into feature-hashed vectors.

        Parameters
        ----------
        texts : list of str
            The documents to embed.

        Returns
        -------
        list of list of float
            One ``dim``-length L2-normalised vector per input text, in
            the same order as ``texts``.
        """
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a feature-hashed vector.

        Parameters
        ----------
        text : str
            The query text to embed.

        Returns
        -------
        list of float
            A ``dim``-length L2-normalised embedding vector.
        """
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = text.lower().split()
        features = list(words)
        features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        for f in features:
            h = hashlib.blake2s(f.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors.

    Robust to length mismatch: only the leading ``min(len(a), len(b))``
    components are compared, so a longer vector is silently truncated to
    the length of the shorter one. A zero-norm vector is treated as
    having norm ``1.0`` to avoid division by zero.

    Parameters
    ----------
    a : Sequence of float
        The first vector.
    b : Sequence of float
        The second vector.

    Returns
    -------
    float
        The cosine similarity in the range ``[-1.0, 1.0]``. Returns
        ``0.0`` when either input is empty.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    num = sum(a[i] * b[i] for i in range(n))
    da = math.sqrt(sum(x * x for x in a[:n])) or 1.0
    db = math.sqrt(sum(x * x for x in b[:n])) or 1.0
    return num / (da * db)


__all__ = ["Embeddings", "HashEmbeddings", "cosine"]
