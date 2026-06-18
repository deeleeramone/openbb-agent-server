"""Vendored SQLiteVec vector store.

Adapted from ``langchain_community.vectorstores.sqlitevec`` (MIT licence)
to remove the ``langchain-community`` dependency, which is now archived.

Only the surface used by this project is kept:

* ``create_table_if_not_exists``
* ``add_texts``
* ``similarity_search_with_score``

The class intentionally does **not** manage connection lifecycle — the
caller is responsible for opening and closing the ``sqlite3.Connection``.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack a float vector into the raw-bytes format sqlite-vec expects."""
    return struct.pack(f"{len(vector)}f", *vector)


class SQLiteVec:
    """Thin wrapper around a ``sqlite-vec`` virtual table.

    Requires a pre-configured ``sqlite3.Connection`` with the ``sqlite-vec``
    extension already loaded.
    """

    def __init__(
        self,
        table: str,
        connection: sqlite3.Connection,
        embedding: Embeddings,
        db_file: str = "vec.db",
    ) -> None:
        self._connection = connection
        self._table = table
        self._embedding = embedding

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def create_table_if_not_exists(self) -> None:
        """Create the data table, vec0 virtual table, and insert trigger."""
        dim = len(self._embedding.embed_query("dim probe"))
        self._connection.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table}"
            " (rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  text TEXT, metadata BLOB, text_embedding BLOB)"
        )
        self._connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_vec"
            f" USING vec0(rowid INTEGER PRIMARY KEY,"
            f" text_embedding float[{dim}])"
        )
        self._connection.execute(
            f"CREATE TRIGGER IF NOT EXISTS {self._table}_embed_text "
            f"AFTER INSERT ON {self._table} BEGIN"
            f"  INSERT INTO {self._table}_vec(rowid, text_embedding)"
            f"  VALUES (new.rowid, new.text_embedding); END;"
        )
        self._connection.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[int]:
        """Embed *texts* and insert them into the store.

        Returns the ``rowid`` values of the newly inserted rows.
        """
        row = self._connection.execute(
            f"SELECT max(rowid) AS rowid FROM {self._table}"
        ).fetchone()
        max_id: int = row["rowid"] if row["rowid"] is not None else 0

        embeds = self._embedding.embed_documents(list(texts))

        self._connection.executemany(
            f"INSERT INTO {self._table}(text, metadata, text_embedding)"
            " VALUES (?, ?, ?)",
            [
                (text, json.dumps(meta), _serialize_f32(emb))
                for text, meta, emb in zip(texts, metadatas, embeds)
            ],
        )
        self._connection.commit()

        rows = self._connection.execute(
            f"SELECT rowid FROM {self._table} WHERE rowid > {max_id}"
        )
        return [r["rowid"] for r in rows]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return the *k* closest documents with their distance scores."""
        embedding = self._embedding.embed_query(query)
        cursor = self._connection.cursor()
        cursor.execute(
            f"SELECT text, metadata, distance"
            f" FROM {self._table} AS e"
            f" INNER JOIN {self._table}_vec AS v ON v.rowid = e.rowid"
            f" WHERE v.text_embedding MATCH ? AND k = ?"
            f" ORDER BY distance",
            [_serialize_f32(embedding), k],
        )
        results: list[tuple[Document, float]] = []
        for row in cursor.fetchall():
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            results.append(
                (Document(page_content=row["text"], metadata=meta), row["distance"])
            )
        return results
