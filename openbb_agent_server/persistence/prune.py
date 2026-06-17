"""Retention pruning for the history and checkpoint databases."""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore

logger = logging.getLogger("openbb_agent_server.prune")

_SQLITE_VAR_LIMIT = 500

# SQLiteVec per-row indexes that are orphan-cleaned once their parent
# history row is deleted: (data table, vec0 table, metadata parent key,
# parent table).
_VECTOR_INDEXES = (("pdf_pages_vec", "pdf_pages_vec_vec", "doc_id", "pdf_documents"),)

# Widget data is queried with SQL — these vector tables are dead weight.
_WIDGET_VECTOR_TABLES = ("widget_rows_vec", "widget_rows_vec_vec")


@dataclass
class PruneStats:
    """Row counts removed by one prune pass.

    Attributes
    ----------
    history : dict[str, int]
        Rows deleted from the history database, keyed by table/category.
    checkpoints : dict[str, int]
        Rows deleted from the checkpoint database, keyed ``"checkpoints"``
        and ``"writes"``.
    """

    history: dict[str, int] = field(default_factory=dict)
    checkpoints: dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        """Return the total rows deleted across both databases."""
        return sum(self.history.values()) + sum(self.checkpoints.values())


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _trace_id_of(thread_id: str) -> str:
    return thread_id.rsplit(":", 1)[-1]


async def prune_checkpoints(
    db_path: str,
    *,
    keep_last: int,
    recent_trace_ids: set[str] | None,
    vacuum: bool = True,
) -> dict[str, int]:
    """Prune a LangGraph SQLite checkpoint file.

    Drops whole threads whose trace is outside the retention window
    (``recent_trace_ids`` is the set still inside it; ``None`` skips the
    age pass), then keeps only the newest ``keep_last`` checkpoints per
    thread namespace. Orphaned ``writes`` rows whose checkpoint no longer
    exists are swept last, then the file is optionally vacuumed.

    Parameters
    ----------
    db_path : str
        Path to the SQLite checkpoint database. A missing file is a
        no-op and returns zero counts.
    keep_last : int
        Number of most-recent checkpoints to retain per
        ``(thread_id, checkpoint_ns)`` partition. Coerced to a minimum
        of 1.
    recent_trace_ids : set[str] | None
        Trace ids still inside the retention window. Threads whose trace
        id is absent from this set are deleted wholesale. ``None`` skips
        the age-based thread pass entirely.
    vacuum : bool, default True
        When True, run ``VACUUM`` after deletion to reclaim disk space.

    Returns
    -------
    dict[str, int]
        Rows deleted, keyed ``"checkpoints"`` and ``"writes"``.
    """
    counts = {"checkpoints": 0, "writes": 0}
    if not Path(db_path).exists():
        return counts
    keep = max(1, keep_last)
    conn = await aiosqlite.connect(db_path, timeout=30.0)
    try:
        await conn.execute("PRAGMA busy_timeout=30000")
        if recent_trace_ids is not None:
            cur = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
            threads = [str(r[0]) for r in await cur.fetchall()]
            stale = [
                t
                for t in threads
                if ":" in t and _trace_id_of(t) not in recent_trace_ids
            ]
            for i in range(0, len(stale), _SQLITE_VAR_LIMIT):
                chunk = stale[i : i + _SQLITE_VAR_LIMIT]
                marks = ",".join("?" * len(chunk))
                for table in ("writes", "checkpoints"):
                    cur = await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id IN ({marks})",  # noqa: S608
                        chunk,
                    )
                    counts[table] += cur.rowcount or 0
        cur = await conn.execute(
            "DELETE FROM checkpoints WHERE rowid IN ("
            "  SELECT rowid FROM ("
            "    SELECT rowid, ROW_NUMBER() OVER ("
            "      PARTITION BY thread_id, checkpoint_ns ORDER BY checkpoint_id DESC"
            "    ) AS rn FROM checkpoints"
            "  ) WHERE rn > ?"
            ")",
            (keep,),
        )
        counts["checkpoints"] += cur.rowcount or 0
        cur = await conn.execute(
            "DELETE FROM writes WHERE (thread_id, checkpoint_ns, checkpoint_id) "
            "NOT IN (SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints)"
        )
        counts["writes"] += cur.rowcount or 0
        await conn.commit()
        if vacuum:
            await conn.execute("VACUUM")
    finally:
        await conn.close()
    return counts


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _prune_history_vectors_sync(db_path: str, *, vacuum: bool) -> dict[str, int]:
    import sqlite_vec

    counts: dict[str, int] = {}
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA busy_timeout=30000")
        if _table_exists(conn, "widget_rows_vec"):
            removed = conn.execute("SELECT COUNT(*) FROM widget_rows_vec").fetchone()[0]
            for tbl in _WIDGET_VECTOR_TABLES:
                if _table_exists(conn, tbl):
                    conn.execute(f"DROP TABLE {tbl}")  # noqa: S608
            counts["widget_rows_vec"] = int(removed)
        for data_tbl, vec_tbl, parent_col, parent_tbl in _VECTOR_INDEXES:
            if not _table_exists(conn, data_tbl):
                continue
            orphans = [
                r[0]
                for r in conn.execute(
                    f"SELECT rowid FROM {data_tbl} WHERE CAST("  # noqa: S608
                    f"json_extract(metadata, '$.{parent_col}') AS INTEGER) "
                    f"NOT IN (SELECT id FROM {parent_tbl})"
                ).fetchall()
            ]
            removed = 0
            for i in range(0, len(orphans), _SQLITE_VAR_LIMIT):
                chunk = orphans[i : i + _SQLITE_VAR_LIMIT]
                marks = ",".join("?" * len(chunk))
                conn.execute(
                    f"DELETE FROM {vec_tbl} WHERE rowid IN ({marks})",  # noqa: S608
                    chunk,
                )
                cur = conn.execute(
                    f"DELETE FROM {data_tbl} WHERE rowid IN ({marks})",  # noqa: S608
                    chunk,
                )
                removed += cur.rowcount or 0
            counts[data_tbl] = removed
        conn.commit()
        if vacuum:
            conn.execute("VACUUM")
    finally:
        conn.close()
    return counts


async def prune_history_vectors(db_path: str, *, vacuum: bool = True) -> dict[str, int]:
    """Reclaim SQLiteVec space in ``history.db``.

    Drops the removed widget per-row index (``widget_rows_vec``)
    wholesale, and orphan-cleans the PDF page index whose rows are not
    foreign-keyed to their parent. A no-op when the file is absent.
    The blocking SQLite work runs in a worker thread.

    Parameters
    ----------
    db_path : str
        Path to the ``history.db`` SQLite file. A missing file returns
        an empty dict.
    vacuum : bool, default True
        When True, run ``VACUUM`` after pruning to reclaim disk space.

    Returns
    -------
    dict[str, int]
        Rows removed per vector table (e.g. ``"widget_rows_vec"``,
        ``"pdf_pages_vec"``). Empty when the file is absent.
    """
    if not Path(db_path).exists():
        return {}
    return await asyncio.to_thread(_prune_history_vectors_sync, db_path, vacuum=vacuum)


async def run_prune(
    *,
    history: SqliteHistoryStore,
    checkpoint_path: str | None,
    history_retention_days: int | None,
    checkpoint_retention_days: int | None,
    checkpoint_keep_last: int,
    vacuum: bool = True,
) -> PruneStats:
    """Prune history and checkpoints per the configured retention windows.

    Runs the history age pass (and its SQLiteVec cleanup) when
    ``history_retention_days`` is set, then the checkpoint pass when
    ``checkpoint_path`` is given, deriving the checkpoint retention
    window from the history store's recent trace ids.

    Parameters
    ----------
    history : SqliteHistoryStore
        History store to prune; also the source of recent trace ids used
        to scope the checkpoint age pass.
    checkpoint_path : str | None
        Path to the LangGraph checkpoint database. ``None`` skips the
        checkpoint pass.
    history_retention_days : int | None
        Delete history older than this many days. ``None`` skips the
        history pass.
    checkpoint_retention_days : int | None
        Drop checkpoint threads whose trace has had no history activity
        within this many days. ``None`` keeps all threads regardless of
        age (only ``checkpoint_keep_last`` applies).
    checkpoint_keep_last : int
        Number of most-recent checkpoints to retain per thread
        namespace.
    vacuum : bool, default True
        When True, vacuum each database after pruning. The history
        store's own vacuum is suppressed when a separate vector file is
        pruned, so the vector pass performs the single reclaim instead.

    Returns
    -------
    PruneStats
        Per-database row counts removed by this pass.
    """
    stats = PruneStats()
    now = _now()
    if history_retention_days is not None:
        cutoff = now - _dt.timedelta(days=history_retention_days)
        db_path = history.db_path
        stats.history = await history.prune_older_than(
            cutoff=cutoff, vacuum=(vacuum and db_path is None)
        )
        if db_path is not None:
            stats.history.update(await prune_history_vectors(db_path, vacuum=vacuum))
    if checkpoint_path is not None:
        recent: set[str] | None = None
        if checkpoint_retention_days is not None:
            since = now - _dt.timedelta(days=checkpoint_retention_days)
            recent = await history.recent_trace_ids(since=since)
        stats.checkpoints = await prune_checkpoints(
            checkpoint_path,
            keep_last=checkpoint_keep_last,
            recent_trace_ids=recent,
            vacuum=vacuum,
        )
    logger.info(
        "prune: removed %d history row(s), %d checkpoint row(s)",
        sum(stats.history.values()),
        sum(stats.checkpoints.values()),
    )
    return stats
