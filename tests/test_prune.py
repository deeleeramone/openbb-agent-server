"""Retention pruning tests."""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from openbb_agent_server.persistence import models as m
from openbb_agent_server.persistence.prune import (
    PruneStats,
    prune_checkpoints,
    prune_history_vectors,
    run_prune,
)
from openbb_agent_server.persistence.sqlite_store import SqliteHistoryStore

_CKPT_SCHEMA = """
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
"""


async def _make_checkpoints_db(path: Path, threads: dict[str, list[str]]) -> None:
    conn = await aiosqlite.connect(str(path))
    await conn.executescript(_CKPT_SCHEMA)
    for thread_id, cids in threads.items():
        for cid in cids:
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) "
                "VALUES (?, ?, ?)",
                (thread_id, "", cid),
            )
            await conn.execute(
                "INSERT INTO writes "
                "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, "", cid, "task", 0, "ch"),
            )
    await conn.commit()
    await conn.close()


async def _count(path: Path, table: str) -> int:
    conn = await aiosqlite.connect(str(path))
    try:
        cur = await conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        row = await cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        await conn.close()


def test_prune_stats_total() -> None:
    stats = PruneStats(history={"traces": 2}, checkpoints={"checkpoints": 3})
    assert stats.total() == 5


@pytest.mark.asyncio
async def test_prune_checkpoints_missing_file_is_noop(tmp_path: Path) -> None:
    counts = await prune_checkpoints(
        str(tmp_path / "nope.db"), keep_last=1, recent_trace_ids=None
    )
    assert counts == {"checkpoints": 0, "writes": 0}


@pytest.mark.asyncio
async def test_prune_checkpoints_keeps_only_latest_per_thread(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.db"
    await _make_checkpoints_db(db, {"u:default:t1": ["c1", "c2", "c3"]})
    counts = await prune_checkpoints(str(db), keep_last=1, recent_trace_ids=None)
    assert counts["checkpoints"] == 2
    assert await _count(db, "checkpoints") == 1
    assert await _count(db, "writes") == 1


@pytest.mark.asyncio
async def test_prune_checkpoints_drops_stale_threads(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.db"
    await _make_checkpoints_db(
        db,
        {
            "u:default:old": ["c1", "c2"],
            "u:default:fresh": ["c3", "c4"],
            "no-colons": ["c5"],
        },
    )
    counts = await prune_checkpoints(str(db), keep_last=5, recent_trace_ids={"fresh"})
    assert counts["checkpoints"] == 2
    rows = []
    conn = await aiosqlite.connect(str(db))
    cur = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
    rows = sorted(r[0] for r in await cur.fetchall())
    await conn.close()
    assert rows == ["no-colons", "u:default:fresh"]


@pytest.mark.asyncio
async def test_prune_checkpoints_keep_last_clamped_to_one(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.db"
    await _make_checkpoints_db(db, {"u:default:t1": ["c1", "c2"]})
    await prune_checkpoints(str(db), keep_last=0, recent_trace_ids=None)
    assert await _count(db, "checkpoints") == 1


@pytest.mark.asyncio
async def test_prune_checkpoints_skips_vacuum(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.db"
    await _make_checkpoints_db(db, {"u:default:t1": ["c1", "c2"]})
    counts = await prune_checkpoints(
        str(db), keep_last=1, recent_trace_ids=None, vacuum=False
    )
    assert counts["checkpoints"] == 1


async def _seed_history(history: SqliteHistoryStore) -> tuple[_dt.datetime, ...]:
    now = _dt.datetime.now(_dt.timezone.utc)
    old = now - _dt.timedelta(days=400)
    async with history._sessionmaker() as session:
        session.add(m.User(user_id="u"))
        session.add(
            m.Trace(trace_id="old", user_id="u", started_at=old, status="completed")
        )
        session.add(
            m.Trace(trace_id="new", user_id="u", started_at=now, status="completed")
        )
        session.add(
            m.WidgetData(
                user_id="u", conversation_id="c", widget_uuid="w", ingested_at=old
            )
        )
        session.add(
            m.WidgetData(
                user_id="u", conversation_id="c", widget_uuid="w2", ingested_at=now
            )
        )
        await session.commit()
    return now, old


@pytest.mark.asyncio
async def test_prune_older_than_drops_aged_rows(
    history: SqliteHistoryStore,
) -> None:
    now, _old = await _seed_history(history)
    counts = await history.prune_older_than(cutoff=now - _dt.timedelta(days=90))
    assert counts["traces"] == 1
    assert counts["widget_data"] == 1
    async with history._sessionmaker() as session:
        from sqlalchemy import select

        traces = sorted(r[0] for r in await session.execute(select(m.Trace.trace_id)))
    assert traces == ["new"]


@pytest.mark.asyncio
async def test_prune_older_than_without_vacuum(
    history: SqliteHistoryStore,
) -> None:
    now, _old = await _seed_history(history)
    counts = await history.prune_older_than(
        cutoff=now - _dt.timedelta(days=90), vacuum=False
    )
    assert counts["traces"] == 1


@pytest.mark.asyncio
async def test_recent_trace_ids(history: SqliteHistoryStore) -> None:
    now, _old = await _seed_history(history)
    recent = await history.recent_trace_ids(since=now - _dt.timedelta(days=14))
    assert recent == {"new"}


@pytest.mark.asyncio
async def test_run_prune_history_and_checkpoints(
    history: SqliteHistoryStore, tmp_path: Path
) -> None:
    now, _old = await _seed_history(history)
    db = tmp_path / "checkpoints.db"
    await _make_checkpoints_db(
        db, {"u:default:old": ["c1", "c2"], "u:default:new": ["c3", "c4"]}
    )
    stats = await run_prune(
        history=history,
        checkpoint_path=str(db),
        history_retention_days=90,
        checkpoint_retention_days=14,
        checkpoint_keep_last=1,
    )
    assert stats.history["traces"] == 1
    assert stats.checkpoints["checkpoints"] == 3
    assert stats.total() > 0


@pytest.mark.asyncio
async def test_run_prune_skips_when_nothing_configured(
    history: SqliteHistoryStore,
) -> None:
    stats = await run_prune(
        history=history,
        checkpoint_path=None,
        history_retention_days=None,
        checkpoint_retention_days=None,
        checkpoint_keep_last=1,
    )
    assert stats.total() == 0


@pytest.mark.asyncio
async def test_run_prune_checkpoints_without_age_window(
    history: SqliteHistoryStore, tmp_path: Path
) -> None:
    db = tmp_path / "checkpoints.db"
    await _make_checkpoints_db(db, {"u:default:t1": ["c1", "c2", "c3"]})
    stats = await run_prune(
        history=history,
        checkpoint_path=str(db),
        history_retention_days=None,
        checkpoint_retention_days=None,
        checkpoint_keep_last=1,
    )
    assert stats.checkpoints["checkpoints"] == 2


def _make_vec_db(path: Path, *, widget_rows: int, pdf_doc_ids: list[int]) -> None:
    import sqlite_vec

    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    emb = sqlite_vec.serialize_float32([0.1, 0.2, 0.3, 0.4])
    for data_tbl, vec_tbl, trigger, parent_tbl, parent_key, ids in (
        (
            "widget_rows_vec",
            "widget_rows_vec_vec",
            "widget_rows_vec_embed_text",
            "widget_data",
            "parent_id",
            [1] * widget_rows,
        ),
        (
            "pdf_pages_vec",
            "pdf_pages_vec_vec",
            "pdf_pages_vec_embed_text",
            "pdf_documents",
            "doc_id",
            pdf_doc_ids,
        ),
    ):
        conn.execute(f"CREATE TABLE {parent_tbl} (id INTEGER PRIMARY KEY)")  # noqa: S608
        conn.execute(f"INSERT INTO {parent_tbl} (id) VALUES (1)")  # noqa: S608
        conn.execute(
            f"CREATE TABLE {data_tbl} (rowid INTEGER PRIMARY KEY AUTOINCREMENT, "  # noqa: S608
            "text TEXT, metadata BLOB, text_embedding BLOB)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE {vec_tbl} USING vec0("  # noqa: S608
            "rowid INTEGER PRIMARY KEY, text_embedding float[4])"
        )
        conn.execute(
            f"CREATE TRIGGER {trigger} AFTER INSERT ON {data_tbl} "  # noqa: S608
            f"BEGIN INSERT INTO {vec_tbl}(rowid, text_embedding) "
            "VALUES (new.rowid, new.text_embedding); END"
        )
        for pid in ids:
            conn.execute(
                f"INSERT INTO {data_tbl} (text, metadata, text_embedding) "  # noqa: S608
                "VALUES (?, ?, ?)",
                ("row", json.dumps({parent_key: pid}), emb),
            )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_prune_history_vectors_missing_file(tmp_path: Path) -> None:
    assert await prune_history_vectors(str(tmp_path / "nope.db")) == {}


@pytest.mark.asyncio
async def test_prune_history_vectors_drops_widget_and_cleans_pdf(
    tmp_path: Path,
) -> None:
    db = tmp_path / "history.db"
    _make_vec_db(db, widget_rows=3, pdf_doc_ids=[1, 1, 999])
    counts = await prune_history_vectors(str(db), vacuum=True)
    assert counts["widget_rows_vec"] == 3
    assert counts["pdf_pages_vec"] == 1
    conn = sqlite3.connect(str(db))
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name='widget_rows_vec'"
            ).fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM pdf_pages_vec").fetchone()[0] == 2
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_prune_history_vectors_skips_absent_tables(
    history: SqliteHistoryStore,
) -> None:
    assert history.db_path is not None
    assert await prune_history_vectors(history.db_path) == {}


@pytest.mark.asyncio
async def test_run_prune_history_db_path_none() -> None:
    class _StubHistory:
        db_path = None

        async def prune_older_than(
            self, *, cutoff: _dt.datetime, vacuum: bool
        ) -> dict[str, int]:
            return {"traces": 4}

    stats = await run_prune(
        history=_StubHistory(),  # type: ignore[arg-type]
        checkpoint_path=None,
        history_retention_days=30,
        checkpoint_retention_days=None,
        checkpoint_keep_last=1,
    )
    assert stats.history == {"traces": 4}


def _build_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    from openbb_agent_server.app.app import create_app
    from openbb_agent_server.app.settings import AgentServerSettings

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv(
        "OPENBB_AGENT_DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_CHECKPOINTER_PROVIDER", "inmemory")
    return create_app(AgentServerSettings())


@pytest.mark.asyncio
async def test_lifespan_runs_prune_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    import openbb_agent_server.persistence.prune as prune_mod

    monkeypatch.setenv("OPENBB_AGENT_PRUNE_INTERVAL_HOURS", "24")
    called = asyncio.Event()

    async def _fake(**_kwargs: Any) -> PruneStats:
        called.set()
        return PruneStats()

    monkeypatch.setattr(prune_mod, "run_prune", _fake)
    app = _build_app(monkeypatch, tmp_path)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(called.wait(), timeout=5)
    assert called.is_set()


@pytest.mark.asyncio
async def test_lifespan_skips_sweep_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    import openbb_agent_server.persistence.prune as prune_mod

    monkeypatch.setenv("OPENBB_AGENT_PRUNE_INTERVAL_HOURS", "0")
    called = False

    async def _fake(**_kwargs: Any) -> PruneStats:
        nonlocal called
        called = True
        return PruneStats()

    monkeypatch.setattr(prune_mod, "run_prune", _fake)
    app = _build_app(monkeypatch, tmp_path)
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0)
    assert called is False


@pytest.mark.asyncio
async def test_lifespan_sweep_survives_prune_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    import openbb_agent_server.persistence.prune as prune_mod

    monkeypatch.setenv("OPENBB_AGENT_PRUNE_INTERVAL_HOURS", "24")
    raised = asyncio.Event()

    async def _boom(**_kwargs: Any) -> PruneStats:
        raised.set()
        raise RuntimeError("prune boom")

    monkeypatch.setattr(prune_mod, "run_prune", _boom)
    app = _build_app(monkeypatch, tmp_path)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(raised.wait(), timeout=5)
    assert raised.is_set()
