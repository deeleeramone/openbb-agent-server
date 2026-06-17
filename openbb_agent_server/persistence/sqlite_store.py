"""Async SQLAlchemy ``HistoryStore`` for SQLite/Postgres."""

from __future__ import annotations

import datetime as _dt
from typing import Any

from sqlalchemy import delete, event, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from openbb_agent_server.persistence import models as m
from openbb_agent_server.persistence.store import (
    HistoryStore,
    MessageRecord,
    UsageRecord,
)
from openbb_agent_server.runtime.principal import UserPrincipal


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _apply_sqlite_pragmas(engine: AsyncEngine, url: str) -> None:
    """Enable WAL and a busy_timeout so writers don't deadlock."""
    if "sqlite" not in url:
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_conn: Any, _: Any) -> None:
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.execute("PRAGMA synchronous=NORMAL")
        finally:
            cur.close()


class SqliteHistoryStore(HistoryStore):
    """Single ``HistoryStore`` implementation for SQLite + Postgres.

    Backs the agent server's persisted history (users, conversations,
    messages, traces, tool calls, usage, artifacts, and citations) with an
    async SQLAlchemy engine. The same class serves both SQLite and Postgres;
    SQLite connections additionally get WAL mode and a busy timeout applied.
    """

    def __init__(self, url: str) -> None:
        """Create the async engine and session factory for ``url``.

        Parameters
        ----------
        url : str
            SQLAlchemy async database URL (e.g.
            ``sqlite+aiosqlite:///history.db`` or an async Postgres DSN).
            When the URL targets SQLite, WAL/busy-timeout pragmas are
            registered on connect.
        """
        self._engine: AsyncEngine = create_async_engine(url, future=True)
        _apply_sqlite_pragmas(self._engine, url)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_schema(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(m.Base.metadata.create_all)

    async def aclose(self) -> None:
        """Dispose of the engine and release its connection pool."""
        await self._engine.dispose()

    async def upsert_user(self, principal: UserPrincipal) -> None:
        """Insert the principal's user row, or refresh it if it exists.

        When no row exists for ``principal.user_id`` a new user is created
        from the principal's display name and email. Otherwise the existing
        row's display name and email are overwritten and ``last_seen_at`` is
        bumped to the current UTC time.

        Parameters
        ----------
        principal : UserPrincipal
            The authenticated user whose record is created or updated.
        """
        async with self._sessionmaker() as session:
            existing = await session.get(m.User, principal.user_id)
            if existing is None:
                session.add(
                    m.User(
                        user_id=principal.user_id,
                        display_name=principal.display_name,
                        email=principal.email,
                    )
                )
            else:
                existing.display_name = principal.display_name
                existing.email = principal.email
                existing.last_seen_at = _now()
            await session.commit()

    async def delete_user(self, principal: UserPrincipal) -> None:
        """Delete the user and cascade-delete all rows they own.

        Removes every history row scoped to ``principal.user_id`` across
        messages, tool calls, usage, artifacts, citations, pending runs,
        runs, traces, conversations, and API keys before deleting the user
        row itself, then commits.

        Parameters
        ----------
        principal : UserPrincipal
            The user whose data is purged.
        """
        async with self._sessionmaker() as session:
            await self._cascade_delete(session, principal.user_id)
            await session.commit()

    async def begin_trace(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
        conversation_id: str | None,
        run_id: str | None,
    ) -> None:
        """Open or reopen a trace and bind it to the principal.

        Creates a new ``Trace`` row for ``trace_id`` when none exists. If the
        trace already exists and belongs to ``principal``, it is reopened:
        the run id is updated, ``ended_at`` is cleared, and status is reset to
        ``"running"``.

        Parameters
        ----------
        principal : UserPrincipal
            The user that owns the trace.
        trace_id : str
            Stable identifier for the trace (primary key).
        conversation_id : str or None
            Conversation the trace belongs to, if any. Only used when the
            trace row is first created.
        run_id : str or None
            Identifier of the run that produced this trace.

        Raises
        ------
        PermissionError
            If ``trace_id`` already exists but is owned by a different user.
        """
        async with self._sessionmaker() as session:
            row = await session.get(m.Trace, trace_id)
            if row is None:
                session.add(
                    m.Trace(
                        trace_id=trace_id,
                        user_id=principal.user_id,
                        conversation_id=conversation_id,
                        run_id=run_id,
                    )
                )
            else:
                if row.user_id != principal.user_id:
                    raise PermissionError(f"trace {trace_id} belongs to another user")
                row.run_id = run_id
                row.ended_at = None
                row.status = "running"
            await session.commit()

    async def end_trace(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
        status: str,
    ) -> None:
        """Mark a trace finished with the given terminal status.

        Sets ``ended_at`` to the current UTC time and records ``status`` on
        the principal's trace. Silently does nothing when the trace does not
        exist or is owned by another user.

        Parameters
        ----------
        principal : UserPrincipal
            The user that owns the trace.
        trace_id : str
            Identifier of the trace to close.
        status : str
            Terminal status to store (e.g. ``"completed"`` or ``"error"``).
        """
        async with self._sessionmaker() as session:
            row = await self._scoped_get_trace(session, principal, trace_id)
            if row is None:
                return
            row.ended_at = _now()
            row.status = status
            await session.commit()

    async def append_message(
        self,
        *,
        principal: UserPrincipal,
        conversation_id: str,
        role: str,
        content: str,
        trace_id: str | None,
    ) -> int:
        """Append a message to a conversation and return its sequence number.

        Creates the conversation on first use (owned by ``principal``). For an
        existing conversation the next ``seq`` is computed as one past the
        current maximum. The conversation's ``updated_at`` is refreshed.

        Parameters
        ----------
        principal : UserPrincipal
            The user that owns the conversation and message.
        conversation_id : str
            Identifier of the conversation to append to.
        role : str
            Author role of the message (e.g. ``"user"`` or ``"assistant"``).
        content : str
            Message body text.
        trace_id : str or None
            Trace that produced the message, if any.

        Returns
        -------
        int
            Zero-based sequence number assigned to the new message within the
            conversation.

        Raises
        ------
        PermissionError
            If the conversation exists but is owned by a different user.
        """
        async with self._sessionmaker() as session:
            conv = await session.get(m.Conversation, conversation_id)
            if conv is None:
                conv = m.Conversation(
                    conversation_id=conversation_id,
                    user_id=principal.user_id,
                )
                session.add(conv)
                seq = 0
            elif conv.user_id != principal.user_id:
                raise PermissionError("conversation not found")
            else:
                last = await session.scalar(
                    select(m.Message.seq)
                    .where(m.Message.conversation_id == conversation_id)
                    .order_by(m.Message.seq.desc())
                    .limit(1)
                )
                seq = 0 if last is None else last + 1
            session.add(
                m.Message(
                    conversation_id=conversation_id,
                    user_id=principal.user_id,
                    seq=seq,
                    role=role,
                    content=content,
                    trace_id=trace_id,
                )
            )
            conv.updated_at = _now()
            await session.commit()
            return seq

    async def list_conversations(
        self,
        *,
        principal: UserPrincipal,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the principal's conversations, most recently updated first.

        Parameters
        ----------
        principal : UserPrincipal
            The user whose conversations are listed.
        limit : int, default 50
            Maximum number of conversations to return.

        Returns
        -------
        list of dict
            One dict per conversation with keys ``conversation_id``,
            ``title``, and ``updated_at`` (ISO-8601 string), ordered by
            descending ``updated_at``.
        """
        async with self._sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        select(m.Conversation)
                        .where(m.Conversation.user_id == principal.user_id)
                        .order_by(m.Conversation.updated_at.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return [
                {
                    "conversation_id": r.conversation_id,
                    "title": r.title,
                    "updated_at": r.updated_at.isoformat(),
                }
                for r in rows
            ]

    async def get_messages(
        self,
        *,
        principal: UserPrincipal,
        conversation_id: str,
        limit: int = 200,
    ) -> list[MessageRecord]:
        """Return a conversation's messages in ascending sequence order.

        Only messages owned by ``principal`` in the given conversation are
        returned, so the user scoping doubles as an access check.

        Parameters
        ----------
        principal : UserPrincipal
            The user that owns the conversation.
        conversation_id : str
            Identifier of the conversation to read.
        limit : int, default 200
            Maximum number of messages to return.

        Returns
        -------
        list of MessageRecord
            Messages ordered by ascending ``seq``.
        """
        async with self._sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        select(m.Message)
                        .where(
                            m.Message.user_id == principal.user_id,
                            m.Message.conversation_id == conversation_id,
                        )
                        .order_by(m.Message.seq.asc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return [
                MessageRecord(
                    conversation_id=r.conversation_id,
                    seq=r.seq,
                    role=r.role,
                    content=r.content,
                    user_id=r.user_id,
                    trace_id=r.trace_id,
                    ts=r.ts,
                )
                for r in rows
            ]

    async def record_tool_call(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any] | None,
        error: str | None,
        latency_ms: int | None,
        side: str,
        state: str,
    ) -> None:
        """Persist a single tool invocation against a trace.

        Assigns the next per-trace sequence number and stores the call's
        arguments, result, error, latency, side, and state.

        Parameters
        ----------
        principal : UserPrincipal
            The user that owns the trace.
        trace_id : str
            Trace the tool call belongs to.
        tool_name : str
            Name of the invoked tool.
        args : dict
            JSON-serializable arguments passed to the tool.
        result : dict or None
            JSON-serializable result, or ``None`` if the call produced none.
        error : str or None
            Error message when the call failed, otherwise ``None``.
        latency_ms : int or None
            Wall-clock duration of the call in milliseconds, if measured.
        side : str
            Which side issued the call (e.g. client vs. server).
        state : str
            Lifecycle/outcome state of the call.
        """
        async with self._sessionmaker() as session:
            seq = await self._next_seq(session, m.ToolCall, trace_id, principal)
            session.add(
                m.ToolCall(
                    trace_id=trace_id,
                    user_id=principal.user_id,
                    seq=seq,
                    tool_name=tool_name,
                    args_json=args,
                    result_json=result,
                    error=error,
                    latency_ms=latency_ms,
                    side=side,
                    state=state,
                )
            )
            await session.commit()

    async def record_usage(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
        usage: UsageRecord,
    ) -> None:
        """Persist a token-usage / cost record against a trace.

        Assigns the next per-trace sequence number and stores model, token
        counts, cache hits, and cost.

        Parameters
        ----------
        principal : UserPrincipal
            The user that owns the trace.
        trace_id : str
            Trace the usage record belongs to.
        usage : UsageRecord
            Usage details to persist. Its ``user_id`` must match
            ``principal.user_id``.

        Raises
        ------
        PermissionError
            If ``usage.user_id`` differs from ``principal.user_id``.
        """
        if usage.user_id != principal.user_id:
            raise PermissionError("usage record user mismatch")
        async with self._sessionmaker() as session:
            seq = await self._next_seq(session, m.Usage, trace_id, principal)
            session.add(
                m.Usage(
                    trace_id=trace_id,
                    user_id=principal.user_id,
                    seq=seq,
                    model=usage.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read=usage.cache_read,
                    cache_creation=usage.cache_creation,
                    cost_usd=usage.cost_usd,
                )
            )
            await session.commit()

    async def get_trace_bundle(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
    ) -> dict[str, Any] | None:
        """Return a full snapshot of a trace and its related rows.

        Gathers the trace plus its messages, tool calls, usage, artifacts,
        and citations (all scoped to ``principal``) into a single nested,
        JSON-serializable dict. Messages are included only when the trace is
        bound to a conversation.

        Parameters
        ----------
        principal : UserPrincipal
            The user that owns the trace.
        trace_id : str
            Identifier of the trace to bundle.

        Returns
        -------
        dict or None
            A dict with ``trace``, ``messages``, ``tool_calls``, ``usage``,
            ``artifacts``, and ``citations`` keys, or ``None`` if the trace
            does not exist or belongs to another user.
        """
        from sqlalchemy import select

        async with self._sessionmaker() as session:
            trace = await session.get(m.Trace, trace_id)
            if trace is None or trace.user_id != principal.user_id:
                return None

            tool_calls = (
                (
                    await session.execute(
                        select(m.ToolCall)
                        .where(
                            m.ToolCall.trace_id == trace_id,
                            m.ToolCall.user_id == principal.user_id,
                        )
                        .order_by(m.ToolCall.seq.asc())
                    )
                )
                .scalars()
                .all()
            )

            usage = (
                (
                    await session.execute(
                        select(m.Usage)
                        .where(
                            m.Usage.trace_id == trace_id,
                            m.Usage.user_id == principal.user_id,
                        )
                        .order_by(m.Usage.seq.asc())
                    )
                )
                .scalars()
                .all()
            )

            artifacts = (
                (
                    await session.execute(
                        select(m.Artifact)
                        .where(
                            m.Artifact.trace_id == trace_id,
                            m.Artifact.user_id == principal.user_id,
                        )
                        .order_by(m.Artifact.seq.asc())
                    )
                )
                .scalars()
                .all()
            )

            citations = (
                (
                    await session.execute(
                        select(m.CitationRow)
                        .where(
                            m.CitationRow.trace_id == trace_id,
                            m.CitationRow.user_id == principal.user_id,
                        )
                        .order_by(m.CitationRow.seq.asc())
                    )
                )
                .scalars()
                .all()
            )

            messages: list[m.Message] = []
            if trace.conversation_id:
                messages = list(
                    (
                        await session.execute(
                            select(m.Message)
                            .where(
                                m.Message.user_id == principal.user_id,
                                m.Message.conversation_id == trace.conversation_id,
                                m.Message.trace_id == trace_id,
                            )
                            .order_by(m.Message.seq.asc())
                        )
                    )
                    .scalars()
                    .all()
                )

            return {
                "trace": {
                    "trace_id": trace.trace_id,
                    "run_id": trace.run_id,
                    "conversation_id": trace.conversation_id,
                    "started_at": trace.started_at.isoformat()
                    if trace.started_at
                    else None,
                    "ended_at": trace.ended_at.isoformat() if trace.ended_at else None,
                    "status": trace.status,
                },
                "messages": [
                    {
                        "seq": r.seq,
                        "role": r.role,
                        "content": r.content,
                        "ts": r.ts.isoformat() if r.ts else None,
                    }
                    for r in messages
                ],
                "tool_calls": [
                    {
                        "seq": r.seq,
                        "tool_name": r.tool_name,
                        "args": r.args_json,
                        "result": r.result_json,
                        "error": r.error,
                        "latency_ms": r.latency_ms,
                        "side": r.side,
                        "state": r.state,
                    }
                    for r in tool_calls
                ],
                "usage": [
                    {
                        "model": r.model,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "cache_read": r.cache_read,
                        "cache_creation": r.cache_creation,
                        "cost_usd": r.cost_usd,
                    }
                    for r in usage
                ],
                "artifacts": [
                    {
                        "seq": r.seq,
                        "kind": r.kind,
                        "payload": r.payload_json,
                        "mime": r.mime,
                    }
                    for r in artifacts
                ],
                "citations": [
                    {
                        "seq": r.seq,
                        "source": r.source,
                        "source_url": r.source_url,
                        "page": r.page,
                        "text_snippet": r.text_snippet,
                    }
                    for r in citations
                ],
            }

    async def usage_summary(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate the principal's token usage and cost, grouped by model.

        Sums input/output tokens, cache reads, cache creation, and cost over
        the user's usage rows, optionally narrowed to a single trace and/or
        conversation, and counts the calls per model.

        Parameters
        ----------
        principal : UserPrincipal
            The user whose usage is summarized.
        trace_id : str or None, optional
            When set, restrict the summary to this trace.
        conversation_id : str or None, optional
            When set, restrict the summary to traces in this conversation
            (joined through the trace table).

        Returns
        -------
        dict
            A dict with a single ``by_model`` key mapping to a list of
            per-model dicts (``model``, ``input_tokens``, ``output_tokens``,
            ``cache_read``, ``cache_creation``, ``cost_usd``, ``calls``).
        """
        from sqlalchemy import func, select

        async with self._sessionmaker() as session:
            stmt = select(
                m.Usage.model,
                func.sum(m.Usage.input_tokens),
                func.sum(m.Usage.output_tokens),
                func.sum(m.Usage.cache_read),
                func.sum(m.Usage.cache_creation),
                func.sum(m.Usage.cost_usd),
                func.count(m.Usage.id),
            ).where(m.Usage.user_id == principal.user_id)
            if trace_id:
                stmt = stmt.where(m.Usage.trace_id == trace_id)
            if conversation_id:
                stmt = stmt.join(m.Trace, m.Trace.trace_id == m.Usage.trace_id).where(
                    m.Trace.conversation_id == conversation_id
                )
            stmt = stmt.group_by(m.Usage.model)
            rows = (await session.execute(stmt)).all()
            return {
                "by_model": [
                    {
                        "model": r[0],
                        "input_tokens": int(r[1] or 0),
                        "output_tokens": int(r[2] or 0),
                        "cache_read": int(r[3] or 0),
                        "cache_creation": int(r[4] or 0),
                        "cost_usd": float(r[5] or 0.0),
                        "calls": int(r[6] or 0),
                    }
                    for r in rows
                ]
            }

    async def _scoped_get_trace(
        self,
        session: AsyncSession,
        principal: UserPrincipal,
        trace_id: str,
    ) -> m.Trace | None:
        row = await session.get(m.Trace, trace_id)
        if row is None or row.user_id != principal.user_id:
            return None
        return row

    async def _next_seq(
        self,
        session: AsyncSession,
        table: type[Any],
        trace_id: str,
        principal: UserPrincipal,
    ) -> int:
        last = await session.scalar(
            select(table.seq)
            .where(
                table.trace_id == trace_id,
                table.user_id == principal.user_id,
            )
            .order_by(table.seq.desc())
            .limit(1)
        )
        return 0 if last is None else last + 1

    async def _cascade_delete(self, session: AsyncSession, user_id: str) -> None:
        for table in (
            m.Message,
            m.ToolCall,
            m.Usage,
            m.Artifact,
            m.CitationRow,
            m.PendingRun,
            m.Run,
            m.Trace,
            m.Conversation,
            m.ApiKey,
        ):
            await session.execute(delete(table).where(table.user_id == user_id))
        await session.execute(delete(m.User).where(m.User.user_id == user_id))

    @property
    def db_path(self) -> str | None:
        """Filesystem path of the SQLite file, or ``None`` for non-SQLite."""
        url = self._engine.url
        if not url.drivername.startswith("sqlite"):
            return None
        return url.database

    async def recent_trace_ids(self, *, since: _dt.datetime) -> set[str]:
        """Return trace ids whose run started on or after ``since``.

        Parameters
        ----------
        since : datetime.datetime
            Inclusive lower bound compared against each trace's
            ``started_at``.

        Returns
        -------
        set of str
            Identifiers of traces started at or after ``since``.
        """
        async with self._sessionmaker() as session:
            rows = await session.execute(
                select(m.Trace.trace_id).where(m.Trace.started_at >= since)
            )
            return {r[0] for r in rows}

    async def prune_older_than(
        self, *, cutoff: _dt.datetime, vacuum: bool = True
    ) -> dict[str, int]:
        """Delete history rows older than ``cutoff`` and report counts.

        Removes time-stamped rows across traces, runs, messages,
        conversations, tool calls, usage, artifacts, widget data, and pending
        runs whose timestamp is strictly before ``cutoff``. Citations orphaned
        by trace deletion are also removed, as are PDF pages and documents for
        PDFs ingested before ``cutoff``. Optionally runs ``VACUUM`` afterward
        to reclaim space.

        Parameters
        ----------
        cutoff : datetime.datetime
            Exclusive upper bound; rows with a timestamp earlier than this are
            deleted.
        vacuum : bool, default True
            When ``True``, run ``VACUUM`` after committing the deletions.

        Returns
        -------
        dict of str to int
            Number of rows deleted, keyed by table name (plus ``citations``,
            ``pdf_pages``, and ``pdf_documents``).
        """
        counts: dict[str, int] = {}
        async with self._sessionmaker() as session:
            for table, ts_col in (
                (m.Trace, m.Trace.started_at),
                (m.Run, m.Run.started_at),
                (m.Message, m.Message.ts),
                (m.Conversation, m.Conversation.updated_at),
                (m.ToolCall, m.ToolCall.ts),
                (m.Usage, m.Usage.ts),
                (m.Artifact, m.Artifact.ts),
                (m.WidgetData, m.WidgetData.ingested_at),
                (m.PendingRun, m.PendingRun.created_at),
            ):
                result = await session.execute(delete(table).where(ts_col < cutoff))
                counts[table.__tablename__] = int(result.rowcount or 0)  # ty: ignore[unresolved-attribute]
            orphan = await session.execute(
                delete(m.CitationRow).where(
                    m.CitationRow.trace_id.not_in(select(m.Trace.trace_id))
                )
            )
            counts["citations"] = int(orphan.rowcount or 0)  # ty: ignore[unresolved-attribute]
            old_pdf_ids = select(m.PdfDocument.id).where(
                m.PdfDocument.ingested_at < cutoff
            )
            pages = await session.execute(
                delete(m.PdfPage).where(m.PdfPage.pdf_id.in_(old_pdf_ids))
            )
            counts["pdf_pages"] = int(pages.rowcount or 0)  # ty: ignore[unresolved-attribute]
            docs = await session.execute(
                delete(m.PdfDocument).where(m.PdfDocument.ingested_at < cutoff)
            )
            counts["pdf_documents"] = int(docs.rowcount or 0)  # ty: ignore[unresolved-attribute]
            await session.commit()
        if vacuum:
            autocommit = self._engine.execution_options(isolation_level="AUTOCOMMIT")
            async with autocommit.connect() as conn:
                await conn.execute(text("VACUUM"))
        return counts
