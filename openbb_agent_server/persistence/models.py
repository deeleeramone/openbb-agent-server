"""SQLAlchemy ORM tables."""

from __future__ import annotations

import datetime as _dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM tables.

    Provides a shared ``type_annotation_map`` so ``Mapped[dict[str, Any]]``
    and ``Mapped[list[Any]]`` attributes are stored using the SQLAlchemy
    ``JSON`` column type.
    """

    type_annotation_map = {
        dict[str, Any]: JSON,
        list[Any]: JSON,
    }


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class User(Base):
    """An end user of the agent server, keyed by ``user_id``.

    Attributes
    ----------
    user_id : str
        Primary-key identifier for the user.
    display_name : str | None
        Optional human-readable name.
    email : str | None
        Optional contact email address.
    created_at : datetime.datetime
        UTC timestamp when the user record was first created.
    last_seen_at : datetime.datetime
        UTC timestamp of the user's most recent activity.
    quota_json : dict[str, Any]
        Arbitrary per-user quota/limit configuration stored as JSON.
    memory_opt_in : bool
        Whether the user has opted in to long-term memory features.
    """

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    last_seen_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    quota_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    memory_opt_in: Mapped[bool] = mapped_column(Boolean, default=False)


class ApiKey(Base):
    """A hashed API key issued to a :class:`User` for authentication.

    Attributes
    ----------
    key_id : str
        Primary-key identifier for the key (the public, non-secret part).
    user_id : str
        Owning user; cascades on user deletion.
    hashed_secret : str
        Hash of the key's secret value; the raw secret is never stored.
    label : str | None
        Optional human-readable label describing the key's purpose.
    scopes : list[Any]
        Permission scopes granted to this key, stored as JSON.
    created_at : datetime.datetime
        UTC timestamp when the key was issued.
    revoked_at : datetime.datetime | None
        UTC timestamp when the key was revoked, or ``None`` if active.
    """

    __tablename__ = "api_keys"

    key_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE")
    )
    hashed_secret: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    scopes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    revoked_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (Index("ix_api_keys_user", "user_id"),)


class Conversation(Base):
    """A chat conversation owned by a :class:`User`.

    Acts as the parent of an ordered set of :class:`Message` rows, which are
    deleted together with the conversation.

    Attributes
    ----------
    conversation_id : str
        Primary-key identifier for the conversation.
    user_id : str
        Owning user; cascades on user deletion.
    title : str | None
        Optional display title for the conversation.
    summary_blob_ref : str | None
        Reference to an externally stored rolling summary of the
        conversation, if one has been generated.
    created_at : datetime.datetime
        UTC timestamp when the conversation was created.
    updated_at : datetime.datetime
        UTC timestamp of the last update to the conversation.
    messages : list[Message]
        Child messages, ordered relationship cascading delete-orphan.
    """

    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE")
    )
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    summary_blob_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    updated_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    __table_args__ = (Index("ix_conversations_user", "user_id", "updated_at"),)

    messages: Mapped[list[Message]] = relationship(
        cascade="all, delete-orphan",
        back_populates="conversation",
    )


class Trace(Base):
    """An execution trace grouping the events of a single agent run.

    Attributes
    ----------
    trace_id : str
        Primary-key identifier for the trace.
    user_id : str
        Owning user; cascades on user deletion.
    conversation_id : str | None
        Conversation this trace belongs to, if any.
    run_id : str | None
        Identifier of the associated :class:`Run`, if any.
    started_at : datetime.datetime
        UTC timestamp when the trace began.
    ended_at : datetime.datetime | None
        UTC timestamp when the trace finished, or ``None`` while running.
    status : str
        Lifecycle status of the trace; defaults to ``"running"``.
    """

    __tablename__ = "traces"

    trace_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE")
    )
    conversation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    ended_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String, default="running")

    __table_args__ = (Index("ix_traces_user", "user_id", "started_at"),)


class Message(Base):
    """A single message within a :class:`Conversation`.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    conversation_id : str
        Parent conversation; cascades on conversation deletion.
    user_id : str
        Owning user identifier.
    seq : int
        Monotonic sequence number ordering messages within a conversation.
    role : str
        Author role of the message, e.g. ``"user"`` or ``"assistant"``.
    content : str
        Textual content of the message.
    widget_refs : list[Any]
        References to widgets attached to the message, stored as JSON.
    file_refs : list[Any]
        References to files attached to the message, stored as JSON.
    trace_id : str | None
        Identifier of the trace that produced this message, if any.
    ts : datetime.datetime
        UTC timestamp when the message was recorded.
    conversation : Conversation
        Back-reference to the owning conversation.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE")
    )
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    widget_refs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    file_refs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    ts: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_messages_user_conv_seq", "user_id", "conversation_id", "seq"),
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Run(Base):
    """A single invocation of the agent against a model.

    Attributes
    ----------
    run_id : str
        Primary-key identifier for the run.
    user_id : str
        Owning user identifier.
    conversation_id : str | None
        Conversation this run is part of, if any.
    trace_id : str | None
        Identifier of the associated :class:`Trace`, if any.
    model : str | None
        Name of the model used for the run.
    system_prompt_hash : str | None
        Hash of the system prompt used, for reproducibility/auditing.
    status : str
        Lifecycle status of the run; defaults to ``"running"``.
    started_at : datetime.datetime
        UTC timestamp when the run began.
    ended_at : datetime.datetime | None
        UTC timestamp when the run finished, or ``None`` while running.
    """

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    system_prompt_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")
    started_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    ended_at: Mapped[_dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (Index("ix_runs_user", "user_id", "started_at"),)


class ToolCall(Base):
    """A single tool invocation recorded against a :class:`Trace`.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    trace_id : str
        Trace this tool call belongs to.
    user_id : str
        Owning user identifier.
    seq : int
        Monotonic sequence number ordering tool calls within a trace.
    tool_name : str
        Name of the tool that was invoked.
    args_json : dict[str, Any]
        Arguments passed to the tool, stored as JSON.
    result_json : dict[str, Any] | None
        Result returned by the tool, or ``None`` if it errored or is pending.
    error : str | None
        Error message if the tool call failed.
    latency_ms : int | None
        Wall-clock duration of the call in milliseconds, if measured.
    side : str
        Where the call executed; defaults to ``"server"`` (vs. client-side).
    state : str
        Execution state of the call; defaults to ``"complete"``.
    ts : datetime.datetime
        UTC timestamp when the call was recorded.
    """

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    args_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    side: Mapped[str] = mapped_column(String, default="server")
    state: Mapped[str] = mapped_column(String, default="complete")
    ts: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_tool_calls_user_trace", "user_id", "trace_id", "seq"),)


class Usage(Base):
    """Token-usage and cost accounting for one model call in a trace.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    trace_id : str
        Trace this usage record belongs to.
    user_id : str
        Owning user identifier.
    seq : int
        Monotonic sequence number ordering usage records within a trace.
    model : str
        Name of the model the usage was billed against.
    input_tokens : int
        Number of input (prompt) tokens consumed.
    output_tokens : int
        Number of output (completion) tokens generated.
    cache_read : int
        Number of tokens read from prompt cache.
    cache_creation : int
        Number of tokens written when creating prompt cache entries.
    cost_usd : float
        Computed cost of the call in US dollars.
    ts : datetime.datetime
        UTC timestamp when the usage was recorded.
    """

    __tablename__ = "usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read: Mapped[int] = mapped_column(Integer, default=0)
    cache_creation: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    ts: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_usage_user_trace", "user_id", "trace_id", "seq"),)


class Artifact(Base):
    """An output artifact produced during a :class:`Trace`.

    Stores either an inline JSON payload or a reference to externally
    stored blob data.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    trace_id : str
        Trace this artifact belongs to.
    user_id : str
        Owning user identifier.
    seq : int
        Monotonic sequence number ordering artifacts within a trace.
    kind : str
        Type discriminator describing what the artifact is.
    payload_blob_ref : str | None
        Reference to externally stored blob payload, if used.
    payload_json : dict[str, Any] | None
        Inline JSON payload, if the artifact is stored inline.
    mime : str | None
        MIME type of the artifact's payload, if known.
    ts : datetime.datetime
        UTC timestamp when the artifact was recorded.
    """

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    payload_blob_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    mime: Mapped[str | None] = mapped_column(String, nullable=True)
    ts: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_artifacts_user_trace", "user_id", "trace_id", "seq"),)


class CitationRow(Base):
    """A single source citation recorded against a :class:`Trace`.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    trace_id : str
        Trace this citation belongs to.
    user_id : str
        Owning user identifier.
    seq : int
        Monotonic sequence number ordering citations within a trace.
    source : str | None
        Name or identifier of the cited source.
    source_url : str | None
        URL of the cited source, if available.
    page : int | None
        Page number within the source document, if applicable.
    bbox_json : list[Any] | None
        Bounding-box coordinates locating the citation on the page,
        stored as JSON.
    text_snippet : str | None
        Excerpt of the cited text.
    """

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    text_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_citations_user_trace", "user_id", "trace_id", "seq"),)


class WidgetData(Base):
    """One ingested widget-data response.

    Captures the tabular payload returned by a widget so it can be reused
    as context within a conversation.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    user_id : str
        Owning user; cascades on user deletion.
    conversation_id : str
        Conversation the widget data was ingested into.
    widget_uuid : str
        Identifier of the widget that produced the data.
    widget_name : str | None
        Human-readable name of the widget, if known.
    origin : str | None
        Source/provider the widget data originated from.
    input_args : dict[str, Any]
        Input arguments used to fetch the data, stored as JSON.
    columns : list[Any] | None
        Column definitions for the tabular payload, stored as JSON.
    rows : list[Any]
        Row data for the tabular payload, stored as JSON.
    ingested_at : datetime.datetime
        UTC timestamp when the data was ingested.
    """

    __tablename__ = "widget_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id: Mapped[str] = mapped_column(String, nullable=False)
    widget_uuid: Mapped[str] = mapped_column(String, nullable=False)
    widget_name: Mapped[str | None] = mapped_column(String, nullable=True)
    origin: Mapped[str | None] = mapped_column(String, nullable=True)
    input_args: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    columns: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    rows: Mapped[list[Any]] = mapped_column(JSON, default=list)
    ingested_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=_now,
    )

    __table_args__ = (
        Index(
            "ix_widget_data_lookup",
            "user_id",
            "conversation_id",
            "widget_uuid",
            "ingested_at",
        ),
    )


class PdfDocument(Base):
    """One ingested PDF with its metadata and table of contents.

    Acts as the parent of its parsed :class:`PdfPage` rows, which are
    deleted together with the document.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    user_id : str
        Owning user identifier.
    file_key : str
        Storage key identifying the underlying file.
    name : str
        Display name of the document.
    url : str | None
        Source URL the document was fetched from, if any.
    mime : str | None
        MIME type of the document, if known.
    total_pages : int
        Total number of pages in the document.
    metadata_json : dict[str, Any]
        Extracted document metadata, stored as JSON.
    toc_json : list[Any]
        Extracted table of contents, stored as JSON.
    status : str
        Ingestion status; defaults to ``"pending"``.
    error : str | None
        Error message if ingestion failed.
    ingested_at : datetime.datetime
        UTC timestamp when the document was ingested.
    """

    __tablename__ = "pdf_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    file_key: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    mime: Mapped[str | None] = mapped_column(String, nullable=True)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    toc_json: Mapped[list[Any]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    __table_args__ = (Index("ix_pdf_documents_lookup", "user_id", "file_key"),)


class PdfPage(Base):
    """One parsed page of a :class:`PdfDocument`.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    pdf_id : int
        Parent document; cascades on document deletion.
    page : int
        One-based page number within the document.
    text : str
        Extracted plain text of the page.
    words_json : list[Any]
        Per-word data (e.g. text and positions) for the page, stored as
        JSON.
    """

    __tablename__ = "pdf_pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pdf_id: Mapped[int] = mapped_column(
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, default="")
    words_json: Mapped[list[Any]] = mapped_column(JSON, default=list)

    __table_args__ = (Index("ix_pdf_pages_doc_page", "pdf_id", "page"),)


class PendingRun(Base):
    """State blob for resuming a run that yielded on a client-side tool call.

    Persists the serialized run state so execution can continue once the
    client returns the result of the tool call.

    Attributes
    ----------
    run_id : str
        Primary-key identifier of the run to resume.
    user_id : str
        Owning user identifier.
    state_blob : dict[str, Any]
        Serialized run state needed to resume execution, stored as JSON.
    created_at : datetime.datetime
        UTC timestamp when the pending state was saved.
    """

    __tablename__ = "pending_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    state_blob: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    __table_args__ = (Index("ix_pending_runs_user", "user_id"),)
