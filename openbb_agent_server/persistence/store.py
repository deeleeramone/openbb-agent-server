"""HistoryStore — the multi-tenant persistence ABC."""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from openbb_agent_server.runtime.principal import UserPrincipal


class TraceRecord(BaseModel):
    """Immutable record describing one agent execution trace.

    A trace bounds a single run of the agent for a user, optionally tied
    to a conversation and a run identifier, with start/end timestamps and
    a terminal status.

    Attributes
    ----------
    trace_id : str
        Unique identifier for the trace.
    user_id : str
        Owner of the trace; all access is scoped to this user.
    conversation_id : str or None
        Conversation the trace belongs to, or None if standalone.
    run_id : str or None
        Identifier of the run that produced the trace, if any.
    started_at : datetime.datetime
        Timestamp at which the trace began.
    ended_at : datetime.datetime or None
        Timestamp at which the trace finished, or None while in flight.
    status : str
        Terminal status of the trace (for example ``"ok"`` or ``"error"``).
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    user_id: str
    conversation_id: str | None
    run_id: str | None
    started_at: _dt.datetime
    ended_at: _dt.datetime | None
    status: str


class MessageRecord(BaseModel):
    """Immutable record for one message in a conversation.

    Attributes
    ----------
    conversation_id : str
        Conversation the message belongs to.
    seq : int
        Monotonic sequence number of the message within the conversation.
    role : str
        Author role of the message (for example ``"user"`` or
        ``"assistant"``).
    content : str
        Text content of the message.
    user_id : str
        Owner of the message; all access is scoped to this user.
    trace_id : str or None
        Trace that produced the message, or None if not associated with
        a trace.
    ts : datetime.datetime
        Timestamp at which the message was recorded.
    """

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    seq: int
    role: str
    content: str
    user_id: str
    trace_id: str | None
    ts: _dt.datetime


class ToolCallRecord(BaseModel):
    """Immutable record for one tool invocation within a trace.

    Attributes
    ----------
    trace_id : str
        Trace the tool call belongs to.
    seq : int
        Monotonic sequence number of the call within the trace.
    user_id : str
        Owner of the call; all access is scoped to this user.
    tool_name : str
        Name of the invoked tool.
    args : dict[str, Any]
        Arguments passed to the tool.
    result : dict[str, Any] or None
        Structured result returned by the tool, or None on failure.
    error : str or None
        Error message if the call failed, otherwise None.
    latency_ms : int or None
        Wall-clock duration of the call in milliseconds, if measured.
    side : str
        Origin of the call (for example client- versus server-side).
    state : str
        Lifecycle state of the call (for example pending or completed).
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    seq: int
    user_id: str
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    latency_ms: int | None
    side: str
    state: str


class UsageRecord(BaseModel):
    """Immutable record of token usage and cost for a trace.

    Attributes
    ----------
    trace_id : str
        Trace the usage applies to.
    user_id : str
        Owner of the usage; all access is scoped to this user.
    model : str
        Identifier of the model that produced the usage.
    input_tokens : int
        Number of prompt (input) tokens consumed.
    output_tokens : int
        Number of completion (output) tokens produced.
    cache_read : int
        Number of tokens served from the prompt cache.
    cache_creation : int
        Number of tokens written to the prompt cache.
    cost_usd : float
        Total cost of the usage in US dollars.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    user_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_creation: int
    cost_usd: float


class HistoryStore(ABC):
    """Persistence ABC. All queries are principal-scoped."""

    @abstractmethod
    async def upsert_user(self, principal: UserPrincipal) -> None:
        """Insert or update the user row for ``principal``.

        Implementers must create the user if absent and refresh stored
        identity attributes otherwise, keyed on ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Authenticated identity to persist.
        """

    @abstractmethod
    async def begin_trace(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
        conversation_id: str | None,
        run_id: str | None,
    ) -> None:
        """Open a new trace owned by the principal.

        Implementers must record the trace with its start time and mark
        it as in-flight, scoped to ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner of the trace.
        trace_id : str
            Unique identifier to assign to the trace.
        conversation_id : str or None
            Conversation the trace belongs to, or None if standalone.
        run_id : str or None
            Identifier of the run producing the trace, if any.
        """

    @abstractmethod
    async def end_trace(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
        status: str,
    ) -> None:
        """Close an open trace with a terminal status.

        Implementers must set the trace end time and status, and apply
        the update only to a trace owned by ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner of the trace.
        trace_id : str
            Identifier of the trace to close.
        status : str
            Terminal status to record (for example ``"ok"`` or
            ``"error"``).
        """

    @abstractmethod
    async def append_message(
        self,
        *,
        principal: UserPrincipal,
        conversation_id: str,
        role: str,
        content: str,
        trace_id: str | None,
    ) -> int:
        """Append one message to a conversation and return its sequence.

        Implementers must allocate the next monotonic ``seq`` within the
        conversation and persist the message scoped to
        ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner of the message.
        conversation_id : str
            Conversation to append to.
        role : str
            Author role of the message (for example ``"user"`` or
            ``"assistant"``).
        content : str
            Text content of the message.
        trace_id : str or None
            Trace that produced the message, or None.

        Returns
        -------
        int
            The newly assigned sequence number of the appended message.
        """

    @abstractmethod
    async def list_conversations(
        self,
        *,
        principal: UserPrincipal,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the principal's conversations, most recent first.

        Parameters
        ----------
        principal : UserPrincipal
            Owner whose conversations are listed.
        limit : int, default 50
            Maximum number of conversations to return.

        Returns
        -------
        list[dict[str, Any]]
            One summary mapping per conversation owned by the principal.
        """

    @abstractmethod
    async def get_messages(
        self,
        *,
        principal: UserPrincipal,
        conversation_id: str,
        limit: int = 200,
    ) -> list[MessageRecord]:
        """Return messages from a conversation in sequence order.

        Implementers must restrict results to a conversation owned by
        ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner of the conversation.
        conversation_id : str
            Conversation to read.
        limit : int, default 200
            Maximum number of messages to return.

        Returns
        -------
        list[MessageRecord]
            Messages ordered by ascending sequence number.
        """

    @abstractmethod
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
        """Record one tool invocation against a trace.

        Implementers must allocate the next sequence number within the
        trace and persist the call scoped to ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner of the trace.
        trace_id : str
            Trace the tool call belongs to.
        tool_name : str
            Name of the invoked tool.
        args : dict[str, Any]
            Arguments passed to the tool.
        result : dict[str, Any] or None
            Structured result returned by the tool, or None on failure.
        error : str or None
            Error message if the call failed, otherwise None.
        latency_ms : int or None
            Wall-clock duration of the call in milliseconds, if measured.
        side : str
            Origin of the call (for example client- versus server-side).
        state : str
            Lifecycle state of the call (for example pending or
            completed).
        """

    @abstractmethod
    async def record_usage(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
        usage: UsageRecord,
    ) -> None:
        """Persist token usage and cost for a trace.

        Implementers must store the usage scoped to ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner of the trace.
        trace_id : str
            Trace the usage applies to.
        usage : UsageRecord
            Token counts and cost to record.
        """

    @abstractmethod
    async def delete_user(self, principal: UserPrincipal) -> None:
        """Cascade-delete every row owned by the user.

        Implementers must remove the user and all dependent records
        (traces, messages, tool calls, and usage) for
        ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner whose data is deleted.
        """

    @abstractmethod
    async def get_trace_bundle(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str,
    ) -> dict[str, Any] | None:
        """Return the full audit join for one trace.

        Implementers must assemble the trace together with its related
        messages, tool calls, and usage, restricted to a trace owned by
        ``principal.user_id``.

        Parameters
        ----------
        principal : UserPrincipal
            Owner of the trace.
        trace_id : str
            Trace to bundle.

        Returns
        -------
        dict[str, Any] or None
            The joined audit bundle, or None if no such trace exists for
            the principal.
        """

    @abstractmethod
    async def usage_summary(
        self,
        *,
        principal: UserPrincipal,
        trace_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Return aggregated usage scoped to the principal.

        Implementers must aggregate usage for ``principal.user_id``,
        optionally narrowing to a single trace or conversation when the
        corresponding filter is supplied.

        Parameters
        ----------
        principal : UserPrincipal
            Owner whose usage is aggregated.
        trace_id : str or None, optional
            Restrict aggregation to this trace when provided.
        conversation_id : str or None, optional
            Restrict aggregation to this conversation when provided.

        Returns
        -------
        dict[str, Any]
            Aggregated usage totals for the selected scope.
        """
