"""ACP-conformant PyWry chat provider over the embedded agent runtime.

``OpenBBAgentProvider`` implements PyWry's ``ChatProvider`` session
lifecycle (``initialize`` / ``new_session`` / ``prompt`` / ``cancel``)
on top of :mod:`openbb_agent_server.runtime.embedded.EmbeddedRuntime`,
so the agent loop, tool sources, middleware, and the layered
``openbb.toml`` config all transfer unchanged into a PyWry chat
component attached to any PyWry widget instance::

    from pywry import PyWry
    from openbb_agent_server.acp import create_chat_manager

    app = PyWry(title="My App")
    chat = create_chat_manager(profile="default")
    widget = app.show(
        content,
        toolbars=[chat.toolbar()],
        callbacks=chat.callbacks(),
    )
    chat.bind(widget)
    app.block()

Wire-format translation (OpenBB SSE → ACP ``SessionUpdate``):

============================  =======================================
``copilotMessageChunk``       ``AgentMessageUpdate`` (text delta)
``copilotStatusUpdate``       ``ThinkingUpdate`` (INFO reasoning) or
                              ``StatusUpdate`` (WARNING / ERROR);
                              ``hidden`` rows are dropped; attached
                              artifacts re-emit as ``ArtifactUpdate``
``copilotMessageArtifact``    ``ArtifactUpdate`` with the matching
                              PyWry artifact class (table → AG Grid,
                              text → markdown, html, code)
``copilotCitationCollection`` one ``CitationUpdate`` per citation
``copilotFunctionCall``       ``StatusUpdate`` notice — client-side
                              Workspace functions cannot run outside
                              OpenBB Workspace, and the stream closes
                              (same contract as the HTTP endpoint)
============================  =======================================

Agent profiles from the TOML surface as ACP session *modes*: when more
than one profile is configured the first prompt of a session announces
them via ``ModeUpdate`` and ``set_mode`` switches the session's profile.

Requires the ``pywry`` extra: ``pip install 'openbb-agent-server[pywry]'``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

try:
    from pywry.chat.artifacts import (
        CodeArtifact,
        HtmlArtifact,
        JsonArtifact,
        MarkdownArtifact,
        TableArtifact,
    )
    from pywry.chat.models import ContentBlock, ImagePart, TextPart
    from pywry.chat.providers import ChatProvider
    from pywry.chat.session import (
        AgentCapabilities,
        ClientCapabilities,
        PromptCapabilities,
        SessionMode,
    )
    from pywry.chat.updates import (
        AgentMessageUpdate,
        ArtifactUpdate,
        CitationUpdate,
        ModeUpdate,
        SessionUpdate,
        StatusUpdate,
        ThinkingUpdate,
    )
except ImportError as exc:  # pragma: no cover — exercised only without pywry
    raise ImportError(
        "The ACP shim requires pywry>=2.0. "
        "Install it with: pip install 'openbb-agent-server[pywry]'"
    ) from exc

from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.protocol.schemas import (
    ChatMessage,
    CitationCollectionSSE,
    ClientArtifact,
    FunctionCallSSE,
    MessageArtifactSSE,
    MessageChunkSSE,
    SSEEvent,
    StatusUpdateSSE,
    UploadedFile,
)
from openbb_agent_server.runtime.embedded import (
    DEFAULT_EMBEDDED_SCOPES,
    EmbeddedRuntime,
)

logger = logging.getLogger("openbb_agent_server.acp")

# INFO statuses come from the reasoning channel (the adapter folds
# ``<thinking>`` segments and tool announcements into them) — they land
# in PyWry's persistent thinking trail. WARNING / ERROR are transient
# alerts and stay as status rows.
_ALERT_EVENT_TYPES = frozenset({"WARNING", "ERROR"})


def _artifact_to_pywry(artifact: ClientArtifact) -> Any:
    """Convert one wire ``ClientArtifact`` into a PyWry artifact instance.

    Charts carry Workspace chart params (not a Plotly figure), so their
    row payload renders as a table — the data survives even though the
    Workspace-specific chart styling does not.
    """
    content = artifact.content
    if artifact.type == "table":
        rows = content if isinstance(content, list) else []
        return TableArtifact(data=rows)
    if artifact.type == "chart":
        if isinstance(content, list):
            return TableArtifact(data=content)
        return JsonArtifact(data={"name": artifact.name, "content": content})
    if artifact.type == "html":
        return HtmlArtifact(content=str(content))
    if artifact.type == "code":
        return CodeArtifact(content=str(content))
    # "text" and anything future-unknown render as markdown.
    return MarkdownArtifact(content=str(content))


def translate_sse(ev: SSEEvent) -> Iterator[SessionUpdate]:
    """Translate one OpenBB wire SSE event into ACP session updates.

    Dispatches on the concrete SSE subtype and yields zero or more
    ``SessionUpdate`` objects per the wire-format mapping documented in
    the module docstring. Empty message deltas and ``hidden`` status
    rows are dropped; INFO statuses become ``ThinkingUpdate`` rows while
    WARNING / ERROR statuses become transient ``StatusUpdate`` rows;
    statuses may also carry attached artifacts. Unknown event types
    yield nothing.

    Parameters
    ----------
    ev : SSEEvent
        One decoded OpenBB SSE event from the agent turn stream.

    Yields
    ------
    SessionUpdate
        ACP updates equivalent to the input event, in emission order.
    """
    if isinstance(ev, MessageChunkSSE):
        if ev.data.delta:
            yield AgentMessageUpdate(text=ev.data.delta)
        return
    if isinstance(ev, StatusUpdateSSE):
        data = ev.data
        if not data.hidden and data.message:
            if data.eventType in _ALERT_EVENT_TYPES:
                yield StatusUpdate(text=f"{data.eventType}: {data.message}")
            else:
                yield ThinkingUpdate(text=f"{data.message}\n\n")
        for attached in data.artifacts or []:
            yield ArtifactUpdate(artifact=_artifact_to_pywry(attached))
        return
    if isinstance(ev, MessageArtifactSSE):
        yield ArtifactUpdate(artifact=_artifact_to_pywry(ev.data))
        return
    if isinstance(ev, CitationCollectionSSE):
        for citation in ev.data.citations:
            info = citation.source_info
            metadata = info.metadata or {}
            yield CitationUpdate(
                url=str(metadata.get("url") or ""),
                title=info.name or info.widget_id or citation.id,
                snippet=info.description or "",
            )
        return
    if isinstance(ev, FunctionCallSSE):
        yield StatusUpdate(
            text=(
                f"The agent requested the Workspace UI function "
                f"'{ev.data.function}', which is unavailable outside "
                f"OpenBB Workspace."
            )
        )
        return


def _content_blocks_to_turn(
    content: list[ContentBlock],
) -> tuple[str, list[UploadedFile]]:
    """Split ACP content blocks into prompt text + uploaded files."""
    texts: list[str] = []
    files: list[UploadedFile] = []
    for block in content:
        if isinstance(block, TextPart):
            if block.text:
                texts.append(block.text)
        elif isinstance(block, ImagePart):
            ext = (block.mime_type or "image/png").rsplit("/", 1)[-1]
            files.append(
                UploadedFile(
                    name=f"pasted-image-{len(files) + 1}.{ext}",
                    mime=block.mime_type,
                    data_base64=block.data,
                )
            )
        else:
            logger.debug(
                "acp: ignoring unsupported content block type %r",
                getattr(block, "type", type(block).__name__),
            )
    return "\n\n".join(texts), files


@dataclass
class _AcpSession:
    """Per-session conversation state owned by the provider."""

    conversation_id: str
    profile: str
    messages: list[ChatMessage] = field(default_factory=list)
    cancel_event: asyncio.Event | None = None
    modes_announced: bool = False


class OpenBBAgentProvider(ChatProvider):
    """PyWry ``ChatProvider`` backed by the embedded agent runtime.

    Parameters
    ----------
    settings : AgentServerSettings | None
        Pre-resolved settings. When omitted, ``runtime`` must be given
        or use :meth:`from_toml` to run the layered TOML cascade.
    profile : str | None
        Profile new sessions start on. Defaults to the settings'
        ``default_profile``.
    user_id : str
        Identity for history / memory scoping — embedded chats are
        single-user, so one stable id is the expected shape.
    runtime : EmbeddedRuntime | None
        Bring-your-own runtime (e.g. shared across providers). Built
        from ``settings`` when omitted.
    """

    def __init__(
        self,
        settings: AgentServerSettings | None = None,
        *,
        profile: str | None = None,
        user_id: str = "pywry-local",
        runtime: EmbeddedRuntime | None = None,
    ) -> None:
        """Build the provider over an embedded runtime.

        Parameters
        ----------
        settings : AgentServerSettings or None, optional
            Settings to construct a runtime from when ``runtime`` is not
            given. Required unless ``runtime`` is supplied.
        profile : str or None, optional
            Agent profile to activate for this provider.
        user_id : str, default "pywry-local"
            Identity used for history and memory scoping.
        runtime : EmbeddedRuntime or None, optional
            A pre-built runtime to reuse; when given, ``settings`` is
            ignored.

        Raises
        ------
        ValueError
            If neither ``settings`` nor ``runtime`` is provided.
        """
        if runtime is None:
            if settings is None:
                raise ValueError(
                    "OpenBBAgentProvider needs settings or a runtime; "
                    "use OpenBBAgentProvider.from_toml() to load openbb.toml"
                )
            runtime = EmbeddedRuntime(settings)
        self._runtime = runtime
        self._default_profile = profile or self._runtime.settings.default_profile
        # Fail fast on unknown profile names instead of at first prompt.
        self._runtime.settings.resolve_profile(self._default_profile)
        self._principal = self._runtime.principal(
            user_id,
            scopes=DEFAULT_EMBEDDED_SCOPES,
        )
        self._sessions: dict[str, _AcpSession] = {}

    @classmethod
    def from_toml(
        cls,
        explicit_path: str | None = None,
        *,
        profile: str | None = None,
        user_id: str = "pywry-local",
    ) -> OpenBBAgentProvider:
        """Build a provider from the layered ``openbb.toml`` cascade.

        The same discovery the server CLI runs — ``.env`` files,
        ``~/.openbb_platform/openbb.toml``, the project's
        ``openbb.toml``, then ``explicit_path`` / ``$OPENBB_CONFIG`` —
        so one TOML drives the HTTP server and the embedded chat alike.

        Parameters
        ----------
        explicit_path : str or None, optional
            Path to a specific ``openbb.toml`` layered on top of the
            cascade. Falls back to ``$OPENBB_CONFIG`` and the standard
            discovery locations when not given.
        profile : str or None, optional
            Profile new sessions start on. Defaults to the settings'
            ``default_profile``.
        user_id : str, default "pywry-local"
            Identity used to scope history and memory.

        Returns
        -------
        OpenBBAgentProvider
            A provider over a runtime built from the resolved settings.
        """
        return cls(
            runtime=EmbeddedRuntime.from_toml(explicit_path),
            profile=profile,
            user_id=user_id,
        )

    @property
    def runtime(self) -> EmbeddedRuntime:
        """The embedded runtime this provider drives."""
        return self._runtime

    async def initialize(
        self,
        capabilities: ClientCapabilities,
    ) -> AgentCapabilities:
        """Start the runtime and advertise agent capabilities.

        Parameters
        ----------
        capabilities : ClientCapabilities
            Capabilities reported by the connecting ACP client.
            Accepted for protocol conformance; not currently inspected.

        Returns
        -------
        AgentCapabilities
            Agent capabilities advertising image prompt support, no
            session loading, and mode support when more than one agent
            profile is configured.
        """
        await self._runtime.start()
        has_modes = len(self._runtime.settings.all_profile_names()) > 1
        return AgentCapabilities(
            promptCapabilities=PromptCapabilities(image=True),
            loadSession=False,
            modes=has_modes,
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a conversation session and return its id.

        Parameters
        ----------
        cwd : str
            Working directory advertised by the client. Accepted for
            protocol conformance; not currently used.
        mcp_servers : list[dict[str, Any]] | None
            Client-proposed MCP servers. Ignored with a log notice —
            tool sources are configured in ``openbb.toml`` under
            ``[agent.tool_source_config]``.

        Returns
        -------
        str
            The new session's UUID, used to address subsequent calls.
        """
        if mcp_servers:
            logger.info(
                "acp: ignoring %d client MCP server(s) — tool sources are "
                "configured in openbb.toml ([agent.tool_source_config])",
                len(mcp_servers),
            )
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = _AcpSession(
            conversation_id=session_id,
            profile=self._default_profile,
        )
        return session_id

    async def prompt(
        self,
        session_id: str,
        content: list[ContentBlock],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[SessionUpdate]:
        """Run one turn against the agent loop, streaming ACP updates.

        Appends the prompt to the session history, announces available
        modes on the first prompt of a multi-profile session, drives the
        embedded runtime, and translates each wire event to ACP updates.
        The assembled assistant reply is appended to the session history
        when the turn completes. Runtime failures are caught and surfaced
        as a single ERROR ``StatusUpdate`` rather than propagating.

        Parameters
        ----------
        session_id : str
            Id of a session created by :meth:`new_session`.
        content : list[ContentBlock]
            ACP content blocks for this turn; text parts join into the
            prompt and image parts become uploaded files.
        cancel_event : asyncio.Event | None
            Cooperative cancellation signal. A fresh event is created
            when omitted, allowing :meth:`cancel` to interrupt the turn.

        Yields
        ------
        SessionUpdate
            ACP updates produced while the turn runs.

        Raises
        ------
        ValueError
            If ``session_id`` does not name a known session.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id!r}")

        session.cancel_event = cancel_event or asyncio.Event()

        if not session.modes_announced:
            session.modes_announced = True
            names = self._runtime.settings.all_profile_names()
            if len(names) > 1:
                yield ModeUpdate(
                    currentModeId=session.profile,
                    availableModes=[SessionMode(id=name, name=name) for name in names],
                )

        text, files = _content_blocks_to_turn(content)
        session.messages.append(ChatMessage(role="human", content=text))

        assembled: list[str] = []
        try:
            async for ev in self._runtime.run_turn(
                principal=self._principal,
                conversation_id=session.conversation_id,
                messages=list(session.messages),
                profile=session.profile,
                uploaded_files=files,
                cancel_event=session.cancel_event,
            ):
                if isinstance(ev, MessageChunkSSE):
                    assembled.append(ev.data.delta)
                for update in translate_sse(ev):
                    yield update
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("acp: agent turn failed")
            yield StatusUpdate(text=f"ERROR: agent turn failed — {exc}")
            return
        finally:
            session.cancel_event = None

        final_text = "".join(assembled)
        if final_text:
            session.messages.append(ChatMessage(role="ai", content=final_text))

    async def cancel(self, session_id: str) -> None:
        """Cooperatively cancel the session's in-flight turn, if any.

        Sets the session's cancel event when a turn is running. Unknown
        session ids and idle sessions are silently ignored.

        Parameters
        ----------
        session_id : str
            Id of the session whose current turn should be cancelled.
        """
        session = self._sessions.get(session_id)
        if session is not None and session.cancel_event is not None:
            session.cancel_event.set()

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        """Switch the session's agent profile (ACP mode).

        Parameters
        ----------
        session_id : str
            Id of the session whose profile should change.
        mode_id : str
            Profile name to switch to; must be a configured profile.

        Raises
        ------
        ValueError
            If ``session_id`` does not name a known session.
        KeyError
            If ``mode_id`` is not a configured profile name.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id!r}")
        # Raises KeyError on unknown profile — surfaced to the caller.
        self._runtime.settings.resolve_profile(mode_id)
        session.profile = mode_id


def create_chat_manager(
    explicit_path: str | None = None,
    *,
    settings: AgentServerSettings | None = None,
    profile: str | None = None,
    user_id: str = "pywry-local",
    **chat_kwargs: Any,
) -> Any:
    """Build a ready-to-attach ``pywry.chat.ChatManager``.

    Convenience wrapper: resolves the TOML cascade (or takes explicit
    ``settings``), wraps the provider, and returns a ``ChatManager``
    whose ``toolbar()`` / ``callbacks()`` / ``bind()`` attach to any
    PyWry widget instance. When the resolved settings carry a metadata
    description and no ``welcome_message`` was supplied, that
    description becomes the chat's welcome message.

    Parameters
    ----------
    explicit_path : str | None
        Path passed to the TOML cascade as the highest-priority config
        source. Ignored when ``settings`` is given.
    settings : AgentServerSettings | None
        Pre-resolved settings. When supplied, the TOML cascade is
        skipped and the provider is built directly from them.
    profile : str | None
        Profile new sessions start on. Defaults to the settings'
        default profile.
    user_id : str
        Identity used to scope history and memory.
    **chat_kwargs : Any
        Extra keyword arguments forwarded to ``ChatManager`` (e.g.
        ``welcome_message``, ``show_sidebar``).

    Returns
    -------
    Any
        A configured ``pywry.chat.ChatManager`` instance.
    """
    from pywry.chat.manager import ChatManager

    if settings is not None:
        provider = OpenBBAgentProvider(
            settings,
            profile=profile,
            user_id=user_id,
        )
    else:
        provider = OpenBBAgentProvider.from_toml(
            explicit_path,
            profile=profile,
            user_id=user_id,
        )
    metadata = provider.runtime.settings.metadata
    if metadata.description:
        chat_kwargs.setdefault("welcome_message", metadata.description)
    return ChatManager(provider=provider, **chat_kwargs)
