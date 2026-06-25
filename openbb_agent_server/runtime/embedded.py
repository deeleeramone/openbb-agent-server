"""In-process agent runtime — the agent loop without the HTTP app.

``EmbeddedRuntime`` owns the same stores and checkpointer the FastAPI
app wires in its lifespan, and exposes ``run_turn`` as the single entry
point: a ``QueryRequest``-shaped turn in, the same ``SSEEvent`` stream
the ``/v1/query`` endpoint produces out. Adapters that re-expose the
loop over another protocol (e.g. the ACP shim in
``openbb_agent_server.acp``) build on this instead of going through
HTTP.

One runtime per process — the shared service slots in
``runtime.services`` are global, the same constraint the HTTP app has.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from openbb_agent_server.app.settings import AgentProfile, AgentServerSettings
from openbb_agent_server.protocol.schemas import (
    ChatMessage,
    FunctionCallSSE,
    MessageChunkSSE,
    QueryRequest,
    SSEEvent,
    UploadedFile,
)
from openbb_agent_server.runtime import (
    context as run_context,
    registry,
    services,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal

if TYPE_CHECKING:  # pragma: no cover — import cycle guard, types only
    from openbb_agent_server.app.app import AgentStores

logger = logging.getLogger("openbb_agent_server.embedded")

DEFAULT_EMBEDDED_SCOPES: tuple[str, ...] = (
    "agent:query",
    "memory:read",
    "memory:write",
)


class EmbeddedRuntime:
    """Drive the agent loop in-process, no FastAPI required.

    Composition mirrors ``app.create_app`` + its lifespan: the same
    ``build_stores`` call, the same checkpointer plugin, the same
    ``run_agent`` seam. The retention prune sweep is the one server
    behaviour intentionally not started here — embedding hosts decide
    their own retention story.
    """

    def __init__(self, settings: AgentServerSettings) -> None:
        """Create the runtime from resolved settings.

        Stores and the checkpointer are bound lazily on first use, so
        construction is cheap and does no I/O.

        Parameters
        ----------
        settings : AgentServerSettings
            The resolved configuration driving the agent loop, model
            provider, tool sources, and persistence.
        """
        self._settings = settings
        self._stores: AgentStores | None = None
        self._checkpointer_provider: Any = None
        self._checkpointer: Any = None
        self._started = False
        self._start_lock = asyncio.Lock()

    @classmethod
    def from_toml(cls, explicit_path: str | None = None) -> EmbeddedRuntime:
        """Build a runtime from the layered TOML cascade.

        Runs the exact bootstrap the server CLI runs — ``.env`` files,
        ``~/.openbb_platform/openbb.toml``, project ``openbb.toml``,
        ``explicit_path`` / ``$OPENBB_CONFIG`` — so the same config
        file drives both deployments.

        Parameters
        ----------
        explicit_path : str or None, optional
            Path to a specific ``openbb.toml`` to layer on top of the
            cascade. Falls back to ``$OPENBB_CONFIG`` and the standard
            discovery locations when not given.

        Returns
        -------
        EmbeddedRuntime
            An unstarted runtime built from the resolved agent settings.
            Call :meth:`start` (or ``run_turn``, which starts lazily)
            before use.
        """
        from openbb_agent_server.app.config import (
            agent_section,
            bootstrap_launcher_config,
        )

        cfg = bootstrap_launcher_config(explicit_path=explicit_path)
        return cls(AgentServerSettings.from_toml(agent_section(cfg)))

    @property
    def settings(self) -> AgentServerSettings:
        """The resolved settings this runtime was built from."""
        return self._settings

    @property
    def started(self) -> bool:
        """True once ``start()`` has bound stores + checkpointer."""
        return self._started

    def principal(
        self,
        user_id: str = "embedded",
        *,
        display_name: str | None = None,
        scopes: Sequence[str] = DEFAULT_EMBEDDED_SCOPES,
    ) -> UserPrincipal:
        """Synthesize a principal for a local single-user embedding.

        Parameters
        ----------
        user_id : str, optional
            Stable identifier for the embedding's single user. Defaults
            to ``"embedded"``.
        display_name : str or None, optional
            Human-readable name for the principal.
        scopes : Sequence[str], optional
            Authorization scopes granted to the principal. Defaults to
            :data:`DEFAULT_EMBEDDED_SCOPES` (agent query plus memory
            read/write).

        Returns
        -------
        UserPrincipal
            A principal carrying the given identity and scopes, suitable
            for passing to :meth:`run_turn`.
        """
        return UserPrincipal(
            user_id=user_id,
            display_name=display_name,
            scopes=tuple(scopes),
        )

    async def start(self) -> None:
        """Bind stores + checkpointer. Idempotent."""
        async with self._start_lock:
            if self._started:
                return
            from openbb_agent_server.app.app import build_stores
            from openbb_agent_server.observability.logging import (
                install_trace_logging,
            )
            from openbb_agent_server.runtime.identity import warn_if_pepper_unset

            self._settings.data_dir.mkdir(parents=True, exist_ok=True)
            install_trace_logging()
            warn_if_pepper_unset()
            self._stores = build_stores(self._settings)
            await self._stores.history.init_schema()
            self._checkpointer_provider = registry.load(
                "openbb_agent_server.checkpointers",
                self._settings.checkpointer_provider,
                self._settings.checkpointer_config,
            )
            self._checkpointer = await self._checkpointer_provider.open(self._settings)
            services.set_services(checkpointer=self._checkpointer)
            self._started = True
            logger.info(
                "embedded runtime started | model=%s/%s checkpointer=%s",
                self._settings.model_provider,
                self._settings.model_name,
                self._settings.checkpointer_provider,
            )

    async def aclose(self) -> None:
        """Release the checkpointer + stores and reset the service slots."""
        if not self._started:
            return
        try:
            if self._checkpointer_provider is not None:
                await self._checkpointer_provider.close(self._checkpointer)
        finally:
            await services.areset()
            self._started = False

    async def run_turn(
        self,
        *,
        principal: UserPrincipal,
        conversation_id: str,
        messages: Sequence[ChatMessage],
        profile: str | None = None,
        timezone: str | None = None,
        uploaded_files: Sequence[UploadedFile] = (),
        cancel_event: asyncio.Event | None = None,
        workspace_options: dict[str, Any] | None = None,
        model_config_overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[SSEEvent]:
        """Run one conversation turn and yield wire-protocol SSE events.

        Mirrors the ``/v1/query`` lifecycle: trace bookkeeping in the
        history store, human/AI message persistence, memory ingestion,
        then the ``run_agent`` stream under a bound ``RunContext``.
        ``messages`` is the full multi-turn history, last entry the new
        human message — the same contract the HTTP endpoint has.

        Starts the runtime lazily if needed. The stream ends after a
        ``FunctionCallSSE`` (status ``"dispatched"``) just like the wire
        protocol, and the assembled AI text is persisted only on a clean
        ``"completed"`` run. The trace is always ended, even on cancel.

        Parameters
        ----------
        principal : UserPrincipal
            The acting user; used for history upsert, message
            attribution and memory scoping. See :meth:`principal`.
        conversation_id : str
            Identifier of the conversation this turn belongs to; also
            the checkpointer thread key.
        messages : Sequence[ChatMessage]
            Full multi-turn history, with the new human message last.
        profile : str or None, optional
            Agent profile name to resolve; falls back to the settings'
            default profile when not given.
        timezone : str or None, optional
            IANA timezone used to localize the run.
        uploaded_files : Sequence[UploadedFile], optional
            Files attached to this turn, ingested before the agent runs.
        cancel_event : asyncio.Event or None, optional
            When set during streaming, stops the run and marks the trace
            ``"cancelled"`` without persisting AI output.
        workspace_options : dict[str, Any] or None, optional
            Per-user feature toggles (e.g. ``{"search-web": True}``).
            Passed through to :class:`RunContext` so tool sources can
            query them via :meth:`RunContext.has_workspace_option`.
        model_config_overrides : dict[str, Any] or None, optional
            Per-session overrides for model configuration keys (e.g.
            ``{"temperature": 0.7}``). Merged over the profile's
            ``model_config_`` before the agent runs.

        Yields
        ------
        SSEEvent
            Wire-protocol SSE events as the agent produces them, the
            same stream the ``/v1/query`` endpoint emits.

        Raises
        ------
        RuntimeError
            If startup completes without binding the stores.
        """
        if not self._started:
            await self.start()
        if self._stores is None:  # pragma: no cover — start() guarantees stores
            raise RuntimeError("EmbeddedRuntime.start() did not bind stores")

        settings = self._settings
        prof: AgentProfile = settings.resolve_profile(
            profile or settings.default_profile
        )
        if model_config_overrides:
            merged = {**prof.model_config_, **model_config_overrides}
            prof = prof.model_copy(update={"model_config_": merged})
        history = self._stores.history

        await history.upsert_user(principal)

        trace_id = str(uuid.uuid4())
        turn_idx = sum(1 for m in messages if m.role == "human")
        run_id = f"{conversation_id}:turn{turn_idx}"

        body = QueryRequest(
            messages=list(messages),
            uploaded_files=list(uploaded_files),
            timezone=timezone,
        )

        from openbb_agent_server.app.router import (
            _collect_uploaded_files_with_ingest,
        )

        ctx = RunContext(
            principal=principal,
            trace_id=trace_id,
            run_id=run_id,
            conversation_id=conversation_id,
            agent_name=prof.name,
            timezone=timezone,
            uploaded_files=await _collect_uploaded_files_with_ingest(
                body, principal=principal
            ),
            workspace_options=dict(workspace_options) if workspace_options else {},
        )

        await history.begin_trace(
            principal=principal,
            trace_id=trace_id,
            conversation_id=conversation_id,
            run_id=run_id,
        )
        if body.messages and body.messages[-1].role == "human":
            last_content = body.messages[-1].content
            await history.append_message(
                principal=principal,
                conversation_id=conversation_id,
                role="human",
                content=last_content
                if isinstance(last_content, str)
                else str(last_content or ""),
                trace_id=trace_id,
            )

        try:
            from openbb_agent_server.memory.ingestion import ingest_request_context

            await ingest_request_context(
                principal=principal,
                store=self._stores.memory,
                body=body,
                trace_id=trace_id,
                char_threshold=settings.ingest_char_threshold,
                chunk_chars=settings.ingest_chunk_chars,
                chunk_overlap=settings.ingest_chunk_overlap,
                translator=self._stores.translator
                if settings.translate_for_ingestion
                else None,
                translate_target_lang=settings.ingest_target_language,
            )
        except Exception:
            logger.warning("context ingestion errored", exc_info=True)

        from openbb_agent_server.runtime.builder import run_agent

        assembled: list[str] = []
        status = "completed"
        # ``run_agent`` is an async-gen function, so the object carries
        # ``aclose`` — the cast surfaces that past the declared
        # ``AsyncIterator`` return type.
        run_iter = cast(
            "AsyncGenerator[SSEEvent, None]",
            run_agent(ctx=ctx, body=body, settings=settings, profile=prof),
        )
        try:
            with run_context.bind(ctx):
                try:
                    async for ev in run_iter:
                        if cancel_event is not None and cancel_event.is_set():
                            status = "cancelled"
                            break
                        if isinstance(ev, MessageChunkSSE):
                            assembled.append(ev.data.delta)
                        yield ev
                        if isinstance(ev, FunctionCallSSE):
                            # The wire protocol closes the stream after a
                            # client-side dispatch; embedded hosts get the
                            # same contract.
                            status = "dispatched"
                            break
                finally:
                    with suppress(Exception):
                        await run_iter.aclose()
            final_ai_text = "".join(assembled)
            if final_ai_text and status == "completed":
                await history.append_message(
                    principal=principal,
                    conversation_id=conversation_id,
                    role="ai",
                    content=final_ai_text,
                    trace_id=trace_id,
                )
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except GeneratorExit:
            # Consumer closed the stream mid-run (host teardown).
            status = "cancelled"
            raise
        finally:
            with suppress(Exception):
                await asyncio.shield(
                    history.end_trace(
                        principal=principal,
                        trace_id=trace_id,
                        status=status,
                    )
                )
