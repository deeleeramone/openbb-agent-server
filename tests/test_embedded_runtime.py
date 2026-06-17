"""Tests for the in-process EmbeddedRuntime (no HTTP, no pywry)."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from openbb_agent_server.app.settings import AgentServerSettings
from openbb_agent_server.protocol.schemas import ChatMessage, MessageChunkSSE
from openbb_agent_server.runtime import services
from openbb_agent_server.runtime.embedded import EmbeddedRuntime
from openbb_agent_server.runtime.principal import UserPrincipal


async def _collect_turn(
    runtime: EmbeddedRuntime,
    principal: UserPrincipal,
    text: str,
    *,
    conversation_id: str = "conv-1",
    cancel_event: asyncio.Event | None = None,
) -> list:
    return [
        ev
        async for ev in runtime.run_turn(
            principal=principal,
            conversation_id=conversation_id,
            messages=[ChatMessage(role="human", content=text)],
            cancel_event=cancel_event,
        )
    ]


async def test_run_turn_streams_message_chunks(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
) -> None:
    runtime = EmbeddedRuntime(settings_env)
    try:
        events = await _collect_turn(runtime, alice, "Hello")
        deltas = [ev.data.delta for ev in events if isinstance(ev, MessageChunkSSE)]
        assert "".join(deltas) == "OK."
    finally:
        await runtime.aclose()


async def test_run_turn_persists_human_and_ai_messages(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
) -> None:
    runtime = EmbeddedRuntime(settings_env)
    try:
        await _collect_turn(runtime, alice, "Hello", conversation_id="conv-hist")
        history = services.get_history()
        records = await history.get_messages(
            principal=alice, conversation_id="conv-hist"
        )
        assert [(r.role, r.content) for r in records] == [
            ("human", "Hello"),
            ("ai", "OK."),
        ]
        conversations = await history.list_conversations(principal=alice)
        assert any(c.get("conversation_id") == "conv-hist" for c in conversations)
    finally:
        await runtime.aclose()


async def test_run_turn_pre_set_cancel_event_yields_nothing(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
) -> None:
    cancel = asyncio.Event()
    cancel.set()
    runtime = EmbeddedRuntime(settings_env)
    try:
        events = await _collect_turn(runtime, alice, "Hello", cancel_event=cancel)
        assert events == []
        history = services.get_history()
        records = await history.get_messages(principal=alice, conversation_id="conv-1")
        # The human message is recorded; the AI reply never lands.
        assert [r.role for r in records] == ["human"]
    finally:
        await runtime.aclose()


async def test_start_is_idempotent_and_aclose_resets(
    settings_env: AgentServerSettings,
) -> None:
    runtime = EmbeddedRuntime(settings_env)
    await runtime.start()
    await runtime.start()
    assert runtime.started
    await runtime.aclose()
    assert not runtime.started
    with pytest.raises(RuntimeError):
        services.get_history()


async def test_unknown_profile_raises(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
) -> None:
    runtime = EmbeddedRuntime(settings_env)
    try:
        with pytest.raises(KeyError):
            async for _ in runtime.run_turn(
                principal=alice,
                conversation_id="conv-x",
                messages=[ChatMessage(role="human", content="hi")],
                profile="not-a-profile",
            ):
                pass
    finally:
        await runtime.aclose()


def test_from_toml_runs_the_layered_cascade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.app import config as cfg_mod

    cfg = tmp_path / "embedded.toml"
    cfg.write_text(
        textwrap.dedent(
            """
            [agent]
            model_provider = "fake"
            checkpointer_provider = "inmemory"
            middleware = []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    # Keep the repo's own openbb.toml and the user-global layer out of
    # the cascade so only the explicit file above contributes.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", tmp_path / "no_user_dir")
    runtime = EmbeddedRuntime.from_toml(str(cfg))
    assert runtime.settings.model_provider == "fake"
    assert runtime.settings.checkpointer_provider == "inmemory"
    assert not runtime.started


def test_principal_defaults_to_embedded_scopes(
    settings_env: AgentServerSettings,
) -> None:
    runtime = EmbeddedRuntime(settings_env)
    principal = runtime.principal("desk-user", display_name="Desk User")
    assert principal.user_id == "desk-user"
    assert principal.has_scope("agent:query")
    assert principal.has_scope("memory:write")


async def test_run_turn_survives_ingestion_failure(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(**_kwargs):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(
        "openbb_agent_server.memory.ingestion.ingest_request_context", _boom
    )
    runtime = EmbeddedRuntime(settings_env)
    try:
        events = await _collect_turn(runtime, alice, "Hello")
        deltas = [ev.data.delta for ev in events if isinstance(ev, MessageChunkSSE)]
        assert "".join(deltas) == "OK."
    finally:
        await runtime.aclose()


async def test_run_turn_closes_after_function_call_dispatch(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.protocol.schemas import (
        FunctionCallSSE,
        FunctionCallSSEData,
        MessageChunkSSE as ChunkSSE,
        MessageChunkSSEData,
    )

    async def _fake_run_agent(**_kwargs):
        yield ChunkSSE(data=MessageChunkSSEData(delta="before "))
        yield FunctionCallSSE(data=FunctionCallSSEData(function="get_widget_data"))
        yield ChunkSSE(data=MessageChunkSSEData(delta="never"))

    monkeypatch.setattr(
        "openbb_agent_server.runtime.builder.run_agent", _fake_run_agent
    )
    runtime = EmbeddedRuntime(settings_env)
    try:
        events = await _collect_turn(runtime, alice, "Hello", conversation_id="conv-fc")
        # The stream closes right after the dispatch event.
        assert [type(ev).__name__ for ev in events] == [
            "MessageChunkSSE",
            "FunctionCallSSE",
        ]
        # A dispatched turn does not persist a partial AI reply.
        history = services.get_history()
        records = await history.get_messages(principal=alice, conversation_id="conv-fc")
        assert [r.role for r in records] == ["human"]
    finally:
        await runtime.aclose()


async def test_run_turn_propagates_cancelled_error(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _cancelled_run_agent(**_kwargs):
        raise asyncio.CancelledError
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(
        "openbb_agent_server.runtime.builder.run_agent", _cancelled_run_agent
    )
    runtime = EmbeddedRuntime(settings_env)
    try:
        with pytest.raises(asyncio.CancelledError):
            await _collect_turn(runtime, alice, "Hello")
    finally:
        await runtime.aclose()


async def test_run_turn_handles_consumer_abort(
    settings_env: AgentServerSettings,
    alice: UserPrincipal,
) -> None:
    """Closing the generator mid-stream still ends the trace cleanly."""
    runtime = EmbeddedRuntime(settings_env)
    try:
        gen = runtime.run_turn(
            principal=alice,
            conversation_id="conv-abort",
            messages=[ChatMessage(role="human", content="Hello")],
        )
        first = await gen.__anext__()
        assert first is not None
        await gen.aclose()
        # The human message landed; the abandoned AI reply did not.
        history = services.get_history()
        records = await history.get_messages(
            principal=alice, conversation_id="conv-abort"
        )
        assert [r.role for r in records] == ["human"]
    finally:
        await runtime.aclose()
