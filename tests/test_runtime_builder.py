"""Builder stream-event normalisation tests."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from openbb_agent_server.runtime.builder import _normalise_stream_event
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def test_normalise_messages_event() -> None:
    """The normaliser projects an AIMessageChunk into a stable dict."""
    msg = AIMessageChunk(content="hi", tool_calls=[])
    out = _normalise_stream_event(((), "messages", (msg, {"meta": "x"})))
    assert out is not None
    assert out["type"] == "messages"
    assert out["ns"] == []
    message = out["data"]["message"]
    assert message["content"] == "hi"
    assert message["tool_calls"] == []
    assert message["tool_call_chunks"] == []
    assert message["additional_kwargs"] == {}
    assert "id" in message


def test_normalise_messages_event_with_tool_calls() -> None:
    msg = AIMessageChunk(
        content="",
        tool_calls=[{"name": "x", "args": {}, "id": "c1"}],
    )
    out = _normalise_stream_event(((), "messages", (msg, {})))
    assert out is not None
    assert out["data"]["message"]["tool_calls"][0]["name"] == "x"


def test_normalise_messages_event_missing_payload_drops() -> None:
    out = _normalise_stream_event(((), "messages", ()))
    assert out is None


def test_normalise_updates_event() -> None:
    out = _normalise_stream_event(((), "updates", {"agent": {"messages": []}}))
    assert out == {
        "type": "updates",
        "ns": [],
        "data": {"agent": {"messages": []}},
    }


def test_normalise_updates_with_none_data() -> None:
    out = _normalise_stream_event(((), "updates", None))
    assert out == {"type": "updates", "ns": [], "data": {}}


def test_normalise_custom_dict_passthrough() -> None:
    payload = {"type": "chunk", "content": "x"}
    out = _normalise_stream_event(((), "custom", payload))
    assert out == {"type": "custom", "ns": [], "data": payload}


def test_normalise_custom_non_dict_wraps_as_message() -> None:
    out = _normalise_stream_event(((), "custom", "raw text"))
    assert out == {"type": "custom", "ns": [], "data": {"message": "raw text"}}


def test_normalise_unknown_mode_returns_none() -> None:
    assert _normalise_stream_event(((), "unknown_mode", {})) is None


def test_normalise_non_tuple_returns_none() -> None:
    assert _normalise_stream_event("not a tuple") is None


def test_normalise_wrong_arity_tuple_returns_none() -> None:
    assert _normalise_stream_event(("a", "b")) is None


def test_normalise_subagent_namespace_propagates() -> None:
    msg = AIMessageChunk(content="from-subagent")
    out = _normalise_stream_event((("tools:abc",), "messages", (msg, {})))
    assert out is not None
    assert out["ns"] == ["tools:abc"]


def test_normalise_messages_event_with_none_content_substitutes_empty_string() -> None:
    """An AIMessageChunk whose content is None normalises to an empty string."""
    msg = AIMessageChunk(content="")
    object.__setattr__(msg, "content", None)
    out = _normalise_stream_event(((), "messages", (msg, {})))
    assert out is not None
    assert out["data"]["message"]["content"] == ""


def test_runtime_builder_tool_messages_round_trip_as_toolmessage() -> None:
    """Round-trip role:tool messages as ToolMessage."""
    from langchain_core.messages import ToolMessage

    from openbb_agent_server.protocol.schemas import ChatMessage
    from openbb_agent_server.runtime.builder import _to_lc_messages

    out = _to_lc_messages(
        [
            ChatMessage(role="human", content="hi"),
            ChatMessage(role="ai", content="ok"),
            ChatMessage(role="tool", content="result", tool_call_id="abc"),
        ]
    )
    assert isinstance(out[0], HumanMessage)
    assert isinstance(out[1], AIMessage)
    assert isinstance(out[-1], ToolMessage)
    assert out[-1].tool_call_id == "abc"


def test_runtime_builder_resolves_middleware_from_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.app.settings import AgentMetadata, AgentProfile
    from openbb_agent_server.runtime import builder

    sentinel_mw = object()

    class _FakeMW:
        def build(self, ctx: Any, config: dict[str, Any]) -> Any:
            assert "model" in config
            return sentinel_mw

    monkeypatch.setattr(
        "openbb_agent_server.runtime.builder.registry.load",
        lambda group, name: _FakeMW(),
    )

    profile = AgentProfile(
        name="p",
        model_provider="fake",
        model_name="x",
        model_config={},
        tool_sources=(),
        subagents=(),
        middleware=("call_limit",),
        skills=(),
        features={},
        system_prompt_file=None,
        metadata=AgentMetadata(),
        tool_source_config={},
    )
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )
    out = builder._resolve_middleware(ctx, profile, model=object())
    assert out == [sentinel_mw]


def test_runtime_builder_subagent_carries_model_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward a non-None spec.model into the subagent dict."""
    from openbb_agent_server.app.settings import (
        AgentMetadata,
        AgentProfile,
    )
    from openbb_agent_server.runtime import builder

    sentinel_model = object()

    class _Spec:
        name = "researcher"
        description = "d"
        system_prompt = "s"
        tools: tuple[str, ...] = ()
        model = sentinel_model

    monkeypatch.setattr(
        "openbb_agent_server.runtime.builder.registry.load",
        lambda _group, _name: _Spec(),
    )

    profile = AgentProfile(
        name="p",
        model_provider="fake",
        model_name="x",
        model_config={},
        tool_sources=(),
        subagents=("researcher",),
        middleware=(),
        skills=(),
        features={},
        system_prompt_file=None,
        metadata=AgentMetadata(),
        tool_source_config={},
    )
    out = builder._resolve_subagents(profile, main_tools=[])
    assert out and out[0]["model"] is sentinel_model


@pytest.mark.asyncio
async def test_runtime_builder_run_agent_resolves_profile_when_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Resolve the profile when run_agent is given none."""
    import sys
    import types

    from openbb_agent_server.app.settings import AgentServerSettings
    from openbb_agent_server.protocol.schemas import QueryRequest

    fake_module = types.ModuleType("deepagents")

    class _FakeAgent:
        async def astream(self, *a: Any, **kw: Any):
            if False:  # pragma: no cover — make this an async generator
                yield None

    fake_module.create_deep_agent = lambda **_kw: _FakeAgent()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepagents", fake_module)

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")

    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.runtime import services as _services_mod

    _services_mod.set_services(checkpointer=InMemorySaver())
    settings = AgentServerSettings()

    real_resolve = AgentServerSettings.resolve_profile
    seen: list[str | None] = []

    def _patched(self: Any, name: str | None = None):
        seen.append(name)
        return real_resolve(self, name)

    monkeypatch.setattr(AgentServerSettings, "resolve_profile", _patched)

    from openbb_agent_server.runtime.builder import run_agent

    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        agent_name="default",
    )
    body = QueryRequest(
        messages=[{"role": "human", "content": "hi"}],
        conversation_id="c",
        run_id="r",
    )

    gen = run_agent(ctx=ctx, body=body, settings=settings, profile=None)
    with contextlib.suppress(StopAsyncIteration, Exception):
        async for _ev in gen:
            pass
    assert "default" in seen


def test_runtime_builder_passes_skills_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Pass profile skills through to the agent kwargs."""
    import sys
    import types

    captured_kwargs: dict[str, Any] = {}

    class _FakeAgent:
        async def astream(self, *a: Any, **kw: Any):
            if False:  # pragma: no cover — make this an async generator
                yield None

    def _capture(**kwargs: Any) -> _FakeAgent:
        captured_kwargs.update(kwargs)
        return _FakeAgent()

    fake_module = types.ModuleType("deepagents")
    fake_module.create_deep_agent = _capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepagents", fake_module)

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["x"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SKILLS", '["/skills/finance"]')

    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.app.settings import AgentServerSettings
    from openbb_agent_server.protocol.schemas import QueryRequest
    from openbb_agent_server.runtime import services
    from openbb_agent_server.runtime.builder import run_agent

    services.set_services(checkpointer=InMemorySaver())

    settings = AgentServerSettings()
    profile = settings.resolve_profile()

    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        agent_name="default",
    )
    body = QueryRequest(
        messages=[{"role": "human", "content": "hi"}],
        conversation_id="c",
        run_id="r",
    )

    seen_events: list[Any] = []

    captured_error: list[BaseException] = []

    async def _drive() -> None:
        gen = run_agent(ctx=ctx, body=body, settings=settings, profile=profile)
        try:
            async for ev in gen:
                seen_events.append(ev)
        except BaseException as exc:  # noqa: BLE001
            captured_error.append(exc)

    asyncio.run(_drive())
    if "skills" not in captured_kwargs:
        raise AssertionError(
            f"create_deep_agent never received skills kwarg. "
            f"captured_kwargs={captured_kwargs!r}, "
            f"seen_events={[(type(e).__name__, getattr(e, 'message', None)) for e in seen_events]}, "
            f"errors={captured_error!r}"
        )
    assert captured_kwargs["skills"] == ["/skills/finance"]
