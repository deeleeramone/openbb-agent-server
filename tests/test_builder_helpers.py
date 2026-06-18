"""Unit tests for the runtime builder helpers."""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage

from openbb_agent_server.observability.logging import TRACE
from openbb_agent_server.persistence import models as m
from openbb_agent_server.protocol.schemas import ChatMessage
from openbb_agent_server.runtime import services
from openbb_agent_server.runtime.builder import (
    _build_turn_addendum,
    _parse_function_call_envelope,
    _render_file_snapshot,
    _render_widget_snapshot,
    _resolve_subagents,
    _to_lc_messages,
    _tool_message_content,
    _tool_message_entry_chunks,
)
from openbb_agent_server.runtime.context import FileRef, RunContext, WidgetRef
from openbb_agent_server.runtime.principal import UserPrincipal
from openbb_agent_server.runtime.widget_store import WidgetDataStore


def _ctx(**kw: Any) -> RunContext:
    defaults: dict[str, Any] = {
        "principal": UserPrincipal(user_id="u"),
        "trace_id": "t",
        "run_id": "r",
        "conversation_id": "c",
    }
    defaults.update(kw)
    return RunContext(**defaults)


def test_render_file_snapshot_returns_empty_when_no_files() -> None:
    assert _render_file_snapshot(_ctx()) == ""


def test_render_file_snapshot_lists_uploaded_files() -> None:
    files = (
        FileRef(name="a.pdf", mime="application/pdf", data_base64="x"),
        FileRef(name="b.png", mime="image/png", data_base64="y"),
    )
    out = _render_file_snapshot(_ctx(uploaded_files=files))
    assert "a.pdf" in out
    assert "application/pdf" in out
    assert "b.png" in out
    assert "image/png" in out
    assert "Uploaded files" in out


def test_render_widget_snapshot_returns_empty_when_no_widgets() -> None:
    assert _render_widget_snapshot(_ctx()) == ""


def test_render_widget_snapshot_lists_widgets() -> None:
    widgets = (
        WidgetRef(
            uuid="u-1",
            widget_id="balance",
            origin="market",
            params={"sym": "AAPL"},
            data=[{"x": 1}],
            name="Balance Sheet",
        ),
    )
    out = _render_widget_snapshot(_ctx(widgets=widgets))
    assert "u-1" in out
    assert "Balance Sheet" in out
    assert "AAPL" in out


def test_parse_function_call_envelope_non_string() -> None:
    assert _parse_function_call_envelope(42) is None


def test_parse_function_call_envelope_non_json_returns_none() -> None:
    assert _parse_function_call_envelope("hello world") is None


def test_parse_function_call_envelope_string_without_function_key() -> None:
    assert _parse_function_call_envelope('{"x": 1}') is None


def test_parse_function_call_envelope_unparseable_json() -> None:
    assert _parse_function_call_envelope('{"function": broken') is None


def test_parse_function_call_envelope_with_partial_payload() -> None:
    """Reject a JSON dict with function but no input_arguments."""
    assert _parse_function_call_envelope('{"function": "x"}') is None


def test_parse_function_call_envelope_valid() -> None:
    raw = '{"function": "get_widget_data", "input_arguments": {"k": "v"}}'
    out = _parse_function_call_envelope(raw)
    assert out is not None
    assert out["function"] == "get_widget_data"


def test_tool_message_content_none() -> None:
    assert _tool_message_content(None) == ""


def test_tool_message_content_string() -> None:
    assert _tool_message_content("plain") == "plain"


def test_tool_message_content_other_type() -> None:
    assert _tool_message_content(42) == "42"


def test_tool_message_content_list_with_items() -> None:
    data = [{"items": [{"content": "a"}, {"content": "b"}]}]
    out = _tool_message_content(data)
    assert "a" in out
    assert "b" in out


def test_tool_message_entry_chunks_non_dict() -> None:
    assert _tool_message_entry_chunks("plain") == ["plain"]


def test_tool_message_entry_chunks_items_with_strings() -> None:
    out = _tool_message_entry_chunks({"items": ["plain", None, 9]})
    assert "plain" in out
    assert "9" in out


def test_tool_message_entry_chunks_items_with_dict_no_content() -> None:
    out = _tool_message_entry_chunks({"items": [{"other": "skip"}]})
    assert out == []


def test_tool_message_entry_chunks_falls_back_to_content_key() -> None:
    out = _tool_message_entry_chunks({"content": "hi"})
    assert out == ["hi"]


def test_tool_message_entry_chunks_falls_back_to_data_key() -> None:
    out = _tool_message_entry_chunks({"data": {"k": "v"}})
    assert out and "v" in out[0]


def test_tool_message_entry_chunks_last_resort_json_dump() -> None:
    out = _tool_message_entry_chunks({"random": "field"})
    assert out[0].startswith("{")
    assert "random" in out[0]


def test_tool_message_entry_chunks_non_string_content_value() -> None:
    out = _tool_message_entry_chunks({"content": 42})
    assert out == ["42"]


def test_tool_message_entry_chunks_non_string_inner_value() -> None:
    out = _tool_message_entry_chunks({"items": [{"content": 42}]})
    assert out == ["42"]


def test_to_lc_messages_synthesises_ai_for_orphan_tool_message() -> None:
    """Synthesise a pairing AIMessage for an orphan tool message."""
    from langchain_core.messages import ToolMessage

    msgs = [ChatMessage(role="tool", content="x", tool_call_id="1")]
    out = _to_lc_messages(msgs)
    assert len(out) == 2
    assert isinstance(out[0], AIMessage)
    assert out[0].tool_calls and out[0].tool_calls[0]["id"] == "1"
    assert isinstance(out[1], ToolMessage)
    assert out[1].tool_call_id == "1"


def test_to_lc_messages_drops_envelope_only_ai() -> None:
    msgs = [
        ChatMessage(
            role="ai",
            content=('{"function": "get_widget_data", "input_arguments": {}}'),
        )
    ]
    assert _to_lc_messages(msgs) == []


def test_to_lc_messages_drops_blank_ai() -> None:
    msgs = [ChatMessage(role="ai", content="   ")]
    assert _to_lc_messages(msgs) == []


def test_to_lc_messages_keeps_real_ai() -> None:
    msgs = [ChatMessage(role="ai", content="hello")]
    out = _to_lc_messages(msgs)
    assert len(out) == 1
    assert isinstance(out[0], AIMessage)


def test_to_lc_messages_keeps_human() -> None:
    msgs = [ChatMessage(role="human", content="hi")]
    out = _to_lc_messages(msgs)
    assert len(out) == 1
    assert isinstance(out[0], HumanMessage)


def test_to_lc_messages_handles_none_content() -> None:
    msgs = [ChatMessage(role="human", content=None)]
    out = _to_lc_messages(msgs)
    assert len(out) == 1
    assert out[0].content == ""


def test_to_lc_messages_serialises_non_string_content() -> None:
    msgs = [ChatMessage(role="human", content={"k": "v"})]
    out = _to_lc_messages(msgs)
    assert len(out) == 1
    assert "k" in out[0].content


@pytest_asyncio.fixture
async def widget_store_in_services() -> Any:
    s = WidgetDataStore("sqlite+aiosqlite:///:memory:")
    async with s._engine.begin() as conn:
        await conn.run_sync(m.Base.metadata.create_all)
    services.set_services(history=None, widget_store=s)
    try:
        yield s
    finally:
        services.reset()
        await s.aclose()


@pytest.mark.asyncio
async def test_turn_addendum_returns_empty_when_not_ingested() -> None:
    out = await _build_turn_addendum(_ctx(), body=None, has_ingested=False)
    assert out == ""


@pytest.mark.asyncio
async def test_turn_addendum_returns_empty_when_no_widget_store() -> None:
    services.reset()
    out = await _build_turn_addendum(_ctx(), body=None, has_ingested=True)
    assert out == ""


@pytest.mark.asyncio
async def test_turn_addendum_returns_empty_when_store_has_no_entries(
    widget_store_in_services: WidgetDataStore,
) -> None:
    out = await _build_turn_addendum(_ctx(), body=None, has_ingested=True)
    assert out == ""


@pytest.mark.asyncio
async def test_turn_addendum_emits_sql_surface(
    widget_store_in_services: WidgetDataStore,
) -> None:
    await widget_store_in_services.record(
        principal=_ctx().principal,
        conversation_id="c",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="market",
        input_args={},
        rows=[{"x": 1}, {"x": 2}],
        columns=["x"],
    )
    out = await _build_turn_addendum(_ctx(), body=None, has_ingested=True)
    assert "prices" in out
    assert "rows=2" in out
    assert "query_widget_data" in out
    assert "read_widget_data" in out


@pytest.mark.asyncio
async def test_turn_addendum_swallows_schema_errors(
    monkeypatch: pytest.MonkeyPatch,
    widget_store_in_services: WidgetDataStore,
) -> None:
    async def broken_schema(**_kw: Any) -> Any:
        raise RuntimeError("nope")

    monkeypatch.setattr(widget_store_in_services, "schema", broken_schema)
    out = await _build_turn_addendum(_ctx(), body=None, has_ingested=True)
    assert out == ""


def test_resolve_subagents_resolves_known_subagent_with_tools() -> None:
    """Attach tools referenced by a known subagent."""
    import types as _types

    from openbb_agent_server.runtime import registry
    from openbb_agent_server.runtime.plugins import SubAgentSpec

    class _ResearcherStub(SubAgentSpec):
        name = "researcher"
        description = "stub researcher"
        system_prompt = "be a researcher"
        tools = ("real-tool",)
        model: Any = None

    profile = _types.SimpleNamespace(subagents=("researcher",))

    class _Tool:
        name = "real-tool"

    main_tools = [_Tool()]

    real_load = registry.load
    try:
        registry.load = (  # type: ignore[assignment]
            lambda group, name, *a, **kw: (
                _ResearcherStub()
                if group == "openbb_agent_server.subagents"
                else real_load(group, name, *a, **kw)
            )
        )
        out = _resolve_subagents(profile, main_tools)
    finally:
        registry.load = real_load  # type: ignore[assignment]

    assert len(out) == 1
    [entry] = out
    assert entry["name"] == "researcher"
    assert entry["tools"] == [main_tools[0]]


def test_resolve_subagents_drops_unknown_tools() -> None:
    """Drop a wanted tool that is not on the main agent."""
    import types as _types

    from openbb_agent_server.runtime import registry
    from openbb_agent_server.runtime.plugins import SubAgentSpec

    class _Stub(SubAgentSpec):
        name = "researcher"
        description = "stub"
        system_prompt = "be a researcher"
        tools = ("missing-tool",)
        model: Any = None

    profile = _types.SimpleNamespace(subagents=("researcher",))

    real_load = registry.load
    try:
        registry.load = (  # type: ignore[assignment]
            lambda group, name, *a, **kw: (
                _Stub()
                if group == "openbb_agent_server.subagents"
                else real_load(group, name, *a, **kw)
            )
        )
        out = _resolve_subagents(profile, main_tools=[])
    finally:
        registry.load = real_load  # type: ignore[assignment]

    assert len(out) == 1
    assert "tools" not in out[0]


def test_render_widget_snapshot_includes_description_when_present() -> None:
    """Include a description line for a widget with a description."""
    w = WidgetRef(
        uuid="u-1",
        widget_id="balance",
        origin="market",
        params={"sym": "AAPL"},
        data=None,
        name="Balance Sheet",
        description="quarterly balance sheet",
    )
    out = _render_widget_snapshot(_ctx(widgets=(w,)))
    assert "description: quarterly balance sheet" in out


def test_to_lc_messages_serialises_unserialisable_content_via_str() -> None:
    """Fall back to str(raw) when json.dumps fails."""
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    msg = ChatMessage(role="human", content=cyclic)
    out = _to_lc_messages([msg])
    assert len(out) == 1
    assert isinstance(out[0].content, str)
    assert out[0].content


def test_tool_message_entry_chunks_falls_back_to_str_on_unserialisable() -> None:
    """Fall back to str(entry) for an unserialisable entry."""
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    out = _tool_message_entry_chunks(cyclic)
    assert len(out) == 1
    assert isinstance(out[0], str)


from openbb_agent_server.runtime.builder import _resolve_tools  # noqa: E402


@pytest.mark.asyncio
async def test_resolve_tools_recognises_workspace_mcp_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag tools with the workspace-MCP prefix as client-side."""
    import types as _types

    from openbb_agent_server.runtime import registry
    from openbb_agent_server.runtime.plugins import ToolSource

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Source(ToolSource):
        name = "workspace_mcp"

        async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
            return [_Tool("mcp:open_widget")]

    real_load = registry.load
    monkeypatch.setattr(
        registry,
        "load",
        lambda group, name, *a, **kw: (
            _Source()
            if group == "openbb_agent_server.tools"
            else real_load(group, name, *a, **kw)
        ),
    )
    profile = _types.SimpleNamespace(
        tool_sources=("workspace_mcp",),
        tool_source_config={},
    )
    tools, client = await _resolve_tools(_ctx(), profile)
    assert "mcp:open_widget" in client
    assert len(tools) == 1


from openbb_agent_server.runtime.builder import (  # noqa: E402
    _load_system_prompt,
)


@pytest.mark.asyncio
async def test_resolve_tools_recognises_client_side_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag tools with the client: prefix as client-side."""
    import types as _types

    from openbb_agent_server.runtime import registry
    from openbb_agent_server.runtime.plugins import ToolSource

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Source(ToolSource):
        name = "client_side"

        async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
            return [_Tool("client:open_widget")]

    real_load = registry.load
    monkeypatch.setattr(
        registry,
        "load",
        lambda group, name, *a, **kw: (
            _Source()
            if group == "openbb_agent_server.tools"
            else real_load(group, name, *a, **kw)
        ),
    )
    profile = _types.SimpleNamespace(
        tool_sources=("client_side",),
        tool_source_config={},
    )
    tools, client = await _resolve_tools(_ctx(), profile)
    assert "client:open_widget" in client
    assert len(tools) == 1


def test_load_system_prompt_handles_format_spec(tmp_path: Any) -> None:
    """Substitute placeholders that carry a format spec."""
    path = tmp_path / "prompt.txt"
    path.write_text("today=[{today:>12}]\ntz={timezone}\n", encoding="utf-8")

    import types as _types

    profile = _types.SimpleNamespace(system_prompt_file=str(path))
    out = _load_system_prompt(_ctx(), profile)
    assert "today=[" in out
    assert "tz=UTC" in out


def test_load_system_prompt_preserves_unknown_placeholders_with_conversion(
    tmp_path: Any,
) -> None:
    """Preserve unknown placeholders along with conversion and spec."""
    path = tmp_path / "prompt.txt"
    path.write_text("known={timezone}\nunknown={mystery!r:>5}\n", encoding="utf-8")
    import types as _types

    profile = _types.SimpleNamespace(system_prompt_file=str(path))
    out = _load_system_prompt(_ctx(), profile)
    assert "known=UTC" in out
    assert "{mystery!r:>5}" in out


def test_render_widget_snapshot_detects_pdf_widget_via_columns() -> None:
    """Detect a PDF widget via its columns."""
    w = WidgetRef(
        uuid="u-pdf",
        widget_id="plain",
        origin="market",
        params={},
        data=None,
        name="Some Widget",
        columns=["title", "content"],
    )
    out = _render_widget_snapshot(_ctx(widgets=(w,)))
    assert "KIND=pdf-document" in out
    assert "expose PDF bytes" in out


def test_render_widget_snapshot_detects_pdf_widget_via_name_token() -> None:
    """Detect a PDF widget via a name token."""
    w = WidgetRef(
        uuid="u-doc",
        widget_id="x",
        origin="market",
        params={},
        data=None,
        name="Fund Prospectus",
    )
    out = _render_widget_snapshot(_ctx(widgets=(w,)))
    assert "KIND=pdf-document" in out


def test_render_widget_snapshot_marks_file_widget_unreadable() -> None:
    """Mark a file- widget as an unreadable user upload."""
    w = WidgetRef(
        uuid="u-f",
        widget_id="file-123",
        origin="upload",
        params={},
        data=None,
        name="My Upload",
    )
    out = _render_widget_snapshot(_ctx(widgets=(w,)))
    assert "UNREADABLE=true" in out
    assert "user uploads" in out


def test_render_widget_snapshot_annotates_in_store_widget() -> None:
    """Annotate an in-store widget with data_in_store=true."""
    w = WidgetRef(
        uuid="u-s",
        widget_id="prices",
        origin="market",
        params={},
        data=None,
        name="Prices",
    )
    out = _render_widget_snapshot(_ctx(widgets=(w,)), in_store=frozenset({"u-s"}))
    assert "data_in_store=true" in out
    assert "read_widget_data" in out


def test_render_widget_snapshot_shows_not_loaded_for_dataless_widget() -> None:
    """Show data=<not loaded> for a dataless widget."""
    w = WidgetRef(
        uuid="u-n",
        widget_id="prices",
        origin="market",
        params={},
        data=None,
        name="Prices",
    )
    out = _render_widget_snapshot(_ctx(widgets=(w,)))
    assert "data=<not loaded>" in out


def test_load_system_prompt_falls_back_when_file_unreadable(tmp_path: Any) -> None:
    """Fall back to the in-memory template for an unreadable prompt file."""
    import types as _types

    profile = _types.SimpleNamespace(
        system_prompt_file=str(tmp_path / "missing-prompt.txt")
    )
    out = _load_system_prompt(_ctx(), profile)
    assert "OpenBB Agent" in out


def test_to_lc_messages_ai_with_function_carries_tool_call() -> None:
    """Pair a tool_call onto an ai message that announced a function."""
    msg = ChatMessage(
        role="ai",
        content="dispatching",
        function="get_widget_data",
        input_arguments={"data_sources": []},
    )
    out = _to_lc_messages([msg])
    assert len(out) == 1
    assert isinstance(out[0], AIMessage)
    assert out[0].tool_calls[0]["name"] == "get_widget_data"


def test_to_lc_messages_rewrites_get_widget_data_tool_message() -> None:
    """Rewrite a get_widget_data tool message to point at the local store."""
    msgs = [
        ChatMessage(
            role="tool",
            function="get_widget_data",
            content="ignored",
            tool_call_id="c-1",
            input_arguments={
                "data_sources": [
                    {"widget_uuid": "w-tab", "id": "price_table"},
                    {"widget_uuid": "w-pdf", "id": "fund_prospectus"},
                    "not-a-dict",
                    {"id": ""},
                ]
            },
        )
    ]
    out = _to_lc_messages(msgs)
    from langchain_core.messages import ToolMessage

    tool_msg = next(m for m in out if isinstance(m, ToolMessage))
    assert "w-tab" in tool_msg.content
    assert "read_widget_data" in tool_msg.content
    assert "w-pdf" in tool_msg.content
    assert "list_pdfs" in tool_msg.content
    assert "Do NOT re-fetch" in tool_msg.content


def test_to_lc_messages_non_widget_tool_message_uses_data_payload() -> None:
    """Flatten the data payload of a non-get_widget_data tool message."""
    msgs = [
        ChatMessage(
            role="tool",
            function="web_search",
            content="fallback",
            data=[{"k": "v"}],
            tool_call_id="c-2",
        )
    ]
    out = _to_lc_messages(msgs)
    from langchain_core.messages import ToolMessage

    tool_msg = next(m for m in out if isinstance(m, ToolMessage))
    assert "k" in tool_msg.content


def test_resolve_subagents_attaches_model_when_set() -> None:
    import types as _types

    from openbb_agent_server.runtime import registry
    from openbb_agent_server.runtime.plugins import SubAgentSpec

    class _Stub(SubAgentSpec):
        name = "researcher"
        description = "stub"
        system_prompt = "be a researcher"
        tools = ()
        model: Any = "fake-model"

    profile = _types.SimpleNamespace(subagents=("researcher",))

    real_load = registry.load
    try:
        registry.load = (  # type: ignore[assignment]
            lambda group, name, *a, **kw: (
                _Stub()
                if group == "openbb_agent_server.subagents"
                else real_load(group, name, *a, **kw)
            )
        )
        out = _resolve_subagents(profile, main_tools=[])
    finally:
        registry.load = real_load  # type: ignore[assignment]

    assert out[0]["model"] == "fake-model"


from openbb_agent_server.runtime.builder import (  # noqa: E402
    _normalise_message_payload,
    _resolve_middleware,
    run_agent,
)


def test_normalise_message_payload_non_tuple_returns_none() -> None:
    """Normalise a non-tuple messages payload to None."""
    assert _normalise_message_payload((), "not-a-tuple") is None


def test_resolve_middleware_builds_each_named_middleware() -> None:
    """Load and build every middleware on the profile."""
    import types as _types

    from openbb_agent_server.runtime import registry
    from openbb_agent_server.runtime.plugins import Middleware

    built: list[Any] = []

    class _FakeMw:
        pass

    class _Factory(Middleware):
        name = "fake_mw"

        def build(self, ctx: RunContext, config: dict[str, Any]) -> Any:
            built.append(config)
            return _FakeMw()

    profile = _types.SimpleNamespace(middleware=("fake_mw",))
    real_load = registry.load
    try:
        registry.load = (  # type: ignore[assignment]
            lambda group, name, *a, **kw: (
                _Factory()
                if group == "openbb_agent_server.middleware"
                else real_load(group, name, *a, **kw)
            )
        )
        out = _resolve_middleware(_ctx(), profile, model="m")
    finally:
        registry.load = real_load  # type: ignore[assignment]

    assert len(out) == 1
    assert isinstance(out[0], _FakeMw)
    assert built[0]["model"] == "m"


class _FakeAgent:
    """Stand-in deepagents agent whose astream yields scripted tuples."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def astream(self, *_a: Any, **_kw: Any) -> Any:
        for ev in self._events:
            yield ev


def _install_fake_deepagents(
    monkeypatch: pytest.MonkeyPatch,
    events: list[Any],
    captured: dict[str, Any] | None = None,
) -> None:
    """Patch deepagents.create_deep_agent to return a scripted agent."""
    import sys
    import types as _types

    module = _types.ModuleType("deepagents")

    def _create(**kwargs: Any) -> _FakeAgent:
        if captured is not None:
            captured.update(kwargs)
        return _FakeAgent(events)

    module.create_deep_agent = _create  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepagents", module)


async def _run_agent_to_list(**kw: Any) -> list[Any]:
    out: list[Any] = []
    async for ev in run_agent(**kw):
        out.append(ev)
    return out


def _settings_for_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    from openbb_agent_server.app.settings import AgentServerSettings

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["ok"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    return AgentServerSettings()


@pytest.mark.asyncio
async def test_run_agent_streams_messages_and_custom_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stream messages and custom events through run_agent."""
    from langchain_core.messages import AIMessageChunk
    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.protocol.schemas import MessageChunkSSE, QueryRequest
    from openbb_agent_server.runtime import services as _services

    _services.set_services(checkpointer=InMemorySaver())
    settings = _settings_for_run(monkeypatch, tmp_path)

    events: list[Any] = [
        (
            "messages",
            (),
            (
                AIMessageChunk(
                    content="hi-chunk",
                    additional_kwargs={"reasoning_content": "thinking"},
                ),
                {},
            ),
        ),
        ("custom", (), {"type": "chunk", "content": "custom-log"}),
        ("messages", (), ("only-one-element",)),
        "not-a-tuple",
        ((), "messages", (AIMessageChunk(content="hello"), {})),
        ((), "messages", ("not-a-message", {})),
        ((), "updates", {"agent": {}}),
        ((), "custom", {"type": "chunk", "content": " world"}),
    ]
    _install_fake_deepagents(monkeypatch, events)

    body = QueryRequest(
        messages=[ChatMessage(role="human", content="hi")],
        conversation_id="c",
        run_id="r",
    )
    with caplog.at_level(TRACE, logger="openbb_agent_server.builder"):
        out = await _run_agent_to_list(ctx=_ctx(), body=body, settings=settings)
    assert any("llm chunk #" in r.getMessage() for r in caplog.records)
    assert any("astream raw #" in r.getMessage() for r in caplog.records)
    deltas = "".join(e.data.delta for e in out if isinstance(e, MessageChunkSSE))
    assert "world" in deltas
    blob = "".join(
        getattr(e.data, "delta", "") or getattr(e.data, "message", "") for e in out
    )
    assert "hello" in blob


@pytest.mark.asyncio
async def test_run_agent_astream_raises_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Propagate an exception raised inside agent.astream."""
    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.protocol.schemas import QueryRequest
    from openbb_agent_server.runtime import services as _services

    _services.set_services(checkpointer=InMemorySaver())
    settings = _settings_for_run(monkeypatch, tmp_path)

    class _BoomAgent:
        async def astream(self, *_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("astream blew up")
            yield  # noqa: F811 — keeps this an async generator

    import sys
    import types as _types

    module = _types.ModuleType("deepagents")
    module.create_deep_agent = lambda **_kw: _BoomAgent()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepagents", module)

    body = QueryRequest(
        messages=[ChatMessage(role="human", content="hi")],
        conversation_id="c",
        run_id="r",
    )
    with pytest.raises(RuntimeError, match="astream blew up"):
        await _run_agent_to_list(ctx=_ctx(), body=body, settings=settings)


@pytest.mark.asyncio
async def test_run_agent_in_store_hint_and_addendum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    widget_store_in_services: WidgetDataStore,
) -> None:
    """Flag in-store data and prepend the SQL-surface addendum."""
    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.protocol.schemas import QueryRequest

    captured: dict[str, Any] = {}
    _install_fake_deepagents(monkeypatch, [], captured)
    settings = _settings_for_run(monkeypatch, tmp_path)

    ctx = _ctx(
        widgets=(
            WidgetRef(
                uuid="w-1",
                widget_id="prices",
                origin="market",
                params={},
                data=None,
                name="Prices",
            ),
        )
    )
    await widget_store_in_services.record(
        principal=ctx.principal,
        conversation_id="prior",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="market",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    services.set_services(
        history=None,
        widget_store=widget_store_in_services,
        checkpointer=InMemorySaver(),
    )

    body = QueryRequest(
        messages=[ChatMessage(role="human", content="hi")],
        conversation_id="c",
        run_id="r",
    )
    await _run_agent_to_list(ctx=ctx, body=body, settings=settings)
    prompt = captured["system_prompt"]
    assert "data_in_store=true" in prompt
    assert "Local widget_data tables" in prompt


@pytest.mark.asyncio
async def test_run_agent_in_store_lookup_timeout_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    widget_store_in_services: WidgetDataStore,
) -> None:
    """Swallow a widget-store lookup timeout."""
    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.protocol.schemas import QueryRequest

    async def _timeout_list(**_kw: Any) -> Any:
        import asyncio as _asyncio

        raise _asyncio.TimeoutError

    monkeypatch.setattr(widget_store_in_services, "list_entries", _timeout_list)
    services.set_services(
        history=None,
        widget_store=widget_store_in_services,
        checkpointer=InMemorySaver(),
    )
    _install_fake_deepagents(monkeypatch, [])
    settings = _settings_for_run(monkeypatch, tmp_path)
    body = QueryRequest(
        messages=[ChatMessage(role="human", content="hi")],
        conversation_id="c",
        run_id="r",
    )
    out = await _run_agent_to_list(ctx=_ctx(), body=body, settings=settings)
    assert out == []


@pytest.mark.asyncio
async def test_run_agent_in_store_lookup_error_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    widget_store_in_services: WidgetDataStore,
) -> None:
    """Swallow an unexpected widget-store error."""
    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.protocol.schemas import QueryRequest

    async def _boom(**_kw: Any) -> Any:
        raise RuntimeError("store offline")

    monkeypatch.setattr(widget_store_in_services, "list_entries", _boom)
    services.set_services(
        history=None,
        widget_store=widget_store_in_services,
        checkpointer=InMemorySaver(),
    )
    _install_fake_deepagents(monkeypatch, [])
    settings = _settings_for_run(monkeypatch, tmp_path)
    body = QueryRequest(
        messages=[ChatMessage(role="human", content="hi")],
        conversation_id="c",
        run_id="r",
    )
    out = await _run_agent_to_list(ctx=_ctx(), body=body, settings=settings)
    assert out == []


@pytest.mark.asyncio
async def test_run_agent_narrows_tools_when_turn_has_tool_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Narrow the tool set when the turn has a tool message."""
    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.protocol.schemas import QueryRequest
    from openbb_agent_server.runtime import registry
    from openbb_agent_server.runtime.plugins import ToolSource

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Source(ToolSource):
        name = "fake_tools"

        async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
            return [_Tool("read_widget_data"), _Tool("some_other_tool")]

    real_load = registry.load
    monkeypatch.setattr(
        registry,
        "load",
        lambda group, name, *a, **kw: (
            _Source()
            if group == "openbb_agent_server.tools"
            else real_load(group, name, *a, **kw)
        ),
    )
    services.set_services(checkpointer=InMemorySaver())

    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["ok"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", '["fake_tools"]')
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    from openbb_agent_server.app.settings import AgentServerSettings

    settings = AgentServerSettings()

    captured: dict[str, Any] = {}
    _install_fake_deepagents(monkeypatch, [], captured)

    body = QueryRequest(
        messages=[
            ChatMessage(role="human", content="hi"),
            ChatMessage(
                role="tool",
                function="get_widget_data",
                content="rows",
                tool_call_id="c-1",
            ),
        ],
        conversation_id="c",
        run_id="r",
    )
    await _run_agent_to_list(ctx=_ctx(), body=body, settings=settings)
    kept = {getattr(t, "name", None) for t in captured["tools"]}
    assert "read_widget_data" in kept
    assert "some_other_tool" not in kept


@pytest.mark.asyncio
async def test_run_agent_passes_skills_when_profile_has_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Forward profile skills to create_deep_agent."""
    from langgraph.checkpoint.memory import InMemorySaver

    from openbb_agent_server.protocol.schemas import QueryRequest
    from openbb_agent_server.runtime import services as _services

    _services.set_services(checkpointer=InMemorySaver())
    monkeypatch.setenv("OPENBB_AGENT_AUTH_BACKEND", "none")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBB_AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("OPENBB_AGENT_FAKE_RESPONSES", '["ok"]')
    monkeypatch.setenv("OPENBB_AGENT_TOOL_SOURCES", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SUBAGENTS", "[]")
    monkeypatch.setenv("OPENBB_AGENT_MIDDLEWARE", "[]")
    monkeypatch.setenv("OPENBB_AGENT_SKILLS", '["/skills/finance"]')
    from openbb_agent_server.app.settings import AgentServerSettings

    settings = AgentServerSettings()

    captured: dict[str, Any] = {}
    _install_fake_deepagents(monkeypatch, [], captured)

    body = QueryRequest(
        messages=[ChatMessage(role="human", content="hi")],
        conversation_id="c",
        run_id="r",
    )
    await _run_agent_to_list(ctx=_ctx(), body=body, settings=settings, profile=None)
    assert captured["skills"] == ["/skills/finance"]


def test_system_prompt_never_contains_user_identity(tmp_path) -> None:
    """Keep user identity out of the system prompt."""
    from openbb_agent_server.app.settings import AgentMetadata, AgentProfile
    from openbb_agent_server.runtime.builder import (
        _build_system_prompt,
        _load_system_prompt,
    )

    sensitive_email = "alice@confidential.example.com"
    ctx = RunContext(
        principal=UserPrincipal(
            user_id=sensitive_email,
            display_name="Alice Confidential",
            email=sensitive_email,
        ),
        trace_id="trace-123",
        run_id="r",
        conversation_id="c",
    )
    out = _build_system_prompt(ctx)
    assert sensitive_email not in out
    assert "Alice Confidential" not in out
    assert "trace-123" not in out

    profile = AgentProfile(
        name="default",
        metadata=AgentMetadata(),
        model_provider="fake",
        model_name="x",
        model_config={},
        tool_sources=(),
        subagents=(),
        middleware=(),
        skills=(),
        features={},
        system_prompt_file=None,
        tool_source_config={},
    )
    out = _load_system_prompt(ctx, profile)
    assert sensitive_email not in out
    assert "Alice Confidential" not in out
    assert "trace-123" not in out

    custom = tmp_path / "custom.md"
    custom.write_text("Custom prompt {user_id} {display_name} {trace_id}.")
    profile_custom = profile.model_copy(update={"system_prompt_file": str(custom)})
    out = _load_system_prompt(ctx, profile_custom)
    assert sensitive_email not in out
    assert "Alice Confidential" not in out
    assert "trace-123" not in out
