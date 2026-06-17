"""Tests for the PyWry ACP shim (skipped when pywry is not installed)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pywry.chat")

from pywry.chat.artifacts import (  # noqa: E402
    CodeArtifact,
    HtmlArtifact,
    MarkdownArtifact,
    TableArtifact,
)
from pywry.chat.models import ImagePart, TextPart  # noqa: E402
from pywry.chat.session import ClientCapabilities  # noqa: E402
from pywry.chat.updates import (  # noqa: E402
    AgentMessageUpdate,
    ArtifactUpdate,
    CitationUpdate,
    ModeUpdate,
    StatusUpdate,
    ThinkingUpdate,
)

from openbb_agent_server.acp import OpenBBAgentProvider, translate_sse  # noqa: E402
from openbb_agent_server.acp.provider import _content_blocks_to_turn  # noqa: E402
from openbb_agent_server.app.settings import AgentServerSettings  # noqa: E402
from openbb_agent_server.protocol.schemas import (  # noqa: E402
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    ClientArtifact,
    FunctionCallSSE,
    FunctionCallSSEData,
    MessageArtifactSSE,
    MessageChunkSSE,
    MessageChunkSSEData,
    SourceInfo,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)

# ---------------------------------------------------------------------------
# translate_sse — wire SSE → ACP SessionUpdate
# ---------------------------------------------------------------------------


def test_translate_message_chunk() -> None:
    out = list(translate_sse(MessageChunkSSE(data=MessageChunkSSEData(delta="Hi"))))
    assert len(out) == 1
    assert isinstance(out[0], AgentMessageUpdate)
    assert out[0].text == "Hi"


def test_translate_empty_chunk_dropped() -> None:
    assert (
        list(translate_sse(MessageChunkSSE(data=MessageChunkSSEData(delta="")))) == []
    )


def test_translate_info_status_becomes_thinking() -> None:
    ev = StatusUpdateSSE(
        data=StatusUpdateSSEData(eventType="INFO", message="Calling tool: web_search")
    )
    out = list(translate_sse(ev))
    assert len(out) == 1
    assert isinstance(out[0], ThinkingUpdate)
    assert "web_search" in out[0].text


def test_translate_error_status_stays_status() -> None:
    ev = StatusUpdateSSE(
        data=StatusUpdateSSEData(eventType="ERROR", message="tool blew up")
    )
    out = list(translate_sse(ev))
    assert len(out) == 1
    assert isinstance(out[0], StatusUpdate)
    assert "ERROR" in out[0].text


def test_translate_hidden_status_dropped() -> None:
    ev = StatusUpdateSSE(
        data=StatusUpdateSSEData(eventType="SUCCESS", message="done", hidden=True)
    )
    assert list(translate_sse(ev)) == []


@pytest.mark.parametrize(
    ("artifact_type", "content", "expected_cls"),
    [
        ("text", "# Title", MarkdownArtifact),
        ("html", "<b>x</b>", HtmlArtifact),
        ("table", [{"a": 1}], TableArtifact),
        ("chart", [{"x": 1, "y": 2}], TableArtifact),
        ("snowflake_query", "SELECT 1", CodeArtifact),
        ("snowflake_python", "print(1)", CodeArtifact),
    ],
)
def test_translate_artifacts_map_to_pywry_classes(
    artifact_type: str, content, expected_cls: type
) -> None:
    ev = MessageArtifactSSE(
        data=ClientArtifact(
            type=artifact_type,
            name="a",
            description="d",
            uuid="u-1",
            content=content,
        )
    )
    out = list(translate_sse(ev))
    assert len(out) == 1
    assert isinstance(out[0], ArtifactUpdate)
    assert isinstance(out[0].artifact, expected_cls)


def test_translate_sql_artifact_language() -> None:
    ev = MessageArtifactSSE(
        data=ClientArtifact(
            type="snowflake_query",
            name="q",
            description="d",
            uuid="u-2",
            content="SELECT 1",
        )
    )
    (update,) = translate_sse(ev)
    assert update.artifact.language == "sql"
    assert update.artifact.content == "SELECT 1"


def test_translate_citations() -> None:
    ev = CitationCollectionSSE(
        data=CitationCollection(
            citations=[
                Citation(
                    id="c1",
                    source_info=SourceInfo(
                        type="web",
                        name="Example",
                        description="An example source",
                        metadata={"url": "https://example.com"},
                    ),
                )
            ]
        )
    )
    out = list(translate_sse(ev))
    assert len(out) == 1
    assert isinstance(out[0], CitationUpdate)
    assert out[0].url == "https://example.com"
    assert out[0].title == "Example"
    assert out[0].snippet == "An example source"


def test_translate_function_call_announces_unavailable() -> None:
    ev = FunctionCallSSE(
        data=FunctionCallSSEData(function="get_widget_data", input_arguments={})
    )
    out = list(translate_sse(ev))
    assert len(out) == 1
    assert isinstance(out[0], StatusUpdate)
    assert "get_widget_data" in out[0].text
    assert "Workspace" in out[0].text


# ---------------------------------------------------------------------------
# Content block conversion
# ---------------------------------------------------------------------------


def test_content_blocks_text_and_images() -> None:
    text, files = _content_blocks_to_turn(
        [
            TextPart(text="What is this chart?"),
            ImagePart(data="aGk=", mimeType="image/jpeg"),
            TextPart(text="Be brief."),
        ]
    )
    assert text == "What is this chart?\n\nBe brief."
    assert len(files) == 1
    assert files[0].mime == "image/jpeg"
    assert files[0].data_base64 == "aGk="
    assert files[0].name.endswith(".jpeg")


# ---------------------------------------------------------------------------
# Provider lifecycle against the embedded runtime (fake model)
# ---------------------------------------------------------------------------


async def test_provider_full_lifecycle(
    settings_env: AgentServerSettings,
) -> None:
    provider = OpenBBAgentProvider(settings_env, user_id="test-user")
    try:
        caps = await provider.initialize(ClientCapabilities())
        assert caps.prompt_capabilities is not None
        assert caps.prompt_capabilities.image is True
        # The built-in settings ship multiple profiles → modes advertised.
        assert caps.modes is True

        session_id = await provider.new_session(cwd=".")
        updates = [
            u async for u in provider.prompt(session_id, [TextPart(text="Hello")])
        ]
        # First prompt of the session announces the profile catalogue.
        assert isinstance(updates[0], ModeUpdate)
        assert updates[0].current_mode_id == "default"
        assert {m.id for m in updates[0].available_modes} == set(
            settings_env.all_profile_names()
        )
        text = "".join(u.text for u in updates if isinstance(u, AgentMessageUpdate))
        assert text == "OK."
    finally:
        await provider.runtime.aclose()


async def test_provider_tracks_conversation_history(
    settings_env: AgentServerSettings,
) -> None:
    provider = OpenBBAgentProvider(settings_env, user_id="test-user")
    try:
        await provider.initialize(ClientCapabilities())
        session_id = await provider.new_session(cwd=".")
        async for _ in provider.prompt(session_id, [TextPart(text="Hello")]):
            pass
        session = provider._sessions[session_id]
        assert [(m.role, m.content) for m in session.messages] == [
            ("human", "Hello"),
            ("ai", "OK."),
        ]
    finally:
        await provider.runtime.aclose()


async def test_provider_cancel_short_circuits(
    settings_env: AgentServerSettings,
) -> None:
    provider = OpenBBAgentProvider(settings_env, user_id="test-user")
    try:
        await provider.initialize(ClientCapabilities())
        session_id = await provider.new_session(cwd=".")
        cancel = asyncio.Event()
        cancel.set()
        updates = [
            u
            async for u in provider.prompt(
                session_id, [TextPart(text="Hello")], cancel_event=cancel
            )
        ]
        assert not any(isinstance(u, AgentMessageUpdate) for u in updates)
    finally:
        await provider.runtime.aclose()


async def test_provider_unknown_session_raises(
    settings_env: AgentServerSettings,
) -> None:
    provider = OpenBBAgentProvider(settings_env)
    try:
        with pytest.raises(ValueError, match="unknown session"):
            async for _ in provider.prompt("nope", [TextPart(text="x")]):
                pass
        await provider.cancel("nope")  # unknown session: silent no-op
        with pytest.raises(ValueError, match="unknown session"):
            await provider.set_mode("nope", "default")
    finally:
        await provider.runtime.aclose()


async def test_provider_set_mode_validates_profile(
    settings_env: AgentServerSettings,
) -> None:
    provider = OpenBBAgentProvider(settings_env)
    try:
        session_id = await provider.new_session(cwd=".")
        await provider.set_mode(session_id, "default")
        with pytest.raises(KeyError):
            await provider.set_mode(session_id, "not-a-profile")
    finally:
        await provider.runtime.aclose()


def test_provider_rejects_unknown_default_profile(
    settings_env: AgentServerSettings,
) -> None:
    with pytest.raises(KeyError):
        OpenBBAgentProvider(settings_env, profile="not-a-profile")


def test_provider_requires_settings_or_runtime() -> None:
    with pytest.raises(ValueError, match="settings or a runtime"):
        OpenBBAgentProvider()


def test_translate_chart_artifact_with_non_list_content() -> None:
    from pywry.chat.artifacts import JsonArtifact

    ev = MessageArtifactSSE(
        data=ClientArtifact(
            type="chart",
            name="c",
            description="d",
            uuid="u-3",
            content="not-rows",
        )
    )
    (update,) = translate_sse(ev)
    assert isinstance(update.artifact, JsonArtifact)
    assert update.artifact.data == {"name": "c", "content": "not-rows"}


def test_translate_status_with_attached_artifacts() -> None:
    ev = StatusUpdateSSE(
        data=StatusUpdateSSEData(
            eventType="SUCCESS",
            message="built a table",
            artifacts=[
                ClientArtifact(
                    type="table",
                    name="t",
                    description="d",
                    uuid="u-4",
                    content=[{"a": 1}],
                )
            ],
        )
    )
    out = list(translate_sse(ev))
    assert isinstance(out[0], ThinkingUpdate)
    assert isinstance(out[1], ArtifactUpdate)
    assert isinstance(out[1].artifact, TableArtifact)


def test_content_blocks_ignore_unsupported_types() -> None:
    from pywry.chat.models import AudioPart

    text, files = _content_blocks_to_turn(
        [TextPart(text="hi"), AudioPart(data="aGk=", mimeType="audio/wav")]
    )
    assert text == "hi"
    assert files == []


def test_provider_from_toml_runs_cascade(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openbb_agent_server.app import config as cfg_mod

    cfg = tmp_path / "acp.toml"
    cfg.write_text(
        '[agent]\nmodel_provider = "fake"\ncheckpointer_provider = "inmemory"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", tmp_path / "no_user_dir")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))
    provider = OpenBBAgentProvider.from_toml(str(cfg), user_id="from-toml-user")
    assert provider.runtime.settings.checkpointer_provider == "inmemory"


async def test_new_session_ignores_client_mcp_servers(
    settings_env: AgentServerSettings,
) -> None:
    provider = OpenBBAgentProvider(settings_env)
    try:
        session_id = await provider.new_session(
            cwd=".",
            mcp_servers=[{"name": "x", "transport": "streamable_http"}],
        )
        assert session_id in provider._sessions
    finally:
        await provider.runtime.aclose()


async def test_prompt_surfaces_run_failure_as_error_status(
    settings_env: AgentServerSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenBBAgentProvider(settings_env)
    try:
        session_id = await provider.new_session(cwd=".")

        async def _boom(**_kwargs):
            raise RuntimeError("loop exploded")
            yield  # pragma: no cover — makes this an async generator

        monkeypatch.setattr(provider._runtime, "run_turn", _boom)
        updates = [u async for u in provider.prompt(session_id, [TextPart(text="x")])]
        errors = [u for u in updates if isinstance(u, StatusUpdate)]
        assert errors
        assert "loop exploded" in errors[-1].text
        # No assistant reply was appended after the failure.
        assert [m.role for m in provider._sessions[session_id].messages] == ["human"]
    finally:
        await provider.runtime.aclose()


async def test_prompt_propagates_cancelled_error(
    settings_env: AgentServerSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenBBAgentProvider(settings_env)
    try:
        session_id = await provider.new_session(cwd=".")

        async def _cancelled(**_kwargs):
            raise asyncio.CancelledError
            yield  # pragma: no cover — makes this an async generator

        monkeypatch.setattr(provider._runtime, "run_turn", _cancelled)
        with pytest.raises(asyncio.CancelledError):
            async for _ in provider.prompt(session_id, [TextPart(text="x")]):
                pass
        # The cancel handle was cleared by the finally block.
        assert provider._sessions[session_id].cancel_event is None
    finally:
        await provider.runtime.aclose()


async def test_cancel_sets_active_turn_event(
    settings_env: AgentServerSettings,
) -> None:
    provider = OpenBBAgentProvider(settings_env)
    try:
        session_id = await provider.new_session(cwd=".")
        event = asyncio.Event()
        provider._sessions[session_id].cancel_event = event
        await provider.cancel(session_id)
        assert event.is_set()
    finally:
        await provider.runtime.aclose()


def test_create_chat_manager_with_settings(
    settings_env: AgentServerSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.acp.provider import create_chat_manager

    captured: dict = {}

    class _FakeManager:
        def __init__(self, provider=None, **kwargs):
            captured["provider"] = provider
            captured["kwargs"] = kwargs

    monkeypatch.setattr("pywry.chat.manager.ChatManager", _FakeManager)
    manager = create_chat_manager(settings=settings_env, user_id="cm-user")
    assert isinstance(manager, _FakeManager)
    assert isinstance(captured["provider"], OpenBBAgentProvider)
    # Welcome message defaulted from the agent metadata description.
    assert "welcome_message" in captured["kwargs"]


def test_create_chat_manager_from_cascade(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.acp.provider import create_chat_manager
    from openbb_agent_server.app import config as cfg_mod

    cfg = tmp_path / "cm.toml"
    cfg.write_text(
        '[agent]\nmodel_provider = "fake"\ncheckpointer_provider = "inmemory"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg_mod, "USER_OPENBB_DIR", tmp_path / "no_user_dir")
    monkeypatch.setenv("OPENBB_AGENT_DATA_DIR", str(tmp_path))

    class _FakeManager:
        def __init__(self, provider=None, **kwargs):
            self.provider = provider

    monkeypatch.setattr("pywry.chat.manager.ChatManager", _FakeManager)
    manager = create_chat_manager(str(cfg))
    assert isinstance(manager.provider, OpenBBAgentProvider)
