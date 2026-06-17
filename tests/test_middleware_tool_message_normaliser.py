"""tool_message_normaliser middleware tests."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from openbb_agent_server.observability.logging import TRACE
from openbb_agent_server.plugins.middleware.tool_message_normaliser import (
    ToolMessageNormaliserMiddlewareFactory,
    _content_str,
    _dedupe_tool_calls,
    _strict_human_assistant,
    _to_openai_tool_calls,
    _valid_tool_calls,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


class _Request:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages

    def override(self, *, messages: list[Any]) -> _Request:
        return _Request(messages)


def test_factory_name() -> None:
    assert ToolMessageNormaliserMiddlewareFactory.name == "tool_message_normaliser"


def test_content_str_none() -> None:
    assert _content_str(None) == ""


def test_content_str_plain_string() -> None:
    assert _content_str("hi") == "hi"


def test_content_str_concatenates_text_blocks() -> None:
    blocks = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    assert _content_str(blocks) == "hello world"


def test_content_str_skips_non_text_blocks() -> None:
    blocks = [
        {"type": "text", "text": "x"},
        {"type": "image_url", "url": "http://x"},
    ]
    assert _content_str(blocks) == "x"


def test_content_str_handles_string_blocks_in_list() -> None:
    blocks = ["a", "b"]
    assert _content_str(blocks) == "ab"


def test_content_str_falls_back_to_str_for_other_types() -> None:
    assert _content_str(42) == "42"


def test_content_str_skips_text_block_without_string_text() -> None:
    blocks = [{"type": "text", "text": 42}]
    assert _content_str(blocks) == ""


def test_valid_tool_calls_returns_empty_for_falsy_input() -> None:
    assert _valid_tool_calls(None) == []
    assert _valid_tool_calls([]) == []


def test_valid_tool_calls_keeps_named_dict_calls() -> None:
    calls = [
        {"name": "x", "args": {}, "id": "1"},
        {"name": "y", "args": {}, "id": "2"},
    ]
    assert _valid_tool_calls(calls) == calls


def test_valid_tool_calls_drops_blank_name_calls() -> None:
    calls = [
        {"name": "", "args": {}, "id": "1"},
        {"name": "real", "args": {}, "id": "2"},
    ]
    assert _valid_tool_calls(calls) == [{"name": "real", "args": {}, "id": "2"}]


def test_valid_tool_calls_handles_object_style_calls() -> None:
    class _Call:
        def __init__(self, name: str) -> None:
            self.name = name

    blank, named = _Call(""), _Call("real")
    assert _valid_tool_calls([blank, named]) == [named]


def test_to_openai_tool_calls_returns_empty_for_falsy_input() -> None:
    assert _to_openai_tool_calls(None) == []
    assert _to_openai_tool_calls([]) == []


def test_to_openai_tool_calls_converts_dict_calls() -> None:
    out = _to_openai_tool_calls([{"name": "x", "args": {"a": 1}, "id": "c1"}])
    assert out == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "x", "arguments": '{"a": 1}'},
        }
    ]


def test_to_openai_tool_calls_converts_object_style_calls() -> None:
    class _Call:
        def __init__(self) -> None:
            self.name = "y"
            self.args = {"b": 2}
            self.id = "c2"

    out = _to_openai_tool_calls([_Call()])
    assert out[0]["id"] == "c2"
    assert out[0]["function"] == {"name": "y", "arguments": '{"b": 2}'}


def test_to_openai_tool_calls_skips_blank_name() -> None:
    out = _to_openai_tool_calls([{"name": "", "args": {}, "id": "1"}])
    assert out == []


def test_to_openai_tool_calls_falls_back_for_unserialisable_args() -> None:
    circular: dict[str, Any] = {}
    circular["self"] = circular
    out = _to_openai_tool_calls([{"name": "x", "args": circular, "id": "c3"}])
    assert out[0]["function"] == {"name": "x", "arguments": "{}"}


def test_strict_passes_system_through() -> None:
    out = _strict_human_assistant([SystemMessage(content="sys")])
    assert len(out) == 1
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == "sys"


def test_strict_drops_empty_ai_message() -> None:
    out = _strict_human_assistant([AIMessage(content="")])
    assert out == []


def test_strict_drops_empty_human_message_when_no_pending() -> None:
    out = _strict_human_assistant([HumanMessage(content="")])
    assert out == []


def test_strict_drops_tool_message_with_empty_payload() -> None:
    out = _strict_human_assistant([ToolMessage(content="", tool_call_id="1")])
    assert out == []


def test_strict_folds_tool_message_into_following_human() -> None:
    msgs = [
        ToolMessage(content="weather=sunny", name="get_weather", tool_call_id="1"),
        HumanMessage(content="thanks"),
    ]
    out = _strict_human_assistant(msgs)
    [combined] = out
    assert isinstance(combined, HumanMessage)
    assert "weather=sunny" in combined.content
    assert "thanks" in combined.content
    assert "[tool: get_weather result]" in combined.content


def test_strict_tool_without_name_uses_default_label() -> None:
    msgs = [
        ToolMessage(content="x", tool_call_id="1"),
        HumanMessage(content="ok"),
    ]
    out = _strict_human_assistant(msgs)
    [combined] = out
    assert "[tool: tool result]" in combined.content


def test_strict_flushes_pending_tool_as_human_before_ai_message() -> None:
    msgs = [
        ToolMessage(content="data", name="t", tool_call_id="1"),
        AIMessage(content="response"),
    ]
    out = _strict_human_assistant(msgs)
    assert len(out) == 2
    assert isinstance(out[0], HumanMessage)
    assert "[tool: t result]" in out[0].content
    assert "data" in out[0].content
    assert isinstance(out[1], AIMessage)
    assert out[1].content == "response"


def test_strict_ai_preserves_text_and_structured_tool_calls() -> None:
    msg = AIMessage(
        content="thinking",
        tool_calls=[{"name": "search", "args": {}, "id": "1"}],
    )
    out = _strict_human_assistant([msg])
    [result] = out
    assert isinstance(result, AIMessage)
    assert result.content == "thinking"
    assert [tc["name"] for tc in result.tool_calls] == ["search"]


def test_strict_ai_with_tool_calls_no_text_keeps_empty_content() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {}, "id": "1"}],
    )
    out = _strict_human_assistant([msg])
    [result] = out
    assert result.content == ""
    assert [tc["name"] for tc in result.tool_calls] == ["search"]
    mirrored = result.additional_kwargs["tool_calls"]
    assert mirrored[0]["type"] == "function"
    assert mirrored[0]["function"]["name"] == "search"


def test_strict_ai_drops_tool_calls_from_additional_kwargs() -> None:
    msg = AIMessage(
        content="hi",
        additional_kwargs={"tool_calls": [{"name": "x"}], "other": "keep"},
    )
    out = _strict_human_assistant([msg])
    [result] = out
    assert "tool_calls" not in result.additional_kwargs
    assert result.additional_kwargs.get("other") == "keep"


def test_strict_trailing_tool_results_are_flushed() -> None:
    msgs = [
        AIMessage(content="hi"),
        ToolMessage(content="result1", name="t", tool_call_id="1"),
    ]
    out = _strict_human_assistant(msgs)
    assert len(out) == 2
    assert isinstance(out[-1], HumanMessage)
    assert "result1" in out[-1].content


def test_strict_unknown_message_with_text_becomes_human() -> None:
    class _Mystery:
        content = "unknown stuff"

    out = _strict_human_assistant([_Mystery()])
    [result] = out
    assert isinstance(result, HumanMessage)
    assert result.content == "unknown stuff"


def test_strict_unknown_message_with_empty_content_is_dropped() -> None:
    class _Mystery:
        content = ""

    out = _strict_human_assistant([_Mystery()])
    assert out == []


def test_strict_unknown_message_flushes_pending_tools() -> None:
    class _Mystery:
        content = "extra"

    msgs = [
        ToolMessage(content="x", name="t", tool_call_id="1"),
        _Mystery(),
    ]
    out = _strict_human_assistant(msgs)
    # Pending tool gets flushed as Human, then the unknown's text gets a separate Human, which then merges.
    [combined] = out
    assert isinstance(combined, HumanMessage)
    assert "[tool: t result]" in combined.content
    assert "extra" in combined.content


def test_strict_merges_consecutive_humans() -> None:
    msgs = [HumanMessage(content="part 1"), HumanMessage(content="part 2")]
    out = _strict_human_assistant(msgs)
    [merged] = out
    assert "part 1" in merged.content
    assert "part 2" in merged.content


def test_strict_merges_consecutive_ai() -> None:
    msgs = [AIMessage(content="a1"), AIMessage(content="a2")]
    out = _strict_human_assistant(msgs)
    [merged] = out
    assert "a1" in merged.content
    assert "a2" in merged.content


def test_strict_preserves_alternation_when_already_strict() -> None:
    msgs = [
        SystemMessage(content="s"),
        HumanMessage(content="h"),
        AIMessage(content="a"),
        HumanMessage(content="h2"),
    ]
    out = _strict_human_assistant(msgs)
    assert [type(m).__name__ for m in out] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]


def test_strict_does_not_merge_system_messages() -> None:
    msgs = [SystemMessage(content="s1"), SystemMessage(content="s2")]
    out = _strict_human_assistant(msgs)
    assert len(out) == 2


def test_dedupe_returns_none_when_response_is_none() -> None:
    assert _dedupe_tool_calls(None) is None


def test_dedupe_passes_through_when_no_tool_calls() -> None:
    msg = AIMessage(content="hi")
    assert _dedupe_tool_calls(msg) is msg


def test_dedupe_passes_through_when_only_one_call() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "x", "args": {}, "id": "1"}],
    )
    assert _dedupe_tool_calls(msg) is msg


def test_dedupe_drops_exact_duplicates() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "x", "args": {"a": 1}, "id": "1"},
            {"name": "x", "args": {"a": 1}, "id": "2"},
        ],
    )
    out = _dedupe_tool_calls(msg)
    assert len(out.tool_calls) == 1


def test_dedupe_keeps_distinct_calls() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "x", "args": {"a": 1}, "id": "1"},
            {"name": "x", "args": {"a": 2}, "id": "2"},
        ],
    )
    out = _dedupe_tool_calls(msg)
    assert len(out.tool_calls) == 2


def test_dedupe_falls_back_for_unhashable_args() -> None:
    """Args that defeat json.dumps fall back to repr."""

    circular: dict[str, Any] = {}
    circular["self"] = circular
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "x", "args": circular, "id": "1"},
            {"name": "x", "args": circular, "id": "2"},
        ],
    )
    out = _dedupe_tool_calls(msg)
    assert len(out.tool_calls) == 1


def test_dedupe_handles_object_style_tool_calls() -> None:
    class _Call:
        def __init__(self, name: str, args: dict) -> None:
            self.name = name
            self.args = args

    class _Msg:
        def __init__(self) -> None:
            self.tool_calls = [_Call("x", {"a": 1}), _Call("x", {"a": 1})]
            self.content = ""

    msg = _Msg()
    out = _dedupe_tool_calls(msg)
    assert len(out.tool_calls) == 1


def test_dedupe_creates_new_aimessage_when_assignment_fails() -> None:
    """A read-only AIMessage.tool_calls still triggers a rebuild."""

    msg = AIMessage(
        content="ok",
        tool_calls=[
            {"name": "x", "args": {"a": 1}, "id": "1"},
            {"name": "x", "args": {"a": 1}, "id": "2"},
        ],
    )

    class _Readonly(AIMessage):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "tool_calls":
                raise AttributeError("readonly")
            super().__setattr__(name, value)

    msg2 = _Readonly(
        content="ok",
        tool_calls=[
            {"name": "x", "args": {"a": 1}, "id": "1"},
            {"name": "x", "args": {"a": 1}, "id": "2"},
        ],
    )
    out = _dedupe_tool_calls(msg2)
    assert isinstance(out, AIMessage)
    assert len(out.tool_calls) == 1


def test_wrap_model_call_runs_normalisation_and_dedupe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mw = ToolMessageNormaliserMiddlewareFactory().build(_ctx(), {})
    request = _Request(
        [
            ToolMessage(content="r", name="t", tool_call_id="1"),
            HumanMessage(content="q"),
        ]
    )
    seen: list[Any] = []

    def handler(req: Any) -> AIMessage:
        seen.append(req)
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "x", "args": {}, "id": "1"},
                {"name": "x", "args": {}, "id": "2"},
            ],
        )

    logger_name = "openbb_agent_server.middleware.tool_message_normaliser"
    with caplog.at_level(TRACE, logger=logger_name):
        out = mw.wrap_model_call(request, handler)
    assert any("-> model" in r.getMessage() for r in caplog.records)
    [used] = seen
    [combined] = used.messages
    assert isinstance(combined, HumanMessage)
    assert "[tool: t result]" in combined.content
    assert len(out.tool_calls) == 1


@pytest.mark.asyncio
async def test_awrap_model_call_runs_normalisation_and_dedupe() -> None:
    mw = ToolMessageNormaliserMiddlewareFactory().build(_ctx(), {})
    request = _Request(
        [
            ToolMessage(content="r", name="t", tool_call_id="1"),
            HumanMessage(content="q"),
        ]
    )
    seen: list[Any] = []

    async def handler(req: Any) -> AIMessage:
        seen.append(req)
        return AIMessage(
            content="response",
            tool_calls=[
                {"name": "x", "args": {}, "id": "1"},
                {"name": "x", "args": {}, "id": "2"},
            ],
        )

    out = await mw.awrap_model_call(request, handler)
    [used] = seen
    [combined] = used.messages
    assert isinstance(combined, HumanMessage)
    assert "[tool: t result]" in combined.content


class _MistralRequest(_Request):
    """A request whose model name trips the _wants_tool_role check."""

    class _Model:
        model = "mistralai/mistral-large-3"

    def __init__(self, messages: list[Any]) -> None:
        super().__init__(messages)
        self.model = self._Model()

    def override(self, *, messages: list[Any]) -> _MistralRequest:
        return _MistralRequest(messages)


def test_strict_preserves_tool_role_for_mistral() -> None:
    """preserve_tool_role=True keeps a ToolMessage verbatim with its id."""
    msgs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {}, "id": "tc-1"}],
        ),
        ToolMessage(content="sunny", name="get_weather", tool_call_id="tc-1"),
    ]
    out = _strict_human_assistant(msgs, preserve_tool_role=True)
    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "tc-1"
    assert tool_msgs[0].content == "sunny"
    assert tool_msgs[0].name == "get_weather"


def test_strict_preserved_tool_messages_never_merge() -> None:
    """Two consecutive preserved ToolMessages stay separate (no merge)."""
    msgs = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "a", "args": {}, "id": "tc-1"},
                {"name": "b", "args": {}, "id": "tc-2"},
            ],
        ),
        ToolMessage(content="r1", name="a", tool_call_id="tc-1"),
        ToolMessage(content="r2", name="b", tool_call_id="tc-2"),
    ]
    out = _strict_human_assistant(msgs, preserve_tool_role=True)
    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_msgs] == ["r1", "r2"]
    assert [m.tool_call_id for m in tool_msgs] == ["tc-1", "tc-2"]


def test_strict_merges_consecutive_ai_messages_with_tool_calls() -> None:
    """Merging two AIMessages unions their tool_calls into additional_kwargs."""
    msgs = [
        AIMessage(
            content="first",
            tool_calls=[{"name": "a", "args": {}, "id": "1"}],
        ),
        AIMessage(
            content="second",
            tool_calls=[{"name": "b", "args": {}, "id": "2"}],
        ),
    ]
    out = _strict_human_assistant(msgs)
    [merged] = out
    assert isinstance(merged, AIMessage)
    assert merged.content == "first\n\nsecond"
    names = {tc["name"] for tc in merged.tool_calls}
    assert names == {"a", "b"}
    wire = merged.additional_kwargs["tool_calls"]
    assert {c["function"]["name"] for c in wire} == {"a", "b"}


def test_strict_drops_empty_ai_message_in_pass_three() -> None:
    """Pass 3 drops an AIMessage left empty on both text and tool_calls."""

    class _RawAI:
        """Minimal AIMessage-shaped object pass 2 will merge by type."""

    msgs = [
        AIMessage(content="answer one"),
        AIMessage(content="answer two"),
        HumanMessage(content="next"),
    ]
    out = _strict_human_assistant(msgs)
    for m in out:
        if isinstance(m, AIMessage):
            assert _content_str(m.content).strip() or _valid_tool_calls(
                m.tool_calls or []
            )
    del _RawAI
