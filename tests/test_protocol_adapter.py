"""DeepAgentEventAdapter translation tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from openbb_agent_server.protocol.adapter import (
    CLIENT_SIDE_TOOL_PREFIX,
    DeepAgentEventAdapter,
)
from openbb_agent_server.protocol.schemas import (
    CitationCollectionSSE,
    FunctionCallSSE,
    MessageArtifactSSE,
    MessageChunkSSE,
    StatusUpdateSSE,
)


async def _drive(
    adapter: DeepAgentEventAdapter, raw: list[dict[str, Any]]
) -> list[Any]:
    async def gen() -> AsyncIterator[dict[str, Any]]:
        for r in raw:
            yield r

    out: list[Any] = []
    async for ev in adapter.adapt(gen()):
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_natural_completion_does_not_emit_turn_complete_marker() -> None:
    """The adapter emits no trailing Turn complete StatusUpdateSSE."""
    adapter = DeepAgentEventAdapter()

    async def gen() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "messages", "ns": [], "data": {"message": {"content": "hi"}}}

    out: list[Any] = []
    async for ev in adapter.adapt(gen()):
        out.append(ev)
    assert not any(
        isinstance(ev, StatusUpdateSSE)
        and getattr(ev.data, "message", None) == "Turn complete."
        for ev in out
    )


@pytest.mark.asyncio
async def test_text_chunk_becomes_message_chunk_sse() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "messages", "ns": [], "data": {"message": {"content": "hi"}}}],
    )
    assert len(out) == 1
    assert isinstance(out[0], MessageChunkSSE)
    assert out[0].data.delta == "hi"


@pytest.mark.asyncio
async def test_empty_text_chunk_is_dropped() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "messages", "data": {"message": {"content": ""}}}],
    )
    assert out == []


@pytest.mark.asyncio
async def test_client_side_tool_call_routes_through_execute_agent_tool() -> None:
    """Non-enum tool names are wrapped as execute_agent_tool."""
    raw = [
        {
            "type": "messages",
            "ns": ["main"],
            "data": {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": f"{CLIENT_SIDE_TOOL_PREFIX}open_widget",
                            "args": {"widget_id": "w1"},
                            "id": "call-1",
                        }
                    ],
                }
            },
        }
    ]
    out = await _drive(DeepAgentEventAdapter(), raw)
    assert len(out) == 1
    fc = out[0]
    assert isinstance(fc, FunctionCallSSE)
    assert fc.data.function == "execute_agent_tool"
    assert fc.data.input_arguments == {
        "server_id": "main",
        "name": "open_widget",
        "arguments": {"widget_id": "w1"},
    }
    assert fc.data.extra_state == {"call_id": "call-1"}


@pytest.mark.asyncio
async def test_native_workspace_function_routes_directly() -> None:
    """Enum names pass through unchanged with top-level args."""
    raw = [
        {
            "type": "messages",
            "ns": [],
            "data": {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "get_widget_data",
                            "args": {"widget_id": "w-1"},
                            "id": "c-2",
                        }
                    ],
                }
            },
        }
    ]
    out = await _drive(
        DeepAgentEventAdapter(client_tool_names=frozenset({"get_widget_data"})),
        raw,
    )
    assert isinstance(out[0], FunctionCallSSE)
    assert out[0].data.function == "get_widget_data"
    assert out[0].data.input_arguments == {"widget_id": "w-1"}


@pytest.mark.asyncio
async def test_explicit_client_tool_set_routes_to_function_call() -> None:
    out = await _drive(
        DeepAgentEventAdapter(client_tool_names=frozenset({"render_chart"})),
        [
            {
                "type": "messages",
                "data": {
                    "message": {
                        "content": "",
                        "tool_calls": [{"name": "render_chart", "args": {}, "id": "x"}],
                    }
                },
            }
        ],
    )
    assert len(out) == 1
    assert isinstance(out[0], FunctionCallSSE)
    assert out[0].data.function == "execute_agent_tool"
    assert out[0].data.input_arguments["name"] == "render_chart"


@pytest.mark.asyncio
async def test_server_side_tool_call_emits_no_function_call() -> None:
    """Plain server-side tool calls emit no FunctionCallSSE."""
    out = await _drive(
        DeepAgentEventAdapter(client_tool_names=frozenset()),
        [
            {
                "type": "messages",
                "data": {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "list_widget_data",
                                "args": {},
                                "id": "c-1",
                            }
                        ],
                    }
                },
            }
        ],
    )
    assert not any(isinstance(ev, FunctionCallSSE) for ev in out)


@pytest.mark.asyncio
async def test_reasoning_content_coalesces_into_one_step_per_aimessage() -> None:
    """All reasoning_content deltas in one AIMessage buffer into one row."""
    from openbb_agent_server.protocol.schemas import StatusUpdateSSE

    raw = [
        {
            "type": "messages",
            "ns": [],
            "data": {
                "message": {
                    "id": "m-r",
                    "content": "",
                    "additional_kwargs": {"reasoning_content": "Let"},
                }
            },
        },
        {
            "type": "messages",
            "ns": [],
            "data": {
                "message": {
                    "id": "m-r",
                    "content": "",
                    "additional_kwargs": {"reasoning_content": " me think."},
                }
            },
        },
    ]
    out = await _drive(DeepAgentEventAdapter(), raw)
    status_msgs = [ev.data.message for ev in out if isinstance(ev, StatusUpdateSSE)]
    assert status_msgs == ["Let me think."]


@pytest.mark.asyncio
async def test_prose_during_tool_call_routes_to_reasoning() -> None:
    """Prose alongside tool_calls routes to a reasoning row."""
    from openbb_agent_server.protocol.schemas import (
        MessageChunkSSE,
        StatusUpdateSSE,
    )

    raw = [
        {
            "type": "messages",
            "ns": [],
            "data": {
                "message": {
                    "id": "m-t",
                    "content": "I'll fetch the balance sheet.",
                    "tool_calls": [
                        {"name": "list_widget_data", "args": {}, "id": "c-1"}
                    ],
                }
            },
        }
    ]
    out = await _drive(DeepAgentEventAdapter(), raw)
    assert any(
        isinstance(ev, StatusUpdateSSE) and "balance sheet" in ev.data.message
        for ev in out
    )
    assert not any(
        isinstance(ev, MessageChunkSSE) and "balance sheet" in ev.data.delta
        for ev in out
    )


@pytest.mark.asyncio
async def test_workspace_mcp_tool_carries_server_id_through_arguments() -> None:
    """An mcp tool name round-trips server id and function name."""
    raw = [
        {
            "type": "messages",
            "ns": ["graph"],
            "data": {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "mcp:wkspc:get_quote",
                            "args": {"ticker": "AAPL"},
                            "id": "c-9",
                        }
                    ],
                }
            },
        }
    ]
    out = await _drive(DeepAgentEventAdapter(), raw)
    assert isinstance(out[0], FunctionCallSSE)
    assert out[0].data.function == "execute_agent_tool"
    assert out[0].data.input_arguments == {
        "server_id": "wkspc",
        "name": "get_quote",
        "arguments": {"ticker": "AAPL"},
    }


@pytest.mark.asyncio
async def test_messages_event_extracts_text_from_block_list_content() -> None:
    """The adapter surfaces only text blocks from block-list content."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "messages",
                "ns": [],
                "data": {
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "internal scratch"},
                            {"type": "text", "text": "Hello "},
                            {"type": "text", "text": "world!"},
                        ],
                        "tool_calls": [],
                    },
                },
            },
        ],
    )
    chunks = [e for e in out if isinstance(e, MessageChunkSSE)]
    assert chunks, "block-list text content must surface as a chunk"
    assert chunks[0].data.delta == "Hello world!"


@pytest.mark.asyncio
async def test_messages_event_string_content_still_works() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"content": "plain string", "tool_calls": []}},
            },
        ],
    )
    chunks = [e for e in out if isinstance(e, MessageChunkSSE)]
    assert chunks[0].data.delta == "plain string"


@pytest.mark.asyncio
async def test_messages_event_skips_thinking_and_tool_use_blocks() -> None:
    """Reasoning / tool_use blocks are model-internal — never broadcast."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "messages",
                "ns": [],
                "data": {
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "secret reasoning"},
                            {"type": "tool_use", "id": "x", "name": "y", "input": {}},
                            {"type": "redacted_thinking", "data": "..."},
                        ],
                        "tool_calls": [],
                    },
                },
            },
        ],
    )
    chunks = [e for e in out if isinstance(e, MessageChunkSSE)]
    assert chunks == []


@pytest.mark.asyncio
async def test_messages_event_empty_content_emits_nothing() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"content": "", "tool_calls": []}},
            },
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"content": None, "tool_calls": []}},
            },
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"content": [], "tool_calls": []}},
            },
        ],
    )
    chunks = [e for e in out if isinstance(e, MessageChunkSSE)]
    assert chunks == []


@pytest.mark.asyncio
async def test_extract_text_handles_string_blocks_in_list() -> None:
    """Older LangChain shapes pack content as ``["chunk-a", "chunk-b"]``."""
    from openbb_agent_server.protocol.adapter import _extract_text

    assert _extract_text(["a", "b", "c"]) == "abc"


@pytest.mark.asyncio
async def test_updates_event_is_dropped_never_reaches_ui() -> None:
    """``updates`` events carry middleware lifecycle steps — never broadcast."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "updates",
                "ns": [],
                "data": {"ModelCallLimitMiddleware.before_model": {}},
            },
            {
                "type": "updates",
                "ns": ["tools:abc"],
                "data": {"_MemoryWriterMiddleware.after_agent": {}},
            },
        ],
    )
    assert out == []


@pytest.mark.asyncio
async def test_custom_chunk_passthrough() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "custom", "data": {"type": "chunk", "content": "passthrough"}}],
    )
    assert isinstance(out[0], MessageChunkSSE)
    assert out[0].data.delta == "passthrough"


@pytest.mark.asyncio
async def test_artifacts_drain_after_final_answer() -> None:
    """Artifacts buffer during the stream and drain after the final chunk."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "artifact",
                    "artifact": {
                        "type": "text",
                        "uuid": "art-1",
                        "name": "summary",
                        "description": "",
                        "content": "hello",
                    },
                },
            },
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"id": "m-final", "content": "All done."}},
            },
        ],
    )
    types = [type(ev).__name__ for ev in out]
    assert types.index("MessageChunkSSE") < types.index("MessageArtifactSSE")


@pytest.mark.asyncio
async def test_custom_artifact_table_is_flattened_to_records() -> None:
    """The adapter converts columns/rows shorthand to list-of-records."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "artifact",
                    "artifact": {
                        "type": "table",
                        "uuid": "u1",
                        "name": "Top 5",
                        "description": "five rows",
                        "columns": ["a", "b"],
                        "rows": [[1, 2], [3, 4]],
                    },
                },
            }
        ],
    )
    a = out[0]
    assert isinstance(a, MessageArtifactSSE)
    assert a.data.type == "table"
    assert a.data.uuid == "u1"
    assert a.data.content == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


@pytest.mark.asyncio
async def test_custom_artifact_html_passthrough() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "artifact",
                    "artifact": {
                        "type": "html",
                        "uuid": "h1",
                        "name": "Report",
                        "description": "...",
                        "content": "<h1>hi</h1>",
                    },
                },
            }
        ],
    )
    a = out[0]
    assert isinstance(a, MessageArtifactSSE)
    assert a.data.type == "html"
    assert a.data.content == "<h1>hi</h1>"


@pytest.mark.asyncio
async def test_custom_artifact_markdown_is_coerced_to_text() -> None:
    """A markdown artifact type is coerced to text."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "artifact",
                    "artifact": {
                        "type": "markdown",
                        "uuid": "m1",
                        "name": "Notes",
                        "description": "...",
                        "content": "# Hi",
                    },
                },
            }
        ],
    )
    a = out[0]
    assert isinstance(a, MessageArtifactSSE)
    assert a.data.type == "text"
    assert a.data.content == "# Hi"


@pytest.mark.asyncio
async def test_custom_artifact_chart_routes_plotly_to_chart_params() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "artifact",
                    "artifact": {
                        "type": "chart",
                        "uuid": "c1",
                        "name": "Trend",
                        "description": "...",
                        "plotly": {"data": [{"x": [1], "y": [2]}]},
                    },
                },
            }
        ],
    )
    a = out[0]
    assert isinstance(a, MessageArtifactSSE)
    assert a.data.type == "chart"
    assert a.data.chart_params == {"data": [{"x": [1], "y": [2]}]}


@pytest.mark.asyncio
async def test_error_event_becomes_error_status() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "error", "data": {"message": "kaboom"}}],
    )
    s = out[0]
    assert isinstance(s, StatusUpdateSSE)
    assert s.data.eventType == "ERROR"
    assert "kaboom" in s.data.message


@pytest.mark.asyncio
async def test_unknown_event_kind_is_silently_dropped() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "future-thing", "data": {"x": 1}}],
    )
    assert out == []


@pytest.mark.asyncio
async def test_custom_step_passthrough() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "step",
                    "event_type": "WARNING",
                    "message": "hot path",
                    "details": {"x": 1},
                },
            }
        ],
    )
    s = out[0]
    assert isinstance(s, StatusUpdateSSE)
    assert s.data.eventType == "WARNING"
    assert s.data.message == "hot path"
    assert s.data.details == [{"x": 1}]


@pytest.mark.asyncio
async def test_custom_step_success_is_coerced_to_info() -> None:
    """A SUCCESS event type is folded into INFO."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "step",
                    "event_type": "SUCCESS",
                    "message": "done",
                },
            }
        ],
    )
    s = out[0]
    assert isinstance(s, StatusUpdateSSE)
    assert s.data.eventType == "INFO"


@pytest.mark.asyncio
async def test_custom_function_call_passthrough() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "function_call",
                    "server_id": "wkspc",
                    "tool_name": "open_widget",
                    "parameters": {"id": "w1"},
                    "call_id": "c-9",
                },
            }
        ],
    )
    fc = out[0]
    assert isinstance(fc, FunctionCallSSE)
    assert fc.data.function == "execute_agent_tool"
    assert fc.data.input_arguments == {
        "server_id": "wkspc",
        "name": "open_widget",
        "arguments": {"id": "w1"},
    }
    assert fc.data.extra_state == {"call_id": "c-9"}


@pytest.mark.asyncio
async def test_custom_citations_passthrough() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "citations",
                    "citations": [
                        {
                            "id": "c-1",
                            "source_info": {
                                "type": "web",
                                "name": "Reuters",
                                "origin": "https://x.example",
                            },
                            "details": [{"text": "Quote"}],
                        }
                    ],
                },
            },
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"content": "Reuters reported the news."}},
            },
        ],
    )
    cc = next(e for e in out if isinstance(e, CitationCollectionSSE))
    assert len(cc.data.citations) == 1
    assert cc.data.citations[0].id == "c-1"
    assert cc.data.citations[0].source_info.origin == "https://x.example"


def test_citation_tokens_filters_short_words_and_stopwords() -> None:
    from openbb_agent_server.protocol.adapter import _citation_tokens

    assert _citation_tokens("The ASML 2026 Outlook is up") == {
        "asml",
        "2026",
        "outlook",
    }


def test_citation_is_relevant_widget_always_kept() -> None:
    from openbb_agent_server.protocol.adapter import _citation_is_relevant

    citation = {"source_info": {"type": "widget", "uuid": "w-1"}}
    assert _citation_is_relevant(citation, set()) is True


def test_citation_is_relevant_untitled_is_kept() -> None:
    from openbb_agent_server.protocol.adapter import _citation_is_relevant

    citation = {"source_info": {"type": "web", "origin": "https://x"}}
    assert _citation_is_relevant(citation, {"unrelated"}) is True


def test_citation_is_relevant_title_overlap_threshold() -> None:
    from openbb_agent_server.protocol.adapter import _citation_is_relevant

    citation = {"source_info": {"type": "web", "name": "ASML lifts 2026 sales outlook"}}
    assert _citation_is_relevant(
        citation, {"asml", "lifts", "2026", "sales", "outlook"}
    )
    assert not _citation_is_relevant(citation, {"hsbc", "roche"})


@pytest.mark.asyncio
async def test_drain_citations_drops_unreferenced_citations() -> None:
    """Citations the final answer never references are filtered out."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "citations",
                    "citations": [
                        {
                            "id": "keep",
                            "source_info": {
                                "type": "web",
                                "name": "ASML lifts 2026 sales outlook",
                                "origin": "https://a.example",
                            },
                        },
                        {
                            "id": "drop",
                            "source_info": {
                                "type": "web",
                                "name": "Celebrity gossip roundup weekly",
                                "origin": "https://b.example",
                            },
                        },
                    ],
                },
            },
            {
                "type": "messages",
                "ns": [],
                "data": {
                    "message": {"content": "ASML lifts its 2026 sales outlook sharply."}
                },
            },
        ],
    )
    cc = next(e for e in out if isinstance(e, CitationCollectionSSE))
    assert [c.id for c in cc.data.citations] == ["keep"]


@pytest.mark.asyncio
async def test_drain_citations_emits_nothing_when_all_unreferenced() -> None:
    """No ``CitationCollectionSSE`` at all when the answer cites nothing."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "citations",
                    "citations": [
                        {
                            "id": "x",
                            "source_info": {
                                "type": "web",
                                "name": "Unrelated celebrity gossip column",
                                "origin": "https://b.example",
                            },
                        }
                    ],
                },
            },
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"content": "The fund holds equities."}},
            },
        ],
    )
    assert not any(isinstance(e, CitationCollectionSSE) for e in out)


@pytest.mark.asyncio
async def test_drain_citations_widget_kept_without_answer_match() -> None:
    """Widget citations survive even when the answer text doesn't match."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "citations",
                    "citations": [
                        {
                            "id": "w",
                            "source_info": {
                                "type": "widget",
                                "uuid": "w-1",
                                "origin": "Blackrock",
                            },
                        }
                    ],
                },
            },
            {
                "type": "messages",
                "ns": [],
                "data": {"message": {"content": "Done."}},
            },
        ],
    )
    cc = next(e for e in out if isinstance(e, CitationCollectionSSE))
    assert [c.id for c in cc.data.citations] == ["w"]


@pytest.mark.asyncio
async def test_drain_citations_matches_against_table_artifact() -> None:
    """A citation referenced only by an emitted table artifact is kept."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "citations",
                    "citations": [
                        {
                            "id": "t",
                            "source_info": {
                                "type": "web",
                                "name": "Siemens orders rise sharply quarter",
                                "origin": "https://s.example",
                            },
                        }
                    ],
                },
            },
            {
                "type": "custom",
                "data": {
                    "type": "artifact",
                    "artifact": {
                        "type": "table",
                        "name": "Holdings",
                        "columns": ["Headline"],
                        "rows": [["Siemens orders rise sharply this quarter"]],
                    },
                },
            },
        ],
    )
    cc = next(e for e in out if isinstance(e, CitationCollectionSSE))
    assert [c.id for c in cc.data.citations] == ["t"]


@pytest.mark.asyncio
async def test_custom_unknown_kind_falls_back_to_info_status() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "custom", "data": {"type": "weird", "message": "hello"}}],
    )
    s = out[0]
    assert isinstance(s, StatusUpdateSSE)
    assert s.data.eventType == "INFO"
    assert s.data.message == "hello"


@pytest.mark.asyncio
async def test_updates_event_with_namespace_still_dropped() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "updates", "ns": ["tools:abc"], "data": {"agent": {}}}],
    )
    assert out == []


@pytest.mark.asyncio
async def test_updates_event_with_empty_data_still_dropped() -> None:
    out = await _drive(
        DeepAgentEventAdapter(),
        [{"type": "updates", "ns": [], "data": {}}],
    )
    assert out == []


def test_resolve_artifact_wire_type_maps_markdown_to_text() -> None:
    from openbb_agent_server.protocol.adapter import _resolve_artifact_wire_type

    assert _resolve_artifact_wire_type("markdown") == "text"


def test_resolve_artifact_wire_type_unknown_type_falls_back_to_text() -> None:
    from openbb_agent_server.protocol.adapter import _resolve_artifact_wire_type

    assert _resolve_artifact_wire_type("nope-foo") == "text"


def test_resolve_artifact_table_content_falls_back_to_raw_list() -> None:
    """When ``columns``/``rows`` are absent, fall back to ``raw_content``."""
    from openbb_agent_server.protocol.adapter import (
        _resolve_artifact_table_content,
    )

    out = _resolve_artifact_table_content({}, [{"a": 1}])
    assert out == [{"a": 1}]


def test_resolve_artifact_table_content_string_fallback_when_no_data() -> None:
    from openbb_agent_server.protocol.adapter import (
        _resolve_artifact_table_content,
    )

    out = _resolve_artifact_table_content({}, "plain text")
    assert out == "plain text"


def test_resolve_artifact_table_content_handles_none_raw_content() -> None:
    from openbb_agent_server.protocol.adapter import (
        _resolve_artifact_table_content,
    )

    out = _resolve_artifact_table_content({}, None)
    assert out == ""


def test_build_artifact_chart_with_non_dict_plotly_falls_back_to_string_content() -> (
    None
):
    """A chart artifact without a dict ``plotly`` payload stringifies content."""
    from openbb_agent_server.protocol.adapter import _build_artifact

    a = _build_artifact({"type": "chart", "plotly": "not-a-dict", "content": "x"})
    assert a.type == "chart"
    assert a.chart_params is None
    assert a.content == "x"


def test_build_artifact_html_content_passes_through() -> None:
    from openbb_agent_server.protocol.adapter import _build_artifact

    a = _build_artifact({"type": "html", "content": "<h1>hi</h1>"})
    assert a.type == "html"
    assert a.content == "<h1>hi</h1>"


def test_build_artifact_none_content_becomes_empty_string() -> None:
    from openbb_agent_server.protocol.adapter import _build_artifact

    a = _build_artifact({"type": "html"})
    assert a.content == ""


def test_build_artifact_non_string_non_list_content_stringifies() -> None:
    from openbb_agent_server.protocol.adapter import _build_artifact

    a = _build_artifact({"type": "html", "content": 42})
    assert a.content == "42"


def test_coerce_status_event_type_normalises_known_levels() -> None:
    from openbb_agent_server.protocol.adapter import _coerce_status_event_type

    assert _coerce_status_event_type("INFO") == "INFO"
    assert _coerce_status_event_type("info") == "INFO"
    assert _coerce_status_event_type("WARNING") == "WARNING"
    assert _coerce_status_event_type("ERROR") == "ERROR"
    assert _coerce_status_event_type(None) == "INFO"
    # SUCCESS not in the whitelist — folded to INFO.
    assert _coerce_status_event_type("SUCCESS") == "INFO"


def test_coerce_status_details_none_returns_none() -> None:
    from openbb_agent_server.protocol.adapter import _coerce_status_details

    assert _coerce_status_details(None) is None


def test_coerce_status_details_list_filters_non_dict_strings() -> None:
    from openbb_agent_server.protocol.adapter import _coerce_status_details

    out = _coerce_status_details([{"k": "v"}, "ok", 42, None])
    assert out == [{"k": "v"}, "ok"]


def test_coerce_status_details_empty_dict_returns_none() -> None:
    from openbb_agent_server.protocol.adapter import _coerce_status_details

    assert _coerce_status_details({}) is None


def test_coerce_status_details_dict_wraps_in_list() -> None:
    from openbb_agent_server.protocol.adapter import _coerce_status_details

    out = _coerce_status_details({"k": "v"})
    assert out == [{"k": "v"}]


def test_coerce_status_details_string_wraps_in_list() -> None:
    from openbb_agent_server.protocol.adapter import _coerce_status_details

    out = _coerce_status_details("hello")
    assert out == ["hello"]


def test_coerce_status_details_other_type_stringifies() -> None:
    from openbb_agent_server.protocol.adapter import _coerce_status_details

    out = _coerce_status_details(42)
    assert out == ["42"]


def test_split_thinking_extracts_blocks() -> None:
    from openbb_agent_server.protocol.adapter import _split_thinking

    thinking, prose = _split_thinking(
        "<think>step 1</think>The answer is 42<think>step 2</think>"
    )
    assert thinking == ["step 1", "step 2"]
    assert prose == "The answer is 42"


def test_split_thinking_skips_empty_thinking_blocks() -> None:
    from openbb_agent_server.protocol.adapter import _split_thinking

    thinking, prose = _split_thinking("<think>   </think>hello")
    assert thinking == []
    assert prose == "hello"


def test_extract_text_handles_block_with_no_type_but_text_field() -> None:
    """A content block with no type but a text field still concatenates."""
    from openbb_agent_server.protocol.adapter import _extract_text

    out = _extract_text([{"text": "hello"}, {"text": "world"}])
    assert "hello" in out
    assert "world" in out


def test_extract_text_handles_non_list_non_string() -> None:
    """Anything other than str or list falls through to str."""
    from openbb_agent_server.protocol.adapter import _extract_text

    assert _extract_text(42) == "42"
    assert _extract_text(None) == ""


def test_thinking_splitter_routes_blocks_to_status_then_chunk() -> None:
    """Inline think blocks route to status, the rest to chunks."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter
    from openbb_agent_server.protocol.schemas import (
        MessageChunkSSE,
        StatusUpdateSSE,
    )

    adapter = DeepAgentEventAdapter()
    adapter._cur_id = "m-1"
    adapter._splitter = _ThinkingStreamSplitter()
    splits = adapter._splitter.feed("<think>step a</think>final answer")
    out = adapter._emit_splits(splits) + adapter._drain_pending()
    assert any(isinstance(e, StatusUpdateSSE) for e in out)
    assert any(isinstance(e, MessageChunkSSE) for e in out)


def test_thinking_splitter_buffers_prose_until_close() -> None:
    """Prose buffers and is routed at boundary time."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    adapter = DeepAgentEventAdapter()
    adapter._splitter = _ThinkingStreamSplitter()
    out = adapter._emit_splits(adapter._splitter.feed("hello world"))
    assert out == []
    assert "".join(adapter._prose_buf) == "hello world"


def test_thinking_splitter_holds_back_only_partial_tag_tail() -> None:
    """Only a partial-tag tail is held back; everything before flushes."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    first = sp.feed("hello <thin")
    assert first == [("prose", "hello ")]
    second = sp.feed("king>secret</thinking>tail")
    channels = [c for c, _ in second]
    assert "thinking" in channels
    assert ("prose", "tail") in second


def test_thinking_splitter_switches_channel_when_full_tag_arrives_in_one_delta() -> (
    None
):
    """A whole thinking block in one chunk routes body and trailing prose."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    out = sp.feed("pre <thinking>secret</thinking>tail")
    assert out == [
        ("prose", "pre "),
        ("thinking", "secret"),
        ("prose", "tail"),
    ]


def test_thinking_splitter_does_not_hold_back_when_no_lt_in_buffer() -> None:
    """With no angle bracket in the buffer, every character flushes."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    out = sp.feed("plain text with no tag")
    assert out == [("prose", "plain text with no tag")]


def test_absorb_citations_skips_non_list_input() -> None:
    adapter = DeepAgentEventAdapter()
    adapter._absorb_citations("not-a-list")
    assert adapter._citations == []


def test_absorb_citations_skips_non_dict_entries() -> None:
    adapter = DeepAgentEventAdapter()
    adapter._absorb_citations(["bad-entry", {"source_info": {"origin": "u"}}])
    assert len(adapter._citations) == 1


def test_absorb_citations_dedupes_by_url_and_snippet() -> None:
    adapter = DeepAgentEventAdapter()
    payload = {
        "source_info": {"origin": "http://x"},
        "details": [{"text": "snippet"}],
    }
    adapter._absorb_citations([payload, payload])
    assert len(adapter._citations) == 1


def test_build_function_call_skips_non_dict_tool_call() -> None:
    adapter = DeepAgentEventAdapter()
    assert adapter._build_function_call_from_tool_call("not-a-dict", ("ns",)) is None


def test_build_function_call_skips_blank_name() -> None:
    adapter = DeepAgentEventAdapter()
    assert (
        adapter._build_function_call_from_tool_call(
            {"name": "", "args": {}, "id": "x"}, ("ns",)
        )
        is None
    )


def test_build_function_call_skips_workspace_native_when_not_client_tool() -> None:
    """Built-in functions are filtered unless the agent declared them."""
    from openbb_agent_server.protocol.adapter import _WORKSPACE_NATIVE_FUNCTIONS

    natives = list(_WORKSPACE_NATIVE_FUNCTIONS)
    if not natives:
        pytest.skip("no built-in native functions to test against")
    adapter = DeepAgentEventAdapter()
    out = adapter._build_function_call_from_tool_call(
        {"name": natives[0], "args": {}, "id": "x"}, ("ns",)
    )
    assert out is None


def test_translate_messages_emits_reasoning_kwarg_as_status_update() -> None:
    """reasoning_content is surfaced as an INFO step."""
    from openbb_agent_server.protocol.schemas import StatusUpdateSSE

    adapter = DeepAgentEventAdapter()
    out = adapter._translate_messages(
        {
            "message": {
                "id": "",
                "content": "",
                "tool_calls": [],
                "additional_kwargs": {"reasoning_content": "thinking step"},
            }
        },
        ns=("agent",),
    )
    out += adapter._drain_pending()
    assert any(
        isinstance(e, StatusUpdateSSE) and "thinking step" in e.data.message
        for e in out
    )


def test_translate_messages_flattens_reasoning_block_list() -> None:
    """reasoning_content as a list of blocks is joined."""
    from openbb_agent_server.protocol.schemas import StatusUpdateSSE

    adapter = DeepAgentEventAdapter()
    out = adapter._translate_messages(
        {
            "message": {
                "id": "",
                "content": "",
                "tool_calls": [],
                "additional_kwargs": {
                    "reasoning_content": [
                        {"text": "step a "},
                        {"text": "step b"},
                    ]
                },
            }
        },
        ns=("agent",),
    )
    out += adapter._drain_pending()
    found = [e for e in out if isinstance(e, StatusUpdateSSE)]
    assert any("step a step b" in e.data.message for e in found)


def test_translate_error_event_emits_error_status() -> None:
    """Error events become StatusUpdateSSE ERRORs."""
    from openbb_agent_server.protocol.schemas import StatusUpdateSSE

    adapter = DeepAgentEventAdapter()
    out = adapter._translate(
        {"type": "error", "ns": ["x"], "data": {"message": "kaboom"}}
    )
    [ev] = out
    assert isinstance(ev, StatusUpdateSSE)
    assert ev.data.eventType == "ERROR"
    assert "kaboom" in ev.data.message


def test_translate_unknown_kind_returns_empty() -> None:
    adapter = DeepAgentEventAdapter()
    assert adapter._translate({"type": "what?", "ns": [], "data": {}}) == []


def test_translate_messages_id_less_with_tool_indicator_routes_to_reasoning() -> None:
    """Content with tool_calls in an id-less message routes to reasoning."""
    from openbb_agent_server.protocol.schemas import MessageChunkSSE, StatusUpdateSSE

    adapter = DeepAgentEventAdapter()
    out = adapter._translate_messages(
        {
            "message": {
                "id": "",
                "content": "I will call the tool now",
                "tool_calls": [{"name": "search", "args": {}, "id": "c1"}],
            }
        },
        ns=("agent",),
    )
    assert any(
        isinstance(e, StatusUpdateSSE) and "call the tool" in e.data.message
        for e in out
    )
    assert all(not isinstance(e, MessageChunkSSE) for e in out)


def test_splitter_empty_delta_returns_empty() -> None:
    """An empty delta is a no-op — the early-return guard."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    assert sp.feed("") == []


def test_splitter_suppresses_after_harmony_marker() -> None:
    """The first Harmony marker truncates the buffer and refuses further deltas."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    out = sp.feed("visible answer <|channel|>functions.tool_call")
    assert out == [("prose", "visible answer ")]
    assert sp.feed("more leaked text") == []
    assert sp._harmony_suppress is True


def test_splitter_stray_close_reclassifies_preceding_prose() -> None:
    """A stray close emits a close_unmatched marker and reroutes prose."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    out = sp.feed("hidden reasoning</think>visible tail")
    assert ("close_unmatched", "") in out
    assert ("thinking", "hidden reasoning") in out
    assert ("prose", "visible tail") in out
    out2 = sp.feed("more</think>again")
    assert ("close_unmatched", "") not in out2


def test_splitter_flush_emits_held_thinking_tail() -> None:
    """A flush inside an unterminated thinking block emits on that channel."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    sp.feed("<thinking>still going <")
    flushed = sp.flush()
    assert flushed == [("thinking", "<")]
    assert sp.flush() == []


def test_splitter_safe_emit_end_holds_back_long_buffer_past_marker() -> None:
    """A ``<`` far from the buffer tail flushes everything before it."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    out = sp.feed("<" + "x" * 64)
    assert out == [("prose", "<" + "x" * 64)]


def test_splitter_holds_unclosed_citation_marker_until_closed() -> None:
    """An unclosed citation marker is held in full until it closes."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    out = sp.feed('answer text 【cite_source text="' + "y" * 200)
    assert out == [("prose", "answer text ")]
    [(channel, text)] = sp.feed('" source="f.pdf"】 and the rest')
    assert channel == "prose"
    assert text.startswith("【cite_source") and text.endswith(" and the rest")


def test_splitter_holds_unclosed_pipe_citation_token_until_closed() -> None:
    """An unclosed pipe-token citation is held in full until it closes."""
    from openbb_agent_server.protocol.adapter import _ThinkingStreamSplitter

    sp = _ThinkingStreamSplitter()
    out = sp.feed("performance.<|start_citation_id|>" + "z" * 200)
    assert out == [("prose", "performance.")]
    [(channel, text)] = sp.feed("<|end_citation_id|> and the remainder of the answer.")
    assert channel == "prose"
    assert text.startswith("<|start_citation_id|>")
    assert text.endswith(" and the remainder of the answer.")


def test_emit_splits_strips_inline_cite_source_marker() -> None:
    """An inline cite_source marker is stripped from prose."""
    adapter = DeepAgentEventAdapter()
    adapter._emit_splits(
        [
            (
                "prose",
                'See page 138.【cite_source text="basket" '
                'source="iefa_prospectus.pdf" source_url=null】 Next sentence.',
            )
        ]
    )
    assert adapter._prose_buf == ["See page 138. Next sentence."]


def test_emit_splits_strips_bare_citation_id_markers() -> None:
    """Bare citation-id refs are stripped, short CJK prose is kept."""
    adapter = DeepAgentEventAdapter()
    adapter._emit_splits(
        [
            (
                "prose",
                "Gold fell on a stronger dollar.【rCW9mowjZMqwr7hu】 "
                "See note 【6月】 for detail.",
            )
        ]
    )
    assert adapter._prose_buf == [
        "Gold fell on a stronger dollar. See note 【6月】 for detail."
    ]


def test_emit_splits_strips_pipe_token_citation_markers() -> None:
    """Pipe-token citation markers are stripped from prose."""
    adapter = DeepAgentEventAdapter()
    adapter._emit_splits(
        [
            (
                "prose",
                "Earnings land July 27 2026."
                "<|start_citation_id|>2w5vZcK9VgU5K3Xh<|end_citation_id|>"
                " That is a catalyst.<|end_citation_id|>",
            )
        ]
    )
    assert adapter._prose_buf == ["Earnings land July 27 2026. That is a catalyst."]


def test_flatten_reasoning_non_string_non_list_stringifies() -> None:
    """A reasoning payload that is neither str nor list is stringified."""
    from openbb_agent_server.protocol.adapter import _flatten_reasoning

    assert _flatten_reasoning(42) == "42"


def test_emit_splits_close_unmatched_moves_prose_to_reasoning() -> None:
    """A close_unmatched signal moves buffered prose into reasoning."""
    adapter = DeepAgentEventAdapter()
    adapter._prose_buf = ["earlier prose"]
    adapter._emit_splits([("close_unmatched", "")])
    assert adapter._prose_buf == []
    assert adapter._reasoning_buf == ["earlier prose"]


def test_emit_splits_skips_empty_text_pairs() -> None:
    """An empty channel-text pair contributes nothing."""
    adapter = DeepAgentEventAdapter()
    adapter._emit_splits([("prose", ""), ("thinking", "")])
    assert adapter._prose_buf == []
    assert adapter._reasoning_buf == []


def test_emit_splits_prose_that_is_all_citation_marker_is_dropped() -> None:
    """Prose consisting solely of a citation marker collapses to nothing."""
    adapter = DeepAgentEventAdapter()
    adapter._emit_splits([("prose", "【cursor:1】")])
    assert adapter._prose_buf == []


@pytest.mark.asyncio
async def test_widget_citation_dedupes_by_uuid() -> None:
    """Widget citations key on the per-instance uuid, not the vendor origin."""
    adapter = DeepAgentEventAdapter()
    base = {
        "source_info": {
            "type": "widget",
            "uuid": "w-uuid-1",
            "origin": "Blackrock",
        }
    }
    adapter._absorb_citations([base, dict(base)])
    assert len(adapter._citations) == 1


@pytest.mark.asyncio
async def test_widget_citation_pages_get_separate_chips() -> None:
    """PDF widget citations include ``Page: N`` in the dedup key."""
    adapter = DeepAgentEventAdapter()
    page1 = {
        "source_info": {
            "type": "widget",
            "uuid": "doc-1",
            "metadata": {"Page": "1"},
        }
    }
    page2 = {
        "source_info": {
            "type": "widget",
            "uuid": "doc-1",
            "metadata": {"Page": "2"},
        }
    }
    adapter._absorb_citations([page1, page2, dict(page1)])
    assert len(adapter._citations) == 2


@pytest.mark.asyncio
async def test_widget_citation_falls_back_to_widget_id() -> None:
    """A widget citation with no ``uuid`` keys on ``widget_id`` instead."""
    adapter = DeepAgentEventAdapter()
    adapter._absorb_citations(
        [{"source_info": {"type": "widget", "widget_id": "slug-x"}}]
    )
    assert len(adapter._citations) == 1


@pytest.mark.asyncio
async def test_artifact_build_failure_is_swallowed() -> None:
    """An artifact payload that breaks _build_artifact emits nothing."""
    out = await _drive(
        DeepAgentEventAdapter(),
        [
            {
                "type": "custom",
                "data": {
                    "type": "artifact",
                    "artifact": {
                        "type": "table",
                        "name": "broken",
                        "uuid": "b1",
                        "columns": ["a"],
                        "rows": [42],
                    },
                },
            }
        ],
    )
    assert out == []


@pytest.mark.asyncio
async def test_custom_step_flushes_splitter_and_reasoning() -> None:
    """A step event flushes the splitter and reasoning before emitting."""
    raw = [
        {
            "type": "messages",
            "ns": [],
            "data": {"message": {"id": "m-1", "content": "thinking out loud "}},
        },
        {
            "type": "custom",
            "data": {"type": "step", "message": "a step"},
        },
    ]
    out = await _drive(DeepAgentEventAdapter(), raw)
    msgs = [e.data.message for e in out if isinstance(e, StatusUpdateSSE)]
    assert "thinking out loud" in " ".join(msgs)
    assert "a step" in msgs


@pytest.mark.asyncio
async def test_mcp_tool_without_inner_colon_uses_namespace_server_id() -> None:
    """An mcp name with no inner colon takes the server id from ns."""
    raw = [
        {
            "type": "messages",
            "ns": ["nspace"],
            "data": {
                "message": {
                    "content": "",
                    "tool_calls": [{"name": "mcp:bare_fn", "args": {}, "id": "c-1"}],
                }
            },
        }
    ]
    out = await _drive(DeepAgentEventAdapter(), raw)
    assert isinstance(out[0], FunctionCallSSE)
    assert out[0].data.input_arguments == {
        "server_id": "nspace",
        "name": "bare_fn",
        "arguments": {},
    }


@pytest.mark.asyncio
async def test_new_message_id_rotates_splitter_and_flushes_old() -> None:
    """A new message id flushes the prior splitter before rotating."""
    raw = [
        {
            "type": "messages",
            "ns": [],
            "data": {"message": {"id": "m-1", "content": "first part "}},
        },
        {
            "type": "messages",
            "ns": [],
            "data": {"message": {"id": "m-2", "content": "second part"}},
        },
    ]
    out = await _drive(DeepAgentEventAdapter(), raw)
    delta = "".join(e.data.delta for e in out if isinstance(e, MessageChunkSSE))
    assert "first part" in delta
    assert "second part" in delta


def test_translate_messages_idless_event_makes_disposable_splitter() -> None:
    """A boundary-less (id-less) event with no splitter yet creates one."""
    adapter = DeepAgentEventAdapter()
    assert adapter._splitter is None
    adapter._translate_messages(
        {"message": {"id": "", "content": "hello", "tool_calls": []}},
        ns=("agent",),
    )
    assert adapter._splitter is not None


def test_translate_custom_inline_artifact_emits_message_artifact() -> None:
    """_translate_custom with an artifact kind emits a MessageArtifactSSE."""
    adapter = DeepAgentEventAdapter()
    out = adapter._translate_custom(
        {
            "type": "artifact",
            "artifact": {
                "type": "text",
                "uuid": "a1",
                "name": "n",
                "content": "body",
            },
        }
    )
    assert len(out) == 1
    assert isinstance(out[0], MessageArtifactSSE)


def test_protocol_adapter_passes_citations_through() -> None:
    """Buffer citations through _translate and flush them as one."""
    from openbb_agent_server.protocol.adapter import DeepAgentEventAdapter
    from openbb_agent_server.protocol.schemas import CitationCollectionSSE

    adapter = DeepAgentEventAdapter(client_tool_names=set())
    inline = adapter._translate(
        {
            "type": "custom",
            "data": {
                "type": "citations",
                "citations": [
                    {
                        "id": "c-1",
                        "source_info": {
                            "type": "web",
                            "name": "T",
                            "origin": "https://x.example",
                        },
                        "details": [{"text": "snippet"}],
                    }
                ],
            },
        }
    )
    assert inline == []
    [cc] = adapter._drain_citations()
    assert isinstance(cc, CitationCollectionSSE)
    assert cc.data.citations[0].id == "c-1"
    assert cc.data.citations[0].source_info.origin == "https://x.example"
