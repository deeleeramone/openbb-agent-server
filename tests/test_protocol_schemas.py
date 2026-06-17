"""Wire-protocol schema conformance tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from openbb_agent_server.protocol.schemas import (
    ChatMessage,
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    CitationHighlightBoundingBox,
    ClientArtifact,
    FunctionCallSSE,
    FunctionCallSSEData,
    MessageArtifactSSE,
    MessageChunkSSE,
    MessageChunkSSEData,
    QueryRequest,
    SourceInfo,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    WidgetParam,
    WidgetsBag,
    WidgetSpec,
)


def test_message_chunk_event_name_is_canonical() -> None:
    sse = MessageChunkSSE(data=MessageChunkSSEData(delta="x"))
    assert sse.event == "copilotMessageChunk"


def test_status_update_event_name_is_canonical() -> None:
    sse = StatusUpdateSSE(data=StatusUpdateSSEData(eventType="INFO", message="x"))
    assert sse.event == "copilotStatusUpdate"


def test_function_call_event_name_is_canonical() -> None:
    sse = FunctionCallSSE(
        data=FunctionCallSSEData(function="execute_agent_tool", input_arguments={})
    )
    assert sse.event == "copilotFunctionCall"


def test_message_artifact_event_name_is_canonical() -> None:
    sse = MessageArtifactSSE(
        data=ClientArtifact(
            type="text", name="n", description="d", uuid="u", content=""
        )
    )
    assert sse.event == "copilotMessageArtifact"


def test_citation_collection_event_name_is_canonical() -> None:
    sse = CitationCollectionSSE(data=CitationCollection(citations=[]))
    assert sse.event == "copilotCitationCollection"


def test_message_chunk_data_payload_is_just_delta() -> None:
    """The wire payload is just a delta field."""
    sse = MessageChunkSSE(data=MessageChunkSSEData(delta="hello"))
    assert json.loads(sse.data.model_dump_json()) == {"delta": "hello"}


def test_status_update_uses_camelcase_eventtype_field() -> None:
    """The status update uses a camelCase eventType field."""
    sse = StatusUpdateSSE(
        data=StatusUpdateSSEData(eventType="WARNING", message="hot path"),
    )
    payload = json.loads(sse.data.model_dump_json())
    assert payload["eventType"] == "WARNING"
    assert payload["group"] == "reasoning"
    assert payload["hidden"] is False


def test_status_update_accepts_success_eventtype() -> None:
    """SUCCESS is a first-class wire eventType value."""
    sse = StatusUpdateSSEData(eventType="SUCCESS", message="ok")
    assert sse.eventType == "SUCCESS"
    with pytest.raises(ValidationError):
        StatusUpdateSSEData(eventType="DEBUG", message="x")  # type: ignore[arg-type]


def test_function_call_function_field_is_closed_enum() -> None:
    """The function field is a closed enum."""
    with pytest.raises(ValidationError):
        FunctionCallSSEData(function="my_custom_tool", input_arguments={})  # type: ignore[arg-type]


def test_function_call_native_function_round_trips() -> None:
    sse = FunctionCallSSE(
        data=FunctionCallSSEData(
            function="get_widget_data",
            input_arguments={"widget_id": "w-1"},
            extra_state={"call_id": "c-1"},
        )
    )
    payload = json.loads(sse.data.model_dump_json())
    assert payload["function"] == "get_widget_data"
    assert payload["input_arguments"] == {"widget_id": "w-1"}
    assert payload["extra_state"] == {"call_id": "c-1"}


def test_client_artifact_round_trip() -> None:
    art = ClientArtifact(
        type="html",
        name="Report",
        description="proof of life",
        uuid="u-1",
        content="<h1>hi</h1>",
    )
    payload = json.loads(art.model_dump_json())
    assert payload["type"] == "html"
    assert payload["uuid"] == "u-1"
    assert payload["content"] == "<h1>hi</h1>"


def test_client_artifact_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        ClientArtifact(  # type: ignore[arg-type]
            type="markdown", name="x", description="x", uuid="u", content=""
        )


def test_query_request_minimal_payload_validates() -> None:
    req = QueryRequest.model_validate(
        {"messages": [{"role": "human", "content": "hi"}]}
    )
    assert len(req.messages) == 1
    assert req.messages[0].role == "human"
    assert req.workspace_options == {}


def test_query_request_workspace_options_keyed_object_passes_through() -> None:
    """The current Workspace shape — option values keyed by id."""
    req = QueryRequest.model_validate(
        {
            "messages": [{"role": "human", "content": "hi"}],
            "workspace_options": {"search-web": True, "fetch-url": False},
        }
    )
    assert req.workspace_options == {"search-web": True, "fetch-url": False}


def test_query_request_workspace_options_list_is_coerced() -> None:
    """A ``list[str]`` of enabled ids coerces to ``{id: True}``."""
    req = QueryRequest.model_validate(
        {
            "messages": [{"role": "human", "content": "hi"}],
            "workspace_options": ["search-web", "deep-research"],
        }
    )
    assert req.workspace_options == {"search-web": True, "deep-research": True}


def test_query_request_workspace_options_null_coerces_to_empty() -> None:
    req = QueryRequest.model_validate(
        {
            "messages": [{"role": "human", "content": "hi"}],
            "workspace_options": None,
        }
    )
    assert req.workspace_options == {}


def test_chat_message_tool_role_carries_tool_call_id() -> None:
    m = ChatMessage(role="tool", content="result", tool_call_id="abc")
    assert m.tool_call_id == "abc"


def test_chat_message_tool_role_can_carry_function_call_result_shape() -> None:
    """A tool-role message can carry the function-call result shape."""
    m = ChatMessage.model_validate(
        {
            "role": "tool",
            "function": "get_widget_data",
            "input_arguments": {"widget_id": "w"},
            "data": [{"items": [{"data_format": {}, "content": "rows"}]}],
        }
    )
    assert m.function == "get_widget_data"
    assert m.input_arguments == {"widget_id": "w"}
    assert m.content is None


def test_widget_spec_id_property_picks_uuid_over_widget_id() -> None:
    w = WidgetSpec(uuid="u-1", widget_id="wid-1")
    assert w.id == "u-1"


def test_widget_spec_id_falls_back_to_widget_id() -> None:
    w = WidgetSpec(widget_id="wid-1")
    assert w.id == "wid-1"


def test_widget_spec_id_empty_when_neither_set() -> None:
    w = WidgetSpec()
    assert w.id == ""


def test_widgets_bag_default_empty_lists() -> None:
    bag = WidgetsBag()
    assert bag.primary == []
    assert bag.secondary == []
    assert bag.extra == []


def test_widget_param_accepts_arbitrary_current_value() -> None:
    p = WidgetParam(name="ticker", type="string", current_value="AAPL")
    assert p.current_value == "AAPL"


def test_citation_bounding_box_carries_pixel_coordinates() -> None:
    box = CitationHighlightBoundingBox(
        text="quote", page=1, x0=72.0, top=117.0, x1=259.0, bottom=135.0
    )
    assert box.x0 == 72.0
    assert box.bottom == 135.0


def test_citation_round_trip_with_widget_source() -> None:
    c = Citation(
        id="c-1",
        source_info=SourceInfo(type="widget", widget_id="w-1", name="Earnings"),
        details=[{"text": "Revenue grew 12%"}],
    )
    assert c.source_info.widget_id == "w-1"
    assert c.details == [{"text": "Revenue grew 12%"}]


def test_citation_round_trip_with_pdf_bounding_boxes() -> None:
    c = Citation(
        id="c-2",
        source_info=SourceInfo(type="direct retrieval", name="10-K"),
        quote_bounding_boxes=[
            [
                CitationHighlightBoundingBox(
                    text="t", page=1, x0=0, top=0, x1=10, bottom=10
                )
            ]
        ],
    )
    assert c.quote_bounding_boxes is not None
    assert c.quote_bounding_boxes[0][0].page == 1


def test_citation_collection_payload_is_wrapped() -> None:
    """Workspace expects ``data: {"citations": [...]}``, not a raw list."""
    sse = CitationCollectionSSE(
        data=CitationCollection(
            citations=[
                Citation(
                    id="c-1",
                    source_info=SourceInfo(type="web", origin="https://x.example"),
                )
            ]
        )
    )
    payload = json.loads(sse.data.model_dump_json())
    assert "citations" in payload
    assert payload["citations"][0]["id"] == "c-1"
