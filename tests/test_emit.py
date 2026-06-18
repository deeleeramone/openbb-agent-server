"""Emit-helper tests."""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.runtime import emit


@pytest.fixture
def captured_writes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def fake_writer() -> Any:
        return captured.append

    monkeypatch.setattr(emit, "_writer", fake_writer)
    return captured


def test_no_writer_falls_back_to_noop() -> None:
    emit.reasoning_step("nope")
    emit.html_artifact(content="<div/>")
    emit.markdown_artifact(content="hi")
    emit.table_artifact(columns=["a"], rows=[[1]])
    emit.chart_artifact(plotly={"data": []})
    emit.image_artifact(data_base64="xx")
    emit.file_artifact(data_base64="yy")
    emit.cite(text="t", source="s")
    emit.function_call(tool_name="t", parameters={})


def test_reasoning_step_writes_step_payload(
    captured_writes: list[dict[str, Any]],
) -> None:
    emit.reasoning_step("starting", event_type="INFO", k=1)
    assert captured_writes == [
        {
            "type": "step",
            "event_type": "INFO",
            "message": "starting",
            "details": {"k": 1},
        }
    ]


def test_html_artifact_emits_artifact_envelope(
    captured_writes: list[dict[str, Any]],
) -> None:
    uuid_returned = emit.html_artifact(
        content="<h1>hi</h1>", name="Report", description="d"
    )
    [payload] = captured_writes
    assert payload["type"] == "artifact"
    inner = payload["artifact"]
    assert inner["type"] == "html"
    assert inner["uuid"] == uuid_returned
    assert inner["content"] == "<h1>hi</h1>"


def test_table_artifact_payload_shape(captured_writes: list[dict[str, Any]]) -> None:
    emit.table_artifact(columns=["a", "b"], rows=[[1, 2]], name="t")
    inner = captured_writes[0]["artifact"]
    assert inner == pytest.approx(
        {
            "type": "table",
            "uuid": inner["uuid"],
            "name": "t",
            "description": "",
            "columns": ["a", "b"],
            "rows": [[1, 2]],
        }
    )


def test_chart_artifact_payload_shape(captured_writes: list[dict[str, Any]]) -> None:
    emit.chart_artifact(plotly={"data": [1, 2]}, name="c")
    inner = captured_writes[0]["artifact"]
    assert inner["type"] == "chart"
    assert inner["plotly"] == {"data": [1, 2]}


def test_image_artifact_renders_inline_via_html(
    captured_writes: list[dict[str, Any]],
) -> None:
    """Wrap an image artifact as inline HTML."""
    emit.image_artifact(data_base64="xx", mime="image/png", name="logo")
    inner = captured_writes[0]["artifact"]
    assert inner["type"] == "html"
    assert 'src="data:image/png;base64,xx"' in inner["content"]


def test_image_artifact_with_url_uses_url_directly(
    captured_writes: list[dict[str, Any]],
) -> None:
    emit.image_artifact(url="https://x.example/p.png")
    inner = captured_writes[0]["artifact"]
    assert 'src="https://x.example/p.png"' in inner["content"]


def test_image_artifact_requires_either_data_or_url() -> None:
    with pytest.raises(ValueError):
        emit.image_artifact()


def test_file_artifact_renders_download_link_via_html(
    captured_writes: list[dict[str, Any]],
) -> None:
    """Surface a file artifact as an HTML download link."""
    emit.file_artifact(url="http://example.com/x.csv", mime="text/csv", name="x.csv")
    inner = captured_writes[0]["artifact"]
    assert inner["type"] == "html"
    assert 'href="http://example.com/x.csv"' in inner["content"]
    assert 'download="x.csv"' in inner["content"]


def test_file_artifact_with_data_base64_uses_data_url(
    captured_writes: list[dict[str, Any]],
) -> None:
    emit.file_artifact(data_base64="aGVsbG8=", mime="text/plain", name="hi.txt")
    inner = captured_writes[0]["artifact"]
    assert "data:text/plain;base64,aGVsbG8=" in inner["content"]


def test_file_artifact_requires_either_data_or_url() -> None:
    with pytest.raises(ValueError):
        emit.file_artifact()


def test_cite_payload_shape(captured_writes: list[dict[str, Any]]) -> None:
    """Build the openbb-ai Citation shape."""
    emit.cite(text="quote", source="Reuters", source_url="http://x")
    [payload] = captured_writes
    assert payload["type"] == "citations"
    [citation] = payload["citations"]
    assert citation["id"]
    assert citation["source_info"]["type"] == "web"
    assert citation["source_info"]["name"] == "Reuters"
    assert citation["source_info"]["origin"] == "http://x"
    assert citation["details"] == [
        {"text": "quote", "url": "http://x", "title": "Reuters"}
    ]


def test_function_call_payload_shape(captured_writes: list[dict[str, Any]]) -> None:
    cid = emit.function_call(tool_name="open_widget", parameters={"id": "w"})
    [payload] = captured_writes
    assert payload["type"] == "function_call"
    assert payload["tool_name"] == "open_widget"
    assert payload["call_id"] == cid


def test_markdown_artifact_payload_shape(captured_writes: list[dict[str, Any]]) -> None:
    emit.markdown_artifact(content="# Hi", name="m")
    inner = captured_writes[0]["artifact"]
    assert inner["type"] == "markdown"
    assert inner["content"] == "# Hi"


def test_code_artifact_payload_shape(captured_writes: list[dict[str, Any]]) -> None:
    uid = emit.code_artifact(content="print(1)", language="python", name="snippet")
    inner = captured_writes[0]["artifact"]
    assert inner["type"] == "code"
    assert inner["content"] == "print(1)"
    assert inner["language"] == "python"
    assert inner["uuid"] == uid
    assert inner["name"] == "snippet"


def test_bind_writer_overrides_resolution() -> None:
    """Make _writer() return the bound sink."""
    captured: list[dict[str, Any]] = []
    with emit.bind_writer(captured.append):
        emit.reasoning_step("hello")
    assert captured == [
        {"type": "step", "event_type": "INFO", "message": "hello", "details": {}}
    ]


def test_bind_writer_reset_after_with_block() -> None:
    captured: list[dict[str, Any]] = []
    with emit.bind_writer(captured.append):
        pass
    emit.reasoning_step("not-captured")
    assert captured == []


def test_cite_no_writer_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Warn instead of emitting when no stream writer is bound."""
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        emit.cite(text="t", source="Reuters", source_url="http://x")
    assert any("no stream writer" in r.message for r in caplog.records)


def test_cite_widget_source_info_shape(
    captured_writes: list[dict[str, Any]],
) -> None:
    """Produce a widget-shaped source_info when citing a widget."""
    emit.cite(text="from-widget", widget="balance-sheet", source="Balance Sheet")
    [payload] = captured_writes
    [citation] = payload["citations"]
    src = citation["source_info"]
    assert src["type"] == "widget"
    assert src["uuid"] == "balance-sheet"
    assert src["metadata"]["widget_uuid"] == "balance-sheet"
    assert src["name"] == "Balance Sheet"


def test_cite_attaches_metadata_from_input_arguments_and_extras(
    captured_writes: list[dict[str, Any]],
) -> None:
    """Attach input_arguments and extra_details to a citation."""
    emit.cite(
        text="t",
        source="Reuters",
        input_arguments={"sym": "AAPL"},
        extra_details={"published_at": "2026-05-01"},
    )
    [payload] = captured_writes
    [citation] = payload["citations"]
    meta = citation["source_info"]["metadata"]
    assert meta["input_args"] == {"sym": "AAPL"}
    assert {"published_at": "2026-05-01"} in citation["details"]


def test_cite_includes_quote_bounding_boxes_when_provided(
    captured_writes: list[dict[str, Any]],
) -> None:
    """Include quote_bounding_boxes on the citation."""
    boxes = [[{"page": 1, "x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4}]]
    emit.cite(text="quote", widget="doc-uuid", quote_bounding_boxes=boxes)
    [payload] = captured_writes
    [citation] = payload["citations"]
    assert citation["quote_bounding_boxes"] == boxes
