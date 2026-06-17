"""artifacts tool source tests."""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.plugins.tools.artifacts import ArtifactsToolSource
from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    monkeypatch.setattr(emit, "_writer", lambda: out.append)
    return out


@pytest.mark.asyncio
async def test_all_six_tools_present() -> None:
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    names = {t.name for t in tools}
    assert names == {
        "emit_html_artifact",
        "emit_markdown_artifact",
        "emit_table_artifact",
        "emit_chart_artifact",
        "emit_reasoning_step",
        "cite_source",
    }


@pytest.mark.asyncio
async def test_emit_html_writes_html_artifact_payload(
    captured: list[dict[str, Any]],
) -> None:
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    html = next(t for t in tools if t.name == "emit_html_artifact")
    html.invoke({"content": "<h1>Hello</h1>", "name": "Greeting"})
    [payload] = captured
    assert payload["type"] == "artifact"
    assert payload["artifact"]["type"] == "html"
    assert payload["artifact"]["content"] == "<h1>Hello</h1>"


@pytest.mark.asyncio
async def test_emit_table_writes_table_payload(captured: list[dict[str, Any]]) -> None:
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    table = next(t for t in tools if t.name == "emit_table_artifact")
    table.invoke({"columns": ["a"], "rows": [[1], [2]], "name": "Counts"})
    [payload] = captured
    assert payload["artifact"]["columns"] == ["a"]
    assert payload["artifact"]["rows"] == [[1], [2]]


@pytest.mark.asyncio
async def test_emit_chart_writes_chart_payload(captured: list[dict[str, Any]]) -> None:
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    chart = next(t for t in tools if t.name == "emit_chart_artifact")
    chart.invoke({"plotly": {"data": [{"x": [1, 2]}]}, "name": "P"})
    [payload] = captured
    assert payload["artifact"]["type"] == "chart"
    assert payload["artifact"]["plotly"] == {"data": [{"x": [1, 2]}]}


@pytest.mark.asyncio
async def test_emit_reasoning_step_writes_step(captured: list[dict[str, Any]]) -> None:
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    step = next(t for t in tools if t.name == "emit_reasoning_step")
    step.invoke({"message": "thinking", "event_type": "INFO"})
    [payload] = captured
    assert payload["type"] == "step"
    assert payload["message"] == "thinking"


@pytest.mark.asyncio
async def test_cite_source_writes_citations(captured: list[dict[str, Any]]) -> None:
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    cite = next(t for t in tools if t.name == "cite_source")
    cite.invoke({"text": "quote", "source": "10-K", "source_url": "http://x"})
    [payload] = captured
    assert payload["type"] == "citations"
    [citation] = payload["citations"]
    assert citation["id"]
    assert citation["source_info"]["name"] == "10-K"
    assert citation["source_info"]["origin"] == "http://x"
    assert citation["details"] == [
        {"text": "quote", "url": "http://x", "title": "10-K"}
    ]


@pytest.mark.asyncio
async def test_emit_markdown_writes_markdown_payload(
    captured: list[dict[str, Any]],
) -> None:
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    md = next(t for t in tools if t.name == "emit_markdown_artifact")
    md.invoke({"content": "# Hi", "name": "doc"})
    [payload] = captured
    assert payload["artifact"]["type"] == "markdown"
    assert payload["artifact"]["content"] == "# Hi"


def test_sanitise_markdown_strips_scratchpad_heading_section() -> None:
    """Remove a Tool activity heading and its body, reporting it in notes."""
    from openbb_agent_server.plugins.tools.artifacts import _sanitise_markdown_body

    body = (
        "# Summary\n\nThe company grew revenue.\n\n"
        "## Tool activity\n\nLooked up the balance sheet.\n\n"
        "# Conclusion\n\nStrong quarter.\n"
    )
    cleaned, notes = _sanitise_markdown_body(body)
    assert "Tool activity" not in cleaned
    assert "Looked up the balance sheet" not in cleaned
    assert "Strong quarter" in cleaned
    assert any("Tool activity" in n for n in notes)


def test_sanitise_markdown_strips_inline_tool_call_lines() -> None:
    """Strip list lines that enumerate tool calls and add a note."""
    from openbb_agent_server.plugins.tools.artifacts import _sanitise_markdown_body

    body = (
        "Analysis follows.\n\n"
        "- pdf_extract(name='10K', page_range=[1, 2])\n"
        "- search_pdf(query='revenue', k=5)\n\n"
        "Final takeaway.\n"
    )
    cleaned, notes = _sanitise_markdown_body(body)
    assert "pdf_extract" not in cleaned
    assert "search_pdf" not in cleaned
    assert "Final takeaway" in cleaned
    assert any("inline tool-call line" in n for n in notes)


def test_sanitise_markdown_strips_trailing_scratchpad_section() -> None:
    """Remove a trailing scratchpad heading with no following heading."""
    from openbb_agent_server.plugins.tools.artifacts import _sanitise_markdown_body

    body = "# Analysis\n\nGood numbers.\n\n## Next steps\n\nReview the filing.\n"
    cleaned, notes = _sanitise_markdown_body(body)
    assert "Next steps" not in cleaned
    assert "Review the filing" not in cleaned
    assert "Good numbers" in cleaned
    assert notes


def test_sanitise_markdown_clean_body_returned_unchanged() -> None:
    """Return content with no scratchpad sections unchanged."""
    from openbb_agent_server.plugins.tools.artifacts import _sanitise_markdown_body

    body = "# Analysis\n\nJust the polished prose, nothing to strip.\n"
    cleaned, notes = _sanitise_markdown_body(body)
    assert cleaned == body
    assert notes == []


def test_decode_if_string_parses_json_and_passes_through() -> None:
    """Decode JSON strings and leave bad input alone."""
    from openbb_agent_server.plugins.tools.artifacts import _decode_if_string

    assert _decode_if_string('["a", "b"]') == ["a", "b"]
    assert _decode_if_string("not json") == "not json"
    assert _decode_if_string([1, 2]) == [1, 2]


@pytest.mark.asyncio
async def test_emit_markdown_appends_sanitiser_warning(
    captured: list[dict[str, Any]],
) -> None:
    """Carry a WARNING in the tool result when the body trips the sanitiser."""
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    md = next(t for t in tools if t.name == "emit_markdown_artifact")
    result = md.invoke(
        {"content": "# Report\n\nText.\n\n## Tool activity\n\nDid stuff.\n"}
    )
    assert "WARNING" in result
    [payload] = captured
    assert "Tool activity" not in payload["artifact"]["content"]


@pytest.mark.asyncio
async def test_emit_html_appends_sanitiser_warning(
    captured: list[dict[str, Any]],
) -> None:
    """Surface the sanitiser WARNING suffix from the HTML emitter."""
    src = ArtifactsToolSource()
    tools = await src.tools(_ctx(), {})
    html = next(t for t in tools if t.name == "emit_html_artifact")
    result = html.invoke({"content": "# Report\n\nText.\n\n## Next steps\n\nReview.\n"})
    assert "WARNING" in result


def test_artifacts_decode_if_string_passes_through_non_string_inputs() -> None:
    from openbb_agent_server.plugins.tools.artifacts import _decode_if_string

    sentinel = {"already": "decoded"}
    assert _decode_if_string(sentinel) is sentinel
    assert _decode_if_string("not json") == "not json"
