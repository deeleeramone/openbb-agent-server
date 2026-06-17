"""Subagent declaration tests — purely shape-checking the spec classes."""

from __future__ import annotations

import pytest

from openbb_agent_server.plugins.subagents.analyst import AnalystSubAgent
from openbb_agent_server.plugins.subagents.charter import CharterSubAgent
from openbb_agent_server.plugins.subagents.pdf_reader import PdfReaderSubAgent
from openbb_agent_server.plugins.subagents.researcher import (
    ResearcherSubAgent,
    factory,
)


@pytest.mark.parametrize(
    "spec_cls",
    [ResearcherSubAgent, CharterSubAgent, AnalystSubAgent, PdfReaderSubAgent],
)
def test_subagent_specs_have_required_fields(spec_cls: type) -> None:
    assert spec_cls.name
    assert spec_cls.description
    assert spec_cls.system_prompt
    assert isinstance(spec_cls.tools, tuple)


def test_pdf_reader_advertises_pdf_extract_tool() -> None:
    assert "pdf_extract" in PdfReaderSubAgent.tools


def test_researcher_module_factory_returns_a_dict() -> None:
    spec = factory()
    assert spec["name"] == "researcher"
    assert "system_prompt" in spec
