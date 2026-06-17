"""Researcher subagent."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are a research subagent. Use the available tools to retrieve
authoritative sources for the user's question (filings, primary
documents, news), summarise crisply, and ALWAYS attach citations
(``source_url`` or named ``source``) to every factual claim.

If a tool fails or returns nothing, say so explicitly — do not invent.
"""


def factory(**_config: Any) -> dict[str, Any]:
    """Return the researcher subagent declaration as a dict.

    Parameters
    ----------
    **_config : Any
        Configuration keys, accepted and ignored; the declaration is
        static.

    Returns
    -------
    dict of str to Any
        Mapping with ``name``, ``description``, and ``system_prompt``
        keys describing the researcher subagent.
    """
    return {
        "name": "researcher",
        "description": (
            "Use when the user asks for facts, sources, filings, news, or "
            "any answer that requires up-to-date retrieval and citation."
        ),
        "system_prompt": SYSTEM_PROMPT,
    }


class ResearcherSubAgent:
    """Declare the researcher subagent as class attributes.

    A retrieval-focused subagent that gathers authoritative sources and
    attaches citations to every factual claim.

    Attributes
    ----------
    name : str
        Subagent identifier, ``"researcher"``.
    description : str
        When-to-use guidance surfaced to the routing agent.
    system_prompt : str
        The system prompt instructing citation-backed retrieval.
    tools : tuple of str
        Allowed tool names; empty means inherit the caller's tools.
    model : str or None
        Model override, or ``None`` to use the default model.
    """

    name = "researcher"
    description = (
        "Use when the user asks for facts, sources, filings, news, or any "
        "answer that requires up-to-date retrieval and citation."
    )
    system_prompt = SYSTEM_PROMPT
    tools: tuple[str, ...] = ()
    model: str | None = None
