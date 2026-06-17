"""Analyst subagent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a data-analyst subagent. Given a dataset reference, compute
descriptive statistics, group-bys, deltas, or cross-tabs the user asked
for. Emit results as a single table artifact via the ``custom`` stream
channel:

  {"type": "artifact", "artifact_type": "table", "title": "...",
   "data": {"columns": [...], "rows": [...]}}

Keep the markdown reply terse — numbers go in the table.
"""


class AnalystSubAgent:
    """Declare the data-analyst subagent and its routing metadata.

    A static configuration object that registers the ``analyst``
    subagent. Its ``description`` drives delegation when the user asks for
    descriptive statistics, group-bys, deltas, or other tabular numeric
    summaries, and its ``system_prompt`` instructs the model to emit
    results as a single ``table`` artifact on the ``custom`` stream
    channel.

    Attributes
    ----------
    name : str
        Subagent identifier, ``"analyst"``.
    description : str
        Routing hint describing when this subagent should be selected.
    system_prompt : str
        System prompt governing the subagent's analysis and output
        format.
    tools : tuple[str, ...]
        Names of tools available to the subagent; empty by default.
    model : str or None
        Override model identifier, or None to use the default model.
    """

    name = "analyst"
    description = (
        "Use when the user asks for descriptive statistics, group-bys, "
        "deltas, or any tabular numeric summary of a dataset reference."
    )
    system_prompt = SYSTEM_PROMPT
    tools: tuple[str, ...] = ()
    model: str | None = None
