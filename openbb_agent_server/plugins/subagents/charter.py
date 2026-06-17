"""Charter subagent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a charting subagent. Given a dataset reference (a widget id, a
``read_file`` path, or inline JSON the user pasted), produce a single
chart artifact via the ``custom`` stream channel with shape:

  {"type": "artifact", "artifact_type": "chart", "title": "...",
   "data": {"plotly": <plotly figure JSON>}}

Pick chart type from the data shape (line for time series, bar for
categorical comparisons, scatter for two numeric axes). Keep titles
short. Never inline the full data array into the chat reply — emit it
only as part of the artifact.
"""


class CharterSubAgent:
    """Subagent that turns a referenced dataset into one chart artifact.

    Declares the static configuration the orchestrator needs to register
    and route to the charting subagent. The agent reads a dataset
    reference (widget id, ``read_file`` path, or inline JSON), picks a
    chart type from the data shape, and emits a single Plotly chart
    artifact on the ``custom`` stream channel.

    Attributes
    ----------
    name : str
        Stable identifier used to select and route to this subagent.
    description : str
        Human-readable trigger guidance shown to the routing agent.
    system_prompt : str
        Instructions given to the model defining the charting behavior
        and the required artifact shape.
    tools : tuple of str
        Names of tools exposed to the subagent. Empty: it needs none.
    model : str or None
        Override model identifier, or ``None`` to use the default model.
    """

    name = "charter"
    description = (
        "Use when the user asks for a chart / visualisation / plot of data "
        "they've referenced (a widget, an uploaded spreadsheet, or a tool result)."
    )
    system_prompt = SYSTEM_PROMPT
    tools: tuple[str, ...] = ()
    model: str | None = None
