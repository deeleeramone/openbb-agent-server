# Writing a sub-agent

A sub-agent is a focused agent with its own system prompt and tool set, invoked by the main agent via a `task` tool. The spec is a `Protocol` — any class with the right attributes is a valid sub-agent; no inheritance required.

```python
from typing import Protocol

class SubAgentSpec(Protocol):
    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]
    model: str | None
    model_profile: str | None
```

## Minimal example

```python
"""PDF-reader sub-agent — focused prompt over the pdf_extract tool."""

from __future__ import annotations


class PdfReaderSubAgent:
    name = "pdf_reader"
    description = (
        "Extract text and bounding boxes from one or more uploaded PDFs. "
        "Returns markdown with page citations. Use when the user asks a "
        "question grounded in a PDF they uploaded."
    )
    system_prompt = (
        "You read PDFs and answer questions from them. Use the "
        "``pdf_extract`` tool for each PDF in scope. For every claim, "
        "call ``cite_source`` with the page number and the smallest "
        "bounding box that supports it. Refuse to answer if the relevant "
        "text isn't in the PDFs you can see."
    )
    tools: tuple[str, ...] = ("pdf_extract", "cite_source")
    model: str | None = None
```

Note: `SubAgentSpec` is a `Protocol`, not a base class — duck-typing only. Inheriting from it adds nothing and is intentionally absent in the bundled subagents. The shipped `plugins/subagents/pdf_reader.py` follows this same pattern, with `tools = ("pdf_extract",)` and a prompt that emits citations through the `custom` stream channel instead of a `cite_source` tool call. (`model_profile` is optional in practice — the builder checks for it with `hasattr` — so a spec that omits the attribute entirely, like `pdf_reader`, is still valid.)

Register it:

```toml
[project.entry-points."openbb_agent_server.subagents"]
pdf_reader = "my_package.pdf_reader:PdfReaderSubAgent"
```

Reference it from the default profile (`[agent]`) or a named one (`[agent.profiles.<name>]`):

```toml
[agent]
subagents = ["pdf_reader", "researcher"]
```

## How DeepAgents uses the spec

`runtime/builder.py::_resolve_subagents` translates each spec into the dict shape DeepAgents wants:

```python
{
    "name": spec.name,
    "description": spec.description,
    "system_prompt": spec.system_prompt,
    "tools": [main_tool_by_name[t] for t in spec.tools if t in main_tool_by_name],
    "model": spec.model,   # optional
}
```

The sub-agent's tools must already exist on the main agent (the same `tool_sources` from the profile). Tool names not found are silently dropped. This means a sub-agent inherits tools from the parent rather than getting its own `tool_sources` config.

The agent loop sees a new `task` tool that takes a sub-agent name and a free-form description. DeepAgents handles the dispatch: spin up a focused mini-agent with the sub-agent's prompt + tool set, run it to completion, return its summary to the parent.

## Picking the model

`spec.model` controls which chat model the sub-agent uses:

| Value | Behaviour |
| --- | --- |
| `None` (default) | inherit the parent's model |
| `str` | provider name — re-resolves through the registry |
| `BaseChatModel` instance | use this exact object |

Alternatively, set `model_profile` to the name of a configured agent profile (`[agent.profiles.<name>]`): the builder resolves that profile's provider + model at build time and runs the sub-agent on it. This is how the bundled model-profile subagents (`deepseek-v4-flash`, `nemotron-3-super`, `qwen3.5`, …) delegate to specialist models; see `plugins/subagents/model_profile.py`.

For a cheap sub-agent (e.g., a "summariser" running over already-fetched text), point at a smaller model:

```python
class SummariserSpec(SubAgentSpec):
    name = "summariser"
    description = "..."
    system_prompt = "..."
    tools = ()
    @property
    def model(self) -> Any:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5-20251001")
```

## When to use a sub-agent vs a tool

| Use a sub-agent | Use a tool |
| --- | --- |
| Multi-step reasoning over its own state (research, drafting, multi-PDF synthesis) | Single deterministic operation (HTTP fetch, SQL query, file read) |
| Wants its own focused prompt | Just executes a function |
| Should be able to call multiple tools in its own loop | One-shot in / out |

Rule of thumb: if you'd write the description as a paragraph of instructions, it's a sub-agent. If it's "execute X and return the result", it's a tool.

## Tool inheritance gotcha

Because tools come from the main `tool_sources`, you can't ship a sub-agent that uses a tool not listed in the parent profile. Either:

- Add the tool to the parent profile (it'll show up on the parent agent too — usually fine), or
- Ship the tool as its own `ToolSource` in the same plugin package and require operators to list both.

## Tests

Spec-level unit tests are tiny:

```python
def test_pdf_reader_spec_defaults() -> None:
    spec = PdfReaderSubAgent()
    assert spec.name == "pdf_reader"
    assert "pdf_extract" in spec.tools
```

End-to-end behavioural tests are harder because they require the agent to choose `task(name="pdf_reader", …)`. The pragmatic alternative: integration tests against the parent agent that assert the sub-agent gets invoked when the input warrants it (e.g., "summarise this 10-K" → expect `pdf_reader` invocation in `tool_calls`).

## Source

- [`runtime.plugins`](../reference/runtime/plugins.md)
- [`runtime.builder`](../reference/runtime/builder.md) — `_resolve_subagents`.
- Worked examples: every file under [`plugins/subagents/`](../reference/plugins/subagents/index.md).
