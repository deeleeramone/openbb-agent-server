# Writing a tool source

A `ToolSource` produces one or more `langchain_core.tools.BaseTool` instances. The agent loop calls them via DeepAgents; the runtime ensures the right context, the right structured types, and right observability.

## Minimal example

```python
"""Hello-world tool source."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource


class _GreetArgs(BaseModel):
    name: str = Field(description="Who to greet.")


class HelloToolSource(ToolSource):
    """Single ``greet`` tool — returns ``Hello, {name}!``."""

    name = "hello"

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        def greet(name: str) -> str:
            emit.reasoning_step("greet", target=name)
            return f"Hello, {name}!"

        return [
            StructuredTool.from_function(
                func=greet,
                name="greet",
                description="Greet someone by name.",
                args_schema=_GreetArgs,
            )
        ]
```

Register it:

```toml
# pyproject.toml in the plugin package
[project.entry-points."openbb_agent_server.tools"]
hello = "my_package.hello:HelloToolSource"
```

Add `hello` to a profile's `tool_sources` and restart the server.

## When to register tools

`tools(ctx, config)` is called **once per run**. The returned list is wired into `create_deep_agent(tools=…)` and stays for the rest of the agent loop. Don't try to dynamically add or remove tools mid-run — DeepAgents doesn't see them.

If a tool depends on per-call config (an API key, a NIM model name, a base URL), close over those values inside `tools()` and capture them in the closures you return:

```python
async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
    api_key = ctx.api_keys.get("MY_API_KEY") or config.get("api_key") or os.environ.get("MY_API_KEY")
    if not api_key:
        logger.warning("hello: no API key; skipping registration")
        return []
    base_url = config.get("base_url", "https://api.example.com")

    def call_api(payload: str) -> str:
        return _post(base_url, payload, api_key=api_key)

    return [StructuredTool.from_function(...)]
```

## Soft-skip pattern

If a tool source can't run (missing key, missing extra), **return an empty list** with a single `logger.warning` rather than raising. The agent simply won't see those tools, and the rest of the profile loads cleanly. This is how `vision_qa`, `paligemma_vision`, `gemma_audio` behave when `NVIDIA_API_KEY` is unset.

## Async vs sync tools

`StructuredTool.from_function` takes either:

- `func=` — a sync callable. LangChain runs it in a thread pool. Fine for fast / pure / CPU-bound work.
- `coroutine=` — an async callable. The agent loop awaits it directly. Required when the tool needs `await asyncio.to_thread`, `httpx`, or any other async API.

**Don't combine async and sync APIs in one function**. If your body does `await ...`, register it as `coroutine=`. If LangChain calls `func=` in its thread pool, that thread has no event loop and `asyncio.run` / `asyncio.get_event_loop` will fail.

## Reading `RunContext` inside a tool body

`ctx` is the run-bound `RunContext`. Inside the tool body, `run_context.current()` returns the same object. Use it for:

```python
from openbb_agent_server.runtime import context as run_context

def list_my_files() -> list[dict]:
    ctx = run_context.current()
    return [
        {"name": f.name, "mime": f.mime}
        for f in ctx.uploaded_files
        if f.mime and f.mime.startswith("image/")
    ]
```

## Emitting reasoning + artifacts

```python
emit.reasoning_step("fetching", url=url, model=model)       # → StatusUpdateSSE INFO
emit.reasoning_step("rate-limited", event_type="WARNING")    # → StatusUpdateSSE WARNING
emit.table_artifact(columns=["a","b"], rows=[[1,2]], name="t")    # → MessageArtifactSSE table
emit.chart_artifact(plotly={"data": [...]}, name="px")       # → MessageArtifactSSE chart
emit.html_artifact(content="<div>…</div>", name="report")
emit.markdown_artifact(content="# title\n…", name="notes")
emit.image_artifact(data_base64="...", mime="image/png")     # rendered inline as HTML
emit.file_artifact(url="...", name="report.pdf")             # rendered as download link
emit.cite(text="…", source="Reuters", source_url="https://…")
```

All helpers no-op when called outside a LangGraph stream context, so unit tests don't need to mock anything.

## Background variants

Long-running tools should ship both `foo` and `submit_foo` variants. The pattern (mirrored across `vision_qa`, `paligemma_vision`, `gemma_audio`, `groq_audio`):

```python
async def submit_foo(arg: str) -> dict[str, Any]:
    from openbb_agent_server.runtime.jobs import get_registry
    job_id = get_registry().submit(
        lambda: foo(arg),
        label=f"foo({arg})",
        metadata={"tool": "foo", "arg": arg},
    )
    emit.reasoning_step("submit_foo", job_id=job_id, source=arg)
    return {"job_id": job_id, "label": f"foo({arg})"}
```

Then register both via `StructuredTool.from_function(coroutine=foo, …)` and `coroutine=submit_foo`. See [Background jobs](../guides/background-jobs.md).

## Server-side vs client-side tools

| Kind | Where it runs | How to write it |
| --- | --- | --- |
| Server-side | inside the agent loop, in this process | normal `StructuredTool.from_function`; returns a value |
| Client-side | inside the Workspace UI | a stub `StructuredTool` whose name starts with `client:` (or whose call raises `PendingClientToolCall`); the runtime emits `FunctionCallSSE` and the next request brings the result back |

The `client_side` tool source is the entry point for the latter. The `workspace_mcp` tool source forwards the user's enabled MCP tools by the same mechanism.

## Tests

Three layers:

1. **Unit** — instantiate the source directly, call `.tools(ctx, config)`, invoke each tool with crafted args, assert output. See `tests/test_tool_vision_qa.py`.
2. **Integration** — mount the FastAPI app with the source active, POST `/v1/query`, assert SSE event sequence. See `tests/test_router_query_integration.py`.
3. **Live** (gated) — real network calls. Gate with `pytest.mark.skipif(not os.environ.get("NVIDIA_API_KEY"), …)`. See `tests/test_nim_integration.py`.

100% line + branch coverage is the project's standard; tools should ship with full coverage of every soft-skip / error path.

## Source

- [`runtime.plugins`](../reference/runtime/plugins.md)
- [`runtime.emit`](../reference/runtime/emit.md)
- [`runtime.jobs`](../reference/runtime/jobs.md)
- Worked examples: every file under [`plugins/tools/`](../reference/plugins/tools/index.md).
