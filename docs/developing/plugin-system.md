# Plugin system

Every swappable piece of the runtime is a plugin discovered via Python entry points. Six plugin groups, ABCs / protocols in `runtime/plugins.py`:

| Group | ABC | Built-ins |
| --- | --- | --- |
| `openbb_agent_server.auth` | `AuthBackend` | `none`, `bearer_static`, `api_key_table`, `oidc_jwt`, `openbb_workspace` |
| `openbb_agent_server.models` | `ModelProvider` | `anthropic`, `openai`, `openai_compat`, `bedrock`, `vertex`, `google_genai`, `groq`, `nvidia`, `fake` |
| `openbb_agent_server.tools` | `ToolSource` | every shipped tool source |
| `openbb_agent_server.middleware` | `Middleware` | `call_limit`, `tool_call_limit`, `tool_call_announcer`, `tool_call_ledger`, `tool_filter`, `tool_message_normaliser`, `loop_guard`, `usage_recorder` |
| `openbb_agent_server.subagents` | `SubAgentSpec` (Protocol) | `researcher`, `analyst`, `charter`, `pdf_reader` |
| `openbb_agent_server.checkpointers` | `CheckpointerProvider` | `sqlite`, `postgres`, `inmemory` |

## Discovery

`runtime/registry.py::load(group, name, config=None)` calls `importlib.metadata.entry_points(group=group)`, finds the matching entry, calls `entry.load()`, instantiates with no args, returns the instance. `config` is passed to `build(ctx, config)` for plugins that take per-call config.

No global plugin cache — entry points resolve fresh each call.

## Declaring a plugin in `pyproject.toml`

```toml
[project.entry-points."openbb_agent_server.tools"]
my_tool = "my_package.my_module:MyToolSource"
```

## ABCs

### `AuthBackend`

```python
class AuthBackend(ABC):
    name: str
    async def authenticate(self, request: Request) -> UserPrincipal: ...
```

Resolves the principal from request headers. Raises `HTTPException` (401 / 403) on rejection. See [Writing an auth backend](writing-an-auth-backend.md).

### `ModelProvider`

```python
class ModelProvider(ABC):
    name: str
    def build(self, ctx: RunContext, config: dict[str, Any]) -> BaseChatModel: ...
```

Returns a LangChain chat model exposing `astream`, `ainvoke`, `bind_tools`. See [Writing a model provider](writing-a-model-provider.md).

### `ToolSource`

```python
class ToolSource(ABC):
    name: str
    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]: ...
```

Returns `langchain_core.tools.BaseTool` instances. Best practice: `StructuredTool.from_function(coroutine=fn, args_schema=…)`. See [Writing a tool source](writing-a-tool-source.md).

### `Middleware`

```python
class Middleware(ABC):
    name: str
    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware: ...
```

Returns a `langchain.agents.middleware.types.AgentMiddleware`. Hookpoints: `wrap_model_call` and `wrap_tool_call` (plus their async `awrap_*` variants). See [Writing a middleware](writing-a-middleware.md).

### `SubAgentSpec` (Protocol, not ABC)

```python
class SubAgentSpec(Protocol):
    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]
    model: str | None
```

Any class with these attributes is a valid spec — no inheritance required. `tools` lists tool names inherited from the parent. See [Writing a sub-agent](writing-a-subagent.md).

### `CheckpointerProvider`

```python
class CheckpointerProvider(ABC):
    name: str
    async def open(self, settings: Any) -> Any: ...
    async def close(self, saver: Any) -> None: ...
```

Builds the LangGraph checkpointer used for resume / replay / HITL. Default is `sqlite`.

## Selection

`[agent]` selects plugins by name:

```toml
[agent]
tool_sources = ["artifacts", "web_search", "widget_data", "inspect_widget_data", "pdf_extract"]
middleware = ["tool_call_announcer", "tool_call_ledger", "usage_recorder", "call_limit"]
subagents = ["researcher", "pdf_reader"]
checkpointer_provider = "sqlite"

[agent.auth]
backend = "oidc_jwt"

[agent.model]
provider = "nvidia"
name = "nvidia/nemotron-3-super-120b-a12b"

[agent.tool_source_config.web_search]
backend = "tavily"
```

Per-plugin kwargs go in `tool_source_config[<name>]`, `embeddings_config`, `reranker_config`, etc.

## RunContext access

```python
from openbb_agent_server.runtime import context as run_context
ctx = run_context.current()
# .principal, .trace_id, .run_id, .conversation_id, .agent_name,
# .timezone, .widgets, .uploaded_files, .api_keys, .api_urls, .tools,
# .workspace_options
```

## Emitting events

```python
from openbb_agent_server.runtime import emit

emit.reasoning_step("Fetching data", source="widget")
emit.table_artifact(columns=["sym", "px"], rows=[["AAPL", 178.4]], name="quotes")
emit.cite(text="…", source="Reuters", source_url="https://reuters.com/...")
```

Frames flow through LangGraph's `get_stream_writer` and get translated to Workspace SSE by `protocol/adapter.py`. Outside a LangGraph context (e.g. unit tests), the helpers no-op.

## Testing a plugin

```python
import pytest
from openbb_agent_server.runtime.context import RunContext, bind
from openbb_agent_server.runtime.principal import UserPrincipal
from my_package.my_module import MyToolSource

@pytest.mark.asyncio
async def test_my_tool() -> None:
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t", run_id="r", conversation_id="c",
    )
    with bind(ctx):
        tools = await MyToolSource().tools(ctx, {})
        [tool] = [t for t in tools if t.name == "do_thing"]
        result = await tool.ainvoke({"arg": 1})
    assert result == "expected"
```

## Source

- [`runtime.plugins`](../reference/runtime/plugins.md) — the ABCs.
- [`runtime.registry`](../reference/runtime/registry.md) — discovery.
- [`runtime.context`](../reference/runtime/context.md) — `RunContext` + `bind`.
- [`runtime.emit`](../reference/runtime/emit.md) — Workspace-event helpers.
