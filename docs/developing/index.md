# Developer guides

For people extending or customising the agent server — writing plugins, hooking into the runtime, contributing back. Every swappable piece of the runtime resolves through one of six entry-point groups; the guides below walk through each ABC plus the standing conventions and the test harness.

## Pages

### [Plugin system](plugin-system.md)
The six entry-point groups (`openbb_agent_server.auth`, `.model_providers`, `.tools`, `.middleware`, `.subagents`, `.checkpointers`) with their ABCs / protocols in `runtime/plugins.py`, how `runtime/registry.py::load(group, name, config)` discovers and instantiates plugins fresh per call (no global cache), how to declare entry points in `pyproject.toml`, and what the built-in roster looks like for each group. Start here before writing any plugin.

### [Writing a tool source](writing-a-tool-source.md)
The `ToolSource` ABC, building `StructuredTool` instances with pydantic arg schemas, the foreground / `submit_*` background variant pattern, emitting status steps and citations via `runtime.emit`, reading uploaded files through `ctx.uploaded_files`, registering through entry points, and the soft-skip pattern for tools that need optional credentials.

### [Writing a model provider](writing-a-model-provider.md)
The `ModelProvider` ABC, returning a `langchain_core.language_models.BaseChatModel`, reading API keys from `ctx.api_keys` first then the environment, optional rate limiting / retry wrapping, the soft-skip path when the requested provider isn't installed, and how the model gets bound into the DeepAgents loop via `create_deep_agent(model=…)`.

### [Writing a middleware](writing-a-middleware.md)
The `Middleware` ABC over LangChain's `AgentMiddleware`, the two hookpoints — `wrap_model_call(request, handler)` for the chat completion and `wrap_tool_call(request, handler)` for tool invocations — ordering semantics within the configured `middleware` list, and worked patterns (redactors, ledgers, call limits, tool filtering, response normalisation).

### [Writing a sub-agent](writing-a-subagent.md)
The `SubAgentSpec` `Protocol` (`name`, `description`, `system_prompt`, `tools`, `model`, `model_profile`), tool inheritance from the parent profile, picking a different model than the host agent (including delegating to a configured profile via `model_profile`), and registration through entry points. Protocol-based so no inheritance is required — any class with the right attributes is a valid spec.

### [Writing an auth backend](writing-an-auth-backend.md)
The `AuthBackend` ABC, the `UserPrincipal` shape (`user_id`, `display_name`, `email`, `scopes`, `raw_claims`), scope conventions (`agent:query`, `memory:read`, `memory:write`, `admin`), header / cookie / JWT / API-key implementation patterns, and how to use `runtime/identity.py::hash_user_id` to keep raw emails out of persisted rows.

### [Testing](testing.md)
The 1300+-test layout (`tests/conftest.py`, `tests/fixtures/`, `tests/test_*.py`), the rule that fixtures are real `pip install -e .` openbb extension packages (no `SimpleNamespace` mock for `obb`), the `alice` / `bob` principal fixtures, `history` and `settings_env` fixtures, async patterns, the 100% line + branch coverage standard, when `# pragma: no cover` is acceptable, and the live-test gating.

## See also

- [API reference](../reference/index.md) — symbol-level mirror of the package tree.
- [`guides/`](../guides/index.md) — what the agent does end-to-end, useful before you start building.
- [`operating/`](../operating/index.md) — configuration, persistence, observability that your plugins will plug into.
- [`docs/index.md`](../index.md) — the parent index, including the full built-in plugin roster.
