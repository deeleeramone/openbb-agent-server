# Writing a model provider

A `ModelProvider` returns a `langchain_core.language_models.BaseChatModel` instance. The runtime wires the model into DeepAgents via `create_deep_agent(model=…)`, then the agent loop drives it through `astream` / `ainvoke` / `bind_tools`.

## Minimal example

```python
"""Anthropic model provider."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider


class AnthropicProvider(ModelProvider):
    """Anthropic Claude via ``langchain_anthropic``."""

    name = "anthropic"

    def build(self, ctx: RunContext, config: dict[str, Any]) -> BaseChatModel:
        api_key = (
            ctx.api_keys.get("ANTHROPIC_API_KEY")
            or config.get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        kwargs: dict[str, Any] = {
            "model": config.get("model_name") or "claude-opus-4-7",
            "api_key": api_key,
        }
        if (max_tokens := config.get("max_tokens")):
            kwargs["max_tokens"] = int(max_tokens)
        if (temperature := config.get("temperature")) is not None:
            kwargs["temperature"] = float(temperature)
        return ChatAnthropic(**kwargs)
```

Register it:

```toml
[project.entry-points."openbb_agent_server.model_providers"]
anthropic = "my_package.anthropic:AnthropicProvider"
```

Select it on the default profile (top-level `[agent]`) or on a named profile under `[agent.profiles.<name>]`:

```toml
[agent.model]
provider = "anthropic"
name = "claude-opus-4-7"

[agent.model.config]
max_tokens = 4096
temperature = 0.4

# named profile overrides
[agent.profiles.fast.model]
provider = "anthropic"
name = "claude-haiku-4-5-20251001"
```

## Contract

`build(ctx, config)` runs **once per run**. The returned object is bound for the rest of the agent loop. Don't try to reconfigure it mid-run.

Whatever `BaseChatModel` you return must:

- Implement `bind_tools(tools)` — DeepAgents binds the tool list before streaming.
- Implement `astream(messages)` — the runtime streams tokens via this.
- Surface `usage_metadata` on the final `AIMessage` — `UsageRecorder` middleware reads it.

Every LangChain v1 chat model adapter implements all three.

## Per-call vs per-profile config

The merged `config` dict comes from the profile's `[agent.model.config]` TOML table (the `model_config_` attribute on `AgentProfile` — trailing underscore because Pydantic reserves `model_config`) + any per-request overrides safely allowed by your profile. Common keys:

| Key | Meaning |
| --- | --- |
| `model_name` | the model id to send to the provider |
| `api_key` | overrides env / `ctx.api_keys` |
| `base_url` | provider endpoint override (Azure / OpenRouter / Bedrock VPC / etc.) |
| `temperature`, `top_p`, `max_tokens` | sampling knobs |
| `extra_headers` | provider-specific (org id, project id, …) |

Be defensive: callers can set any key, so cast / validate before forwarding.

## API-key precedence

Same pattern across every built-in:

```
config.api_key
  → ctx.api_keys["X_API_KEY"]      # forwarded by Workspace
  → os.environ.get("X_API_KEY")    # operator-set
```

Never raise if the key is missing — return a `BaseChatModel` that will fail loudly on first call instead. (Or, if your model strictly requires a key at construction time, raise a clear `RuntimeError` with installation hints.)

## Rate-limit-aware providers

Groq has tight per-minute caps. The Groq provider uses a token-bucket limiter (`plugins/models/groq_rate_limiter.py`) that wraps the underlying model so the agent loop sleeps rather than getting 429'd. If your provider has similar constraints, see that file as the pattern — it's a `BaseChatModel` proxy that delegates to the wrapped model after acquiring tokens.

## Sub-agent models

Sub-agents can specify their own model in `SubAgentSpec.model`. DeepAgents accepts either a string (provider name — falls back to `build` again) or a fully-constructed `BaseChatModel`. Returning a fully-constructed model is typically simpler for sub-agents that use a smaller / faster model than the main loop.

## Soft-skip

Unlike tool sources, a model provider failing to load is fatal — there's no run without a model. Don't soft-skip; raise a clear error.

## Tests

A model provider is a thin shim, so unit tests are simple:

```python
def test_build_returns_chat_anthropic_with_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    provider = AnthropicProvider()
    ctx = RunContext(principal=UserPrincipal(user_id="u"), trace_id="t", run_id="r", conversation_id="c")
    model = provider.build(ctx, {"model_name": "claude-opus-4-7", "max_tokens": 128})
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-opus-4-7"
    assert model.max_tokens == 128
```

For real streaming behaviour, mock the underlying LangChain class — don't stand up a fake HTTP server.

## Source

- [`runtime.plugins`](../reference/runtime/plugins.md)
- Worked examples: every file under [`plugins/models/`](../reference/plugins/models/index.md).
