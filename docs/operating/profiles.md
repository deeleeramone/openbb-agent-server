# Profiles

One process hosts one default profile plus any number of named overrides. Each profile shows up as a separately-selectable agent in Workspace.

## URL routing

| URL | Profile |
| --- | --- |
| `POST /v1/query` | `default_profile` (default name: `"default"`) |
| `POST /agents/{name}/v1/query` | profile named `{name}` |
| `GET /agents.json` | every profile present |

Profile names must match `^[a-z0-9-]+$` — the Workspace spec. The router silently drops any profile whose name doesn't match.

## Shape

A profile is a frozen `AgentProfile`:

```python
class AgentProfile(BaseSettings):
    name: str
    metadata: AgentMetadata
    model_provider: str
    model_name: str
    model_config_: dict[str, Any]      # alias "model_config"
    tool_sources: tuple[str, ...]
    tool_source_config: dict[str, dict[str, Any]]
    subagents: tuple[str, ...]
    middleware: tuple[str, ...]
    skills: tuple[str, ...]
    features: dict[str, Any]
    system_prompt_file: str | None
```

## Declaring profiles

Top-level `[agent]` sets the defaults. `[agent.profiles.<name>]` adds an overlay that overrides selected keys.

```toml
[agent]
default_profile = "default"

[agent.metadata]
name = "OpenBB Agent"
description = "General-purpose research assistant."

[agent.model]
provider = "nvidia"
name = "nvidia/nemotron-3-super-120b-a12b"

# Named profile inheriting defaults, swapping model + trimming tools.
[agent.profiles.fast]
[agent.profiles.fast.metadata]
name = "OpenBB · Fast"
description = "Smaller model, web search only."

[agent.profiles.fast.model]
provider = "anthropic"
name = "claude-haiku-4-5-20251001"

[agent.profiles.fast]
tool_sources = ["web_search", "rerank"]
```

`agents.json` advertises two top-level keys (`default`, `fast`).

## Per-profile system prompt

```toml
[agent.profiles.fast]
system_prompt_file = "/etc/openbb/prompts/fast.md"
```

Inline `system_prompt` strings are rejected — the loader raises with a clear message. The bundled default is at `openbb_agent_server/prompts/default_system_prompt.md`.

Templating: prompts can include `{timezone}`, `{today}`, `{widget_snapshot}`, `{file_snapshot}`. Unknown `{name}` placeholders round-trip unchanged.

## Features

`features` becomes the `features` map in `/agents.json`. Reserved boolean keys:

| Key | Effect when `true` |
| --- | --- |
| `streaming` | tokens stream chunk-by-chunk |
| `widget-dashboard-select` | Workspace passes pinned widgets in `widgets.primary` |
| `widget-dashboard-search` | Workspace lets the agent search across all dashboard widgets |
| `widget-global-search` | Workspace exposes global search across the user's widget catalogue |
| `file-upload` | Workspace allows file uploads on the request |
| `generative-ui` | Workspace renders client-side generative UI from the agent's output |

Custom features take a dict shape:

```toml
[agent.profiles.default.features.deep-research]
label = "Deep Research"
description = "Branch into the researcher sub-agent before drafting."
default = false
```

The names `web search`, `web-search`, and `websearch` are reserved — use `search-web` (kebab-case) for the web-search toggle.

## Selecting plugins

`tool_sources`, `middleware`, `subagents` are lists of plugin names. Loading is fail-fast — a typo'd name raises at startup, not at first request.

## Reading the active profile inside a plugin

```python
from openbb_agent_server.runtime import context as run_context
ctx = run_context.current()
profile_name = ctx.agent_name
```

Don't cache settings — they're loaded fresh per call so the operator can change `openbb.toml` and have the next run pick it up.

## Source

- [`app/settings.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/app/settings.py) — `AgentProfile`, `AgentServerSettings.resolve_profile`.
- [`app/router.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/app/router.py) — `_registration`, `_run_query`.
