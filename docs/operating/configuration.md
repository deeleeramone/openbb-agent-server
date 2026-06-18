# Configuration

Settings come from three sources, highest wins:

1. CLI flags (`--host`, `--port`, `--auth`, `--model-provider`, `--model-name`).
2. Environment variables prefixed `OPENBB_AGENT_`.
3. `openbb.toml` `[agent]` section.

Unset keys fall through to built-in defaults.

## Generating a config

```sh
openbb-agent-server --generate-config openbb.toml
```

Writes a working `openbb.toml` to the path you supply (or stdout with no path). Pass `--config-file <path>` to use a non-default location.

## Top-level keys

| Key | Default | Env var |
| --- | --- | --- |
| `host` | `127.0.0.1` | `OPENBB_AGENT_HOST` |
| `port` | `6900` | `OPENBB_AGENT_PORT` |
| `default_profile` | `default` | `OPENBB_AGENT_DEFAULT_PROFILE` |
| `db_url` | unset → `sqlite+aiosqlite:///$data_dir/history.db` | `OPENBB_AGENT_DB_URL` |
| `data_dir` | `~/.openbb_platform/agent` | `OPENBB_AGENT_DATA_DIR` |
| `checkpointer_provider` | `sqlite` | `OPENBB_AGENT_CHECKPOINTER_PROVIDER` |

## Retention / pruning

Left unbounded, `checkpoints.db` and `history.db` grow forever — the LangGraph checkpointer writes a full state snapshot every super-step. These keys cap that; see [persistence.md](persistence.md#retention--pruning).

| Key | Default | Env var | Purpose |
| --- | --- | --- | --- |
| `checkpoint_keep_last` | `1` | `OPENBB_AGENT_CHECKPOINT_KEEP_LAST` | Checkpoints kept per conversation thread. Only the latest is needed to resume; this is the main size control. |
| `checkpoint_retention_days` | `14` | `OPENBB_AGENT_CHECKPOINT_RETENTION_DAYS` | Drop checkpoint threads whose trace is older than this. `null` disables the age pass. |
| `history_retention_days` | `90` | `OPENBB_AGENT_HISTORY_RETENTION_DAYS` | Drop history rows (traces, messages, tool calls, usage, artifacts, citations, widget data, PDF ingest) older than this. `null` disables. |
| `prune_interval_hours` | `24` | `OPENBB_AGENT_PRUNE_INTERVAL_HOURS` | Cadence of the in-process retention sweep (one runs at startup). `0` disables the sweep — prune stays available via the CLI. |

Run a prune on demand: `openbb-agent-server prune` (see CLI flags below).

## Auth

| Key | Default |
| --- | --- |
| `auth.backend` | `none` |
| `auth.config` | `{}` |

See [auth.md](auth.md) for per-backend config keys.

## Model

| Key | Default |
| --- | --- |
| `model.provider` | `nvidia` |
| `model.name` | `nvidia/nemotron-3-super-120b-a12b` |
| `model.config` | `{temperature=0.4, max_completion_tokens=8192, top_p=0.95}` |

Other providers: `anthropic`, `openai`, `openai_compat`, `bedrock`, `vertex`, `google_genai`, `groq`, `fake`. Each is an optional install extra; see [plugin-system.md](../developing/plugin-system.md#modelprovider).

## Plugin selection

| Key | Default |
| --- | --- |
| `tool_sources` | `["artifacts", "web_search", "fetch_url", "widget_data", "inspect_widget_data", "pdf_extract", "dashboard", "recall_user_memory", "translate", "rerank", "vision_qa", "workspace_mcp"]` |
| `tool_source_config` | `{}` (per-name kwargs dict) |
| `subagents` | `["researcher", "charter", "analyst", "pdf_reader"]` |
| `middleware` | `["tool_message_normaliser", "tool_filter", "tool_call_announcer", "usage_recorder", "tool_call_ledger", "loop_guard", "call_limit", "tool_call_limit"]` |
| `skills` | `[]` (filesystem paths for DeepAgents `SkillsMiddleware`) |
| `system_prompt_file` | unset → bundled `prompts/default_system_prompt.md` |

## Memory pipeline

| Key | Default |
| --- | --- |
| `embeddings_provider` | `nvidia` |
| `embeddings_model` | `nvidia/nv-embed-v1` |
| `embeddings_config` | `{}` |
| `embeddings_code_provider` | `nvidia-code` (set to empty string to disable code routing) |
| `embeddings_code_model` | `nvidia/nv-embedcode-7b-v1` |
| `embeddings_code_config` | `{}` |
| `reranker_provider` | `nvidia` (empty string disables) |
| `reranker_model` | `nv-rerank-qa-mistral-4b:1` |
| `reranker_config` | `{}` |
| `rerank_fanout` | `32` |
| `translation_provider` | `nvidia` (empty string disables) |
| `translation_model` | `nvidia/riva-translate-4b-instruct-v1_1` |
| `translation_config` | `{}` |
| `translate_for_ingestion` | `true` |
| `ingest_target_language` | `English` |
| `ingest_char_threshold` | `2000` |
| `ingest_chunk_chars` | `1500` |
| `ingest_chunk_overlap` | `200` |

## Features

`features` is the map advertised in `/agents.json`. The built-in defaults:

```toml
[agent.features]
streaming = true
widget-dashboard-select = true
widget-dashboard-search = true
widget-global-search = true
mcp-tools = true
file-upload = true
generative-ui = true

[agent.features.search-web]
label = "Search Web"
description = "Allow the agent to search the public web when answering."
default = false

[agent.features.fetch-url]
label = "Fetch URL"
description = "Allow the agent to fetch and read the full text of a web page from a URL."
default = false
```

Reserved boolean keys: `streaming`, `widget-dashboard-select`, `widget-dashboard-search`, `widget-global-search`, `mcp-tools`, `file-upload`, `generative-ui`. Custom feature names beginning with `web search`, `web-search`, or `websearch` are rejected — use `search-web` (kebab-case).

## Metadata

`[agent.metadata]` populates `/agents.json` for the default profile:

```toml
[agent.metadata]
name = "OpenBB · NVIDIA Stack"
description = "..."
image_url = "https://..."   # optional
```

## Profiles

A profile is a named override bundle. `[agent.profiles.<name>]` inherits everything from `[agent]` and overrides selected keys. See [profiles.md](profiles.md).

## Worked example

```toml
[agent]
host = "0.0.0.0"
port = 6900
db_url = "postgresql+psycopg://openbb:***@db:5432/openbb"

[agent.auth]
backend = "oidc_jwt"
config = { jwks_url = "https://idp.example.com/.well-known/jwks.json", issuer = "https://idp.example.com", audience = "openbb-agent" }

[agent.model]
provider = "anthropic"
name = "claude-opus-4-7"
config = { max_tokens = 4096, temperature = 0.4 }

[agent.tool_source_config.web_search]
backend = "tavily"
```

Env-var equivalents use `OPENBB_AGENT_` + uppercase key, double-underscore for nesting:

```sh
OPENBB_AGENT_HOST=0.0.0.0
OPENBB_AGENT_PORT=6900
OPENBB_AGENT_DB_URL=postgresql+psycopg://...
OPENBB_AGENT_AUTH_BACKEND=oidc_jwt
OPENBB_AGENT_MODEL_PROVIDER=anthropic
OPENBB_AGENT_MODEL_NAME=claude-opus-4-7
```

## API key env vars

Tools and model providers read keys from the environment in this order: per-request (`QueryRequest.api_keys` forwarded by Workspace) → per-plugin config (`*_config.api_key`) → environment.

| Service | Env var |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI / OpenAI-compat | `OPENAI_API_KEY` |
| Google (Vertex / Gemini) | `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| NVIDIA NIM | `NVIDIA_API_KEY` |
| Tavily web-search backend | `TAVILY_API_KEY` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| Identity-hashing pepper | `OPENBB_AGENT_USER_ID_PEPPER` |

The agent server never logs or persists secrets. Identity emails are hashed via `OPENBB_AGENT_USER_ID_PEPPER` before they appear in any persisted row.

## CLI flags

```
openbb-agent-server [serve]
  --host HOST                # override agent.host
  --port PORT                # override agent.port
  --auth NAME                # override agent.auth.backend
  --model-provider NAME      # override agent.model.provider
  --model-name NAME          # override agent.model.name
  --reload                   # hot-reload (dev)
  --log-level info|debug|trace|...  # root log level (TRACE is the custom level)
  --config-file PATH         # explicit openbb.toml

openbb-agent-server keys issue --user-id ID [--scope agent:query --scope memory:read --scope memory:write] [--label NAME]
openbb-agent-server keys revoke --key-id ID
openbb-agent-server keys list [--user-id ID]

openbb-agent-server prune [--keep-last N] [--checkpoint-days N] [--history-days N] [--no-vacuum]
  # one-shot retention prune of checkpoints.db + history.db; flags override config.

openbb-agent-server --generate-config [PATH]   # write template
```

## Source

- [`app/settings.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/app/settings.py) — every key, its type, its default.
- [`app/config.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/app/config.py) — TOML loader (`[agent]` section + env-var overlay).
- [`main.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/main.py) — CLI.
