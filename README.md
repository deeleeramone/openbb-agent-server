# openbb-agent-server

Pluggable, multi-tenant agent backend that speaks the [OpenBB Workspace
custom-agent SSE protocol][workspace-protocol] and runs the agent loop
on top of the [LangChain DeepAgents harness][deepagents]. One process
hosts many agent profiles; auth, model provider, tools, sub-agents,
middleware, checkpointer, and persistence are independent plugin axes
— anything can be swapped without forking the package.

The full OpenBB Platform — every command across every installed
provider — is reachable via the optional `mcp_local` tool source,
which spawns the
[`openbb-mcp-server`](https://github.com/OpenBB-finance/OpenBB/tree/main/openbb_platform/extensions/mcp_server)
extension over stdio.

The default setup uses 100% free tokens and embedding models available from [NVIDIA](https://build.nvidia.com/) by registering for an API key [here](https://developer.nvidia.com/login)

## Install & run

```bash
# from a checkout of this repository
pip install -e '.[workspace-mcp]'
# add or combine more extras: [anthropic] [openai] [bedrock]
#                             [vertex] [google_genai] [groq]
#                             [snowflake] [tavily] [postgres]
#                             [pywry]  (desktop chat embedding)

export NVIDIA_API_KEY=...

openbb-agent-server
```

In OpenBB Workspace, add a custom agent pointing at
`http://localhost:8010`. Workspace fetches this once and
reads every agent profile the server registers in a single payload.

The `[workspace-mcp]` extra installs
[openbb-workspace-mcp](https://github.com/OpenBB-finance/workspace-mcp)
from its GitHub zip (Python ≥3.13 only). To run it in-process and skip
the separate `workspace-mcp` sidecar, set `mount_workspace_mcp = true`
in `openbb.toml` after installing the extra, then point the Workspace
UI's MCP-servers setting at `http://localhost:8010/mcp/workspace/mcp`.
The mount is **opt-in** (default `false`) so installing the extra alone
does not change the server's behavior.

For production, generate the config template and edit it:

```bash
openbb-agent-server --generate-config /etc/openbb/openbb.toml
openbb-agent-server --config-file /etc/openbb/openbb.toml --host 0.0.0.0
```

## Documentation

Documentation currently lives in [`docs/`](docs/README.md), and may move in the future:

| Audience                     | Start here                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| First-time user              | [Getting started](docs/guides/getting-started.md) → [Architecture](docs/guides/architecture.md) → [Workspace integration](docs/guides/workspace-integration.md)                                                                                                                                                                                                                                                                                                                                            |
| Desktop embedding            | [PyWry chat (ACP)](docs/guides/pywry-chat.md) — `openbb-agent-canvas` opens a window whose main page is the agent's live canvas (charts, tables, HTML) with the chat attached; or attach the chat to any PyWry widget via the `[pywry]` extra. Same `openbb.toml`, same loop, no HTTP server                                                                                                                                                                                                                |
| Operator / SRE               | [Configuration](docs/operating/configuration.md) → [Auth](docs/operating/auth.md) → [Persistence](docs/operating/persistence.md) → [Observability](docs/operating/observability.md)                                                                                                                                                                                                                                                                                                                        |
| Plugin author                | [Plugin system](docs/developing/plugin-system.md) → writing a [tool source](docs/developing/writing-a-tool-source.md) / [model provider](docs/developing/writing-a-model-provider.md) / [middleware](docs/developing/writing-a-middleware.md) / [sub-agent](docs/developing/writing-a-subagent.md) / [auth backend](docs/developing/writing-an-auth-backend.md) → [Conventions](docs/developing/conventions.md) → [Testing](docs/developing/testing.md)                                                     |
| API lookup                   | [Reference](docs/reference/) — module-by-module, mirrors the package tree                                                                                                                                                                                                                                                                                                                                                                                                                                  |
