# Getting Started

The agent server is a Python package that ships its own CLI. Out of the box it speaks the [OpenBB Workspace](https://docs.openbb.co/workspace) custom-agent SSE protocol and runs every OpenBB Platform command via an in-process MCP server.

## Prerequisites

- Python 3.10+
- An NVIDIA API key if you want production-quality embeddings, reranking, and multimodal tools (vision / audio). Without one, the server falls back to deterministic hash embeddings and the NIM-backed tools quietly skip registration.
- A model-provider key for whichever chat model you choose (Anthropic / OpenAI / Bedrock / Vertex / Groq / Snowflake Cortex). The default profile uses NVIDIA `nemotron-3-super-120b-a12b`.

## Install

From the monorepo:

```sh
cd openbb_platform/extensions/agent_server
pip install -e .
```

The library installs `langchain-nvidia-ai-endpoints`, which provides free access to dozens of models. For other providers:

```sh
pip install -e ".[anthropic]"     # langchain-anthropic
pip install -e ".[openai]"        # langchain-openai
pip install -e ".[bedrock]"       # langchain-aws
pip install -e ".[vertex]"        # langchain-google-genai
pip install -e ".[groq]"          # langchain-groq
pip install -e ".[snowflake]"     # snowflake-connector-python + sqlglot
pip install -e ".[postgres]"      # psycopg + langgraph-checkpoint-postgres
```

`langchain-community`, `langchain-text-splitters`, and `sqlite-vec` are base dependencies — the memory pipeline needs them in every install.

## Configure

The minimal configuration is one environment variable, NVIDIA_API_KEY.

```sh
export NVIDIA_API_KEY=nvapi-...
```

For all keys + defaults, see [Configuration](../operating/configuration.md). The same keys can live in an `openbb.toml` under `[agent]`.

## Run

```sh
openbb-agent-server
```

The server logs:

```json
{"level":"INFO","logger":"openbb_agent_server.main","message":"agent server listening on 127.0.0.1:8010"}
```

Verify it responds:

```sh
curl http://127.0.0.1:8010/agents.json
```

You should see a JSON map of profile metadata. See [Architecture → Wire protocol](architecture.md#wire-protocol) for the shape.

## Add the agent to OpenBB Workspace

Open Workspace → AI Agents → Add Agent → paste `http://127.0.0.1:8010` → save. Detailed walk-through: [Workspace integration](workspace-integration.md).

## First conversation

Open a workspace, pick the new agent in the chat panel, and try:

> What can you do?

The default system prompt explains the agent's capabilities. From there:

- **PDF / image / spreadsheet:** add Workspace files into the chat context. See [Multimodal tools](multimodal.md).
- **Widget data:** pin a widget to your dashboard and ask a question grounded in it. See [Widgets and data](widgets-and-data.md).
- **Background jobs:** ask the agent to "transcribe these three clips in parallel". See [Background jobs](background-jobs.md).
- **Cross-thread memory:** ask the agent to remember a preference (the principal needs the `memory:write` scope — `none` auth doesn't grant it; use `bearer_static`, `api_key_table`, `oidc_jwt`, or `openbb_workspace`). See [Memory and recall](memory-and-recall.md).

## What's next

| You want to… | Read |
| --- | --- |
| Understand the request lifecycle | [Architecture](architecture.md) |
| Configure for production | [Configuration](../operating/configuration.md) |
| Plug in a custom tool | [Writing a tool source](../developing/writing-a-tool-source.md) |
| Swap the chat model | [Configuration](../operating/configuration.md#model) |
| Look up an exact function | [API reference](../reference/index.md) |
