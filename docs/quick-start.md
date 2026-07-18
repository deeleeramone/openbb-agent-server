# Quick Start

Two ways to run the agent: as an **HTTP server** behind OpenBB Workspace, or as a **desktop canvas** window. Both share the same configurable agent loop.

## A. HTTP server for OpenBB Workspace

1.  Install (see [Installation](installation.md)) and set the one required key:

    ```sh
    export NVIDIA_API_KEY=nvapi-...
    ```

2.  Run the server:

    ```sh
    openbb-agent-server
    ```

    It logs where it is listening:

    ```json
    {"level":"INFO","logger":"openbb_agent_server.main","message":"agent server listening on 127.0.0.1:6900"}
    ```

3.  Verify it responds:

    ```sh
    curl http://127.0.0.1:6900/agents.json
    ```

    You should get a JSON map of profile metadata (the shape is documented in the [Wire-protocol contract](explanation/wire-contract.md)).

4.  Add it to Workspace: **AI Agents → Add Agent → paste `http://127.0.0.1:6900` → save**. Full walk-through: [Workspace integration](guides/workspace-integration.md).

5.  Pick the agent in the chat panel and ask:

    > What can you do?

From there, explore [widgets & data](guides/widgets-and-data.md), [multimodal files](guides/multimodal.md), [background jobs](guides/background-jobs.md), and [cross-thread memory](guides/memory-and-recall.md).

## B. Desktop canvas (PyWry)

The same agent loop, embedded in a native window whose main page is a live canvas the agent draws on (charts, tables, documents, TradingView widgets).

```sh
pip install -e ".[pywry]"
export NVIDIA_API_KEY=nvapi-...
openbb-agent-canvas
```

A window opens with the chat panel attached; the agent renders to the page through the [`pywry_canvas`](reference/plugins/tools/pywry_canvas.md) tools. See [PyWry chat (ACP)](guides/pywry-chat.md).

## Configuration cascade

Configuration resolves in this precedence (highest wins):

```
CLI flags
  → environment variables (OPENBB_AGENT_*)
    → explicit --config-file / OPENBB_CONFIG
      → ./openbb.toml  →  ~/.openbb_platform/openbb.toml
        → pyproject.toml [tool.openbb]
          → built-in defaults
```

The same keys can live under `[agent]` in an `openbb.toml`. Generate a starter config and see every key in [Advanced Configuration](operating/configuration.md).
