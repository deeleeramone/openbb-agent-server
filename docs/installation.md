# Installation

## Requirements

- **Python 3.11+** (`requires-python = ">=3.11,<3.15"`).
- An **NVIDIA API key** for production-quality embeddings, reranking, and the NIM-backed multimodal tools (vision / audio). Without one the server falls back to deterministic hash embeddings and the NIM-backed tools quietly skip registration. Get one at [build.nvidia.com](https://build.nvidia.com).
- A **model-provider key** for whichever chat model you run (Anthropic / OpenAI / Bedrock / Vertex / Groq). The default profile uses an NVIDIA-hosted model, so an NVIDIA key alone is enough to start.

## Install

Clone the repository and install it in editable mode — with `pip`:

```sh
git clone https://github.com/deeleeramone/openbb-agent-server
cd openbb-agent-server
pip install -e .
```

…or with [`uv`](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/deeleeramone/openbb-agent-server
cd openbb-agent-server
uv sync
```

`langchain-nvidia-ai-endpoints`, `langchain-text-splitters`, and `sqlite-vec` are base dependencies — the default model and the memory pipeline work with no extras installed.

## Optional extras

Install only what your deployment needs. Each model provider is opt-in so a minimal install stays small.

| Extra | Installs | Enables |
| --- | --- | --- |
| `anthropic` | `langchain-anthropic` | Claude chat models |
| `openai` | `langchain-openai` | OpenAI chat models |
| `bedrock` | `langchain-aws` | Amazon Bedrock models |
| `vertex` / `google_genai` | `langchain-google-genai`, `google-genai` | Google Gemini / Vertex models |
| `groq` | `langchain-groq` | Groq-hosted models |
| `postgres` | `psycopg[binary]`, `pgvector`, `langgraph-checkpoint-postgres` | Postgres history + the Postgres checkpointer (multi-worker prod) |
| `tavily` | `tavily-python` | Tavily web-search backend (better quality, needs an API key) |
| `pywry` | `pywry` | The ACP desktop canvas (`openbb-agent-canvas`) — **not** in `[all]` |
| `all` | every extra **except** `pywry` | Everything for a server deployment |

```sh
pip install -e ".[anthropic]"            # one provider
pip install -e ".[postgres,tavily]"      # combine
pip install -e ".[all]"                  # full server stack (no pywry)
pip install -e ".[pywry]"                # the desktop canvas
```

## Console scripts

Two entry points are installed:

| Command | Purpose |
| --- | --- |
| `openbb-agent-server` | Run the HTTP server (the Workspace custom-agent backend). |
| `openbb-agent-canvas` | Open the PyWry desktop window with the agent attached as a chat panel + live canvas. Needs the `[pywry]` extra. |

Next: the [**Quick Start**](quick-start.md) to run your first query, or [**Advanced Configuration**](operating/configuration.md) for the full settings cascade.
