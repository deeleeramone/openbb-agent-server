# Memory and recall

The agent has two storage layers:

1. **Chat history** — every turn of every conversation, partitioned by `user_id`. Used to replay context and audit traces. Lives in `HistoryStore` (`persistence/sqlite_store.py`).
2. **Cross-thread memory** — durable facts that persist across conversations, recalled semantically. Lives in `MemoryStore` (`memory/sqlite_store.py`), backed by `sqlite-vec` and an optional cross-encoder reranker.

This guide covers the second one.

## What gets remembered

Anything written to `MemoryStore.write` for the current `user_id`.

In the default HTTP server flow, writes come from request-time ingestion:

| Source | Trigger | Default state |
| --- | --- | --- |
| `ingest_request_context` | Long uploaded files / messages get chunked + embedded at request time. | On when the user has `memory:write` scope. |
| `memory/writer.py` helper | Optional post-run extractor helper for custom integrations. | Not wired into the default middleware list. |

All writes are scoped to `principal.user_id`; cross-user reads are physically impossible — the SQL queries filter on `json_extract(metadata, '$.user_id')`.

## What gets recalled

Two read paths:

- **Implicit** — the `recall_user_memory` tool gives the agent on-demand semantic search. The agent calls it when it needs context that might exist from prior sessions.
- **Explicit** — `GET /v1/memory` returns the user's own memory list; `PATCH /v1/memory/{id}` pins or relabels; `DELETE /v1/memory/{id}` forgets.

The recall pipeline:

```
query → embeddings.aembed_query → SQLiteVec ANN (fanout × 4)
                               ↓
               Python user_id filter → fanout rows
                               ↓
                pinned-rescue: always include user's pinned rows
                               ↓
                optional NvidiaReranker (cross-encoder) → top-k
                               ↓
                return [Memory(text, kind, pinned, score, source_trace_id, …)]
```

The two-stage shape (broad ANN fanout → narrow cross-encoder rerank) follows the [LangChain RAG recipe](https://docs.langchain.com/oss/python/integrations/retrievers).

## Configuration

```toml
# openbb.toml
[agent]
embeddings_provider = "nvidia"             # default
embeddings_model = "nvidia/nv-embed-v1"    # optional override
embeddings_code_provider = "nvidia-code"   # routes _code rows to nv-embedcode
embeddings_code_model = "nvidia/nv-embedcode-7b-v1"
reranker_provider = "nvidia"               # empty string disables
reranker_model = "nv-rerank-qa-mistral-4b:1"
translation_provider = "nvidia"            # empty string disables
translation_model = "nvidia/riva-translate-4b-instruct"
translate_for_ingestion = true             # auto-translate non-English chunks before embedding
rerank_fanout = 32                         # ANN pool size before rerank
```

Plumbing: `memory/factory.py::make_embeddings / make_reranker / make_translator`.

## Gating

Memory writes require the `memory:write` scope on the principal. There is no separate per-user opt-in flag wired up today — granting the scope is the only gate. The `none` auth backend does NOT grant it; use `bearer_static`, `api_key_table`, `oidc_jwt`, or `openbb_workspace`.

```sh
openbb-agent-server keys issue \
  --user-id alice@example.com \
  --scope agent:query --scope memory:read --scope memory:write
```

## Right to erasure

`DELETE /v1/me` purges the user's data from every table — including the vector index. The router calls `memory.delete_all_for_user(principal)` before `history.delete_user(principal)` so SQLiteVec rows are dropped first; the SQLAlchemy cascade then handles `traces`, `runs`, `messages`, `tool_calls`, `usage`, `artifacts`, `citations`, `pending_runs`, `api_keys`, and `users`.

## Use the retriever in your own code

`MemoryStoreRetriever` (`memory/retrievers.py`) is a `langchain_core.retrievers.BaseRetriever`. Drop it into any LangChain composition:

```python
from openbb_agent_server.memory.retrievers import MemoryStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_nvidia_ai_endpoints import NVIDIARerank

retriever = MemoryStoreRetriever(store=memory_store, principal=principal, k=12)
compressed = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=NVIDIARerank(model="nv-rerank-qa-mistral-4b:1", api_key=...),
)
docs = await compressed.ainvoke("which equities does the user actively trade options on?")
```

`WidgetDataRetriever` is the per-conversation analogue over `WidgetDataStore`. Both return `Document` objects with full metadata.

## Pinning

Pinned memories always pass through into the rerank pool regardless of ANN score, and their final score is floored at 1.0. Use pinning for facts that must show up every turn — e.g. "the user manages a fund with a $500M AUM" stays prominent even when the current query doesn't mention "fund" or "AUM".

```http
PATCH /v1/memory/{memory_id}
{"pinned": true}
```

## What's a "memory entry"

`memory/store.py::Memory`:

```python
class Memory(BaseModel):
    memory_id: str
    user_id: str
    text: str
    kind: str = "fact"              # "fact" | "context_text" | "context_code"
    pinned: bool = False
    source_trace_id: str | None     # the trace that produced this entry
    score: float | None             # populated on retrieval
```

`kind` drives embedder routing — `context_code` rows (anything written by `ingest_request_context` for a source classified as code) go through the code-tuned embedder (`nvidia/nv-embedcode-7b-v1` by default).

## How the `memories` tables look

Two SQLiteVec tables on the same SQLite file as `HistoryStore`:

- `memories_text` — `(rowid, text, metadata, text_embedding)`
- `memories_code` — same shape (only created when `code_embeddings` is configured)

`metadata` is JSON: `{memory_id, user_id, kind, pinned, source_trace_id, created_at}`. The companion `memories_text_vec` / `memories_code_vec` virtual tables hold the `sqlite-vec` index. Cross-user isolation: every query carries `WHERE json_extract(metadata, '$.user_id') = ?`.

See [`memory.sqlite_store`](../reference/memory/sqlite_store.md) and [`memory.store`](../reference/memory/store.md).
