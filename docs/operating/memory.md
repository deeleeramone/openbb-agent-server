# Memory

Per-user cross-thread vector memory. Stored in `langchain_community.vectorstores.SQLiteVec` tables on the same file as the rest of persistence. For the read/write/recall API see [memory-and-recall.md](../guides/memory-and-recall.md).

## Configuration

```toml
[agent]
embeddings_provider = "nvidia"             # "nvidia" | "nvidia-code" | "hash"
embeddings_model = "nvidia/nv-embed-v1"
embeddings_code_provider = "nvidia-code"   # empty string disables code routing
embeddings_code_model = "nvidia/nv-embedcode-7b-v1"
reranker_provider = "nvidia"               # empty string disables
reranker_model = "nv-rerank-qa-mistral-4b:1"
rerank_fanout = 32                         # ANN candidates fed to the reranker
translation_provider = "nvidia"            # empty string disables
translation_model = "nvidia/riva-translate-4b-instruct-v1_1"
translate_for_ingestion = true             # auto-translate non-English chunks before embedding
ingest_target_language = "English"
ingest_char_threshold = 2000               # only ingest content over this many chars
ingest_chunk_chars = 1500
ingest_chunk_overlap = 200
```

`*_config` dicts (e.g. `embeddings_config`, `reranker_config`) carry per-provider kwargs.

## Embedders

| Provider | Default model | Vector dim | When |
| --- | --- | --- | --- |
| `nvidia` | `nvidia/nv-embed-v1` | 4096 | text recall |
| `nvidia-code` | `nvidia/nv-embedcode-7b-v1` | 4096 | code-classified chunks (paired with `embeddings_code_provider`) |
| `hash` | feature-hashed | 256 | tests / zero-config dev |

The classifier in `memory/classifier.py` routes each chunk to `nvidia-code` when `looks_like_code()` is true; everything else goes to `nvidia`.

## Reranker

Setting `reranker_provider = "nvidia"` enables a two-stage retrieval:

1. SQLiteVec ANN pulls `rerank_fanout` candidates (default 32).
2. Pinned rows are merged in unconditionally.
3. The cross-encoder rescores the pool; top-k returned.

Reranker failures fall back to ANN-only order. Logged at WARNING.

## Translator

When `translation_provider = "nvidia"` and `translate_for_ingestion = true`, any uploaded chunk classified as non-English (≥ 5% non-ASCII in the first 4 KB) is auto-translated to `ingest_target_language` before embedding. The translated text is what gets stored; the original-language tag is preserved in the chunk header.

Failures fall back to storing the original text. Logged at WARNING.

## Gating

Memory writes require the `memory:write` scope. There is no separate user-row opt-in flag wired up today — the `users.memory_opt_in` column exists but isn't read. Granting `memory:write` is the only gate.

Reads (`recall_user_memory` tool, `GET /v1/memory`) require `memory:read`.

## What writes memory

| Source | When |
| --- | --- |
| `MemoryWriter` middleware | post-turn, extracts durable facts from the assistant reply (requires `memory:write`) |
| `ingest_request_context` | long uploaded files / human messages exceeding `ingest_char_threshold` get chunked + embedded at request time |

Every write carries `source_trace_id` so any recalled fact can be traced back to the conversation that produced it.

## What reads memory

| API | Scope |
| --- | --- |
| `recall_user_memory(query, k=8)` tool | `memory:read` |
| `GET /v1/memory?limit=N` | `memory:read` |
| `PATCH /v1/memory/{memory_id}` (`{"pinned": true}`) | `memory:write` |
| `DELETE /v1/memory/{memory_id}` | `memory:write` |
| `DELETE /v1/me` | (caller's own data) |

## Right-to-erasure

`DELETE /v1/me` runs `memory.delete_all_for_user(principal)` before `history.delete_user(principal)`. Memory rows go first (SQLiteVec rowid deletes); SQLAlchemy cascade handles everything else.

## Source

- [`memory/store.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/memory/store.py) — `MemoryStore` ABC.
- [`memory/sqlite_store.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/memory/sqlite_store.py) — SQLiteVec impl.
- [`memory/factory.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/memory/factory.py) — embedder / reranker / translator builders.
- [`memory/ingestion.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/memory/ingestion.py) — request-time ingestion.
- [`memory/writer.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/memory/writer.py) — `MemoryWriter` middleware.
- [`memory/classifier.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/memory/classifier.py) — code vs text routing.
