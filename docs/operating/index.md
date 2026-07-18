# Operator guides

Everything an SRE or platform team needs to run the agent server in production — the configuration cascade, identity, the persistence schema, the memory pipeline knobs, and what gets logged. These guides assume you have already followed the [Getting started](../guides/getting-started.md) flow and are now hardening the deployment.

## Pages

### [Configuration](configuration.md)
Every settable `[agent]` key with its default, env-var name, and (where applicable) CLI flag. Covers the precedence cascade (CLI > env > TOML > built-in), top-level keys (`host`, `port`, `db_url`, `data_dir`, `checkpointer_provider`), plugin selection lists (`tool_sources`, `subagents`, `middleware`, `skills`), the full memory pipeline (embedder, reranker, translator, fanout, chunk sizing), reserved vs custom `features`, the metadata block, API-key env vars, and the bundled CLI flags — including `--generate-config` to scaffold a working TOML.

### [Profiles](profiles.md)
URL routing for the default profile vs named profiles (`POST /v1/query` vs `POST /agents/{name}/v1/query`), the frozen `AgentProfile` shape, declaring overlays under `[agent.profiles.<name>]`, per-profile `system_prompt_file` (inline strings are rejected) and the template placeholders prompts accept, the reserved feature-flag keys, custom dict-shaped features, and the `^[a-z0-9-]+$` name constraint Workspace enforces.

### [Auth](auth.md)
The five built-in backends with the headers each consumes and the `UserPrincipal` it produces — `none` (anonymous, no `memory:write`), `bearer_static` (shared secret), `api_key_table` (argon2id-hashed `oba_<id>.<secret>` keys in the DB), `oidc_jwt` (JWKS-verified IdP tokens), and `openbb_workspace` (HMAC-hashed email from `X-OpenBB-User`). Includes the `openbb-agent-server keys` CLI for mint / revoke / list, the scope matrix (`agent:query`, `memory:read`, `memory:write`), and the 404-not-403 cross-user isolation rule.

### [Persistence](persistence.md)
The dual-layer schema. The SQLAlchemy ORM tables (`users`, `api_keys`, `conversations`, `messages`, `traces`, `runs`, `tool_calls`, `usage`, `artifacts`, `citations`, `pending_runs`, `widget_data`, `pdf_documents`, `pdf_pages`) are documented column-by-column with their primary and business keys. `SQLiteVec` vector tables (`memories_text`, `memories_code`, `pdf_pages_vec`) live in the same file under SQLite mode and are skipped under Postgres. Covers `init_schema()` bootstrap, `db_url` formats, and backup procedure.

### [Memory](memory.md)
The memory pipeline configuration cross-referenced to the runtime flows — embedder providers (`nvidia`, `nvidia-code`, `hash`) with their vector dimensions, the `looks_like_code()` classifier that routes code chunks to the code-tuned embedder, the optional cross-encoder reranker with `rerank_fanout`, the auto-translator for non-English ingestion, the active writer (`ingest_request_context`) plus optional post-run helper (`memory/writer.py`), readers (`recall_user_memory` tool, `GET /v1/memory`), and the right-to-erasure ordering (vector rows first, then SQLAlchemy cascade).

### [Observability](observability.md)
Structured JSON-to-stdout logs with the trace contextvar attaching `trace_id`, `run_id`, `conversation_id`, and the hashed `user_id` to every line. Covers the PII redaction filter (emails, bearer tokens, API-key-shaped strings), `--log-level` including the in-package `trace` level, `usage` rows from `UsageRecorder` and how they are aggregated via `GET /v1/usage`, the `tool_calls` ledger (including the two-row pattern for client-side tool calls), the `GET /v1/traces/{id}` bundle, and the response headers (`X-Trace-ID`, `X-Server-Trace-ID`).

## See also

- [`guides/`](../guides/index.md) — user-facing walk-throughs of the agent's surfaces.
- [`developing/`](../developing/index.md) — writing plugins, conventions, the test harness.
- [`reference/`](../reference/index.md) — symbol-level API reference.
- [`docs/index.md`](../index.md) — the parent index.
