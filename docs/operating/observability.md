# Observability

Two signals are wired up: structured JSON logs and per-call usage/tool ledgers.

## Logs

Every log line is JSON to stdout. The trace contextvar adds `trace_id`, `run_id`, `conversation_id`, and `user_id` (the hashed identity, never an email).

```jsonc
{
  "level": "INFO",
  "logger": "openbb_agent_server.router",
  "message": "query received | agent=default msgs=4 last_role=human ...",
  "trace": {"trace_id":"01HZ…","run_id":"01HZ…","conversation_id":"01HZ…","user_id":"u-0cf19c8f…"}
}
```

The formatter also runs a PII redaction filter that scrubs:

- Email addresses → opaque `u-<hash>` (same hash as `user_id`).
- `Bearer <token>` and `Authorization: …` headers.
- API-key-shaped strings (`sk-…`, `nvapi-…`, `tvly-…`, `gsk_…`, `X-API-KEY=…`).

Set level with the CLI flag `--log-level info|debug|warning|error|trace`. Default is `info`. `trace` is the in-package level below `debug` used for body dumps and per-stream-frame detail.

## Usage rows

The `UsageRecorder` middleware writes one `usage` row per model call:

```
usage(trace_id, seq, user_id, model, input_tokens, output_tokens,
      cache_read, cache_creation, cost_usd)
```

`cost_usd` is always `0.0` today — the column exists for a future pricing plugin that's not implemented.

Aggregated via `GET /v1/usage`:

```
GET /v1/usage?trace_id=01HZ...
GET /v1/usage?conversation_id=01HZ...
GET /v1/usage?from=2026-05-01&to=2026-05-13
```

Principal-scoped: a user only ever sees their own rows.

## Tool-call ledger

The `ToolCallLedger` middleware writes one row per server-side tool invocation:

```
tool_calls(trace_id, seq, user_id, tool_name, args_json, result_json,
           error, latency_ms, side, state)
```

`side` is `server` for in-process tools. Client-side tool calls (the `copilotFunctionCall` roundtrip through the Workspace UI) get two rows — one when the `FunctionCallSSE` is emitted (`state="pending"`) and one when the next request returns the result (`state="complete"` or `state="error"`).

## Trace bundle

```
GET /v1/traces/{trace_id}
```

Returns the join of `messages` + `tool_calls` + `usage` + `artifacts` + `citations` for one trace. Useful for support and audits.

## Wire-level trace id

`POST /v1/query` responses carry:

| Header | Value |
| --- | --- |
| `X-Trace-ID` | mirrors the conversation id (from request body, `X-Trace-ID` header, or a generated UUID) |
| `X-Server-Trace-ID` | server-generated UUIDv7 unique to this exchange |

## Source

- [`observability/logging.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/observability/logging.py) — JSON formatter, PII redaction filter, trace contextvar, `TRACE` level.
- [`plugins/middleware/usage_recorder.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/middleware/usage_recorder.py) — captures `usage_metadata` from each `AIMessage` and writes `usage` rows.
- [`plugins/middleware/tool_call_ledger.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/middleware/tool_call_ledger.py)
- [`app/router.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/app/router.py) — `/v1/usage` and `/v1/traces/{id}` handlers.
