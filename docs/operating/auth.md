# Auth

Five built-in backends, picked via `agent.auth.backend` / `OPENBB_AGENT_AUTH_BACKEND`. Each returns a `UserPrincipal` from request headers. Every persisted row keys by `principal.user_id`; cross-user reads return `404`.

## Picking a backend

| Backend | When |
| --- | --- |
| `none` | local dev, single user |
| `bearer_static` | shared secret for CI / smoke |
| `api_key_table` | service-to-service, hashed keys in the DB |
| `oidc_jwt` | IdP-issued JWTs (Auth0 / Cognito / Okta / custom OIDC) |
| `openbb_workspace` | upstream Workspace gateway forwarding the user's email |

## `none`

```toml
[agent.auth]
backend = "none"
```

Every request resolves to `UserPrincipal(user_id="anonymous", display_name="Anonymous", scopes=("agent:query", "memory:read"))`. Logs a loud warning at startup. No `memory:write` scope — memory writes are off by default in this backend.

## `bearer_static`

```toml
[agent.auth]
backend = "bearer_static"
```

```sh
export OPENBB_AGENT_AUTH_BEARER=shared-secret-token
```

Clients send `Authorization: Bearer shared-secret-token`. Every accepted request resolves to `UserPrincipal(user_id="static-user", scopes=("agent:query", "memory:read", "memory:write"))`. Single shared identity; not multi-user.

Constructor knobs (set via `agent.auth.config`):

| Key | Default |
| --- | --- |
| `token` | env `OPENBB_AGENT_AUTH_BEARER` |
| `user_id` | `static-user` |
| `display_name` | `None` |
| `scopes` | `("agent:query", "memory:read", "memory:write")` |

## `api_key_table`

```toml
[agent.auth]
backend = "api_key_table"
```

Hashed-key lookup against the `api_keys` table. The full plaintext key is `oba_<key_id>.<secret>` — the `oba_` prefix is required. Clients send either `X-API-KEY: <plaintext>` or `Authorization: Bearer <plaintext>`.

Mint, revoke, and list keys with the bundled CLI:

```sh
openbb-agent-server keys issue --user-id alice@example.com --scope agent:query --scope memory:read --scope memory:write --label "alice-laptop"
# prints the plaintext key once: oba_<id>.<secret>

openbb-agent-server keys list --user-id alice@example.com
openbb-agent-server keys revoke --key-id <id>
```

`keys issue` upserts the `users` row, hashes the secret with argon2id, and returns the plaintext one time. Default scopes when `--scope` is omitted: `agent:query`, `memory:read`.

## `oidc_jwt`

```toml
[agent.auth]
backend = "oidc_jwt"

[agent.auth.config]
jwks_url = "https://idp.example.com/.well-known/jwks.json"
audience = "openbb-agent"
issuer = "https://idp.example.com"
algorithms = ["RS256"]
jwks_cache_seconds = 3600
```

Clients send `Authorization: Bearer <jwt>`. The backend verifies signature + audience + issuer with `PyJWT`, then maps claims to a `UserPrincipal`:

| Claim | Maps to |
| --- | --- |
| `sub` | `user_id` (required) |
| `name` or `preferred_username` | `display_name` |
| `email` | `email` |
| `scope` (space-separated string) or `scopes` (list) | `scopes` |
| _all_ | `raw_claims` |

`jwks_cache_seconds` defaults to 3600 (one hour). The JWKS is refetched on cache miss.

## `openbb_workspace`

```toml
[agent.auth]
backend = "openbb_workspace"
```

Trusts an `X-OpenBB-User` header from an upstream gateway (typically Workspace itself). The header value must be an email; the backend hashes it via HMAC-SHA256 (peppered with `OPENBB_AGENT_USER_ID_PEPPER`) and uses the hash as `user_id`. The raw email is preserved on `UserPrincipal.email` but never written to logs or persistence rows.

Constructor knobs:

| Key | Default |
| --- | --- |
| `header` | `X-OpenBB-User` |
| `scopes` | `("agent:query", "memory:read", "memory:write")` |
| `require_email` | `true` (non-email header → 403) |

## Scopes the runtime checks

| Scope | Required by |
| --- | --- |
| `agent:query` | `POST /v1/query` |
| `memory:read` | `GET /v1/memory`, `recall_user_memory` tool |
| `memory:write` | `PATCH /v1/memory/{id}`, `DELETE /v1/memory/{id}`, `ingest_request_context` |

`POST /v1/conversations/{id}/cancel` requires authentication but no scope — it only cancels runs owned by the calling principal.

A backend can issue any scope string; plugins check via `principal.has_scope("foo")`.

## Cross-user isolation

Every store method takes a `UserPrincipal` and filters by `user_id`. A request for a resource owned by a different user returns `404` (not `403`) to avoid leaking existence.

## Right-to-erasure

`DELETE /v1/me` purges the caller's data:

1. `memory.delete_all_for_user(principal)` — drops every row from `memories_text` / `memories_code` SQLiteVec tables.
2. `history.delete_user(principal)` — cascade through `users`, `api_keys`, `conversations`, `messages`, `traces`, `runs`, `tool_calls`, `usage`, `artifacts`, `citations`, `pending_runs`, `widget_data`.

## Source

- [`runtime/principal.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/runtime/principal.py) — `UserPrincipal`.
- [`runtime/identity.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/runtime/identity.py) — `hash_user_id`.
- [`plugins/auth/none.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/auth/none.py)
- [`plugins/auth/bearer_static.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/auth/bearer_static.py)
- [`plugins/auth/api_key_table.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/auth/api_key_table.py)
- [`plugins/auth/oidc_jwt.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/auth/oidc_jwt.py)
- [`plugins/auth/openbb_workspace.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/plugins/auth/openbb_workspace.py)
