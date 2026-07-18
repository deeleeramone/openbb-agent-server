# Writing an auth backend

An `AuthBackend` resolves an incoming HTTP request into a `UserPrincipal`. Everything else in the runtime is scoped by `principal.user_id`, so this is the only place identity is determined.

```python
class AuthBackend(ABC):
    name: str
    async def authenticate(self, request: Request) -> UserPrincipal: ...
```

```python
class UserPrincipal(BaseModel):
    user_id: str                  # stable across sessions; partition key
    display_name: str | None
    email: str | None
    scopes: tuple[str, ...]       # "agent:query", "memory:read", "memory:write", "admin"
    raw_claims: dict[str, Any]    # opaque, for downstream plugins
```

## Minimal example

```python
"""HMAC-signed cookie auth — issue your own session cookies, verify on each request."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any

from fastapi import HTTPException, Request

from openbb_agent_server.runtime.plugins import AuthBackend
from openbb_agent_server.runtime.principal import UserPrincipal


class CookieAuthBackend(AuthBackend):
    name = "cookie"

    def __init__(self) -> None:
        self._secret = os.environ.get("OPENBB_AGENT_COOKIE_SECRET", "").encode()
        if not self._secret:
            raise RuntimeError("OPENBB_AGENT_COOKIE_SECRET must be set")

    async def authenticate(self, request: Request) -> UserPrincipal:
        token = request.cookies.get("openbb_session")
        if not token:
            raise HTTPException(status_code=401, detail="missing session cookie")
        try:
            payload_b64, sig = token.rsplit(".", 1)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail="malformed token") from exc
        expected = hmac.new(self._secret, payload_b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(status_code=401, detail="bad signature")
        claims = json.loads(_b64url_decode(payload_b64))
        return UserPrincipal(
            user_id=str(claims["sub"]),
            display_name=claims.get("name"),
            email=claims.get("email"),
            scopes=tuple(claims.get("scopes") or ()),
            raw_claims=claims,
        )
```

Register:

```toml
[project.entry-points."openbb_agent_server.auth"]
cookie = "my_package.cookie_auth:CookieAuthBackend"
```

Select:

```toml
[agent.auth]
backend = "cookie"
```

(or via the environment: `OPENBB_AGENT_AUTH_BACKEND=cookie`).

## Contract

- **Return** a `UserPrincipal` on success.
- **Raise** `HTTPException` with the appropriate status on failure:
  - `401` — missing / unparsable credentials.
  - `403` — credentials parsed but insufficient scopes.
  - Never `200 with anonymous principal` — use the `none` backend if that's what you actually want (dev-only).

Every authenticated principal must have a stable `user_id`. Treat it as the partition key — every persistence write keys by it.

## Scopes

The runtime enforces three named scopes today:

| Scope | Required by |
| --- | --- |
| `agent:query` | `POST /v1/query` |
| `memory:read` | `GET /v1/memory`, `recall_user_memory` tool |
| `memory:write` | `PATCH /v1/memory/{id}`, `DELETE /v1/memory/{id}`, `ingest_request_context` |

Plus an open-ended `admin` scope for future admin endpoints. Backends can issue any scope string — plug-ins check via `principal.has_scope("foo")`.

## Built-in backends

| Backend | Source | Identity from | Use case |
| --- | --- | --- | --- |
| `none` | `plugins/auth/none.py` | hard-coded `anonymous` (scopes: `agent:query`, `memory:read`) | dev / single-tenant local |
| `bearer_static` | `plugins/auth/bearer_static.py` | env-set token compared to `Authorization: Bearer` | dev / smoke tests |
| `api_key_table` | `plugins/auth/api_key_table.py` | hashed lookup in `api_keys` SQL table (`oba_<id>.<secret>` plaintext) | service-to-service |
| `oidc_jwt` | `plugins/auth/oidc_jwt.py` | JWT verified against JWKS URL | production with an IdP |
| `openbb_workspace` | `plugins/auth/openbb_workspace.py` | `X-OpenBB-User: <email>` from upstream gateway | Workspace-gated deployments |

Each backend's source file is the best worked example.

## Caching JWKS

`oidc_jwt` caches the JWKS for `jwks_cache_seconds` (default 3600 / one hour). Backends are instantiated once per process by the registry, so the cache lives for the process lifetime.

## `user_id` collisions across backends

If the same user can authenticate via multiple backends (e.g. OIDC for the browser, API keys for scripts), pick one canonical `user_id` source — usually the IdP's `sub`. Backends that issue alternative credentials must map their key id back to that canonical `user_id`. Otherwise the same human gets two separate buckets of history and memory.

## Testing

```python
@pytest.mark.asyncio
async def test_valid_cookie_returns_principal(monkeypatch) -> None:
    monkeypatch.setenv("OPENBB_AGENT_COOKIE_SECRET", "test")
    backend = CookieAuthBackend()
    request = _make_request(cookies={"openbb_session": _sign({"sub": "alice", "scopes": ["agent:query"]})})
    p = await backend.authenticate(request)
    assert p.user_id == "alice"
    assert "agent:query" in p.scopes


@pytest.mark.asyncio
async def test_bad_signature_raises_401(monkeypatch) -> None:
    monkeypatch.setenv("OPENBB_AGENT_COOKIE_SECRET", "test")
    backend = CookieAuthBackend()
    with pytest.raises(HTTPException) as exc:
        await backend.authenticate(_make_request(cookies={"openbb_session": "x.y"}))
    assert exc.value.status_code == 401
```

## Source

- [`runtime.plugins`](../reference/runtime/plugins.md)
- [`runtime.principal`](../reference/runtime/principal.md)
- Worked examples: every file under [`plugins/auth/`](../reference/plugins/auth/index.md).
