"""oidc_jwt auth backend tests."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from threading import Thread

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from starlette.requests import Request

from openbb_agent_server.plugins.auth.oidc_jwt import OidcJwtAuthBackend


def _make_jwks(public_key_pem: bytes, kid: str = "test-kid") -> dict[str, object]:
    from cryptography.hazmat.primitives import serialization

    pub = serialization.load_pem_public_key(public_key_pem)
    numbers = pub.public_numbers()  # type: ignore[union-attr]
    import base64

    def b64u(n: int) -> str:
        as_bytes = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(as_bytes).rstrip(b"=").decode()

    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": kid,
                "use": "sig",
                "alg": "RS256",
                "n": b64u(numbers.n),
                "e": b64u(numbers.e),
            }
        ]
    }


@pytest.fixture(scope="module")
def keypair() -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives import serialization

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


@pytest.fixture(scope="module")
def jwks_server(keypair: tuple[bytes, bytes]) -> Iterator[str]:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    _, pub_pem = keypair
    jwks = _make_jwks(pub_pem)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps(jwks).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # noqa: D401
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/jwks.json"
    finally:
        server.shutdown()


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw})


def _sign(priv_pem: bytes, claims: dict[str, object]) -> str:
    return jwt.encode(claims, priv_pem, algorithm="RS256", headers={"kid": "test-kid"})


def test_constructor_requires_jwks_url() -> None:
    with pytest.raises(RuntimeError):
        OidcJwtAuthBackend(jwks_url="")


def test_missing_authorization_header_returns_401(jwks_server: str) -> None:
    backend = OidcJwtAuthBackend(jwks_url=jwks_server)
    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(backend.authenticate(_request({})))
    assert exc.value.status_code == 401


def test_valid_jwt_round_trip(jwks_server: str, keypair: tuple[bytes, bytes]) -> None:
    priv, _ = keypair
    backend = OidcJwtAuthBackend(
        jwks_url=jwks_server, audience="agent-server", issuer="https://idp"
    )
    token = _sign(
        priv,
        {
            "sub": "user-42",
            "iss": "https://idp",
            "aud": "agent-server",
            "exp": int(time.time()) + 60,
            "scope": "agent:query memory:read",
            "name": "Test User",
            "email": "u@example.com",
        },
    )
    import asyncio

    p = asyncio.run(
        backend.authenticate(_request({"authorization": f"Bearer {token}"}))
    )
    assert p.user_id == "user-42"
    assert "agent:query" in p.scopes
    assert "memory:read" in p.scopes
    assert p.email == "u@example.com"


def test_expired_jwt_returns_403(
    jwks_server: str, keypair: tuple[bytes, bytes]
) -> None:
    priv, _ = keypair
    backend = OidcJwtAuthBackend(jwks_url=jwks_server)
    token = _sign(
        priv,
        {"sub": "u", "exp": int(time.time()) - 60},
    )
    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            backend.authenticate(_request({"authorization": f"Bearer {token}"}))
        )
    assert exc.value.status_code == 403


def test_wrong_audience_returns_403(
    jwks_server: str, keypair: tuple[bytes, bytes]
) -> None:
    priv, _ = keypair
    backend = OidcJwtAuthBackend(jwks_url=jwks_server, audience="something-else")
    token = _sign(
        priv,
        {"sub": "u", "aud": "agent-server", "exp": int(time.time()) + 60},
    )
    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            backend.authenticate(_request({"authorization": f"Bearer {token}"}))
        )
    assert exc.value.status_code == 403


def test_token_without_sub_returns_403(
    jwks_server: str, keypair: tuple[bytes, bytes]
) -> None:
    priv, _ = keypair
    backend = OidcJwtAuthBackend(jwks_url=jwks_server)
    token = _sign(priv, {"exp": int(time.time()) + 60})
    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            backend.authenticate(_request({"authorization": f"Bearer {token}"}))
        )
    assert exc.value.status_code == 403


def test_scopes_as_list_claim_are_extracted(
    jwks_server: str, keypair: tuple[bytes, bytes]
) -> None:
    priv, _ = keypair
    backend = OidcJwtAuthBackend(jwks_url=jwks_server)
    token = _sign(
        priv,
        {
            "sub": "u",
            "scopes": ["a:b", "c:d"],
            "exp": int(time.time()) + 60,
        },
    )
    import asyncio

    p = asyncio.run(
        backend.authenticate(_request({"authorization": f"Bearer {token}"}))
    )
    assert p.scopes == ("a:b", "c:d")


def test_invalid_signature_returns_403(jwks_server: str) -> None:
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    from cryptography.hazmat.primitives import serialization

    priv = other.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    backend = OidcJwtAuthBackend(jwks_url=jwks_server)
    token = jwt.encode(
        {"sub": "u", "exp": int(time.time()) + 60},
        priv,
        algorithm="RS256",
        headers={"kid": "missing-kid"},
    )
    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            backend.authenticate(_request({"authorization": f"Bearer {token}"}))
        )
    assert exc.value.status_code == 403


def test_non_bearer_scheme_returns_401(jwks_server: str) -> None:
    backend = OidcJwtAuthBackend(jwks_url=jwks_server)
    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(backend.authenticate(_request({"authorization": "Basic abc"})))
    assert exc.value.status_code == 401


def test_oidc_jwt_extract_scopes_handles_list_value() -> None:
    from openbb_agent_server.plugins.auth.oidc_jwt import OidcJwtAuthBackend

    assert OidcJwtAuthBackend._extract_scopes({"scopes": ["a", "b", "c"]}) == (
        "a",
        "b",
        "c",
    )
    assert OidcJwtAuthBackend._extract_scopes({"scope": ("x", "y")}) == ("x", "y")
    assert OidcJwtAuthBackend._extract_scopes({"scope": 42}) == ()
