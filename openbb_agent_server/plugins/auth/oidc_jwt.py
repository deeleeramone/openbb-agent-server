"""OIDC JWT auth backend."""

from __future__ import annotations

import logging
from typing import Any

import jwt
from fastapi import HTTPException, Request
from jwt import PyJWKClient

from openbb_agent_server.runtime.plugins import AuthBackend
from openbb_agent_server.runtime.principal import UserPrincipal

logger = logging.getLogger("openbb_agent_server.auth.oidc_jwt")


class OidcJwtAuthBackend(AuthBackend):
    """Verify JWT bearer tokens against a JWKS URL."""

    name = "oidc_jwt"

    def __init__(
        self,
        *,
        jwks_url: str,
        audience: str | None = None,
        issuer: str | None = None,
        algorithms: tuple[str, ...] = ("RS256",),
        jwks_cache_seconds: int = 3600,
    ) -> None:
        """Configure the backend with a JWKS endpoint and claim checks.

        Parameters
        ----------
        jwks_url : str
            URL of the OIDC provider's JWKS document. Required; the signing
            keys are fetched and cached from here.
        audience : str or None, optional
            Expected ``aud`` claim. When set, tokens whose audience does not
            match are rejected. ``None`` skips the audience check.
        issuer : str or None, optional
            Expected ``iss`` claim. When set, tokens from a different issuer
            are rejected. ``None`` skips the issuer check.
        algorithms : tuple of str, optional
            Allowed JWT signing algorithms. Defaults to ``("RS256",)``.
        jwks_cache_seconds : int, optional
            Lifespan, in seconds, of the cached JWKS signing keys. Defaults
            to 3600.

        Raises
        ------
        RuntimeError
            If ``jwks_url`` is empty.
        """
        if not jwks_url:
            raise RuntimeError("OidcJwtAuthBackend requires jwks_url")
        self._jwks_client = PyJWKClient(
            jwks_url, cache_keys=True, lifespan=jwks_cache_seconds
        )
        self._audience = audience
        self._issuer = issuer
        self._algorithms = list(algorithms)

    async def authenticate(self, request: Request) -> UserPrincipal:
        """Verify the request's bearer token and build a principal.

        Extract the ``Authorization: Bearer <token>`` header, resolve the
        signing key from the JWKS endpoint, decode and validate the token
        against the configured audience, issuer, and algorithms, then map
        its claims onto a :class:`~openbb_agent_server.runtime.principal.UserPrincipal`.

        Parameters
        ----------
        request : fastapi.Request
            Incoming request whose ``Authorization`` header carries the JWT.

        Returns
        -------
        UserPrincipal
            Principal built from the token's ``sub``, name/email claims, and
            extracted scopes, with the full claim set preserved in
            ``raw_claims``.

        Raises
        ------
        fastapi.HTTPException
            401 if the bearer token is missing; 403 if the signing key
            cannot be resolved, the token fails validation, or the ``sub``
            claim is absent.
        """
        auth = request.headers.get("authorization") or ""
        if not auth.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = auth[len("bearer ") :].strip()

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token).key
        except Exception as exc:
            logger.warning("oidc_jwt: JWKS lookup failed: %s", exc)
            raise HTTPException(
                status_code=403, detail="cannot verify signing key"
            ) from exc

        try:
            claims = jwt.decode(
                token,
                key=signing_key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
            )
        except jwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=403, detail=f"invalid token: {exc}"
            ) from exc

        sub = claims.get("sub")
        if not sub:
            raise HTTPException(status_code=403, detail="token missing sub")

        scopes = self._extract_scopes(claims)
        return UserPrincipal(
            user_id=str(sub),
            display_name=claims.get("name") or claims.get("preferred_username"),
            email=claims.get("email"),
            scopes=tuple(scopes),
            raw_claims=dict(claims),
        )

    @staticmethod
    def _extract_scopes(claims: dict[str, Any]) -> tuple[str, ...]:
        raw = claims.get("scope") or claims.get("scopes") or ""
        if isinstance(raw, str):
            return tuple(s for s in raw.split() if s)
        if isinstance(raw, (list, tuple)):
            return tuple(str(s) for s in raw)
        return ()
