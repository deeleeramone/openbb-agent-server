"""Single shared secret, single-user dev auth backend."""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request

from openbb_agent_server.runtime.plugins import AuthBackend
from openbb_agent_server.runtime.principal import UserPrincipal

logger = logging.getLogger("openbb_agent_server.auth.bearer_static")


class BearerStaticAuthBackend(AuthBackend):
    """Authenticate every request against one shared bearer token.

    Intended for development and testing only: any valid token maps to a
    single fixed user identity. Use the ``api_key_table`` or ``oidc_jwt``
    backend in production.
    """

    name = "bearer_static"

    def __init__(
        self,
        *,
        token: str | None = None,
        user_id: str = "static-user",
        display_name: str | None = None,
        scopes: tuple[str, ...] = ("agent:query", "memory:read", "memory:write"),
    ) -> None:
        """Resolve and store the shared token plus the static principal.

        Parameters
        ----------
        token : str or None, optional
            The shared bearer token. When ``None``, it is read from the
            ``OPENBB_AGENT_AUTH_BEARER`` environment variable.
        user_id : str, default "static-user"
            Identifier assigned to every authenticated principal.
        display_name : str or None, optional
            Human-readable name attached to the principal.
        scopes : tuple of str, optional
            Authorization scopes granted to every authenticated request.
            Defaults to query plus memory read/write.

        Raises
        ------
        RuntimeError
            If no token is supplied via ``token`` or the environment.
        """
        token = token or os.environ.get("OPENBB_AGENT_AUTH_BEARER")
        if not token:
            raise RuntimeError(
                "BearerStaticAuthBackend requires a non-empty token "
                "(``token`` config key or OPENBB_AGENT_AUTH_BEARER env var)."
            )
        self._token = token
        self._user_id = user_id
        self._display_name = display_name
        self._scopes = scopes
        logger.warning(
            "BearerStaticAuthBackend resolves every authenticated request "
            "to user '%s'. Use the api_key_table or oidc_jwt backend in "
            "production.",
            user_id,
        )

    async def authenticate(self, request: Request) -> UserPrincipal:
        """Validate the request's bearer token and return the principal.

        Read the ``Authorization`` header, require a ``Bearer`` scheme,
        and compare the supplied token against the configured one with a
        constant-time comparison.

        Parameters
        ----------
        request : Request
            The incoming FastAPI request whose ``Authorization`` header
            carries the bearer token.

        Returns
        -------
        UserPrincipal
            The fixed static principal (user id, display name, scopes)
            when the token matches.

        Raises
        ------
        HTTPException
            With status 401 if the bearer token is missing, or 403 if the
            supplied token does not match the configured token.
        """
        header = request.headers.get("authorization") or ""
        if not header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        supplied = header[len("bearer ") :].strip()
        if not hmac.compare_digest(supplied, self._token):
            raise HTTPException(status_code=403, detail="invalid bearer token")
        return UserPrincipal(
            user_id=self._user_id,
            display_name=self._display_name,
            scopes=self._scopes,
        )
