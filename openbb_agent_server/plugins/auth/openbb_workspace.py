"""OpenBB Workspace auth backend."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from openbb_agent_server.runtime.identity import hash_user_id, is_email
from openbb_agent_server.runtime.plugins import AuthBackend
from openbb_agent_server.runtime.principal import UserPrincipal

logger = logging.getLogger("openbb_agent_server.auth.openbb_workspace")

DEFAULT_HEADER = "X-OpenBB-User"

DEFAULT_SCOPES: tuple[str, ...] = (
    "agent:query",
    "memory:read",
    "memory:write",
)


class OpenBBWorkspaceAuthBackend(AuthBackend):
    """Trust the X-OpenBB-User header from an upstream Workspace gateway.

    This backend assumes a trusted reverse proxy (the OpenBB Workspace
    gateway) has already authenticated the caller and forwards the user's
    identity in a request header. It performs no token verification of its
    own, so it must only be used behind such a gateway.
    """

    name = "openbb_workspace"

    def __init__(
        self,
        *,
        header: str = DEFAULT_HEADER,
        scopes: tuple[str, ...] = DEFAULT_SCOPES,
        require_email: bool = True,
    ) -> None:
        """Configure the header name, granted scopes, and email policy.

        Parameters
        ----------
        header : str, optional
            Name of the request header carrying the user identity. Defaults
            to ``X-OpenBB-User``.
        scopes : tuple[str, ...], optional
            Scopes granted to every authenticated principal. Defaults to
            ``agent:query``, ``memory:read``, ``memory:write``.
        require_email : bool, optional
            If ``True`` (default), reject requests whose header value is not
            a valid email address with HTTP 403.
        """
        self._header = header
        self._scopes = tuple(scopes)
        self._require_email = require_email

    async def authenticate(self, request: Request) -> UserPrincipal:
        """Build a :class:`~openbb_agent_server.runtime.principal.UserPrincipal` from the trusted identity header.

        The header value is stripped and lower-cased. When ``require_email``
        is set, a value that is not a valid email is rejected. The cleaned
        value is hashed into an opaque ``user_id`` and, if it is an email,
        retained as the principal's ``email``.

        Parameters
        ----------
        request : fastapi.Request
            The incoming request whose identity header is inspected.

        Returns
        -------
        UserPrincipal
            A principal carrying the hashed ``user_id``, optional ``email``,
            and the configured scopes.

        Raises
        ------
        fastapi.HTTPException
            401 if the header is missing or empty; 403 if ``require_email``
            is set and the value is not a valid email address.
        """
        raw = request.headers.get(self._header)
        if not raw:
            raise HTTPException(
                status_code=401,
                detail=f"missing required header {self._header}",
            )
        cleaned = raw.strip().lower()
        if not cleaned:
            raise HTTPException(
                status_code=401,
                detail=f"empty {self._header} header",
            )
        if self._require_email and not is_email(cleaned):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"{self._header} must be an email address; got a value "
                    "that doesn't match RFC-5321"
                ),
            )
        email = cleaned if is_email(cleaned) else None
        user_id = hash_user_id(cleaned)
        return UserPrincipal(
            user_id=user_id,
            email=email,
            scopes=self._scopes,
        )
