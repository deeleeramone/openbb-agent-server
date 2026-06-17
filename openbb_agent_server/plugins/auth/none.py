"""No-auth, single-user dev backend."""

from __future__ import annotations

import logging

from fastapi import Request

from openbb_agent_server.runtime.plugins import AuthBackend
from openbb_agent_server.runtime.principal import UserPrincipal

logger = logging.getLogger("openbb_agent_server.auth.none")

ANONYMOUS_USER_ID = "anonymous"


class NoneAuthBackend(AuthBackend):
    """Resolve every request to a single shared anonymous user.

    Development-only auth backend that performs no authentication. Every
    request is treated as the same user, so it must never be enabled in
    production.
    """

    name = "none"

    def __init__(self, **_config: object) -> None:
        """Initialise the backend and log a one-time security warning.

        Parameters
        ----------
        **_config : object
            Ignored configuration accepted for interface compatibility
            with other auth backends.
        """
        logger.warning(
            "NoneAuthBackend is enabled — every request resolves to the "
            "single shared user '%s'. Do not use in production.",
            ANONYMOUS_USER_ID,
        )

    async def authenticate(self, request: Request) -> UserPrincipal:
        """Return the shared anonymous principal, ignoring the request.

        Parameters
        ----------
        request : Request
            Incoming request; not inspected by this backend.

        Returns
        -------
        UserPrincipal
            The fixed anonymous user with the ``agent:query`` and
            ``memory:read`` scopes.
        """
        return UserPrincipal(
            user_id=ANONYMOUS_USER_ID,
            display_name="Anonymous",
            scopes=("agent:query", "memory:read"),
        )
