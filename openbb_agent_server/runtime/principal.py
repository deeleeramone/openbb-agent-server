"""User principal — the resolved identity attached to every request."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserPrincipal(BaseModel):
    """The authenticated identity for one request.

    A frozen, immutable model produced by an :class:`~openbb_agent_server.runtime.plugins.AuthBackend` and
    attached to the run context so downstream code can identify the caller
    and check authorization. Unknown fields are rejected (``extra="forbid"``).

    Attributes
    ----------
    user_id : str
        Stable, non-empty identifier for the authenticated user.
    display_name : str or None
        Human-readable name, when available.
    email : str or None
        The user's email address, when available.
    scopes : tuple[str, ...]
        Authorization scopes granted to this principal.
    raw_claims : dict[str, Any]
        The original token/claims payload the principal was derived from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(..., min_length=1)
    display_name: str | None = None
    email: str | None = None
    scopes: tuple[str, ...] = ()
    raw_claims: dict[str, Any] = Field(default_factory=dict)

    def has_scope(self, scope: str) -> bool:
        """Return ``True`` iff ``scope`` is granted (exact string match).

        Parameters
        ----------
        scope : str
            The scope to check for membership in :attr:`scopes`.

        Returns
        -------
        bool
            ``True`` when ``scope`` is present in the principal's granted
            scopes, otherwise ``False``. Matching is exact; no wildcard or
            hierarchical expansion is performed.
        """
        return scope in self.scopes
