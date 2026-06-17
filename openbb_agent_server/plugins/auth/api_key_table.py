"""API key table auth backend."""

from __future__ import annotations

import datetime as _dt
import logging
import secrets
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from openbb_agent_server.persistence import models as m
from openbb_agent_server.runtime.plugins import AuthBackend
from openbb_agent_server.runtime.principal import UserPrincipal

logger = logging.getLogger("openbb_agent_server.auth.api_key_table")

KEY_PREFIX = "oba_"
KEY_SEP = "."


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class IssuedKey:
    """Plaintext key and its row metadata.

    Returned by :meth:`ApiKeyTableAuthBackend.issue`. The plaintext key is only
    available at issue time and is never persisted; only the Argon2 hash of its
    secret half is stored.

    Attributes
    ----------
    plaintext : str
        The full key string to hand to the caller, formatted as
        ``oba_<key_id>.<secret>``. Cannot be recovered later.
    key_id : str
        Stable identifier of the key row, used for revocation and lookup.
    user_id : str
        Identifier of the user the key authenticates as.
    scopes : tuple[str, ...]
        Authorization scopes granted to the key.
    label : str or None
        Optional human-readable label for the key.
    """

    __slots__ = ("plaintext", "key_id", "user_id", "scopes", "label")

    def __init__(
        self,
        *,
        plaintext: str,
        key_id: str,
        user_id: str,
        scopes: tuple[str, ...],
        label: str | None,
    ) -> None:
        """Store the plaintext key and its associated row metadata.

        Parameters
        ----------
        plaintext : str
            Full key string formatted as ``oba_<key_id>.<secret>``.
        key_id : str
            Stable identifier of the key row.
        user_id : str
            Identifier of the user the key authenticates as.
        scopes : tuple[str, ...]
            Authorization scopes granted to the key.
        label : str or None
            Optional human-readable label for the key.
        """
        self.plaintext = plaintext
        self.key_id = key_id
        self.user_id = user_id
        self.scopes = scopes
        self.label = label


class ApiKeyTableAuthBackend(AuthBackend):
    """Hashed API keys keyed off the ``api_keys`` table."""

    name = "api_key_table"

    def __init__(self, *, db_url: str) -> None:
        """Create the async engine, session factory, and Argon2 hasher.

        Parameters
        ----------
        db_url : str
            SQLAlchemy async database URL backing the ``api_keys`` and
            ``users`` tables (for example, an ``sqlite+aiosqlite`` or
            ``postgresql+asyncpg`` DSN).
        """
        self._engine = create_async_engine(db_url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        self._hasher = PasswordHasher()

    @staticmethod
    def _extract_key(request: Request) -> str | None:
        header = request.headers.get("x-api-key")
        if header:
            return header.strip()
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth[len("bearer ") :].strip()
        return None

    @staticmethod
    def _split(raw: str) -> tuple[str, str]:
        if not (raw.startswith(KEY_PREFIX) and KEY_SEP in raw):
            raise HTTPException(status_code=403, detail="invalid api key")
        key_id, secret = raw[len(KEY_PREFIX) :].split(KEY_SEP, 1)
        if not key_id or not secret:
            raise HTTPException(status_code=403, detail="invalid api key")
        return key_id, secret

    async def authenticate(self, request: Request) -> UserPrincipal:
        """Authenticate a request by its API key and return the principal.

        Extract the key from the ``x-api-key`` header or a bearer
        ``Authorization`` header, split it into id and secret, look up the
        matching non-revoked row, verify the secret against the stored Argon2
        hash, and resolve the owning user.

        Parameters
        ----------
        request : fastapi.Request
            Incoming request whose headers carry the API key.

        Returns
        -------
        UserPrincipal
            Principal for the authenticated user, carrying the scopes granted
            to the matched key.

        Raises
        ------
        fastapi.HTTPException
            ``401`` if no key is present; ``403`` if the key is malformed,
            unknown, revoked, fails secret verification, or its user is missing.
        """
        raw = self._extract_key(request)
        if not raw:
            raise HTTPException(status_code=401, detail="missing api key")
        key_id, secret = self._split(raw)

        async with self._sessionmaker() as session:
            row = await session.get(m.ApiKey, key_id)
            if row is None or row.revoked_at is not None:
                raise HTTPException(status_code=403, detail="invalid api key")
            try:
                self._hasher.verify(row.hashed_secret, secret)
            except VerifyMismatchError as exc:
                raise HTTPException(status_code=403, detail="invalid api key") from exc
            user = await session.get(m.User, row.user_id)
            if user is None:
                raise HTTPException(status_code=403, detail="invalid api key")
            return UserPrincipal(
                user_id=user.user_id,
                display_name=user.display_name,
                email=user.email,
                scopes=tuple(row.scopes or ()),
            )

    async def aclose(self) -> None:
        """Dispose of the underlying async engine and its connection pool."""
        await self._engine.dispose()

    async def issue(
        self,
        *,
        user_id: str,
        scopes: tuple[str, ...] = ("agent:query", "memory:read"),
        label: str | None = None,
        display_name: str | None = None,
        email: str | None = None,
    ) -> IssuedKey:
        """Mint a key, upsert the user row, and return the plaintext key.

        Generate a random key id and secret, hash the secret with Argon2, and
        persist a new ``api_keys`` row. If the user does not exist it is
        created; otherwise any missing display name or email is filled in and
        the last-seen timestamp is refreshed. The plaintext key is returned
        once and never stored.

        Parameters
        ----------
        user_id : str
            Identifier of the user the key belongs to. Created if absent.
        scopes : tuple[str, ...], optional
            Authorization scopes granted to the key. Defaults to
            ``("agent:query", "memory:read")``.
        label : str or None, optional
            Optional human-readable label stored with the key.
        display_name : str or None, optional
            Display name used when creating the user, or to backfill a missing
            display name on an existing user.
        email : str or None, optional
            Email used when creating the user, or to backfill a missing email
            on an existing user.

        Returns
        -------
        IssuedKey
            The issued key, including the one-time plaintext value.
        """
        key_id = secrets.token_urlsafe(8)
        secret = secrets.token_urlsafe(32)
        hashed = self._hasher.hash(secret)
        async with self._sessionmaker() as session:
            user = await session.get(m.User, user_id)
            if user is None:
                user = m.User(
                    user_id=user_id,
                    display_name=display_name,
                    email=email,
                )
                session.add(user)
            else:
                if display_name and not user.display_name:
                    user.display_name = display_name
                if email and not user.email:
                    user.email = email
                user.last_seen_at = _now()
            session.add(
                m.ApiKey(
                    key_id=key_id,
                    user_id=user_id,
                    hashed_secret=hashed,
                    label=label,
                    scopes=list(scopes),
                )
            )
            await session.commit()
        return IssuedKey(
            plaintext=f"{KEY_PREFIX}{key_id}{KEY_SEP}{secret}",
            key_id=key_id,
            user_id=user_id,
            scopes=tuple(scopes),
            label=label,
        )

    async def revoke(self, *, key_id: str) -> bool:
        """Revoke a key by stamping its ``revoked_at`` timestamp.

        Parameters
        ----------
        key_id : str
            Identifier of the key row to revoke.

        Returns
        -------
        bool
            ``True`` if the key existed and was revoked; ``False`` if no row
            matched ``key_id``.
        """
        async with self._sessionmaker() as session:
            row = await session.get(m.ApiKey, key_id)
            if row is None:
                return False
            row.revoked_at = _now()
            await session.commit()
            return True

    async def list_keys(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return non-secret metadata for keys.

        Parameters
        ----------
        user_id : str or None, optional
            If given, restrict the result to keys owned by this user;
            otherwise return metadata for all keys.

        Returns
        -------
        list[dict[str, Any]]
            One dict per key with ``key_id``, ``user_id``, ``label``,
            ``scopes``, and ISO-formatted ``created_at`` and ``revoked_at``
            (the timestamps are ``None`` when unset). Secrets and hashes are
            never included.
        """
        async with self._sessionmaker() as session:
            stmt = select(m.ApiKey)
            if user_id:
                stmt = stmt.where(m.ApiKey.user_id == user_id)
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "key_id": r.key_id,
                    "user_id": r.user_id,
                    "label": r.label,
                    "scopes": list(r.scopes or ()),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "revoked_at": (r.revoked_at.isoformat() if r.revoked_at else None),
                }
                for r in rows
            ]
