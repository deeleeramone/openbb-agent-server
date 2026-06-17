"""Stable hashing helper that maps emails / external IDs to ``user_id`` strings."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re

logger = logging.getLogger("openbb_agent_server.runtime.identity")

_PEPPER_ENV = "OPENBB_AGENT_USER_ID_PEPPER"
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,}\b")
_HASH_PREFIX = "u-"
_HASH_BYTES = 12


def _pepper() -> bytes:
    return os.environ.get(_PEPPER_ENV, "").encode("utf-8")


def warn_if_pepper_unset() -> None:
    """Log a startup WARNING when the user-id pepper is unset."""
    if not os.environ.get(_PEPPER_ENV):
        logger.warning(
            "%s is not set — falling back to an empty pepper. Set a stable "
            "secret value before going to production; rotating it later "
            "orphans every user's persisted data.",
            _PEPPER_ENV,
        )


def hash_user_id(value: str) -> str:
    """Return a stable opaque ``user_id`` for an email or external identifier.

    The input is normalized (stripped and lower-cased) and run through an
    HMAC-SHA256 keyed by the ``OPENBB_AGENT_USER_ID_PEPPER`` pepper. The
    first ``_HASH_BYTES`` bytes of the digest are hex-encoded and prefixed
    with ``u-`` to form the identifier. The mapping is deterministic for a
    fixed pepper, so the same email always yields the same ``user_id``.

    Parameters
    ----------
    value : str
        An email address or external identifier. Leading/trailing
        whitespace and case are ignored.

    Returns
    -------
    str
        The opaque, pepper-keyed ``user_id`` (e.g. ``"u-1a2b3c..."``).

    Raises
    ------
    ValueError
        If ``value`` is empty or whitespace-only after normalization.
    """
    normalized = (value or "").strip().lower()
    if not normalized:
        raise ValueError("hash_user_id: value must be non-empty")
    digest = hmac.new(_pepper(), normalized.encode("utf-8"), hashlib.sha256).digest()
    return f"{_HASH_PREFIX}{digest.hex()[: 2 * _HASH_BYTES]}"


def is_email(value: str) -> bool:
    """Return True iff ``value`` looks like an RFC-5321-ish email address.

    The check is a full-match against a permissive email regex after
    stripping surrounding whitespace; it does not validate deliverability.

    Parameters
    ----------
    value : str
        The candidate string to test. An empty or ``None``-like value
        returns ``False``.

    Returns
    -------
    bool
        ``True`` if ``value`` fully matches the email pattern, else
        ``False``.
    """
    if not value:
        return False
    return _EMAIL_RE.fullmatch(value.strip()) is not None


def redact_email_in_text(text: str) -> str:
    """Replace every email address in ``text`` with its ``user_id`` hash.

    Each email-looking substring is substituted in place by the opaque
    identifier from :func:`hash_user_id`, leaving the rest of the text
    intact. Useful for scrubbing PII from logs or model-visible content.

    Parameters
    ----------
    text : str
        Arbitrary text that may contain one or more email addresses. An
        empty or falsy value is returned unchanged.

    Returns
    -------
    str
        ``text`` with every matched email replaced by its hashed
        ``user_id``.
    """
    if not text:
        return text
    return _EMAIL_RE.sub(lambda m: hash_user_id(m.group(0)), text)


__all__ = [
    "hash_user_id",
    "is_email",
    "redact_email_in_text",
    "warn_if_pepper_unset",
]
