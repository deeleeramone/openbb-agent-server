"""Tests for the runtime identity module."""

from __future__ import annotations

import pytest

from openbb_agent_server.runtime import identity
from openbb_agent_server.runtime.identity import (
    hash_user_id,
    is_email,
    redact_email_in_text,
)


def test_is_email_accepts_simple_addresses() -> None:
    assert is_email("alice@example.com")
    assert is_email("a.b+tag@host.subdomain.io")


def test_is_email_rejects_garbage() -> None:
    assert not is_email("alice")
    assert not is_email("alice@")
    assert not is_email("@example.com")
    assert not is_email("")
    assert not is_email("alice@.com")
    assert not is_email("alice@example")


def test_is_email_strips_whitespace() -> None:
    assert is_email("  alice@example.com  ")


def test_hash_user_id_returns_opaque_prefix() -> None:
    h = hash_user_id("alice@example.com")
    assert h.startswith("u-")
    assert "@" not in h
    assert "alice" not in h
    assert len(h) == 2 + 24


def test_hash_user_id_is_stable() -> None:
    a = hash_user_id("alice@example.com")
    b = hash_user_id("alice@example.com")
    assert a == b


def test_hash_user_id_lowercases_and_trims() -> None:
    a = hash_user_id("Alice@Example.COM")
    b = hash_user_id("  alice@example.com  ")
    c = hash_user_id("alice@example.com")
    assert a == b == c


def test_hash_user_id_differs_for_different_inputs() -> None:
    assert hash_user_id("alice@example.com") != hash_user_id("bob@example.com")


def test_hash_user_id_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hash_user_id("")
    with pytest.raises(ValueError, match="non-empty"):
        hash_user_id("   ")


def test_hash_user_id_uses_env_pepper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBB_AGENT_USER_ID_PEPPER", "first-pepper")
    a = hash_user_id("alice@example.com")
    monkeypatch.setenv("OPENBB_AGENT_USER_ID_PEPPER", "second-pepper")
    b = hash_user_id("alice@example.com")
    assert a != b


def test_warn_if_pepper_unset_logs_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The startup check warns once when the pepper env var is unset."""
    import logging

    monkeypatch.delenv("OPENBB_AGENT_USER_ID_PEPPER", raising=False)
    with caplog.at_level(logging.WARNING):
        identity.warn_if_pepper_unset()
    assert any("PEPPER" in r.message for r in caplog.records)


def test_warn_if_pepper_unset_silent_when_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The startup check stays silent when the pepper is configured."""
    import logging

    monkeypatch.setenv("OPENBB_AGENT_USER_ID_PEPPER", "a-stable-secret")
    with caplog.at_level(logging.WARNING):
        identity.warn_if_pepper_unset()
    assert not any("PEPPER" in r.message for r in caplog.records)


def test_redact_email_in_text_replaces_with_hash() -> None:
    out = redact_email_in_text("contact alice@example.com for details")
    assert "alice@example.com" not in out
    assert "u-" in out


def test_redact_email_in_text_preserves_non_email() -> None:
    assert redact_email_in_text("no emails here") == "no emails here"


def test_redact_email_in_text_handles_empty() -> None:
    assert redact_email_in_text("") == ""
    assert redact_email_in_text(None) is None  # type: ignore[arg-type]


def test_redact_email_in_text_replaces_multiple() -> None:
    out = redact_email_in_text("alice@x.com and bob@y.com")
    assert "alice@x.com" not in out
    assert "bob@y.com" not in out
    assert out.count("u-") == 2
