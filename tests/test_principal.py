"""UserPrincipal contract tests."""

from __future__ import annotations

import pytest

from openbb_agent_server.runtime.principal import UserPrincipal


def test_principal_minimum_fields() -> None:
    p = UserPrincipal(user_id="u1")
    assert p.user_id == "u1"
    assert p.scopes == ()
    assert p.display_name is None


def test_principal_is_frozen() -> None:
    p = UserPrincipal(user_id="u1")
    with pytest.raises(Exception):
        p.user_id = "u2"  # type: ignore[misc]


def test_principal_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        UserPrincipal(user_id="u1", surprise="x")  # type: ignore[call-arg]


def test_principal_rejects_empty_user_id() -> None:
    with pytest.raises(Exception):
        UserPrincipal(user_id="")


def test_has_scope_exact_match() -> None:
    p = UserPrincipal(user_id="u1", scopes=("memory:read", "agent:query"))
    assert p.has_scope("memory:read")
    assert p.has_scope("agent:query")
    assert not p.has_scope("memory:write")
    assert not p.has_scope("memory")
