"""RunContext contextvar binding tests."""

from __future__ import annotations

import pytest

from openbb_agent_server.runtime import context as run_context
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _make_ctx(user_id: str = "u1") -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id=user_id),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


def test_unbound_lookup_raises() -> None:
    with pytest.raises(LookupError):
        run_context.current()


def test_bind_then_current() -> None:
    ctx = _make_ctx()
    with run_context.bind(ctx):
        assert run_context.current() is ctx


def test_bind_restores_previous_state() -> None:
    with pytest.raises(LookupError):
        run_context.current()
    with run_context.bind(_make_ctx("a")):
        with run_context.bind(_make_ctx("b")):
            assert run_context.current().principal.user_id == "b"
        assert run_context.current().principal.user_id == "a"
    with pytest.raises(LookupError):
        run_context.current()


def test_runtime_state_raises_when_no_run_bound() -> None:
    from openbb_agent_server.runtime.context import runtime_state

    with pytest.raises(LookupError, match="runtime state"):
        runtime_state()


def test_has_workspace_option_truthy_value_is_enabled() -> None:
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        workspace_options={"search-web": True, "model": "gpt-4o"},
    )
    assert ctx.has_workspace_option("search-web") is True
    assert ctx.has_workspace_option("model") is True


def test_has_workspace_option_false_value_is_disabled() -> None:
    """A toggle the user left off arrives as ``False`` — not enabled."""
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        workspace_options={"search-web": False},
    )
    assert ctx.has_workspace_option("search-web") is False


def test_has_workspace_option_absent_key_is_disabled() -> None:
    assert _make_ctx().has_workspace_option("search-web") is False
