"""call_limit + tool_call_limit middleware factory tests."""

from __future__ import annotations

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

from openbb_agent_server.plugins.middleware.call_limit import (
    CallLimitMiddlewareFactory,
    ToolCallLimitMiddlewareFactory,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


def test_call_limit_factory_uses_constructor_defaults() -> None:
    factory = CallLimitMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    assert isinstance(mw, ModelCallLimitMiddleware)
    assert mw.run_limit == 40
    assert mw.exit_behavior == "end"


def test_call_limit_factory_constructor_overrides() -> None:
    factory = CallLimitMiddlewareFactory(model_run_limit=5, exit_behavior="error")
    mw = factory.build(_ctx(), {})
    assert mw.run_limit == 5
    assert mw.exit_behavior == "error"


def test_call_limit_factory_per_call_config_wins_over_constructor() -> None:
    factory = CallLimitMiddlewareFactory(model_run_limit=5)
    mw = factory.build(_ctx(), {"model_run_limit": 99, "exit_behavior": "error"})
    assert mw.run_limit == 99
    assert mw.exit_behavior == "error"


def test_call_limit_name_attribute() -> None:
    assert CallLimitMiddlewareFactory.name == "call_limit"


def test_tool_call_limit_factory_uses_constructor_defaults() -> None:
    factory = ToolCallLimitMiddlewareFactory()
    mw = factory.build(_ctx(), {})
    assert isinstance(mw, ToolCallLimitMiddleware)
    assert mw.run_limit == 80
    assert mw.exit_behavior == "end"


def test_tool_call_limit_factory_constructor_overrides() -> None:
    factory = ToolCallLimitMiddlewareFactory(tool_run_limit=8, exit_behavior="error")
    mw = factory.build(_ctx(), {})
    assert mw.run_limit == 8
    assert mw.exit_behavior == "error"


def test_tool_call_limit_factory_per_call_config_wins_over_constructor() -> None:
    factory = ToolCallLimitMiddlewareFactory(tool_run_limit=8)
    mw = factory.build(_ctx(), {"tool_run_limit": 100, "exit_behavior": "error"})
    assert mw.run_limit == 100
    assert mw.exit_behavior == "error"


def test_tool_call_limit_name_attribute() -> None:
    assert ToolCallLimitMiddlewareFactory.name == "tool_call_limit"
