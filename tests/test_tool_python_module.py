"""python_module tool source tests."""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool, tool

from openbb_agent_server.plugins.tools.python_module import (
    PythonModuleToolSource,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


@tool
def _hello(name: str) -> str:
    """Say hello."""
    return f"hello, {name}"


_HELLO_LIST = [_hello]


def _factory_tool() -> StructuredTool:
    return _hello  # type: ignore[return-value]


_BARE_INSTANCE = _hello


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


@pytest.mark.asyncio
async def test_loads_single_basetool_instance() -> None:
    src = PythonModuleToolSource(
        modules=["tests.test_tool_python_module:_BARE_INSTANCE"]
    )
    tools = await src.tools(_ctx(), {})
    assert len(tools) == 1
    assert tools[0].name == "_hello"


@pytest.mark.asyncio
async def test_loads_list_of_tools() -> None:
    src = PythonModuleToolSource(modules=["tests.test_tool_python_module:_HELLO_LIST"])
    tools = await src.tools(_ctx(), {})
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_loads_factory_callable() -> None:
    src = PythonModuleToolSource(
        modules=["tests.test_tool_python_module:_factory_tool"]
    )
    tools = await src.tools(_ctx(), {})
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_invalid_spec_format_raises() -> None:
    src = PythonModuleToolSource(modules=["no_colon_in_spec"])
    with pytest.raises(ValueError):
        await src.tools(_ctx(), {})


@pytest.mark.asyncio
async def test_unsupported_resolved_type_raises() -> None:
    src = PythonModuleToolSource(modules=["builtins:dict"])
    with pytest.raises(TypeError):
        await src.tools(_ctx(), {})


def test_python_module_flatten_keeps_bad_factory_intact() -> None:
    from openbb_agent_server.plugins.tools.python_module import _flatten

    def factory_with_required_arg(x):  # noqa: ANN001
        return x

    with pytest.raises(TypeError):
        _flatten(factory_with_required_arg)
