"""Tests for the ``background_jobs`` ToolSource."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import BaseTool

from openbb_agent_server.plugins.tools.background_jobs import (
    BackgroundJobsToolSource,
)
from openbb_agent_server.runtime.context import RunContext, bind
from openbb_agent_server.runtime.jobs import JobState, get_registry
from openbb_agent_server.runtime.principal import UserPrincipal


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u-bg"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


async def _by_name(
    src: BackgroundJobsToolSource, name: str, ctx: RunContext
) -> BaseTool:
    tools = await src.tools(ctx, {})
    [tool] = [t for t in tools if t.name == name]
    return tool


@pytest.mark.asyncio
async def test_tool_source_returns_four_tools(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    tools = await src.tools(ctx, {})
    names = {t.name for t in tools}
    assert names == {
        "list_background_jobs",
        "check_job",
        "wait_for_job",
        "cancel_job",
    }


@pytest.mark.asyncio
async def test_list_returns_every_submitted_job(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    list_tool = await _by_name(src, "list_background_jobs", ctx)

    async def work() -> int:
        return 1

    with bind(ctx):
        reg = get_registry()
        reg.submit(work(), label="a")
        reg.submit(work(), label="b")
        out = await list_tool.ainvoke({})
        assert {row["label"] for row in out} == {"a", "b"}


@pytest.mark.asyncio
async def test_check_returns_running_snapshot(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    check_tool = await _by_name(src, "check_job", ctx)

    async def slow() -> int:
        await asyncio.sleep(0.2)
        return 1

    with bind(ctx):
        job_id = get_registry().submit(slow(), label="slow")
        out = await check_tool.ainvoke({"job_id": job_id})
        assert out["state"] in (JobState.RUNNING.value, JobState.DONE.value)
        assert "result" not in out
        await get_registry().wait(job_id, timeout_s=1)


@pytest.mark.asyncio
async def test_check_unknown_job_raises_value_error(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    check_tool = await _by_name(src, "check_job", ctx)

    with bind(ctx):
        with pytest.raises(Exception) as exc_info:
            await check_tool.ainvoke({"job_id": "bogus"})
    assert "bogus" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_job_returns_done_with_result(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    wait_tool = await _by_name(src, "wait_for_job", ctx)

    async def quick() -> dict[str, int]:
        return {"answer": 42}

    with bind(ctx):
        job_id = get_registry().submit(quick(), label="q")
        out = await wait_tool.ainvoke({"job_id": job_id, "timeout_s": 1})
        assert out["state"] == JobState.DONE.value
        assert out["result"] == {"answer": 42}


@pytest.mark.asyncio
async def test_wait_for_job_unknown_id_raises(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    wait_tool = await _by_name(src, "wait_for_job", ctx)

    with bind(ctx):
        with pytest.raises(Exception) as exc_info:
            await wait_tool.ainvoke({"job_id": "bogus", "timeout_s": 1})
    assert "bogus" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cancel_job_returns_true_when_cancellable(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    cancel_tool = await _by_name(src, "cancel_job", ctx)

    async def never() -> None:
        await asyncio.sleep(60)

    with bind(ctx):
        job_id = get_registry().submit(lambda: never(), label="never")
        out = await cancel_tool.ainvoke({"job_id": job_id})
        assert out == {"job_id": job_id, "cancelled": True}
        assert get_registry().status(job_id)["state"] == JobState.CANCELED.value


@pytest.mark.asyncio
async def test_cancel_job_returns_false_when_already_done(ctx: RunContext) -> None:
    src = BackgroundJobsToolSource()
    cancel_tool = await _by_name(src, "cancel_job", ctx)

    async def quick() -> None:
        return None

    with bind(ctx):
        job_id = get_registry().submit(quick(), label="q")
        await get_registry().wait(job_id, timeout_s=1)
        out = await cancel_tool.ainvoke({"job_id": job_id})
        assert out == {"job_id": job_id, "cancelled": False}


def test_tool_source_is_registered_as_entry_point() -> None:
    """Confirm the background_jobs group entry points at this class."""
    from importlib.metadata import entry_points

    eps = {
        e.name: e.load()
        for e in entry_points(group="openbb_agent_server.tools")
        if e.name == "background_jobs"
    }
    assert eps == {"background_jobs": BackgroundJobsToolSource}
