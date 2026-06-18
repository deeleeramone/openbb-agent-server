"""Tests for :mod:`openbb_agent_server.runtime.jobs`."""

from __future__ import annotations

import asyncio
import warnings

import pytest

from openbb_agent_server.runtime.context import RunContext, bind, runtime_state
from openbb_agent_server.runtime.jobs import (
    WAIT_TIMEOUT,
    JobRegistry,
    JobState,
    get_registry,
)
from openbb_agent_server.runtime.principal import UserPrincipal


@pytest.fixture
def ctx() -> RunContext:
    """Build a real RunContext for binding inside tests."""
    return RunContext(
        principal=UserPrincipal(user_id="u-test"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


@pytest.mark.asyncio
async def test_submit_with_coroutine_runs_to_done(ctx: RunContext) -> None:
    async def work() -> int:
        await asyncio.sleep(0.01)
        return 42

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(work(), label="work")
        out = await reg.wait(job_id, timeout_s=1)

    assert out["state"] == JobState.DONE.value
    assert out["result"] == 42
    assert out["label"] == "work"


@pytest.mark.asyncio
async def test_submit_with_factory_also_runs_to_done(ctx: RunContext) -> None:
    async def work() -> str:
        return "ok"

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(lambda: work(), label="factory")
        out = await reg.wait(job_id, timeout_s=1)

    assert out["state"] == JobState.DONE.value
    assert out["result"] == "ok"


@pytest.mark.asyncio
async def test_submit_rejects_non_awaitable(ctx: RunContext) -> None:
    with bind(ctx), pytest.raises(TypeError):
        get_registry().submit(123, label="nope")


@pytest.mark.asyncio
async def test_status_returns_snapshot_without_result(ctx: RunContext) -> None:
    async def work() -> str:
        await asyncio.sleep(0.01)
        return "done"

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(work(), label="snap", metadata={"k": "v"})
        st = reg.status(job_id)
        assert st["job_id"] == job_id
        assert st["label"] == "snap"
        assert st["state"] in (JobState.RUNNING.value, JobState.DONE.value)
        assert st["metadata"] == {"k": "v"}
        assert "result" not in st
        await reg.wait(job_id, timeout_s=1)


@pytest.mark.asyncio
async def test_status_raises_on_unknown_id(ctx: RunContext) -> None:
    with bind(ctx), pytest.raises(KeyError):
        get_registry().status("bogus")


@pytest.mark.asyncio
async def test_list_all_returns_oldest_first(ctx: RunContext) -> None:
    async def work(n: int) -> int:
        await asyncio.sleep(0.01)
        return n

    with bind(ctx):
        reg = get_registry()
        first = reg.submit(work(1), label="a")
        await asyncio.sleep(0.005)
        second = reg.submit(work(2), label="b")
        all_jobs = reg.list_all()
        ids = [j["job_id"] for j in all_jobs]
        assert ids == [first, second]
        await reg.wait(second, timeout_s=1)


@pytest.mark.asyncio
async def test_wait_returns_timeout_sentinel_when_too_slow(ctx: RunContext) -> None:
    async def slow() -> str:
        await asyncio.sleep(0.5)
        return "late"

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(slow(), label="slow")
        out = await reg.wait(job_id, timeout_s=0.01)
        assert out["state"] == WAIT_TIMEOUT
        full = await reg.wait(job_id, timeout_s=2)
        assert full["state"] == JobState.DONE.value
        assert full["result"] == "late"


@pytest.mark.asyncio
async def test_wait_on_already_finished_returns_done_immediately(
    ctx: RunContext,
) -> None:
    async def quick() -> int:
        return 7

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(quick(), label="q")
        first = await reg.wait(job_id, timeout_s=1)
        assert first["state"] == JobState.DONE.value
        again = await reg.wait(job_id, timeout_s=0.001)
        assert again["state"] == JobState.DONE.value
        assert again["result"] == 7


@pytest.mark.asyncio
async def test_wait_raises_on_unknown_id(ctx: RunContext) -> None:
    with bind(ctx), pytest.raises(KeyError):
        await get_registry().wait("bogus")


@pytest.mark.asyncio
async def test_exception_is_captured_onto_job(ctx: RunContext) -> None:
    async def boom() -> None:
        raise ValueError("kaboom")

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(boom(), label="boom")
        out = await reg.wait(job_id, timeout_s=1)

    assert out["state"] == JobState.ERROR.value
    assert "kaboom" in out["error"]
    assert "ValueError" in out["error"]


@pytest.mark.asyncio
async def test_cancel_before_first_step_with_factory_no_warning(
    ctx: RunContext,
) -> None:
    """Cancelling a factory job before its first step emits no warning."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)

        async def never() -> None:
            await asyncio.sleep(60)

        with bind(ctx):
            reg = get_registry()
            job_id = reg.submit(lambda: never(), label="never")
            assert reg.cancel(job_id) is True
            assert reg.status(job_id)["state"] == JobState.CANCELED.value

        # Yield to the event loop so any un-awaited coroutine warnings fire.
        await asyncio.sleep(0)

    coro_warnings = [w for w in recorded if "coroutine" in str(w.message).lower()]
    assert coro_warnings == [], (
        f"factory path leaked un-awaited coroutines: {coro_warnings}"
    )


@pytest.mark.asyncio
async def test_cancel_mid_run(ctx: RunContext) -> None:
    """Cancel a running task and transition it to CANCELED."""

    async def slow() -> str:
        await asyncio.sleep(60)
        return "never"

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(slow(), label="mid")
        await asyncio.sleep(0.01)
        assert reg.cancel(job_id) is True
        await asyncio.sleep(0.05)
        assert reg.status(job_id)["state"] == JobState.CANCELED.value


@pytest.mark.asyncio
async def test_cancel_already_done_returns_false(ctx: RunContext) -> None:
    async def quick() -> None:
        return None

    with bind(ctx):
        reg = get_registry()
        job_id = reg.submit(quick(), label="q")
        await reg.wait(job_id, timeout_s=1)
        assert reg.cancel(job_id) is False


def test_cancel_on_unknown_id_returns_false() -> None:
    reg = JobRegistry()
    assert reg.cancel("bogus") is False


@pytest.mark.asyncio
async def test_cancel_all_cancels_every_running_job(ctx: RunContext) -> None:
    async def sleepy() -> None:
        await asyncio.sleep(60)

    with bind(ctx):
        reg = get_registry()
        for i in range(3):
            reg.submit(lambda: sleepy(), label=f"j{i}")
        assert reg.cancel_all() == 3
        assert reg.cancel_all() == 0


@pytest.mark.asyncio
async def test_bind_finally_cancels_outstanding_jobs(ctx: RunContext) -> None:
    """Leaving bind() cancels every still-running background job."""

    captured: dict[str, JobRegistry] = {}

    async def never() -> None:
        await asyncio.sleep(60)

    with bind(ctx):
        reg = get_registry()
        captured["reg"] = reg
        reg.submit(lambda: never(), label="leak")

    [job] = captured["reg"].list_all()
    assert job["state"] == JobState.CANCELED.value


@pytest.mark.asyncio
async def test_get_registry_outside_bind_raises() -> None:
    with pytest.raises(LookupError):
        get_registry()


@pytest.mark.asyncio
async def test_get_registry_is_idempotent_inside_bind(ctx: RunContext) -> None:
    with bind(ctx):
        a = get_registry()
        b = get_registry()
        assert a is b
        from openbb_agent_server.runtime.jobs import _STATE_KEY

        assert runtime_state()[_STATE_KEY] is a


@pytest.mark.asyncio
async def test_cleanup_state_with_no_registry_returns_zero() -> None:
    from openbb_agent_server.runtime.jobs import cleanup_state

    assert cleanup_state({}) == 0


@pytest.mark.asyncio
async def test_cleanup_state_cancels_running_jobs(ctx: RunContext) -> None:
    from openbb_agent_server.runtime.jobs import _STATE_KEY, cleanup_state

    async def never() -> None:
        await asyncio.sleep(60)

    reg = JobRegistry()
    reg.submit(lambda: never(), label="x")
    reg.submit(lambda: never(), label="y")
    n = cleanup_state({_STATE_KEY: reg})
    assert n == 2
