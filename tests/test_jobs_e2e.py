"""End-to-end tests for the background-job pipeline."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from openbb_agent_server.plugins.tools._media import fetch_url
from openbb_agent_server.runtime.context import RunContext, bind
from openbb_agent_server.runtime.jobs import (
    WAIT_TIMEOUT,
    JobRegistry,
    JobState,
    get_registry,
)
from openbb_agent_server.runtime.principal import UserPrincipal
from tests.test_media_integration import (
    _free_port,
    _ServerHandle as _BaseServer,
)


def _build_e2e_app() -> Starlette:
    async def fixed(request: Any) -> Response:
        n = int(request.path_params["ms"])
        await asyncio.sleep(n / 1000.0)
        return Response(content=b"ok", media_type="text/plain")

    async def cancellable(request: Any) -> StreamingResponse:
        """Drip-feed bytes until the client disconnects."""

        async def gen() -> AsyncIterator[bytes]:
            while not await request.is_disconnected():
                yield b"X" * 64
                await asyncio.sleep(0.05)

        return StreamingResponse(gen(), media_type="application/octet-stream")

    return Starlette(
        routes=[
            Route("/sleep/{ms:int}", fixed),
            Route("/cancellable", cancellable),
        ]
    )


class _E2EServer(_BaseServer):
    async def start(self) -> None:
        config = uvicorn.Config(
            app=_build_e2e_app(),
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(200):
            if self._server.started:
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("e2e HTTP server never reached started=True")


@pytest.fixture
async def http_server() -> AsyncIterator[_E2EServer]:
    handle = _E2EServer("127.0.0.1", _free_port())
    await handle.start()
    try:
        yield handle
    finally:
        await handle.stop()


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u-e2e"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


async def test_submit_real_fetch_and_collect(
    ctx: RunContext, http_server: _E2EServer
) -> None:
    with bind(ctx):
        reg = get_registry()
        url = f"{http_server.base}/sleep/50"
        job_id = reg.submit(
            lambda: fetch_url(url, max_bytes=1024, timeout_s=5.0),
            label="fetch",
        )
        out = await reg.wait(job_id, timeout_s=5.0)
        assert out["state"] == JobState.DONE.value
        fetched = out["result"]
        assert fetched.data == b"ok"
        assert fetched.mime == "text/plain"


async def test_parallel_submits_finish_in_concurrent_wall_clock(
    ctx: RunContext, http_server: _E2EServer
) -> None:
    """Run parallel fetches concurrently rather than sequentially.

    The concurrency signal is the RATIO of a parallel batch's wall-clock
    to a one-job-at-a-time baseline measured in the same run — not an
    absolute threshold. Both scale together with machine speed / load, so
    this stays reliable under full-suite CPU contention (an absolute floor
    false-fails when concurrent execution is merely slowed, not serial).
    """
    n_jobs = 5
    per_job_ms = 300
    url = f"{http_server.base}/sleep/{per_job_ms}"

    def _fetch() -> Any:
        return fetch_url(url, max_bytes=128, timeout_s=5.0)

    with bind(ctx):
        reg = get_registry()

        # Parallel: submit all, then await them.
        start = time.perf_counter()
        job_ids = [reg.submit(_fetch, label=f"par-{i}") for i in range(n_jobs)]
        par_outs = [await reg.wait(j, timeout_s=5.0) for j in job_ids]
        parallel_s = time.perf_counter() - start

        # Sequential baseline: submit+await one at a time, same environment.
        start = time.perf_counter()
        seq_outs = []
        for i in range(n_jobs):
            jid = reg.submit(_fetch, label=f"seq-{i}")
            seq_outs.append(await reg.wait(jid, timeout_s=5.0))
        sequential_s = time.perf_counter() - start

    assert all(o["state"] == JobState.DONE.value for o in (*par_outs, *seq_outs))
    # A parallel batch must finish well under running the jobs back-to-back.
    assert parallel_s < sequential_s * 0.7, (
        f"parallel submits did not run concurrently: parallel={parallel_s:.2f}s "
        f"vs sequential baseline={sequential_s:.2f}s"
    )


async def test_cancel_aborts_inflight_fetch(
    ctx: RunContext, http_server: _E2EServer
) -> None:
    """Abort an in-flight fetch when the job is cancelled."""
    with bind(ctx):
        reg = get_registry()
        url = f"{http_server.base}/cancellable"
        job_id = reg.submit(
            lambda: fetch_url(url, max_bytes=10 * 1024 * 1024, timeout_s=60.0),
            label="cancel-me",
        )
        await asyncio.sleep(0.05)
        assert reg.status(job_id)["state"] == JobState.RUNNING.value

        start = time.perf_counter()
        assert reg.cancel(job_id) is True
        await asyncio.sleep(0.1)
        elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"cancellation took too long: {elapsed:.2f}s"
    assert reg.status(job_id)["state"] == JobState.CANCELED.value


async def test_bind_unbind_cancels_inflight_jobs(
    ctx: RunContext, http_server: _E2EServer
) -> None:
    """Cancel in-flight jobs when the bind() block is exited."""
    captured: dict[str, JobRegistry] = {}

    url = f"{http_server.base}/cancellable"

    with bind(ctx):
        reg = get_registry()
        captured["reg"] = reg
        reg.submit(
            lambda: fetch_url(url, max_bytes=10 * 1024 * 1024, timeout_s=60.0),
            label="leaked-fetch",
        )
        await asyncio.sleep(0.05)

    [snapshot] = captured["reg"].list_all()
    assert snapshot["state"] == JobState.CANCELED.value


async def test_mixed_workload_each_terminal_state(
    ctx: RunContext, http_server: _E2EServer
) -> None:
    with bind(ctx):
        reg = get_registry()
        slow_url = f"{http_server.base}/sleep/200"
        quick_url = f"{http_server.base}/sleep/10"
        forever_url = f"{http_server.base}/cancellable"

        j_slow = reg.submit(
            lambda: fetch_url(slow_url, max_bytes=128, timeout_s=5.0),
            label="slow",
        )
        j_quick = reg.submit(
            lambda: fetch_url(quick_url, max_bytes=128, timeout_s=5.0),
            label="quick",
        )
        j_cancel = reg.submit(
            lambda: fetch_url(forever_url, max_bytes=10 * 1024 * 1024, timeout_s=60.0),
            label="forever",
        )

        q_out = await reg.wait(j_quick, timeout_s=5.0)
        assert q_out["state"] == JobState.DONE.value

        assert reg.cancel(j_cancel) is True

        s_out = await reg.wait(j_slow, timeout_s=5.0)
        assert s_out["state"] == JobState.DONE.value

    final = {row["label"]: row["state"] for row in reg.list_all()}
    assert final["quick"] == JobState.DONE.value
    assert final["slow"] == JobState.DONE.value
    assert final["forever"] == JobState.CANCELED.value


async def test_wait_timeout_does_not_kill_the_job(
    ctx: RunContext, http_server: _E2EServer
) -> None:
    """Return the TIMEOUT sentinel without killing the job."""
    with bind(ctx):
        reg = get_registry()
        url = f"{http_server.base}/sleep/300"
        job_id = reg.submit(
            lambda: fetch_url(url, max_bytes=128, timeout_s=5.0),
            label="slow",
        )

        first = await reg.wait(job_id, timeout_s=0.05)
        assert first["state"] == WAIT_TIMEOUT
        assert reg.status(job_id)["state"] in (
            JobState.RUNNING.value,
            JobState.DONE.value,
        )
        second = await reg.wait(job_id, timeout_s=5.0)
        assert second["state"] == JobState.DONE.value
        assert second["result"].data == b"ok"
