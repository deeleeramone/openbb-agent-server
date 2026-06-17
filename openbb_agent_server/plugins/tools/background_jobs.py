"""``background_jobs`` tool source — poll / wait / cancel for fan-out jobs."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.jobs import get_registry
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.background_jobs")


class _ListArgs(BaseModel):
    pass


class _CheckArgs(BaseModel):
    job_id: str = Field(
        description="The id returned by a previous ``submit_*`` call.",
    )


class _WaitArgs(BaseModel):
    job_id: str = Field(
        description="The id returned by a previous ``submit_*`` call.",
    )
    timeout_s: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description=(
            "Maximum seconds to block waiting for the job. Returns "
            "``state='TIMEOUT'`` (not an error) if the job is still "
            "running when this elapses — call again to keep waiting."
        ),
    )


class _CancelArgs(BaseModel):
    job_id: str = Field(
        description="The id returned by a previous ``submit_*`` call.",
    )


class BackgroundJobsToolSource(ToolSource):
    """``list_background_jobs`` / ``check_job`` / ``wait_for_job`` / ``cancel_job``."""

    name = "background_jobs"

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Build the background-job management tools for a run.

        Closes over the process-wide job registry to expose four tools:
        ``list_background_jobs`` (enumerate jobs), ``check_job``
        (non-blocking status snapshot), ``wait_for_job`` (block until a
        job finishes or a timeout elapses), and ``cancel_job`` (request
        cancellation of a running job).

        Parameters
        ----------
        ctx : RunContext
            Active run context. Currently unused; accepted to satisfy the
            :class:`~openbb_agent_server.runtime.plugins.ToolSource` contract.
        config : dict[str, Any]
            Per-invocation tool configuration. Currently unused.

        Returns
        -------
        list[BaseTool]
            The four LangChain ``StructuredTool`` instances described above.
        """

        def list_background_jobs() -> list[dict[str, Any]]:
            jobs = get_registry().list_all()
            emit.reasoning_step("list_background_jobs", count=len(jobs))
            return jobs

        def check_job(job_id: str) -> dict[str, Any]:
            try:
                return get_registry().status(job_id)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc

        async def wait_for_job(job_id: str, timeout_s: float = 300.0) -> dict[str, Any]:
            registry = get_registry()
            try:
                result = await registry.wait(job_id, timeout_s=float(timeout_s))
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            emit.reasoning_step(
                "wait_for_job",
                job_id=job_id,
                state=result.get("state"),
            )
            return result

        def cancel_job(job_id: str) -> dict[str, Any]:
            cancelled = get_registry().cancel(job_id)
            emit.reasoning_step("cancel_job", job_id=job_id, cancelled=cancelled)
            return {"job_id": job_id, "cancelled": cancelled}

        return [
            StructuredTool.from_function(
                func=list_background_jobs,
                name="list_background_jobs",
                description=(
                    "List every background job started by a ``submit_*`` "
                    "tool in this run. Returns ``[{job_id, label, state, "
                    "started_at, finished_at}, ...]``. ``state`` is one of "
                    "``running`` / ``done`` / ``error`` / ``canceled``."
                ),
                args_schema=_ListArgs,
            ),
            StructuredTool.from_function(
                func=check_job,
                name="check_job",
                description=(
                    "Non-blocking status snapshot for one background job. "
                    "Returns the same shape as ``list_background_jobs`` "
                    "entries. Use this when you want to poll without "
                    "stalling the agent. If the state is ``running``, "
                    "continue with other work and come back later."
                ),
                args_schema=_CheckArgs,
            ),
            StructuredTool.from_function(
                coroutine=wait_for_job,
                name="wait_for_job",
                description=(
                    "Block until a background job finishes (or "
                    "``timeout_s`` elapses) and return its final state "
                    "plus ``result``. On timeout, ``state`` is "
                    "``TIMEOUT`` (the job keeps running — call this "
                    "again to wait longer). On success, ``state`` is "
                    "``done`` and ``result`` holds whatever the "
                    "underlying tool would have returned synchronously."
                ),
                args_schema=_WaitArgs,
            ),
            StructuredTool.from_function(
                func=cancel_job,
                name="cancel_job",
                description=(
                    "Request cancellation of a still-running background "
                    "job. Returns ``{job_id, cancelled}`` where "
                    "``cancelled`` is ``True`` iff the task was alive at "
                    "the time of the call. Already-finished jobs return "
                    "``False`` and are unaffected."
                ),
                args_schema=_CancelArgs,
            ),
        ]
