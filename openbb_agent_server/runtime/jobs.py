"""Per-run background-task registry."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openbb_agent_server.runtime.context import runtime_state

logger = logging.getLogger("openbb_agent_server.jobs")


_STATE_KEY = "_background_jobs"


class JobState(str, Enum):
    """Lifecycle states for a background job.

    Attributes
    ----------
    PENDING : str
        Submitted but not yet started (reserved; jobs start ``RUNNING``).
    RUNNING : str
        The job's coroutine is executing.
    DONE : str
        The job completed successfully; its result is available.
    ERROR : str
        The job raised; the exception is recorded on the job.
    CANCELED : str
        The job was cancelled before completing.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELED = "canceled"


WAIT_TIMEOUT = "TIMEOUT"


@dataclass
class _Job:
    id: str
    label: str
    state: JobState
    started_at: float
    task: asyncio.Task[Any] | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class JobRegistry:
    """Per-run registry of fire-and-forget asyncio tasks.

    One registry is created lazily per run via :func:`get_registry` and
    stored on the run's :func:`~openbb_agent_server.runtime.context.runtime_state`. It tracks each submitted
    coroutine as a :class:`_Job`, records lifecycle state and timing, and
    exposes non-blocking status snapshots plus blocking waits.
    """

    def __init__(self) -> None:
        """Initialize an empty registry with no tracked jobs."""
        self._jobs: dict[str, _Job] = {}

    def submit(
        self,
        target: Any,
        *,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Schedule ``target`` as a background task and return its job id.

        Parameters
        ----------
        target : Any
            Either an already-created coroutine object, or a zero-argument
            callable that returns one when invoked. The callable form is
            deferred until the task actually runs.
        label : str
            Human-readable name for the job; used in the asyncio task name
            and in log lines.
        metadata : dict[str, Any] or None, optional
            Arbitrary metadata copied onto the job and echoed back in status
            snapshots. Defaults to an empty mapping.

        Returns
        -------
        str
            A URL-safe job id that uniquely identifies the new job within
            this run.

        Raises
        ------
        TypeError
            If ``target`` is neither a coroutine nor a zero-argument
            callable returning one.
        """
        coro_obj: Awaitable[Any] | None
        factory: Callable[[], Awaitable[Any]] | None
        if asyncio.iscoroutine(target):
            coro_obj = target
            factory = None
        elif callable(target):
            coro_obj = None
            factory = target
        else:
            raise TypeError(
                "JobRegistry.submit: target must be a coroutine OR a "
                f"zero-arg callable returning one; got {type(target).__name__}"
            )

        job_id = secrets.token_urlsafe(12)
        job = _Job(
            id=job_id,
            label=label,
            state=JobState.RUNNING,
            started_at=time.time(),
            metadata=dict(metadata or {}),
        )
        job.task = asyncio.create_task(
            self._run(job, coro_obj, factory),
            name=f"job:{label}:{job_id}",
        )
        self._jobs[job_id] = job
        logger.debug("job submitted id=%s label=%s", job_id, label)
        return job_id

    async def _run(
        self,
        job: _Job,
        coro: Awaitable[Any] | None,
        factory: Callable[[], Awaitable[Any]] | None,
    ) -> None:
        if coro is None and factory is not None:
            coro = factory()
        if coro is None:  # pragma: no cover - submit() validates upstream
            job.state = JobState.ERROR
            job.error = "submit() received neither coroutine nor factory"
            job.finished_at = time.time()
            return
        try:
            job.result = await coro
        except asyncio.CancelledError:
            if asyncio.iscoroutine(coro):
                coro.close()
            job.state = JobState.CANCELED
            if job.finished_at is None:  # pragma: no cover - cancel() sets it first
                job.finished_at = time.time()
            logger.debug("job cancelled id=%s", job.id)
            raise
        except Exception as exc:  # noqa: BLE001 — recorded onto the job
            job.state = JobState.ERROR
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = time.time()
            logger.warning(
                "job failed id=%s label=%s error=%s",
                job.id,
                job.label,
                job.error,
            )
            return
        job.state = JobState.DONE
        job.finished_at = time.time()
        logger.debug("job finished id=%s", job.id)

    def status(self, job_id: str) -> dict[str, Any]:
        """Return a non-blocking snapshot of one job.

        Parameters
        ----------
        job_id : str
            Id of a job previously returned by :meth:`submit`.

        Returns
        -------
        dict[str, Any]
            Snapshot with ``job_id``, ``label``, ``state``, ``started_at``
            and ``finished_at`` keys, plus ``metadata`` and ``error`` when
            present. The job's result is never included.

        Raises
        ------
        KeyError
            If no job with ``job_id`` exists in this run.
        """
        return self._snapshot(self._require(job_id), include_result=False)

    def list_all(self) -> list[dict[str, Any]]:
        """Return snapshots of all registered jobs, oldest-first.

        Returns
        -------
        list[dict[str, Any]]
            One status snapshot per job, ordered by ``started_at`` ascending.
            Results are omitted from every snapshot.
        """
        return [
            self._snapshot(j, include_result=False)
            for j in sorted(self._jobs.values(), key=lambda j: j.started_at)
        ]

    async def wait(
        self, job_id: str, *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Await a job to completion or until ``timeout_s`` elapses.

        If the job has already finished, returns immediately. The underlying
        task is shielded, so a timeout here does not cancel the job.

        Parameters
        ----------
        job_id : str
            Id of a job previously returned by :meth:`submit`.
        timeout_s : float or None, optional
            Maximum seconds to wait. ``None`` (the default) waits forever.

        Returns
        -------
        dict[str, Any]
            A status snapshot. When the job is done its ``result`` is
            included. On timeout the snapshot's ``state`` is replaced with
            :data:`WAIT_TIMEOUT` and no result is included.

        Raises
        ------
        KeyError
            If no job with ``job_id`` exists in this run.
        """
        job = self._require(job_id)
        if job.state in (JobState.DONE, JobState.ERROR, JobState.CANCELED):
            return self._snapshot(job, include_result=True)
        assert job.task is not None  # noqa: S101
        try:
            await asyncio.wait_for(asyncio.shield(job.task), timeout=timeout_s)
        except asyncio.TimeoutError:
            return {
                **self._snapshot(job, include_result=False),
                "state": WAIT_TIMEOUT,
            }
        return self._snapshot(job, include_result=True)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of one job.

        A job that is unknown or already in a terminal state (done, errored
        or canceled) is treated as not cancellable.

        Parameters
        ----------
        job_id : str
            Id of the job to cancel.

        Returns
        -------
        bool
            ``True`` if a cancellation request was delivered to the task,
            ``False`` otherwise. On success the job's state is set to
            ``CANCELED`` and ``finished_at`` is stamped if not already set.
        """
        job = self._jobs.get(job_id)
        if job is None or job.state in (
            JobState.DONE,
            JobState.ERROR,
            JobState.CANCELED,
        ):
            return False
        assert job.task is not None  # noqa: S101
        cancelled = job.task.cancel()
        if cancelled:
            job.state = JobState.CANCELED
            if job.finished_at is None:
                job.finished_at = time.time()
        return cancelled

    def cancel_all(self) -> int:
        """Cancel every still-running task in the registry.

        Iterates all jobs currently in the ``RUNNING`` state, requests
        cancellation, and marks each successfully cancelled job as
        ``CANCELED`` with a shared ``finished_at`` timestamp.

        Returns
        -------
        int
            The number of jobs for which cancellation was delivered.
        """
        n = 0
        now = time.time()
        for job in self._jobs.values():
            if (
                job.state == JobState.RUNNING
                and job.task is not None
                and job.task.cancel()
            ):
                job.state = JobState.CANCELED
                if job.finished_at is None:
                    job.finished_at = now
                n += 1
        if n:
            logger.debug("cancel_all cancelled %d background jobs", n)
        return n

    def _require(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"no background job {job_id!r} in this run")
        return job

    def _snapshot(self, job: _Job, *, include_result: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": job.id,
            "label": job.label,
            "state": job.state.value,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
        if job.metadata:
            out["metadata"] = dict(job.metadata)
        if job.error:
            out["error"] = job.error
        if include_result and job.state == JobState.DONE:
            out["result"] = job.result
        return out


def get_registry() -> JobRegistry:
    """Return the :class:`JobRegistry` for the current run.

    The registry is stored on the run's :func:`~openbb_agent_server.runtime.context.runtime_state` under a
    private key and created lazily on first access.

    Returns
    -------
    JobRegistry
        The registry bound to the active run.
    """
    state = runtime_state()
    reg = state.get(_STATE_KEY)
    if reg is None:
        reg = JobRegistry()
        state[_STATE_KEY] = reg
    return reg


def cleanup_state(state: dict[str, Any]) -> int:
    """Cancel every background job recorded in ``state``.

    Intended for teardown of a run: looks up the job registry on the given
    runtime-state mapping and cancels all of its still-running tasks. A no-op
    when no registry was ever created.

    Parameters
    ----------
    state : dict[str, Any]
        A run's runtime-state mapping, as returned by :func:`~openbb_agent_server.runtime.context.runtime_state`.

    Returns
    -------
    int
        The number of background jobs cancelled, or ``0`` if no registry was
        present.
    """
    reg = state.get(_STATE_KEY)
    if reg is None:
        return 0
    return reg.cancel_all()


__all__ = [
    "JobRegistry",
    "JobState",
    "WAIT_TIMEOUT",
    "cleanup_state",
    "get_registry",
]
