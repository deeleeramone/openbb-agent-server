"""Groq rate limiter."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.rate_limiters import BaseRateLimiter


@dataclass(frozen=True)
class GroqLimits:
    """Hold the published per-model rate quotas for a Groq model.

    Each field is the documented free-tier ceiling for one dimension; a
    value of ``None`` means that dimension is not enforced for the model.

    Attributes
    ----------
    rpm : int | None
        Requests permitted per minute.
    rpd : int | None
        Requests permitted per day.
    tpm : int | None
        Tokens permitted per minute (prompt plus completion).
    tpd : int | None
        Tokens permitted per day (prompt plus completion).
    audio_per_hour : int | None
        Seconds of transcribed audio permitted per hour.
    audio_per_day : int | None
        Seconds of transcribed audio permitted per day.
    """

    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None
    audio_per_hour: int | None = None
    audio_per_day: int | None = None


GROQ_LIMITS: dict[str, GroqLimits] = {
    "llama-3.1-8b-instant": GroqLimits(rpm=30, rpd=14_400, tpm=6_000, tpd=500_000),
    "llama-3.3-70b-versatile": GroqLimits(rpm=30, rpd=1_000, tpm=12_000, tpd=100_000),
    "meta-llama/llama-4-scout-17b-16e-instruct": GroqLimits(
        rpm=30, rpd=1_000, tpm=30_000, tpd=500_000
    ),
    "moonshotai/kimi-k2-instruct": GroqLimits(
        rpm=60, rpd=1_000, tpm=10_000, tpd=300_000
    ),
    "qwen/qwen3-32b": GroqLimits(rpm=60, rpd=1_000, tpm=6_000, tpd=500_000),
    "openai/gpt-oss-120b": GroqLimits(rpm=30, rpd=1_000, tpm=8_000, tpd=200_000),
    "openai/gpt-oss-20b": GroqLimits(rpm=30, rpd=1_000, tpm=8_000, tpd=200_000),
    "groq/compound": GroqLimits(rpm=30, rpd=250, tpm=70_000),
    "groq/compound-mini": GroqLimits(rpm=30, rpd=250, tpm=70_000),
    "allam-2-7b": GroqLimits(rpm=30, rpd=7_000, tpm=6_000, tpd=500_000),
    "whisper-large-v3": GroqLimits(
        rpm=20, rpd=2_000, audio_per_hour=7_200, audio_per_day=28_800
    ),
    "whisper-large-v3-turbo": GroqLimits(
        rpm=20, rpd=2_000, audio_per_hour=7_200, audio_per_day=28_800
    ),
}

_DEFAULT_LIMITS = GroqLimits(rpm=30, rpd=1_000, tpm=6_000, tpd=100_000)


@dataclass
class _Bucket:
    """Token-bucket counter that refills over a fixed period."""

    capacity: float
    period_seconds: float
    available: float
    last_refill: float

    @classmethod
    def of(cls, capacity: float, period_seconds: float) -> _Bucket:
        return cls(
            capacity=capacity,
            period_seconds=period_seconds,
            available=float(capacity),
            last_refill=time.monotonic(),
        )

    def refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return
        per_second = self.capacity / self.period_seconds
        self.available = min(self.capacity, self.available + per_second * elapsed)
        self.last_refill = now

    def consume(self, n: float) -> None:
        self.refill()
        self.available -= n

    def time_until_at_least_one(self) -> float:
        self.refill()
        if self.available >= 1:
            return 0.0
        deficit = 1.0 - self.available
        per_second = self.capacity / self.period_seconds
        return deficit / per_second


class GroqRateLimiter(BaseRateLimiter):
    """Throttle Groq calls across every published quota dimension.

    Implements the LangChain :class:`BaseRateLimiter` contract using one
    independent token bucket per active dimension (requests per minute and
    day, tokens per minute and day, audio seconds per hour and day). A call
    is admitted only when every bucket has at least one unit available, so
    the slowest-refilling dimension governs the effective wait. Requests are
    charged on admission; token and audio usage must be recorded after the
    fact via :meth:`record_tokens` and :meth:`record_audio_seconds`.

    The limiter is safe to share across threads (a `threading.Lock`)
    and across asyncio tasks (an additional `asyncio.Lock`).
    """

    def __init__(
        self,
        *,
        rpm: int,
        rpd: int | None = None,
        tpm: int | None = None,
        tpd: int | None = None,
        audio_per_hour: int | None = None,
        audio_per_day: int | None = None,
        check_every_n_seconds: float = 0.1,
    ) -> None:
        """Build a limiter with a bucket for each supplied quota.

        Parameters
        ----------
        rpm : int
            Requests permitted per minute. Must be greater than zero; the
            per-minute request bucket is always created.
        rpd : int | None, optional
            Requests permitted per day. When ``None`` the daily request
            dimension is not enforced.
        tpm : int | None, optional
            Tokens permitted per minute. When ``None`` the per-minute token
            dimension is not enforced.
        tpd : int | None, optional
            Tokens permitted per day. When ``None`` the daily token
            dimension is not enforced.
        audio_per_hour : int | None, optional
            Seconds of audio permitted per hour. When ``None`` the hourly
            audio dimension is not enforced.
        audio_per_day : int | None, optional
            Seconds of audio permitted per day. When ``None`` the daily
            audio dimension is not enforced.
        check_every_n_seconds : float, optional
            Maximum poll interval, in seconds, between availability checks
            while blocking. Clamped to a floor of ``0.01``.

        Raises
        ------
        ValueError
            If ``rpm`` is not greater than zero.
        """
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self._req_min = _Bucket.of(rpm, 60.0)
        self._req_day = _Bucket.of(rpd, 86_400.0) if rpd else None
        self._tok_min = _Bucket.of(tpm, 60.0) if tpm else None
        self._tok_day = _Bucket.of(tpd, 86_400.0) if tpd else None
        self._audio_hour = (
            _Bucket.of(audio_per_hour, 3_600.0) if audio_per_hour else None
        )
        self._audio_day = _Bucket.of(audio_per_day, 86_400.0) if audio_per_day else None
        self._check_every = max(0.01, check_every_n_seconds)
        self._lock = threading.Lock()
        self._alock = asyncio.Lock()

    @classmethod
    def from_limits(cls, limits: GroqLimits, **overrides: Any) -> GroqRateLimiter:
        """Construct a limiter from a :class:`GroqLimits` quota set.

        Parameters
        ----------
        limits : GroqLimits
            Published quotas to seed the buckets. A missing ``rpm`` falls
            back to ``30`` since a per-minute request limit is mandatory.
        **overrides : Any
            Keyword arguments forwarded to :mod:`__init__` that take
            precedence over the corresponding values from ``limits`` (for
            example ``check_every_n_seconds`` or an adjusted ``rpm``).

        Returns
        -------
        GroqRateLimiter
            A new limiter configured from the quotas and overrides.
        """
        kwargs: dict[str, Any] = {
            "rpm": limits.rpm or 30,
            "rpd": limits.rpd,
            "tpm": limits.tpm,
            "tpd": limits.tpd,
            "audio_per_hour": limits.audio_per_hour,
            "audio_per_day": limits.audio_per_day,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def acquire(self, *, blocking: bool = True) -> bool:  # noqa: D401
        """Reserve one request slot, blocking until every quota allows it.

        Polls all active buckets and charges a request once each one has at
        least a full unit available.

        Parameters
        ----------
        blocking : bool, optional
            When ``True`` (the default) sleep in increments of at most
            ``check_every_n_seconds`` until a slot is free. When ``False``
            return immediately if any dimension is currently exhausted.

        Returns
        -------
        bool
            ``True`` once a request was charged; ``False`` only when
            ``blocking`` is ``False`` and a slot was not immediately
            available.
        """
        while True:
            with self._lock:
                wait = self._compute_wait()
                if wait <= 0.0:
                    self._consume_request()
                    return True
            if not blocking:
                return False
            time.sleep(min(wait, self._check_every))

    async def aacquire(self, *, blocking: bool = True) -> bool:  # noqa: D401
        """Reserve one request slot asynchronously across every quota.

        Async counterpart to :meth:`acquire`. Holds the asyncio lock while
        checking buckets and awaits `asyncio.sleep` between polls so
        the event loop is not blocked.

        Parameters
        ----------
        blocking : bool, optional
            When ``True`` (the default) await until a slot is free. When
            ``False`` return immediately if any dimension is exhausted.

        Returns
        -------
        bool
            ``True`` once a request was charged; ``False`` only when
            ``blocking`` is ``False`` and no slot was immediately available.
        """
        while True:
            async with self._alock:
                with self._lock:
                    wait = self._compute_wait()
                    if wait <= 0.0:
                        self._consume_request()
                        return True
            if not blocking:
                return False
            await asyncio.sleep(min(wait, self._check_every))

    def _compute_wait(self) -> float:
        candidates = [self._req_min.time_until_at_least_one()]
        if self._req_day is not None:
            candidates.append(self._req_day.time_until_at_least_one())
        if self._tok_min is not None:
            candidates.append(self._tok_min.time_until_at_least_one())
        if self._tok_day is not None:
            candidates.append(self._tok_day.time_until_at_least_one())
        if self._audio_hour is not None:
            candidates.append(self._audio_hour.time_until_at_least_one())
        if self._audio_day is not None:
            candidates.append(self._audio_day.time_until_at_least_one())
        return max(candidates)

    def _consume_request(self) -> None:
        self._req_min.consume(1.0)
        if self._req_day is not None:
            self._req_day.consume(1.0)

    def record_tokens(self, n: int) -> None:
        """Charge ``n`` tokens against the per-minute and daily buckets.

        Call after a completion to account for token usage that is unknown
        at admission time. Non-positive counts are ignored, and dimensions
        that were not configured are skipped.

        Parameters
        ----------
        n : int
            Total tokens consumed by the response (prompt plus completion).
        """
        if n <= 0:
            return
        with self._lock:
            if self._tok_min is not None:
                self._tok_min.consume(n)
            if self._tok_day is not None:
                self._tok_day.consume(n)

    def record_audio_seconds(self, seconds: float) -> None:
        """Charge transcribed audio seconds against the audio buckets.

        Call after a transcription to account for audio duration. Non-positive
        values are ignored, and audio dimensions that were not configured are
        skipped.

        Parameters
        ----------
        seconds : float
            Seconds of audio processed by the response.
        """
        if seconds <= 0:
            return
        with self._lock:
            if self._audio_hour is not None:
                self._audio_hour.consume(seconds)
            if self._audio_day is not None:
                self._audio_day.consume(seconds)

    @property
    def callback_handler(self) -> BaseCallbackHandler:
        """Return a LangChain callback that feeds token usage back in.

        Returns
        -------
        BaseCallbackHandler
            A handler that, on ``on_llm_end``, reads the response's token
            usage and forwards it to :meth:`record_tokens`.
        """
        return _GroqUsageHandler(self)

    def snapshot(self) -> dict[str, float | None]:
        """Return the current remaining capacity of every dimension.

        Refills each bucket to the present moment before reading, so the
        returned values reflect capacity available right now.

        Returns
        -------
        dict[str, float | None]
            Mapping with keys ``rpm_remaining``, ``rpd_remaining``,
            ``tpm_remaining``, ``tpd_remaining``,
            ``audio_seconds_per_hour_remaining`` and
            ``audio_seconds_per_day_remaining``. Each value is the units
            still available, or ``None`` for a dimension that is not
            configured.
        """
        with self._lock:
            for b in (
                self._req_min,
                self._req_day,
                self._tok_min,
                self._tok_day,
                self._audio_hour,
                self._audio_day,
            ):
                if b is not None:
                    b.refill()
            return {
                "rpm_remaining": self._req_min.available,
                "rpd_remaining": (self._req_day.available if self._req_day else None),
                "tpm_remaining": (self._tok_min.available if self._tok_min else None),
                "tpd_remaining": (self._tok_day.available if self._tok_day else None),
                "audio_seconds_per_hour_remaining": (
                    self._audio_hour.available if self._audio_hour else None
                ),
                "audio_seconds_per_day_remaining": (
                    self._audio_day.available if self._audio_day else None
                ),
            }


class _GroqUsageHandler(BaseCallbackHandler):
    """Read response usage metadata and feed it into a limiter."""

    raise_error = False

    def __init__(self, limiter: GroqRateLimiter) -> None:
        self._limiter = limiter

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        total = int(usage.get("total_tokens") or 0)
        if total <= 0:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    meta = getattr(msg, "usage_metadata", None)
                    if isinstance(meta, dict):
                        total += int(meta.get("total_tokens") or 0)
        if total > 0:
            self._limiter.record_tokens(total)


_LIMITER_CACHE: dict[tuple[str, str], GroqRateLimiter] = {}
_CACHE_LOCK = threading.Lock()


def get_limiter(*, api_key: str, model_name: str) -> GroqRateLimiter:
    """Return the process-shared limiter for an API key and model.

    Looks up a cached limiter keyed by ``(api_key, model_name)`` and creates
    one on first use, seeded from :data:`GROQ_LIMITS` (or the default quotas
    when the model is unknown). Subsequent calls with the same key return the
    same instance, so all callers for that key and model share its quotas.
    Access to the cache is guarded by a module-level lock.

    Parameters
    ----------
    api_key : str
        Groq API key the limiter applies to; quotas are tracked per key.
    model_name : str
        Model identifier used to select published quotas.

    Returns
    -------
    GroqRateLimiter
        The shared limiter for this key and model.
    """
    cache_key = (api_key, model_name)
    with _CACHE_LOCK:
        existing = _LIMITER_CACHE.get(cache_key)
        if existing is not None:
            return existing
        limits = GROQ_LIMITS.get(model_name) or _DEFAULT_LIMITS
        limiter = GroqRateLimiter.from_limits(limits)
        _LIMITER_CACHE[cache_key] = limiter
        return limiter


def reset_cache() -> None:
    """Drop every cached limiter."""
    with _CACHE_LOCK:
        _LIMITER_CACHE.clear()
