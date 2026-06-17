"""GroqRateLimiter unit tests."""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from openbb_agent_server.plugins.models.groq_rate_limiter import (
    GROQ_LIMITS,
    GroqRateLimiter,
    _Bucket,
    get_limiter,
    reset_cache,
)


def test_bucket_refills_at_constant_rate() -> None:
    bucket = _Bucket.of(capacity=60, period_seconds=60.0)
    bucket.consume(60)
    bucket.last_refill -= 1.0
    bucket.refill()
    assert bucket.available == pytest.approx(1.0, abs=0.05)


def test_bucket_caps_at_capacity() -> None:
    bucket = _Bucket.of(capacity=10, period_seconds=10.0)
    bucket.last_refill -= 100.0
    bucket.refill()
    assert bucket.available == 10.0


def test_bucket_time_until_one_when_drained() -> None:
    bucket = _Bucket.of(capacity=60, period_seconds=60.0)
    bucket.consume(60)
    wait = bucket.time_until_at_least_one()
    assert 0.95 <= wait <= 1.05


def test_acquire_consumes_one_request_per_call() -> None:
    limiter = GroqRateLimiter(rpm=30, rpd=14_400, tpm=6_000, tpd=500_000)
    snap_before = limiter.snapshot()
    limiter.acquire(blocking=False)
    snap_after = limiter.snapshot()
    assert snap_after["rpm_remaining"] < snap_before["rpm_remaining"]
    assert snap_after["rpd_remaining"] < snap_before["rpd_remaining"]
    assert snap_after["tpm_remaining"] == pytest.approx(
        snap_before["tpm_remaining"], abs=0.5
    )


def test_acquire_nonblocking_returns_false_when_drained() -> None:
    limiter = GroqRateLimiter(rpm=2, rpd=None, tpm=None, tpd=None)
    assert limiter.acquire(blocking=False) is True
    assert limiter.acquire(blocking=False) is True
    assert limiter.acquire(blocking=False) is False


def test_acquire_blocking_waits_until_refill() -> None:
    limiter = GroqRateLimiter(rpm=60, rpd=None, tpm=None, tpd=None)
    for _ in range(60):
        limiter.acquire(blocking=False)
    t0 = time.monotonic()
    assert limiter.acquire(blocking=True) is True
    elapsed = time.monotonic() - t0
    assert 0.7 <= elapsed <= 2.0


def test_acquire_blocked_by_token_bucket_when_drained() -> None:
    limiter = GroqRateLimiter(rpm=60, rpd=None, tpm=10, tpd=None)
    assert limiter.acquire(blocking=False) is True
    limiter.record_tokens(10)
    assert limiter.acquire(blocking=False) is False


def test_record_tokens_decrements_both_minute_and_day_buckets() -> None:
    limiter = GroqRateLimiter(rpm=30, rpd=None, tpm=6_000, tpd=500_000)
    before = limiter.snapshot()
    limiter.record_tokens(150)
    after = limiter.snapshot()
    assert before["tpm_remaining"] - after["tpm_remaining"] == pytest.approx(
        150, abs=0.5
    )
    assert before["tpd_remaining"] - after["tpd_remaining"] == pytest.approx(
        150, abs=0.5
    )


def test_record_tokens_zero_is_noop() -> None:
    limiter = GroqRateLimiter(rpm=30, tpm=6_000)
    before = limiter.snapshot()
    limiter.record_tokens(0)
    limiter.record_tokens(-5)
    after = limiter.snapshot()
    assert before == after


def test_callback_handler_reads_usage_metadata_off_message() -> None:
    limiter = GroqRateLimiter(rpm=30, tpm=6_000)
    handler = limiter.callback_handler

    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    response = LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output={})
    handler.on_llm_end(response)

    assert limiter.snapshot()["tpm_remaining"] == pytest.approx(6_000 - 15, abs=0.5)


def test_callback_handler_reads_token_usage_from_llm_output() -> None:
    limiter = GroqRateLimiter(rpm=30, tpm=6_000)
    handler = limiter.callback_handler

    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="x"))]],
        llm_output={"token_usage": {"total_tokens": 200}},
    )
    handler.on_llm_end(response)

    assert limiter.snapshot()["tpm_remaining"] == pytest.approx(5_800, abs=0.5)


def test_get_limiter_caches_per_api_key_and_model() -> None:
    reset_cache()
    a = get_limiter(api_key="k1", model_name="llama-3.1-8b-instant")
    b = get_limiter(api_key="k1", model_name="llama-3.1-8b-instant")
    c = get_limiter(api_key="k2", model_name="llama-3.1-8b-instant")
    d = get_limiter(api_key="k1", model_name="llama-3.3-70b-versatile")
    assert a is b
    assert a is not c
    assert a is not d


def test_get_limiter_uses_published_table_for_known_model() -> None:
    reset_cache()
    limiter = get_limiter(api_key="k", model_name="llama-3.1-8b-instant")
    snap = limiter.snapshot()
    limits = GROQ_LIMITS["llama-3.1-8b-instant"]
    assert snap["rpm_remaining"] == pytest.approx(limits.rpm, abs=0.5)
    assert snap["rpd_remaining"] == pytest.approx(limits.rpd, abs=0.5)
    assert snap["tpm_remaining"] == pytest.approx(limits.tpm, abs=0.5)
    assert snap["tpd_remaining"] == pytest.approx(limits.tpd, abs=0.5)
    assert snap["audio_seconds_per_hour_remaining"] is None
    assert snap["audio_seconds_per_day_remaining"] is None


def test_get_limiter_picks_audio_buckets_for_whisper() -> None:
    reset_cache()
    limiter = get_limiter(api_key="k", model_name="whisper-large-v3")
    snap = limiter.snapshot()
    limits = GROQ_LIMITS["whisper-large-v3"]
    assert snap["rpm_remaining"] == pytest.approx(limits.rpm, abs=0.5)
    assert snap["rpd_remaining"] == pytest.approx(limits.rpd, abs=0.5)
    assert snap["audio_seconds_per_hour_remaining"] == pytest.approx(
        limits.audio_per_hour, abs=0.5
    )
    assert snap["audio_seconds_per_day_remaining"] == pytest.approx(
        limits.audio_per_day, abs=0.5
    )
    assert snap["tpm_remaining"] is None
    assert snap["tpd_remaining"] is None


def test_record_audio_seconds_decrements_audio_buckets() -> None:
    from openbb_agent_server.plugins.models.groq_rate_limiter import GroqLimits

    limiter = GroqRateLimiter.from_limits(
        GroqLimits(rpm=20, rpd=2_000, audio_per_hour=7_200, audio_per_day=28_800)
    )
    before = limiter.snapshot()
    limiter.record_audio_seconds(180)
    after = limiter.snapshot()
    assert (
        before["audio_seconds_per_hour_remaining"]
        - after["audio_seconds_per_hour_remaining"]
    ) == pytest.approx(180, abs=0.5)
    assert (
        before["audio_seconds_per_day_remaining"]
        - after["audio_seconds_per_day_remaining"]
    ) == pytest.approx(180, abs=0.5)


def test_record_audio_seconds_zero_or_negative_is_noop() -> None:
    from openbb_agent_server.plugins.models.groq_rate_limiter import GroqLimits

    limiter = GroqRateLimiter.from_limits(
        GroqLimits(rpm=20, audio_per_hour=7_200, audio_per_day=28_800)
    )
    before = limiter.snapshot()
    limiter.record_audio_seconds(0)
    limiter.record_audio_seconds(-10)
    after = limiter.snapshot()
    assert before == after


def test_acquire_blocked_by_audio_bucket_when_drained() -> None:
    from openbb_agent_server.plugins.models.groq_rate_limiter import GroqLimits

    limiter = GroqRateLimiter.from_limits(
        GroqLimits(rpm=20, audio_per_hour=10, audio_per_day=100)
    )
    assert limiter.acquire(blocking=False) is True
    limiter.record_audio_seconds(10)
    assert limiter.acquire(blocking=False) is False


def test_get_limiter_falls_back_to_default_for_unknown_model() -> None:
    reset_cache()
    limiter = get_limiter(api_key="k", model_name="not-a-published-model")
    snap = limiter.snapshot()
    assert snap["rpm_remaining"] >= 1
    assert snap["tpm_remaining"] is not None


def test_invalid_rpm_rejected() -> None:
    with pytest.raises(ValueError, match="rpm"):
        GroqRateLimiter(rpm=0)


def test_acquire_skips_optional_buckets_when_none() -> None:
    limiter = GroqRateLimiter(rpm=30, rpd=None, tpm=None, tpd=None)
    snap = limiter.snapshot()
    assert snap["rpd_remaining"] is None
    assert snap["tpm_remaining"] is None
    assert snap["tpd_remaining"] is None
    assert limiter.acquire(blocking=False) is True


def test_aacquire_consumes_request() -> None:
    async def go() -> None:
        limiter = GroqRateLimiter(rpm=30, rpd=14_400, tpm=6_000, tpd=500_000)
        before = limiter.snapshot()
        assert await limiter.aacquire(blocking=False) is True
        after = limiter.snapshot()
        assert before["rpm_remaining"] > after["rpm_remaining"]

    asyncio.run(go())


def test_aacquire_nonblocking_returns_false_when_drained() -> None:
    async def go() -> None:
        limiter = GroqRateLimiter(rpm=1, rpd=None, tpm=None, tpd=None)
        assert await limiter.aacquire(blocking=False) is True
        assert await limiter.aacquire(blocking=False) is False

    asyncio.run(go())


def test_bucket_refill_is_noop_when_no_time_elapsed() -> None:
    from openbb_agent_server.plugins.models.groq_rate_limiter import _Bucket

    bucket = _Bucket.of(capacity=10, period_seconds=10)
    bucket.consume(5)
    snap_a = bucket.available
    bucket.last_refill = time.monotonic() + 1.0
    bucket.refill()
    assert bucket.available == snap_a


@pytest.mark.asyncio
async def test_aacquire_blocking_waits_for_refill() -> None:
    from openbb_agent_server.plugins.models.groq_rate_limiter import GroqRateLimiter

    limiter = GroqRateLimiter(rpm=60, rpd=None, tpm=None, tpd=None)
    for _ in range(60):
        assert await limiter.aacquire(blocking=False) is True
    t0 = time.monotonic()
    assert await limiter.aacquire(blocking=True) is True
    elapsed = time.monotonic() - t0
    assert 0.5 <= elapsed <= 2.5
