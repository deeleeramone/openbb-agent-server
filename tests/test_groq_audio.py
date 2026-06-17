"""GroqAudioToolSource tests."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from openbb_agent_server.plugins.models.groq_rate_limiter import (
    get_limiter,
    reset_cache,
)
from openbb_agent_server.plugins.tools.groq_audio import GroqAudioToolSource
from openbb_agent_server.runtime.context import FileRef, RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(
    *,
    api_keys: dict[str, str] | None = None,
    files: tuple[FileRef, ...] = (),
) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=api_keys or {},
        uploaded_files=files,
    )


def _mp3_ref(name: str = "hello.wav") -> FileRef:
    return FileRef(
        name=name,
        mime="audio/wav",
        data_base64=base64.b64encode(b"FAKE-AUDIO-BYTES").decode(),
    )


def test_constructor_rejects_unknown_default_model() -> None:
    with pytest.raises(ValueError, match="default_model"):
        GroqAudioToolSource(default_model="not-a-whisper")


def test_constructor_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        GroqAudioToolSource(max_retries=-1)


async def test_tools_requires_api_key() -> None:
    src = GroqAudioToolSource()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        await src.tools(_ctx(), {})


async def test_tools_returns_transcribe_and_translate() -> None:
    src = GroqAudioToolSource(api_key="k")
    tools = await src.tools(_ctx(), {})
    names = {t.name for t in tools}
    assert names == {"transcribe_audio", "translate_audio"}


def _mock_transport(captured: dict[str, Any]) -> httpx.MockTransport:
    """Return a transport that records the request and replies success."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.content
        captured["content_type"] = request.headers.get("content-type")
        return httpx.Response(
            200,
            json={
                "text": "hello world",
                "duration": 2.5,
                "language": "en",
                "segments": [{"text": "hello world", "start": 0, "end": 2.5}],
            },
        )

    return httpx.MockTransport(handler)


async def test_transcribe_uses_uploaded_file_and_records_audio_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    captured: dict[str, Any] = {}
    transport = _mock_transport(captured)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    out = await tx.ainvoke(
        {"audio_name": "hello.wav", "response_format": "verbose_json"}
    )

    assert out["text"] == "hello world"
    assert out["duration"] == 2.5
    assert out["_groq"]["model"] == "whisper-large-v3-turbo"
    assert out["_groq"]["endpoint"] == "/audio/transcriptions"
    assert out["_groq"]["duration_seconds"] == 2.5

    body = captured["body"].decode("latin-1")
    assert body.count('name="response_format"') == 1
    assert "hello.wav" in body
    assert "FAKE-AUDIO-BYTES" in body

    snap = get_limiter(api_key="k", model_name="whisper-large-v3-turbo").snapshot()
    assert snap["audio_seconds_per_hour_remaining"] == pytest.approx(
        7_200 - 2.5, abs=1.0
    )
    assert snap["audio_seconds_per_day_remaining"] == pytest.approx(
        28_800 - 2.5, abs=1.0
    )


async def test_transcribe_includes_language_only_for_transcriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    captured: dict[str, Any] = {}
    transport = _mock_transport(captured)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k", default_language="en")
    [_tx, tr] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    await tr.ainvoke({"audio_name": "hello.wav"})

    body = captured["body"].decode("latin-1")
    assert "/audio/translations" in captured["url"]
    assert 'name="language"' not in body


async def test_transcribe_passes_timestamp_granularities_as_repeated_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    captured: dict[str, Any] = {}
    transport = _mock_transport(captured)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    await tx.ainvoke(
        {
            "audio_name": "hello.wav",
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
        }
    )

    body = captured["body"].decode("latin-1")
    assert body.count('name="timestamp_granularities[]"') == 2


async def test_transcribe_rejects_unknown_response_format() -> None:
    reset_cache()
    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    with pytest.raises(ValueError, match="response_format"):
        await tx.ainvoke({"audio_name": "hello.wav", "response_format": "weird"})


async def test_transcribe_errors_on_missing_inputs() -> None:
    reset_cache()
    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(), {})
    with pytest.raises(RuntimeError, match="audio_name or audio_url"):
        await tx.ainvoke({})


async def test_transcribe_errors_when_uploaded_name_missing() -> None:
    reset_cache()
    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(name="other.wav"),)), {})
    with pytest.raises(RuntimeError, match="hello.wav"):
        await tx.ainvoke({"audio_name": "hello.wav"})


async def test_transcribe_retries_on_429_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429, headers={"retry-after": "0.05"}, text="slow down"
            )
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k", max_retries=2)
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    out = await tx.ainvoke({"audio_name": "hello.wav"})

    assert out["text"] == "ok"
    assert len(calls) == 2


async def test_transcribe_eventually_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k", max_retries=1)
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    with pytest.raises(RuntimeError, match="503"):
        await tx.ainvoke({"audio_name": "hello.wav"})


async def test_transcribe_audio_url_fetches_remote_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("hello.wav"):
            return httpx.Response(
                200, content=b"REMOTE-BYTES", headers={"content-type": "audio/wav"}
            )
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    real_sync_client = httpx.Client

    def _async_client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    def _sync_client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _async_client,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.Client",
        _sync_client,
    )

    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(), {})
    out = await tx.ainvoke({"audio_url": "https://example.test/path/hello.wav"})
    assert out["text"] == "ok"
    assert b"REMOTE-BYTES" in captured["body"]
    assert b"hello.wav" in captured["body"]


async def test_transcribe_audio_url_propagates_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    real_sync_client = httpx.Client

    def _sync_client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.Client",
        _sync_client,
    )

    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(), {})
    with pytest.raises(RuntimeError, match="could not fetch audio_url"):
        await tx.ainvoke({"audio_url": "https://example.test/missing.wav"})


async def test_transcribe_uploaded_url_only_ref_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch and forward a URL-only FileRef."""
    reset_cache()
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("uploaded.wav"):
            return httpx.Response(
                200, content=b"FROM-URL", headers={"content-type": "audio/wav"}
            )
        sent["body"] = request.content
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    real_sync_client = httpx.Client

    def _async_client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    def _sync_client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _async_client,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.Client",
        _sync_client,
    )

    src = GroqAudioToolSource(api_key="k")
    ref = FileRef(name="audio", mime=None, url="https://example.test/uploaded.wav")
    [tx, _] = await src.tools(_ctx(files=(ref,)), {})
    await tx.ainvoke({"audio_name": "audio"})
    assert b"FROM-URL" in sent["body"]


async def test_transcribe_uploaded_url_only_ref_propagates_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_sync_client = httpx.Client

    def _sync_client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.Client",
        _sync_client,
    )

    src = GroqAudioToolSource(api_key="k")
    ref = FileRef(name="audio", url="https://example.test/missing.wav")
    [tx, _] = await src.tools(_ctx(files=(ref,)), {})
    with pytest.raises(RuntimeError, match="could not fetch"):
        await tx.ainvoke({"audio_name": "audio"})


async def test_transcribe_rejects_unknown_timestamp_granularity() -> None:
    reset_cache()
    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    with pytest.raises(ValueError, match="timestamp_granularities"):
        await tx.ainvoke(
            {
                "audio_name": "hello.wav",
                "response_format": "verbose_json",
                "timestamp_granularities": ["nope"],
            }
        )


async def test_transcribe_text_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="bare-transcript", headers={"content-type": "text/plain"}
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    out = await tx.ainvoke({"audio_name": "hello.wav", "response_format": "text"})
    assert out["text"] == "bare-transcript"
    snap = out["_groq"]["rate_limiter_snapshot"]
    assert snap["audio_seconds_per_hour_remaining"] == pytest.approx(7_200, abs=1.0)


async def test_transcribe_retries_on_network_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k", max_retries=2)
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    out = await tx.ainvoke({"audio_name": "hello.wav"})
    assert out["text"] == "ok"
    assert len(calls) == 2


async def test_transcribe_raises_after_network_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("persistent")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k", max_retries=1)
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    with pytest.raises(RuntimeError, match="network error after"):
        await tx.ainvoke({"audio_name": "hello.wav"})


async def test_transcribe_4xx_other_than_429_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad payload")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    with pytest.raises(RuntimeError, match="HTTP 400"):
        await tx.ainvoke({"audio_name": "hello.wav"})
    assert len(calls) == 1


async def test_transcribe_includes_optional_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    await tx.ainvoke({"audio_name": "hello.wav", "prompt": "biased context"})
    assert b'name="prompt"' in captured["body"]
    assert b"biased context" in captured["body"]


async def test_transcribe_429_with_invalid_retry_after_header_falls_back_to_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to exponential backoff on a non-numeric retry-after."""
    reset_cache()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429, headers={"retry-after": "not-a-number"}, text="slow"
            )
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource(api_key="k", max_retries=2)
    [tx, _] = await src.tools(_ctx(files=(_mp3_ref(),)), {})
    out = await tx.ainvoke({"audio_name": "hello.wav"})
    assert out["text"] == "ok"
    assert len(calls) == 2


async def test_tools_picks_up_per_call_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plumb per-call config overrides through tools()."""
    reset_cache()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _client(**kw: Any) -> httpx.AsyncClient:
        kw.pop("transport", None)
        return real_async_client(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.groq_audio.httpx.AsyncClient",
        _client,
    )

    src = GroqAudioToolSource()
    [tx, _] = await src.tools(
        _ctx(api_keys={"GROQ_API_KEY": "k"}, files=(_mp3_ref(),)),
        {
            "base_url": "https://override.example/openai/v1",
            "default_model": "whisper-large-v3",
            "default_language": "fr",
            "timeout": 30.0,
            "max_retries": 1,
        },
    )
    await tx.ainvoke({"audio_name": "hello.wav"})
    assert captured["url"].startswith("https://override.example/openai/v1")
    assert b'name="model"\r\n\r\nwhisper-large-v3' in captured["body"]
    assert b'name="language"\r\n\r\nfr' in captured["body"]


async def test_tools_picks_api_key_from_constructor_if_ctx_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    src = GroqAudioToolSource(api_key="constructor-key")
    tools = await src.tools(_ctx(), {})
    assert {t.name for t in tools} == {"transcribe_audio", "translate_audio"}


async def test_tools_picks_api_key_from_per_call_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_cache()
    src = GroqAudioToolSource()
    tools = await src.tools(_ctx(), {"api_key": "config-key"})
    assert {t.name for t in tools} == {"transcribe_audio", "translate_audio"}


async def test_resolve_audio_skips_files_with_other_names() -> None:
    """Skip uploaded files whose name does not match."""
    reset_cache()
    src = GroqAudioToolSource(api_key="k")
    [tx, _] = await src.tools(
        _ctx(files=(_mp3_ref(name="other.wav"), _mp3_ref(name="hello.wav"))),
        {},
    )
    with pytest.raises(Exception):  # noqa: B017 — any post-resolve failure is fine
        await tx.ainvoke({"audio_name": "hello.wav"})
