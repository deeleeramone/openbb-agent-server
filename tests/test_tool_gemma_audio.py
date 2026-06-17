"""gemma_audio tool source tests."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest

from openbb_agent_server.plugins.tools._media import AudioSegment, MediaError
from openbb_agent_server.plugins.tools.gemma_audio import (
    GemmaAudioToolSource,
    _format_clock,
    _is_audio,
)
from openbb_agent_server.runtime.context import FileRef, RunContext, bind
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(*files: FileRef, api_key: str | None = None) -> RunContext:
    keys: dict[str, str] = {}
    if api_key is not None:
        keys["NVIDIA_API_KEY"] = api_key
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=keys,
        uploaded_files=tuple(files),
    )


def test_is_audio_audio_mime() -> None:
    assert _is_audio(FileRef(name="x.dat", mime="audio/mp3", data_base64="x"))


def test_is_audio_video_mime() -> None:
    assert _is_audio(FileRef(name="x.dat", mime="video/mp4", data_base64="x"))


def test_is_audio_extension() -> None:
    assert _is_audio(FileRef(name="x.mp3", data_base64="x"))


def test_is_audio_uppercase_extension() -> None:
    assert _is_audio(FileRef(name="X.WAV", data_base64="x"))


def test_is_audio_rejects_image() -> None:
    assert not _is_audio(FileRef(name="x.png", mime="image/png", data_base64="x"))


def test_is_audio_no_name_no_mime() -> None:
    assert not _is_audio(FileRef(name="", data_base64="x"))


def test_format_clock_minutes() -> None:
    assert _format_clock(65) == "1:05"


def test_format_clock_seconds_only() -> None:
    assert _format_clock(45) == "0:45"


def test_format_clock_with_hours() -> None:
    assert _format_clock(3725) == "1:02:05"


def test_format_clock_clamps_negative() -> None:
    assert _format_clock(-5) == "0:00"


def test_format_clock_handles_float() -> None:
    assert _format_clock(90.7) == "1:30"


class _StubChatNVIDIA:
    instances: list[_StubChatNVIDIA] = []
    fake_reply: str = "hello world"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _StubChatNVIDIA.instances.append(self)

    def astream(self, _messages: Any) -> AsyncIterator[Any]:
        return self._astream()

    async def _astream(self) -> AsyncIterator[Any]:
        class _Chunk:
            def __init__(self, c: str) -> None:
                self.content = c

        yield _Chunk(self.fake_reply)


@pytest.fixture
def stub_nvidia_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    import types

    fake_mod = types.ModuleType("langchain_nvidia_ai_endpoints")
    fake_mod.ChatNVIDIA = _StubChatNVIDIA  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", fake_mod)
    _StubChatNVIDIA.instances.clear()
    _StubChatNVIDIA.fake_reply = "hello world"
    return fake_mod


@pytest.fixture
def stub_split(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bytes, float]]:
    """Replace split_audio_bytes so tests don't need ffmpeg."""
    from openbb_agent_server.plugins.tools import gemma_audio

    called: list[tuple[bytes, float]] = []

    async def fake_split(raw: bytes, *, max_seconds: float) -> list[AudioSegment]:
        called.append((raw, max_seconds))
        return [
            AudioSegment(
                index=0,
                total=1,
                start_s=0.0,
                end_s=1.0,
                data=raw,
                mime="audio/ogg",
            )
        ]

    monkeypatch.setattr(gemma_audio, "split_audio_bytes", fake_split)
    return called


@pytest.fixture
def stub_split_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace split_audio_bytes with one that returns multiple segments."""
    from openbb_agent_server.plugins.tools import gemma_audio

    async def fake_split(raw: bytes, *, max_seconds: float) -> list[AudioSegment]:
        return [
            AudioSegment(
                index=0,
                total=2,
                start_s=0.0,
                end_s=60.0,
                data=raw[: len(raw) // 2],
                mime="audio/ogg",
            ),
            AudioSegment(
                index=1,
                total=2,
                start_s=60.0,
                end_s=120.0,
                data=raw[len(raw) // 2 :],
                mime="audio/ogg",
            ),
        ]

    monkeypatch.setattr(gemma_audio, "split_audio_bytes", fake_split)


@pytest.mark.asyncio
async def test_tools_returns_empty_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    src = GemmaAudioToolSource()
    out = await src.tools(_ctx(), {})
    assert out == []
    assert any("NVIDIA_API_KEY is not set" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_tools_registers_three_tools(stub_nvidia_module: Any) -> None:
    src = GemmaAudioToolSource(api_key="k")
    out = await src.tools(_ctx(), {})
    assert {t.name for t in out} == {
        "list_audio",
        "transcribe_audio",
        "submit_transcribe_audio",
    }


@pytest.mark.asyncio
async def test_tools_pulls_key_from_context(stub_nvidia_module: Any) -> None:
    src = GemmaAudioToolSource()
    out = await src.tools(_ctx(api_key="ctx"), {})
    assert len(out) == 3


@pytest.mark.asyncio
async def test_tools_pulls_key_from_config(stub_nvidia_module: Any) -> None:
    src = GemmaAudioToolSource()
    out = await src.tools(_ctx(), {"api_key": "cfg"})
    assert len(out) == 3


@pytest.mark.asyncio
async def test_tools_pulls_key_from_env(
    stub_nvidia_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env")
    src = GemmaAudioToolSource()
    out = await src.tools(_ctx(), {})
    assert len(out) == 3


@pytest.mark.asyncio
async def test_list_audio_returns_uploaded(stub_nvidia_module: Any) -> None:
    clip = FileRef(name="x.mp3", mime="audio/mp3", data_base64="x")
    img = FileRef(name="x.png", mime="image/png", data_base64="x")
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [list_tool] = [t for t in tools if t.name == "list_audio"]
        out = list_tool.invoke({})
    assert len(out) == 1
    assert out[0]["name"] == "x.mp3"


@pytest.mark.asyncio
async def test_transcribe_audio_uses_base64_data(
    stub_nvidia_module: Any,
    stub_split: list[tuple[bytes, float]],
) -> None:
    import base64

    raw = b"\x00" * 100
    clip = FileRef(
        name="x.wav",
        mime="audio/wav",
        data_base64=base64.b64encode(raw).decode(),
    )
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        out = await tool.ainvoke({"name": "x.wav"})
    assert out["text"] == "hello world"
    assert out["segments"] == 1
    assert out["target"] == "x.wav"
    [(passed_raw, _)] = stub_split
    assert passed_raw == raw


@pytest.mark.asyncio
async def test_transcribe_audio_strips_data_prefix(
    stub_nvidia_module: Any,
    stub_split: list[tuple[bytes, float]],
) -> None:
    import base64

    raw = b"abc"
    clip = FileRef(
        name="x.wav",
        mime="audio/wav",
        data_base64="data:audio/wav;base64," + base64.b64encode(raw).decode(),
    )
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        await tool.ainvoke({"name": "x.wav"})
    [(passed_raw, _)] = stub_split
    assert passed_raw == raw


@pytest.mark.asyncio
async def test_transcribe_audio_oversized_base64_rejected(
    stub_nvidia_module: Any,
    stub_split: list[tuple[bytes, float]],
) -> None:
    import base64

    raw = b"x" * 1000
    clip = FileRef(
        name="x.wav",
        mime="audio/wav",
        data_base64=base64.b64encode(raw).decode(),
    )
    src = GemmaAudioToolSource(max_fetch_bytes=10)
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        with pytest.raises(Exception, match="exceeds the configured max"):
            await tool.ainvoke({"name": "x.wav"})


@pytest.mark.asyncio
async def test_transcribe_audio_url_upload(
    stub_nvidia_module: Any,
    stub_split: list[tuple[bytes, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import gemma_audio

    class _Fetched:
        data = b"audio bytes"
        mime = "audio/mpeg"

    async def fake_fetch(*_a: Any, **_kw: Any) -> _Fetched:
        return _Fetched()

    monkeypatch.setattr(gemma_audio, "fetch_url", fake_fetch)

    clip = FileRef(name="x.mp3", mime="audio/mp3", url="https://x/y.mp3")
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        out = await tool.ainvoke({"name": "x.mp3"})
    assert out["text"] == "hello world"


@pytest.mark.asyncio
async def test_transcribe_audio_uploaded_without_data_or_url_raises(
    stub_nvidia_module: Any,
) -> None:
    clip = FileRef(name="x.mp3", mime="audio/mp3")
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        with pytest.raises(Exception, match="neither data_base64 nor url"):
            await tool.ainvoke({"name": "x.mp3"})


@pytest.mark.asyncio
async def test_transcribe_audio_unknown_name(stub_nvidia_module: Any) -> None:
    src = GemmaAudioToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        with pytest.raises(Exception, match="not among this run's uploads"):
            await tool.ainvoke({"name": "missing.mp3"})


@pytest.mark.asyncio
async def test_transcribe_audio_url_only(
    stub_nvidia_module: Any,
    stub_split: list[tuple[bytes, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import gemma_audio

    class _Fetched:
        data = b"audio bytes"
        mime = "audio/mpeg"

    async def fake_fetch(*_a: Any, **_kw: Any) -> _Fetched:
        return _Fetched()

    monkeypatch.setattr(gemma_audio, "fetch_url", fake_fetch)

    src = GemmaAudioToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        out = await tool.ainvoke({"url": "https://x/y.mp3"})
    assert out["text"] == "hello world"
    assert out["target"] == "https://x/y.mp3"


@pytest.mark.asyncio
async def test_transcribe_audio_rejects_no_source(
    stub_nvidia_module: Any,
) -> None:
    src = GemmaAudioToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        with pytest.raises(Exception, match="needs either 'name' or 'url'"):
            await tool.ainvoke({"instruction": "x"})


@pytest.mark.asyncio
async def test_transcribe_audio_wraps_resolver_media_errors(
    stub_nvidia_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import gemma_audio

    async def broken_fetch(*_a: Any, **_kw: Any) -> Any:
        raise MediaError("fetch failed")

    monkeypatch.setattr(gemma_audio, "fetch_url", broken_fetch)

    src = GemmaAudioToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        with pytest.raises(Exception, match="gemma_audio: fetch failed"):
            await tool.ainvoke({"url": "https://x"})


@pytest.mark.asyncio
async def test_transcribe_audio_wraps_split_errors(
    stub_nvidia_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64

    from openbb_agent_server.plugins.tools import gemma_audio

    async def broken_split(*_a: Any, **_kw: Any) -> list[AudioSegment]:
        raise MediaError("ffmpeg not happy")

    monkeypatch.setattr(gemma_audio, "split_audio_bytes", broken_split)

    raw = b"audio"
    clip = FileRef(
        name="x.wav",
        mime="audio/wav",
        data_base64=base64.b64encode(raw).decode(),
    )
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        with pytest.raises(Exception, match="ffmpeg not happy"):
            await tool.ainvoke({"name": "x.wav"})


@pytest.mark.asyncio
async def test_transcribe_audio_multi_segment_concatenates_with_headers(
    stub_nvidia_module: Any,
    stub_split_multi: None,
) -> None:
    import base64

    raw = b"\x00" * 200
    clip = FileRef(
        name="x.wav",
        mime="audio/wav",
        data_base64=base64.b64encode(raw).decode(),
    )
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        out = await tool.ainvoke({"name": "x.wav"})
    assert out["segments"] == 2
    assert "--- segment 1/2 " in out["text"]
    assert "--- segment 2/2 " in out["text"]


@pytest.mark.asyncio
async def test_transcribe_audio_uploaded_no_mime_falls_back(
    stub_nvidia_module: Any,
    stub_split: list[tuple[bytes, float]],
) -> None:
    import base64

    raw = b"\x00" * 10
    clip = FileRef(
        name="clip",
        mime=None,
        data_base64=base64.b64encode(raw).decode(),
    )
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "transcribe_audio"]
        out = await tool.ainvoke({"name": "clip"})
    assert out["segments"] == 1


@pytest.mark.asyncio
async def test_submit_transcribe_audio_returns_job_id(
    stub_nvidia_module: Any,
    stub_split: list[tuple[bytes, float]],
) -> None:
    import base64

    raw = b"\x00" * 10
    clip = FileRef(
        name="x.wav",
        mime="audio/wav",
        data_base64=base64.b64encode(raw).decode(),
    )
    src = GemmaAudioToolSource()
    ctx = _ctx(clip, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [submit] = [t for t in tools if t.name == "submit_transcribe_audio"]
        out = await submit.ainvoke({"name": "x.wav"})
    assert "job_id" in out
    assert "label" in out
    assert "x.wav" in out["label"]


@pytest.mark.asyncio
async def test_tools_respects_config_overrides(stub_nvidia_module: Any) -> None:
    """Propagate a custom model/base_url to ChatNVIDIA kwargs."""

    import base64

    from openbb_agent_server.plugins.tools import gemma_audio

    async def fake_split(*_a: Any, **_kw: Any) -> list[AudioSegment]:
        return [
            AudioSegment(
                index=0,
                total=1,
                start_s=0.0,
                end_s=1.0,
                data=b"x",
                mime="audio/ogg",
            )
        ]

    import pytest as _pt

    mp = _pt.MonkeyPatch()
    try:
        mp.setattr(gemma_audio, "split_audio_bytes", fake_split)
        raw = b"\x00" * 10
        clip = FileRef(
            name="x.wav",
            mime="audio/wav",
            data_base64=base64.b64encode(raw).decode(),
        )
        src = GemmaAudioToolSource()
        ctx = _ctx(clip, api_key="k")
        with bind(ctx):
            tools = await src.tools(
                ctx,
                {
                    "model": "alt",
                    "base_url": "https://x",
                    "temperature": 0.5,
                    "max_fetch_bytes": 100_000,
                    "max_audio_seconds_per_call": 30.0,
                    "fetch_timeout_s": 5.0,
                },
            )
            [tool] = [t for t in tools if t.name == "transcribe_audio"]
            await tool.ainvoke({"name": "x.wav"})
        [instance] = _StubChatNVIDIA.instances
        assert instance.kwargs["model"] == "alt"
        assert instance.kwargs["base_url"] == "https://x"
        assert instance.kwargs["temperature"] == 0.5
    finally:
        mp.undo()
