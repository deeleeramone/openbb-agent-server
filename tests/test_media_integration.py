"""Integration tests for the media tools."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import math
import socket
import struct
import wave
from collections.abc import AsyncIterator
from typing import Any

import pytest
import uvicorn
from langchain_core.messages import AIMessageChunk
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route

from openbb_agent_server.plugins.tools._media import (
    FFmpegUnavailableError,
    MediaFetchError,
    MediaTooLargeError,
    fetch_url,
    ffmpeg_available,
    flatten_message_content,
    probe_audio_duration,
    split_audio_bytes,
    to_data_url,
)


def _free_port() -> int:
    """Return a kernel-allocated free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _payload_bytes(request: Any) -> bytes:
    """Return deterministic bytes sized by a query param."""
    size = int(request.query_params.get("size", "1024"))
    return bytes((i * 7) % 256 for i in range(size))


def _build_app() -> Starlette:
    async def small(_: Any) -> Response:
        return Response(content=b"small-body-OK", media_type="application/octet-stream")

    async def large(request: Any) -> Response:
        body = await _payload_bytes(request)
        return Response(content=body, media_type="application/octet-stream")

    async def slow_stream(request: Any) -> StreamingResponse:
        size = int(request.query_params.get("size", "4096"))
        chunk = bytes(256)

        async def gen() -> AsyncIterator[bytes]:
            sent = 0
            while sent < size:
                n = min(len(chunk), size - sent)
                yield chunk[:n]
                sent += n
                await asyncio.sleep(0.001)

        return StreamingResponse(gen(), media_type="application/octet-stream")

    async def png(_: Any) -> Response:
        body = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae4260"
            "82"
        )
        return Response(content=body, media_type="image/png")

    async def with_charset(_: Any) -> Response:
        return Response(content=b"hello", media_type="text/plain; charset=utf-8")

    async def notfound(_: Any) -> JSONResponse:
        return JSONResponse({"err": "no"}, status_code=404)

    async def no_ct_asgi(scope: Any, receive: Any, send: Any) -> None:
        """Bare ASGI app that emits no content-type header."""
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"nocontenttype"})

    return Starlette(
        routes=[
            Route("/small.bin", small),
            Route("/large.bin", large),
            Route("/slow", slow_stream),
            Route("/image.png", png),
            Route("/with-charset", with_charset),
            Mount("/no-ct", app=no_ct_asgi),
            Route("/notfound", notfound),
        ]
    )


class _ServerHandle:
    """A running uvicorn server on a known port."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.base = f"http://{host}:{port}"
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        config = uvicorn.Config(
            app=_build_app(),
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
        raise RuntimeError("test HTTP server never reached started=True")

    async def stop(self) -> None:
        """Tear down the uvicorn server promptly."""
        if self._server is not None:
            self._server.should_exit = True
            self._server.force_exit = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=2.0)


@pytest.fixture
async def http_server() -> AsyncIterator[_ServerHandle]:
    handle = _ServerHandle("127.0.0.1", _free_port())
    await handle.start()
    try:
        yield handle
    finally:
        await handle.stop()


async def test_fetch_url_round_trips_real_bytes(http_server: _ServerHandle) -> None:
    fetched = await fetch_url(
        f"{http_server.base}/small.bin",
        max_bytes=1024,
        timeout_s=5.0,
    )
    assert fetched.data == b"small-body-OK"
    assert fetched.mime == "application/octet-stream"
    assert fetched.size_bytes == 13


async def test_fetch_url_streams_large_payload_in_chunks(
    http_server: _ServerHandle,
) -> None:
    expected = bytes((i * 7) % 256 for i in range(8192))
    fetched = await fetch_url(
        f"{http_server.base}/large.bin?size=8192",
        max_bytes=16384,
        timeout_s=5.0,
    )
    assert fetched.data == expected
    assert len(fetched.data) == 8192


async def test_fetch_url_strips_content_type_parameters(
    http_server: _ServerHandle,
) -> None:
    """Strip content-type parameters down to the bare MIME."""
    fetched = await fetch_url(
        f"{http_server.base}/with-charset",
        max_bytes=1024,
        timeout_s=5.0,
        fallback_mime="application/json",
    )
    assert fetched.mime == "text/plain"
    assert fetched.data == b"hello"


async def test_fetch_url_uses_fallback_mime_when_server_omits_header(
    http_server: _ServerHandle,
) -> None:
    fetched = await fetch_url(
        f"{http_server.base}/no-ct/",
        max_bytes=1024,
        timeout_s=5.0,
        fallback_mime="audio/wav",
    )
    assert fetched.mime == "audio/wav"
    assert fetched.data == b"nocontenttype"


async def test_fetch_url_keeps_server_mime_for_image_path(
    http_server: _ServerHandle,
) -> None:
    fetched = await fetch_url(
        f"{http_server.base}/image.png",
        max_bytes=1024,
        timeout_s=5.0,
        fallback_mime="application/octet-stream",
    )
    assert fetched.mime == "image/png"
    assert fetched.data[:8] == b"\x89PNG\r\n\x1a\n"


async def test_fetch_url_raises_too_large_on_oversized_payload(
    http_server: _ServerHandle,
) -> None:
    """Raise MediaTooLargeError on an oversized streamed payload."""
    with pytest.raises(MediaTooLargeError) as exc_info:
        await fetch_url(
            f"{http_server.base}/slow?size=4096",
            max_bytes=1024,
            timeout_s=5.0,
        )
    assert "1024" in str(exc_info.value)


async def test_fetch_url_raises_too_large_on_static_oversized_payload(
    http_server: _ServerHandle,
) -> None:
    with pytest.raises(MediaTooLargeError):
        await fetch_url(
            f"{http_server.base}/large.bin?size=8192",
            max_bytes=2048,
            timeout_s=5.0,
        )


async def test_fetch_url_raises_fetch_error_on_404(
    http_server: _ServerHandle,
) -> None:
    with pytest.raises(MediaFetchError):
        await fetch_url(
            f"{http_server.base}/notfound",
            max_bytes=1024,
            timeout_s=5.0,
        )


async def test_fetch_url_raises_fetch_error_on_connection_refused() -> None:
    """Raise MediaFetchError on a refused connection."""
    port = _free_port()
    with pytest.raises(MediaFetchError):
        await fetch_url(
            f"http://127.0.0.1:{port}/nope",
            max_bytes=1024,
            timeout_s=1.0,
        )


async def test_to_data_url_encodes_real_bytes_off_thread() -> None:
    raw = b"\x00\x01\x02\x03 hello \xff\xfe"
    url = await to_data_url(raw, mime="application/octet-stream")
    head, _, payload = url.partition(",")
    assert head == "data:application/octet-stream;base64"
    assert base64.b64decode(payload) == raw


async def test_to_data_url_respects_explicit_max_bytes() -> None:
    with pytest.raises(MediaTooLargeError):
        await to_data_url(b"x" * 1024, mime="text/plain", max_bytes=512)


def test_flatten_handles_plain_string_chunks() -> None:
    chunk = AIMessageChunk(content="hello")
    assert flatten_message_content(chunk.content) == "hello"


def test_flatten_handles_typed_block_chunks() -> None:
    chunk = AIMessageChunk(
        content=[
            {"type": "text", "text": "part-A "},
            {"type": "image_url", "image_url": {"url": "ignored"}},
            {"type": "text", "text": "part-B"},
        ]
    )
    assert flatten_message_content(chunk.content) == "part-A part-B"


def test_flatten_handles_mixed_list_of_strings_and_dicts() -> None:
    content = ["raw-string ", {"type": "text", "text": "block-text"}]
    assert flatten_message_content(content) == "raw-string block-text"


def test_flatten_handles_none_and_empty() -> None:
    assert flatten_message_content(None) == ""
    assert flatten_message_content("") == ""
    assert flatten_message_content([]) == ""


def _synthesize_wav(duration_s: float, freq_hz: float = 440.0) -> bytes:
    """Generate a mono 24 kHz PCM-16 WAV of a given duration."""
    sample_rate = 24000
    n = int(duration_s * sample_rate)
    amplitude = 0.25
    frames = bytearray()
    for i in range(n):
        v = int(amplitude * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        frames.extend(struct.pack("<h", v))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg / ffprobe not installed")
async def test_probe_audio_duration_returns_close_to_expected() -> None:
    wav = _synthesize_wav(duration_s=2.0)
    duration = await probe_audio_duration(wav)
    assert 1.95 < duration < 2.05


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg / ffprobe not installed")
async def test_probe_audio_duration_raises_on_garbage() -> None:
    from openbb_agent_server.plugins.tools._media import MediaError

    with pytest.raises(MediaError):
        await probe_audio_duration(b"not-an-audio-file")


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg / ffprobe not installed")
async def test_split_audio_bytes_passes_short_clip_through_unchanged() -> None:
    wav = _synthesize_wav(duration_s=1.0)
    [seg] = await split_audio_bytes(wav, max_seconds=10.0)
    assert seg.data == wav
    assert seg.index == 0 and seg.total == 1


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg / ffprobe not installed")
async def test_split_audio_bytes_chunks_long_clip_into_segments() -> None:
    wav = _synthesize_wav(duration_s=5.0)
    segments = await split_audio_bytes(wav, max_seconds=2.0)
    assert len(segments) == 3
    assert [s.index for s in segments] == [0, 1, 2]
    assert {s.total for s in segments} == {3}
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(2.0)
    assert segments[1].start_s == pytest.approx(2.0)
    assert segments[1].end_s == pytest.approx(4.0)
    assert segments[2].start_s == pytest.approx(4.0)
    assert segments[2].end_s == pytest.approx(5.0)
    for seg in segments:
        assert seg.mime == "audio/ogg"
        assert seg.data[:4] == b"OggS"


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg / ffprobe not installed")
async def test_split_audio_bytes_rejects_non_positive_max_seconds() -> None:
    with pytest.raises(ValueError):
        await split_audio_bytes(b"", max_seconds=0)


def test_ffmpeg_required_helpers_raise_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise FFmpegUnavailableError when the binary is missing."""
    import openbb_agent_server.plugins.tools._media as media

    monkeypatch.setattr(media.shutil, "which", lambda _: None)
    with pytest.raises(FFmpegUnavailableError):
        media._require_binary("ffmpeg")


class _FakeProc:
    """Stand-in for asyncio.subprocess.Process."""

    def __init__(
        self, *, returncode: int, stdout: bytes = b"", stderr: bytes = b""
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:  # noqa: A002
        return self._stdout, self._stderr


async def test_probe_audio_duration_raises_when_ffprobe_returns_non_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise MediaError on non-numeric ffprobe output."""
    from openbb_agent_server.plugins.tools._media import (
        MediaError,
        probe_audio_duration,
    )

    async def fake_exec(*_args: object, **_kwargs: object) -> _FakeProc:
        return _FakeProc(returncode=0, stdout=b"not-a-number\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools._media._require_binary",
        lambda name: f"/fake/{name}",
    )
    with pytest.raises(MediaError, match="non-numeric"):
        await probe_audio_duration(b"any-payload")


async def test_split_audio_bytes_raises_when_ffmpeg_segment_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise MediaError when an ffmpeg segment call fails."""
    from openbb_agent_server.plugins.tools._media import (
        MediaError,
        split_audio_bytes,
    )

    calls: list[str] = []

    async def fake_exec(*args: object, **_kwargs: object) -> _FakeProc:
        binary = str(args[0])
        if binary.endswith("ffprobe"):
            calls.append("probe")
            return _FakeProc(returncode=0, stdout=b"10.0\n")
        calls.append("ffmpeg")
        return _FakeProc(returncode=1, stderr=b"some ffmpeg error")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools._media._require_binary",
        lambda name: f"/fake/{name}",
    )
    with pytest.raises(MediaError, match="ffmpeg failed"):
        await split_audio_bytes(b"long-clip", max_seconds=2.0)
    assert "ffmpeg" in calls
