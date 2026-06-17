"""Live NVIDIA NIM integration tests for the multimodal tool sources."""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import struct
import wave
from typing import Any

import pytest

from openbb_agent_server.plugins.tools.gemma_audio import GemmaAudioToolSource
from openbb_agent_server.plugins.tools.paligemma_vision import (
    PaliGemmaVisionToolSource,
)
from openbb_agent_server.plugins.tools.vision_qa import (
    VisionQaToolSource,
)
from openbb_agent_server.runtime.context import FileRef, RunContext, bind
from openbb_agent_server.runtime.jobs import JobState, get_registry
from openbb_agent_server.runtime.principal import UserPrincipal

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY"),
        reason="NVIDIA_API_KEY not set — live NIM tests skipped",
    ),
    pytest.mark.timeout(120),
]


def _silence_wav_b64(duration_s: float = 1.0) -> str:
    """Build a base64-encoded mono 24 kHz PCM-16 WAV of silence."""
    sample_rate = 24000
    n = int(duration_s * sample_rate)
    frames = bytes(2 * n)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tone_wav_b64(duration_s: float = 1.0, freq_hz: float = 440.0) -> str:
    """Build a base64-encoded short tone WAV."""
    sample_rate = 24000
    n = int(duration_s * sample_rate)
    frames = bytearray()
    for i in range(n):
        v = int(0.25 * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        frames.extend(struct.pack("<h", v))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _hello_png_b64() -> str:
    """Build a base64-encoded PNG with the rendered word HELLO."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (256, 96), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = "HELLO"
    try:
        font = ImageFont.load_default(size=48)
    except (AttributeError, TypeError):
        font = ImageFont.load_default()
    draw.text((20, 20), text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_ctx(*files: FileRef) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u-nim"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        uploaded_files=tuple(files),
    )


async def _tool_named(src: Any, ctx: RunContext, name: str) -> Any:
    tools = await src.tools(ctx, {})
    [t] = [x for x in tools if x.name == name]
    return t


_NIM_DEADLINE_S = 30.0
_PARALLEL_NIM_DEADLINE_S = 60.0


async def test_gemma_transcribe_audio_returns_non_empty_text() -> None:
    async with asyncio.timeout(_NIM_DEADLINE_S):
        f = FileRef(name="clip.wav", mime="audio/wav", data_base64=_tone_wav_b64(1.0))
        ctx = _make_ctx(f)
        src = GemmaAudioToolSource()
        with bind(ctx):
            tool = await _tool_named(src, ctx, "transcribe_audio")
            out = await tool.ainvoke(
                {
                    "name": "clip.wav",
                    "instruction": (
                        "Describe what you hear in one short sentence. "
                        "If there is no speech, say 'no speech'."
                    ),
                    "max_output_tokens": 256,
                }
            )
    assert isinstance(out, dict)
    assert isinstance(out.get("text"), str)
    assert out["text"].strip() != ""
    assert out["model"] == GemmaAudioToolSource()._model
    assert out["segments"] == 1


async def test_gemma_submit_transcribe_audio_completes_via_wait_for_job() -> None:
    """Submit then wait collects the real transcript bytes."""
    async with asyncio.timeout(_NIM_DEADLINE_S):
        f = FileRef(
            name="clip.wav", mime="audio/wav", data_base64=_silence_wav_b64(1.0)
        )
        ctx = _make_ctx(f)
        src = GemmaAudioToolSource()
        with bind(ctx):
            submit_tool = await _tool_named(src, ctx, "submit_transcribe_audio")
            submit_out = await submit_tool.ainvoke(
                {
                    "name": "clip.wav",
                    "instruction": "Transcribe. If silent, say '(silence)'.",
                    "max_output_tokens": 128,
                }
            )
            assert "job_id" in submit_out
            job_id = submit_out["job_id"]

            result = await get_registry().wait(job_id, timeout_s=_NIM_DEADLINE_S)
            assert result["state"] == JobState.DONE.value
            body = result["result"]
            assert isinstance(body, dict)
            assert isinstance(body.get("text"), str)
            assert body["text"].strip() != ""


async def test_phi4_understand_image_reads_visible_text() -> None:
    async with asyncio.timeout(_NIM_DEADLINE_S):
        f = FileRef(name="hello.png", mime="image/png", data_base64=_hello_png_b64())
        ctx = _make_ctx(f)
        src = VisionQaToolSource()
        with bind(ctx):
            tool = await _tool_named(src, ctx, "understand_image")
            text = await tool.ainvoke(
                {
                    "name": "hello.png",
                    "instruction": (
                        "What is the text written in this image? Reply "
                        "with just the visible word, nothing else."
                    ),
                    "max_output_tokens": 64,
                }
            )
    assert isinstance(text, str)
    assert "hello" in text.lower()


async def test_phi4_submit_understand_image_round_trips_via_registry() -> None:
    async with asyncio.timeout(_NIM_DEADLINE_S):
        f = FileRef(name="hello.png", mime="image/png", data_base64=_hello_png_b64())
        ctx = _make_ctx(f)
        src = VisionQaToolSource()
        with bind(ctx):
            submit_tool = await _tool_named(src, ctx, "submit_understand_image")
            out = await submit_tool.ainvoke(
                {
                    "name": "hello.png",
                    "instruction": "OCR the image. One word.",
                    "max_output_tokens": 64,
                }
            )
            job_id = out["job_id"]
            result = await get_registry().wait(job_id, timeout_s=_NIM_DEADLINE_S)
            assert result["state"] == JobState.DONE.value
            assert "hello" in str(result["result"]).lower()


async def test_paligemma_caption_image_returns_description() -> None:
    async with asyncio.timeout(_NIM_DEADLINE_S):
        f = FileRef(name="hello.png", mime="image/png", data_base64=_hello_png_b64())
        ctx = _make_ctx(f)
        src = PaliGemmaVisionToolSource()
        with bind(ctx):
            tool = await _tool_named(src, ctx, "caption_image")
            out = await tool.ainvoke({"name": "hello.png", "language": "en"})
    assert isinstance(out, dict)
    assert isinstance(out.get("caption"), str)
    assert out["caption"].strip() != ""


async def test_paligemma_read_image_text_returns_ocr() -> None:
    async with asyncio.timeout(_NIM_DEADLINE_S):
        f = FileRef(name="hello.png", mime="image/png", data_base64=_hello_png_b64())
        ctx = _make_ctx(f)
        src = PaliGemmaVisionToolSource()
        with bind(ctx):
            tool = await _tool_named(src, ctx, "read_image_text")
            out = await tool.ainvoke({"name": "hello.png"})
    assert isinstance(out, dict)
    assert isinstance(out.get("text"), str)
    assert out["text"].strip() != ""


async def test_two_real_nim_jobs_run_concurrently() -> None:
    """Two understand_image jobs run concurrently faster than serially."""
    import time

    async with asyncio.timeout(_PARALLEL_NIM_DEADLINE_S):
        f = FileRef(name="hello.png", mime="image/png", data_base64=_hello_png_b64())
        ctx = _make_ctx(f)
        src = VisionQaToolSource()
        with bind(ctx):
            submit_tool = await _tool_named(src, ctx, "submit_understand_image")
            start = time.perf_counter()
            j1 = (
                await submit_tool.ainvoke(
                    {
                        "name": "hello.png",
                        "instruction": "One word: what does the image say?",
                        "max_output_tokens": 64,
                    }
                )
            )["job_id"]
            j2 = (
                await submit_tool.ainvoke(
                    {
                        "name": "hello.png",
                        "instruction": "One word: describe the colour scheme.",
                        "max_output_tokens": 64,
                    }
                )
            )["job_id"]
            results = await asyncio.gather(
                get_registry().wait(j1, timeout_s=_PARALLEL_NIM_DEADLINE_S),
                get_registry().wait(j2, timeout_s=_PARALLEL_NIM_DEADLINE_S),
            )
            elapsed = time.perf_counter() - start

    assert all(r["state"] == JobState.DONE.value for r in results), results
    assert elapsed < _PARALLEL_NIM_DEADLINE_S, (
        f"two parallel NIM jobs took {elapsed:.1f}s"
    )
