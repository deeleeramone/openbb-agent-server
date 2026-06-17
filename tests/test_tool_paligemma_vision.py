"""paligemma_vision tool source tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from openbb_agent_server.plugins.tools.paligemma_vision import (
    PaliGemmaVisionToolSource,
    _is_image,
    _resolve_data_url,
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


def test_is_image_mime() -> None:
    assert _is_image(FileRef(name="x.dat", mime="image/png", data_base64="x"))


def test_is_image_extension() -> None:
    assert _is_image(FileRef(name="x.jpg", data_base64="x"))


def test_is_image_uppercase_extension() -> None:
    assert _is_image(FileRef(name="X.PNG", data_base64="x"))


def test_is_image_rejects_audio() -> None:
    assert not _is_image(FileRef(name="x.mp3", data_base64="x"))


@pytest.mark.asyncio
async def test_resolve_data_url_from_base64() -> None:
    f = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    out = await _resolve_data_url(f, max_bytes=100, timeout_s=1)
    assert out == "data:image/png;base64,ZGF0YQ=="


@pytest.mark.asyncio
async def test_resolve_data_url_strips_existing_data_prefix() -> None:
    f = FileRef(
        name="x.png", mime="image/png", data_base64="data:image/jpeg;base64,ZGF0YQ=="
    )
    out = await _resolve_data_url(f, max_bytes=100, timeout_s=1)
    assert out == "data:image/png;base64,ZGF0YQ=="


@pytest.mark.asyncio
async def test_resolve_data_url_guesses_mime() -> None:
    f = FileRef(name="x.png", mime=None, data_base64="ZGF0YQ==")
    out = await _resolve_data_url(f, max_bytes=100, timeout_s=1)
    assert out.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_resolve_data_url_fetches_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import paligemma_vision

    class _Fetched:
        data = b"bin"
        mime = "image/png"

    async def fake_fetch(*_a: Any, **_kw: Any) -> _Fetched:
        return _Fetched()

    async def fake_to_data_url(data: bytes, *, mime: str) -> str:
        return f"data:{mime};base64,Ymlu"

    monkeypatch.setattr(paligemma_vision, "fetch_url", fake_fetch)
    monkeypatch.setattr(paligemma_vision, "to_data_url", fake_to_data_url)

    f = FileRef(name="x.png", mime="image/png", url="https://x/y.png")
    out = await _resolve_data_url(f, max_bytes=100, timeout_s=1)
    assert out == "data:image/png;base64,Ymlu"


@pytest.mark.asyncio
async def test_resolve_data_url_raises_when_no_source() -> None:
    f = FileRef(name="x.png", mime="image/png")
    with pytest.raises(ValueError, match="has neither data_base64 nor url"):
        await _resolve_data_url(f, max_bytes=100, timeout_s=1)


class _SseTransport(httpx.AsyncBaseTransport):
    """Fake NIM VLM endpoint that streams SSE chunks."""

    def __init__(
        self,
        *,
        frames: list[str] | None = None,
        status_code: int = 200,
    ) -> None:
        if frames is None:
            frames = [
                'data: {"choices": [{"delta": {"content": "HEL"}}]}\n\n',
                'data: {"choices": [{"delta": {"content": "LO"}}]}\n\n',
                "data: [DONE]\n\n",
            ]
        self.frames = frames
        self.status_code = status_code
        self.received_requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.received_requests.append(request)
        body = "".join(self.frames).encode("utf-8")
        return httpx.Response(
            self.status_code,
            content=body,
            headers={"content-type": "text/event-stream"},
        )


@pytest.fixture
def stub_httpx(monkeypatch: pytest.MonkeyPatch) -> _SseTransport:
    """Patch httpx.AsyncClient to use a controllable transport."""
    transport = _SseTransport()
    real_async_client = httpx.AsyncClient

    def _factory(*_a: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*_a, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return transport


@pytest.mark.asyncio
async def test_tools_returns_empty_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    src = PaliGemmaVisionToolSource()
    out = await src.tools(_ctx(), {})
    assert out == []
    assert any("NVIDIA_API_KEY is not set" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_tools_registers_all_tools_when_key_present() -> None:
    src = PaliGemmaVisionToolSource(api_key="k")
    out = await src.tools(_ctx(), {})
    names = {t.name for t in out}
    assert names == {
        "list_images",
        "caption_image",
        "read_image_text",
        "ask_about_image",
        "submit_caption_image",
        "submit_read_image_text",
        "submit_ask_about_image",
    }


@pytest.mark.asyncio
async def test_tools_pulls_key_from_context() -> None:
    src = PaliGemmaVisionToolSource()
    out = await src.tools(_ctx(api_key="ctx"), {})
    assert len(out) == 7


@pytest.mark.asyncio
async def test_tools_pulls_key_from_config() -> None:
    src = PaliGemmaVisionToolSource()
    out = await src.tools(_ctx(), {"api_key": "cfg"})
    assert len(out) == 7


@pytest.mark.asyncio
async def test_tools_pulls_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env")
    src = PaliGemmaVisionToolSource()
    out = await src.tools(_ctx(), {})
    assert len(out) == 7


@pytest.mark.asyncio
async def test_list_images_returns_uploaded() -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="x")
    audio = FileRef(name="x.mp3", mime="audio/mp3", data_base64="x")
    ctx = _ctx(img, audio, api_key="k")
    src = PaliGemmaVisionToolSource()
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [list_tool] = [t for t in tools if t.name == "list_images"]
        out = list_tool.invoke({})
    assert len(out) == 1
    assert out[0]["name"] == "x.png"


@pytest.mark.asyncio
async def test_caption_image_calls_vlm_endpoint(stub_httpx: _SseTransport) -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    ctx = _ctx(img, api_key="k")
    src = PaliGemmaVisionToolSource()
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        out = await tool.ainvoke({"name": "x.png", "language": "en"})

    assert out["caption"] == "HELLO"
    assert out["target"] == "x.png"
    [req] = stub_httpx.received_requests
    assert req.method == "POST"
    assert "vlm" in str(req.url)
    body = json.loads(req.content.decode())
    assert body["stream"] is True
    assert body["max_tokens"] <= 1024
    assert "caption en" in body["messages"][0]["content"]


@pytest.mark.asyncio
async def test_caption_image_caps_max_tokens_at_paligemma_ceiling(
    stub_httpx: _SseTransport,
) -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    ctx = _ctx(img, api_key="k")
    src = PaliGemmaVisionToolSource()
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        await tool.ainvoke({"name": "x.png"})
    body = json.loads(stub_httpx.received_requests[0].content.decode())
    assert body["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_read_image_text_runs_ocr(stub_httpx: _SseTransport) -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    ctx = _ctx(img, api_key="k")
    src = PaliGemmaVisionToolSource()
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "read_image_text"]
        out = await tool.ainvoke({"name": "x.png"})
    assert out["text"] == "HELLO"
    body = json.loads(stub_httpx.received_requests[0].content.decode())
    assert "ocr" in body["messages"][0]["content"]


@pytest.mark.asyncio
async def test_ask_about_image(stub_httpx: _SseTransport) -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    ctx = _ctx(img, api_key="k")
    src = PaliGemmaVisionToolSource()
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "ask_about_image"]
        out = await tool.ainvoke(
            {"question": "How many?", "name": "x.png", "language": "fr"}
        )
    assert out["answer"] == "HELLO"
    assert out["question"] == "How many?"
    body = json.loads(stub_httpx.received_requests[0].content.decode())
    assert "answer fr How many?" in body["messages"][0]["content"]


@pytest.mark.asyncio
async def test_caption_image_fetches_from_url(
    stub_httpx: _SseTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import paligemma_vision

    class _Fetched:
        data = b"bin"
        mime = "image/png"

    async def fake_fetch(*_a: Any, **_kw: Any) -> _Fetched:
        return _Fetched()

    async def fake_to_data_url(data: bytes, *, mime: str) -> str:
        return f"data:{mime};base64,Ymlu"

    monkeypatch.setattr(paligemma_vision, "fetch_url", fake_fetch)
    monkeypatch.setattr(paligemma_vision, "to_data_url", fake_to_data_url)

    src = PaliGemmaVisionToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        out = await tool.ainvoke({"url": "https://x/y.png"})
    assert out["target"] == "https://x/y.png"


@pytest.mark.asyncio
async def test_caption_image_rejects_unknown_name(
    stub_httpx: _SseTransport,
) -> None:
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        with pytest.raises(Exception, match="not among this run's uploads"):
            await tool.ainvoke({"name": "missing.png"})


@pytest.mark.asyncio
async def test_caption_image_rejects_when_no_name_or_url(
    stub_httpx: _SseTransport,
) -> None:
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        with pytest.raises(Exception, match="must pass 'name' or 'url'"):
            await tool.ainvoke({})


@pytest.mark.asyncio
async def test_caption_image_wraps_media_error_with_name(
    stub_httpx: _SseTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import (
        _media as media_mod,
        paligemma_vision,
    )

    async def broken_resolve(*_a: Any, **_kw: Any) -> str:
        raise media_mod.MediaError("nope")

    monkeypatch.setattr(paligemma_vision, "_resolve_data_url", broken_resolve)

    img = FileRef(name="x.png", mime="image/png", data_base64="x")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        with pytest.raises(Exception, match="paligemma_vision: nope"):
            await tool.ainvoke({"name": "x.png"})


@pytest.mark.asyncio
async def test_caption_image_wraps_media_error_with_url(
    stub_httpx: _SseTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import (
        _media as media_mod,
        paligemma_vision,
    )

    async def broken_fetch(*_a: Any, **_kw: Any) -> Any:
        raise media_mod.MediaError("nope")

    monkeypatch.setattr(paligemma_vision, "fetch_url", broken_fetch)

    src = PaliGemmaVisionToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        with pytest.raises(Exception, match="paligemma_vision: nope"):
            await tool.ainvoke({"url": "https://x"})


@pytest.mark.asyncio
async def test_call_skips_done_marker(stub_httpx: _SseTransport) -> None:
    """Terminate the streaming loop on data: [DONE]."""
    stub_httpx.frames = [
        'data: {"choices": [{"delta": {"content": "PART1"}}]}\n\n',
        "data: [DONE]\n\n",
        'data: {"choices": [{"delta": {"content": "IGNORED"}}]}\n\n',
    ]
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        out = await tool.ainvoke({"name": "x.png"})
    assert out["caption"] == "PART1"


@pytest.mark.asyncio
async def test_call_skips_empty_lines(stub_httpx: _SseTransport) -> None:
    stub_httpx.frames = [
        "\n\n",
        ": comment\n",
        'data: {"choices": [{"delta": {"content": "OK"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        out = await tool.ainvoke({"name": "x.png"})
    assert out["caption"] == "OK"


@pytest.mark.asyncio
async def test_call_skips_malformed_json(stub_httpx: _SseTransport) -> None:
    stub_httpx.frames = [
        "data: {invalid json\n\n",
        'data: {"choices": [{"delta": {"content": "FINE"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        out = await tool.ainvoke({"name": "x.png"})
    assert out["caption"] == "FINE"


@pytest.mark.asyncio
async def test_call_handles_no_delta(stub_httpx: _SseTransport) -> None:
    """Drop frames without a delta.content field."""
    stub_httpx.frames = [
        'data: {"choices": [{"finish_reason": "stop"}]}\n\n',
        'data: {"choices": []}\n\n',
        "data: {}\n\n",
        'data: {"choices": [{"delta": {"content": "Z"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        out = await tool.ainvoke({"name": "x.png"})
    assert out["caption"] == "Z"


@pytest.mark.asyncio
async def test_call_raises_on_http_error(stub_httpx: _SseTransport) -> None:
    stub_httpx.status_code = 500
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "caption_image"]
        with pytest.raises(httpx.HTTPStatusError):
            await tool.ainvoke({"name": "x.png"})


@pytest.mark.asyncio
async def test_submit_caption_image_runs_via_registry(
    stub_httpx: _SseTransport,
) -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [submit] = [t for t in tools if t.name == "submit_caption_image"]
        out = await submit.ainvoke({"name": "x.png"})
    assert "job_id" in out
    assert "caption_image" in out["label"]


@pytest.mark.asyncio
async def test_submit_read_image_text(stub_httpx: _SseTransport) -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [submit] = [t for t in tools if t.name == "submit_read_image_text"]
        out = await submit.ainvoke({"name": "x.png"})
    assert "job_id" in out
    assert "read_image_text" in out["label"]


@pytest.mark.asyncio
async def test_submit_ask_about_image_carries_question_in_metadata(
    stub_httpx: _SseTransport,
) -> None:
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    src = PaliGemmaVisionToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [submit] = [t for t in tools if t.name == "submit_ask_about_image"]
        out = await submit.ainvoke({"question": "What?", "name": "x.png"})
    assert "job_id" in out
    assert "ask_about_image" in out["label"]
