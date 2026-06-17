"""vision_qa tool source tests."""

from __future__ import annotations

import base64
import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest

from openbb_agent_server.plugins.tools.vision_qa import (
    VisionQaToolSource,
    _is_image,
    _resolve_data_url,
)
from openbb_agent_server.runtime.context import FileRef, RunContext, bind
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(*files: FileRef, api_key: str | None = None) -> RunContext:
    api_keys: dict[str, str] = {}
    if api_key is not None:
        api_keys["NVIDIA_API_KEY"] = api_key
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=api_keys,
        uploaded_files=tuple(files),
    )


def test_is_image_recognises_mime() -> None:
    assert _is_image(FileRef(name="x.dat", mime="image/png", data_base64="x"))


def test_is_image_recognises_extension() -> None:
    assert _is_image(FileRef(name="hello.jpg", data_base64="x"))


def test_is_image_recognises_uppercase_extension() -> None:
    assert _is_image(FileRef(name="HELLO.PNG", data_base64="x"))


def test_is_image_rejects_non_image() -> None:
    assert not _is_image(FileRef(name="audio.mp3", data_base64="x"))


def test_is_image_rejects_when_no_name_or_mime() -> None:
    assert not _is_image(FileRef(name="", data_base64="x"))


@pytest.mark.asyncio
async def test_resolve_data_url_from_base64_keeps_explicit_mime() -> None:
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
async def test_resolve_data_url_guesses_mime_from_extension() -> None:
    f = FileRef(name="x.png", mime=None, data_base64="ZGF0YQ==")
    out = await _resolve_data_url(f, max_bytes=100, timeout_s=1)
    assert out.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_resolve_data_url_falls_back_to_default_mime() -> None:
    f = FileRef(name="", mime=None, data_base64="ZGF0YQ==")
    out = await _resolve_data_url(f, max_bytes=100, timeout_s=1)
    assert out.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_resolve_data_url_raises_when_no_source() -> None:
    f = FileRef(name="x.png", mime="image/png")
    with pytest.raises(RuntimeError, match="has no data_base64 or url"):
        await _resolve_data_url(f, max_bytes=100, timeout_s=1)


@pytest.mark.asyncio
async def test_resolve_data_url_fetches_from_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch the URL when data_base64 is absent."""

    from openbb_agent_server.plugins.tools import _media, vision_qa

    class _Fetched:
        def __init__(self) -> None:
            self.data = b"binarydata"
            self.mime = "image/png"

    async def fake_fetch_url(*_a: Any, **_kw: Any) -> _Fetched:
        return _Fetched()

    async def fake_to_data_url(data: bytes, *, mime: str) -> str:
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    monkeypatch.setattr(vision_qa, "fetch_url", fake_fetch_url)
    monkeypatch.setattr(vision_qa, "to_data_url", fake_to_data_url)

    f = FileRef(name="x.png", mime="image/png", url="https://x/y.png")
    out = await _resolve_data_url(f, max_bytes=100, timeout_s=1)
    assert out.startswith("data:image/png;base64,")
    assert _media.fetch_url is not None


class _StubChatNVIDIA:
    """Fake LangChain chat client; astream yields one chunk per call."""

    instances: list[_StubChatNVIDIA] = []
    fake_reply: str = "HELLO"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _StubChatNVIDIA.instances.append(self)

    def astream(self, _messages: Any) -> AsyncIterator[Any]:
        return self._astream()

    async def _astream(self) -> AsyncIterator[Any]:
        class _Chunk:
            def __init__(self, content: str) -> None:
                self.content = content

        yield _Chunk(self.fake_reply[:3])
        yield _Chunk(self.fake_reply[3:])


@pytest.fixture
def stub_nvidia_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake langchain_nvidia_ai_endpoints module."""
    import types

    fake_mod = types.ModuleType("langchain_nvidia_ai_endpoints")
    fake_mod.ChatNVIDIA = _StubChatNVIDIA  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", fake_mod)
    _StubChatNVIDIA.instances.clear()
    _StubChatNVIDIA.fake_reply = "HELLO"
    return fake_mod


@pytest.mark.asyncio
async def test_tools_returns_empty_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    src = VisionQaToolSource()
    out = await src.tools(_ctx(), {})
    assert out == []
    assert any("NVIDIA_API_KEY is not set" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_tools_registers_three_tools_when_key_available(
    stub_nvidia_module: Any,
) -> None:
    src = VisionQaToolSource(api_key="testkey")
    out = await src.tools(_ctx(), {})
    names = {t.name for t in out}
    assert names == {"list_images", "understand_image", "submit_understand_image"}


@pytest.mark.asyncio
async def test_tools_pulls_api_key_from_context(stub_nvidia_module: Any) -> None:
    src = VisionQaToolSource()
    out = await src.tools(_ctx(api_key="ctx-key"), {})
    assert len(out) == 3


@pytest.mark.asyncio
async def test_tools_pulls_api_key_from_config(stub_nvidia_module: Any) -> None:
    src = VisionQaToolSource()
    out = await src.tools(_ctx(), {"api_key": "cfg-key"})
    assert len(out) == 3


@pytest.mark.asyncio
async def test_tools_pulls_api_key_from_env(
    stub_nvidia_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    src = VisionQaToolSource()
    out = await src.tools(_ctx(), {})
    assert len(out) == 3


@pytest.mark.asyncio
async def test_list_images_returns_uploaded(stub_nvidia_module: Any) -> None:
    img = FileRef(name="hello.png", mime="image/png", data_base64="x")
    audio = FileRef(name="clip.mp3", mime="audio/mp3", data_base64="y")
    ctx = _ctx(img, audio, api_key="k")
    src = VisionQaToolSource()
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [list_tool] = [t for t in tools if t.name == "list_images"]
        result = list_tool.invoke({})
    assert len(result) == 1
    assert result[0]["name"] == "hello.png"


@pytest.mark.asyncio
async def test_understand_image_returns_streamed_reply(
    stub_nvidia_module: Any,
) -> None:
    img = FileRef(name="hello.png", mime="image/png", data_base64="ZGF0YQ==")
    ctx = _ctx(img, api_key="k")
    src = VisionQaToolSource()
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "understand_image"]
        out = await tool.ainvoke(
            {
                "name": "hello.png",
                "instruction": "What is this?",
                "max_output_tokens": 128,
            }
        )
    assert out == "HELLO"


@pytest.mark.asyncio
async def test_understand_image_rejects_missing_name_and_url(
    stub_nvidia_module: Any,
) -> None:
    src = VisionQaToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "understand_image"]
        with pytest.raises(Exception, match="provide either"):
            await tool.ainvoke({"instruction": "hi"})


@pytest.mark.asyncio
async def test_understand_image_rejects_unknown_name(
    stub_nvidia_module: Any,
) -> None:
    src = VisionQaToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "understand_image"]
        with pytest.raises(Exception, match="not among this run's uploads"):
            await tool.ainvoke({"name": "missing.png", "instruction": "x"})


@pytest.mark.asyncio
async def test_understand_image_fetches_from_url(
    stub_nvidia_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import vision_qa

    class _Fetched:
        data = b"bin"
        mime = "image/png"

    async def fake_fetch_url(*_a: Any, **_kw: Any) -> _Fetched:
        return _Fetched()

    async def fake_to_data_url(data: bytes, *, mime: str) -> str:
        return f"data:{mime};base64,Ymlu"

    monkeypatch.setattr(vision_qa, "fetch_url", fake_fetch_url)
    monkeypatch.setattr(vision_qa, "to_data_url", fake_to_data_url)

    src = VisionQaToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "understand_image"]
        out = await tool.ainvoke({"url": "https://x/y.png", "instruction": "What?"})
    assert out == "HELLO"


@pytest.mark.asyncio
async def test_understand_image_wraps_media_errors_with_name(
    stub_nvidia_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import (
        _media as media_mod,
        vision_qa,
    )

    async def broken_resolve(*_a: Any, **_kw: Any) -> str:
        raise media_mod.MediaError("nope")

    monkeypatch.setattr(vision_qa, "_resolve_data_url", broken_resolve)

    img = FileRef(name="x.png", mime="image/png", data_base64="x")
    src = VisionQaToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "understand_image"]
        with pytest.raises(Exception, match="vision_qa: nope"):
            await tool.ainvoke({"name": "x.png", "instruction": "x"})


@pytest.mark.asyncio
async def test_understand_image_wraps_media_errors_with_url(
    stub_nvidia_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools import (
        _media as media_mod,
        vision_qa,
    )

    async def broken_fetch(*_a: Any, **_kw: Any) -> Any:
        raise media_mod.MediaError("nope")

    monkeypatch.setattr(vision_qa, "fetch_url", broken_fetch)

    src = VisionQaToolSource()
    ctx = _ctx(api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [tool] = [t for t in tools if t.name == "understand_image"]
        with pytest.raises(Exception, match="vision_qa: nope"):
            await tool.ainvoke({"url": "https://x", "instruction": "x"})


@pytest.mark.asyncio
async def test_submit_understand_image_returns_job_id(
    stub_nvidia_module: Any,
) -> None:
    img = FileRef(name="hello.png", mime="image/png", data_base64="ZGF0YQ==")
    src = VisionQaToolSource()
    ctx = _ctx(img, api_key="k")
    with bind(ctx):
        tools = await src.tools(ctx, {})
        [submit] = [t for t in tools if t.name == "submit_understand_image"]
        result = await submit.ainvoke({"name": "hello.png", "instruction": "x"})
    assert "job_id" in result
    assert "label" in result
    assert "hello.png" in result["label"]


@pytest.mark.asyncio
async def test_tools_respects_config_overrides(stub_nvidia_module: Any) -> None:
    """Reflect the config-supplied overrides in the client kwargs."""

    src = VisionQaToolSource(model="default-model")
    out = await src.tools(
        _ctx(api_key="k"),
        {
            "model": "custom",
            "base_url": "https://x",
            "temperature": 0.9,
            "max_fetch_bytes": 1_000,
            "fetch_timeout_s": 5.0,
        },
    )
    assert len(out) == 3
    img = FileRef(name="x.png", mime="image/png", data_base64="ZGF0YQ==")
    ctx = _ctx(img, api_key="k")
    src2 = VisionQaToolSource()
    with bind(ctx):
        tools = await src2.tools(
            ctx, {"model": "custom-m", "base_url": "https://example.com"}
        )
        [tool] = [t for t in tools if t.name == "understand_image"]
        await tool.ainvoke({"name": "x.png", "instruction": "x"})
    [instance] = _StubChatNVIDIA.instances
    assert instance.kwargs["model"] == "custom-m"
    assert instance.kwargs["base_url"] == "https://example.com"
