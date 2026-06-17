"""Tests for the gemini_image + gemini_embeddings tool sources."""

from __future__ import annotations

import base64
import sys
import types as pytypes
from typing import Any

import httpx
import pytest

from openbb_agent_server.runtime.context import FileRef, RunContext, bind
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


def test_gemini_image_constructor_rejects_bad_max_retries() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    with pytest.raises(ValueError, match="max_retries"):
        GeminiImageToolSource(max_retries=-1)


def test_gemini_image_constructor_rejects_bad_timeout() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    with pytest.raises(ValueError, match="timeout"):
        GeminiImageToolSource(timeout=0)


def test_gemini_embeddings_constructor_rejects_unknown_task_type() -> None:
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    with pytest.raises(ValueError, match="default_task_type"):
        GeminiEmbeddingsToolSource(default_task_type="WEIRD")


def test_gemini_embeddings_constructor_rejects_bad_dimensionality() -> None:
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    with pytest.raises(ValueError, match="dimensionality"):
        GeminiEmbeddingsToolSource(default_output_dimensionality=0)


async def test_image_tools_require_api_key() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        await src.tools(_ctx(), {})


async def test_embeddings_tools_require_api_key() -> None:
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        await src.tools(_ctx(), {})


async def test_image_tools_returns_generate_and_edit() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="k")
    tools = await src.tools(_ctx(), {})
    assert {t.name for t in tools} == {"generate_image", "edit_image"}


async def test_embeddings_tools_returns_embed_text() -> None:
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource(api_key="k")
    tools = await src.tools(_ctx(), {})
    assert {t.name for t in tools} == {"embed_text"}


def _install_fake_google_genai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    images_to_return: list[tuple[bytes, str]] | None = None,
    raise_exc: BaseException | None = None,
    on_call: Any = None,
) -> dict[str, Any]:
    """Replace google.genai and google.genai.types with stubs."""
    captured: dict[str, Any] = {"calls": []}

    fake_types = pytypes.ModuleType("google.genai.types")

    class _GenerateImagesConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Part:
        def __init__(
            self,
            *,
            data: Any = None,
            mime_type: str | None = None,
            text: str | None = None,
        ) -> None:
            self.inline_data = (
                pytypes.SimpleNamespace(data=data, mime_type=mime_type)
                if data is not None
                else None
            )
            self.text = text

        @classmethod
        def from_bytes(cls, *, data: bytes, mime_type: str) -> _Part:
            return cls(data=data, mime_type=mime_type)

        @classmethod
        def from_text(cls, *, text: str) -> _Part:
            return cls(text=text)

    class _Content:
        def __init__(self, *, role: str, parts: list[_Part]) -> None:
            self.role = role
            self.parts = parts

    class _GenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_types.GenerateImagesConfig = _GenerateImagesConfig
    fake_types.GenerateContentConfig = _GenerateContentConfig
    fake_types.Part = _Part
    fake_types.Content = _Content

    fake_genai = pytypes.ModuleType("google.genai")

    class _Models:
        def generate_content(
            self, *, model: str, contents: Any, config: Any = None
        ) -> Any:
            captured["calls"].append(
                {
                    "kind": "generate_content",
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
            )
            if on_call is not None:
                on_call(captured["calls"][-1])
            if raise_exc is not None:
                raise raise_exc
            parts = [
                _Part(data=img, mime_type=mime)
                for img, mime in (images_to_return or [])
            ]
            cand = pytypes.SimpleNamespace(content=pytypes.SimpleNamespace(parts=parts))
            return pytypes.SimpleNamespace(candidates=[cand])

        def generate_images(
            self, *, model: str, prompt: str, config: Any = None
        ) -> Any:
            captured["calls"].append(
                {
                    "kind": "generate_images",
                    "model": model,
                    "prompt": prompt,
                    "config": config.kwargs if config else None,
                }
            )
            if raise_exc is not None:
                raise raise_exc
            generated = []
            for img, mime in images_to_return or []:
                generated.append(
                    pytypes.SimpleNamespace(
                        image=pytypes.SimpleNamespace(image_bytes=img, mime_type=mime)
                    )
                )
            return pytypes.SimpleNamespace(generated_images=generated)

    class _Client:
        def __init__(self, *, api_key: str | None = None, **_kw: Any) -> None:
            captured["api_key"] = api_key
            self.models = _Models()

    fake_genai.Client = _Client

    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return captured


async def test_generate_image_gemini_backend_emits_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    captured = _install_fake_google_genai(
        monkeypatch,
        images_to_return=[(b"PNG-BYTES", "image/png")],
    )

    src = GeminiImageToolSource(api_key="kx")
    [gen, _edit] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        out = await gen.ainvoke({"prompt": "a tiny ship", "backend": "gemini"})
    assert out["image_count"] == 1
    assert out["byte_sizes"] == [len(b"PNG-BYTES")]
    assert len(out["artifact_uuids"]) == 1
    assert captured["api_key"] == "kx"
    [call] = captured["calls"]
    assert call["kind"] == "generate_content"
    assert call["model"].startswith("gemini-2.5-flash-image")


async def test_generate_image_imagen_backend_forwards_full_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    captured = _install_fake_google_genai(
        monkeypatch,
        images_to_return=[(b"IMG1", "image/png"), (b"IMG2", "image/png")],
    )

    src = GeminiImageToolSource(api_key="kx")
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        out = await gen.ainvoke(
            {
                "prompt": "two cats",
                "backend": "imagen",
                "aspect_ratio": "16:9",
                "number_of_images": 2,
                "negative_prompt": "blurry",
                "seed": 42,
            }
        )
    assert out["image_count"] == 2
    [call] = captured["calls"]
    assert call["kind"] == "generate_images"
    cfg = call["config"]
    assert cfg["aspect_ratio"] == "16:9"
    assert cfg["number_of_images"] == 2
    assert cfg["negative_prompt"] == "blurry"
    assert cfg["seed"] == 42


async def test_generate_image_rejects_unknown_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[(b"x", "image/png")])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx")
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()), pytest.raises(ValueError, match="backend"):
        await gen.ainvoke({"prompt": "x", "backend": "weird"})


async def test_generate_image_rejects_unknown_aspect_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[(b"x", "image/png")])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx")
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()), pytest.raises(ValueError, match="aspect_ratio"):
        await gen.ainvoke({"prompt": "x", "aspect_ratio": "weird"})


async def test_generate_image_raises_when_response_carries_no_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx")
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()), pytest.raises(RuntimeError, match="no image parts"):
        await gen.ainvoke({"prompt": "x"})


async def test_generate_image_imagen_raises_on_zero_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx")
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()), pytest.raises(RuntimeError, match="zero images"):
        await gen.ainvoke({"prompt": "x", "backend": "imagen"})


async def test_edit_image_with_uploaded_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    captured = _install_fake_google_genai(
        monkeypatch,
        images_to_return=[(b"EDITED", "image/png")],
    )

    base_b64 = base64.b64encode(b"ORIGINAL").decode()
    ref = FileRef(name="hat.png", mime="image/png", data_base64=base_b64)

    src = GeminiImageToolSource(api_key="kx")
    [_, edit] = await src.tools(_ctx(files=(ref,)), {})
    with bind(_ctx()):
        out = await edit.ainvoke(
            {"prompt": "add a wizard hat", "base_image_name": "hat.png"}
        )
    assert out["image_count"] == 1
    [call] = captured["calls"]
    assert call["kind"] == "generate_content"
    [content] = call["contents"]
    assert content.role == "user"
    assert content.parts[0].text == "add a wizard hat"
    assert content.parts[1].inline_data.data == b"ORIGINAL"


async def test_edit_image_with_url_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    captured = _install_fake_google_genai(
        monkeypatch,
        images_to_return=[(b"EDITED", "image/png")],
    )

    real_sync = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"FROM-URL", headers={"content-type": "image/jpeg"}
        )

    transport = httpx.MockTransport(handler)

    def _client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.gemini_image.httpx.Client", _client
    )

    src = GeminiImageToolSource(api_key="kx")
    [_, edit] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        await edit.ainvoke(
            {
                "prompt": "make it sepia",
                "base_image_url": "https://x.test/photo.jpg",
                "seed": 7,
            }
        )
    [call] = captured["calls"]
    [content] = call["contents"]
    assert content.parts[1].inline_data.data == b"FROM-URL"
    assert call["config"].kwargs == {"seed": 7}


async def test_edit_image_url_fetch_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[(b"x", "image/png")])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    real_sync = httpx.Client

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    transport = httpx.MockTransport(handler)

    def _client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.gemini_image.httpx.Client", _client
    )

    src = GeminiImageToolSource(api_key="kx")
    [_, edit] = await src.tools(_ctx(), {})
    with (
        bind(_ctx()),
        pytest.raises(RuntimeError, match="could not fetch base_image_url"),
    ):
        await edit.ainvoke(
            {"prompt": "x", "base_image_url": "https://x.test/missing.jpg"}
        )


async def test_edit_image_uploaded_url_only_ref_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    captured = _install_fake_google_genai(
        monkeypatch, images_to_return=[(b"EDITED", "image/png")]
    )

    real_sync = httpx.Client

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("uploaded.png"):
            return httpx.Response(
                200,
                content=b"FROM-UPLOAD-URL",
                headers={"content-type": "image/png"},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    def _client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.gemini_image.httpx.Client", _client
    )

    ref = FileRef(name="img.png", url="https://example.test/uploaded.png")
    src = GeminiImageToolSource(api_key="kx")
    [_, edit] = await src.tools(_ctx(files=(ref,)), {})
    with bind(_ctx()):
        await edit.ainvoke({"prompt": "x", "base_image_name": "img.png"})
    [call] = captured["calls"]
    assert call["contents"][0].parts[1].inline_data.data == b"FROM-UPLOAD-URL"


async def test_edit_image_uploaded_url_only_ref_propagates_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[(b"x", "image/png")])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    real_sync = httpx.Client

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    def _client(**kw: Any) -> httpx.Client:
        kw.pop("transport", None)
        return real_sync(transport=transport, **kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.gemini_image.httpx.Client", _client
    )

    ref = FileRef(name="img.png", url="https://example.test/missing.png")
    src = GeminiImageToolSource(api_key="kx")
    [_, edit] = await src.tools(_ctx(files=(ref,)), {})
    with bind(_ctx()), pytest.raises(RuntimeError, match="could not fetch"):
        await edit.ainvoke({"prompt": "x", "base_image_name": "img.png"})


async def test_edit_image_missing_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[(b"x", "image/png")])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx")
    [_, edit] = await src.tools(_ctx(), {})
    with (
        bind(_ctx()),
        pytest.raises(RuntimeError, match="base_image_name or base_image_url"),
    ):
        await edit.ainvoke({"prompt": "x"})


async def test_edit_image_unknown_uploaded_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_genai(monkeypatch, images_to_return=[(b"x", "image/png")])
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    ref = FileRef(name="other.png", data_base64=base64.b64encode(b"x").decode())
    src = GeminiImageToolSource(api_key="kx")
    [_, edit] = await src.tools(_ctx(files=(ref,)), {})
    with bind(_ctx()), pytest.raises(RuntimeError, match="other.png"):
        await edit.ainvoke({"prompt": "x", "base_image_name": "missing.png"})


async def test_generate_image_retries_then_succeeds_on_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry then succeed on a 429."""
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.gemini_image._backoff", lambda _n: 0.0
    )

    class _ClientError(Exception):
        code = 429
        details = {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "0s",
                    }
                ]
            }
        }

    state = {"calls": 0}

    def _on_call(rec: dict[str, Any]) -> None:
        state["calls"] += 1
        if state["calls"] == 1:
            raise _ClientError("first 429")

    captured = _install_fake_google_genai(
        monkeypatch,
        images_to_return=[(b"OK", "image/png")],
        on_call=_on_call,
    )
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.gemini_image._retryable",
        lambda exc: True,
    )

    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx", max_retries=2)
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        out = await gen.ainvoke({"prompt": "x"})
    assert out["image_count"] == 1
    assert state["calls"] == 2


async def test_generate_image_propagates_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom(Exception):
        code = 400

    _install_fake_google_genai(
        monkeypatch, images_to_return=[], raise_exc=_Boom("invalid")
    )

    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx", max_retries=3)
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()), pytest.raises(_Boom):
        await gen.ainvoke({"prompt": "x"})


async def test_generate_image_eventually_raises_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openbb_agent_server.plugins.tools.gemini_image._backoff", lambda _n: 0.0
    )

    class _Boom(Exception):
        code = 500

    _install_fake_google_genai(
        monkeypatch, images_to_return=[], raise_exc=_Boom("server")
    )

    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx", max_retries=1)
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()), pytest.raises(RuntimeError, match="exhausted 1 retries"):
        await gen.ainvoke({"prompt": "x"})


def test_retryable_handles_status_code_attribute() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _retryable

    class _E(Exception):
        status_code = 503

    assert _retryable(_E()) is True


def test_retryable_handles_named_exceptions() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _retryable

    class ResourceExhausted(Exception):
        pass

    assert _retryable(ResourceExhausted()) is True


def test_retryable_returns_false_for_unknown() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _retryable

    assert _retryable(ValueError("boom")) is False


def test_retryable_handles_httpx_timeout() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _retryable

    assert _retryable(httpx.ConnectTimeout("timeout")) is True


def test_retry_delay_extracts_seconds_from_retry_info() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _retry_delay

    class _E(Exception):
        details = {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "12s",
                    }
                ]
            }
        }

    assert _retry_delay(_E()) == 12.0


def test_retry_delay_returns_none_for_missing_or_malformed() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _retry_delay

    assert _retry_delay(Exception("plain")) is None

    class _E(Exception):
        details = {"error": {"details": [{"@type": "RetryInfo", "retryDelay": "wat"}]}}

    assert _retry_delay(_E()) is None


def test_retry_delay_returns_none_when_details_not_dict() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _retry_delay

    class _E(Exception):
        details = "string-not-dict"

    assert _retry_delay(_E()) is None


def test_backoff_is_capped() -> None:
    from openbb_agent_server.plugins.tools.gemini_image import _backoff

    assert _backoff(1) == 0.5
    assert _backoff(20) == 30.0


def _install_fake_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vectors: list[list[float]] | None = None,
    raise_exc: BaseException | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {"init_kwargs": None}

    class _Embedder:
        def __init__(self, **kwargs: Any) -> None:
            captured["init_kwargs"] = kwargs

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            captured["texts"] = texts
            if raise_exc is not None:
                raise raise_exc
            return vectors or [[0.0] * 8 for _ in texts]

    fake_module = pytypes.ModuleType("langchain_google_genai")
    fake_module.GoogleGenerativeAIEmbeddings = _Embedder
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    return captured


async def test_embed_text_round_trips_through_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_embeddings(monkeypatch, vectors=[[0.1, 0.2], [0.3, 0.4]])
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource(api_key="kx")
    [tool] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        out = await tool.ainvoke(
            {
                "texts": ["hello", "world"],
                "task_type": "RETRIEVAL_QUERY",
                "output_dimensionality": 256,
            }
        )
    assert out["count"] == 2
    assert out["dimensions"] == 2
    assert out["task_type"] == "RETRIEVAL_QUERY"
    assert captured["texts"] == ["hello", "world"]
    assert captured["init_kwargs"]["task_type"] == "RETRIEVAL_QUERY"
    assert captured["init_kwargs"]["output_dimensionality"] == 256
    assert captured["init_kwargs"]["model"].startswith("models/")


async def test_embed_text_empty_list_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_embeddings(monkeypatch, vectors=[])
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource(api_key="kx")
    [tool] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        out = await tool.ainvoke({"texts": []})
    assert out == {"vectors": [], "model": "gemini-embedding-001", "dimensions": 0}


async def test_embed_text_rejects_unknown_task_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_embeddings(monkeypatch, vectors=[[0.1]])
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource(api_key="kx")
    [tool] = await src.tools(_ctx(), {})
    with bind(_ctx()), pytest.raises(ValueError, match="task_type"):
        await tool.ainvoke({"texts": ["x"], "task_type": "WEIRD"})


async def test_embed_text_passes_model_through_unchanged_when_already_prefixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_embeddings(monkeypatch, vectors=[[0.1]])
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource(api_key="kx")
    [tool] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        await tool.ainvoke({"texts": ["x"], "model": "models/gemini-embedding-001"})
    assert captured["init_kwargs"]["model"] == "models/gemini-embedding-001"


async def test_embeddings_per_call_config_overrides_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_embeddings(monkeypatch, vectors=[[0.1]])
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource(
        api_key="kx",
        default_task_type="RETRIEVAL_DOCUMENT",
        default_output_dimensionality=512,
    )
    [tool] = await src.tools(
        _ctx(),
        {
            "default_task_type": "CLASSIFICATION",
            "default_output_dimensionality": 128,
            "default_model": "gemini-embedding-2",
        },
    )
    with bind(_ctx()):
        await tool.ainvoke({"texts": ["x"]})
    assert captured["init_kwargs"]["task_type"] == "CLASSIFICATION"
    assert captured["init_kwargs"]["output_dimensionality"] == 128
    assert captured["init_kwargs"]["model"] == "models/gemini-embedding-2"


async def test_embeddings_picks_api_key_from_per_call_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_embeddings(monkeypatch, vectors=[[0.1]])
    from openbb_agent_server.plugins.tools.gemini_embeddings import (
        GeminiEmbeddingsToolSource,
    )

    src = GeminiEmbeddingsToolSource()
    [tool] = await src.tools(_ctx(), {"api_key": "config-key"})
    with bind(_ctx()):
        await tool.ainvoke({"texts": ["x"]})
    assert captured["init_kwargs"]["google_api_key"] == "config-key"


async def test_image_picks_api_key_from_per_call_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_google_genai(
        monkeypatch, images_to_return=[(b"x", "image/png")]
    )
    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource()
    [gen, _] = await src.tools(_ctx(), {"api_key": "config-key"})
    with bind(_ctx()):
        await gen.ainvoke({"prompt": "x"})
    assert captured["api_key"] == "config-key"


async def test_imagen_skips_generated_entries_with_no_image_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip generated_images entries with no image_bytes."""
    captured: dict[str, Any] = {}

    fake_types = pytypes.ModuleType("google.genai.types")

    class _GenerateImagesConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_types.GenerateImagesConfig = _GenerateImagesConfig

    class _Models:
        def generate_images(
            self, *, model: str, prompt: str, config: Any = None
        ) -> Any:
            captured["called"] = True
            generated = [
                pytypes.SimpleNamespace(
                    image=pytypes.SimpleNamespace(image_bytes=None, mime_type=None)
                ),
                pytypes.SimpleNamespace(
                    image=pytypes.SimpleNamespace(
                        image_bytes=b"REAL", mime_type="image/png"
                    )
                ),
            ]
            return pytypes.SimpleNamespace(generated_images=generated)

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.Client = lambda *, api_key=None, **_kw: pytypes.SimpleNamespace(
        models=_Models()
    )

    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx")
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        out = await gen.ainvoke({"prompt": "x", "backend": "imagen"})
    assert out["image_count"] == 1
    assert out["byte_sizes"] == [len(b"REAL")]


async def test_gemini_skips_parts_without_inline_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip parts without inline_data.data."""
    fake_types = pytypes.ModuleType("google.genai.types")

    class _Part:
        def __init__(self, *, data: Any = None, mime_type: str | None = None) -> None:
            self.inline_data = pytypes.SimpleNamespace(data=data, mime_type=mime_type)

    fake_types.Part = _Part
    fake_types.Content = lambda **kw: pytypes.SimpleNamespace(**kw)
    fake_types.GenerateContentConfig = lambda **kw: pytypes.SimpleNamespace(**kw)

    class _Models:
        def generate_content(
            self, *, model: str, contents: Any, config: Any = None
        ) -> Any:
            parts = [
                _Part(data=None, mime_type=None),
                _Part(data=b"REAL", mime_type="image/png"),
            ]
            cand = pytypes.SimpleNamespace(content=pytypes.SimpleNamespace(parts=parts))
            return pytypes.SimpleNamespace(candidates=[cand])

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.Client = lambda *, api_key=None, **_kw: pytypes.SimpleNamespace(
        models=_Models()
    )
    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    from openbb_agent_server.plugins.tools.gemini_image import GeminiImageToolSource

    src = GeminiImageToolSource(api_key="kx")
    [gen, _] = await src.tools(_ctx(), {})
    with bind(_ctx()):
        out = await gen.ainvoke({"prompt": "x"})
    assert out["image_count"] == 1


def test_retry_delay_skips_non_dict_entries() -> None:
    """Skip a non-dict entry inside error.details."""
    from openbb_agent_server.plugins.tools.gemini_image import _retry_delay

    class _E(Exception):
        details = {
            "error": {
                "details": [
                    "not-a-dict",
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "3s",
                    },
                ]
            }
        }

    assert _retry_delay(_E()) == 3.0


def test_retry_delay_returns_none_on_unparsable_seconds() -> None:
    """Return None on an unparsable retryDelay string."""
    from openbb_agent_server.plugins.tools.gemini_image import _retry_delay

    class _E(Exception):
        details = {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "abcs",
                    }
                ]
            }
        }

    assert _retry_delay(_E()) is None
