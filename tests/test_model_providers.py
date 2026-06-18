"""Model provider construction tests."""

from __future__ import annotations

from typing import Any

import pytest

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(api_keys: dict[str, str] | None = None) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=api_keys or {},
    )


def test_anthropic_provider_builds_with_explicit_key() -> None:
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic import ChatAnthropic

    from openbb_agent_server.plugins.models.anthropic_provider import (
        AnthropicProvider,
    )

    provider = AnthropicProvider(model_name="claude-opus-4-7", api_key="kx")
    model = provider.build(_ctx(), {})
    assert isinstance(model, ChatAnthropic)


def test_anthropic_provider_prefers_runtime_api_key_from_ctx() -> None:
    pytest.importorskip("langchain_anthropic")
    from openbb_agent_server.plugins.models.anthropic_provider import (
        AnthropicProvider,
    )

    provider = AnthropicProvider(api_key="default-key")
    model = provider.build(_ctx(api_keys={"ANTHROPIC_API_KEY": "ctx-key"}), {})
    assert _extract_api_key(model) == "ctx-key"


def test_anthropic_provider_config_overrides_model_name() -> None:
    pytest.importorskip("langchain_anthropic")
    from openbb_agent_server.plugins.models.anthropic_provider import (
        AnthropicProvider,
    )

    provider = AnthropicProvider(api_key="kx")
    model = provider.build(_ctx(), {"model_name": "claude-haiku-4-5"})
    assert (
        getattr(model, "model", None) == "claude-haiku-4-5"
        or getattr(model, "model_name", None) == "claude-haiku-4-5"
    )


def test_openai_provider_constructs_when_package_available() -> None:
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    from openbb_agent_server.plugins.models.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="kx")
    model = provider.build(_ctx(), {})
    assert isinstance(model, ChatOpenAI)


def test_nvidia_provider_constructs_when_package_available() -> None:
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    provider = NvidiaProvider(api_key="kx")
    model = provider.build(_ctx(), {})
    assert isinstance(model, ChatNVIDIA)


def test_nvidia_provider_propagates_extra_body_and_model_kwargs() -> None:
    """extra_body keys merge into model_kwargs cleanly."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")

    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    provider = NvidiaProvider(
        api_key="kx",
        extra_body={"frequency_penalty": 0.1},
        model_kwargs={"reasoning_effort": "high"},
    )
    model = provider.build(_ctx(), {})
    mk = getattr(model, "model_kwargs", {}) or {}
    assert mk.get("reasoning_effort") == "high"
    assert mk.get("frequency_penalty") == 0.1
    assert "extra_body" not in mk


def test_bedrock_provider_constructs_when_package_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_aws")
    from openbb_agent_server.plugins.models.bedrock_provider import BedrockProvider

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    BedrockProvider(region_name="us-east-1").build(_ctx(), {})


def test_vertex_provider_constructs_when_package_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_google_genai")
    from langchain_google_genai import ChatGoogleGenerativeAI

    from openbb_agent_server.plugins.models.vertex_provider import VertexProvider

    model = VertexProvider(project="test-project").build(_ctx(), {})
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.vertexai is True
    assert model.project == "test-project"


def test_google_genai_provider_constructs_when_package_available() -> None:
    pytest.importorskip("langchain_google_genai")
    from langchain_google_genai import ChatGoogleGenerativeAI

    from openbb_agent_server.plugins.models.google_genai_provider import (
        GoogleGenAIProvider,
    )

    provider = GoogleGenAIProvider(api_key="kx")
    model = provider.build(_ctx(), {})
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("gemini-2.5-flash")


def test_google_genai_provider_prefers_runtime_api_key_from_ctx() -> None:
    pytest.importorskip("langchain_google_genai")
    from openbb_agent_server.plugins.models.google_genai_provider import (
        GoogleGenAIProvider,
    )

    provider = GoogleGenAIProvider(api_key="default")
    model = provider.build(_ctx(api_keys={"GOOGLE_API_KEY": "ctx-key"}), {})
    raw = model.google_api_key
    resolved = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
    assert resolved == "ctx-key"


def test_google_genai_provider_accepts_gemini_api_key_alias() -> None:
    pytest.importorskip("langchain_google_genai")
    from openbb_agent_server.plugins.models.google_genai_provider import (
        GoogleGenAIProvider,
    )

    provider = GoogleGenAIProvider()
    model = provider.build(_ctx(api_keys={"GEMINI_API_KEY": "alt"}), {})
    raw = model.google_api_key
    resolved = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
    assert resolved == "alt"


def test_google_genai_provider_config_overrides_model_name() -> None:
    pytest.importorskip("langchain_google_genai")
    from openbb_agent_server.plugins.models.google_genai_provider import (
        GoogleGenAIProvider,
    )

    provider = GoogleGenAIProvider(api_key="kx")
    model = provider.build(_ctx(), {"model_name": "gemini-2.5-pro"})
    assert model.model.endswith("gemini-2.5-pro")


def test_google_genai_provider_forwards_full_sampling_block() -> None:
    pytest.importorskip("langchain_google_genai")
    from openbb_agent_server.plugins.models.google_genai_provider import (
        GoogleGenAIProvider,
    )

    provider = GoogleGenAIProvider(
        api_key="kx",
        temperature=0.7,
        max_output_tokens=1024,
        top_p=0.9,
        top_k=40,
        timeout=45.0,
        max_retries=3,
        seed=42,
        stop=["</done>"],
        safety_settings={},
        base_url="https://generativelanguage.googleapis.example",
        additional_headers={"x-goog-user-project": "abc"},
        cached_content="cachedContents/abc",
        response_mime_type="application/json",
        response_schema={"type": "object"},
        thinking_budget=1024,
        thinking_level="medium",
        include_thoughts=False,
        labels={"team": "research"},
    )
    model = provider.build(_ctx(), {})
    assert model.temperature == 0.7
    assert model.max_output_tokens == 1024
    assert model.top_p == 0.9
    assert model.top_k == 40
    assert model.max_retries == 3
    assert model.seed == 42
    assert model.stop == ["</done>"]
    assert model.timeout == 45.0
    assert model.cached_content == "cachedContents/abc"
    assert model.response_mime_type == "application/json"
    assert model.response_schema == {"type": "object"}
    assert model.thinking_budget == 1024
    assert model.thinking_level == "medium"
    assert model.include_thoughts is False
    assert dict(model.labels or {}).get("team") == "research"
    assert dict(model.additional_headers or {}).get("x-goog-user-project") == "abc"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.5}, "temperature"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_p": -0.01}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_retries": -1}, "max_retries"),
        ({"thinking_level": "extreme"}, "thinking_level"),
    ],
)
def test_google_genai_provider_rejects_out_of_range(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.google_genai_provider import (
        GoogleGenAIProvider,
    )

    with pytest.raises(ValueError, match=match):
        GoogleGenAIProvider(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 1.5}, "temperature"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_anthropic_provider_rejects_out_of_range(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.anthropic_provider import (
        AnthropicProvider,
    )

    with pytest.raises(ValueError, match=match):
        AnthropicProvider(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 2.1}, "temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"presence_penalty": -2.5}, "presence_penalty"),
        ({"frequency_penalty": 2.5}, "frequency_penalty"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"n": 0}, "n"),
        ({"reasoning_effort": "extreme"}, "reasoning_effort"),
    ],
)
def test_openai_provider_rejects_out_of_range(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.openai_provider import OpenAIProvider

    with pytest.raises(ValueError, match=match):
        OpenAIProvider(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 2.5}, "temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"max_completion_tokens": 0}, "max_completion_tokens"),
    ],
)
def test_nvidia_provider_rejects_out_of_range(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    with pytest.raises(ValueError, match=match):
        NvidiaProvider(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 1.5}, "temperature"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_bedrock_provider_rejects_out_of_range(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.bedrock_provider import BedrockProvider

    with pytest.raises(ValueError, match=match):
        BedrockProvider(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 2.5}, "temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_retries": -1}, "max_retries"),
        ({"thinking_budget": -1}, "thinking_budget"),
        ({"thinking_level": "extreme"}, "thinking_level"),
    ],
)
def test_vertex_provider_rejects_out_of_range(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.vertex_provider import VertexProvider

    with pytest.raises(ValueError, match=match):
        VertexProvider(**kwargs)


def test_anthropic_provider_forwards_sampling_block() -> None:
    pytest.importorskip("langchain_anthropic")
    from openbb_agent_server.plugins.models.anthropic_provider import (
        AnthropicProvider,
    )

    provider = AnthropicProvider(
        api_key="kx",
        api_url="https://api.anthropic.example/v1",
        temperature=0.5,
        max_tokens=2048,
        top_p=0.95,
        top_k=20,
        stop_sequences=["</done>"],
        timeout=45.0,
        max_retries=3,
        default_headers={"X-Tenant": "abc"},
        thinking={"type": "enabled", "budget_tokens": 1024},
        betas=["prompt-caching-2024-07-31"],
    )
    model = provider.build(_ctx(), {})
    assert model.temperature == 0.5
    assert model.max_tokens == 2048
    assert model.top_p == 0.95
    assert model.top_k == 20
    assert model.stop_sequences == ["</done>"]
    assert model.max_retries == 3
    assert model.anthropic_api_url == "https://api.anthropic.example/v1"
    assert dict(model.default_headers or {}).get("X-Tenant") == "abc"
    assert model.thinking == {"type": "enabled", "budget_tokens": 1024}
    assert model.betas == ["prompt-caching-2024-07-31"]


def test_openai_provider_forwards_sampling_block() -> None:
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_provider import OpenAIProvider

    provider = OpenAIProvider(
        api_key="kx",
        base_url="https://api.openai.example/v1",
        organization="org-abc",
        temperature=1.2,
        max_tokens=4096,
        top_p=0.85,
        presence_penalty=0.5,
        frequency_penalty=-0.5,
        seed=7,
        n=2,
        stop=["</done>"],
        timeout=30.0,
        reasoning_effort="high",
        default_headers={"X-Tenant": "abc"},
        streaming=False,
    )
    model = provider.build(_ctx(), {})
    assert model.temperature == 1.2
    assert model.max_tokens == 4096
    assert model.top_p == 0.85
    assert model.presence_penalty == 0.5
    assert model.frequency_penalty == -0.5
    assert model.seed == 7
    assert model.n == 2
    assert model.reasoning_effort == "high"
    assert model.openai_api_base == "https://api.openai.example/v1"
    assert model.openai_organization == "org-abc"
    assert model.stop == ["</done>"]
    assert dict(model.default_headers or {}).get("X-Tenant") == "abc"


def test_nvidia_provider_forwards_sampling_block() -> None:
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    provider = NvidiaProvider(
        api_key="kx",
        base_url="https://nim.example/v1",
        temperature=0.3,
        max_completion_tokens=512,
        top_p=0.7,
        seed=11,
        stop=["</done>"],
        default_headers={"X-Tenant": "abc"},
    )
    model = provider.build(_ctx(), {})
    assert model.temperature == 0.3
    assert model.top_p == 0.7
    assert model.seed == 11
    assert model.stop == ["</done>"]
    assert model.base_url == "https://nim.example/v1"
    assert dict(model.default_headers).get("X-Tenant") == "abc"


def test_nvidia_provider_forwards_reasoning_fields() -> None:
    """Reasoning fields piggy-back on model_kwargs for the NIM server."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    provider = NvidiaProvider(
        api_key="kx",
        reasoning_effort="high",
        reasoning_budget=16384,
        chat_template_kwargs={"reasoning_budget": 16384},
    )
    model = provider.build(_ctx(), {})
    mk = dict(getattr(model, "model_kwargs", {}) or {})
    assert mk.get("reasoning_effort") == "high"
    assert mk.get("reasoning_budget") == 16384
    assert mk.get("chat_template_kwargs") == {"reasoning_budget": 16384}


def test_nvidia_provider_reasoning_budget_disabled_sentinel() -> None:
    """reasoning_budget=-1 is the NIM no-enforcement sentinel."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    provider = NvidiaProvider(api_key="kx", reasoning_budget=-1)
    model = provider.build(_ctx(), {})
    mk = dict(getattr(model, "model_kwargs", {}) or {})
    assert mk.get("reasoning_budget") == -1


def test_nvidia_provider_streaming_toggle_maps_to_disable_streaming() -> None:
    """streaming=False translates to ChatNVIDIA's disable_streaming=True."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    on = NvidiaProvider(api_key="kx", streaming=True).build(_ctx(), {})
    off = NvidiaProvider(api_key="kx", streaming=False).build(_ctx(), {})
    assert on.disable_streaming is False
    assert off.disable_streaming is True


def test_nvidia_provider_max_tokens_alias_to_max_completion_tokens() -> None:
    """max_tokens wins over max_completion_tokens when both are set."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    a = NvidiaProvider(api_key="kx", max_completion_tokens=512).build(_ctx(), {})
    b = NvidiaProvider(api_key="kx", max_tokens=1024).build(_ctx(), {})
    c = NvidiaProvider(api_key="kx", max_tokens=2048, max_completion_tokens=512).build(
        _ctx(), {}
    )
    assert a.max_tokens == 512
    assert b.max_tokens == 1024
    assert c.max_tokens == 2048


def test_nvidia_provider_rejects_invalid_reasoning_effort() -> None:
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    with pytest.raises(ValueError, match="reasoning_effort"):
        NvidiaProvider(api_key="k", reasoning_effort="extreme")


def test_nvidia_provider_rejects_invalid_reasoning_budget() -> None:
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    with pytest.raises(ValueError, match="reasoning_budget"):
        NvidiaProvider(api_key="k", reasoning_budget=-5)


def test_nvidia_provider_forwards_penalty_fields() -> None:
    """Penalty fields ride along in model_kwargs."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    provider = NvidiaProvider(
        api_key="kx", frequency_penalty=0.5, presence_penalty=-0.3
    )
    model = provider.build(_ctx(), {})
    mk = dict(getattr(model, "model_kwargs", {}) or {})
    assert mk.get("frequency_penalty") == 0.5
    assert mk.get("presence_penalty") == -0.3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frequency_penalty": 2.5}, "frequency_penalty"),
        ({"frequency_penalty": -2.5}, "frequency_penalty"),
        ({"presence_penalty": 2.5}, "presence_penalty"),
        ({"presence_penalty": -2.5}, "presence_penalty"),
    ],
)
def test_nvidia_provider_rejects_out_of_range_penalty(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    with pytest.raises(ValueError, match=match):
        NvidiaProvider(api_key="k", **kwargs)


def test_openai_compat_provider_forwards_reasoning_fields() -> None:
    """openai_compat forwards reasoning fields to compat servers."""
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="https://nim.example/v1",
        model_name="meta/llama-3.3-70b-instruct",
        api_key="kx",
        reasoning_effort="high",
        reasoning_budget=16384,
        chat_template_kwargs={"reasoning_budget": 16384},
    )
    model = provider.build(_ctx(), {})
    mk = dict(getattr(model, "model_kwargs", {}) or {})
    assert mk.get("reasoning_budget") == 16384
    assert mk.get("chat_template_kwargs") == {"reasoning_budget": 16384}
    assert getattr(model, "reasoning_effort", None) == "high"


def test_openai_compat_provider_rejects_invalid_reasoning_fields() -> None:
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    with pytest.raises(ValueError, match="reasoning_effort"):
        OpenAICompatProvider(
            base_url="https://x/v1",
            model_name="m",
            reasoning_effort="extreme",
        )
    with pytest.raises(ValueError, match="reasoning_budget"):
        OpenAICompatProvider(
            base_url="https://x/v1",
            model_name="m",
            reasoning_budget=-7,
        )


def test_nvidia_provider_suppresses_not_known_to_support_tools_warning(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """The provider silences the misleading not-known-to-support-tools warning."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from langchain_core.tools import tool

    from openbb_agent_server.plugins.models.nvidia_provider import NvidiaProvider

    @tool
    def echo(x: str) -> str:
        """Echo back."""
        return x

    provider = NvidiaProvider(
        api_key="k", model_name="mistralai/mistral-large-3-675b-instruct-2512"
    )
    model = provider.build(_ctx(), {})
    model.bind_tools([echo])
    messages = [str(w.message) for w in recwarn.list]
    assert not any("is not known to support tools" in m for m in messages), (
        f"warning leaked: {messages}"
    )


def test_nvidia_silence_unknown_tools_warning_swallows_attribute_errors() -> None:
    """_silence_unknown_tools_warning is best-effort and swallows errors."""
    from openbb_agent_server.plugins.models.nvidia_provider import (
        _silence_unknown_tools_warning,
    )

    class _ExplodingClient:
        @property
        def model(self) -> Any:
            raise RuntimeError("ChatNVIDIA internal shape changed")

    class _Model:
        _client = _ExplodingClient()

    _silence_unknown_tools_warning(_Model())


def test_openai_compat_provider_merges_extra_body_and_model_kwargs() -> None:
    """extra_body and model_kwargs both fold into model_kwargs."""
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://localhost:8000/v1",
        model_name="llama3.1",
        extra_body={"guided_json": {"type": "object"}, "shared": "from_extra"},
        model_kwargs={"guided_regex": "[0-9]+", "shared": "from_model_kwargs"},
    )
    model = provider.build(_ctx(), {})
    mk = dict(getattr(model, "model_kwargs", {}) or {})
    assert mk["guided_json"] == {"type": "object"}
    assert mk["guided_regex"] == "[0-9]+"
    assert mk["shared"] == "from_model_kwargs"


def test_bedrock_provider_routes_top_p_top_k_through_model_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_aws")
    from openbb_agent_server.plugins.models.bedrock_provider import BedrockProvider

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    provider = BedrockProvider(
        region_name="us-east-1",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        provider="anthropic",
        temperature=0.4,
        max_tokens=1024,
        top_p=0.9,
        top_k=50,
        stop_sequences=["</done>"],
        timeout=60,
        max_retries=4,
        guardrails={"guardrailIdentifier": "g-1", "guardrailVersion": "1"},
    )
    model = provider.build(_ctx(), {})
    assert model.temperature == 0.4
    assert model.max_tokens == 1024
    assert model.max_retries == 4
    assert model.model_kwargs == {
        "top_p": 0.9,
        "top_k": 50,
        "stop_sequences": ["</done>"],
    }
    assert model.endpoint_url == "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert model.provider == "anthropic"
    assert model.timeout == 60
    assert dict(model.guardrails or {}).get("guardrailIdentifier") == "g-1"


def test_bedrock_provider_credentials_profile_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """credentials_profile_name lands in kwargs without requiring AWS auth."""
    pytest.importorskip("langchain_aws")
    captured: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr(
        "openbb_agent_server.plugins.models.bedrock_provider.__import__",
        __import__,
        raising=False,
    )
    monkeypatch.setattr("langchain_aws.ChatBedrock", _Recorder, raising=False)

    from openbb_agent_server.plugins.models.bedrock_provider import BedrockProvider

    provider = BedrockProvider(
        region_name="us-east-1",
        credentials_profile_name="my-profile",
    )
    provider.build(_ctx(), {})
    assert captured.get("credentials_profile_name") == "my-profile"


def test_openai_compat_provider_requires_base_url_and_model() -> None:
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatProvider(base_url="", model_name="x")
    with pytest.raises(ValueError, match="model_name"):
        OpenAICompatProvider(base_url="http://x", model_name="")


def test_openai_compat_provider_builds_against_explicit_base_url() -> None:
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://localhost:11434/v1",
        model_name="llama3.1",
    )
    model = provider.build(_ctx(), {})
    assert isinstance(model, ChatOpenAI)
    base = getattr(model, "openai_api_base", None)
    assert base == "http://localhost:11434/v1"


def test_openai_compat_provider_uses_placeholder_key_when_unset() -> None:
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://localhost:8000/v1", model_name="llama3.1"
    )
    model = provider.build(_ctx(), {})
    raw = model.openai_api_key
    resolved = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
    assert resolved == "EMPTY"


def test_openai_compat_provider_prefers_compat_key_over_openai_key() -> None:
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://localhost:8000/v1", model_name="llama3.1"
    )
    model = provider.build(
        _ctx(
            api_keys={
                "OPENAI_COMPAT_API_KEY": "compat-key",
                "OPENAI_API_KEY": "openai-key",
            }
        ),
        {},
    )
    raw = model.openai_api_key
    resolved = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
    assert resolved == "compat-key"


def test_openai_compat_provider_falls_back_to_openai_key() -> None:
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://localhost:8000/v1", model_name="llama3.1"
    )
    model = provider.build(_ctx(api_keys={"OPENAI_API_KEY": "fallback"}), {})
    raw = model.openai_api_key
    resolved = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
    assert resolved == "fallback"


def test_openai_compat_provider_per_call_base_url_and_model_override() -> None:
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://default:8000/v1", model_name="default-model"
    )
    model = provider.build(
        _ctx(),
        {"base_url": "http://override:9000/v1", "model_name": "override-model"},
    )
    assert model.openai_api_base == "http://override:9000/v1"
    assert model.model_name == "override-model"


def test_openai_compat_provider_forwards_full_sampling_block() -> None:
    pytest.importorskip("langchain_openai")
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    provider = OpenAICompatProvider(
        base_url="http://localhost:8000/v1",
        model_name="llama3.1",
        organization="org-abc",
        temperature=0.4,
        max_tokens=2048,
        top_p=0.85,
        presence_penalty=0.1,
        frequency_penalty=-0.1,
        seed=99,
        stop=["</done>"],
        timeout=45.0,
        max_retries=4,
        default_headers={"X-Tenant": "abc"},
        default_query={"trace": "1"},
    )
    model = provider.build(_ctx(), {})
    assert model.temperature == 0.4
    assert model.max_tokens == 2048
    assert model.top_p == 0.85
    assert model.presence_penalty == 0.1
    assert model.frequency_penalty == -0.1
    assert model.seed == 99
    assert model.stop == ["</done>"]
    assert model.max_retries == 4
    assert dict(model.default_headers or {}).get("X-Tenant") == "abc"
    assert dict(model.default_query or {}).get("trace") == "1"
    assert model.openai_organization == "org-abc"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 2.5}, "temperature"),
        ({"top_p": 1.1}, "top_p"),
        ({"presence_penalty": -2.5}, "presence_penalty"),
        ({"frequency_penalty": 2.5}, "frequency_penalty"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"n": 0}, "n"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_openai_compat_provider_rejects_out_of_range(
    kwargs: dict[str, Any], match: str
) -> None:
    from openbb_agent_server.plugins.models.openai_compat_provider import (
        OpenAICompatProvider,
    )

    with pytest.raises(ValueError, match=match):
        OpenAICompatProvider(
            base_url="http://localhost:8000/v1",
            model_name="llama3.1",
            **kwargs,
        )


def test_groq_provider_constructs_with_default_rate_limiter() -> None:
    pytest.importorskip("langchain_groq")
    from langchain_groq import ChatGroq

    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import (
        GroqRateLimiter,
        reset_cache,
    )

    reset_cache()
    provider = GroqProvider(api_key="kx")
    model = provider.build(_ctx(), {})
    assert isinstance(model, ChatGroq)
    assert isinstance(model.rate_limiter, GroqRateLimiter)
    assert any(
        type(cb).__name__ == "_GroqUsageHandler" for cb in (model.callbacks or [])
    )


def test_groq_provider_prefers_runtime_api_key_from_ctx() -> None:
    pytest.importorskip("langchain_groq")
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import reset_cache

    reset_cache()
    provider = GroqProvider(api_key="default")
    model = provider.build(_ctx(api_keys={"GROQ_API_KEY": "ctx-key"}), {})
    raw = model.groq_api_key
    resolved = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
    assert resolved == "ctx-key"


def test_groq_provider_picks_per_model_limits_from_published_table() -> None:
    pytest.importorskip("langchain_groq")
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import (
        GROQ_LIMITS,
        reset_cache,
    )

    reset_cache()
    provider = GroqProvider(model_name="moonshotai/kimi-k2-instruct", api_key="kx")
    model = provider.build(_ctx(), {})
    snap = model.rate_limiter.snapshot()
    limits = GROQ_LIMITS["moonshotai/kimi-k2-instruct"]
    assert snap["rpm_remaining"] == pytest.approx(limits.rpm, abs=0.5)
    assert snap["rpd_remaining"] == pytest.approx(limits.rpd, abs=0.5)
    assert snap["tpm_remaining"] == pytest.approx(limits.tpm, abs=0.5)
    assert snap["tpd_remaining"] == pytest.approx(limits.tpd, abs=0.5)


def test_groq_provider_shares_limiter_across_calls_with_same_api_key_and_model() -> (
    None
):
    pytest.importorskip("langchain_groq")
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import reset_cache

    reset_cache()
    p1 = GroqProvider(model_name="llama-3.1-8b-instant", api_key="shared")
    p2 = GroqProvider(model_name="llama-3.1-8b-instant", api_key="shared")
    m1 = p1.build(_ctx(), {})
    m2 = p2.build(_ctx(), {})
    assert m1.rate_limiter is m2.rate_limiter


def test_groq_provider_isolates_limiter_across_api_keys() -> None:
    pytest.importorskip("langchain_groq")
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import reset_cache

    reset_cache()
    p1 = GroqProvider(model_name="llama-3.1-8b-instant", api_key="org-a")
    p2 = GroqProvider(model_name="llama-3.1-8b-instant", api_key="org-b")
    m1 = p1.build(_ctx(), {})
    m2 = p2.build(_ctx(), {})
    assert m1.rate_limiter is not m2.rate_limiter


def test_groq_provider_accepts_explicit_rate_limit_override() -> None:
    pytest.importorskip("langchain_groq")
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import (
        GroqRateLimiter,
        reset_cache,
    )

    reset_cache()
    custom = GroqRateLimiter(rpm=5, rpd=None, tpm=None, tpd=None)
    provider = GroqProvider(api_key="kx", rate_limit=custom)
    model = provider.build(_ctx(), {})
    assert model.rate_limiter is custom


def test_groq_provider_record_rate_limit_table_overrides_known_model() -> None:
    pytest.importorskip("langchain_groq")
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import (
        GROQ_LIMITS,
        GroqLimits,
        reset_cache,
    )

    original = GROQ_LIMITS.get("test-org-only-model")
    reset_cache()
    GroqProvider(
        model_name="test-org-only-model",
        api_key="kx",
        record_rate_limit_table={
            "test-org-only-model": GroqLimits(rpm=10, rpd=100, tpm=1000, tpd=10_000),
        },
    )
    assert GROQ_LIMITS["test-org-only-model"].rpm == 10
    assert GROQ_LIMITS["test-org-only-model"].tpd == 10_000

    GroqProvider(
        model_name="test-org-only-model-dict",
        api_key="kx",
        record_rate_limit_table={
            "test-org-only-model-dict": {"rpm": 5, "rpd": 50},
        },
    )
    assert GROQ_LIMITS["test-org-only-model-dict"].rpm == 5
    assert GROQ_LIMITS["test-org-only-model-dict"].rpd == 50

    if original is None:
        del GROQ_LIMITS["test-org-only-model"]
    else:
        GROQ_LIMITS["test-org-only-model"] = original
    GROQ_LIMITS.pop("test-org-only-model-dict", None)


def test_groq_provider_forwards_full_sampling_block() -> None:
    pytest.importorskip("langchain_groq")
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider
    from openbb_agent_server.plugins.models.groq_rate_limiter import reset_cache

    reset_cache()
    provider = GroqProvider(
        api_key="kx",
        base_url="https://api.groq.example/openai/v1",
        temperature=0.6,
        max_tokens=1024,
        top_p=0.9,
        max_retries=7,
        n=1,
        stop=["</done>"],
        timeout=30.0,
        reasoning_effort="medium",
        reasoning_format="parsed",
        service_tier="flex",
        default_headers={"X-Tenant": "abc"},
    )
    model = provider.build(_ctx(), {})
    assert model.temperature == 0.6
    assert model.max_tokens == 1024
    assert model.model_kwargs.get("top_p") == 0.9
    assert model.max_retries == 7
    assert model.stop == ["</done>"]
    assert model.reasoning_effort == "medium"
    assert model.reasoning_format == "parsed"
    assert model.service_tier == "flex"
    assert model.groq_api_base == "https://api.groq.example/openai/v1"
    assert dict(model.default_headers or {}).get("X-Tenant") == "abc"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 2.5}, "temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_retries": -1}, "max_retries"),
        ({"n": 0}, "n"),
        ({"reasoning_effort": "extreme"}, "reasoning_effort"),
        ({"reasoning_format": "spicy"}, "reasoning_format"),
        ({"service_tier": "platinum"}, "service_tier"),
    ],
)
def test_groq_provider_rejects_out_of_range(kwargs: dict[str, Any], match: str) -> None:
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider

    with pytest.raises(ValueError, match=match):
        GroqProvider(**kwargs)


def test_groq_provider_rejects_malformed_record_rate_limit_table() -> None:
    from openbb_agent_server.plugins.models.groq_provider import GroqProvider

    with pytest.raises(ValueError, match="record_rate_limit_table"):
        GroqProvider(record_rate_limit_table={"bad-shape": (1, 2, 3)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="record_rate_limit_table"):
        GroqProvider(record_rate_limit_table={"bad-shape": "not-a-mapping"})  # type: ignore[arg-type]


def test_vertex_provider_forwards_sampling_block() -> None:
    pytest.importorskip("langchain_google_genai")
    from openbb_agent_server.plugins.models.vertex_provider import VertexProvider

    provider = VertexProvider(
        project="test-project",
        location="us-central1",
        temperature=0.6,
        max_output_tokens=2048,
        top_p=0.9,
        top_k=40,
        seed=42,
        stop=["</done>"],
        max_retries=4,
        timeout=45.0,
        safety_settings={},
        response_mime_type="application/json",
        response_schema={"type": "object"},
        thinking_budget=1024,
        thinking_level="medium",
        include_thoughts=True,
        cached_content="projects/p/cachedContents/abc",
        labels={"team": "research"},
        additional_headers={"X-Goog-User-Project": "abc"},
    )
    model = provider.build(_ctx(), {})
    assert model.vertexai is True
    assert model.project == "test-project"
    assert model.location == "us-central1"
    assert model.temperature == 0.6
    assert model.max_output_tokens == 2048
    assert model.top_p == 0.9
    assert model.top_k == 40
    assert model.seed == 42
    assert model.max_retries == 4
    assert model.stop == ["</done>"]
    assert model.timeout == 45.0
    assert model.response_mime_type == "application/json"
    assert model.response_schema == {"type": "object"}
    assert model.thinking_budget == 1024
    assert model.thinking_level == "medium"
    assert model.include_thoughts is True
    assert model.cached_content == "projects/p/cachedContents/abc"
    assert dict(model.labels or {}).get("team") == "research"
    assert dict(model.additional_headers or {}).get("X-Goog-User-Project") == "abc"


def test_fake_provider_ignores_unknown_model_name_kwarg() -> None:
    from openbb_agent_server.plugins.models.fake_provider import FakeProvider

    provider = FakeProvider(model_name="anything", responses=["x"])
    out = provider.build(_ctx(), {})
    assert out is not None


def _extract_api_key(model: Any) -> str | None:
    """Extract the API key from a ChatAnthropic model."""
    raw = getattr(model, "anthropic_api_key", None) or getattr(model, "api_key", None)
    if raw is None:
        return None
    return raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)


def test_fake_provider_yields_string_messages_as_ai_chunks() -> None:
    """Yield string responses as AIMessageChunks."""
    from openbb_agent_server.plugins.models.fake_provider import FakeProvider

    provider = FakeProvider(responses=["plain string"])
    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )
    model = provider.build(ctx, {})
    out = model.invoke("hello")
    assert "plain string" in out.content


@pytest.mark.asyncio
async def test_fake_provider_streams_string_chunk_via_astream() -> None:
    from openbb_agent_server.plugins.models.fake_provider import FakeProvider

    provider = FakeProvider(responses=["streamed"])
    model = provider.build(
        RunContext(
            principal=UserPrincipal(user_id="u"),
            trace_id="t",
            run_id="r",
            conversation_id="c",
        ),
        {},
    )
    chunks = []
    async for chunk in model.astream("hi"):
        chunks.append(chunk.content)
    assert "".join(chunks) == "streamed"


def test_fake_provider_string_message_yields_chunk_via_stream() -> None:
    """Yield a string message as a chunk via stream."""
    from openbb_agent_server.plugins.models.fake_provider import (
        _ToolAwareFakeChatModel,
    )

    model = _ToolAwareFakeChatModel(messages=iter(["raw-string"]))
    chunks = list(model.stream("hi"))
    assert any("raw-string" in str(c.content) for c in chunks)


def test_vertex_provider_forwards_credentials_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_google_genai")
    captured: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _Recorder)

    from openbb_agent_server.plugins.models.vertex_provider import VertexProvider

    sentinel = object()
    VertexProvider(project="p", credentials=sentinel).build(
        RunContext(
            principal=UserPrincipal(user_id="u"),
            trace_id="t",
            run_id="r",
            conversation_id="c",
        ),
        {},
    )
    assert captured.get("credentials") is sentinel
