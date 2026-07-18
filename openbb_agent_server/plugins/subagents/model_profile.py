"""Subagent specs that delegate to other configured agent profiles."""

from __future__ import annotations

from typing import Any


def _make_system_prompt(profile_name: str, description: str) -> str:
    return f"""\
You are the {profile_name} specialist subagent. {description}

You have access to the same tool set as the orchestrator. Answer the
delegated task directly, using tools when needed, and always cite
sources when you retrieve facts from PDFs or the web.
"""


class _ModelProfileSubAgent:
    """Subagent declaration that resolves to a configured profile."""

    def __init__(
        self,
        profile_name: str,
        description: str,
        tools: tuple[str, ...] = (),
    ) -> None:
        self.name = profile_name
        self.description = description
        self.system_prompt = _make_system_prompt(profile_name, description)
        self.tools = tools
        self.model: str | None = None
        self.model_profile = profile_name


class ModelProfileSubAgent:
    """Factory for subagents backed by another agent profile.

    This lets the main orchestrator hand work to any configured
    profile (``deepseek-v4-pro``, ``minimax-m3``, ``nemotron-3-ultra``,
    ...) as if it were a named subagent. The runtime resolves the
    profile name into a model identifier at build time.

    The entry point target is the class itself, so the registry passes
    no arguments when instantiating it. The resulting callable is then
    invoked with the subagent config, returning the actual spec object.
    """

    def __init__(self, **_config: Any) -> None:
        self._profile_name = ""
        self._description = ""
        self._tools: tuple[str, ...] = ()

    def with_profile(
        self,
        profile_name: str,
        description: str,
        tools: tuple[str, ...] = (),
    ) -> ModelProfileSubAgent:
        self._profile_name = profile_name
        self._description = description
        self._tools = tools
        return self

    def __call__(self, **_config: Any) -> _ModelProfileSubAgent:
        return _ModelProfileSubAgent(
            profile_name=self._profile_name,
            description=self._description,
            tools=self._tools,
        )


_deepseek_v4_flash_subagent = ModelProfileSubAgent().with_profile(
    profile_name="deepseek-v4-flash",
    description=(
        "Fast multilingual reasoning specialist powered by DeepSeek v4 Flash."
    ),
)

_deepseek_v4_pro_subagent = ModelProfileSubAgent().with_profile(
    profile_name="deepseek-v4-pro",
    description=(
        "High-accuracy reasoning and coding specialist powered by DeepSeek v4 Pro."
    ),
)

_nemotron_3_super_subagent = ModelProfileSubAgent().with_profile(
    profile_name="nemotron-3-super",
    description=(
        "Agentic reasoning and tool-use specialist backed by Nemotron 3 Super."
    ),
)

_nemotron_ultra_subagent = ModelProfileSubAgent().with_profile(
    profile_name="nemotron-ultra",
    description=(
        "Largest Nemotron reasoning specialist for complex analysis and coding."
    ),
)

_nemotron_3_nano_subagent = ModelProfileSubAgent().with_profile(
    profile_name="nemotron-3-nano",
    description=(
        "Fast, lightweight text reasoning specialist for summaries and quick tasks."
    ),
)

_mistral_small_4_subagent = ModelProfileSubAgent().with_profile(
    profile_name="mistral-small-4",
    description=(
        "Multimodal text/image reasoning specialist with configurable reasoning effort."
    ),
)

_llama_4_maverick_subagent = ModelProfileSubAgent().with_profile(
    profile_name="llama-4-maverick",
    description=(
        "Long-context multimodal specialist for vision, dashboards, and image research."
    ),
)

_gemma_4_subagent = ModelProfileSubAgent().with_profile(
    profile_name="gemma-4",
    description=(
        "Lightweight multimodal specialist for chart OCR, visual Q&A, and short video."
    ),
)

_gpt_oss_120b_subagent = ModelProfileSubAgent().with_profile(
    profile_name="gpt-oss-120b",
    description=(
        "Tool and structured-output specialist powered by OpenAI GPT-OSS 120B."
    ),
)

_glm_5_1_subagent = ModelProfileSubAgent().with_profile(
    profile_name="glm-5.1",
    description=(
        "Long-context text synthesis specialist powered by GLM 5.1."
    ),
)

_qwen3_5_subagent = ModelProfileSubAgent().with_profile(
    profile_name="qwen3.5",
    description=(
        "Code and SQL generation specialist powered by Qwen3.5."
    ),
)

_minimax_m3_subagent = ModelProfileSubAgent().with_profile(
    profile_name="minimax-m3",
    description=(
        "Multimodal vision-language specialist powered by MiniMax M3."
    ),
)

_nemotron_3_ultra_subagent = ModelProfileSubAgent().with_profile(
    profile_name="nemotron-3-ultra",
    description=(
        "Frontier long-context reasoning specialist backed by "
        "NVIDIA Nemotron-3 Ultra 550B."
    ),
)

_step_3_7_flash_subagent = ModelProfileSubAgent().with_profile(
    profile_name="step-3.7-flash",
    description=(
        "Fast multimodal vision-language specialist with native tool "
        "use, powered by StepFun Step-3.7-Flash."
    ),
)


