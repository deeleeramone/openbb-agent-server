---
name: subagent-delegation
description: Decide when and how to delegate work to specialist subagents and model-profile subagents.
---

# Subagent Delegation

The orchestrator can hand off specialized work via `task(subagent_name,
instruction)`. Use delegation when a task has a clear specialist owner or
when parallel work would improve latency/quality.

## Specialist subagents

- `researcher` — multi-hop web and document research; returns cited notes.
- `analyst` — quantitative analysis, ratios, and data-frame operations.
- `charter` — plotting, chart design, and visualization artifacts.
- `pdf_reader` — deep reading and extraction from uploaded PDFs.

## Model-profile subagents

- `deepseek-v4-flash` — fast multilingual reasoning.
- `deepseek-v4-pro` — high-accuracy reasoning and coding.
- `nemotron-3-super` — agentic reasoning and tool use.
- `nemotron-ultra` — largest Nemotron model for complex analysis and coding.
- `nemotron-3-ultra` — frontier long-context reasoning (Nemotron-3 Ultra 550B).
- `nemotron-3-nano` — low-latency text tasks, summaries, quick answers.
- `mistral-small-4` — multimodal reasoning over images + text.
- `llama-4-maverick` — vision and long-context understanding.
- `gemma-4` — lightweight multimodal OCR and visual Q&A.
- `gpt-oss-120b` — tool-heavy structured output tasks.
- `glm-5.1` — long-context text synthesis.
- `qwen3.5` — coding and SQL generation.
- `minimax-m3` — multimodal vision-language tasks.
- `step-3.7-flash` — fast multimodal vision-language work with native tool use.

## Delegation rules

1. One specialist per distinct sub-task; do not nest more than two levels.
2. Include all context the subagent needs in the instruction (widget IDs,
   PDF names, question, desired output format).
3. If a subagent fails, switch to a different profile or do the work
   yourself instead of retrying identically.
4. Always integrate the subagent result into the final answer and cite
   sources.
