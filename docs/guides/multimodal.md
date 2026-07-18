# Multimodal

This guide explains the multimodal tool sources that are currently available, their exact tool names, and when background jobs are supported.

Uploads arrive in `QueryRequest.uploaded_files` as `FileRef` objects and are exposed to tools through `RunContext.uploaded_files`.

## What Is Available

| Tool source | Main modality | Tool names | Async `submit_*` support | Required key |
| --- | --- | --- | --- | --- |
| `pdf_extract` | PDF | `list_pdfs`, `get_pdf_outline`, `search_pdf`, `pdf_extract` | no | none |
| `vision_qa` | image understanding | `list_images`, `understand_image`, `submit_understand_image` | yes | `NVIDIA_API_KEY` |
| `paligemma_vision` | image caption/OCR/QA | `list_images`, `caption_image`, `read_image_text`, `ask_about_image`, `submit_caption_image`, `submit_read_image_text`, `submit_ask_about_image` | yes | `NVIDIA_API_KEY` |
| `nemotron_ocr` | OCR | `list_ocr_images`, `extract_text_from_image`, `submit_extract_text_from_image` | yes | `NVIDIA_API_KEY` |
| `gemma_audio` | transcription | `list_audio`, `transcribe_audio`, `submit_transcribe_audio` | yes | `NVIDIA_API_KEY` |
| `groq_audio` | transcription/translation | `transcribe_audio`, `translate_audio` | no | `GROQ_API_KEY` |
| `gemini_image` | image generation/edit | `generate_image`, `edit_image` | no | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

Important: not every multimodal source has `submit_*` tools.

## How To Configure

Enable only the sources you need in profile config:

```toml
[agent]
tool_sources = [
  "pdf_extract",
  "vision_qa",
  "gemma_audio",
  "background_jobs",
]
```

If a required API key is missing, many multimodal sources soft-skip registration, so the tools simply do not appear to the agent.

## Typical Patterns

### PDF extraction

- Use `pdf_extract` for text + layout-aware extraction.
- Use `search_pdf` when you need targeted retrieval first.

### Image understanding

- Use `vision_qa` for general image Q/A.
- Use `paligemma_vision` or `nemotron_ocr` when OCR-focused workflows are needed.

### Audio

- Use `gemma_audio` for transcription with optional async fan-out.
- Use `groq_audio` for direct transcribe/translate calls (sync-only tools).

### Image generation/edit

- Use `gemini_image` for create/edit image workflows.

## Background Jobs

For tools that provide `submit_*` variants, include `background_jobs` and use:

- `list_background_jobs`
- `check_job(job_id)`
- `wait_for_job(job_id, timeout_s)`
- `cancel_job(job_id)`

See [Background jobs](background-jobs.md).

## Troubleshooting

- Tool missing from the model's tool list:
  - check the tool source is enabled in `tool_sources`.
  - check the required API key is present.
- `submit_*` expected but missing:
  - some sources are sync-only (`groq_audio`, `gemini_image`, `pdf_extract`).
- Upload not found by name:
  - verify the file name in `uploaded_files` matches what the tool call uses.

## References

- [pdf_extract](../reference/plugins/tools/pdf_extract.md)
- [vision_qa](../reference/plugins/tools/vision_qa.md)
- [paligemma_vision](../reference/plugins/tools/paligemma_vision.md)
- [nemotron_ocr](../reference/plugins/tools/nemotron_ocr.md)
- [gemma_audio](../reference/plugins/tools/gemma_audio.md)
- [groq_audio](../reference/plugins/tools/groq_audio.md)
- [gemini_image](../reference/plugins/tools/gemini_image.md)
- [background_jobs](../reference/plugins/tools/background_jobs.md)
