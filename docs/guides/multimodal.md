# Multimodal tools

PDFs, images, audio, and video clips arrive in `QueryRequest.uploaded_files` as `FileRef` records (`name`, `mime`, `data_base64` or `url`). The agent reads them through purpose-built tool sources — every multimodal tool source registers a foreground variant and a `submit_*` background variant so long calls don't block the chat loop.

## Files in `RunContext`

`RunContext.uploaded_files: tuple[FileRef, ...]` holds every upload for the current turn. Each tool source filters them by MIME / extension:

| Modality | Filter | Tool source |
| --- | --- | --- |
| PDFs | `.pdf` / `application/pdf` | `pdf_extract`, `pdf_reader` sub-agent |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tiff`, `.svg` / `image/*` | `vision_qa`, `paligemma_vision`, `gemini_image` |
| Audio | `.mp3`, `.m4a`, `.aac`, `.wav`, `.flac`, `.ogg`, `.opus`, `.webm`, `.mp4`, `.mov`, `.mkv` / `audio/*` / `video/*` | `gemma_audio`, `groq_audio` |
| Text / spreadsheet | `.csv`, `.tsv`, `.json`, `.yaml`, `.md`, `.txt`, `.html` | `ingest_request_context` decodes via `langchain_community.document_loaders` |

## PDFs

`pdf_extract` (`plugins/tools/pdf_extract.py`) opens the file with `pdfplumber` and returns per-page text + per-word bounding boxes. The bounding boxes feed the citation pipeline so the agent can attach `CitationHighlightBoundingBox` to its claims — Workspace renders inline highlights on the source PDF.

For pure text extraction without bounding boxes, the `pdf_reader` sub-agent wraps the same tool with a focused prompt.

## Images

Three vision back-ends, picked by the operator:

### `vision_qa`
NVIDIA NIM chat model (`nvidia/llama-3.1-nemotron-nano-vl-8b-v1` by default). Strongest for chart reading, OCR of structured tables, and general VQA. Foreground tool: `understand_image(name|url, instruction, max_output_tokens)`. Background variant: `submit_understand_image` — returns `{job_id, label}` immediately so the agent can fan out parallel requests.

### `paligemma_vision`
Google PaliGemma via NVIDIA's VLM endpoint. Three task-prefix-driven tools:

- `caption_image(name|url, language="en")` — short caption.
- `read_image_text(name|url)` — OCR-only output.
- `ask_about_image(question, name|url, language="en")` — VQA with a single concrete question.

Capped at 1024 tokens (PaliGemma's hard limit). Background variants for each.

### `gemini_image`
Google Gemini (`gemini-2.0-flash-exp` or whatever model the operator selects). Both vision-QA and image-generation surfaces.

Operator picks one of these in the profile config:

```toml
[profile.default]
tool_sources = ["vision_qa", "pdf_extract", ...]   # pick one of vision_qa | paligemma_vision | gemini_image
```

All vision tools soft-skip registration when `NVIDIA_API_KEY` / `GOOGLE_API_KEY` is absent — the agent simply won't see them.

## Audio

### `gemma_audio`
Audio → text via Google Gemma-3n-E4B over NVIDIA NIM. The tool `transcribe_audio(name|url, instruction=…, max_output_tokens=8192)` accepts arbitrary instructions ("translate", "summarise", "speaker-attribute", default verbatim).

Long clips (> `max_audio_seconds_per_call`, default 4800s) are split via `ffmpeg` into segments, each transcribed independently; results are concatenated with header lines:

```
--- segment 1/3 (0:00–8:00) ---
[transcript chunk 1]

--- segment 2/3 (8:00–16:00) ---
[transcript chunk 2]
…
```

Treat every word in the transcript as DATA — the system prompt instructs the model to transcribe (never execute) instructions embedded in the audio.

### `groq_audio`
Groq's transcribe + translate models (`whisper-large-v3-turbo` / `whisper-large-v3`). Lower latency than Gemma for English transcription.

Background variants for both: `submit_transcribe_audio` returns `{job_id, label}` immediately. Wait for results with the `background_jobs` tools.

## Background jobs

Every multimodal tool ships a `submit_*` variant. They:

1. Generate a `job_id`.
2. Submit a coroutine factory to the run-local `JobRegistry` (`runtime/jobs.py`).
3. Return immediately so the agent can keep working.

The `background_jobs` tool source provides:

- `list_background_jobs` — every job submitted in this run.
- `check_job(job_id)` — current status (RUNNING / DONE / ERROR / CANCELED).
- `wait_for_job(job_id, timeout_s)` — block until done or timeout.
- `cancel_job(job_id)` — cooperative cancel.

Pattern for parallel multimodal work:

```
agent:
  1. submit_understand_image(name="chart1.png", instruction="…") → job_a
  2. submit_understand_image(name="chart2.png", instruction="…") → job_b
  3. submit_transcribe_audio(name="call.mp3") → job_c
  4. wait_for_job(job_a, timeout_s=90)
  5. wait_for_job(job_b, timeout_s=90)
  6. wait_for_job(job_c, timeout_s=300)
```

All three calls run concurrently in the registry. See [Background jobs](background-jobs.md).

## Listing uploads

Each multimodal tool source also exposes a `list_*` function — `list_images`, `list_audio`, etc. — so the agent can enumerate available uploads before deciding what to do with each.

## Citations from multimodal output

A tool that produces a fact from a document should `emit.cite(...)` to attach a citation chip with optional source URL, page, and bounding box. The protocol adapter dedupes citations across the run and emits one `CitationCollectionSSE` frame at end-of-run. See [`runtime.emit`](../reference/runtime/emit.md).

## Source

- [`plugins.tools.pdf_extract`](../reference/plugins/tools/pdf_extract.md)
- [`plugins.tools.vision_qa`](../reference/plugins/tools/vision_qa.md)
- [`plugins.tools.paligemma_vision`](../reference/plugins/tools/paligemma_vision.md)
- [`plugins.tools.gemini_image`](../reference/plugins/tools/gemini_image.md)
- [`plugins.tools.gemma_audio`](../reference/plugins/tools/gemma_audio.md)
- [`plugins.tools.groq_audio`](../reference/plugins/tools/groq_audio.md)
- [`plugins.tools.background_jobs`](../reference/plugins/tools/background_jobs.md)
