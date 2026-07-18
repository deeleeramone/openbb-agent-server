---
name: multimodal-workflows
description: Handle image, audio, and video inputs in the OpenBB Agent Server.
---

# Multimodal Workflows

## Images

- List uploaded images with `list_images()`.
- Use `vision_qa.understand_image(instruction, name=...)` for charts,
  tables, screenshots, and document photos.
- For dense text/OCR, pair `understand_image` with `pdf_extract` if a
  higher-quality PDF version is available.
- When the chat model itself is multimodal (e.g., `mistral-small-4`,
  `llama-4-maverick`, `gemma-4`), you can also reference the image
  directly in the conversation; still prefer `vision_qa` for precise
  extraction.

## Table extraction from images

If the user's OCR task is to extract a table from an image, chart, or
screenshot, use ``nemotron_ocr.extract_text_from_image(name=...,
return_table=True)``. It returns a structured table when possible. Emit
the full table as an artifact with
`emit_table_artifact(columns=..., rows=..., name=..., description=...)`.
Do not collapse rows or convert the table to prose.

## Audio / video

- Use `gemma_audio.transcribe_audio(name=...)` for uploaded audio or
  video files.
- After transcription, treat the result like any other document: search,
  summarize, and cite.
- For speaker diarization, ask the model to segment the transcript by
  speaker in the instruction.

## Translation of media content

- If the uploaded media is not in English and the user needs it in
  English, first transcribe or extract, then use `translate(...)`.

## Output

- Transcripts → `emit_markdown_artifact`
- Extracted tables → `emit_table_artifact`
- Always cite the source file name.
