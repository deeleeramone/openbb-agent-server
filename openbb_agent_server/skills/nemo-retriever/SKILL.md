---
name: nemo-retriever
description: Ingest and query document collections with NVIDIA NeMo Retriever-style retrieval workflows.
---

# NVIDIA NeMo Retriever Workflows

Use this skill whenever the user asks to search, summarize, or extract
facts from a corpus of uploaded documents, filings, PDFs, images,
spreadsheets, presentations, audio, or video.

## Ingestion pipeline

1. Identify the source material:
   - PDF widgets / uploaded PDFs → `list_pdfs()` then `pdf_extract(name, page_range)`
   - Images containing tables/charts → `vision_qa.understand_image(...)`
   - Audio / video clips → `gemma_audio.transcribe_audio(...)`
   - Web pages → `fetch_url(url)`
2. For large documents (>800 pages or many files), split ingestion into
   batches and summarize each batch before final synthesis.
3. If the material is in a non-English language and you need to embed or
   compare it, use `translate(text, target_language="English")` first.
4. Durable user memory is written automatically by the server's
   ingestion pipeline; `recall_user_memory(query, k)` is read-only and
   retrieves facts the user has accumulated across prior conversations.

## Table extraction rule

When an OCR or PDF-extraction task is explicitly about tables, return the
full table as an `emit_table_artifact(columns=..., rows=..., name=...,
description=...)` — do not summarize the table into prose or drop rows.
Include a concise `description` explaining the source document/page.

Use the dedicated ``nemotron_ocr.extract_text_from_image(...,
return_table=True)`` tool for image-based tables; it understands layout
and returns structured rows when possible.

## Query workflow

1. Reformulate the user's question into 1-3 retrieval queries.
2. Search the extracted content with the appropriate tool:
   - PDFs → `search_pdf(query, k=5)`
   - Widget data tables → `query_widget_data(sql=...)`
   - Web → `web_search(query, k=5)`
3. Rerank by relevance: prefer primary sources, recent filings, and tables
   that directly answer the question.
4. Synthesize a concise answer and cite every claim with
   `cite_source(text, source, source_url)`.

## Output

- Factual summaries → `emit_markdown_artifact(...)`
- Extracted tables → `emit_table_artifact(...)`
- Do not return raw OCR dumps as chat text; always summarize.
