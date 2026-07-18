---
name: tool-use
description: Best practices for using the OpenBB Agent Server tool catalog, artifacts, and citations.
---

# Tool Use Best Practices

## When to use a tool

- The answer requires data, computation, or a side effect the tool provides.
- Pick the smallest useful tool; avoid chains for simple questions.
- Never retry a failed tool with the same input; change something
  meaningful or tell the user what went wrong.

## Widget data discipline

- The Attached widgets snapshot at the top of the system prompt already
  lists pinned widgets. Do not call `list_widgets()` unless you need to
  discover newly added widgets.
- Call `get_widget_data(widget_ids=[...])` once with every widget you
  need.
- After ingestion, read tabular widgets with `read_widget_data` or
  `query_widget_data`; do not re-fetch unchanged widgets.

## PDF / image discipline

- Use `list_pdfs()` to see uploaded PDFs.
- Use `get_pdf_outline(name)` to find relevant pages, then
  `pdf_extract(name, page_range=(start, end))`.
- For image-based documents, charts, or screenshots use
  `nemotron_ocr.extract_text_from_image(name=..., return_table=True)`;
  for general visual Q&A use `vision_qa.understand_image(...)`.
- Cite every claim drawn from a PDF or image with `cite_source(...)`.

## Web discipline

- Use `web_search(query, k=3)` for current events and post-training facts.
- Use `fetch_url(url)` only when the user asks for a specific article.
- Both tools attach citation cards automatically; cite them in the answer.

## Artifact discipline

- Render long-form results as `emit_markdown_artifact`.
- Render tables as `emit_table_artifact`.
- Render plots as `emit_chart_artifact` with Plotly JSON.
- Keep the chat reply terse; the artifact is the deliverable.

## Reasoning visibility

- Use `emit_reasoning_step(message)` for multi-step status updates.
- Surface uncertainty rather than inventing numbers.
