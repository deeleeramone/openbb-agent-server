# User guides

Conceptual walk-throughs for people running the agent through OpenBB Workspace — what the agent does, how requests flow through it, and which surfaces are user-facing. Start with [Getting started](getting-started.md) if you have never run the server before; everything else assumes the server is up.

## Pages

### [Getting started](getting-started.md)
Prerequisites, the optional install extras for each model provider, the one environment variable needed for a first boot (`NVIDIA_API_KEY`), running `openbb-agent-server`, registering `http://127.0.0.1:6900/agents.json` in Workspace, and the first conversation. Ends with a "what to read next" matrix.

### [Architecture](architecture.md)
The full request lifecycle: FastAPI router → `runtime/builder.py` → DeepAgents loop → `protocol/adapter.py` → SSE on the wire. Covers `GET /agents.json`, `POST /v1/query`, the five SSE event types, cancellation semantics, the four trace ids (`user_id`, `trace_id`, `run_id`, `conversation_id`), the plugin slots, and the storage surfaces. The map to look at if you are debugging a stream.

### [Workspace integration](workspace-integration.md)
Registering the agent in Workspace, the auth-backend header matrix, reserved vs custom `features`, auto-fetch of pinned widgets through `FunctionCallSSE(function="get_widget_data")`, file uploads as `FileRef`, citations with PDF bounding boxes, conversation sharing across devices, stop-button behaviour, and a troubleshooting table.

### [Memory and recall](memory-and-recall.md)
The two storage layers (chat history vs cross-thread vector memory), the active write path (`ingest_request_context`), optional post-run extraction helpers in `memory/writer.py` for custom integrations, the recall pipeline (`ANN fanout → pinned rescue → optional cross-encoder rerank`), scope gating (`memory:write` is the only gate), right-to-erasure, and dropping `MemoryStoreRetriever` into your own LangChain composition.

### [Widgets and data](widgets-and-data.md)
The `widgets.{primary,secondary,extra}` wire field, the auto-fetch round-trip Workspace performs, `WidgetDataStore` ingestion into SQL rows, and the five `inspect_widget_data` tools — `list / read / search / describe / query`. The read-only SQLite SQL surface via `query_widget_data` is documented with view-naming and JSON-unfold rules.

### [Multimodal tools](multimodal.md)
How `FileRef` uploads reach the agent, MIME-to-tool-source routing, PDFs through `pdf_extract` (text + per-word bounding boxes), the three vision back-ends (`vision_qa`, `paligemma_vision`, `gemini_image`) and how the operator picks one, audio through `gemma_audio` (with long-clip `ffmpeg` splitting) and `groq_audio`, and which sources do or do not expose `submit_*` background variants.

### [PyWry chat (ACP)](pywry-chat.md)
Embedding the agent in a desktop window: the `pywry` extra, the `openbb-agent-canvas` app whose main content page is the agent's live canvas (`canvas_html` / `canvas_plotly` / `canvas_table` / ... tools), `create_chat_manager()` attached to any PyWry widget, what transfers from the shared `openbb.toml` (model, tools, middleware, profiles-as-modes, persistence) and what does not (auth backend, client-side Workspace tools), and the SSE → ACP `SessionUpdate` translation underneath.

### [Background jobs](background-jobs.md)
The run-scoped `JobRegistry`, why it exists (latency hiding + parallel fan-out), `submit(factory, label, metadata) -> job_id`, the four `background_jobs` tools (`list_background_jobs`, `check_job`, `wait_for_job`, `cancel_job`), the `submit → running → done/error/canceled` lifecycle, cooperative cancellation, and a worked example of writing a tool source that submits jobs.

## See also

- [`developing/`](../developing/index.md) — writing your own plugins (tool sources, models, middleware, sub-agents, auth backends).
- [`operating/`](../operating/index.md) — configuration, profiles, auth, persistence, memory, observability.
- [`reference/`](../reference/index.md) — symbol-level API reference that mirrors the package tree.
- [`docs/index.md`](../index.md) — the parent index.
