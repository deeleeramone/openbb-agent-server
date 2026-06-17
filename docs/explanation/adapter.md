# `openbb_agent_server.protocol.adapter`

Translate a DeepAgents (LangGraph) event stream into Workspace SSE events. The adapter is the only place in the server that knows about Workspace's wire shapes — every other module emits dicts via `runtime.emit` and lets the adapter route them to the right `SSEEvent` variant.

**Source:** [`openbb_agent_server/protocol/adapter.py`](https://github.com/deeleeramone/openbb-agent-server/blob/main/openbb_agent_server/protocol/adapter.py)

## `class _ThinkingStreamSplitter`

Stateful streaming splitter for inline `<thinking>` markers. Consumes text deltas, yields `(channel, text)` pairs where `channel` is `"prose"` or `"thinking"`. Designed so first-token latency is one chunk, not one full AIMessage.

| Concern | Behaviour |
| --- | --- |
| Recognised tags | `<thinking>`, `<think>`, `<reasoning>` (case-insensitive, attributes allowed). |
| Hold-back | At most `_PARTIAL_TAG_HOLD = 32` chars at the buffer tail when it could still be the leading edge of an XML tag OR a Harmony marker. |
| Stream end | `flush()` returns any held-back tail as a final `(channel, text)` pair. |

### Implicit thinking blocks

The `langchain_nvidia_ai_endpoints` adapter is known to STRIP the opening `<think>` tag from streamed content while preserving the closing `</think>`. When the splitter sees a stray close with no matching open in this run, it:

1. Reclassifies the preceding content as `thinking`.
2. Emits a single `("close_unmatched", "")` signal so the adapter can retroactively move buffered prose into the reasoning buffer.
3. Sets `_signalled_implicit` so it's a single-shot — later interleaved thinking/prose doesn't keep moving prose around.

### Harmony-format leak suppression

When `gpt-oss` emits OpenAI Harmony tool-call markup inline — either `【assistant to=functions.X` (channel header) or `<|channel|>` / `<|start|>` delimiters — the LangChain adapter doesn't parse it into `tool_calls`; the markers leak as plain text into the chat bubble. The `_harmony_suppress` flag is set on first sight of either pattern, and EVERY byte from the marker onward is discarded for the remainder of the message. The markers always appear AFTER the visible answer, so suppression is safe.

## `class DeepAgentEventAdapter`

```python
DeepAgentEventAdapter(*, client_tool_names: frozenset[str] = frozenset())
```

| Arg | Purpose |
| --- | --- |
| `client_tool_names` | Tool names registered as client-side. Used to decide whether to emit `copilotFunctionCall` for a given tool call (server-side tools run inline; emitting a function call would make Workspace try to re-execute them remotely). |

### State

| Field | Purpose |
| --- | --- |
| `_cur_id` | The current `AIMessage.id`. Splitter is rotated when this changes. |
| `_splitter` | Active `_ThinkingStreamSplitter`. |
| `_reasoning_buf` | List-of-strings buffer for the current reasoning segment. Flushed as ONE `StatusUpdateSSE` row at the next tool dispatch. |
| `_prose_buf` | Buffer for plain prose since the last tool dispatch — ambiguous mid-stream (reasoning preface vs. final answer). |
| `_artifacts` | All `MessageArtifactSSE` events emitted during the run. Drained AFTER the final answer drains — so cards stack below the chat bubble. |
| `_citations` / `_citation_keys` | Citation buffer + dedup set. Drained as ONE `CitationCollectionSSE` at end of stream. |

### Reasoning/prose routing

Prose between tool calls is reasoning preface; prose after the LAST tool call is the final answer. The adapter can't distinguish until either (a) the next tool call fires, in which case `_prose_buf` flushes as a reasoning row, or (b) the stream ends, in which case it flushes as `MessageChunkSSE`. Server-side tools that don't dispatch a `FunctionCallSSE` (e.g. `emit_table_artifact`, `read_widget_data`) STILL close the reasoning segment — without that, planning prose preceding a server-side tool would leak as the final answer.

### Citation buffer

`_absorb_citations(items)` accumulates citations across the run with stable de-dup:

| Citation type | Dedup key |
| --- | --- |
| Web / PDF (`source_info.type != "widget"`) | `(origin_url, first_quote_snippet)`. |
| Widget | `("widget", widget_uuid)` — OR `("widget", f"{uuid}#p{page}")` when `source_info.metadata.Page` is set, so each PDF page gets its own chip. |

`origin` is intentionally NOT the dedup key for widget citations — `origin` is the vendor name and is shared across every widget from that vendor.

### Citation relevance filter

`web_search` and `fetch_url` auto-cite every result and fetched page, so a multi-search run accumulates far more citations than the answer uses. `_drain_citations` keeps only the citations the **final answer references**: as events stream, `_note_for_citations` records the chat-bubble text and every emitted artifact's content; at drain time a web / PDF citation survives only when ≥ 60% of its title's significant words (`_CITATION_MATCH_THRESHOLD`) appear in that text. Widget citations and citations with no usable title are always kept. If no citation survives, no `CitationCollectionSSE` is emitted.

### `async def adapt(stream)`

Translate a DeepAgents event stream (`type` ∈ `messages` / `updates` / `custom` / `error`) into Workspace SSE events. Tail order:

1. `_drain_pending` — flush buffered prose as the final `MessageChunkSSE`.
2. `_drain_artifacts` — emit every buffered artifact in arrival order.
3. `_drain_citations` — emit ONE `CitationCollectionSSE` with the citations the final answer references (see *Citation relevance filter*).

This ordering guarantees the chat-bubble text lands first, every artifact card stacks below it, and citation chips render in one batch instead of trickling in.

### `_build_function_call_from_tool_call`

Maps one `message.tool_calls[i]` entry to a `FunctionCallSSE` or `None`. Rules:

| Tool name | `server_id` | `function` | Emitted? |
| --- | --- | --- | --- |
| Starts with `mcp:<server>:<fn>` | `<server>` | `<fn>` | Yes (`execute_agent_tool`). |
| Starts with `mcp:<fn>` (no server id) | namespace head | `<fn>` | Yes. |
| Starts with `client:<fn>` | namespace head | `<fn>` | Yes. |
| In `client_tool_names` set | `"agent"` | `<name>` | Yes. |
| In `_WORKSPACE_NATIVE_FUNCTIONS` (`get_widget_data`, `add_widget_to_dashboard`, etc.) AND NOT in `client_tool_names` | n/a | n/a | NO — it's a Workspace built-in dispatched separately. |
| Anything else (server-side) | n/a | n/a | NO — runs inline; `ToolMessage` feeds back to the model. |

`_WORKSPACE_NATIVE_FUNCTIONS` is the closed set Workspace dispatches itself: `get_widget_data`, `get_extra_widget_data`, `get_params_options`, `add_widget_to_dashboard`, `add_generative_widget`, `update_widget_in_dashboard`, `assign_tasks_to_agents`, `execute_agent_tool`, `manage_navigation_bar`, `get_skill_content`.

## Module helpers

| Function | Purpose |
| --- | --- |
| `_flatten_reasoning(raw)` | Normalise `additional_kwargs.reasoning_content` (str / list of blocks) to flat string. |
| `_split_thinking(text)` | Non-streaming variant — returns `(thinking_blocks, remaining_prose)`. Also strips `【…cursor/loc/source/ref…】` citation markers and collapses the whitespace they leave behind. |
| `_build_artifact(payload)` | Coerce a tool's artifact dict into a `ClientArtifact`. Resolves `markdown` → `text` (Workspace has no markdown type) and shapes table content from `{columns, rows}`. |
| `_extract_text(content)` | Pull user-visible text out of any LangChain `AIMessage.content` shape (string / list of block dicts). |
| `_coerce_status_event_type` / `_coerce_status_details` | Defensive coercion for plugin-emitted step rows. |

## See also

- [`protocol/sse.md`](../reference/protocol/sse.md) — the wire encoder.
- [`protocol/schemas.md`](../reference/protocol/schemas.md) — every event variant.
- [`runtime/emit.md`](../reference/runtime/emit.md) — the plugin-side API the adapter consumes.
