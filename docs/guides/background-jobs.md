# Background jobs

The `JobRegistry` (`runtime/jobs.py`) lets the agent submit long-running coroutines and keep working in parallel — no separate thread / queue / worker pool needed. It is run-scoped (one registry per agent invocation, lives inside the `bind(ctx)` block) and cancels every still-running task when the run ends.

## Why

Real-world agent loops hit two costs that aren't friendly to the model's loop:

- **Latency.** A multimodal NIM call can take 5-90s. Blocking the chat loop for 90s of "thinking" silence is a bad UX.
- **Sequencing.** A typical task needs three transcriptions, two image OCR passes, and one web search. Doing them serially wastes wall-clock time.

The registry inverts both: every long call has a `submit_*` variant that returns a `job_id` immediately. The agent collects results with `wait_for_job` when it actually needs them.

## Submitting work

The registry is bound at `bind(ctx)` entry and lives in `runtime_state[_STATE_KEY]`. Plug-ins reach it via:

```python
from openbb_agent_server.runtime.jobs import get_registry

job_id = get_registry().submit(
    lambda: my_async_work(arg=1, arg2=2),
    label="my_async_work(arg=1)",
    metadata={"tool": "my_async_work", "arg": 1},
)
```

`submit(awaitable_or_factory, *, label, metadata=None) -> str`:

- A coroutine (already-awaited callable) runs immediately.
- A factory (`lambda: my_async_work(...)`) is preferred — it produces the coroutine lazily so cancel-before-start doesn't leak un-awaited coroutines. `tests/test_runtime_jobs.py::test_cancel_before_first_step_with_factory_no_warning` pins this behaviour.

## Built-in `background_jobs` tools

`plugins/tools/background_jobs.py` exposes the registry as four `StructuredTool`s the agent can call:

| Tool | Args | Returns |
| --- | --- | --- |
| `list_background_jobs` | — | `[{job_id, label, state, metadata, submitted_at, finished_at?}]` |
| `check_job` | `job_id` | snapshot without the result blob |
| `wait_for_job` | `job_id, timeout_s` | snapshot WITH the result on success, or `{state: "TIMEOUT"}` |
| `cancel_job` | `job_id` | `{cancelled: bool}` |

State strings (lowercase) returned in payloads: `running`, `done`, `error`, `canceled`. The `wait_for_job` timeout sentinel is `TIMEOUT` (uppercase). Re-awaiting a finished job is a fast-path that returns the stored result without re-running.

## Job lifecycle

```
submit → running ─┬─ task succeeds → done      (result kept)
                  ├─ task raises   → error     (error string kept)
                  └─ cancel()      → canceled
```

After the host run ends, `bind()`'s `finally` calls `cleanup_state(runtime_state)` which invokes `cancel_all()` and discards the registry. Submitted tasks that haven't finished are signalled with `asyncio.CancelledError`.

## End-to-end pattern

```
agent:
  1. submit_understand_image(name="chart1.png", instruction="What's the y-axis label?")
     → {job_id: "j-1", label: "..."}
  2. submit_understand_image(name="chart2.png", instruction="What's the trend?")
     → {job_id: "j-2", label: "..."}
  3. <continue reasoning / call other tools>
  4. wait_for_job(job_id="j-1", timeout_s=60)
     → {state: "done", result: "Quarterly revenue"}
  5. wait_for_job(job_id="j-2", timeout_s=60)
     → {state: "done", result: "Up and to the right"}
```

The two NIM calls in steps 1-2 run concurrently. `tests/test_nim_integration.py::test_two_real_nim_jobs_run_concurrently` is the live wall-clock proof.

## Cancellation semantics

`cancel(job_id)` is **cooperative** — it sets the underlying `asyncio.Task`'s cancel flag. A task in a blocking `await` returns control immediately; a task that's mid-CPU returns at the next yield point. Pending coroutine factories never start.

`cancel_all()` is what `bind()`'s `finally` calls. Use it during graceful shutdown.

## Writing a tool source that uses jobs

```python
from langchain_core.tools import StructuredTool
from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.jobs import get_registry

async def my_long_call(name: str) -> str:
    # do the real work …
    return ...

async def submit_my_long_call(name: str) -> dict[str, str]:
    job_id = get_registry().submit(
        lambda: my_long_call(name),
        label=f"my_long_call({name})",
        metadata={"tool": "my_long_call", "source": name},
    )
    emit.reasoning_step("submit_my_long_call", job_id=job_id, source=name)
    return {"job_id": job_id, "label": f"my_long_call({name})"}

# Register both via StructuredTool.from_function as `coroutine=` so they run async.
```

See [Writing a tool source](../developing/writing-a-tool-source.md) for the full ABC.

## Observability

Every job carries metadata that flows into logs:

```json
{"level":"INFO","logger":"openbb_agent_server.runtime.jobs",
 "message":"job submitted","details":{"job_id":"j-1","label":"transcribe_audio(call.mp3)",
 "metadata":{"tool":"transcribe_audio","source":"call.mp3"}}}
```

`JobRegistry.list_all()` returns every job submitted in this run (oldest first), suitable for emitting a final "background jobs run this turn" status step at the end of the loop.

## Source

- [`runtime.jobs`](../reference/runtime/jobs.md)
- [`plugins.tools.background_jobs`](../reference/plugins/tools/background_jobs.md)
- Tests: `tests/test_runtime_jobs.py`, `tests/test_tool_background_jobs.py`, `tests/test_jobs_e2e.py`, `tests/test_nim_integration.py`
