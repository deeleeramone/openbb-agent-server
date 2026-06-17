# Testing

The agent server has 1300+ tests and the project's hard rule is **100% line + branch coverage** on every new module, with `# pragma: no cover` only on genuinely-unreachable code accompanied by a WHY comment.

## Layout

```
tests/
├── conftest.py                    # fixtures (alice/bob principals, history, settings_env)
├── fixtures/                      # real installable openbb extension stubs
├── test_*.py                      # one file per source module (mirror or near-mirror)
```

Stub extensions in `fixtures/` are real `pip install -e .` packages — the project's standing rule is **no SimpleNamespace / mock for `obb`**. Real installable extensions surface integration bugs that mocks hide.

## Conventions

- **One test per behaviour.** `test_<verb>_<state>` naming. Several small tests beat one big one — easier to read failure messages.
- **No narrative chatter** in test files. Numpy-style docstrings always.
- **Real fixtures over mocks** wherever practical. SQLite is fast; spin up a real store with `tmp_path`.
- **Use real classes.** `_BrokenEmbeddings(Embeddings)` that subclasses the LangChain ABC beats `Mock(spec=Embeddings)`.

## Running

```sh
# whole suite
pytest tests/

# one file
pytest tests/test_memory_store.py

# one test
pytest tests/test_memory_store.py::test_pinned_memories_dominate_recall_score

# with coverage
pytest tests/ --cov=openbb_agent_server --cov-report=term-missing

# without coverage (faster)
pytest tests/ --no-cov
```

`pyproject.toml` sets `asyncio_mode = auto` so async tests don't need an explicit `@pytest.mark.asyncio` decorator on every function. `pytest-timeout` is in place — long-running async tests use `async with asyncio.timeout(...)` for python-level deadlines (pytest-timeout's `thread` method can't kill async tasks stuck in httpx C-level reads).

## Three layers

### 1. Unit

Direct, isolated. Instantiate the class, call the method, assert. `tests/test_memory_factory.py`, `tests/test_router_helpers.py`.

### 2. Integration

Mount the FastAPI app, drive it with `httpx.AsyncClient`, assert SSE event sequences. `tests/test_router_query_integration.py`, `tests/test_jobs_e2e.py`.

### 3. Live

Real network calls, gated behind an env-var check. Use `pytest.mark.skipif(not os.environ.get("NVIDIA_API_KEY"), reason="…")`. `tests/test_nim_integration.py`, `tests/test_embeddings_live_integration.py`, `tests/test_media_integration.py`.

Live tests should:

- Wrap the body in `async with asyncio.timeout(60)` for python-level deadlines.
- Bound the assertion to behaviour, not exact text ("contains 'apple'" beats "equals 'Apple Inc'").
- Use synthesized payloads (Pillow for images, `wave` for audio) rather than checked-in binaries.

## Fixtures the suite provides

`conftest.py` exposes:

| Fixture | Yields |
| --- | --- |
| `alice` | `UserPrincipal(user_id="alice", scopes=("agent:query","memory:read","memory:write"))` |
| `bob` | `UserPrincipal(user_id="bob", scopes=("agent:query",))` |
| `history` | `SqliteHistoryStore` over `tmp_path` |
| `bearer_env` | sets `OPENBB_AGENT_AUTH_BEARER=test-bearer-token` |
| `settings_env` | full env to build an `AgentServerSettings` with the `fake` model + `none` auth |

Use them — re-rolling principals in every test gets tedious.

## The autouse env isolation

`conftest.py::_isolate_env` runs automatically and:

- Sets `HOME` to `tmp_path` so per-user settings can't leak between tests.
- Strips every `OPENBB_AGENT_*` env var so a test always starts from a known baseline.
- Forces `EMBEDDINGS_PROVIDER=hash`, `RERANKER_PROVIDER=""`, `TRANSLATION_PROVIDER=""` so live NIM calls don't accidentally happen during unit tests.

It does **not** strip `NVIDIA_API_KEY` / `GOOGLE_API_KEY` etc. — live tests need those.

## Coverage targets per module

Goal: 100%. Pragma allowed only for:

```python
except ImportError as exc:  # pragma: no cover - install hint
    raise RuntimeError("install [nvidia] extra") from exc
```

and similar genuinely-unreachable defensive paths, with a WHY comment.

## When tests can't reach a path

If a branch is hard to reach (e.g. only triggered by a specific NIM error response), prefer:

1. Refactor so the branch lives in a small, isolatable helper that's easy to unit-test.
2. Inject the failure with `monkeypatch.setattr(target, "method", broken_method)`.
3. Last resort: `# pragma: no cover - WHY`.

## Failing tests are mine

The project's standing rule is "all test failures are mine to fix" — including ones that look pre-existing. If a test breaks because of code I touched, I fix it; if it breaks for another reason, I still fix it. Don't deflect with "wasn't me".

## Test recipes

### Async fixture pattern

```python
@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteMemoryStore]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'm.db'}"
    history = SqliteHistoryStore(url)
    await history.init_schema()
    try:
        yield SqliteMemoryStore(url, embeddings=HashEmbeddings(dim=64))
    finally:
        await history.aclose()
```

### Patching `ctx`

```python
from openbb_agent_server.runtime.context import RunContext, bind

ctx = RunContext(principal=UserPrincipal(user_id="u"), trace_id="t", run_id="r", conversation_id="c")
with bind(ctx):
    # tools that call run_context.current() see this ctx
    ...
```

### Stubbing a LangChain integration

```python
import sys
import types

def _install_fake_nvidia(monkeypatch: pytest.MonkeyPatch, **attrs: Any) -> None:
    mod = types.ModuleType("langchain_nvidia_ai_endpoints")
    for name, val in attrs.items():
        setattr(mod, name, val)
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", mod)
```

### Driving SSE

```python
async with httpx.AsyncClient(app=app, base_url="http://test") as client:
    async with client.stream("POST", "/v1/query", json=body) as resp:
        async for line in resp.aiter_lines():
            ...
```

## Source

Test files referenced above:

- `tests/test_memory_store.py`
- `tests/test_memory_factory.py`
- `tests/test_router_helpers.py`
- `tests/test_router_query_integration.py`
- `tests/test_jobs_e2e.py`
- `tests/test_nim_integration.py`
- `tests/test_embeddings_live_integration.py`
- `tests/conftest.py`
