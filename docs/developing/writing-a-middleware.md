# Writing a middleware

Middleware intercepts the agent's interaction with the model and tools. The ABC is a thin LangChain wrapper:

```python
class Middleware(ABC):
    name: str
    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware: ...
```

Where `AgentMiddleware` is `langchain.agents.middleware.types.AgentMiddleware`. The two hookpoints (sync + async) are:

| Hook | Wraps | Use cases |
| --- | --- | --- |
| `wrap_model_call(request, handler)` | every model call (the chat completion that drives the agent loop) | rewrite messages, drop / clamp tools, dedupe tool calls in the response, observe usage |
| `wrap_tool_call(request, handler)` | every tool invocation | announce the call, log args / latency, transform the result, gate execution |

## Minimal example: a redactor

```python
"""Redact obvious PII before sending to the model."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import Middleware


_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _redact(text: str) -> str:
    return _SSN_RE.sub("[REDACTED-SSN]", text)


class _RedactorMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        new_messages = [
            HumanMessage(content=_redact(m.content)) if isinstance(m, HumanMessage)
            else m
            for m in request.messages
        ]
        return await handler(request.override(messages=new_messages))

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        new_messages = [
            HumanMessage(content=_redact(m.content)) if isinstance(m, HumanMessage)
            else m
            for m in request.messages
        ]
        return handler(request.override(messages=new_messages))


class RedactorMiddlewareFactory(Middleware):
    name = "redactor"

    def build(self, ctx: RunContext, config: dict[str, Any]) -> AgentMiddleware:
        return _RedactorMiddleware()
```

Register:

```toml
[project.entry-points."openbb_agent_server.middleware"]
redactor = "my_package.redactor:RedactorMiddlewareFactory"
```

Add to a profile's `middleware = [...]` and restart. Middleware order matters — outermost first.

## When to use `wrap_model_call` vs `wrap_tool_call`

- **`wrap_model_call`** sees `request.messages`, `request.tools`, `request.tool_choice`, and `request.system_prompt`. Use it for prompt mutation, tool filtering, response post-processing.
- **`wrap_tool_call`** sees `request.tool_call` (LangChain v1: a `dict[str, Any]` with `name`, `args`, `id`) or `request.tool_name` / `request.args` (older shape). Use it for per-tool observability, gating, retry.

`tool_call_announcer` (`plugins/middleware/tool_call_announcer.py`) is the reference for tool-call observability. `tool_message_normaliser` is the reference for `wrap_model_call` (rewriting message lists).

## Sync vs async hooks

The agent loop is async, so most middlewares only implement `awrap_model_call` / `awrap_tool_call`. If your middleware needs to participate in sync code paths (rare), implement the matching `wrap_*` method as well. Factor the actual logic out so the two stay consistent:

```python
def _filter(self, request): ...

async def awrap_model_call(self, request, handler):
    return await handler(self._filter(request))

def wrap_model_call(self, request, handler):       # only if needed
    return handler(self._filter(request))
```

## Re-emitting tool errors

`tool_call_announcer` shows the pattern for catching tool failures and surfacing them as reasoning steps without swallowing the exception:

```python
async def awrap_tool_call(self, request, handler):
    from langgraph.errors import GraphBubbleUp
    name = _tool_name(request)
    emit.reasoning_step(f"Calling tool: {name}", event_type="INFO", tool_name=name)
    try:
        return await handler(request)
    except GraphBubbleUp:
        raise                          # control-flow signal; not a real error
    except Exception as exc:
        emit.reasoning_step(f"Tool {name} errored: {exc}", event_type="ERROR", tool_name=name)
        raise
```

Special-case `GraphBubbleUp` — it's LangGraph's interrupt signal, not a tool failure. Letting your error path swallow it breaks resume / HITL flows.

## Per-call config

`build(ctx, config)` is called once per run. `config` comes from the profile's `middleware_config[name]` plus any per-request kwargs. Capture knobs (`excluded` lists, thresholds, rate limits) here and close over them in the middleware instance.

## Ordering

Middleware in the profile list runs **outer-to-inner**. The first entry wraps the second, etc. Inside `wrap_model_call`, `handler` is the next middleware down — you can choose to short-circuit (don't call `handler`) to skip the rest, or pre/post-process around it.

Suggested order (matches the default profile):

1. `tool_call_announcer` (outermost — observe everything that follows)
2. `tool_call_ledger` (next — record what happened)
3. `usage_recorder` (record token usage from model responses)
4. `tool_filter` (drop tools the agent shouldn't see)
5. `tool_message_normaliser` (rewrite messages right before they hit the model)
6. `call_limit` / `tool_call_limit` (hard caps — innermost so they see the final shape)

## Tests

Direct unit tests against the middleware class are cheap:

```python
@pytest.mark.asyncio
async def test_redacts_ssn_before_model_sees_it() -> None:
    mw = RedactorMiddlewareFactory().build(_ctx(), {})
    captured = []
    async def handler(req):
        captured.append(req.messages)
        return AIMessage(content="ok")
    request = _Request([HumanMessage(content="My SSN is 123-45-6789")])
    await mw.awrap_model_call(request, handler)
    assert "[REDACTED-SSN]" in captured[0][0].content
```

`tests/test_middleware_tool_filter.py` and `tests/test_middleware_tool_message_normaliser.py` are full worked examples covering both sync and async paths.

## Source

- [`runtime.plugins`](../reference/runtime/plugins.md)
- Worked examples: every file under [`plugins/middleware/`](../reference/plugins/middleware/index.md).
