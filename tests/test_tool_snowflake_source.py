"""SnowflakeToolSource tests — full LangChain tool surface."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest

from openbb_agent_server.plugins.tools.snowflake_tools import SnowflakeToolSource
from openbb_agent_server.plugins.tools.snowflake_tools.client import (
    SnowflakeClient,
    SnowflakeCredentials,
)
from openbb_agent_server.runtime import (
    emit,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


class _Conn(sqlite3.Connection):
    def cursor(self, *a, **k):  # type: ignore[no-untyped-def]
        return _Cursor(super().cursor(*a, **k))


class _Cursor:
    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur
        self.sfqid = "test-id"

    def execute(self, sql: str, params: Any = None):  # type: ignore[no-untyped-def]
        if params and isinstance(params, dict):
            converted = sql
            for key in params:
                converted = converted.replace(f"%({key})s", f":{key}")
            return self._cur.execute(converted, params)
        return self._cur.execute(sql, params or ())

    def fetchall(self):
        return list(self._cur.fetchall())

    def fetchmany(self, size):
        return list(self._cur.fetchmany(size))

    def fetchone(self):
        return self._cur.fetchone()

    @property
    def description(self):
        return self._cur.description

    def close(self) -> None:
        self._cur.close()


@pytest.fixture
def sqlite_factory() -> Iterator:
    conn = sqlite3.connect(":memory:", factory=_Conn)
    cur = conn.cursor()
    cur.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
    for i in range(5):
        cur.execute("INSERT INTO customers VALUES (?, ?)", (i, f"c{i}"))
    conn.commit()
    yield lambda creds: conn
    conn.close()


def _ctx(api_keys: dict[str, str] | None = None) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys=api_keys or {},
    )


@pytest.fixture
def captured_emits(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    monkeypatch.setattr(emit, "_writer", lambda: out.append)
    return out


@pytest.mark.asyncio
async def test_tool_source_yields_full_surface(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    names = {t.name for t in tools}
    expected = {
        "snowflake_query",
        "snowflake_list_databases",
        "snowflake_list_schemas",
        "snowflake_list_tables",
        "snowflake_describe",
        "snowflake_search_catalog",
        "snowflake_explain",
        "snowflake_query_history",
        "snowflake_cortex_complete",
        "snowflake_cortex_summarize",
        "snowflake_cortex_sentiment",
        "snowflake_cortex_translate",
        "snowflake_cortex_classify",
        "snowflake_cortex_extract_answer",
        "snowflake_cortex_embed",
        "snowflake_cortex_search",
        "snowflake_cortex_analyst",
    }
    assert expected.issubset(names)


@pytest.mark.asyncio
async def test_query_tool_runs_and_emits_table_artifact(
    sqlite_factory, captured_emits: list[dict[str, Any]]
) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    query = next(t for t in tools if t.name == "snowflake_query")
    out = query.invoke({"sql": "SELECT id, name FROM customers", "max_rows": 3})
    assert out["row_count"] == 3
    assert out["truncated"] is False
    assert "id" in out["columns"] and "name" in out["columns"]
    assert out["query_id"]
    artifact_emits = [e for e in captured_emits if e.get("type") == "artifact"]
    assert len(artifact_emits) == 1
    assert artifact_emits[0]["artifact"]["type"] == "table"


@pytest.mark.asyncio
async def test_query_tool_rejects_mutating(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    query = next(t for t in tools if t.name == "snowflake_query")
    with pytest.raises(Exception):
        query.invoke({"sql": "DELETE FROM customers"})


@pytest.mark.asyncio
async def test_credentials_layer_via_ctx_api_keys(sqlite_factory) -> None:
    """Override plugin defaults with per-request api_keys."""
    captured_creds: dict[str, Any] = {}

    def factory(creds: SnowflakeCredentials):
        captured_creds["account"] = creds.account
        captured_creds["user"] = creds.user
        captured_creds["role"] = creds.role
        return sqlite_factory(creds)

    src = SnowflakeToolSource(
        credentials={"account": "default-acc", "user": "default-u"},
        connection_factory=factory,
    )
    ctx = _ctx(
        api_keys={
            "SNOWFLAKE_ACCOUNT": "ctx-acc",
            "SNOWFLAKE_USER": "ctx-user",
            "SNOWFLAKE_ROLE": "ANALYST",
        }
    )
    tools = await src.tools(ctx, {})
    query = next(t for t in tools if t.name == "snowflake_query")
    query.invoke({"sql": "SELECT 1"})
    assert captured_creds["account"] == "ctx-acc"
    assert captured_creds["user"] == "ctx-user"
    assert captured_creds["role"] == "ANALYST"


@pytest.mark.asyncio
async def test_describe_runs_native(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    explain = next(t for t in tools if t.name == "snowflake_explain")
    out = explain.invoke({"sql": "SELECT 1"})
    assert out["statement_kind"]
    assert out["query_id"]


@pytest.mark.asyncio
async def test_per_call_max_rows_via_config(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
        max_rows=2,
    )
    tools = await src.tools(_ctx(), {})
    query = next(t for t in tools if t.name == "snowflake_query")
    out = query.invoke({"sql": "SELECT id FROM customers", "max_rows": 4})
    assert out["row_count"] == 4


@pytest.mark.asyncio
async def test_read_only_off_via_constructor(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
        read_only=False,
    )
    tools = await src.tools(_ctx(), {})
    query = next(t for t in tools if t.name == "snowflake_query")
    query.invoke({"sql": "DELETE FROM customers WHERE id = 0"})


@pytest.mark.asyncio
async def test_runtime_emit_records_query_metadata(
    sqlite_factory, captured_emits: list[dict[str, Any]]
) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    query = next(t for t in tools if t.name == "snowflake_query")
    query.invoke({"sql": "SELECT 1 AS v"})
    step_emits = [e for e in captured_emits if e.get("type") == "step"]
    assert len(step_emits) >= 2
    success = [e for e in step_emits if e.get("event_type") == "SUCCESS"]
    assert success and "query_id" in success[0]["details"]


@pytest.mark.asyncio
async def test_search_catalog_uses_named_param(sqlite_factory) -> None:
    """Verify the catalog search SQL pass-through and param substitution."""
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    search = next(t for t in tools if t.name == "snowflake_search_catalog")
    with pytest.raises(Exception):
        search.invoke({"pattern": "%customer%"})


def test_build_tools_static_returns_tool_list(sqlite_factory) -> None:
    creds = SnowflakeCredentials(account="acc", user="u")
    client = SnowflakeClient(creds, connection_factory=sqlite_factory)
    tools = SnowflakeToolSource.build_tools(client, creds, max_rows=5)
    assert any(t.name == "snowflake_query" for t in tools)


def test_snowflake_tool_surface_invokes_every_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive each snowflake tool through a stub client."""
    from openbb_agent_server.plugins.tools.snowflake_tools import (
        SnowflakeToolSource,
        cortex as cortex_mod,
    )
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeCredentials,
    )

    class _Result:
        def __init__(self) -> None:
            self.columns = ("a",)
            self.rows = [(1,)]
            self.row_count = 1
            self.truncated = False
            self.query_id = "qid"
            self.elapsed_ms = 1
            self.statement_kind = "SELECT"

    class _StubClient:
        max_rows = 10

        def execute(
            self, sql: str, params: Any = None, max_rows: int | None = None
        ) -> _Result:
            return _Result()

    cortex_calls: list[str] = []

    def _record(name: str):
        def fn(*_a: Any, **_k: Any) -> Any:
            cortex_calls.append(name)
            if name == "cortex_sentiment":
                return 0.5
            if name == "cortex_embed":
                return [0.1, 0.2]
            if name == "cortex_classify_text":
                return {"label": "LABEL", "score": 0.9}
            if name == "cortex_extract_answer":
                return {"answer": "x"}
            if name == "cortex_search":
                return {"results": [{"chunk": "c", "title": "T", "url": "u"}]}
            if name == "cortex_analyst":
                return {"messages": []}
            return "stub"

        return fn

    for fn_name in (
        "cortex_complete",
        "cortex_summarize",
        "cortex_sentiment",
        "cortex_translate",
        "cortex_classify_text",
        "cortex_extract_answer",
        "cortex_embed",
        "cortex_search",
        "cortex_analyst",
    ):
        monkeypatch.setattr(cortex_mod, fn_name, _record(fn_name))

    creds = SnowflakeCredentials(account="a", user="u")
    tools = SnowflakeToolSource.build_tools(_StubClient(), creds, max_rows=10)
    by_name = {t.name: t for t in tools}

    by_name["snowflake_list_databases"].invoke({})
    by_name["snowflake_list_schemas"].invoke({"database": "DB"})
    by_name["snowflake_list_tables"].invoke({"database": "DB", "schema": "S"})
    by_name["snowflake_describe"].invoke({"object_path": "DB.S.T"})
    by_name["snowflake_get_table_info"].invoke({"table": "DB.S.T"})
    by_name["snowflake_get_table_sample_data"].invoke({"table": "DB.S.T"})
    by_name["snowflake_get_multiple_table_definitions"].invoke({"tables": ["DB.S.T"]})
    by_name["snowflake_search_catalog"].invoke({"pattern": "%cust%"})
    by_name["snowflake_explain"].invoke({"sql": "SELECT 1"})
    by_name["snowflake_query_history"].invoke({})
    by_name["snowflake_cortex_complete"].invoke({"prompt": "hi"})
    by_name["snowflake_cortex_summarize"].invoke({"text": "the quick brown"})
    by_name["snowflake_cortex_sentiment"].invoke({"text": "happy"})
    by_name["snowflake_cortex_translate"].invoke(
        {"text": "hi", "target_language": "fr"}
    )
    by_name["snowflake_cortex_classify"].invoke({"text": "x", "categories": ["a", "b"]})
    by_name["snowflake_cortex_extract_answer"].invoke({"question": "q", "context": "c"})
    by_name["snowflake_cortex_embed"].invoke({"text": "x"})
    by_name["snowflake_cortex_search"].invoke(
        {"database": "DB", "schema": "S", "service": "SVC", "query": "q"}
    )
    by_name["snowflake_cortex_analyst"].invoke({"messages": [{"role": "user"}]})

    assert {
        "cortex_complete",
        "cortex_summarize",
        "cortex_sentiment",
        "cortex_translate",
        "cortex_classify_text",
        "cortex_extract_answer",
        "cortex_embed",
        "cortex_search",
        "cortex_analyst",
    } <= set(cortex_calls)


def test_snowflake_get_table_info_rejects_short_path() -> None:
    """Reject a short table path in snowflake_get_table_info."""
    from openbb_agent_server.plugins.tools.snowflake_tools import SnowflakeToolSource
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeCredentials,
    )

    class _StubClient:
        max_rows = 10

        def execute(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("execute should not be reached")

    creds = SnowflakeCredentials(account="a", user="u")
    tools = SnowflakeToolSource.build_tools(_StubClient(), creds, max_rows=10)
    info = next(t for t in tools if t.name == "snowflake_get_table_info")
    with pytest.raises(Exception, match="DB.SCHEMA.TABLE"):
        info.invoke({"table": "OnlyOnePart"})


def test_snowflake_get_multiple_table_definitions_collects_per_table_errors() -> None:
    """Collect per-table errors in snowflake_get_multiple_table_definitions."""
    from openbb_agent_server.plugins.tools.snowflake_tools import SnowflakeToolSource
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeCredentials,
    )

    class _Boom:
        max_rows = 10

        def execute(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("boom")

    creds = SnowflakeCredentials(account="a", user="u")
    tools = SnowflakeToolSource.build_tools(_Boom(), creds, max_rows=10)
    multi = next(
        t for t in tools if t.name == "snowflake_get_multiple_table_definitions"
    )
    out = multi.invoke({"tables": ["DB.S.T"]})
    assert out["tables"]["DB.S.T"]["error"]
