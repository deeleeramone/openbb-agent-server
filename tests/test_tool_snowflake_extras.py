"""Tests for the openbb-snowflake-ai-style helper tools."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest

from openbb_agent_server.plugins.tools.snowflake_tools import SnowflakeToolSource
from openbb_agent_server.plugins.tools.snowflake_tools.client import (
    SESSION_EXPIRED_CODES,
    SnowflakeClient,
    SnowflakeCredentials,
    _is_session_expired,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


class _Conn(sqlite3.Connection):
    def cursor(self, *a, **k):  # type: ignore[no-untyped-def]
        return _Cursor(super().cursor(*a, **k))


class _Cursor:
    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur
        self.sfqid = "test"

    def execute(self, sql: str, params: Any = None):  # type: ignore[no-untyped-def]
        sql = sql.replace(
            "FROM CUSTDB.INFORMATION_SCHEMA.COLUMNS",
            "FROM INFORMATION_SCHEMA_COLUMNS",
        )
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
    cur.execute(
        "CREATE TABLE INFORMATION_SCHEMA_COLUMNS ("
        "  ordinal_position INTEGER, column_name TEXT, data_type TEXT,"
        "  is_nullable TEXT, column_default TEXT, comment TEXT,"
        "  table_schema TEXT, table_name TEXT)"
    )
    cur.execute(
        "INSERT INTO INFORMATION_SCHEMA_COLUMNS VALUES "
        "(1, 'id', 'INTEGER', 'NO', NULL, 'pk', 'PUBLIC', 'CUSTOMERS')"
    )
    cur.execute(
        "INSERT INTO INFORMATION_SCHEMA_COLUMNS VALUES "
        "(2, 'name', 'TEXT', 'YES', NULL, NULL, 'PUBLIC', 'CUSTOMERS')"
    )
    cur.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
    for i in range(8):
        cur.execute("INSERT INTO customers VALUES (?, ?)", (i, f"c{i}"))
    conn.commit()
    yield lambda creds: conn
    conn.close()


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


@pytest.mark.asyncio
async def test_get_table_info_returns_column_metadata(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    info = next(t for t in tools if t.name == "snowflake_get_table_info")
    out = info.invoke({"table": "CUSTDB.PUBLIC.CUSTOMERS"})
    columns = out["columns"]
    assert "ordinal_position" in columns
    assert "column_name" in columns
    assert out["row_count"] == 2
    rows = out["rows"]
    assert rows[0][1] == "id"
    assert rows[1][1] == "name"


@pytest.mark.asyncio
async def test_get_table_info_rejects_unqualified_name(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    info = next(t for t in tools if t.name == "snowflake_get_table_info")
    with pytest.raises(Exception):
        info.invoke({"table": "CUSTOMERS"})


@pytest.mark.asyncio
async def test_get_table_sample_data_caps_rows(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    sample = next(t for t in tools if t.name == "snowflake_get_table_sample_data")
    out = sample.invoke({"table": "customers", "limit": 3})
    assert out["row_count"] == 3
    assert "id" in out["columns"] and "name" in out["columns"]


@pytest.mark.asyncio
async def test_get_multiple_table_definitions_fan_out(sqlite_factory) -> None:
    src = SnowflakeToolSource(
        credentials={"account": "acc", "user": "u"},
        connection_factory=sqlite_factory,
    )
    tools = await src.tools(_ctx(), {})
    multi = next(
        t for t in tools if t.name == "snowflake_get_multiple_table_definitions"
    )
    out = multi.invoke(
        {
            "tables": [
                "CUSTDB.PUBLIC.CUSTOMERS",
                "CUSTDB.PUBLIC.NONE",
            ]
        }
    )
    assert "tables" in out
    assert "CUSTDB.PUBLIC.CUSTOMERS" in out["tables"]
    bad = out["tables"]["CUSTDB.PUBLIC.NONE"]
    assert "error" in bad or bad.get("row_count") == 0


def test_is_session_expired_recognises_codes() -> None:
    class FakeExc(Exception):
        errno = 390112

    assert _is_session_expired(FakeExc())


def test_is_session_expired_recognises_string_match() -> None:
    assert _is_session_expired(Exception("oh no error 390111 happened"))


def test_is_session_expired_returns_false_for_other_errors() -> None:
    assert not _is_session_expired(Exception("wrong password"))


def test_session_expired_codes_constant_unchanged() -> None:
    assert frozenset({390111, 390112, 390114}) == SESSION_EXPIRED_CODES


@pytest.mark.asyncio
async def test_client_retries_after_session_expired() -> None:
    """Reconnect after the factory raises 390112 once, then succeeds."""
    calls = {"count": 0}

    def factory(creds: SnowflakeCredentials):
        calls["count"] += 1
        if calls["count"] == 2:

            class BrokenConn:
                def cursor(self):  # noqa: D401
                    raise Exception("session expired 390112")

                def close(self):
                    pass

            return BrokenConn()
        return _make_real_conn()

    creds = SnowflakeCredentials(account="acc", user="u")
    client = SnowflakeClient(creds, connection_factory=factory)
    client.open()
    real_conn = client._conn  # noqa: SLF001
    raised = {"once": False}

    class _OneShotBroken:
        def cursor(self_inner):  # noqa: N805 — sqlite-style API
            if raised["once"]:
                return real_conn.cursor()
            raised["once"] = True
            raise Exception("390112 session expired")

        def close(self_inner):
            pass

    client._conn = _OneShotBroken()  # noqa: SLF001
    client._factory = lambda c: real_conn  # noqa: SLF001
    result = client.execute("SELECT 1 AS x")
    assert result.row_count == 1


def _make_real_conn():
    conn = sqlite3.connect(":memory:", factory=_Conn)
    return conn
