"""Snowflake client tests using a real sqlite3 connection."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from openbb_agent_server.plugins.tools.snowflake_tools.client import (
    SnowflakeClient,
    SnowflakeCredentials,
)
from openbb_agent_server.plugins.tools.snowflake_tools.safety import (
    SnowflakeSafetyViolation,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


class _TestConn(sqlite3.Connection):
    """sqlite3 connection adding the Snowflake-specific bits we use."""

    def cursor(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        cur = super().cursor(*args, **kwargs)
        return _TestCursor(cur)


class _TestCursor:
    """Wrap a sqlite3 cursor with sfqid and Snowflake placeholder support."""

    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur
        self.sfqid = "synthetic-test-id"

    def execute(self, sql: str, params=None):  # type: ignore[no-untyped-def]
        if params is None:
            return self._cur.execute(sql)
        if isinstance(params, dict):
            converted = sql
            for key in params:
                converted = converted.replace(f"%({key})s", f":{key}")
            return self._cur.execute(converted, params)
        return self._cur.execute(sql, params)

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


def _seed(conn: _TestConn) -> None:
    cur = conn.cursor()
    cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    for i in range(50):
        cur.execute("INSERT INTO t VALUES (?, ?)", (i, f"row-{i}"))
    conn.commit()


@pytest.fixture
def sqlite_factory() -> Iterator:
    conn = sqlite3.connect(":memory:", factory=_TestConn)
    _seed(conn)
    try:
        yield lambda creds: conn
    finally:
        conn.close()


def _client(factory, **overrides) -> SnowflakeClient:
    creds = SnowflakeCredentials(account="acc", user="u")
    return SnowflakeClient(creds, connection_factory=factory, **overrides)


def test_execute_select_returns_rows_and_query_id(sqlite_factory) -> None:
    client = _client(sqlite_factory, max_rows=10)
    result = client.execute("SELECT id, name FROM t")
    assert result.statement_kind == "SELECT"
    assert result.row_count == 10
    assert result.truncated is False
    assert result.columns == ["id", "name"]
    assert result.query_id
    assert result.elapsed_ms is not None


def test_execute_show_tables(sqlite_factory) -> None:
    client = _client(sqlite_factory)
    result = client.execute("SELECT name FROM sqlite_master WHERE type='table'")
    assert result.row_count >= 1


def test_execute_rejects_mutating_when_read_only(sqlite_factory) -> None:
    client = _client(sqlite_factory, read_only=True)
    with pytest.raises(SnowflakeSafetyViolation):
        client.execute("DROP TABLE t")


def test_execute_allows_mutating_when_read_only_disabled(sqlite_factory) -> None:
    client = _client(sqlite_factory, read_only=False)
    client.execute("UPDATE t SET name = 'updated' WHERE id = 0")


def test_max_rows_truncates_results(sqlite_factory) -> None:
    client = _client(sqlite_factory, max_rows=5)
    result = client.execute("SELECT id FROM t")
    assert result.row_count == 5


def test_per_call_max_rows_overrides_default(sqlite_factory) -> None:
    client = _client(sqlite_factory, max_rows=100)
    result = client.execute("SELECT id FROM t", max_rows=3)
    assert result.row_count == 3


def test_named_parameters_are_translated(sqlite_factory) -> None:
    client = _client(sqlite_factory)
    result = client.execute("SELECT id FROM t WHERE id = %(target)s", {"target": 7})
    assert result.rows == [[7]]


def test_open_close_round_trip(sqlite_factory) -> None:
    client = _client(sqlite_factory)
    client.open()
    client.close()
    client.open()
    client.close()


def test_credentials_to_connect_kwargs_excludes_empties() -> None:
    creds = SnowflakeCredentials(account="acc", user="u")
    kwargs = creds.to_connect_kwargs()
    assert kwargs["account"] == "acc"
    assert kwargs["user"] == "u"
    assert "password" not in kwargs
    assert "private_key" not in kwargs


def test_credentials_loads_pem_string() -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    creds = SnowflakeCredentials(account="acc", user="u", private_key=pem)
    kwargs = creds.to_connect_kwargs()
    assert isinstance(kwargs["private_key"], (bytes, bytearray))
    assert kwargs["authenticator"] == "snowflake"


def test_credentials_to_connect_kwargs_carries_schema_via_alias() -> None:
    creds = SnowflakeCredentials(account="acc", user="u", schema="S")
    kwargs = creds.to_connect_kwargs()
    assert kwargs["schema"] == "S"


def test_snowflake_credentials_layer_picks_every_supported_env_var() -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools.source import (
        _credentials_from_ctx,
    )

    ctx = RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        api_keys={
            "SNOWFLAKE_ACCOUNT": "acct",
            "SNOWFLAKE_USER": "user",
            "SNOWFLAKE_PASSWORD": "pw",
            "SNOWFLAKE_PRIVATE_KEY": "pk",
            "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE": "pkpass",
            "SNOWFLAKE_AUTHENTICATOR": "oauth",
            "SNOWFLAKE_TOKEN": "tok",
            "SNOWFLAKE_ROLE": "role",
            "SNOWFLAKE_WAREHOUSE": "wh",
            "SNOWFLAKE_DATABASE": "db",
            "SNOWFLAKE_SCHEMA": "sch",
            "SNOWFLAKE_HOST": "h.example",
            "SNOWFLAKE_REGION": "us-east-1",
        },
    )
    creds = _credentials_from_ctx(ctx, base=None)
    assert creds.account == "acct"
    assert creds.user == "user"
    assert creds.password == "pw"
    assert creds.private_key == "pk"
    assert creds.private_key_passphrase == "pkpass"
    assert creds.authenticator == "oauth"
    assert creds.token == "tok"
    assert creds.role == "role"
    assert creds.warehouse == "wh"
    assert creds.database == "db"
    assert creds.schema_ == "sch"
    assert creds.host == "h.example"
    assert creds.region == "us-east-1"


def test_snowflake_client_context_manager_open_close() -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeClient,
        SnowflakeCredentials,
    )

    creds = SnowflakeCredentials(account="a", user="u")
    factory_calls = {"open": 0, "close": 0}

    class _Conn:
        def cursor(self) -> Any:
            class _C:
                def execute(self_inner, *_a: Any, **_k: Any) -> None:
                    pass

                def close(self_inner) -> None:
                    pass

            return _C()

        def close(self) -> None:
            factory_calls["close"] += 1

    def _factory(_creds: Any) -> Any:
        factory_calls["open"] += 1
        return _Conn()

    client = SnowflakeClient(creds, connection_factory=_factory)
    with client as bound:
        assert bound is client
        assert factory_calls["open"] == 1
    assert factory_calls["close"] == 1


def test_snowflake_client_truncates_when_rows_exceed_cap() -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeClient,
        SnowflakeCredentials,
    )

    rows = [(i,) for i in range(10)]

    class _Cursor:
        sfqid = "qid"
        description = [("v",)]

        def __init__(self) -> None:
            self._rows: list[tuple[Any, ...]] | None = rows

        def execute(self, *_a: Any, **_k: Any) -> None:
            pass

        def fetchmany(self, n: int) -> list[tuple[Any, ...]]:
            assert self._rows is not None
            return self._rows[:n]

        def fetchall(self) -> list[tuple[Any, ...]]:  # pragma: no cover
            assert self._rows is not None
            return list(self._rows)

        def close(self) -> None:
            pass

    class _Conn:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def close(self) -> None:
            pass

    creds = SnowflakeCredentials(account="a", user="u")
    client = SnowflakeClient(creds, connection_factory=lambda _c: _Conn(), max_rows=3)
    result = client.execute("SELECT v FROM x")
    assert result.row_count == 3
    assert result.truncated is True


def test_snowflake_client_reconnects_when_session_expired_on_cursor_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconnect when the session expires on cursor open."""
    from openbb_agent_server.plugins.tools.snowflake_tools import client as client_mod
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeClient,
        SnowflakeCredentials,
    )

    monkeypatch.setattr(client_mod, "_is_session_expired", lambda _exc: True)

    class _Cursor:
        sfqid = "q"
        description = [("v",)]

        def execute(self, *_a: Any, **_k: Any) -> None:
            pass

        def fetchmany(self, n: int) -> list[tuple[Any, ...]]:
            return [(1,)]

        def fetchall(self) -> list[tuple[Any, ...]]:
            return [(1,)]

        def close(self) -> None:
            pass

    class _Conn:
        def __init__(self, expire_on_first: bool) -> None:
            self._expire_on_first = expire_on_first
            self._cursor_calls = 0

        def cursor(self) -> _Cursor:
            self._cursor_calls += 1
            if self._cursor_calls == 1 and self._expire_on_first:
                raise RuntimeError("session expired")
            return _Cursor()

        def close(self) -> None:
            pass

    class _ConnInitFirst(_Conn):
        def __init__(self) -> None:
            super().__init__(expire_on_first=False)
            self._init_done = False

        def cursor(self) -> _Cursor:
            self._cursor_calls += 1
            if not self._init_done:
                self._init_done = True
                return _Cursor()
            raise RuntimeError("session expired")

    seq = iter([_ConnInitFirst(), _Conn(expire_on_first=False)])

    def _factory(_creds: Any) -> Any:
        return next(seq)

    creds = SnowflakeCredentials(account="a", user="u")
    client = SnowflakeClient(creds, connection_factory=_factory, max_rows=10)
    result = client.execute("SELECT 1")
    assert result.row_count == 1


def test_snowflake_client_reconnects_when_session_expired_on_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconnect when the session expires on execute."""
    from openbb_agent_server.plugins.tools.snowflake_tools import client as client_mod
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeClient,
        SnowflakeCredentials,
    )

    monkeypatch.setattr(client_mod, "_is_session_expired", lambda _exc: True)

    class _Cursor:
        sfqid = "q"
        description = [("v",)]

        def __init__(self, fail: bool) -> None:
            self._fail = fail

        def execute(self, *_a: Any, **_k: Any) -> None:
            if self._fail:
                raise RuntimeError("session expired during execute")

        def fetchmany(self, n: int) -> list[tuple[Any, ...]]:
            return [(1,)]

        def fetchall(self) -> list[tuple[Any, ...]]:
            return [(1,)]

        def close(self) -> None:
            pass

    class _Conn:
        def __init__(self, cursors: list[_Cursor]) -> None:
            self._cursors = cursors

        def cursor(self) -> _Cursor:
            return self._cursors.pop(0)

        def close(self) -> None:
            pass

    conn1 = _Conn([_Cursor(fail=False), _Cursor(fail=True)])
    conn2 = _Conn([_Cursor(fail=False), _Cursor(fail=False)])
    factory = iter([conn1, conn2])

    creds = SnowflakeCredentials(account="a", user="u")
    client = SnowflakeClient(
        creds, connection_factory=lambda _c: next(factory), max_rows=10
    )
    result = client.execute("SELECT 1", params={"x": 1})
    assert result.row_count == 1


def test_snowflake_credentials_load_private_key_from_file(tmp_path: Path) -> None:
    """Load a private key from a file path."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeCredentials,
    )

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pem_path = tmp_path / "key.pem"
    pem_path.write_bytes(pem)

    creds = SnowflakeCredentials(account="a", user="u", private_key=str(pem_path))
    der = creds._loaded_private_key()
    assert isinstance(der, bytes)


def test_snowflake_client_propagates_non_session_error_on_cursor_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate a non-session error raised on cursor open."""
    from openbb_agent_server.plugins.tools.snowflake_tools import client as client_mod
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeClient,
        SnowflakeCredentials,
    )

    monkeypatch.setattr(client_mod, "_is_session_expired", lambda _exc: False)

    class _Cursor:
        def execute(self, *_a: Any, **_k: Any) -> None:
            pass

        def close(self) -> None:
            pass

    class _Conn:
        def __init__(self) -> None:
            self._calls = 0

        def cursor(self) -> _Cursor:
            self._calls += 1
            if self._calls > 1:
                raise RuntimeError("hard failure")
            return _Cursor()

        def close(self) -> None:
            pass

    creds = SnowflakeCredentials(account="a", user="u")
    client = SnowflakeClient(creds, connection_factory=lambda _c: _Conn())
    with pytest.raises(RuntimeError, match="hard failure"):
        client.execute("SELECT 1")


def test_snowflake_client_retries_execute_with_no_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry execute when params is None."""
    from openbb_agent_server.plugins.tools.snowflake_tools import client as client_mod
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeClient,
        SnowflakeCredentials,
    )

    monkeypatch.setattr(client_mod, "_is_session_expired", lambda _exc: True)

    class _Cursor:
        sfqid = "q"
        description = [("v",)]

        def __init__(self, fail: bool) -> None:
            self._fail = fail

        def execute(self, *_a: Any, **_k: Any) -> None:
            if self._fail:
                raise RuntimeError("session expired")

        def fetchmany(self, n: int) -> list[tuple[Any, ...]]:
            return [(1,)]

        def fetchall(self) -> list[tuple[Any, ...]]:
            return [(1,)]

        def close(self) -> None:
            pass

    class _Conn:
        def __init__(self, cursors: list[_Cursor]) -> None:
            self._cursors = cursors

        def cursor(self) -> _Cursor:
            return self._cursors.pop(0)

        def close(self) -> None:
            pass

    conn1 = _Conn([_Cursor(fail=False), _Cursor(fail=True)])
    conn2 = _Conn([_Cursor(fail=False), _Cursor(fail=False)])
    factory = iter([conn1, conn2])

    creds = SnowflakeCredentials(account="a", user="u")
    client = SnowflakeClient(
        creds, connection_factory=lambda _c: next(factory), max_rows=10
    )
    result = client.execute("SELECT 1")
    assert result.row_count == 1
