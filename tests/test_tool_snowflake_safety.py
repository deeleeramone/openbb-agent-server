"""SQL safety classifier tests (sqlglot-backed)."""

from __future__ import annotations

import pytest

from openbb_agent_server.plugins.tools.snowflake_tools.safety import (
    SnowflakeSafetyViolation,
    classify,
    enforce_read_only,
    inject_limit,
    is_read_only,
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT * FROM t WHERE a = 1",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT 1 UNION SELECT 2",
        "SHOW TABLES",
        "SHOW DATABASES",
        "DESCRIBE my_table",
        "DESC my_view",
        "EXPLAIN SELECT 1",
        "USE WAREHOUSE WH",
        "USE DATABASE DB",
        "USE SCHEMA SCH",
        "USE ROLE R",
        "CALL my_proc()",
    ],
)
def test_read_only_statements_pass(sql: str) -> None:
    assert is_read_only(sql)
    enforce_read_only(sql)  # no raise


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "CREATE TABLE t (a INT)",
        "ALTER TABLE t ADD COLUMN b INT",
        "MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN UPDATE SET t.b = s.b",
        "TRUNCATE TABLE t",
        "COPY INTO t FROM @stage",
    ],
)
def test_mutating_statements_rejected(sql: str) -> None:
    assert not is_read_only(sql)
    with pytest.raises(SnowflakeSafetyViolation):
        enforce_read_only(sql)


def test_empty_sql_is_not_read_only() -> None:
    assert not is_read_only("")


def test_classify_returns_short_label() -> None:
    assert classify("SELECT 1") == "SELECT"
    assert classify("SHOW TABLES") == "SHOW"
    assert classify("DROP TABLE x") == "DROP"
    assert classify("USE WAREHOUSE x") == "USE"


def test_inject_limit_appends_when_missing() -> None:
    out = inject_limit("SELECT * FROM t", 5)
    assert "LIMIT 5" in out


def test_inject_limit_preserves_existing_limit() -> None:
    out = inject_limit("SELECT * FROM t LIMIT 9", 5)
    assert "LIMIT 9" in out
    assert "LIMIT 5" not in out


def test_inject_limit_skips_show() -> None:
    sql = "SHOW TABLES"
    assert inject_limit(sql, 5) == sql


def test_inject_limit_skips_describe() -> None:
    sql = "DESCRIBE my_table"
    assert inject_limit(sql, 5) == sql


def test_inject_limit_skips_multi_statement() -> None:
    sql = "SELECT 1; SELECT 2"
    assert inject_limit(sql, 5) == sql


def test_inject_limit_zero_or_negative_passthrough() -> None:
    assert inject_limit("SELECT 1", 0) == "SELECT 1"
    assert inject_limit("SELECT 1", -1) == "SELECT 1"


def test_inject_limit_handles_with() -> None:
    out = inject_limit("WITH x AS (SELECT 1) SELECT * FROM x", 7)
    assert "LIMIT 7" in out


def test_safety_classify_returns_empty_for_blank_sql() -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools.safety import classify

    assert classify("   ") == "EMPTY"


def test_safety_is_read_only_accepts_show_describe_etc() -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools.safety import is_read_only

    assert is_read_only("SHOW DATABASES")
    assert is_read_only("DESCRIBE DB.S.T")
    assert is_read_only("EXPLAIN SELECT 1")
    assert is_read_only("USE DB.S")


def test_safety_is_read_only_rejects_with_returning() -> None:
    """Reject a statement with no read-only base type."""
    from openbb_agent_server.plugins.tools.snowflake_tools.safety import is_read_only

    assert is_read_only("COPY INTO @stage FROM tbl") is False


def test_safety_is_read_only_rejects_unsafe_command() -> None:
    """Reject a Command-shaped statement outside the allow-list."""
    from openbb_agent_server.plugins.tools.snowflake_tools.safety import is_read_only

    assert is_read_only("GRANT ROLE analyst TO USER bob") is False


def test_safety_is_read_only_rejects_neither_mutating_nor_read_only() -> None:
    """Reject a statement in neither the mutating nor read-only type set."""
    from openbb_agent_server.plugins.tools.snowflake_tools.safety import is_read_only

    assert is_read_only("SET QUERY_TAG = 'x'") is False
