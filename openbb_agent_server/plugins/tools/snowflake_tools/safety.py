"""SQL safety: statement classification + read-only enforcement + LIMIT injection."""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

_PYFORMAT_NAMED = re.compile(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s")
_PYFORMAT_POSITIONAL = re.compile(r"%s")


def _normalize_for_parse(sql: str) -> str:
    """Substitute pyformat placeholders with sqlglot-friendly equivalents."""
    sql = _PYFORMAT_NAMED.sub(r":\1", sql)
    sql = _PYFORMAT_POSITIONAL.sub("?", sql)
    return sql


def _has_pyformat_placeholders(sql: str) -> bool:
    return bool(_PYFORMAT_NAMED.search(sql) or _PYFORMAT_POSITIONAL.search(sql))


READ_ONLY_TYPES: frozenset[type] = frozenset(
    {
        exp.Select,
        exp.Show,
        exp.Describe,
        exp.With,
        exp.Union,
        exp.Intersect,
        exp.Except,
        exp.Use,
    }
)

MUTATING_TYPES: frozenset[type] = frozenset(
    {
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.AlterColumn,
        exp.Merge,
        exp.TruncateTable,
        exp.Copy,
    }
)


class SnowflakeSafetyViolation(RuntimeError):
    """Raised when a statement is rejected by the safety classifier."""


def parse(sql: str) -> list[exp.Expression]:
    """Parse one or more Snowflake SQL statements into sqlglot expressions.

    Pyformat placeholders (``%(name)s`` and ``%s``) are first rewritten to
    sqlglot-friendly bind markers (``:name`` and ``?``) so the parser does
    not choke on driver-style parameters. Empty statements produced by
    trailing separators are dropped.

    Parameters
    ----------
    sql : str
        One or more semicolon-separated Snowflake SQL statements.

    Returns
    -------
    list[exp.Expression]
        The parsed top-level expressions, with ``None`` results removed.
    """
    normalised = _normalize_for_parse(sql)
    from typing import cast

    parsed = sqlglot.parse(normalised, read="snowflake")
    return cast(list[exp.Expression], [s for s in parsed if s is not None])


def is_read_only(sql: str) -> bool:
    """Return True iff every statement in ``sql`` is read-only.

    A statement counts as read-only when it parses to one of
    :data:`READ_ONLY_TYPES` (or is a bare ``Command`` whose verb is one of
    ``EXPLAIN``, ``USE``, ``CALL``, ``SHOW``, ``DESCRIBE``, ``DESC``) and is
    not one of :data:`MUTATING_TYPES`. Empty input is treated as not
    read-only.

    Parameters
    ----------
    sql : str
        One or more semicolon-separated Snowflake SQL statements.

    Returns
    -------
    bool
        True when there is at least one statement and every statement is
        classified as read-only; False otherwise.
    """
    statements = parse(sql)
    if not statements:
        return False
    for stmt in statements:
        if isinstance(stmt, exp.Command):
            head = (stmt.name or "").upper()
            if head in {"EXPLAIN", "USE", "CALL", "SHOW", "DESCRIBE", "DESC"}:
                continue
            return False
        if any(isinstance(stmt, t) for t in MUTATING_TYPES):
            return False
        if not any(isinstance(stmt, t) for t in READ_ONLY_TYPES):
            return False
    return True


def classify(sql: str) -> str:
    """Return a short human-readable label for the statement kind.

    Only the first parsed statement is inspected. Bare commands report their
    SQL verb (e.g. ``EXPLAIN``); other statements report the uppercased name
    of their sqlglot expression class (e.g. ``SELECT``, ``INSERT``).

    Parameters
    ----------
    sql : str
        One or more semicolon-separated Snowflake SQL statements.

    Returns
    -------
    str
        ``"EMPTY"`` when nothing parses, otherwise an uppercase label for
        the leading statement.
    """
    statements = parse(sql)
    if not statements:
        return "EMPTY"
    head = statements[0]
    if isinstance(head, exp.Command):
        return (head.name or "COMMAND").upper()
    return type(head).__name__.upper()


def enforce_read_only(sql: str) -> None:
    """Raise :class:`SnowflakeSafetyViolation` if ``sql`` is mutating.

    Parameters
    ----------
    sql : str
        One or more semicolon-separated Snowflake SQL statements.

    Raises
    ------
    SnowflakeSafetyViolation
        If any statement is not read-only, with the offending statement's
        classification embedded in the message.
    """
    if not is_read_only(sql):
        raise SnowflakeSafetyViolation(
            f"refused to run mutating statement: {classify(sql)}"
        )


def inject_limit(sql: str, max_rows: int) -> str:
    """Return ``sql`` with ``LIMIT max_rows`` applied to top-level SELECTs.

    The original SQL is returned unchanged when any of the following hold:
    ``max_rows`` is not positive; the SQL contains pyformat placeholders
    (rewriting would corrupt parameter binding); it does not parse to exactly
    one statement; the statement is not a ``SELECT``/``UNION``/``WITH``; or a
    top-level ``LIMIT`` is already present.

    Parameters
    ----------
    sql : str
        A single Snowflake SQL statement.
    max_rows : int
        The row cap to inject. Non-positive values disable injection.

    Returns
    -------
    str
        The rewritten statement serialized in the Snowflake dialect, or the
        original ``sql`` when no safe rewrite is possible.
    """
    if max_rows <= 0:
        return sql
    if _has_pyformat_placeholders(sql):
        return sql
    statements = parse(sql)
    if len(statements) != 1:
        return sql
    [stmt] = statements
    rewriteable = isinstance(stmt, (exp.Select, exp.Union, exp.With))
    if not rewriteable:
        return sql
    if stmt.args.get("limit") is not None:
        return sql
    stmt = stmt.copy()
    stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    return stmt.sql(dialect="snowflake")
