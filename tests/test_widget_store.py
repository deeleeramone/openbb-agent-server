"""WidgetDataStore + parse_widget_data_messages tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from openbb_agent_server.persistence import models as m
from openbb_agent_server.runtime.principal import UserPrincipal
from openbb_agent_server.runtime.widget_store import (
    WidgetDataStore,
    _extract_columns,
    _extract_rows,
    _extract_rows_from_dict,
    _row_text,
    _slugify_table_name,
    parse_widget_data_messages,
)


def _principal(user_id: str = "u") -> UserPrincipal:
    return UserPrincipal(user_id=user_id)


@pytest_asyncio.fixture
async def store() -> AsyncIterator[WidgetDataStore]:
    s = WidgetDataStore("sqlite+aiosqlite:///:memory:")
    async with s._engine.begin() as conn:
        await conn.run_sync(m.Base.metadata.create_all)
    yield s
    await s._engine.dispose()


def test_slugify_empty_yields_default() -> None:
    assert _slugify_table_name("") == "widget"


def test_slugify_whitespace_yields_default() -> None:
    assert _slugify_table_name("   ") == "widget"


def test_slugify_alphanumeric_passes_through() -> None:
    assert _slugify_table_name("Foo_Bar 9") == "foo_bar_9"


def test_slugify_strips_unsafe_chars() -> None:
    assert _slugify_table_name('a"b/c.d') == "a_b_c_d"


def test_slugify_prefixes_leading_digit() -> None:
    assert _slugify_table_name("9foo") == "_9foo"


def test_slugify_strips_leading_trailing_underscores() -> None:
    assert _slugify_table_name("---abc---") == "abc"


def test_row_text_joins_keys() -> None:
    assert _row_text({"a": 1, "b": "x"}) == "a: 1 | b: x"


def test_row_text_skips_none() -> None:
    assert _row_text({"a": 1, "b": None, "c": "ok"}) == "a: 1 | c: ok"


def test_row_text_empty_dict() -> None:
    assert _row_text({}) == ""


@pytest.mark.asyncio
async def test_record_stores_basic_row(store: WidgetDataStore) -> None:
    rid = await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="market",
        input_args={"sym": "AAPL"},
        rows=[{"a": 1}, {"a": 2}],
        columns=["a"],
    )
    assert rid > 0
    entries = await store.list_entries(principal=_principal(), conversation_id="c1")
    assert len(entries) == 1
    assert entries[0]["row_count"] == 2
    assert entries[0]["widget_name"] == "Prices"
    assert entries[0]["ingested_at"]


@pytest.mark.asyncio
async def test_list_entries_isolated_per_user(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal("a"),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    await store.record(
        principal=_principal("b"),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"x": 2}],
        columns=["x"],
    )

    a_entries = await store.list_entries(
        principal=_principal("a"), conversation_id="c1"
    )
    b_entries = await store.list_entries(
        principal=_principal("b"), conversation_id="c1"
    )
    assert len(a_entries) == 1
    assert len(b_entries) == 1


@pytest.mark.asyncio
async def test_list_entries_empty(store: WidgetDataStore) -> None:
    out = await store.list_entries(principal=_principal(), conversation_id="cx")
    assert out == []


@pytest.mark.asyncio
async def test_list_entries_spans_conversations_for_user(
    store: WidgetDataStore,
) -> None:
    """Return every ingest the user has when conversation_id is None."""
    await store.record(
        principal=_principal("u"),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    await store.record(
        principal=_principal("u"),
        conversation_id="c2",
        widget_uuid="w-b",
        widget_name="B",
        origin="o",
        input_args={},
        rows=[{"x": 2}],
        columns=["x"],
    )
    await store.record(
        principal=_principal("other"),
        conversation_id="c1",
        widget_uuid="w-c",
        widget_name="C",
        origin="o",
        input_args={},
        rows=[{"x": 3}],
        columns=["x"],
    )
    out = await store.list_entries(principal=_principal("u"), conversation_id=None)
    uuids = sorted(e["widget_uuid"] for e in out)
    assert uuids == ["w-a", "w-b"]


@pytest.mark.asyncio
async def test_list_entries_default_spans_conversations(
    store: WidgetDataStore,
) -> None:
    """Default to cross-convo when no conversation_id kwarg is given."""
    await store.record(
        principal=_principal("u"),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    await store.record(
        principal=_principal("u"),
        conversation_id="c2",
        widget_uuid="w-b",
        widget_name="B",
        origin="o",
        input_args={},
        rows=[{"x": 2}],
        columns=["x"],
    )
    out = await store.list_entries(principal=_principal("u"))
    assert {e["widget_uuid"] for e in out} == {"w-a", "w-b"}


@pytest.mark.asyncio
async def test_read_latest_returns_none_when_missing(store: WidgetDataStore) -> None:
    assert (
        await store.read_latest(principal=_principal(), conversation_id="empty") is None
    )


@pytest.mark.asyncio
async def test_read_latest_cross_conversation(store: WidgetDataStore) -> None:
    """Find data ingested in any prior conversation when conversation_id is None."""
    await store.record(
        principal=_principal("u"),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"x": "old"}],
        columns=["x"],
    )
    out = await store.read_latest(
        principal=_principal("u"),
        conversation_id=None,
        widget_uuid="w-1",
    )
    assert out is not None
    assert out["rows"] == [{"x": "old"}]


@pytest.mark.asyncio
async def test_read_latest_cross_conversation_still_isolates_users(
    store: WidgetDataStore,
) -> None:
    """Cross-conversation lookup must NEVER leak across users."""
    await store.record(
        principal=_principal("alice"),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"x": "alice-data"}],
        columns=["x"],
    )
    out = await store.read_latest(
        principal=_principal("bob"),
        conversation_id=None,
        widget_uuid="w-1",
    )
    assert out is None


@pytest.mark.asyncio
async def test_read_latest_returns_newest(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={"v": 1},
        rows=[{"x": "first"}],
        columns=["x"],
    )
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={"v": 2},
        rows=[{"x": "second"}],
        columns=["x"],
    )
    out = await store.read_latest(principal=_principal(), conversation_id="c1")
    assert out is not None
    assert out["rows"] == [{"x": "second"}]


@pytest.mark.asyncio
async def test_read_latest_filters_by_widget_uuid(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"x": "a"}],
        columns=["x"],
    )
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-b",
        widget_name="B",
        origin="o",
        input_args={},
        rows=[{"x": "b"}],
        columns=["x"],
    )
    out = await store.read_latest(
        principal=_principal(), conversation_id="c1", widget_uuid="w-a"
    )
    assert out is not None
    assert out["rows"] == [{"x": "a"}]


@pytest.mark.asyncio
async def test_read_latest_widget_name_falls_through_search(
    store: WidgetDataStore,
) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-b",
        widget_name="B",
        origin="o",
        input_args={},
        rows=[{"x": 2}],
        columns=["x"],
    )
    out = await store.read_latest(
        principal=_principal(), conversation_id="c1", widget_name="A"
    )
    assert out is not None
    assert out["widget_name"] == "A"


@pytest.mark.asyncio
async def test_read_latest_widget_name_fallthrough_returns_none(
    store: WidgetDataStore,
) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    out = await store.read_latest(
        principal=_principal(), conversation_id="c1", widget_name="Nonexistent"
    )
    assert out is None


@pytest.mark.asyncio
async def test_read_latest_max_rows_truncates(store: WidgetDataStore) -> None:
    rows = [{"x": i} for i in range(10)]
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={},
        rows=rows,
        columns=["x"],
    )
    out = await store.read_latest(
        principal=_principal(), conversation_id="c1", max_rows=3
    )
    assert out is not None
    assert len(out["rows"]) == 3


@pytest.mark.asyncio
async def test_search_empty_returns_empty(store: WidgetDataStore) -> None:
    out = await store.search(principal=_principal(), conversation_id="cx", query="foo")
    assert out == []


@pytest.mark.asyncio
async def test_search_substring_fallback_when_no_embeddings(
    store: WidgetDataStore,
) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"city": "Paris"}, {"city": "Rome"}, {"city": "Berlin"}],
        columns=["city"],
    )
    out = await store.search(
        principal=_principal(), conversation_id="c1", query="paris"
    )
    assert len(out) == 1
    assert out[0]["row"] == {"city": "Paris"}


@pytest.mark.asyncio
async def test_search_filters_by_widget_uuid(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"v": "paris"}],
        columns=["v"],
    )
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-b",
        widget_name="B",
        origin="o",
        input_args={},
        rows=[{"v": "paris"}],
        columns=["v"],
    )
    out = await store.search(
        principal=_principal(),
        conversation_id="c1",
        query="paris",
        widget_uuid="w-a",
    )
    assert len(out) == 1


@pytest.mark.asyncio
async def test_search_returns_empty_for_blank_query(
    store: WidgetDataStore,
) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    out = await store.search(principal=_principal(), conversation_id="c1", query="")
    assert out == []


@pytest.mark.asyncio
async def test_substring_search_spans_conversations_when_unscoped(
    store: WidgetDataStore,
) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="old-convo",
        widget_uuid="w",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"city": "Paris"}],
        columns=["city"],
    )
    out = await store.search(
        principal=_principal(), conversation_id=None, query="paris"
    )
    assert len(out) == 1
    assert out[0]["row"] == {"city": "Paris"}


@pytest.mark.asyncio
async def test_substring_search_returns_only_k_results(
    store: WidgetDataStore,
) -> None:
    """Exit the substring path early after collecting k matches."""
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w",
        widget_name="W",
        origin="o",
        input_args={},
        rows=[{"v": "match"}] * 10,
        columns=["v"],
    )
    out = await store.search(
        principal=_principal(), conversation_id="c1", query="match", k=3
    )
    assert len(out) == 3


@pytest.mark.asyncio
async def test_schema_spans_conversations_when_unscoped(
    store: WidgetDataStore,
) -> None:
    """Expose widgets from every conversation when conversation_id is None."""
    await store.record(
        principal=_principal("u"),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="Prices",
        origin="o",
        input_args={},
        rows=[{"a": 1}],
        columns=["a"],
    )
    await store.record(
        principal=_principal("u"),
        conversation_id="c2",
        widget_uuid="w-b",
        widget_name="Volumes",
        origin="o",
        input_args={},
        rows=[{"b": 9}],
        columns=["b"],
    )
    sch = await store.schema(principal=_principal("u"), conversation_id=None)
    tables = {e["table"] for e in sch}
    assert tables == {"prices", "volumes"}


@pytest.mark.asyncio
async def test_query_spans_conversations_when_unscoped(
    store: WidgetDataStore,
) -> None:
    """See widgets ingested in other conversations from SQL queries."""
    await store.record(
        principal=_principal("u"),
        conversation_id="c-old",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="o",
        input_args={},
        rows=[{"sym": "AAPL"}, {"sym": "GOOG"}],
        columns=["sym"],
    )
    out = await store.query(
        principal=_principal("u"),
        conversation_id=None,
        sql='SELECT "sym" FROM "prices"',
    )
    assert out["table_count"] == 1
    assert {r["sym"] for r in out["rows"]} == {"AAPL", "GOOG"}


@pytest.mark.asyncio
async def test_query_unscoped_still_isolates_users(store: WidgetDataStore) -> None:
    """Isolate users on an unscoped cross-conversation query."""
    await store.record(
        principal=_principal("alice"),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="o",
        input_args={},
        rows=[{"sym": "AAPL"}],
        columns=["sym"],
    )
    import sqlalchemy.exc

    with pytest.raises(sqlalchemy.exc.OperationalError):
        await store.query(
            principal=_principal("bob"),
            conversation_id=None,
            sql='SELECT "sym" FROM "prices"',
        )


@pytest.mark.asyncio
async def test_schema_dedupes_by_slug(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="o",
        input_args={},
        rows=[{"a": 1}],
        columns=["a"],
    )
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="o",
        input_args={},
        rows=[{"a": 2}, {"a": 3}],
        columns=["a"],
    )
    sch = await store.schema(principal=_principal(), conversation_id="c1")
    assert len(sch) == 1
    assert sch[0]["table"] == "prices"
    assert sch[0]["row_count"] == 2


@pytest.mark.asyncio
async def test_schema_falls_back_to_uuid_and_id(store: WidgetDataStore) -> None:
    rid = await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name=None,
        origin=None,
        input_args={},
        rows=[],
        columns=[],
    )
    sch = await store.schema(principal=_principal(), conversation_id="c1")
    assert len(sch) == 1
    assert sch[0]["table"] == "w_1"
    assert sch[0]["row_count"] == 0
    assert rid > 0


@pytest.mark.asyncio
async def test_schema_uses_widget_id_when_uuid_blank(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="",
        widget_name=None,
        origin=None,
        input_args={},
        rows=[],
        columns=[],
    )
    sch = await store.schema(principal=_principal(), conversation_id="c1")
    assert sch[0]["table"].startswith("widget_")


@pytest.mark.asyncio
async def test_validate_query_sql_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        WidgetDataStore._validate_query_sql("   ")


@pytest.mark.asyncio
async def test_validate_query_sql_rejects_non_select() -> None:
    with pytest.raises(ValueError, match="SELECT / WITH"):
        WidgetDataStore._validate_query_sql("INSERT INTO foo VALUES (1)")


def test_validate_query_sql_accepts_select() -> None:
    assert WidgetDataStore._validate_query_sql("SELECT 1") == "SELECT 1"


def test_validate_query_sql_strips_trailing_semicolon() -> None:
    assert WidgetDataStore._validate_query_sql("SELECT 1;") == "SELECT 1"


def test_validate_query_sql_accepts_with() -> None:
    sql = "WITH x AS (SELECT 1) SELECT * FROM x"
    assert WidgetDataStore._validate_query_sql(sql) == sql


def test_resolve_view_columns_uses_explicit_columns() -> None:
    ing = m.WidgetData(
        user_id="u",
        conversation_id="c",
        widget_uuid="w",
        widget_name="n",
        origin="o",
        input_args={},
        columns=["a", "b"],
        rows=[{"a": 1, "c": 9}],
    )
    assert WidgetDataStore._resolve_view_columns(ing) == ["a", "b"]


def test_resolve_view_columns_infers_from_rows_when_no_columns() -> None:
    ing = m.WidgetData(
        user_id="u",
        conversation_id="c",
        widget_uuid="w",
        widget_name="n",
        origin="o",
        input_args={},
        columns=None,
        rows=[{"a": 1, "b": 2}],
    )
    assert WidgetDataStore._resolve_view_columns(ing) == ["a", "b"]


def test_resolve_view_columns_empty_when_no_rows_no_columns() -> None:
    ing = m.WidgetData(
        user_id="u",
        conversation_id="c",
        widget_uuid="w",
        widget_name="n",
        origin="o",
        input_args={},
        columns=None,
        rows=[],
    )
    assert WidgetDataStore._resolve_view_columns(ing) == []


def test_resolve_view_columns_skips_non_dict_rows() -> None:
    ing = m.WidgetData(
        user_id="u",
        conversation_id="c",
        widget_uuid="w",
        widget_name="n",
        origin="o",
        input_args={},
        columns=None,
        rows=[{"x": 1}, "not a dict", {"y": 2}],
    )
    out = WidgetDataStore._resolve_view_columns(ing)
    assert out == ["x", "y"]


def test_build_view_sql_empty_columns_makes_empty_view() -> None:
    sql = WidgetDataStore._build_view_sql("foo", [], 1)
    assert 'CREATE TEMP VIEW "foo"' in sql
    assert "WHERE 0" in sql


def test_build_view_sql_quotes_column_names() -> None:
    sql = WidgetDataStore._build_view_sql("t", ['a"b', "c"], 9)
    assert 'AS "a""b"' in sql
    assert "wd.id = 9" in sql


@pytest.mark.asyncio
async def test_query_rejects_postgres_dialect() -> None:
    store = WidgetDataStore("sqlite+aiosqlite:///:memory:")

    class _FakeDialect:
        name = "postgresql"

    store._engine.dialect = _FakeDialect()  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="not supported"):
            await store.query(
                principal=_principal(),
                conversation_id="c1",
                sql="SELECT 1",
            )
    finally:
        await store._engine.dispose()


@pytest.mark.asyncio
async def test_query_runs_basic_select(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="Prices",
        origin="o",
        input_args={},
        rows=[{"sym": "AAPL", "v": 100}, {"sym": "GOOG", "v": 200}],
        columns=["sym", "v"],
    )
    out = await store.query(
        principal=_principal(),
        conversation_id="c1",
        sql='SELECT "sym" FROM "prices" ORDER BY "v" DESC',
    )
    assert out["columns"] == ["sym"]
    assert out["rows"] == [{"sym": "GOOG"}, {"sym": "AAPL"}]
    assert out["truncated"] is False
    assert out["table_count"] == 1


@pytest.mark.asyncio
async def test_query_truncates_to_max_rows(store: WidgetDataStore) -> None:
    rows = [{"i": i} for i in range(20)]
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-1",
        widget_name="Nums",
        origin="o",
        input_args={},
        rows=rows,
        columns=["i"],
    )
    out = await store.query(
        principal=_principal(),
        conversation_id="c1",
        sql='SELECT "i" FROM "nums"',
        max_rows=5,
    )
    assert len(out["rows"]) == 5
    assert out["truncated"] is True


@pytest.mark.asyncio
async def test_query_creates_view_per_widget(store: WidgetDataStore) -> None:
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-a",
        widget_name="A",
        origin="o",
        input_args={},
        rows=[{"x": 1}],
        columns=["x"],
    )
    await store.record(
        principal=_principal(),
        conversation_id="c1",
        widget_uuid="w-b",
        widget_name="B",
        origin="o",
        input_args={},
        rows=[{"y": 9}],
        columns=["y"],
    )
    out = await store.query(
        principal=_principal(),
        conversation_id="c1",
        sql='SELECT "a"."x" AS x, "b"."y" AS y FROM "a", "b"',
    )
    assert out["table_count"] == 2
    assert out["rows"] == [{"x": 1, "y": 9}]


def test_extract_rows_handles_none() -> None:
    assert _extract_rows(None) == []


def test_extract_rows_handles_scalars() -> None:
    assert _extract_rows(42) == []
    assert _extract_rows(1.5) == []
    assert _extract_rows(True) == []


def test_extract_rows_recurses_list() -> None:
    assert _extract_rows([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]


def test_extract_rows_parses_json_string() -> None:
    assert _extract_rows('[{"a": 1}]') == [{"a": 1}]


def test_extract_rows_unparseable_string_returns_empty() -> None:
    assert _extract_rows("not json") == []


def test_extract_rows_unknown_type_returns_empty() -> None:
    class _Mystery:
        pass

    assert _extract_rows(_Mystery()) == []


def test_extract_rows_from_items_key() -> None:
    payload = {"items": [{"x": 1}, {"x": 2}]}
    assert _extract_rows(payload) == [{"x": 1}, {"x": 2}]


def test_extract_rows_from_content_key() -> None:
    payload = {"content": [{"x": 1}]}
    assert _extract_rows(payload) == [{"x": 1}]


def test_extract_rows_scalar_dict_becomes_single_row() -> None:
    payload = {"sym": "AAPL", "v": 100}
    assert _extract_rows(payload) == [{"sym": "AAPL", "v": 100}]


def test_extract_rows_nested_dict_returns_empty() -> None:
    payload = {"meta": {"nested": "value"}, "x": 1}
    assert _extract_rows(payload) == []


def test_extract_rows_from_dict_passes_through_to_helper() -> None:
    assert _extract_rows_from_dict({"items": [{"x": 1}]}) == [{"x": 1}]


def test_extract_columns_empty_rows() -> None:
    assert _extract_columns([]) is None


def test_extract_columns_unions_keys_in_order() -> None:
    assert _extract_columns([{"a": 1, "b": 2}, {"b": 3, "c": 4}]) == ["a", "b", "c"]


def test_extract_columns_skips_non_dict_rows() -> None:
    assert _extract_columns([{"a": 1}, "bad", {"b": 2}]) == ["a", "b"]


def test_extract_columns_all_non_dict_rows_returns_none() -> None:
    assert _extract_columns([None, "x"]) is None  # type: ignore[arg-type]


class _AIMessage:
    def __init__(self, content: str) -> None:
        self.role = "ai"
        self.content = content


class _ToolMessage:
    def __init__(self, data: Any) -> None:
        self.role = "tool"
        self.data = data


class _HumanMessage:
    def __init__(self, content: str) -> None:
        self.role = "human"
        self.content = content


class _StructuredAIMessage:
    """AI tool-call envelope using structured function fields."""

    def __init__(self, function: str, input_arguments: dict[str, Any]) -> None:
        self.role = "ai"
        self.content = None
        self.function = function
        self.input_arguments = input_arguments


class _ToolMessageWithArgs:
    """Tool message carrying its own function and input_arguments."""

    def __init__(
        self,
        data: Any,
        *,
        function: str | None = None,
        input_arguments: dict[str, Any] | None = None,
    ) -> None:
        self.role = "tool"
        self.data = data
        self.function = function
        self.input_arguments = input_arguments or {}


def test_parse_widget_data_messages_reads_structured_ai_envelope() -> None:
    """Read a structured AI envelope and flow widget_uuid through."""
    ai = _StructuredAIMessage(
        function="get_widget_data",
        input_arguments={
            "data_sources": [
                {
                    "widget_uuid": "uuid-1",
                    "origin": "blackrock",
                    "id": "blk_alloc_currency",
                    "input_args": {"ticker": "IEFA"},
                }
            ]
        },
    )
    tool = _ToolMessage([{"items": [{"label": "USD", "weight_pct": 0.5}]}])
    out = parse_widget_data_messages([ai, tool])
    assert len(out) == 1
    assert out[0]["widget_uuid"] == "uuid-1"
    assert out[0]["widget_name"] == "blk_alloc_currency"
    assert out[0]["rows"] == [{"label": "USD", "weight_pct": 0.5}]


def test_parse_widget_data_messages_reads_data_sources_from_tool_msg() -> None:
    """Read data_sources from the tool message itself."""
    tool = _ToolMessageWithArgs(
        [{"items": [{"x": 1}]}],
        function="get_widget_data",
        input_arguments={
            "data_sources": [{"widget_uuid": "uuid-2", "id": "blk_alloc_sector"}]
        },
    )
    out = parse_widget_data_messages([tool])
    assert len(out) == 1
    assert out[0]["widget_uuid"] == "uuid-2"
    assert out[0]["widget_name"] == "blk_alloc_sector"


def test_parse_widget_data_messages_synthesises_from_widget_ids() -> None:
    """Synthesise minimal sources from a forwarded widget_ids list."""
    ai = _StructuredAIMessage(
        function="get_widget_data",
        input_arguments={"widget_ids": ["uuid-a", "uuid-b"]},
    )
    tool = _ToolMessage([{"items": [{"x": 1}]}, {"items": [{"y": 2}]}])
    out = parse_widget_data_messages([ai, tool])
    assert [i["widget_uuid"] for i in out] == ["uuid-a", "uuid-b"]
    assert out[0]["rows"] == [{"x": 1}]
    assert out[1]["rows"] == [{"y": 2}]


def test_parse_widget_data_messages_pairs_ai_envelope_with_tool_payload() -> None:
    envelope = {
        "function": "get_widget_data",
        "input_arguments": {
            "data_sources": [
                {
                    "widget_uuid": "w-1",
                    "origin": "market",
                    "id": "prices",
                    "input_args": {"sym": "AAPL"},
                }
            ]
        },
    }
    tool_payload = [{"items": [{"sym": "AAPL", "v": 100}]}]
    out = parse_widget_data_messages(
        [
            _AIMessage(json.dumps(envelope)),
            _ToolMessage(tool_payload),
        ]
    )
    assert len(out) == 1
    [ingest] = out
    assert ingest["widget_uuid"] == "w-1"
    assert ingest["widget_name"] == "prices"
    assert ingest["origin"] == "market"
    assert ingest["input_args"] == {"sym": "AAPL"}
    assert ingest["rows"] == [{"sym": "AAPL", "v": 100}]
    assert ingest["columns"] == ["sym", "v"]


def test_parse_widget_data_messages_handles_multiple_data_sources() -> None:
    envelope = {
        "function": "get_widget_data",
        "input_arguments": {
            "data_sources": [
                {"widget_uuid": "w-a", "id": "a", "input_args": {}},
                {"widget_uuid": "w-b", "id": "b", "input_args": {}},
            ]
        },
    }
    payload = [
        {"items": [{"x": 1}]},
        {"items": [{"x": 2}]},
    ]
    out = parse_widget_data_messages(
        [_AIMessage(json.dumps(envelope)), _ToolMessage(payload)]
    )
    assert [i["widget_name"] for i in out] == ["a", "b"]


def test_parse_widget_data_messages_skips_when_no_envelope() -> None:
    out = parse_widget_data_messages([_ToolMessage([{"items": [{"x": 1}]}])])
    assert out == []


def test_parse_widget_data_messages_human_role_resets_envelope() -> None:
    envelope = {"function": "get_widget_data", "input_arguments": {"data_sources": []}}
    out = parse_widget_data_messages(
        [
            _AIMessage(json.dumps(envelope)),
            _HumanMessage("excuse me"),
            _ToolMessage([{"items": [{"x": 1}]}]),
        ]
    )
    assert out == []


def test_parse_widget_data_messages_ignores_non_json_ai_content() -> None:
    out = parse_widget_data_messages(
        [
            _AIMessage("just chatting"),
            _ToolMessage([{"items": [{"x": 1}]}]),
        ]
    )
    assert out == []


def test_parse_widget_data_messages_ignores_unparseable_json() -> None:
    out = parse_widget_data_messages(
        [
            _AIMessage("{ not valid"),
            _ToolMessage([{"items": [{"x": 1}]}]),
        ]
    )
    assert out == []


def test_parse_widget_data_messages_ignores_wrong_function() -> None:
    bad_envelope = {"function": "something_else", "input_arguments": {}}
    out = parse_widget_data_messages(
        [
            _AIMessage(json.dumps(bad_envelope)),
            _ToolMessage([{"items": [{"x": 1}]}]),
        ]
    )
    assert out == []


def test_parse_widget_data_messages_skips_non_dict_data_sources() -> None:
    envelope = {
        "function": "get_widget_data",
        "input_arguments": {"data_sources": ["not a dict", {"widget_uuid": "ok"}]},
    }
    out = parse_widget_data_messages(
        [
            _AIMessage(json.dumps(envelope)),
            _ToolMessage([{"items": []}, {"items": [{"x": 1}]}]),
        ]
    )
    assert len(out) == 1
    assert out[0]["widget_uuid"] == "ok"


def test_parse_widget_data_messages_pads_when_payload_shorter() -> None:
    envelope = {
        "function": "get_widget_data",
        "input_arguments": {
            "data_sources": [
                {"widget_uuid": "w-1", "id": "a", "input_args": {}},
                {"widget_uuid": "w-2", "id": "b", "input_args": {}},
            ]
        },
    }
    out = parse_widget_data_messages(
        [
            _AIMessage(json.dumps(envelope)),
            _ToolMessage([{"items": [{"x": 1}]}]),
        ]
    )
    # Second data source has no payload — should still produce a record (with empty rows).
    assert len(out) == 2
    assert out[0]["rows"] == [{"x": 1}]
    assert out[1]["rows"] == []


def test_parse_widget_data_messages_normalises_non_list_data() -> None:
    envelope = {
        "function": "get_widget_data",
        "input_arguments": {
            "data_sources": [{"widget_uuid": "w-1", "id": "a", "input_args": {}}]
        },
    }
    # data is a single dict, not a list — gets wrapped.
    out = parse_widget_data_messages(
        [
            _AIMessage(json.dumps(envelope)),
            _ToolMessage({"items": [{"x": 1}]}),
        ]
    )
    assert len(out) == 1
    assert out[0]["rows"] == [{"x": 1}]


def test_parse_widget_data_messages_skips_role_none() -> None:
    """Reset the envelope for a message without a recognisable role."""

    class _Bare:
        role = None
        content = "x"

    envelope = {"function": "get_widget_data", "input_arguments": {"data_sources": []}}
    out = parse_widget_data_messages(
        [
            _AIMessage(json.dumps(envelope)),
            _Bare(),
            _ToolMessage([{"items": [{"x": 1}]}]),
        ]
    )
    assert out == []


def test_apply_sqlite_pragmas_skips_non_sqlite_url() -> None:
    """Return early from _apply_sqlite_pragmas for non-SQLite URLs."""
    from openbb_agent_server.runtime.widget_store import _apply_sqlite_pragmas

    class _BoomEngine:
        @property
        def sync_engine(self) -> Any:
            raise AssertionError("engine must not be touched for non-sqlite URLs")

    _apply_sqlite_pragmas(_BoomEngine(), "postgresql+asyncpg://host/db")
