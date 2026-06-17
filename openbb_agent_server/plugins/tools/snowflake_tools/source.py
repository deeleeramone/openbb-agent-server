"""``snowflake`` tool source."""

from __future__ import annotations

import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema" in ".*" shadows an attribute in parent "BaseModel"',
)

import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from openbb_agent_server.plugins.tools.snowflake_tools import cortex as cortex_lib
from openbb_agent_server.plugins.tools.snowflake_tools.client import (
    QueryResult,
    SnowflakeClient,
    SnowflakeCredentials,
)
from openbb_agent_server.runtime import emit
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.snowflake")


class _QueryArgs(BaseModel):
    sql: str = Field(description="A read-only SQL query (SELECT/WITH/SHOW/...).")
    max_rows: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        description="Cap on rows returned. LIMIT is appended automatically.",
    )


class _ListSchemasArgs(BaseModel):
    database: str = Field(description="Database name.")


class _ListTablesArgs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    database: str
    schema: str = Field(description="Schema name within the database.")


class _DescribeArgs(BaseModel):
    object_path: str = Field(
        description="Fully-qualified table/view (e.g. DB.SCHEMA.TABLE).",
    )


class _TableInfoArgs(BaseModel):
    table: str = Field(
        description="Fully-qualified table/view (DB.SCHEMA.TABLE).",
    )


class _TableSampleArgs(BaseModel):
    table: str = Field(description="Fully-qualified table to sample from.")
    limit: int = Field(default=5, ge=1, le=200)


class _MultiTableArgs(BaseModel):
    tables: list[str] = Field(
        description="List of fully-qualified tables/views.",
        min_length=1,
        max_length=50,
    )


class _CatalogSearchArgs(BaseModel):
    pattern: str = Field(
        description=(
            "ILIKE pattern matched against table_catalog.table_schema.table_name "
            "and column names. Use % wildcards. Case-insensitive."
        )
    )
    limit: int = Field(default=50, ge=1, le=500)


class _ExplainArgs(BaseModel):
    sql: str = Field(description="SQL to EXPLAIN (read-only).")


class _HistoryArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)


class _CortexCompleteArgs(BaseModel):
    prompt: str
    model: str = Field(default="claude-3-5-sonnet")
    options: dict[str, Any] | None = None


class _CortexSummarizeArgs(BaseModel):
    text: str


class _CortexSentimentArgs(BaseModel):
    text: str


class _CortexTranslateArgs(BaseModel):
    text: str
    target_language: str = Field(description="ISO code (e.g. 'en', 'es', 'fr').")
    source_language: str = ""


class _CortexClassifyArgs(BaseModel):
    text: str
    categories: list[str] = Field(min_length=2)


class _CortexExtractArgs(BaseModel):
    question: str
    context: str


class _CortexEmbedArgs(BaseModel):
    text: str
    model: str = Field(default="snowflake-arctic-embed-l-v2.0")
    dim: int = Field(default=1024, description="768 or 1024")


class _CortexSearchArgs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    database: str
    schema: str
    service: str
    query: str
    columns: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=200)
    filter: dict[str, Any] | None = None


class _CortexAnalystArgs(BaseModel):
    messages: list[dict[str, Any]] = Field(
        description=(
            "OpenAI-style chat history: list of {role, content[]}. The "
            "last message is the user's question."
        )
    )
    semantic_model: str | None = None
    semantic_view: str | None = None


def _result_to_payload(result: QueryResult) -> dict[str, Any]:
    """Render a QueryResult as a tool-output dict and table artifact."""
    if result.rows and result.columns:
        emit.table_artifact(
            columns=list(result.columns),
            rows=[list(r) for r in result.rows],
            name=f"Snowflake result ({result.row_count} rows)",
            description=(
                f"{result.statement_kind} • query_id={result.query_id} • "
                f"{result.elapsed_ms or 0} ms"
                + (" • truncated" if result.truncated else "")
            ),
        )
    emit.reasoning_step(
        f"snowflake_query ok: {result.row_count} rows in {result.elapsed_ms or 0}ms",
        event_type="SUCCESS",
        query_id=result.query_id,
        statement_kind=result.statement_kind,
        truncated=result.truncated,
    )
    return {
        "columns": list(result.columns),
        "rows": [list(r) for r in result.rows],
        "row_count": result.row_count,
        "truncated": result.truncated,
        "query_id": result.query_id,
        "elapsed_ms": result.elapsed_ms,
        "statement_kind": result.statement_kind,
    }


def _credentials_from_ctx(  # noqa: PLR0912 — many independent overrides.
    ctx: RunContext, base: SnowflakeCredentials | None
) -> SnowflakeCredentials:
    """Layer credentials: ctx.api_keys > plugin base > env (SNOWFLAKE_*)."""
    fields: dict[str, Any] = {}
    if base is not None:
        fields = base.model_dump(by_alias=True, exclude_none=True)

    def _pick(*names: str) -> str | None:
        for n in names:
            v = ctx.api_keys.get(n) or os.environ.get(n)
            if v:
                return v
        return None

    overrides: dict[str, Any] = {}
    if v := _pick("SNOWFLAKE_ACCOUNT"):
        overrides["account"] = v
    if v := _pick("SNOWFLAKE_USER"):
        overrides["user"] = v
    if v := _pick("SNOWFLAKE_PASSWORD"):
        overrides["password"] = v
    if v := _pick("SNOWFLAKE_PRIVATE_KEY"):
        overrides["private_key"] = v
    if v := _pick("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"):
        overrides["private_key_passphrase"] = v
    if v := _pick("SNOWFLAKE_AUTHENTICATOR"):
        overrides["authenticator"] = v
    if v := _pick("SNOWFLAKE_TOKEN"):
        overrides["token"] = v
    if v := _pick("SNOWFLAKE_ROLE"):
        overrides["role"] = v
    if v := _pick("SNOWFLAKE_WAREHOUSE"):
        overrides["warehouse"] = v
    if v := _pick("SNOWFLAKE_DATABASE"):
        overrides["database"] = v
    if v := _pick("SNOWFLAKE_SCHEMA"):
        overrides["schema"] = v
    if v := _pick("SNOWFLAKE_HOST"):
        overrides["host"] = v
    if v := _pick("SNOWFLAKE_REGION"):
        overrides["region"] = v

    fields.update(overrides)
    return SnowflakeCredentials(**fields)


class SnowflakeToolSource(ToolSource):
    """Register every Snowflake tool."""

    name = "snowflake"

    def __init__(
        self,
        *,
        credentials: dict[str, Any] | None = None,
        read_only: bool = True,
        max_rows: int = 10_000,
        connection_factory: Any = None,
    ) -> None:
        """Configure the plugin's default Snowflake connection settings.

        Parameters
        ----------
        credentials : dict[str, Any] or None, optional
            Base credentials used to construct a
            :class:`~openbb_agent_server.plugins.tools.snowflake_tools.client.SnowflakeCredentials`. These are the lowest-priority
            layer and are overridden per-run by ``ctx.api_keys`` and
            ``SNOWFLAKE_*`` environment variables. ``None`` means no plugin
            base credentials.
        read_only : bool, optional
            Default read-only enforcement. When ``True``, mutating
            statements are rejected. Defaults to ``True``. Overridable per
            run via the tool ``config``.
        max_rows : int, optional
            Default cap on rows returned by a query. Defaults to 10000.
            Overridable per run via the tool ``config``.
        connection_factory : Any, optional
            Optional callable used to create the underlying Snowflake
            connection, primarily for testing. ``None`` uses the default
            connector.
        """
        self._base_creds = SnowflakeCredentials(**credentials) if credentials else None
        self._read_only = read_only
        self._max_rows = max_rows
        self._connection_factory = connection_factory

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[Any]:
        """Resolve credentials and build the Snowflake tool set for a run.

        Layer credentials (``ctx.api_keys`` > plugin base > ``SNOWFLAKE_*``
        env vars), open a :class:`~openbb_agent_server.plugins.tools.snowflake_tools.client.SnowflakeClient`, and return every
        Snowflake tool bound to that client.

        Parameters
        ----------
        ctx : RunContext
            Active run context; ``ctx.api_keys`` supplies the
            highest-priority credential overrides.
        config : dict[str, Any]
            Per-invocation config. Recognised keys: ``read_only`` and
            ``max_rows``, which override the constructor defaults.

        Returns
        -------
        list[Any]
            The LangChain ``StructuredTool`` instances for querying,
            schema introspection, and Cortex functions.
        """
        creds = _credentials_from_ctx(ctx, self._base_creds)
        client = SnowflakeClient(
            creds,
            read_only=config.get("read_only", self._read_only),
            max_rows=config.get("max_rows", self._max_rows),
            connection_factory=self._connection_factory,
        )

        return _build_tools(client, creds, max_rows=client.max_rows)

    @staticmethod
    def build_tools(
        client: SnowflakeClient,
        creds: SnowflakeCredentials,
        *,
        max_rows: int,
    ) -> list[Any]:
        """Build the Snowflake tool set from an existing client.

        Convenience entry point that bypasses per-run credential layering,
        useful when a configured :class:`~openbb_agent_server.plugins.tools.snowflake_tools.client.SnowflakeClient` is already in
        hand (for example, in tests or embedding code).

        Parameters
        ----------
        client : SnowflakeClient
            Connected client the tools execute against.
        creds : SnowflakeCredentials
            Credentials used by Cortex Search / Analyst tools, which call
            the REST API directly rather than going through ``client``.
        max_rows : int
            Default row cap applied to ``snowflake_query``.

        Returns
        -------
        list[Any]
            The LangChain ``StructuredTool`` instances for this client.
        """
        return _build_tools(client, creds, max_rows=max_rows)


def _build_tools(
    client: SnowflakeClient,
    creds: SnowflakeCredentials,
    *,
    max_rows: int,
) -> list[Any]:
    def snowflake_query(sql: str, max_rows: int = max_rows) -> dict[str, Any]:
        emit.reasoning_step("snowflake_query", sql=sql[:160], max_rows=max_rows)
        return _result_to_payload(client.execute(sql, max_rows=max_rows))

    def snowflake_list_databases() -> dict[str, Any]:
        return _result_to_payload(client.execute("SHOW DATABASES"))

    def snowflake_list_schemas(database: str) -> dict[str, Any]:
        return _result_to_payload(
            client.execute(f"SHOW SCHEMAS IN DATABASE {database}")
        )

    def snowflake_list_tables(database: str, schema: str) -> dict[str, Any]:
        return _result_to_payload(
            client.execute(f"SHOW TABLES IN SCHEMA {database}.{schema}")
        )

    def snowflake_describe(object_path: str) -> dict[str, Any]:
        return _result_to_payload(client.execute(f"DESCRIBE {object_path}"))

    def snowflake_get_table_info(table: str) -> dict[str, Any]:
        """Return rich column metadata for a table."""
        parts = [p.strip('"') for p in table.split(".")]
        if len(parts) != 3:
            raise ValueError(
                f"snowflake_get_table_info expects DB.SCHEMA.TABLE, got {table!r}"
            )
        db, schema, table_name = parts
        sql_prefix = (
            "SELECT ordinal_position, column_name, data_type, is_nullable, "
            "       column_default, comment "
        )
        sql = (
            sql_prefix + f"FROM {db}.INFORMATION_SCHEMA.COLUMNS "  # noqa: S608
            "WHERE table_schema = %(schema)s AND table_name = %(table)s "
            "ORDER BY ordinal_position"
        )
        return _result_to_payload(
            client.execute(sql, {"schema": schema, "table": table_name})
        )

    def snowflake_get_table_sample_data(table: str, limit: int = 5) -> dict[str, Any]:
        """Read the first ``limit`` rows of ``table`` and emit as a table artifact."""
        sample_sql = f"SELECT * FROM {table}"  # noqa: S608
        return _result_to_payload(client.execute(sample_sql, max_rows=limit))

    def snowflake_get_multiple_table_definitions(
        tables: list[str],
    ) -> dict[str, Any]:
        """Fan out column-info lookups over a list of tables."""
        out: dict[str, Any] = {}
        for fq in tables:
            try:
                out[fq] = snowflake_get_table_info(fq)
            except Exception as exc:
                out[fq] = {"error": str(exc)}
        return {"tables": out}

    def snowflake_search_catalog(pattern: str, limit: int = 50) -> dict[str, Any]:
        sql = (
            "SELECT table_catalog, table_schema, table_name, column_name, data_type "
            "FROM SNOWFLAKE.ACCOUNT_USAGE.COLUMNS "
            "WHERE (LOWER(table_name) LIKE LOWER(%(p)s) "
            "       OR LOWER(column_name) LIKE LOWER(%(p)s)) "
            "ORDER BY table_catalog, table_schema, table_name "
        )
        return _result_to_payload(client.execute(sql, {"p": pattern}, max_rows=limit))

    def snowflake_explain(sql: str) -> dict[str, Any]:
        return _result_to_payload(client.execute(f"EXPLAIN {sql}"))

    def snowflake_query_history(limit: int = 20) -> dict[str, Any]:
        sql = (
            "SELECT query_id, user_name, role_name, warehouse_name, "
            "       database_name, schema_name, query_type, execution_status, "
            "       error_message, total_elapsed_time, start_time, query_text "
            "FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) "
            "ORDER BY start_time DESC "
        )
        return _result_to_payload(client.execute(sql, max_rows=limit))

    def snowflake_cortex_complete(
        prompt: str,
        model: str = "claude-3-5-sonnet",
        options: dict[str, Any] | None = None,
    ) -> str:
        emit.reasoning_step("cortex_complete", model=model, prompt=prompt[:80])
        return cortex_lib.cortex_complete(
            client, prompt=prompt, model=model, options=options
        )

    def snowflake_cortex_summarize(text: str) -> str:
        emit.reasoning_step("cortex_summarize", chars=len(text))
        return cortex_lib.cortex_summarize(client, text=text)

    def snowflake_cortex_sentiment(text: str) -> float:
        return cortex_lib.cortex_sentiment(client, text=text)

    def snowflake_cortex_translate(
        text: str, target_language: str, source_language: str = ""
    ) -> str:
        return cortex_lib.cortex_translate(
            client,
            text=text,
            target_language=target_language,
            source_language=source_language,
        )

    def snowflake_cortex_classify(text: str, categories: list[str]) -> dict[str, Any]:
        return cortex_lib.cortex_classify_text(client, text=text, categories=categories)

    def snowflake_cortex_extract_answer(question: str, context: str) -> dict[str, Any]:
        return cortex_lib.cortex_extract_answer(
            client, question=question, context=context
        )

    def snowflake_cortex_embed(
        text: str,
        model: str = "snowflake-arctic-embed-l-v2.0",
        dim: int = 1024,
    ) -> list[float]:
        return cortex_lib.cortex_embed(client, text=text, model=model, dim=dim)

    def snowflake_cortex_search(
        database: str,
        schema: str,
        service: str,
        query: str,
        columns: list[str] | None = None,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        emit.reasoning_step(
            "cortex_search",
            service=f"{database}.{schema}.{service}",
            query=query[:80],
            limit=limit,
        )
        result = cortex_lib.cortex_search(
            creds,
            database=database,
            schema=schema,
            service=service,
            query=query,
            columns=columns,
            limit=limit,
            filter_=filter,
        )
        for hit in result.get("results", []) or []:
            emit.cite(
                text=str(hit.get("chunk") or hit.get("content") or "")[:240],
                source=str(hit.get("title") or service),
                source_url=hit.get("url") or hit.get("source_url"),
            )
        return result

    def snowflake_cortex_analyst(
        messages: list[dict[str, Any]],
        semantic_model: str | None = None,
        semantic_view: str | None = None,
    ) -> dict[str, Any]:
        emit.reasoning_step(
            "cortex_analyst",
            semantic_model=semantic_model,
            semantic_view=semantic_view,
        )
        return cortex_lib.cortex_analyst(
            creds,
            messages=messages,
            semantic_model=semantic_model,
            semantic_view=semantic_view,
        )

    return [
        StructuredTool.from_function(
            snowflake_query,
            name="snowflake_query",
            description=(
                "Run a read-only SQL query against Snowflake. Mutating "
                "statements are rejected. Results render as a table "
                "artifact when non-empty; the function also returns "
                "rows / columns / query_id for further chaining."
            ),
            args_schema=_QueryArgs,
        ),
        StructuredTool.from_function(
            snowflake_list_databases,
            name="snowflake_list_databases",
            description="List databases visible to the current Snowflake role.",
        ),
        StructuredTool.from_function(
            snowflake_list_schemas,
            name="snowflake_list_schemas",
            description="List schemas in a Snowflake database.",
            args_schema=_ListSchemasArgs,
        ),
        StructuredTool.from_function(
            snowflake_list_tables,
            name="snowflake_list_tables",
            description="List tables/views in a Snowflake schema.",
            args_schema=_ListTablesArgs,
        ),
        StructuredTool.from_function(
            snowflake_describe,
            name="snowflake_describe",
            description="DESCRIBE a Snowflake table/view (column types).",
            args_schema=_DescribeArgs,
        ),
        StructuredTool.from_function(
            snowflake_get_table_info,
            name="snowflake_get_table_info",
            description=(
                "Return rich column metadata (ordinal, type, nullable, "
                "default, comment) for one fully-qualified table from "
                "INFORMATION_SCHEMA.COLUMNS."
            ),
            args_schema=_TableInfoArgs,
        ),
        StructuredTool.from_function(
            snowflake_get_table_sample_data,
            name="snowflake_get_table_sample_data",
            description=(
                "Sample the first N rows of a table. Renders as a table "
                "artifact in the Workspace UI."
            ),
            args_schema=_TableSampleArgs,
        ),
        StructuredTool.from_function(
            snowflake_get_multiple_table_definitions,
            name="snowflake_get_multiple_table_definitions",
            description=(
                "Fetch column definitions for many tables in one call. "
                "Returns a dict keyed by fully-qualified table name."
            ),
            args_schema=_MultiTableArgs,
        ),
        StructuredTool.from_function(
            snowflake_search_catalog,
            name="snowflake_search_catalog",
            description=("Search ACCOUNT_USAGE.COLUMNS by table/column ILIKE pattern."),
            args_schema=_CatalogSearchArgs,
        ),
        StructuredTool.from_function(
            snowflake_explain,
            name="snowflake_explain",
            description="Return the EXPLAIN plan for a query.",
            args_schema=_ExplainArgs,
        ),
        StructuredTool.from_function(
            snowflake_query_history,
            name="snowflake_query_history",
            description=(
                "Fetch recent queries from this session's "
                "INFORMATION_SCHEMA.QUERY_HISTORY()."
            ),
            args_schema=_HistoryArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_complete,
            name="snowflake_cortex_complete",
            description="Call SNOWFLAKE.CORTEX.COMPLETE for a chat-style completion.",
            args_schema=_CortexCompleteArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_summarize,
            name="snowflake_cortex_summarize",
            description="Call SNOWFLAKE.CORTEX.SUMMARIZE on the supplied text.",
            args_schema=_CortexSummarizeArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_sentiment,
            name="snowflake_cortex_sentiment",
            description="Score sentiment via SNOWFLAKE.CORTEX.SENTIMENT (-1..+1).",
            args_schema=_CortexSentimentArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_translate,
            name="snowflake_cortex_translate",
            description="Translate text via SNOWFLAKE.CORTEX.TRANSLATE.",
            args_schema=_CortexTranslateArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_classify,
            name="snowflake_cortex_classify",
            description=(
                "Classify text into one of the supplied categories via "
                "SNOWFLAKE.CORTEX.CLASSIFY_TEXT."
            ),
            args_schema=_CortexClassifyArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_extract_answer,
            name="snowflake_cortex_extract_answer",
            description=(
                "Extract an answer + score from a context paragraph via "
                "SNOWFLAKE.CORTEX.EXTRACT_ANSWER."
            ),
            args_schema=_CortexExtractArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_embed,
            name="snowflake_cortex_embed",
            description=(
                "Embed text via SNOWFLAKE.CORTEX.EMBED_TEXT_768 / EMBED_TEXT_1024."
            ),
            args_schema=_CortexEmbedArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_search,
            name="snowflake_cortex_search",
            description=(
                "Query a Cortex Search service. Returns ranked hits and "
                "automatically cites each result."
            ),
            args_schema=_CortexSearchArgs,
        ),
        StructuredTool.from_function(
            snowflake_cortex_analyst,
            name="snowflake_cortex_analyst",
            description=(
                "Ask Cortex Analyst a question against a semantic model / view. "
                "Provide either a stage path to the YAML semantic model or the "
                "fully-qualified semantic view name."
            ),
            args_schema=_CortexAnalystArgs,
        ),
    ]
