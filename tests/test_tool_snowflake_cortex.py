"""Cortex helper tests."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from openbb_agent_server.plugins.tools.snowflake_tools import cortex as cortex_lib
from openbb_agent_server.plugins.tools.snowflake_tools.client import (
    SnowflakeClient,
    SnowflakeCredentials,
)


class _RecordingConn(sqlite3.Connection):
    last_sql: str = ""
    last_params: Any = None
    canned: Any = None

    def cursor(self, *a, **k):  # type: ignore[no-untyped-def]
        return _RecordingCursor(super().cursor(*a, **k), self)


class _RecordingCursor:
    def __init__(self, cur: sqlite3.Cursor, conn: _RecordingConn) -> None:
        self._cur = cur
        self._conn = conn
        self.sfqid = "test"

    def execute(self, sql: str, params: Any = None):  # type: ignore[no-untyped-def]
        self._conn.last_sql = sql
        self._conn.last_params = params
        if "SNOWFLAKE.CORTEX" in sql.upper() or "CORTEX" in sql.upper():
            value = self._conn.canned
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return self._cur.execute("SELECT ? AS response", (value,))
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
def recording_conn() -> Iterator[_RecordingConn]:
    conn = sqlite3.connect(":memory:", factory=_RecordingConn)
    yield conn
    conn.close()


def _make_client(conn: _RecordingConn) -> SnowflakeClient:
    creds = SnowflakeCredentials(account="acc", user="u")
    return SnowflakeClient(creds, connection_factory=lambda c: conn)


def test_cortex_complete_sends_correct_sql_and_returns_text(
    recording_conn: _RecordingConn,
) -> None:
    recording_conn.canned = "Hello world."
    client = _make_client(recording_conn)
    out = cortex_lib.cortex_complete(
        client, prompt="Say hi.", model="claude-3-5-sonnet"
    )
    assert out == "Hello world."
    assert "SNOWFLAKE.CORTEX.COMPLETE" in recording_conn.last_sql


def test_cortex_complete_with_options(
    recording_conn: _RecordingConn,
) -> None:
    recording_conn.canned = "ok"
    client = _make_client(recording_conn)
    cortex_lib.cortex_complete(
        client,
        prompt="hi",
        model="claude-3-5-sonnet",
        options={"temperature": 0.0},
    )
    assert "PARSE_JSON" in recording_conn.last_sql


def test_cortex_summarize_returns_text(recording_conn: _RecordingConn) -> None:
    recording_conn.canned = "summary"
    client = _make_client(recording_conn)
    assert cortex_lib.cortex_summarize(client, text="big text") == "summary"


def test_cortex_sentiment_returns_float(recording_conn: _RecordingConn) -> None:
    recording_conn.canned = "0.42"
    client = _make_client(recording_conn)
    assert cortex_lib.cortex_sentiment(client, text="great") == pytest.approx(0.42)


def test_cortex_translate_passes_languages(recording_conn: _RecordingConn) -> None:
    recording_conn.canned = "hola"
    client = _make_client(recording_conn)
    assert (
        cortex_lib.cortex_translate(client, text="hello", target_language="es")
        == "hola"
    )


def test_cortex_classify_text_parses_json(recording_conn: _RecordingConn) -> None:
    recording_conn.canned = {"label": "positive", "score": 0.9}
    client = _make_client(recording_conn)
    out = cortex_lib.cortex_classify_text(
        client, text="great", categories=["positive", "negative"]
    )
    assert out == {"label": "positive", "score": 0.9}


def test_cortex_extract_answer_parses_json(recording_conn: _RecordingConn) -> None:
    recording_conn.canned = {"answer": "42", "score": 0.99}
    client = _make_client(recording_conn)
    out = cortex_lib.cortex_extract_answer(
        client, question="meaning?", context="42 is the answer"
    )
    assert out["answer"] == "42"


def test_cortex_embed_returns_floats(recording_conn: _RecordingConn) -> None:
    recording_conn.canned = [0.1, 0.2, 0.3]
    client = _make_client(recording_conn)
    out = cortex_lib.cortex_embed(client, text="anything")
    assert out == [0.1, 0.2, 0.3]


def test_cortex_embed_dim_768(recording_conn: _RecordingConn) -> None:
    recording_conn.canned = [0.5, 0.5]
    client = _make_client(recording_conn)
    cortex_lib.cortex_embed(client, text="t", dim=768)
    assert "EMBED_TEXT_768" in recording_conn.last_sql


def _keypair_creds() -> tuple[SnowflakeCredentials, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return (
        SnowflakeCredentials(account="acc", user="u", private_key=pem),
        pem,
    )


def test_keypair_jwt_minted_with_correct_claims() -> None:
    creds, _ = _keypair_creds()
    token = cortex_lib.keypair_jwt(creds)
    import jwt as pyjwt

    decoded = pyjwt.decode(token, options={"verify_signature": False})
    assert decoded["sub"] == "ACC.U"
    assert decoded["iss"].startswith("ACC.U.")
    assert "iat" in decoded
    assert "exp" in decoded


def test_auth_headers_keypair_marks_token_type() -> None:
    creds, _ = _keypair_creds()
    headers = cortex_lib.auth_headers(creds)
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"


def test_auth_headers_oauth() -> None:
    creds = SnowflakeCredentials(
        account="acc", user="u", authenticator="oauth", token="oauth-token"
    )
    headers = cortex_lib.auth_headers(creds)
    assert headers["Authorization"] == "Bearer oauth-token"
    assert headers["X-Snowflake-Authorization-Token-Type"] == "OAUTH"


def test_auth_headers_pat() -> None:
    creds = SnowflakeCredentials(
        account="acc",
        user="u",
        authenticator="programmatic_access_token",
        token="pat-x",
    )
    headers = cortex_lib.auth_headers(creds)
    assert headers["Authorization"] == "Bearer pat-x"
    assert (
        headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
    )


def test_auth_headers_without_credentials_raises() -> None:
    creds = SnowflakeCredentials(account="acc", user="u")
    with pytest.raises(RuntimeError):
        cortex_lib.auth_headers(creds)


def test_cortex_search_round_trip() -> None:
    creds, _ = _keypair_creds()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "chunk": "snippet",
                        "title": "doc",
                        "url": "https://example.com/d",
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = cortex_lib.cortex_search(
        creds,
        database="DB",
        schema="S",
        service="SVC",
        query="hello",
        client=client,
    )
    assert out["results"][0]["title"] == "doc"
    assert captured["url"].endswith(
        "/api/v2/databases/DB/schemas/S/cortex-search-services/SVC:query"
    )
    assert captured["body"]["query"] == "hello"
    assert captured["headers"]["x-snowflake-authorization-token-type"] == "KEYPAIR_JWT"


def test_cortex_analyst_round_trip() -> None:
    creds, _ = _keypair_creds()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"message": {"content": [{"type": "sql", "statement": "SELECT 1"}]}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = cortex_lib.cortex_analyst(
        creds,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Q?"}]}],
        semantic_model="@db.s.stg/model.yaml",
        client=client,
    )
    assert "message" in out
    assert captured["body"]["semantic_model_file"] == "@db.s.stg/model.yaml"


def test_cortex_analyst_requires_model_or_view() -> None:
    creds, _ = _keypair_creds()
    with pytest.raises(RuntimeError):
        cortex_lib.cortex_analyst(creds, messages=[])


def test_account_host_prefers_explicit_host() -> None:
    creds = SnowflakeCredentials(account="acc", user="u", host="https://my.snowflake")
    assert cortex_lib._account_host(creds) == "https://my.snowflake"


def test_account_host_default_format() -> None:
    creds = SnowflakeCredentials(account="acc", user="u")
    assert cortex_lib._account_host(creds) == "https://acc.snowflakecomputing.com"


def test_account_host_with_region() -> None:
    creds = SnowflakeCredentials(account="acc", user="u", region="us-east-1")
    assert (
        cortex_lib._account_host(creds)
        == "https://acc.us-east-1.snowflakecomputing.com"
    )


def test_account_host_requires_account() -> None:
    creds = SnowflakeCredentials(user="u")
    with pytest.raises(RuntimeError):
        cortex_lib._account_host(creds)


def test_keypair_jwt_requires_full_credentials() -> None:
    with pytest.raises(RuntimeError):
        cortex_lib.keypair_jwt(SnowflakeCredentials(account="acc", user="u"))


def test_cortex_scalar_raises_when_no_rows() -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools import cortex as cortex_mod
    from openbb_agent_server.plugins.tools.snowflake_tools.client import QueryResult

    empty = QueryResult(
        sql="SELECT",
        statement_kind="SELECT",
        columns=[],
        rows=[],
        row_count=0,
        truncated=False,
        query_id="q",
        elapsed_ms=0,
    )
    with pytest.raises(RuntimeError, match="no rows"):
        cortex_mod._scalar(empty)


def test_cortex_parse_json_passes_dict_and_list_through() -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools.cortex import _parse_json

    assert _parse_json({"a": 1}) == {"a": 1}
    assert _parse_json([1, 2]) == [1, 2]
    assert _parse_json("not-json") == "not-json"
    assert _parse_json(42) == 42


def test_cortex_search_forwards_optional_columns_and_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools import cortex as cortex_mod
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeCredentials,
    )

    captured: dict[str, Any] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"results": []}

    class _Client:
        def post(
            self, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> _Resp:
            captured["body"] = json
            return _Resp()

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cortex_mod.httpx, "Client", lambda **_kw: _Client())
    cortex_mod.cortex_search(
        SnowflakeCredentials(account="a", user="u", token="tok", authenticator="oauth"),
        database="DB",
        schema="S",
        service="SVC",
        query="q",
        columns=["c1", "c2"],
        filter_={"status": "ok"},
    )
    assert captured["body"]["columns"] == ["c1", "c2"]
    assert captured["body"]["filter"] == {"status": "ok"}
    assert captured.get("closed") is True


def test_cortex_analyst_supports_semantic_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.plugins.tools.snowflake_tools import cortex as cortex_mod
    from openbb_agent_server.plugins.tools.snowflake_tools.client import (
        SnowflakeCredentials,
    )

    captured: dict[str, Any] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {}

    class _Client:
        def post(
            self, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> _Resp:
            captured["body"] = json
            return _Resp()

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cortex_mod.httpx, "Client", lambda **_kw: _Client())
    cortex_mod.cortex_analyst(
        SnowflakeCredentials(account="a", user="u", token="tok", authenticator="oauth"),
        messages=[{"role": "user"}],
        semantic_view="DB.S.MY_VIEW",
    )
    assert captured["body"]["semantic_view"] == "DB.S.MY_VIEW"
    assert captured.get("closed") is True
