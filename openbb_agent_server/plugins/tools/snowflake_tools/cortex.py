"""Snowflake Cortex helpers — SQL functions + REST endpoints."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from base64 import b64encode
from typing import Any

import httpx
import jwt as pyjwt

from openbb_agent_server.plugins.tools.snowflake_tools.client import (
    QueryResult,
    SnowflakeClient,
    SnowflakeCredentials,
)

logger = logging.getLogger("openbb_agent_server.tools.snowflake.cortex")


def cortex_complete(
    client: SnowflakeClient,
    *,
    prompt: str,
    model: str = "claude-3-5-sonnet",
    options: dict[str, Any] | None = None,
) -> str:
    """Call ``SNOWFLAKE.CORTEX.COMPLETE`` and return the text response.

    Run the COMPLETE SQL function over a single prompt, optionally passing
    model options as a parsed JSON object when ``options`` is supplied.

    Parameters
    ----------
    client : SnowflakeClient
        Connected client used to execute the COMPLETE SQL statement.
    prompt : str
        The prompt text sent to the model.
    model : str, default "claude-3-5-sonnet"
        Name of the Cortex model to invoke.
    options : dict[str, Any] | None, optional
        Extra model options (for example temperature or max tokens) serialized
        to JSON and passed through ``PARSE_JSON``. When ``None`` the two-argument
        form of COMPLETE is used.

    Returns
    -------
    str
        The text completion returned by the model.
    """
    if options:
        sql = (
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(%(model)s, %(prompt)s, "
            "PARSE_JSON(%(options)s)) AS response"
        )
        result = client.execute(
            sql,
            {
                "model": model,
                "prompt": prompt,
                "options": json.dumps(options),
            },
        )
    else:
        sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(%(model)s, %(prompt)s) AS response"
        result = client.execute(sql, {"model": model, "prompt": prompt})
    return _scalar(result)


def cortex_summarize(client: SnowflakeClient, *, text: str) -> str:
    """Call ``SNOWFLAKE.CORTEX.SUMMARIZE`` and return the summary.

    Parameters
    ----------
    client : SnowflakeClient
        Connected client used to execute the SUMMARIZE SQL statement.
    text : str
        The text to summarize.

    Returns
    -------
    str
        The generated summary of ``text``.
    """
    return _scalar(
        client.execute(
            "SELECT SNOWFLAKE.CORTEX.SUMMARIZE(%(text)s) AS response",
            {"text": text},
        )
    )


def cortex_sentiment(client: SnowflakeClient, *, text: str) -> float:
    """Call ``SNOWFLAKE.CORTEX.SENTIMENT`` and return the sentiment score.

    Parameters
    ----------
    client : SnowflakeClient
        Connected client used to execute the SENTIMENT SQL statement.
    text : str
        The text to score for sentiment.

    Returns
    -------
    float
        Sentiment score, typically in the range -1 (negative) to 1 (positive).
    """
    raw = _scalar(
        client.execute(
            "SELECT SNOWFLAKE.CORTEX.SENTIMENT(%(text)s) AS response",
            {"text": text},
        )
    )
    return float(raw)


def cortex_translate(
    client: SnowflakeClient,
    *,
    text: str,
    target_language: str,
    source_language: str = "",
) -> str:
    """Call ``SNOWFLAKE.CORTEX.TRANSLATE`` and return the translated text.

    Parameters
    ----------
    client : SnowflakeClient
        Connected client used to execute the TRANSLATE SQL statement.
    text : str
        The text to translate.
    target_language : str
        Language code to translate into (the TRANSLATE destination argument).
    source_language : str, default ""
        Source language code. When empty, Snowflake auto-detects the language.

    Returns
    -------
    str
        The translated text.
    """
    return _scalar(
        client.execute(
            "SELECT SNOWFLAKE.CORTEX.TRANSLATE(%(t)s, %(src)s, %(dst)s) AS response",
            {"t": text, "src": source_language, "dst": target_language},
        )
    )


def cortex_classify_text(
    client: SnowflakeClient,
    *,
    text: str,
    categories: list[str],
) -> dict[str, Any]:
    """Call ``SNOWFLAKE.CORTEX.CLASSIFY_TEXT`` and return the classification.

    Parameters
    ----------
    client : SnowflakeClient
        Connected client used to execute the CLASSIFY_TEXT SQL statement.
    text : str
        The text to classify.
    categories : list[str]
        Candidate category labels to choose from.

    Returns
    -------
    dict[str, Any]
        Parsed JSON result containing the chosen label (typically under a
        ``"label"`` key).
    """
    raw = _scalar(
        client.execute(
            "SELECT SNOWFLAKE.CORTEX.CLASSIFY_TEXT(%(t)s, %(c)s) AS response",
            {"t": text, "c": categories},
        )
    )
    return _parse_json(raw)


def cortex_extract_answer(
    client: SnowflakeClient,
    *,
    question: str,
    context: str,
) -> dict[str, Any]:
    """Call ``SNOWFLAKE.CORTEX.EXTRACT_ANSWER`` and return the answer payload.

    Parameters
    ----------
    client : SnowflakeClient
        Connected client used to execute the EXTRACT_ANSWER SQL statement.
    question : str
        The question to answer.
    context : str
        The source text from which the answer is extracted.

    Returns
    -------
    dict[str, Any]
        Parsed JSON result containing the extracted answer and its score.
    """
    raw = _scalar(
        client.execute(
            "SELECT SNOWFLAKE.CORTEX.EXTRACT_ANSWER(%(q)s, %(c)s) AS response",
            {"q": question, "c": context},
        )
    )
    return _parse_json(raw)


def cortex_embed(
    client: SnowflakeClient,
    *,
    text: str,
    model: str = "snowflake-arctic-embed-l-v2.0",
    dim: int = 1024,
) -> list[float]:
    """Embed text with Cortex and return the resulting vector.

    Select ``EMBED_TEXT_1024`` when ``dim`` is 1024, otherwise
    ``EMBED_TEXT_768``, and coerce the response into a list of floats.

    Parameters
    ----------
    client : SnowflakeClient
        Connected client used to execute the embedding SQL statement.
    text : str
        The text to embed.
    model : str, default "snowflake-arctic-embed-l-v2.0"
        Name of the Cortex embedding model.
    dim : int, default 1024
        Target embedding dimension. Any value other than 1024 routes to the
        768-dimension function.

    Returns
    -------
    list[float]
        The embedding vector as floats.
    """
    fn = "EMBED_TEXT_1024" if dim == 1024 else "EMBED_TEXT_768"
    raw = _scalar(
        client.execute(
            f"SELECT SNOWFLAKE.CORTEX.{fn}(%(model)s, %(text)s) AS response",
            {"model": model, "text": text},
        )
    )
    if isinstance(raw, str):
        raw = _parse_json(raw)
    return [float(x) for x in raw]


def _scalar(result: QueryResult) -> Any:
    if not result.rows or not result.rows[0]:
        raise RuntimeError("cortex call returned no rows")
    return result.rows[0][0]


def _parse_json(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _account_host(creds: SnowflakeCredentials) -> str:
    if creds.host:
        return creds.host.rstrip("/")
    if not creds.account:
        raise RuntimeError("Snowflake account is required for Cortex REST calls")
    region_part = f".{creds.region}" if creds.region else ""
    return f"https://{creds.account}{region_part}.snowflakecomputing.com"


def _public_key_fingerprint(private_key_pem: str, passphrase: str | None) -> str:
    """Compute the SHA-256 fingerprint Snowflake expects for KeyPair JWTs."""
    from cryptography.hazmat.primitives import serialization

    priv = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=passphrase.encode() if passphrase else None,
    )
    pub_der = priv.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    digest = hashlib.sha256(pub_der).digest()
    return "SHA256:" + b64encode(digest).decode()


def keypair_jwt(creds: SnowflakeCredentials, *, lifetime_seconds: int = 3600) -> str:
    """Mint a Snowflake-compatible KeyPair JWT for the REST API.

    Build the issuer/subject from the uppercased account and user plus the
    public-key fingerprint, then sign the claims with the private key using
    RS256.

    Parameters
    ----------
    creds : SnowflakeCredentials
        Credentials providing ``account``, ``user``, ``private_key``, and an
        optional ``private_key_passphrase``.
    lifetime_seconds : int, default 3600
        Number of seconds from now until the token's ``exp`` claim.

    Returns
    -------
    str
        The signed JWT string.

    Raises
    ------
    RuntimeError
        If ``account``, ``user``, or ``private_key`` is missing.
    """
    if not (creds.account and creds.user and creds.private_key):
        raise RuntimeError(
            "KeyPair JWT requires account, user, and private_key on the credentials"
        )
    fingerprint = _public_key_fingerprint(
        creds.private_key, creds.private_key_passphrase
    )
    now = _dt.datetime.now(_dt.timezone.utc)
    qualified_user = f"{creds.account.upper()}.{creds.user.upper()}"
    payload = {
        "iss": f"{qualified_user}.{fingerprint}",
        "sub": qualified_user,
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + lifetime_seconds,
    }
    return pyjwt.encode(payload, creds.private_key, algorithm="RS256")


def auth_headers(creds: SnowflakeCredentials) -> dict[str, str]:
    """Build the auth headers Snowflake REST endpoints accept.

    Prefer a bearer token when ``authenticator`` is ``"oauth"`` or
    ``"programmatic_access_token"``; otherwise fall back to minting a KeyPair
    JWT from the private key. The returned headers include the bearer token and
    the matching ``X-Snowflake-Authorization-Token-Type`` value.

    Parameters
    ----------
    creds : SnowflakeCredentials
        Credentials carrying either an OAuth/PAT ``token`` (with matching
        ``authenticator``) or a ``private_key`` for KeyPair JWT auth.

    Returns
    -------
    dict[str, str]
        Header mapping with ``Authorization`` and the token-type header.

    Raises
    ------
    RuntimeError
        If no OAuth token, PAT, or private key is available.
    """
    if creds.token and creds.authenticator in {
        "oauth",
        "programmatic_access_token",
    }:
        kind = (
            "OAUTH" if creds.authenticator == "oauth" else "PROGRAMMATIC_ACCESS_TOKEN"
        )
        return {
            "Authorization": f"Bearer {creds.token}",
            "X-Snowflake-Authorization-Token-Type": kind,
        }
    if creds.private_key:
        return {
            "Authorization": f"Bearer {keypair_jwt(creds)}",
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
        }
    raise RuntimeError(
        "Cortex REST endpoints require OAuth, PAT, or KeyPair JWT credentials"
    )


def cortex_search(
    creds: SnowflakeCredentials,
    *,
    database: str,
    schema: str,
    service: str,
    query: str,
    columns: list[str] | None = None,
    limit: int = 10,
    filter_: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Call a Cortex Search service and return the raw JSON response.

    Issue an authenticated POST to the search service's ``:query`` endpoint,
    including optional column projection and filter in the request body.

    Parameters
    ----------
    creds : SnowflakeCredentials
        Credentials used to derive the account host and auth headers.
    database : str
        Database containing the search service.
    schema : str
        Schema containing the search service.
    service : str
        Name of the Cortex Search service to query.
    query : str
        The search query text.
    columns : list[str] | None, optional
        Columns to return for each hit. When ``None`` the service defaults
        apply.
    limit : int, default 10
        Maximum number of results to return.
    filter_ : dict[str, Any] | None, optional
        Filter expression applied to the search, sent as the ``filter`` field.
    client : httpx.Client | None, optional
        Existing HTTP client to reuse. When ``None`` a temporary client is
        created and closed automatically.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response from the search service.

    Raises
    ------
    httpx.HTTPStatusError
        If the search request returns a non-success status code.
    """
    url = (
        f"{_account_host(creds)}/api/v2/databases/{database}"
        f"/schemas/{schema}/cortex-search-services/{service}:query"
    )
    body: dict[str, Any] = {"query": query, "limit": limit}
    if columns:
        body["columns"] = columns
    if filter_:
        body["filter"] = filter_
    headers = auth_headers(creds)
    headers["Content-Type"] = "application/json"
    closing = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()
    finally:
        if closing:
            client.close()


def cortex_analyst(
    creds: SnowflakeCredentials,
    *,
    messages: list[dict[str, Any]],
    semantic_model: str | None = None,
    semantic_view: str | None = None,
    stream: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Call Cortex Analyst (text-to-SQL via semantic models).

    Send a conversation to the Analyst message endpoint, grounding it with
    either a semantic model file or a semantic view. Exactly one grounding
    source must be provided.

    Parameters
    ----------
    creds : SnowflakeCredentials
        Credentials used to derive the account host and auth headers.
    messages : list[dict[str, Any]]
        Conversation messages in the Analyst request format.
    semantic_model : str | None, optional
        Path or stage reference to a semantic model file, sent as
        ``semantic_model_file``.
    semantic_view : str | None, optional
        Name of a semantic view, sent as ``semantic_view``.
    stream : bool, default False
        Whether to request a streamed response.
    client : httpx.Client | None, optional
        Existing HTTP client to reuse. When ``None`` a temporary client is
        created and closed automatically.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response from Cortex Analyst.

    Raises
    ------
    RuntimeError
        If neither ``semantic_model`` nor ``semantic_view`` is supplied.
    httpx.HTTPStatusError
        If the Analyst request returns a non-success status code.
    """
    if not (semantic_model or semantic_view):
        raise RuntimeError("Cortex Analyst requires semantic_model or semantic_view")
    url = f"{_account_host(creds)}/api/v2/cortex/analyst/message"
    body: dict[str, Any] = {"messages": messages, "stream": stream}
    if semantic_model:
        body["semantic_model_file"] = semantic_model
    if semantic_view:
        body["semantic_view"] = semantic_view
    headers = auth_headers(creds)
    headers["Content-Type"] = "application/json"
    closing = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()
    finally:
        if closing:
            client.close()
