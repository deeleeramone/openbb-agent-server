"""fetch_url tool source tests."""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

from openbb_agent_server.plugins.tools import fetch_url as fu
from openbb_agent_server.runtime import (
    context as run_context,
    emit,
)
from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.principal import UserPrincipal


def _ctx(*, web: bool = True) -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
        workspace_options={"fetch-url": True} if web else {},
    )


@pytest.mark.parametrize(
    "ip",
    [
        "10.0.0.1",
        "192.168.1.1",
        "172.16.0.1",
        "127.0.0.1",
        "::1",
        "169.254.169.254",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "::ffff:10.0.0.1",
        "not-an-ip",
    ],
)
def test_ip_blocked_rejects_unsafe(ip: str) -> None:
    assert fu._ip_blocked(ip) is True


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
def test_ip_blocked_allows_public(ip: str) -> None:
    assert fu._ip_blocked(ip) is False


def test_validate_url_returns_host_for_http_and_https() -> None:
    assert fu._validate_url("https://example.com/article") == "example.com"
    assert fu._validate_url("http://news.example.org") == "news.example.org"


@pytest.mark.parametrize("url", ["ftp://x/y", "file:///etc/passwd", "gopher://x"])
def test_validate_url_rejects_non_http_scheme(url: str) -> None:
    with pytest.raises(fu.FetchUrlError, match="only http"):
        fu._validate_url(url)


def test_validate_url_rejects_missing_host() -> None:
    with pytest.raises(fu.FetchUrlError, match="no host"):
        fu._validate_url("http:///just-a-path")


def _patch_getaddrinfo(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    """Patch the running loop's getaddrinfo with a canned result."""
    import asyncio

    async def _fake(host: str, port: Any, **_kw: Any) -> Any:
        if isinstance(result, BaseException):
            raise result
        return result

    class _FakeLoop:
        getaddrinfo = staticmethod(_fake)

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _FakeLoop())


@pytest.mark.asyncio
async def test_resolve_and_check_passes_for_public_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_getaddrinfo(
        monkeypatch,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))],
    )
    await fu._resolve_and_check("example.com")


@pytest.mark.asyncio
async def test_resolve_and_check_blocks_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_getaddrinfo(
        monkeypatch,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))],
    )
    with pytest.raises(fu.FetchUrlError, match="SSRF guard"):
        await fu._resolve_and_check("metadata.example")


@pytest.mark.asyncio
async def test_resolve_and_check_raises_on_dns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_getaddrinfo(monkeypatch, socket.gaierror("name resolution failed"))
    with pytest.raises(fu.FetchUrlError, match="could not resolve"):
        await fu._resolve_and_check("nonexistent.invalid")


@pytest.mark.asyncio
async def test_resolve_and_check_raises_on_empty_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_getaddrinfo(monkeypatch, [])
    with pytest.raises(fu.FetchUrlError, match="no addresses"):
        await fu._resolve_and_check("empty.example")


def test_extract_text_html_strips_script_and_style() -> None:
    html = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><p>Real content.</p>"
        "<script>alert('no')</script><p>More text.</p></body></html>"
    )
    text = fu._extract_text(html.encode(), "text/html; charset=utf-8")
    assert "Real content." in text
    assert "More text." in text
    assert "alert" not in text
    assert "color:red" not in text


def test_extract_text_plain_and_json_passthrough() -> None:
    assert fu._extract_text(b"just words", "text/plain") == "just words"
    assert fu._extract_text(b'{"k": 1}', "application/json") == '{"k": 1}'


def test_extract_text_binary_returns_empty() -> None:
    assert fu._extract_text(b"\x89PNG\r\n", "image/png") == ""


def _mock_httpx(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> None:
    """Make _fetch's httpx.AsyncClient use a MockTransport."""
    real_client = httpx.AsyncClient

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


async def _no_dns(_host: str) -> None:
    """Stand in for _resolve_and_check, allowing every host."""


@pytest.mark.asyncio
async def test_fetch_returns_body_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fu, "_resolve_and_check", _no_dns)
    _mock_httpx(
        monkeypatch,
        lambda req: httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<p>hi</p>"
        ),
    )
    out = await fu._fetch("https://example.com/a")
    assert out["status"] == 200
    assert out["body"] == b"<p>hi</p>"
    assert out["final_url"] == "https://example.com/a"


@pytest.mark.asyncio
async def test_fetch_follows_and_revalidates_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fu, "_resolve_and_check", _no_dns)

    def _handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"done"
        )

    _mock_httpx(monkeypatch, _handler)
    out = await fu._fetch("https://example.com/start")
    assert out["body"] == b"done"
    assert out["final_url"] == "https://example.com/final"


@pytest.mark.asyncio
async def test_fetch_redirect_to_blocked_host_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a redirect onto an internal host with the per-hop guard."""

    async def _guard(host: str) -> None:
        if host == "169.254.169.254":
            raise fu.FetchUrlError("refusing: SSRF guard")

    monkeypatch.setattr(fu, "_resolve_and_check", _guard)
    _mock_httpx(
        monkeypatch,
        lambda req: httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data"}
        ),
    )
    with pytest.raises(fu.FetchUrlError, match="SSRF guard"):
        await fu._fetch("https://example.com/start")


@pytest.mark.asyncio
async def test_fetch_redirect_without_location_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fu, "_resolve_and_check", _no_dns)
    _mock_httpx(monkeypatch, lambda req: httpx.Response(302))
    with pytest.raises(fu.FetchUrlError, match="no Location"):
        await fu._fetch("https://example.com/x")


@pytest.mark.asyncio
async def test_fetch_too_many_redirects_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fu, "_resolve_and_check", _no_dns)
    _mock_httpx(
        monkeypatch,
        lambda req: httpx.Response(302, headers={"location": "/again"}),
    )
    with pytest.raises(fu.FetchUrlError, match="too many redirects"):
        await fu._fetch("https://example.com/loop")


@pytest.mark.asyncio
async def test_fetch_body_over_size_cap_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fu, "_resolve_and_check", _no_dns)
    monkeypatch.setattr(fu, "_MAX_BYTES", 8)
    _mock_httpx(
        monkeypatch,
        lambda req: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"x" * 64
        ),
    )
    with pytest.raises(fu.FetchUrlError, match="cap"):
        await fu._fetch("https://example.com/big")


@pytest.mark.asyncio
async def test_fetch_url_returns_text_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    async def _fake_fetch(url: str) -> dict[str, Any]:
        return {
            "final_url": url,
            "status": 200,
            "content_type": "text/html",
            "body": b"<p>Article body.</p>",
        }

    monkeypatch.setattr(fu, "_fetch", _fake_fetch)
    with emit.bind_writer(captured.append):
        out = await fu.fetch_url("https://example.com/a")
    assert out["text"] == "Article body."
    assert out["truncated"] is False
    assert any(w.get("type") == "citations" for w in captured)


@pytest.mark.asyncio
async def test_fetch_url_truncates_long_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fu, "_MAX_TEXT_CHARS", 10)

    async def _fake_fetch(url: str) -> dict[str, Any]:
        return {
            "final_url": url,
            "status": 200,
            "content_type": "text/plain",
            "body": b"x" * 200,
        }

    monkeypatch.setattr(fu, "_fetch", _fake_fetch)
    with emit.bind_writer(lambda _w: None):
        out = await fu.fetch_url("https://example.com/long")
    assert out["truncated"] is True
    assert len(out["text"]) == 10


@pytest.mark.asyncio
async def test_fetch_url_non_text_returns_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_fetch(url: str) -> dict[str, Any]:
        return {
            "final_url": url,
            "status": 200,
            "content_type": "image/png",
            "body": b"\x89PNG",
        }

    monkeypatch.setattr(fu, "_fetch", _fake_fetch)
    with emit.bind_writer(lambda _w: None):
        out = await fu.fetch_url("https://example.com/pic.png")
    assert "note" in out
    assert "text" not in out


@pytest.mark.asyncio
async def test_fetch_url_returns_error_on_fetch_url_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(url: str) -> dict[str, Any]:
        raise fu.FetchUrlError("refusing: SSRF guard")

    monkeypatch.setattr(fu, "_fetch", _boom)
    with emit.bind_writer(lambda _w: None):
        out = await fu.fetch_url("http://10.0.0.1/")
    assert "SSRF guard" in out["error"]
    assert out["url"] == "http://10.0.0.1/"


@pytest.mark.asyncio
async def test_fetch_url_returns_error_on_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(url: str) -> dict[str, Any]:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(fu, "_fetch", _boom)
    with emit.bind_writer(lambda _w: None):
        out = await fu.fetch_url("https://example.com/down")
    assert out["error"].startswith("fetch failed: ConnectError")


@pytest.mark.asyncio
async def test_tool_source_skips_when_web_feature_disabled() -> None:
    src = fu.FetchUrlToolSource()
    assert await src.tools(_ctx(web=False), {}) == []


@pytest.mark.asyncio
async def test_tool_source_yields_fetch_url_when_enabled() -> None:
    src = fu.FetchUrlToolSource()
    tools = await src.tools(_ctx(web=True), {})
    assert [t.name for t in tools] == ["fetch_url"]


@pytest.mark.asyncio
async def test_tool_source_tool_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke the SSRF-guarded pipeline through the bound tool."""
    monkeypatch.setattr(fu, "_resolve_and_check", _no_dns)
    _mock_httpx(
        monkeypatch,
        lambda req: httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<p>News.</p>"
        ),
    )
    src = fu.FetchUrlToolSource()
    [tool] = await src.tools(_ctx(web=True), {})
    ctx = _ctx(web=True)
    with run_context.bind(ctx), emit.bind_writer(lambda _w: None):
        out = await tool.ainvoke({"url": "https://example.com/news"})
    assert out["text"] == "News."
