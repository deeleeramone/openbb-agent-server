"""``mcp_http`` tool source — connect to a running MCP server over HTTP/SSE."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource

logger = logging.getLogger("openbb_agent_server.tools.mcp_http")


_CONFIG_FILE_ENVS: tuple[str, ...] = (
    "OPENBB_AGENT_MCP_CONFIG",
    "OPENBB_MCP_CONFIG",
    "OPENBB_AGENT_CONFIG",
    "OPENBB_API_CONFIG",
    "OPENBB_CONFIG",
)

_VALID_TRANSPORTS: frozenset[str] = frozenset({"streamable_http", "sse", "websocket"})

_TRANSPORT_ALIASES: dict[str, str] = {
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "sse": "sse",
    "websocket": "websocket",
    "ws": "websocket",
}


def _normalise_transport(value: str) -> str:
    out = _TRANSPORT_ALIASES.get(value, value)
    if out not in _VALID_TRANSPORTS:
        raise ValueError(
            f"unsupported transport {value!r}; expected one of "
            f"{sorted(_VALID_TRANSPORTS)} (or the dash form 'streamable-http')."
        )
    return out


def _resolve_config_file(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for var in _CONFIG_FILE_ENVS:
        v = os.environ.get(var)
        if v:
            return v
    return None


def _read_mcp_table(config_path: str | None) -> dict[str, Any]:
    """Re-walk the cascade and return the mcp table."""
    from openbb_agent_server.app.config import bootstrap_launcher_config

    try:
        cfg = bootstrap_launcher_config(explicit_path=config_path)
    except Exception:
        logger.debug("mcp_http: could not re-read TOML cascade", exc_info=True)
        return {}
    section = cfg.get("mcp")
    return dict(section) if isinstance(section, dict) else {}


def _build_url(host: str, port: int | str, transport: str) -> str:
    path = "/sse" if transport == "sse" else "/mcp"
    if host.startswith(("http://", "https://")):
        base = host.rstrip("/")
        host_part = base.split("//", 1)[1]
        if port and ":" not in host_part:
            base = f"{base}:{port}"
        return f"{base}{path}"
    return f"http://{host}:{port}{path}"


class HttpMcpToolSource(ToolSource):
    """Connect to a running MCP server over HTTP/SSE/WebSocket.

    Expose the tools served by a remote Model Context Protocol (MCP)
    server as LangChain tools. Connection details may be supplied
    directly to the constructor, overridden per run via ``config``, or
    discovered from the ``[mcp]`` table of the resolved ``openbb.toml``
    cascade.
    """

    name = "mcp_http"

    def __init__(
        self,
        *,
        url: str | None = None,
        transport: str | None = None,
        headers: dict[str, str] | None = None,
        server_name: str = "openbb",
        config_file: str | None = None,
    ) -> None:
        """Configure the default MCP connection parameters.

        Parameters
        ----------
        url : str or None, optional
            Full MCP endpoint URL. When omitted, the URL is taken from
            run ``config`` or built from ``[mcp].host`` and ``[mcp].port``
            at connection time.
        transport : str or None, optional
            Transport to use. Accepts ``"streamable_http"`` /
            ``"streamable-http"``, ``"sse"``, and ``"websocket"`` /
            ``"ws"``; aliases are normalised. When omitted, defaults are
            resolved at connection time (ultimately ``"streamable_http"``).
        headers : dict of str to str or None, optional
            Static HTTP headers sent with every request. Copied
            defensively. Additional headers may be merged in from config
            and the run's API keys at connection time.
        server_name : str, optional
            Logical name under which the server is registered in the
            underlying multi-server client. Defaults to ``"openbb"``.
        config_file : str or None, optional
            Explicit path to a TOML config file used to resolve the
            ``[mcp]`` table. When omitted, well-known environment
            variables are consulted.

        Raises
        ------
        ValueError
            If ``transport`` is given but is not a supported value.
        """
        if transport is not None:
            transport = _normalise_transport(transport)
        self._url = url
        self._transport = transport
        self._headers = dict(headers or {})
        self._server_name = server_name
        self._config_file = config_file

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Connect to the MCP server and return its tools.

        Resolve transport, URL, and headers by layering, in precedence
        order, the per-run ``config``, the constructor defaults, and the
        ``[mcp]`` table from the TOML cascade. Headers from ``[mcp].spec``
        and from ``config`` are merged in, and each of the run's API keys
        is added as an ``X-OPENBB-<KEY>`` header (without overriding an
        existing value). A :class:`MultiServerMCPClient` is then opened
        and its tools fetched.

        Parameters
        ----------
        ctx : RunContext
            The run context; ``ctx.api_keys`` supplies per-run
            authentication headers.
        config : dict
            Per-run overrides. Recognised keys include ``"config_file"``,
            ``"transport"``, ``"url"``, and ``"headers"``.

        Returns
        -------
        list of BaseTool
            The LangChain tools exposed by the connected MCP server.

        Raises
        ------
        RuntimeError
            If no URL can be resolved and ``[mcp].host``/``[mcp].port``
            are not both available to build one.
        ValueError
            If the resolved transport is not a supported value.
        """
        config_file = _resolve_config_file(
            config.get("config_file") or self._config_file
        )
        mcp_section = _read_mcp_table(config_file)

        raw_transport = (
            config.get("transport")
            or self._transport
            or mcp_section.get("transport")
            or "streamable_http"
        )
        transport = _normalise_transport(str(raw_transport))

        url = config.get("url") or self._url
        if not url:
            host = mcp_section.get("host")
            port = mcp_section.get("port")
            if not host or not port:
                raise RuntimeError(
                    "mcp_http: no URL configured. Either set ``url`` "
                    "directly (constructor or [agent.tool_source_config.mcp_http].url) "
                    "or provide ``[mcp].host`` and ``[mcp].port`` in your openbb.toml."
                )
            url = _build_url(str(host), port, transport)

        headers: dict[str, str] = {**self._headers}
        spec = mcp_section.get("spec")
        if isinstance(spec, dict):
            spec_headers = spec.get("headers") or {}
            if isinstance(spec_headers, dict):
                for k, v in spec_headers.items():
                    headers.setdefault(str(k), str(v))
        for k, v in dict(config.get("headers", {})).items():
            headers[k] = v
        for k, v in ctx.api_keys.items():
            headers.setdefault(f"X-OPENBB-{k}", v)

        if mcp_section:
            logger.debug(
                "mcp_http: connecting to %s (transport=%s, [mcp] keys=%s)",
                url,
                transport,
                sorted(mcp_section.keys()),
            )
        else:
            logger.debug("mcp_http: connecting to %s (transport=%s)", url, transport)

        from typing import cast

        connections: dict[str, Any] = {
            self._server_name: {
                "transport": transport,
                "url": url,
                "headers": headers,
            }
        }
        client = MultiServerMCPClient(connections=cast(Any, connections))
        return await client.get_tools()
