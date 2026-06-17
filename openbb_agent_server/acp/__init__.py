"""ACP shim — expose the agent loop to PyWry chat components.

Lazy re-exports so ``import openbb_agent_server.acp`` works without
pywry installed; the ImportError with install instructions surfaces on
first attribute access instead. ``PyWryCanvas`` / ``build_canvas_html``
are pywry-free and always importable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "OpenBBAgentProvider": "openbb_agent_server.acp.provider",
    "create_chat_manager": "openbb_agent_server.acp.provider",
    "translate_sse": "openbb_agent_server.acp.provider",
    "launch": "openbb_agent_server.acp.canvas_app",
    "CanvasApp": "openbb_agent_server.acp.canvas_app",
    "PyWryCanvas": "openbb_agent_server.acp.canvas",
    "build_canvas_html": "openbb_agent_server.acp.canvas",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Defer submodule imports until a symbol is actually used."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_path), name)
