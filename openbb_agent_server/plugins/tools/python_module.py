"""``python_module`` tool source."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ToolSource


def _resolve(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(
            f"python_module tool spec must be 'pkg.mod:attribute', got {spec!r}"
        )
    module, attr = spec.split(":", 1)
    return getattr(importlib.import_module(module), attr)


def _flatten(value: Any) -> list[BaseTool]:
    import contextlib

    if callable(value) and not isinstance(value, BaseTool):
        with contextlib.suppress(TypeError):
            value = value()
    if isinstance(value, BaseTool):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[BaseTool] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    raise TypeError(
        f"python_module spec resolved to unsupported type {type(value).__name__}"
    )


class PythonModuleToolSource(ToolSource):
    """Discover LangChain tools from dotted-path locations."""

    name = "python_module"

    def __init__(self, *, modules: Sequence[str] | None = None) -> None:
        """Store the default tool specs to resolve when none are configured.

        Parameters
        ----------
        modules : sequence of str or None
            Default ``"pkg.mod:attribute"`` specs used by :meth:`~openbb_agent_server.runtime.plugins.ToolSource.tools`
            when the per-call config omits ``"modules"``. ``None`` is
            treated as an empty sequence.
        """
        self._specs = tuple(modules or ())

    async def tools(self, ctx: RunContext, config: dict[str, Any]) -> list[BaseTool]:
        """Resolve the configured specs into a flat list of LangChain tools.

        Each spec is a ``"pkg.mod:attribute"`` string. The attribute is
        imported and flattened: callables are invoked, and lists or tuples
        are expanded recursively, so every resolved value yields zero or
        more :class:`BaseTool` instances.

        Parameters
        ----------
        ctx : RunContext
            Active run context (unused; present for the
            :class:`~openbb_agent_server.runtime.plugins.ToolSource` interface).
        config : dict
            Per-call configuration; its ``"modules"`` key, when present,
            overrides the specs supplied at construction.

        Returns
        -------
        list of BaseTool
            All tools discovered across the resolved specs.

        Raises
        ------
        ValueError
            If a spec is not in ``"pkg.mod:attribute"`` form.
        TypeError
            If a spec resolves to a value that is not a tool, callable,
            list, or tuple.
        """
        specs: Iterable[str] = config.get("modules", self._specs)
        out: list[BaseTool] = []
        for spec in specs:
            out.extend(_flatten(_resolve(spec)))
        return out
