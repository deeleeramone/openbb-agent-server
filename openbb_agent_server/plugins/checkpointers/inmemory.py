"""In-memory checkpointer provider."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from openbb_agent_server.runtime.plugins import CheckpointerProvider


class InMemoryCheckpointerProvider(CheckpointerProvider):
    """Provide a process-local LangGraph ``InMemorySaver`` checkpointer.

    Checkpoints live only in the current process and are lost on restart.
    Intended for tests and single-process development; not durable.
    """

    name = "inmemory"

    def __init__(self, **_config: Any) -> None:
        """Accept and ignore any configuration keyword arguments.

        Parameters
        ----------
        **_config : Any
            Configuration options, accepted for interface compatibility
            with other checkpointer providers but unused here.
        """

    async def open(self, settings: Any) -> InMemorySaver:
        """Return a fresh in-memory checkpoint saver.

        Parameters
        ----------
        settings : Any
            Runtime settings, accepted for interface compatibility but
            unused by this provider.

        Returns
        -------
        InMemorySaver
            A new, empty process-local checkpoint saver.
        """
        return InMemorySaver()

    async def close(self, saver: Any) -> None:
        """Release the checkpoint saver (a no-op for in-memory storage).

        Parameters
        ----------
        saver : Any
            The previously opened saver. Ignored, since no external
            resources are held.

        Returns
        -------
        None
            Always ``None``; nothing needs releasing.
        """
        return None
