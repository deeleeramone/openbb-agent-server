"""Fake model provider for tests and demos."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider

DEFAULT_RESPONSES = ("OK.",)


class _ToolAwareFakeChatModel(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools and streams."""

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        return self

    def _stream(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.messages import AIMessage, AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        message = next(self.messages)
        if isinstance(message, str):
            chunk = AIMessageChunk(content=message, chunk_position="last")
        elif isinstance(message, AIMessage):
            chunk = AIMessageChunk(
                content=message.content,
                tool_call_chunks=[
                    {
                        "name": tc.get("name"),
                        "args": json.dumps(tc.get("args") or {}),
                        "id": tc.get("id"),
                        "index": idx,
                    }
                    for idx, tc in enumerate(message.tool_calls or [])
                ],
                chunk_position="last",
            )
        else:  # pragma: no cover — only str / AIMessage are valid inputs
            chunk = AIMessageChunk(content=str(message), chunk_position="last")
        yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        for chunk in self._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk


class FakeProvider(ModelProvider):
    """Deterministic fake model provider for tests and demos.

    Builds a chat model that replays a fixed sequence of canned responses
    instead of calling a real LLM, so agent flows can be exercised without
    network access or credentials.

    Attributes
    ----------
    name : str
        Provider key (``"fake"``) used to select this provider.
    """

    name = "fake"

    def __init__(
        self,
        *,
        responses: tuple[str, ...] | list[str] | None = None,
        **_ignored: Any,
    ) -> None:
        """Initialize the provider with its canned response sequence.

        Parameters
        ----------
        responses : tuple[str, ...] or list[str] or None, optional
            Ordered strings the fake model will emit. When ``None``, the
            sequence is read as JSON from the ``OPENBB_AGENT_FAKE_RESPONSES``
            environment variable, falling back to :data:`DEFAULT_RESPONSES`
            (``("OK.",)``) when that is unset.
        **_ignored : Any
            Extra keyword arguments are accepted and discarded so the
            provider stays drop-in compatible with real provider configs.
        """
        if responses is None:
            env = os.environ.get("OPENBB_AGENT_FAKE_RESPONSES")
            responses = tuple(json.loads(env)) if env else DEFAULT_RESPONSES
        self._responses = tuple(responses)

    def build(self, ctx: RunContext, config: dict[str, Any]) -> BaseChatModel:
        """Build a fake chat model that replays the configured responses.

        Parameters
        ----------
        ctx : RunContext
            The active run context. Unused; accepted to satisfy the
            :class:`~openbb_agent_server.runtime.plugins.ModelProvider` interface.
        config : dict[str, Any]
            Per-build configuration. A ``"responses"`` key, if present,
            overrides the provider's default response sequence for this
            model only.

        Returns
        -------
        BaseChatModel
            A tool-aware fake chat model that yields one canned
            :class:`AIMessage` per configured response.
        """
        responses = config.get("responses", self._responses)
        return _ToolAwareFakeChatModel(
            messages=iter([AIMessage(content=r) for r in responses])
        )
