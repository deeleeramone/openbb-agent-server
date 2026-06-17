"""Snowflake Cortex model provider."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from openbb_agent_server.runtime.context import RunContext
from openbb_agent_server.runtime.plugins import ModelProvider


def _pick(ctx: RunContext, *names: str) -> str | None:
    for n in names:
        v = ctx.api_keys.get(n)
        if v:
            return v
    return None


class SnowflakeProvider(ModelProvider):
    """Provide chat models backed by Snowflake Cortex.

    Wrap ``langchain_community.chat_models.ChatSnowflakeCortex``, sourcing
    connection details from the run context's API keys when present and
    falling back to the values supplied at construction.
    """

    name = "snowflake"

    def __init__(
        self,
        *,
        model_name: str = "claude-3-5-sonnet",
        cortex_function: str = "complete",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        account: str | None = None,
        user: str | None = None,
        password: str | None = None,
        role: str | None = None,
        warehouse: str | None = None,
        database: str | None = None,
        schema: str | None = None,
        session: Any = None,
        **extra: Any,
    ) -> None:
        """Store the Cortex model and connection defaults.

        Parameters
        ----------
        model_name : str
            Default Cortex model id; overridable per run via ``config``.
        cortex_function : str
            Snowflake Cortex function to invoke, e.g. ``"complete"``.
        temperature : float
            Sampling temperature passed to Cortex.
        max_tokens : int or None
            Maximum tokens to generate; omitted from kwargs when ``None``.
        top_p : float or None
            Nucleus-sampling cutoff; omitted from kwargs when ``None``.
        account : str or None
            Snowflake account identifier, used when the context lacks
            ``SNOWFLAKE_ACCOUNT``.
        user : str or None
            Snowflake username, used when the context lacks
            ``SNOWFLAKE_USERNAME`` / ``SNOWFLAKE_USER``.
        password : str or None
            Snowflake password, used when the context lacks
            ``SNOWFLAKE_PASSWORD``.
        role : str or None
            Default Snowflake role.
        warehouse : str or None
            Default Snowflake warehouse.
        database : str or None
            Default Snowflake database.
        schema : str or None
            Default Snowflake schema.
        session : Any
            Pre-built Snowpark session reused instead of opening a new one.
        **extra : Any
            Additional keyword arguments forwarded verbatim to
            ``ChatSnowflakeCortex``.
        """
        self._model_name = model_name
        self._cortex_function = cortex_function
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._account = account
        self._user = user
        self._password = password
        self._role = role
        self._warehouse = warehouse
        self._database = database
        self._schema = schema
        self._session = session
        self._extra = extra

    def build_kwargs(self, ctx: RunContext, config: dict[str, Any]) -> dict[str, Any]:
        """Construct the kwargs that feed ChatSnowflakeCortex.

        Resolve each connection setting by preferring the context's API
        keys (e.g. ``SNOWFLAKE_ACCOUNT``) over the construction-time
        default, and emit only the kwargs that have a non-empty value.

        Parameters
        ----------
        ctx : RunContext
            Active run context whose ``api_keys`` supply credentials and
            connection overrides.
        config : dict of str to Any
            Per-run configuration; ``"model_name"`` overrides the default
            Cortex model when present.

        Returns
        -------
        dict of str to Any
            Keyword arguments ready to pass to ``ChatSnowflakeCortex``.
        """
        kwargs: dict[str, Any] = {
            "model": config.get("model_name", self._model_name),
            "cortex_function": self._cortex_function,
            "temperature": self._temperature,
            **self._extra,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p

        if self._session is not None:
            kwargs["session"] = self._session

        account = _pick(ctx, "SNOWFLAKE_ACCOUNT") or self._account
        user = _pick(ctx, "SNOWFLAKE_USERNAME", "SNOWFLAKE_USER") or self._user
        password = _pick(ctx, "SNOWFLAKE_PASSWORD") or self._password
        if account:
            kwargs["snowflake_account"] = account
        if user:
            kwargs["snowflake_username"] = user
        if password:
            kwargs["snowflake_password"] = password
        for ctx_name, kw_name, default in (
            ("SNOWFLAKE_ROLE", "snowflake_role", self._role),
            ("SNOWFLAKE_WAREHOUSE", "snowflake_warehouse", self._warehouse),
            ("SNOWFLAKE_DATABASE", "snowflake_database", self._database),
            ("SNOWFLAKE_SCHEMA", "snowflake_schema", self._schema),
        ):
            v = _pick(ctx, ctx_name) or default
            if v:
                kwargs[kw_name] = v
        return kwargs

    def build(  # pragma: no cover — ChatSnowflakeCortex opens a live Snowpark session at construction
        self, ctx: RunContext, config: dict[str, Any]
    ) -> BaseChatModel:
        """Build a ``ChatSnowflakeCortex`` chat model for this run.

        Parameters
        ----------
        ctx : RunContext
            Active run context supplying credentials via ``api_keys``.
        config : dict of str to Any
            Per-run configuration forwarded to :meth:`build_kwargs`.

        Returns
        -------
        BaseChatModel
            A configured ``ChatSnowflakeCortex`` instance.

        Raises
        ------
        RuntimeError
            If ``langchain-community`` and the Snowflake connector are not
            installed (install the ``[snowflake]`` extra).
        """
        try:
            from langchain_community.chat_models import ChatSnowflakeCortex
        except ImportError as exc:
            raise RuntimeError(
                "SnowflakeProvider requires langchain-community + "
                "snowflake-connector-python. Install the agent_server "
                "with the [snowflake] extra."
            ) from exc

        return ChatSnowflakeCortex(**self.build_kwargs(ctx, config))
