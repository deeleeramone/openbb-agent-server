"""Trace-aware structured logging with PII redaction + ``TRACE`` level."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openbb_agent_server.runtime import context as run_context
from openbb_agent_server.runtime.identity import redact_email_in_text

TRACE = 5

if logging.getLevelName(TRACE) == f"Level {TRACE}":
    logging.addLevelName(TRACE, "TRACE")

LOG_LEVEL_ENV = "OPENBB_AGENT_LOG_LEVEL"

_NOISY_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "aiosqlite",
    "asyncio",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "httpcore",
    "httpx",
    "urllib3",
    "watchfiles",
    "python_multipart",
    "multipart",
)


def resolve_level(level: int | str | None) -> int:
    """Resolve a level int or name to a numeric logging level.

    Parameters
    ----------
    level : int or str or None
        A numeric level, a level name such as ``"DEBUG"``, or ``None``.
        When ``None``, the ``OPENBB_AGENT_LOG_LEVEL`` environment variable
        is consulted, falling back to :mod:`logging.INFO`.

    Returns
    -------
    int
        The numeric logging level. Unrecognized names resolve to
        :mod:`logging.INFO`.
    """
    if level is None:
        level = os.environ.get(LOG_LEVEL_ENV) or logging.INFO
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).strip().upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def trace(logger: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Emit a ``TRACE``-level record on ``logger``.

    The record is only built and logged when ``logger`` is enabled for the
    custom ``TRACE`` level (numeric value 5, below ``DEBUG``).

    Parameters
    ----------
    logger : logging.Logger
        The logger to emit the record on.
    message : str
        The log message, optionally containing ``%``-style placeholders.
    *args : Any
        Positional arguments interpolated into ``message``.
    **kwargs : Any
        Keyword arguments forwarded to `logging.Logger.log`
        (for example ``exc_info`` or ``stacklevel``).
    """
    if logger.isEnabledFor(TRACE):
        logger.log(TRACE, message, *args, **kwargs)


_BEARER_RE = re.compile(r"(?i)Bearer\s+[A-Za-z0-9._\-+/=]+")
_AUTH_HEADER_RE = re.compile(r"(?i)(authorization)\s*[:=]\s*[A-Za-z0-9._\-+/= ]+")
_API_KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|x-api-key|sk-[A-Za-z0-9]+|nvapi-[A-Za-z0-9]+|"
    r"grok-[A-Za-z0-9]+|gsk_[A-Za-z0-9]+|tvly-[A-Za-z0-9]+)[A-Za-z0-9._\-]*"
)


def redact_pii(text: str) -> str:
    """Strip emails, bearer tokens, and API-key-shaped strings from ``text``.

    Emails are replaced via :func:`~openbb_agent_server.runtime.identity.redact_email_in_text`; ``Bearer`` tokens
    and ``Authorization`` headers are masked; and strings shaped like common
    API keys (``sk-``, ``nvapi-``, ``grok-``, ``gsk_``, ``tvly-``,
    ``api_key``, ``x-api-key``) are replaced with ``<redacted-key>``.

    Parameters
    ----------
    text : str
        The text to redact.

    Returns
    -------
    str
        The redacted text. Non-string or empty input is returned unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    out = redact_email_in_text(text)
    out = _BEARER_RE.sub("Bearer <redacted>", out)
    out = _AUTH_HEADER_RE.sub(r"\1: <redacted>", out)
    out = _API_KEY_RE.sub("<redacted-key>", out)
    return out


class TraceContextFilter(logging.Filter):
    """Inject trace IDs from the contextvar into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach trace identifiers from the current run context to ``record``.

        Reads the active :class:`~openbb_agent_server.runtime.context.RunContext` (if any) and sets
        ``trace_id``, ``run_id``, ``conversation_id``, ``user_id``, and a
        combined ``trace`` mapping on the record. When no context is active,
        these are set to empty values.

        Parameters
        ----------
        record : logging.LogRecord
            The record to enrich in place.

        Returns
        -------
        bool
            Always ``True`` so the record is never dropped.
        """
        try:
            ctx = run_context.current()
        except LookupError:
            ctx = None
        if ctx is not None:
            record.trace_id = ctx.trace_id
            record.run_id = ctx.run_id
            record.conversation_id = ctx.conversation_id
            record.user_id = ctx.principal.user_id
            record.trace = {
                "trace_id": ctx.trace_id,
                "run_id": ctx.run_id,
                "conversation_id": ctx.conversation_id,
                "user_id": ctx.principal.user_id,
            }
        else:
            record.trace_id = ""
            record.run_id = ""
            record.conversation_id = ""
            record.user_id = ""
            record.trace = {}
        return True


class PIIRedactionFilter(logging.Filter):
    """Redact emails / bearer tokens / API keys from every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact PII from ``record``'s message and interpolation args in place.

        The string ``record.msg`` is passed through :func:`redact_pii`, and
        any positional/dict ``record.args`` are recursively redacted so that
        secrets cannot leak through ``%``-style formatting.

        Parameters
        ----------
        record : logging.LogRecord
            The record to redact in place.

        Returns
        -------
        bool
            Always ``True`` so the record is never dropped.
        """
        msg = record.msg
        if isinstance(msg, str):
            record.msg = redact_pii(msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = _redact_arg(record.args)
            else:
                record.args = tuple(_redact_arg(a) for a in record.args)
        return True


def _redact_arg(arg: Any) -> Any:
    if isinstance(arg, str):
        return redact_pii(arg)
    if isinstance(arg, dict):
        return {k: _redact_arg(v) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        seq = [_redact_arg(v) for v in arg]
        return type(arg)(seq) if isinstance(arg, tuple) else seq
    return arg


class JsonTraceFormatter(logging.Formatter):
    """One-line JSON formatter that surfaces the trace IDs."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a single-line JSON object.

        The payload carries ``level``, ``logger``, the redacted
        ``message``, and the ``trace`` mapping attached by
        :class:`TraceContextFilter`. When the record has exception info, a
        redacted ``exc_info`` traceback string is included.

        Parameters
        ----------
        record : logging.LogRecord
            The record to format.

        Returns
        -------
        str
            A JSON document serialized with ``default=str`` for
            non-serializable values.
        """
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": redact_pii(record.getMessage()),
            "trace": getattr(record, "trace", {}),
        }
        if record.exc_info:
            payload["exc_info"] = redact_pii(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def _quiet_noisy_loggers(root_level: int) -> None:
    """Cap chatty third-party loggers at WARNING or coarser."""
    quiet_level = max(root_level, logging.WARNING)
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(quiet_level)


def install_trace_logging(level: int | str | None = None) -> None:
    """Attach trace and redaction filters and a JSON formatter to root.

    Set the root logger level, cap noisy third-party loggers at ``WARNING``
    or coarser, and install :class:`TraceContextFilter` plus
    :class:`PIIRedactionFilter` on the root logger and on a new
    :mod:`logging.StreamHandler` that uses :class:`JsonTraceFormatter`.
    The call is idempotent: if a :class:`TraceContextFilter` is already
    present on the root logger, it returns without adding duplicates.

    Parameters
    ----------
    level : int or str or None, optional
        The desired root level, resolved via :func:`resolve_level`. When
        ``None``, the ``OPENBB_AGENT_LOG_LEVEL`` environment variable or
        :mod:`logging.INFO` is used.
    """
    root = logging.getLogger()
    root_level = resolve_level(level)
    root.setLevel(root_level)
    _quiet_noisy_loggers(root_level)
    if any(isinstance(f, TraceContextFilter) for f in root.filters):
        return
    root.addFilter(TraceContextFilter())
    root.addFilter(PIIRedactionFilter())
    handler = logging.StreamHandler()
    handler.setFormatter(JsonTraceFormatter())
    handler.addFilter(TraceContextFilter())
    handler.addFilter(PIIRedactionFilter())
    root.addHandler(handler)
