"""OpenBB Platform Agent Server — pluggable, multi-tenant agent backend."""

from __future__ import annotations

import warnings

__version__ = "0.1.0"

from langchain_core._api.deprecation import (  # noqa: E402
    LangChainPendingDeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change.*",
    category=LangChainPendingDeprecationWarning,
)
