"""Bundled system-prompt templates."""

from importlib import resources
from pathlib import Path


def default_system_prompt_path() -> Path:
    """Filesystem path to the bundled default system prompt."""
    return Path(str(resources.files(__name__).joinpath("default_system_prompt.md")))


__all__ = ["default_system_prompt_path"]
