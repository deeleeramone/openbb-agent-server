"""Shared services container tests."""

from __future__ import annotations

import pytest

from openbb_agent_server.runtime import services


def test_get_history_before_set_raises() -> None:
    services.reset()
    with pytest.raises(RuntimeError):
        services.get_history()


def test_get_memory_before_set_returns_none() -> None:
    services.reset()
    assert services.get_memory() is None


def test_set_then_get_history_round_trip() -> None:
    class FakeHistory:
        pass

    services.reset()
    fh = FakeHistory()
    services.set_services(history=fh)  # type: ignore[arg-type]
    assert services.get_history() is fh
    services.reset()


def test_set_partial_then_overwrite() -> None:
    services.reset()
    h1, h2 = object(), object()
    services.set_services(history=h1)  # type: ignore[arg-type]
    services.set_services(history=h2)  # type: ignore[arg-type]
    assert services.get_history() is h2
    services.reset()


def test_extra_keys_pass_through() -> None:
    services.reset()
    services.set_services(custom_value=42)
    services.reset()
