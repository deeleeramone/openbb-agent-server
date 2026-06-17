"""Tests for :mod:`openbb_agent_server.memory.classifier.looks_like_code`."""

from __future__ import annotations

from openbb_agent_server.memory.classifier import looks_like_code


def test_extension_python_matches() -> None:
    assert looks_like_code("anything", filename="x.py") is True


def test_extension_typescript_matches() -> None:
    assert looks_like_code("anything", filename="X.TS") is True


def test_extension_dockerfile_with_no_extension_matches() -> None:
    assert looks_like_code("anything", filename="Dockerfile") is True


def test_extension_makefile_with_no_extension_matches() -> None:
    assert looks_like_code("anything", filename="path/to/Makefile") is True


def test_extension_unknown_no_signal() -> None:
    assert looks_like_code("hello world", filename="readme.unknown") is False


def test_mime_application_json_matches() -> None:
    assert looks_like_code("anything", mime="application/json") is True


def test_mime_text_x_python_matches() -> None:
    assert looks_like_code("anything", mime="text/x-python") is True


def test_mime_unknown_no_signal() -> None:
    assert looks_like_code("plain prose", mime="text/plain") is False


def test_empty_text_returns_false() -> None:
    assert looks_like_code("") is False


def test_whitespace_only_returns_false() -> None:
    assert looks_like_code("    \n\t ") is False


def test_code_dense_text_via_density() -> None:
    code = "def f(x): return x + 1\nclass C: pass\nimport os\nif True: pass\n" * 5
    assert looks_like_code(code) is True


def test_prose_returns_false() -> None:
    prose = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a sentence of regular English prose. "
        "There are no programming language tokens here whatsoever."
    ) * 5
    assert looks_like_code(prose) is False


def test_only_first_4000_chars_examined() -> None:
    """Classify a doc with code only past position 4000 as prose."""
    prose_head = "ordinary words " * 300
    code_tail = "def f(): return 1\n" * 200
    assert looks_like_code(prose_head + code_tail) is False
