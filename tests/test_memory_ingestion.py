"""Tests for memory ingestion."""

from __future__ import annotations

import base64
from typing import Any

import pytest
from langchain_text_splitters import Language

from openbb_agent_server.memory.ingestion import (
    _decode_file_text,
    _detect_language,
    _iter_long_messages,
    _likely_non_english,
    chunk_text,
    ingest_request_context,
)
from openbb_agent_server.memory.store import Memory, MemoryStore
from openbb_agent_server.runtime.principal import UserPrincipal


def _alice_with_scope() -> UserPrincipal:
    return UserPrincipal(user_id="alice", scopes=("memory:write",))


def _alice_no_scope() -> UserPrincipal:
    return UserPrincipal(user_id="alice", scopes=())


class _FakeStore(MemoryStore):
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []
        self.raise_on_call: int | None = None

    async def write(
        self,
        *,
        principal: UserPrincipal,
        text: str,
        kind: str = "fact",
        source_trace_id: str | None = None,
    ) -> Memory:
        if self.raise_on_call is not None and len(self.writes) >= self.raise_on_call:
            raise RuntimeError("simulated write failure")
        self.writes.append(
            {
                "principal_id": principal.user_id,
                "text": text,
                "kind": kind,
                "source_trace_id": source_trace_id,
            }
        )
        return Memory(
            memory_id=f"m{len(self.writes)}",
            user_id=principal.user_id,
            text=text,
            kind=kind,
            source_trace_id=source_trace_id,
        )

    async def recall(self, **_kw: Any) -> list[Memory]:
        return []

    async def list_memories(self, **_kw: Any) -> list[Memory]:
        return []

    async def pin(self, **_kw: Any) -> Memory | None:
        return None

    async def forget(self, **_kw: Any) -> bool:
        return False

    async def delete_all_for_user(self, principal: UserPrincipal) -> int:
        return 0


class _PermissionStore(_FakeStore):
    async def write(self, **kwargs: Any) -> Memory:
        raise PermissionError("nope")


class _StubTranslator:
    def __init__(self, *, translated: str = "TRANSLATED") -> None:
        self.translated = translated
        self.calls: list[tuple[str, str, str]] = []

    async def translate(
        self,
        text: str,
        *,
        source_language: str = "auto",
        target_language: str = "English",
    ) -> str:
        self.calls.append((text, source_language, target_language))
        return self.translated


class _BrokenTranslator:
    async def translate(self, text: str, **_kw: Any) -> str:
        raise RuntimeError("translator down")


def test_likely_non_english_empty_returns_false() -> None:
    assert _likely_non_english("") is False


def test_likely_non_english_with_ascii_only() -> None:
    assert _likely_non_english("hello world from the test") is False


def test_likely_non_english_with_non_ascii_dense() -> None:
    assert _likely_non_english("これは日本語です。" * 5) is True


def test_likely_non_english_with_non_english_target_always_translates() -> None:
    assert _likely_non_english("hello", target_lang="Spanish") is True


def test_chunk_text_rejects_zero_size() -> None:
    with pytest.raises(ValueError, match="chunk_chars must be positive"):
        chunk_text("hi", chunk_chars=0)


def test_chunk_text_rejects_invalid_overlap_negative() -> None:
    with pytest.raises(ValueError, match="overlap must be in"):
        chunk_text("hi", chunk_chars=10, overlap=-1)


def test_chunk_text_rejects_invalid_overlap_too_big() -> None:
    with pytest.raises(ValueError, match="overlap must be in"):
        chunk_text("hi", chunk_chars=10, overlap=10)


def test_chunk_text_short_text_returns_single_chunk() -> None:
    assert chunk_text("hello", chunk_chars=100, overlap=10) == ["hello"]


def test_chunk_text_whitespace_only_returns_empty() -> None:
    assert chunk_text("    ", chunk_chars=100, overlap=10) == []


def test_chunk_text_long_text_creates_overlapping_chunks() -> None:
    text = "a" * 100
    chunks = chunk_text(text, chunk_chars=30, overlap=10)
    assert len(chunks) >= 3
    assert all(len(c) <= 30 for c in chunks)


def test_chunk_text_terminates_on_last_chunk() -> None:
    text = "x" * 50
    chunks = chunk_text(text, chunk_chars=20, overlap=5)
    assert len(chunks) > 0


def test_chunk_text_with_language_uses_language_splitter() -> None:
    """Use a syntactic splitter for a known language."""
    code = (
        "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n"
    )
    chunks = chunk_text(code, chunk_chars=30, overlap=5, language=Language.PYTHON)
    assert len(chunks) > 1


def test_detect_language_python() -> None:
    assert _detect_language("module.py") is Language.PYTHON


def test_detect_language_uppercase_extension() -> None:
    assert _detect_language("X.PY") is Language.PYTHON


def test_detect_language_unknown_extension_returns_none() -> None:
    assert _detect_language("readme.unknownext") is None


def test_detect_language_no_filename_returns_none() -> None:
    assert _detect_language(None) is None
    assert _detect_language("") is None


def test_decode_file_text_no_b64_returns_none() -> None:
    assert _decode_file_text("x.txt", "text/plain", None) is None


def test_decode_file_text_non_text_mime_returns_none() -> None:
    assert _decode_file_text("x.bin", "application/octet-stream", "Zm9v") is None


def test_decode_file_text_text_mime_decodes_utf8() -> None:
    b64 = base64.b64encode(b"hello").decode()
    assert _decode_file_text("x.txt", "text/plain", b64) == "hello"


def test_decode_file_text_application_json_mime() -> None:
    b64 = base64.b64encode(b"{}").decode()
    assert _decode_file_text("x.json", "application/json", b64) == "{}"


def test_decode_file_text_extension_match() -> None:
    """Trigger decode from a known text extension."""
    b64 = base64.b64encode(b"# header").decode()
    assert _decode_file_text("README.md", "application/octet-stream", b64) == "# header"


def test_decode_file_text_invalid_b64_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert _decode_file_text("x.txt", "text/plain", "!!! not valid base64 !!!") is None
    assert any("base64 decode failed" in r.message for r in caplog.records)


def test_decode_file_text_latin1_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to latin-1 for non-UTF-8 bytes when no loader handles the file."""
    from openbb_agent_server.memory import ingestion

    monkeypatch.setattr(ingestion, "_load_via_loader", lambda path, ext: None)
    raw = bytes([0xC0, 0xC1, 0xC2])
    b64 = base64.b64encode(raw).decode()
    out = _decode_file_text("x.txt", "text/plain", b64)
    assert isinstance(out, str)
    assert len(out) == 3


def test_decode_file_text_csv_uses_csvloader() -> None:
    """Decode CSV files through CSVLoader."""
    raw = b"name,age\nAlice,30\nBob,25\n"
    b64 = base64.b64encode(raw).decode()
    out = _decode_file_text("people.csv", "text/csv", b64)
    assert out is not None
    assert "Alice" in out
    assert "Bob" in out


def test_decode_file_text_tsv_uses_csvloader_with_tab_sep() -> None:
    raw = b"name\tage\nAlice\t30\n"
    b64 = base64.b64encode(raw).decode()
    out = _decode_file_text("people.tsv", "text/tab-separated-values", b64)
    assert out is not None
    assert "Alice" in out


def test_decode_file_text_psv_uses_csvloader_with_pipe_sep() -> None:
    raw = b"name|age\nAlice|30\n"
    b64 = base64.b64encode(raw).decode()
    out = _decode_file_text("people.psv", "text/plain", b64)
    assert out is not None
    assert "Alice" in out


def test_decode_file_text_html_uses_bs_loader() -> None:
    raw = b"<html><body><p>hello world</p></body></html>"
    b64 = base64.b64encode(raw).decode()
    out = _decode_file_text("page.html", "text/html", b64)
    assert out is not None
    assert "hello world" in out


def test_decode_file_text_loader_failure_falls_back_to_plain_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to plain decode when the loader fails."""
    from openbb_agent_server.memory import ingestion

    def broken(*_a: object, **_kw: object) -> None:
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(ingestion, "_load_via_loader", broken)
    raw = b"plain text content"
    b64 = base64.b64encode(raw).decode()
    out = _decode_file_text("note.txt", "text/plain", b64)
    assert out == "plain text content"


def test_iter_long_messages_skips_short() -> None:
    class _M:
        role = "human"
        content = "short"

    out = list(_iter_long_messages([_M()], threshold=100))
    assert out == []


def test_iter_long_messages_skips_non_human() -> None:
    class _M:
        role = "ai"
        content = "a" * 200

    out = list(_iter_long_messages([_M()], threshold=10))
    assert out == []


def test_iter_long_messages_handles_dict_messages() -> None:
    out = list(
        _iter_long_messages(
            [{"role": "human", "content": "x" * 200}],
            threshold=50,
        )
    )
    assert out == [(0, "x" * 200)]


def test_iter_long_messages_skips_non_string_content() -> None:
    class _M:
        role = "human"
        content = ["x" * 200]

    out = list(_iter_long_messages([_M()], threshold=10))
    assert out == []


def test_iter_long_messages_dict_content_field() -> None:
    out = list(
        _iter_long_messages(
            [{"role": "human", "content": "x" * 200}],
            threshold=10,
        )
    )
    assert out == [(0, "x" * 200)]


class _FakeBody:
    def __init__(
        self, *, uploaded_files: list[Any] = (), messages: list[Any] = ()
    ) -> None:
        self.uploaded_files = list(uploaded_files)
        self.messages = list(messages)


class _FakeFile:
    def __init__(
        self,
        *,
        name: str,
        mime: str,
        data_base64: str | None,
    ) -> None:
        self.name = name
        self.mime = mime
        self.data_base64 = data_base64


@pytest.mark.asyncio
async def test_ingest_skips_when_no_memory_write_scope(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    store = _FakeStore()
    with caplog.at_level(logging.DEBUG):
        out = await ingest_request_context(
            principal=_alice_no_scope(),
            store=store,
            body=_FakeBody(),
            trace_id="t",
        )
    assert out == 0
    assert store.writes == []
    assert any("lacks memory:write" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_ingest_no_sources_returns_zero() -> None:
    store = _FakeStore()
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=_FakeBody(),
        trace_id="t",
    )
    assert out == 0
    assert store.writes == []


@pytest.mark.asyncio
async def test_ingest_short_messages_not_recorded() -> None:
    store = _FakeStore()
    body = _FakeBody(messages=[type("M", (), {"role": "human", "content": "hi"})()])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=1000,
    )
    assert out == 0


@pytest.mark.asyncio
async def test_ingest_uploads_text_file() -> None:
    store = _FakeStore()
    long_text = "abc " * 600
    b64 = base64.b64encode(long_text.encode()).decode()
    file = _FakeFile(name="notes.txt", mime="text/plain", data_base64=b64)
    body = _FakeBody(uploaded_files=[file])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t-1",
        char_threshold=100,
        chunk_chars=500,
        chunk_overlap=50,
    )
    assert out > 0
    assert all(w["principal_id"] == "alice" for w in store.writes)
    assert all(w["kind"] == "context_text" for w in store.writes)
    assert all(w["source_trace_id"] == "t-1" for w in store.writes)


@pytest.mark.asyncio
async def test_ingest_long_human_message() -> None:
    store = _FakeStore()
    msg_text = "z" * 5000

    class _M:
        role = "human"
        content = msg_text

    body = _FakeBody(messages=[_M()])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=100,
        chunk_chars=1000,
        chunk_overlap=100,
    )
    assert out >= 5
    labels = [w["text"].split("]", 1)[0] for w in store.writes]
    assert all("message:0" in lbl for lbl in labels)


@pytest.mark.asyncio
async def test_ingest_code_file_uses_code_kind() -> None:
    store = _FakeStore()
    code = "def f():\n    return 1\n" * 200
    b64 = base64.b64encode(code.encode()).decode()
    file = _FakeFile(name="x.py", mime="text/x-python", data_base64=b64)
    body = _FakeBody(uploaded_files=[file])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=100,
        chunk_chars=500,
        chunk_overlap=50,
    )
    assert out > 0
    assert all(w["kind"] == "context_code" for w in store.writes)


@pytest.mark.asyncio
async def test_ingest_translates_non_english_chunks() -> None:
    store = _FakeStore()
    translator = _StubTranslator(translated="HOLA-TRANSLATED")
    # Dense non-ASCII text that surpasses the 5%/4000 char heuristic.
    text = "これは日本語です。" * 200
    file = _FakeFile(
        name="notes.txt",
        mime="text/plain",
        data_base64=base64.b64encode(text.encode()).decode(),
    )
    body = _FakeBody(uploaded_files=[file])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=100,
        chunk_chars=200,
        chunk_overlap=20,
        translator=translator,  # type: ignore[arg-type]
    )
    assert out > 0
    assert translator.calls
    assert any("HOLA-TRANSLATED" in w["text"] for w in store.writes)


@pytest.mark.asyncio
async def test_ingest_translation_failure_falls_back_to_original(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore()
    text = "これは日本語です。" * 200
    file = _FakeFile(
        name="notes.txt",
        mime="text/plain",
        data_base64=base64.b64encode(text.encode()).decode(),
    )
    body = _FakeBody(uploaded_files=[file])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=100,
        chunk_chars=200,
        chunk_overlap=20,
        translator=_BrokenTranslator(),  # type: ignore[arg-type]
    )
    assert out > 0
    assert any("translation failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_ingest_translation_skips_code_chunks() -> None:
    store = _FakeStore()
    translator = _StubTranslator()
    code = ("def f():\n" + "  # comment éèà ñ\n" + "  return 1\n") * 200
    file = _FakeFile(
        name="x.py",
        mime="text/x-python",
        data_base64=base64.b64encode(code.encode()).decode(),
    )
    body = _FakeBody(uploaded_files=[file])
    await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=100,
        chunk_chars=500,
        chunk_overlap=50,
        translator=translator,  # type: ignore[arg-type]
    )
    assert translator.calls == []


@pytest.mark.asyncio
async def test_ingest_permission_error_aborts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    store = _PermissionStore()
    long_text = "a" * 1500
    file = _FakeFile(
        name="x.txt",
        mime="text/plain",
        data_base64=base64.b64encode(long_text.encode()).decode(),
    )
    body = _FakeBody(uploaded_files=[file])
    with caplog.at_level(logging.INFO):
        out = await ingest_request_context(
            principal=_alice_with_scope(),
            store=store,
            body=body,
            trace_id="t",
            char_threshold=100,
            chunk_chars=500,
            chunk_overlap=50,
        )
    assert out == 0
    assert any("PermissionError" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_ingest_continues_past_write_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore()
    store.raise_on_call = 1
    long_text = "a" * 3000
    file = _FakeFile(
        name="x.txt",
        mime="text/plain",
        data_base64=base64.b64encode(long_text.encode()).decode(),
    )
    body = _FakeBody(uploaded_files=[file])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=100,
        chunk_chars=500,
        chunk_overlap=50,
    )
    assert out == 1
    assert any("failed to write chunk" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_ingest_uploaded_file_without_name() -> None:
    store = _FakeStore()
    text = "x" * 3000
    file = _FakeFile(
        name="",
        mime="text/plain",
        data_base64=base64.b64encode(text.encode()).decode(),
    )
    body = _FakeBody(uploaded_files=[file])
    out = await ingest_request_context(
        principal=_alice_with_scope(),
        store=store,
        body=body,
        trace_id="t",
        char_threshold=100,
        chunk_chars=500,
        chunk_overlap=50,
    )
    assert out > 0
    assert any("uploaded_file" in w["text"] for w in store.writes)
