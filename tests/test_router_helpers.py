"""Unit tests for the router helper functions."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest

from openbb_agent_server.app.router import (
    _ai_envelope_from_message_safe,
    _build_pdf_filename,
    _coerce_custom_feature,
    _coerce_feature,
    _collect_uploaded_files,
    _collect_uploaded_files_with_ingest,
    _extract_pdf_b64_from_string,
    _extract_pdf_url_from_string,
    _has_pdf_data_format_marker,
    _is_pdf_ref,
    _looks_like_pdf_ref,
    _params_to_input_args,
    _pdf_ref_from_dict,
    _post_run_extractor,
    _require_scope,
    _rows_from_inline_widget,
    _safe_end_trace,
    _scan_for_http_url,
    _scan_for_pdf_b64,
    _slug_to_label,
    _slugify_filename_segment,
    _string_is_pdf_b64,
    _string_is_pdf_mime,
    _string_is_pdf_url,
    _to_widget_ref,
    _walk_pdf_refs,
)
from openbb_agent_server.app.settings import AgentMetadata, AgentProfile
from openbb_agent_server.observability.logging import TRACE
from openbb_agent_server.protocol.schemas import QueryRequest
from openbb_agent_server.runtime.context import FileRef, RunContext
from openbb_agent_server.runtime.principal import UserPrincipal

_REAL_PDF_BYTES = b"%PDF-1.4\nfake pdf body\n%%EOF"
_REAL_PDF_B64 = base64.b64encode(_REAL_PDF_BYTES).decode("ascii")


class _Param:
    def __init__(
        self, *, name: str | None, current_value: Any = None, default_value: Any = None
    ) -> None:
        self.name = name
        self.current_value = current_value
        self.default_value = default_value


def test_params_to_input_args_empty_list() -> None:
    assert _params_to_input_args([]) == {}


def test_params_to_input_args_skips_unnamed_entries() -> None:
    out = _params_to_input_args([_Param(name=None, current_value="x")])
    assert out == {}


def test_params_to_input_args_prefers_current_value() -> None:
    out = _params_to_input_args(
        [_Param(name="a", current_value="now", default_value="default")]
    )
    assert out == {"a": "now"}


def test_params_to_input_args_falls_back_to_default_value() -> None:
    out = _params_to_input_args(
        [_Param(name="a", current_value=None, default_value="def")]
    )
    assert out == {"a": "def"}


def test_params_to_input_args_dict_passthrough() -> None:
    assert _params_to_input_args({"a": 1}) == {"a": 1}


def test_params_to_input_args_other_returns_empty() -> None:
    assert _params_to_input_args("not a list or dict") == {}
    assert _params_to_input_args(None) == {}


class _Widget:
    def __init__(
        self,
        *,
        uuid: str = "",
        widget_id: str = "",
        origin: str = "",
        params: Any = None,
        data: Any = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.uuid = uuid
        self.widget_id = widget_id
        self.origin = origin
        self.params = params
        self.data = data
        self.name = name
        self.description = description


def test_to_widget_ref_with_list_params() -> None:
    w = _Widget(
        uuid="u-1",
        widget_id="w",
        origin="o",
        params=[_Param(name="a", current_value=1)],
    )
    ref = _to_widget_ref(w)
    assert ref.uuid == "u-1"
    assert ref.params == {"a": 1}


def test_to_widget_ref_with_dict_params() -> None:
    w = _Widget(uuid="u-1", widget_id="w", origin="o", params={"k": "v"})
    ref = _to_widget_ref(w)
    assert ref.params == {"k": "v"}


def test_to_widget_ref_with_other_params() -> None:
    w = _Widget(uuid="u-1", widget_id="w", params="not a list or dict")
    ref = _to_widget_ref(w)
    assert ref.params == {}


def test_to_widget_ref_skips_unnamed_param_in_list() -> None:
    w = _Widget(
        uuid="u-1",
        widget_id="w",
        params=[_Param(name=None, current_value=1), _Param(name="ok", current_value=9)],
    )
    ref = _to_widget_ref(w)
    assert ref.params == {"ok": 9}


def test_to_widget_ref_falls_back_param_default_value() -> None:
    w = _Widget(
        uuid="u-1",
        widget_id="w",
        params=[_Param(name="a", current_value=None, default_value=2)],
    )
    ref = _to_widget_ref(w)
    assert ref.params == {"a": 2}


def test_to_widget_ref_uses_widget_id_when_uuid_blank() -> None:
    w = _Widget(uuid="", widget_id="wid-fall")
    ref = _to_widget_ref(w)
    assert ref.uuid == "wid-fall"


def test_to_widget_ref_includes_name_and_description() -> None:
    w = _Widget(uuid="u", widget_id="w", name="My Widget", description="desc")
    ref = _to_widget_ref(w)
    assert ref.name == "My Widget"
    assert ref.description == "desc"


def test_to_widget_ref_omits_blank_name_and_description() -> None:
    w = _Widget(uuid="u", widget_id="w", name=None, description=None)
    ref = _to_widget_ref(w)
    assert getattr(ref, "name", "") in (None, "", "widget")


def test_coerce_feature_bool_true() -> None:
    assert _coerce_feature(True) is True


def test_coerce_feature_bool_false() -> None:
    assert _coerce_feature(False) is False


def test_coerce_feature_dict_with_default_true() -> None:
    assert _coerce_feature({"default": True, "other": "x"}) is True


def test_coerce_feature_dict_no_default_defaults_false() -> None:
    assert _coerce_feature({}) is False


def test_coerce_feature_truthy_other() -> None:
    assert _coerce_feature("yes") is True
    assert _coerce_feature(1) is True
    assert _coerce_feature(0) is False


def test_coerce_custom_feature_bool_returns_none() -> None:
    assert _coerce_custom_feature("foo", True) is None


def test_coerce_custom_feature_non_dict_returns_none() -> None:
    assert _coerce_custom_feature("foo", "value") is None


def test_coerce_custom_feature_missing_description_returns_none() -> None:
    assert _coerce_custom_feature("foo", {"label": "X"}) is None


def test_coerce_custom_feature_reserved_name_raises() -> None:
    with pytest.raises(ValueError, match="reserved by Workspace"):
        _coerce_custom_feature("web-search", {"description": "x"})


def test_coerce_custom_feature_returns_wire_shape() -> None:
    out = _coerce_custom_feature(
        "deep-research",
        {"label": "Deep Research", "description": "Detailed analysis", "default": True},
    )
    assert out == {
        "label": "Deep Research",
        "default": True,
        "description": "Detailed analysis",
    }


def test_coerce_custom_feature_synthesises_label_from_slug() -> None:
    out = _coerce_custom_feature(
        "deep-research",
        {"description": "Analysis"},
    )
    assert out is not None
    assert out["label"] == "Deep Research"
    assert out["default"] is False


def test_slug_to_label_kebab_case() -> None:
    assert _slug_to_label("deep-research") == "Deep Research"


def test_slug_to_label_single_word() -> None:
    assert _slug_to_label("research") == "Research"


def test_slug_to_label_strips_empty_segments() -> None:
    assert _slug_to_label("--deep---research-") == "Deep Research"


def test_slug_to_label_empty_string() -> None:
    assert _slug_to_label("") == ""


def _profile(provider: str = "missing-provider") -> AgentProfile:
    return AgentProfile(
        name="prof",
        metadata=AgentMetadata(name="x", description="d", image_url=None),
        model_provider=provider,
        model_name="m",
    )


def _ctx() -> RunContext:
    return RunContext(
        principal=UserPrincipal(user_id="u"),
        trace_id="t",
        run_id="r",
        conversation_id="c",
    )


def test_post_run_extractor_returns_none_when_provider_unknown() -> None:
    """An unresolvable provider name produces None."""
    assert _post_run_extractor(_profile("does-not-exist-x9z"), _ctx()) is None


def test_post_run_extractor_returns_none_on_build_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception during provider.build produces None."""

    from openbb_agent_server.runtime import registry

    class _Bad:
        def build(self, _ctx: RunContext, _cfg: dict[str, Any]) -> Any:
            raise RuntimeError("nope")

    def fake_load(_group: str, _name: str, _cfg: dict[str, Any]) -> Any:
        return _Bad()

    monkeypatch.setattr(registry, "load", fake_load)
    assert _post_run_extractor(_profile(), _ctx()) is None


def test_post_run_extractor_returns_provider_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openbb_agent_server.runtime import registry

    sentinel = object()

    class _OK:
        def build(self, _ctx: RunContext, _cfg: dict[str, Any]) -> Any:
            return sentinel

    monkeypatch.setattr(registry, "load", lambda *_a, **_kw: _OK())
    assert _post_run_extractor(_profile(), _ctx()) is sentinel


def test_safe_end_trace_calls_history() -> None:
    """The happy path forwards principal / trace_id / status to ``end_trace``."""
    seen: dict[str, Any] = {}

    class _History:
        async def end_trace(self, **kw: Any) -> None:
            seen.update(kw)

    p = UserPrincipal(user_id="u")
    asyncio.run(_safe_end_trace(_History(), p, "trace-1", "dispatched"))
    assert seen == {"principal": p, "trace_id": "trace-1", "status": "dispatched"}


def test_safe_end_trace_swallows_exception(caplog: pytest.LogCaptureFixture) -> None:
    """A failing ``end_trace`` is logged and never raised."""

    class _History:
        async def end_trace(self, **_kw: Any) -> None:
            raise RuntimeError("boom")

    with caplog.at_level("WARNING"):
        asyncio.run(_safe_end_trace(_History(), UserPrincipal(user_id="u"), "t", "x"))
    assert any("background end_trace failed" in r.message for r in caplog.records)


def test_rows_from_inline_widget_empty() -> None:
    assert _rows_from_inline_widget(None) == []
    assert _rows_from_inline_widget([]) == []


def test_rows_from_inline_widget_plain_row_list() -> None:
    rows = _rows_from_inline_widget([{"a": 1}, {"b": 2}])
    assert rows == [{"a": 1}, {"b": 2}]


def test_rows_from_inline_widget_items_envelope_list() -> None:
    data = [{"items": [{"x": 1}, "bad"]}, {"items": [{"y": 2}]}, {"no_items": 1}]
    assert _rows_from_inline_widget(data) == [{"x": 1}, {"y": 2}]


def test_rows_from_inline_widget_items_envelope_dict() -> None:
    assert _rows_from_inline_widget({"items": [{"z": 9}, 7]}) == [{"z": 9}]


def test_rows_from_inline_widget_dict_without_items() -> None:
    assert _rows_from_inline_widget({"nope": 1}) == []


def test_rows_from_inline_widget_non_dict_first_element() -> None:
    assert _rows_from_inline_widget(["string", "another"]) == []


def test_extract_pdf_b64_non_string_or_empty() -> None:
    assert _extract_pdf_b64_from_string(123) is None  # type: ignore[arg-type]
    assert _extract_pdf_b64_from_string("") is None


def test_extract_pdf_b64_bare_base64() -> None:
    assert _extract_pdf_b64_from_string("  JVBERi0abc") == "JVBERi0abc"


def test_extract_pdf_b64_data_url() -> None:
    out = _extract_pdf_b64_from_string("data:application/pdf;base64,JVBERi0xyz")
    assert out == "JVBERi0xyz"


def test_extract_pdf_b64_data_url_payload_not_pdf() -> None:
    assert _extract_pdf_b64_from_string("data:application/pdf;base64,notpdf") is None


def test_extract_pdf_b64_raw_pdf_bytes_reencoded() -> None:
    out = _extract_pdf_b64_from_string("%PDF-1.4 raw")
    assert out is not None
    assert base64.b64decode(out).startswith(b"%PDF-")


def test_extract_pdf_b64_unrecognised() -> None:
    assert _extract_pdf_b64_from_string("just text") is None


def test_extract_pdf_b64_raw_bytes_encode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside the raw-bytes re-encode path returns ``None``."""
    import base64 as _b64

    def boom(_data: Any) -> Any:
        raise RuntimeError("encode down")

    monkeypatch.setattr(_b64, "b64encode", boom)
    assert _extract_pdf_b64_from_string("%PDF-raw") is None


def test_extract_pdf_url_non_string_or_empty() -> None:
    assert _extract_pdf_url_from_string(None) is None  # type: ignore[arg-type]
    assert _extract_pdf_url_from_string("") is None


def test_extract_pdf_url_http() -> None:
    assert _extract_pdf_url_from_string("  https://x/y.pdf ") == "https://x/y.pdf"


def test_extract_pdf_url_data_url() -> None:
    assert (
        _extract_pdf_url_from_string("data:application/octet-stream;base64,AAA")
        == "data:application/octet-stream;base64,AAA"
    )


def test_extract_pdf_url_plain_text() -> None:
    assert _extract_pdf_url_from_string("not a url") is None


def test_scan_for_http_url_depth_guard() -> None:
    deep: Any = "https://x/y.pdf"
    for _ in range(8):
        deep = {"k": deep}
    assert _scan_for_http_url(deep) is None


def test_scan_for_http_url_in_dict_and_list() -> None:
    assert _scan_for_http_url({"a": {"b": "https://x/z.pdf"}}) == "https://x/z.pdf"
    assert _scan_for_http_url(["x", ["https://x/q.pdf"]]) == "https://x/q.pdf"


def test_scan_for_http_url_no_match() -> None:
    assert _scan_for_http_url({"a": 1, "b": [None, 2]}) is None


def test_scan_for_pdf_b64_depth_guard() -> None:
    deep: Any = "JVBERi0abc"
    for _ in range(8):
        deep = [deep]
    assert _scan_for_pdf_b64(deep) is None


def test_scan_for_pdf_b64_in_dict_and_list() -> None:
    assert _scan_for_pdf_b64({"a": {"b": "JVBERi0deep"}}) == "JVBERi0deep"
    assert _scan_for_pdf_b64(["nope", ["JVBERi0list"]]) == "JVBERi0list"


def test_scan_for_pdf_b64_no_match() -> None:
    assert _scan_for_pdf_b64({"a": 1, "b": (None,)}) is None


def test_string_is_pdf_url() -> None:
    assert _string_is_pdf_url("https://x/a.pdf") is True
    assert _string_is_pdf_url("https://x/a.txt") is False
    assert _string_is_pdf_url("ftp://x/a.pdf") is False
    assert _string_is_pdf_url(123) is False  # type: ignore[arg-type]


def test_string_is_pdf_mime() -> None:
    assert _string_is_pdf_mime("application/pdf") is True
    assert _string_is_pdf_mime("text/plain") is False
    assert _string_is_pdf_mime(None) is False  # type: ignore[arg-type]


def test_string_is_pdf_b64() -> None:
    assert _string_is_pdf_b64("JVBERi0abc") is True
    assert _string_is_pdf_b64("short") is False
    assert _string_is_pdf_b64(123) is False  # type: ignore[arg-type]
    assert _string_is_pdf_b64("not-a-pdf-base64") is False


def test_has_pdf_data_format_marker_string() -> None:
    assert _has_pdf_data_format_marker({"data_format": "PdfDataFormat"}) is True
    assert _has_pdf_data_format_marker({"data_format": "csv"}) is False


def test_has_pdf_data_format_marker_dict() -> None:
    assert _has_pdf_data_format_marker({"data_format": {"kind": "pdf"}}) is True
    assert _has_pdf_data_format_marker({"data_format": {"kind": "json"}}) is False


def test_has_pdf_data_format_marker_absent() -> None:
    assert _has_pdf_data_format_marker({"other": 1}) is False


def test_looks_like_pdf_ref_by_data_format() -> None:
    assert _looks_like_pdf_ref({"data_format": "PdfDataFormat"}) is True


def test_looks_like_pdf_ref_by_url() -> None:
    assert _looks_like_pdf_ref({"url": "https://x/a.pdf"}) is True


def test_looks_like_pdf_ref_by_mime() -> None:
    assert _looks_like_pdf_ref({"mime": "application/pdf"}) is True


def test_looks_like_pdf_ref_by_b64_field() -> None:
    assert _looks_like_pdf_ref({"base64": "JVBERi0abc"}) is True


def test_looks_like_pdf_ref_by_filename() -> None:
    assert _looks_like_pdf_ref({"name": "report.PDF"}) is True


def test_looks_like_pdf_ref_value_fallback_b64() -> None:
    assert _looks_like_pdf_ref({"weird_key": "JVBERi0value"}) is True


def test_looks_like_pdf_ref_value_fallback_data_url() -> None:
    assert _looks_like_pdf_ref({"x": "data:application/pdf;base64,Q"}) is True


def test_looks_like_pdf_ref_value_fallback_pdf_url() -> None:
    assert _looks_like_pdf_ref({"x": "https://host/some.pdf"}) is True


def test_looks_like_pdf_ref_negative() -> None:
    assert _looks_like_pdf_ref({"name": "data.csv", "value": 7}) is False


def test_looks_like_pdf_ref_value_fallback_non_pdf_url() -> None:
    """A non-PDF http URL value does not flip the value-based fallback."""
    assert _looks_like_pdf_ref({"x": "https://host/page.html"}) is False


def test_pdf_ref_from_dict_non_dict_or_not_pdf() -> None:
    assert _pdf_ref_from_dict("string") is None  # type: ignore[arg-type]
    assert _pdf_ref_from_dict({"name": "x.csv"}) is None


def test_pdf_ref_from_dict_url_and_name() -> None:
    ref = _pdf_ref_from_dict({"url": "https://x/report.pdf", "name": "Q3"})
    assert ref == {
        "name": "Q3.pdf",
        "url": "https://x/report.pdf",
        "data_base64": None,
        "mime": "application/pdf",
    }


def test_pdf_ref_from_dict_url_scanned_from_nested() -> None:
    ref = _pdf_ref_from_dict(
        {"data_format": "PdfDataFormat", "nested": {"link2": "https://x/a.pdf"}}
    )
    assert ref is not None
    assert ref["url"] == "https://x/a.pdf"
    assert ref["name"] == "a.pdf"


def test_pdf_ref_from_dict_name_from_url_with_query() -> None:
    ref = _pdf_ref_from_dict({"url": "https://x/doc.pdf?token=abc"})
    assert ref is not None
    assert ref["name"] == "doc.pdf"


def test_pdf_ref_from_dict_b64_only_name_hashed() -> None:
    ref = _pdf_ref_from_dict({"data_format": {"kind": "pdf"}, "base64": "JVBERi0xyz"})
    assert ref is not None
    assert ref["data_base64"] == "JVBERi0xyz"
    assert ref["name"].startswith("document_")
    assert ref["name"].endswith(".pdf")


def test_pdf_ref_from_dict_no_url_no_b64_returns_none() -> None:
    assert _pdf_ref_from_dict({"data_format": "PdfDataFormat"}) is None


def test_pdf_ref_from_dict_name_from_short_string_value() -> None:
    ref = _pdf_ref_from_dict(
        {"data_format": {"kind": "pdf"}, "base64": "JVBERi0aaa", "ref": "doc-key"}
    )
    assert ref is not None
    assert ref["name"] == "doc-key.pdf"


def test_pdf_ref_from_dict_skips_long_and_url_string_values_for_name() -> None:
    ref = _pdf_ref_from_dict(
        {
            "data_format": {"kind": "pdf"},
            "base64": "JVBERi0aaa",
            "huge": "z" * 300,
            "blank": "   ",
        }
    )
    assert ref is not None
    assert ref["name"].startswith("document_")


def test_pdf_ref_from_dict_name_fallback_skips_url_valued_string() -> None:
    """A non-PDF data-URL string value is skipped by the name scan."""
    ref = _pdf_ref_from_dict(
        {
            "data_format": {"kind": "pdf"},
            "base64": "JVBERi0aaa",
            "thumb": "data:image/png;base64,iVBOR",
        }
    )
    assert ref is not None
    assert ref["name"].startswith("document_")


def test_pdf_ref_from_dict_mime_without_slash() -> None:
    ref = _pdf_ref_from_dict({"url": "https://x/a.pdf", "mime": "pdf", "name": "doc"})
    assert ref is not None
    assert ref["mime"] == "application/pdf"


def test_pdf_ref_from_dict_mime_with_slash() -> None:
    ref = _pdf_ref_from_dict(
        {"url": "https://x/a.pdf", "content_type": "application/pdf", "name": "doc"}
    )
    assert ref is not None
    assert ref["mime"] == "application/pdf"


def test_walk_pdf_refs_none_and_scalar() -> None:
    assert _walk_pdf_refs(None) == []
    assert _walk_pdf_refs(42) == []


def test_walk_pdf_refs_direct_dict() -> None:
    refs = _walk_pdf_refs({"url": "https://x/a.pdf", "name": "A"})
    assert len(refs) == 1
    assert refs[0]["name"] == "A.pdf"


def test_walk_pdf_refs_nested_dict_recurses() -> None:
    refs = _walk_pdf_refs({"wrapper": {"url": "https://x/b.pdf", "name": "B"}})
    assert len(refs) == 1
    assert refs[0]["name"] == "B.pdf"


def test_walk_pdf_refs_list_envelope() -> None:
    refs = _walk_pdf_refs(
        {"items": [{"url": "https://x/c.pdf"}, {"url": "https://x/d.pdf"}]}
    )
    assert {r["url"] for r in refs} == {"https://x/c.pdf", "https://x/d.pdf"}


def test_ai_envelope_from_message_safe_handles_none() -> None:
    assert _ai_envelope_from_message_safe(object()) is None


def test_ai_envelope_from_message_safe_swallows_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside the underlying detector yields None."""
    import openbb_agent_server.runtime.widget_store as ws

    def boom(_msg: Any) -> Any:
        raise RuntimeError("detector down")

    monkeypatch.setattr(ws, "_ai_envelope_from_message", boom)
    assert _ai_envelope_from_message_safe(object()) is None


def test_slugify_filename_segment_list_uses_first_nonempty() -> None:
    assert _slugify_filename_segment(["", "  ", "US|IEFA"]) == "US_IEFA"


def test_slugify_filename_segment_list_all_empty() -> None:
    assert _slugify_filename_segment(["", None]) == ""


def test_slugify_filename_segment_none_and_blank() -> None:
    assert _slugify_filename_segment(None) == ""
    assert _slugify_filename_segment("   ") == ""


def test_slugify_filename_segment_truncates_to_64() -> None:
    assert len(_slugify_filename_segment("a" * 200)) == 64


def test_build_pdf_filename_with_params() -> None:
    name = _build_pdf_filename("blk_docs", {"ticker": "IEFA", "doc_name": "prospectus"})
    assert name == "blk_docs-IEFA-prospectus.pdf"


def test_build_pdf_filename_dedupes_param_segments() -> None:
    name = _build_pdf_filename("w", {"ticker": "AAPL", "symbol": "AAPL"})
    assert name == "w-AAPL.pdf"


def test_build_pdf_filename_no_params_falls_back_to_hint() -> None:
    assert _build_pdf_filename("just_hint", None) == "just_hint.pdf"


def test_build_pdf_filename_empty_everything() -> None:
    assert _build_pdf_filename("", {}) == "document.pdf"


def test_is_pdf_ref_by_mime() -> None:
    assert _is_pdf_ref(FileRef(name="x", mime="application/pdf")) is True


def test_is_pdf_ref_by_name() -> None:
    assert _is_pdf_ref(FileRef(name="report.pdf", mime="application/octet-stream"))


def test_is_pdf_ref_negative() -> None:
    assert _is_pdf_ref(FileRef(name="data.csv", mime="text/csv")) is False


def test_require_scope_passes_when_present() -> None:
    p = UserPrincipal(user_id="u", scopes=("agent:query",))
    _require_scope(p, "agent:query")


def test_require_scope_raises_403_when_missing() -> None:
    from fastapi import HTTPException

    p = UserPrincipal(user_id="u", scopes=())
    with pytest.raises(HTTPException) as exc:
        _require_scope(p, "memory:write")
    assert exc.value.status_code == 403


def _query(**kw: Any) -> QueryRequest:
    base: dict[str, Any] = {"messages": []}
    base.update(kw)
    return QueryRequest.model_validate(base)


def test_collect_uploaded_files_explicit_pdf() -> None:
    body = _query(
        uploaded_files=[
            {"name": "a.pdf", "mime": "application/pdf", "url": "https://x/a.pdf"}
        ]
    )
    refs = _collect_uploaded_files(body)
    assert len(refs) == 1
    assert refs[0].name == "a.pdf"


def test_collect_uploaded_files_dedupes() -> None:
    body = _query(
        uploaded_files=[
            {"name": "a.pdf", "url": "https://x/a.pdf"},
            {"name": "a.pdf", "url": "https://x/a.pdf"},
        ]
    )
    assert len(_collect_uploaded_files(body)) == 1


def test_collect_uploaded_files_recovers_url_from_extras(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-canonical download_url extra is recovered by the scan."""
    body = _query(uploaded_files=[{"name": "a.pdf", "download_url": "https://x/a.pdf"}])
    with caplog.at_level(TRACE):
        refs = _collect_uploaded_files(body)
    assert refs[0].url == "https://x/a.pdf"
    assert any("recovered url" in r.message for r in caplog.records)


def test_collect_uploaded_files_recovers_b64_from_extras(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _query(uploaded_files=[{"name": "a.pdf", "raw_bytes": "JVBERi0recovered"}])
    with caplog.at_level(TRACE):
        refs = _collect_uploaded_files(body)
    assert refs[0].data_base64 == "JVBERi0recovered"
    assert any("recovered base64" in r.message for r in caplog.records)


def test_collect_uploaded_files_no_url_no_b64_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An uploaded file with no resolvable bytes logs a shape sketch."""
    body = _query(
        uploaded_files=[
            {
                "name": "a.pdf",
                "str_extra": "x" * 100,
                "list_extra": [{"k": 1}],
                "empty_list": [],
                "dict_extra": {"k": "v"},
                "int_extra": 7,
            }
        ]
    )
    with caplog.at_level("WARNING"):
        refs = _collect_uploaded_files(body)
    assert refs[0].name == "a.pdf"
    assert any("neither url nor data_base64" in r.message for r in caplog.records)


def test_collect_uploaded_files_from_widget_data() -> None:
    body = _query(
        widgets={
            "primary": [
                {
                    "uuid": "w-1",
                    "widget_id": "doc-widget",
                    "data": [{"url": "https://x/w.pdf", "name": "W"}],
                }
            ],
            "secondary": [],
            "extra": [],
        }
    )
    refs = _collect_uploaded_files(body)
    pdf = next(r for r in refs if r.name == "W.pdf")
    assert pdf.model_extra is not None
    assert pdf.model_extra.get("source_widget_uuid") == "w-1"
    assert pdf.model_extra.get("source_widget_id") == "doc-widget"


def test_collect_uploaded_files_dedupes_same_pdf_across_widgets() -> None:
    """The same PDF in two widgets is de-duplicated."""
    pdf = {"url": "https://x/shared.pdf", "name": "Shared"}
    body = _query(
        widgets={
            "primary": [
                {"uuid": "w-1", "widget_id": "doc", "data": [pdf]},
                {"uuid": "w-2", "widget_id": "doc", "data": [pdf]},
            ],
            "secondary": [],
            "extra": [],
        }
    )
    refs = _collect_uploaded_files(body)
    assert sum(1 for r in refs if r.name == "Shared.pdf") == 1


def test_collect_uploaded_files_from_tool_message_with_hint() -> None:
    """A tool-message PDF gets a meaningful filename from the AI envelope hint."""
    body = _query(
        messages=[
            {
                "role": "ai",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "id": "blk_docs",
                            "widget_uuid": "uuid-9",
                            "input_args": {"ticker": "IEFA"},
                        }
                    ]
                },
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "data": [{"data_format": {"kind": "pdf"}, "content": _REAL_PDF_B64}],
            },
        ]
    )
    refs = _collect_uploaded_files(body)
    pdf = next(r for r in refs if r.name.endswith(".pdf"))
    assert "blk_docs" in pdf.name
    assert pdf.model_extra is not None
    assert pdf.model_extra.get("source_widget_uuid") == "uuid-9"


def test_collect_uploaded_files_tool_message_resets_hint_on_human() -> None:
    """A non-ai / non-tool message clears the pending widget hint."""
    body = _query(
        messages=[
            {
                "role": "ai",
                "function": "get_widget_data",
                "input_arguments": {"data_sources": [{"id": "h"}]},
            },
            {"role": "human", "content": "interleaved"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "data": [{"url": "https://x/t.pdf", "name": "document_x"}],
            },
        ]
    )
    refs = _collect_uploaded_files(body)
    assert any(r.name == "document_x.pdf" for r in refs)


def test_collect_uploaded_files_from_context_channel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PDFs riding on body.context are promoted."""
    body = _query(context=[{"url": "https://x/ctx.pdf", "name": "Ctx"}])
    with caplog.at_level(TRACE):
        refs = _collect_uploaded_files(body)
    assert any(r.name == "Ctx.pdf" for r in refs)
    assert any("promoted" in r.message for r in caplog.records)


def test_collect_uploaded_files_from_body_extra_channel() -> None:
    """An unknown top-level body field carrying a PDF is also scanned."""
    body = _query(custom_field={"url": "https://x/extra.pdf", "name": "E"})
    refs = _collect_uploaded_files(body)
    assert any(r.name == "E.pdf" for r in refs)


def test_collect_uploaded_files_zero_pdf_with_doc_widget_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A document widget that yields no PDF refs surfaces a diagnostic."""
    body = _query(
        widgets={
            "primary": [
                {"uuid": "w", "widget_id": "document-list", "data": [{"x": 1}]}
            ],
            "secondary": [],
            "extra": [],
        }
    )
    with caplog.at_level(TRACE, logger="openbb_agent_server.router"):
        _collect_uploaded_files(body)
    assert any("zero PDF refs resolved" in r.message for r in caplog.records)


def test_collect_with_ingest_no_pdf_store_returns_refs() -> None:
    """When no PdfStore is bound, the refs are returned untouched."""
    body = _query(uploaded_files=[{"name": "a.pdf", "url": "https://x/a.pdf"}])
    refs = asyncio.run(
        _collect_uploaded_files_with_ingest(body, principal=UserPrincipal(user_id="u"))
    )
    assert len(refs) == 1


def test_collect_with_ingest_dispatches_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every resolved PDF ref is handed to the bound PdfStore."""
    import openbb_agent_server.app.router as router_mod

    ingested: list[str] = []

    class _PdfStore:
        async def ingest_async(self, *, name: str, **_kw: Any) -> None:
            ingested.append(name)

    monkeypatch.setattr(router_mod.services, "get_pdf_store", lambda: _PdfStore())
    body = _query(
        uploaded_files=[
            {"name": "a.pdf", "url": "https://x/a.pdf"},
            {"name": "b.csv", "url": "https://x/b.csv"},
        ]
    )
    refs = asyncio.run(
        _collect_uploaded_files_with_ingest(body, principal=UserPrincipal(user_id="u"))
    )
    assert len(refs) == 2
    assert ingested == ["a.pdf"]


def test_collect_with_ingest_swallows_dispatch_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing ingest_async is logged but does not abort collection."""
    import openbb_agent_server.app.router as router_mod

    class _PdfStore:
        async def ingest_async(self, **_kw: Any) -> None:
            raise RuntimeError("ingest down")

    monkeypatch.setattr(router_mod.services, "get_pdf_store", lambda: _PdfStore())
    body = _query(uploaded_files=[{"name": "a.pdf", "url": "https://x/a.pdf"}])
    with caplog.at_level("WARNING"):
        refs = asyncio.run(
            _collect_uploaded_files_with_ingest(
                body, principal=UserPrincipal(user_id="u")
            )
        )
    assert len(refs) == 1
    assert any("background ingest dispatch failed" in r.message for r in caplog.records)
