import json
from decimal import Decimal

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.errors import ServiceError
from app.ingestion import (
    parse_document,
    parse_questions,
    split_documents,
    validate_file_type,
)
from tests.pdf_factory import make_pdf


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_questions_preserve_original_order_and_duplicates(settings):
    values = ["  Hosting?  ", "Hosting?", "  Hosting?  "]
    assert parse_questions(json.dumps(values).encode(), settings) == values


@pytest.mark.parametrize("data", [b"[]", b"{}", b"[null]", b"[true]", b"[1]", b'[" "]'])
def test_questions_reject_empty_or_nonstring_values(data, settings):
    with pytest.raises(ServiceError) as error:
        parse_questions(data, settings)
    assert error.value.status_code == 422


@pytest.mark.parametrize(
    ("data", "overrides", "code"),
    [
        (b'["x", "y"]', {"max_questions": 1}, "too_many_questions"),
        (b'["long"]', {"max_question_chars": 3}, "question_too_long"),
        (b'["x"]', {"max_questions_bytes": 4}, "file_too_large"),
    ],
)
def test_question_limits(data, overrides, code):
    with pytest.raises(ServiceError) as error:
        parse_questions(data, Settings(_env_file=None, **overrides))
    assert (error.value.status_code, error.value.code) == (413, code)


@pytest.mark.parametrize(
    ("filename", "mime", "kind"),
    [
        ("report.PDF", "application/pdf", "pdf"),
        ("report.json", "application/json; charset=utf-8", "json"),
        ("report.pdf", "application/octet-stream", "pdf"),
        ("report.json", None, "json"),
    ],
)
def test_valid_file_types(filename, mime, kind):
    assert validate_file_type(filename, mime, {"json", "pdf"}) == kind


@pytest.mark.parametrize(
    ("filename", "mime"),
    [("report.txt", "application/json"), ("report.pdf", "application/json"), (None, None)],
)
def test_rejects_unsupported_or_mismatched_file_types(filename, mime):
    with pytest.raises(ServiceError) as error:
        validate_file_type(filename, mime, {"json", "pdf"})
    assert error.value.status_code == 415


def test_pdf_preserves_physical_pages_and_skips_blank_pages(settings):
    documents = parse_document(
        make_pdf("", "Hosted on AWS.", "Encrypted at rest."), "pdf", settings
    )
    assert [doc.metadata["page"] for doc in documents] == [2, 3]
    assert "Hosted on AWS." in documents[0].page_content


@pytest.mark.parametrize(
    ("pdf", "code"),
    [
        (b"plain text", "invalid_pdf"),
        (b"%PDF-1.4\ntruncated", "invalid_pdf"),
        (make_pdf(""), "empty_document"),
        (make_pdf("Secret", password="example-password"), "encrypted_pdf"),
    ],
)
def test_pdf_failures_are_friendly(pdf, code, settings):
    with pytest.raises(ServiceError) as error:
        parse_document(pdf, "pdf", settings)
    assert (error.value.status_code, error.value.code) == (422, code)
    assert "Secret" not in error.value.message


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"max_pdf_pages": 1}, "too_many_pages"),
        ({"max_extracted_chars": 3}, "document_too_long"),
        ({"max_document_bytes": 10}, "file_too_large"),
    ],
)
def test_pdf_resource_limits(overrides, code):
    with pytest.raises(ServiceError) as error:
        parse_document(make_pdf("first", "second"), "pdf", Settings(_env_file=None, **overrides))
    assert (error.value.status_code, error.value.code) == (413, code)


def test_json_keeps_related_fields_and_unicode(settings):
    root = {"hosting": {"provider": "AWS", "region": "Zürich"}, "encrypted": True}
    documents = parse_document(json.dumps(root).encode(), "json", settings)
    assert len(documents) == 1
    assert json.loads(documents[0].page_content) == root
    assert "Zürich" in documents[0].page_content
    assert documents[0].metadata == {"page": None, "json_path": "$"}


@pytest.mark.parametrize("value", [[], {}, None, "", False, 0])
def test_json_preserves_fields_even_when_their_values_express_absence(value, settings):
    record = {"third_party_processors": value}
    for root in (record, [record, {"hosting": "AWS"}]):
        documents = parse_document(json.dumps(root).encode(), "json", settings)
        assert json.loads(documents[0].page_content) == record


@pytest.mark.parametrize("value", ["99.999999999999999", "1.234567890123456789e-15", "1e-10000"])
def test_json_normalization_preserves_decimal_values(value, settings):
    documents = parse_document(f'{{"value": {value}}}'.encode(), "json", settings)
    normalized = json.loads(documents[0].page_content, parse_float=Decimal)
    assert normalized["value"] == Decimal(value)
    assert not isinstance(normalized["value"], str)


@pytest.mark.parametrize("escape", [r"\ud800", r"\udfff"])
def test_questions_reject_lone_surrogates(escape, settings):
    with pytest.raises(ServiceError) as caught:
        parse_questions(f'["{escape}"]'.encode(), settings)
    assert (caught.value.status_code, caught.value.code) == (422, "invalid_json")


@pytest.mark.parametrize(
    "data",
    [rb'{"nested": [{"value": "\ud800"}]}', rb'{"nested": {"\udfff": "value"}}'],
)
def test_json_rejects_lone_surrogates_in_values_and_keys(data, settings):
    with pytest.raises(ServiceError) as caught:
        parse_document(data, "json", settings)
    assert (caught.value.status_code, caught.value.code) == (422, "invalid_json")


def test_valid_surrogate_pairs_and_unicode_remain_supported(settings):
    assert parse_questions(rb'["Hosting \ud83d\ude00?"]', settings) == ["Hosting 😀?"]
    documents = parse_document(rb'{"\ud83d\ude00": "Z\u00fcrich"}', "json", settings)
    assert json.loads(documents[0].page_content) == {"😀": "Zürich"}


def test_json_array_preserves_record_identity_when_empty_records_are_skipped(settings):
    documents = parse_document(
        b'[{}, {"provider": "AWS"}, null, "Retention: 30 days"]', "json", settings
    )
    assert [doc.metadata["json_path"] for doc in documents] == ["$[1]", "$[3]"]
    assert all(doc.metadata["page"] is None for doc in documents)


@pytest.mark.parametrize(
    "data",
    [b"{", b'"scalar"', b"123", b"[]", b"{}", b"[null, [], {}]", b'{"bad":"\xff"}'],
)
def test_invalid_or_unusable_json_documents(data, settings):
    with pytest.raises(ServiceError) as error:
        parse_document(data, "json", settings)
    assert error.value.status_code == 422


@pytest.mark.parametrize(
    "value", ["NaN", "Infinity", "-Infinity", "1e10000", "1e-999999999999999999999999999"]
)
def test_json_rejects_nonfinite_numbers(value, settings):
    with pytest.raises(ServiceError) as error:
        parse_document(f'{{"value": {value}}}'.encode(), "json", settings)
    assert error.value.code == "invalid_json"


def test_json_depth_limit_ignores_escaped_strings():
    settings = Settings(_env_file=None, max_json_depth=2)
    root = {"value": '[{[ \\" [[[[]]]'}
    assert len(parse_document(json.dumps(root).encode(), "json", settings)) == 1
    with pytest.raises(ServiceError) as error:
        parse_document(b'{"a": {"b": {"c": "deep"}}}', "json", settings)
    assert (error.value.status_code, error.value.code) == (413, "json_too_deep")


@pytest.mark.parametrize(
    ("data", "overrides", "code"),
    [
        (b'{"a": "long text"}', {"max_extracted_chars": 4}, "document_too_long"),
        (b'["first", "second"]', {"max_chunks": 1}, "too_many_chunks"),
    ],
)
def test_json_resource_limits(data, overrides, code):
    with pytest.raises(ServiceError) as error:
        parse_document(data, "json", Settings(_env_file=None, **overrides))
    assert (error.value.status_code, error.value.code) == (413, code)


def test_chunking_retains_exact_offsets_and_source_boundaries():
    settings = Settings(_env_file=None, chunk_size=40, chunk_overlap=10)
    originals = {
        1: "First page hosting. " * 10,
        2: "Second page security. " * 10,
    }
    documents = [
        Document(page_content=text, metadata={"page": page}) for page, text in originals.items()
    ]
    chunks = split_documents(documents, settings)
    assert len(chunks) > 2
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        original = originals[chunk.metadata["page"]]
        offset = chunk.metadata["start_index"]
        assert offset >= 0
        assert original[offset : offset + len(chunk.page_content)] == chunk.page_content
        assert len(chunk.page_content) <= settings.chunk_size
    assert all("chunk_id" not in doc.metadata for doc in documents)


def test_chunk_limit_and_empty_documents():
    settings = Settings(_env_file=None, max_chunks=1, chunk_size=20, chunk_overlap=0)
    with pytest.raises(ServiceError) as error:
        split_documents([Document(page_content="lengthy content " * 20)], settings)
    assert (error.value.status_code, error.value.code) == (413, "too_many_chunks")
    with pytest.raises(ServiceError) as error:
        split_documents([], settings)
    assert (error.value.status_code, error.value.code) == (422, "empty_document")
