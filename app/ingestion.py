"""Bounded parsing and chunking that retains source locations for citations."""

import math
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import PurePath
from typing import Any

import simplejson as json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings
from app.errors import ServiceError


def validate_file_type(filename: str | None, content_type: str | None, allowed: set[str]) -> str:
    """Require a supported extension and a compatible declared media type."""
    kind = PurePath(filename or "").suffix.lower().lstrip(".")
    media_type = (content_type or "").partition(";")[0].strip().lower()
    if kind not in allowed:
        raise ServiceError(
            415, "unsupported_file_type", "Upload a file with a supported extension."
        )
    expected = {"pdf": "application/pdf", "json": "application/json"}[kind]
    if media_type not in {"", "application/octet-stream", expected}:
        raise ServiceError(
            415, "file_type_mismatch", "File extension and content type do not match."
        )
    return kind


def _check_size(data: bytes, limit: int) -> None:
    if len(data) > limit:
        raise ServiceError(413, "file_too_large", "An uploaded file exceeds its byte limit.")


def _check_json_depth(text: str, limit: int) -> None:
    """Bound nesting before json.loads, ignoring braces inside string values."""
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > limit:
                raise ServiceError(413, "json_too_deep", "JSON nesting exceeds the allowed depth.")
        elif char in "]}":
            depth -= 1


def _reject_constant(_: str) -> None:
    raise ValueError("Non-finite numbers are not valid JSON.")


def _finite_decimal(value: str) -> Decimal:
    if not math.isfinite(float(value)):
        raise ValueError("JSON numbers must be finite.")
    return Decimal(value)


def _validate_json_strings(root: Any) -> None:
    pending = [root]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            value.encode("utf-8")
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def _load_json(data: bytes, settings: Settings) -> Any:
    try:
        text = data.decode("utf-8")
        _check_json_depth(text, settings.max_json_depth)
        root = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_decimal)
        _validate_json_strings(root)
        return root
    except (UnicodeError, ValueError, RecursionError, InvalidOperation) as exc:
        raise ServiceError(
            422, "invalid_json", "Upload valid UTF-8 JSON with finite numbers."
        ) from exc


def parse_questions(data: bytes, settings: Settings) -> list[str]:
    _check_size(data, settings.max_questions_bytes)
    questions = _load_json(data, settings)
    if not isinstance(questions, list) or not questions:
        raise ServiceError(422, "invalid_questions", "Questions must be a nonempty JSON array.")
    if len(questions) > settings.max_questions:
        raise ServiceError(
            413, "too_many_questions", "The question count exceeds the allowed limit."
        )
    for question in questions:
        if not isinstance(question, str) or not question.strip():
            raise ServiceError(422, "invalid_question", "Each question must be a nonempty string.")
        if len(question) > settings.max_question_chars:
            raise ServiceError(413, "question_too_long", "A question exceeds the character limit.")
    return questions


def _has_content(value: Any) -> bool:
    """Keep named fields even when their values explicitly express absence."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item:
                return True
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str):
            if item.strip():
                return True
        elif item is not None:
            return True
    return False


def _parse_json_document(data: bytes, settings: Settings) -> list[Document]:
    root = _load_json(data, settings)
    if not isinstance(root, (dict, list)):
        raise ServiceError(
            422, "invalid_document", "Document JSON must contain an object or array."
        )
    records = enumerate(root) if isinstance(root, list) else [(None, root)]
    encoder = json.JSONEncoder(
        ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, use_decimal=True
    )
    documents = []
    total_chars = 0
    for index, record in records:
        if not _has_content(record):
            continue
        parts = []
        for part in encoder.iterencode(record):
            total_chars += len(part)
            if total_chars > settings.max_extracted_chars:
                raise ServiceError(413, "document_too_long", "Extracted text exceeds the limit.")
            parts.append(part)
        documents.append(
            Document(
                page_content="".join(parts),
                metadata={"page": None, "json_path": "$" if index is None else f"$[{index}]"},
            )
        )
        # Each nonempty record requires at least one chunk.
        if len(documents) > settings.max_chunks:
            raise ServiceError(413, "too_many_chunks", "The document produces too many chunks.")
    if not documents:
        raise ServiceError(422, "empty_document", "The document contains no usable text.")
    return documents


def _normalize_pdf_text(text: str) -> str:
    """Collapse extractor-produced whitespace while preserving source words and punctuation.

    PDF text operators often position each word independently. ``pypdf`` can therefore return
    visually adjacent words separated by newlines (and, for some table-heavy reports, blank
    lines). Those layout artifacts are harmful to chunking and embeddings and are not reliable
    paragraph boundaries, so PDF source text uses one canonical space between tokens.
    """
    return " ".join(text.split())


def _parse_pdf(data: bytes, settings: Settings) -> list[Document]:
    if not data.startswith(b"%PDF-"):
        raise ServiceError(422, "invalid_pdf", "The uploaded file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise ServiceError(422, "encrypted_pdf", "Encrypted PDFs are not supported.")
        if len(reader.pages) > settings.max_pdf_pages:
            raise ServiceError(413, "too_many_pages", "The PDF exceeds the page limit.")
        documents = []
        total_chars = 0
        for number, page in enumerate(reader.pages, start=1):
            text = _normalize_pdf_text(page.extract_text() or "")
            total_chars += len(text)
            if total_chars > settings.max_extracted_chars:
                raise ServiceError(413, "document_too_long", "Extracted text exceeds the limit.")
            if text:
                documents.append(Document(page_content=text, metadata={"page": number}))
    except ServiceError:
        raise
    except Exception as exc:
        # Parser exceptions can contain source text; never expose their messages.
        raise ServiceError(
            422, "invalid_pdf", "The PDF could not be read. Upload a valid PDF."
        ) from exc
    if not documents:
        raise ServiceError(
            422, "empty_document", "The PDF contains no extractable text. OCR is not supported."
        )
    return documents


def parse_document(data: bytes, kind: str, settings: Settings) -> list[Document]:
    _check_size(data, settings.max_document_bytes)
    if kind == "json":
        return _parse_json_document(data, settings)
    if kind == "pdf":
        return _parse_pdf(data, settings)
    raise ServiceError(415, "unsupported_file_type", "Document must be a PDF or JSON file.")


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = []
    for document in documents:
        # Split one source record at a time; chunks cannot cross page/record boundaries.
        for chunk in splitter.split_documents([document]):
            if len(chunks) >= settings.max_chunks:
                raise ServiceError(413, "too_many_chunks", "The document produces too many chunks.")
            chunk.metadata["chunk_id"] = f"chunk_{len(chunks) + 1}"
            chunks.append(chunk)
    if not chunks:
        raise ServiceError(422, "empty_document", "The document contains no usable text.")
    return chunks
