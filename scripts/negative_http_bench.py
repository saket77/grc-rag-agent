"""Exercise validation, limit, timeout, and safe-failure behavior over real HTTP."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from collections.abc import Iterable

import httpx
from pypdf import PdfWriter

QUESTIONS = json.dumps(["Which cloud provider?"]).encode()
DOCUMENT = b'{"hosting":"AWS"}'


def _files(
    *,
    questions: bytes = QUESTIONS,
    document: bytes = DOCUMENT,
    question_name: str = "questions.json",
    document_name: str = "document.json",
    question_type: str = "application/json",
    document_type: str = "application/json",
):
    return {
        "questions": (question_name, questions, question_type),
        "document": (document_name, document, document_type),
    }


def _many_page_pdf(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _assert_error(
    response: httpx.Response,
    *,
    case: str,
    status: int,
    code: str,
) -> dict[str, object]:
    assert response.status_code == status, (
        f"{case}: expected HTTP {status}, got {response.status_code}: {response.text[:500]}"
    )
    payload = response.json()
    assert set(payload) == {"error"}, f"{case}: unexpected response envelope: {payload}"
    error = payload["error"]
    assert error["code"] == code, f"{case}: expected {code}, got {error}"
    assert isinstance(error["message"], str) and error["message"].strip(), (
        f"{case}: missing friendly error message"
    )
    request_id = response.headers.get("x-request-id")
    assert request_id and error["request_id"] == request_id, (
        f"{case}: response and body request IDs differ"
    )
    lowered = response.text.casefold()
    for forbidden in ("traceback", "api_key=", "invented", "provider.invalid"):
        assert forbidden not in lowered, f"{case}: leaked internal detail {forbidden!r}"
    result = {
        "case": case,
        "status": response.status_code,
        "code": error["code"],
        "message": error["message"],
        "request_id_present": True,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def _validation_cases(client: httpx.Client) -> Iterable[dict[str, object]]:
    yield _assert_error(
        client.post("/qa", json={}),
        case="multipart required",
        status=415,
        code="multipart_required",
    )
    yield _assert_error(
        client.post(
            "/qa",
            files={"questions": ("questions.json", QUESTIONS, "application/json")},
        ),
        case="missing document field",
        status=422,
        code="invalid_uploads",
    )
    yield _assert_error(
        client.post("/qa", files=_files(questions=b"{")),
        case="malformed questions JSON",
        status=422,
        code="invalid_json",
    )
    yield _assert_error(
        client.post(
            "/qa",
            files=_files(question_name="questions.txt", question_type="text/plain"),
        ),
        case="unsupported questions file type",
        status=415,
        code="unsupported_file_type",
    )
    yield _assert_error(
        client.post(
            "/qa",
            files=_files(
                document=b"%PDF-1.4\ntruncated",
                document_name="document.pdf",
                document_type="application/pdf",
            ),
        ),
        case="malformed PDF",
        status=422,
        code="invalid_pdf",
    )
    yield _assert_error(
        client.post("/qa", files=_files(questions=json.dumps(["Q?"] * 51).encode())),
        case="excessive question count",
        status=413,
        code="too_many_questions",
    )
    yield _assert_error(
        client.post("/qa", files=_files(questions=json.dumps(["x" * 2001]).encode())),
        case="question too long",
        status=413,
        code="question_too_long",
    )
    yield _assert_error(
        client.post("/qa", files=_files(questions=b"[" + b'"x",' * 32768 + b'"x"]')),
        case="oversized questions file",
        status=413,
        code="file_too_large",
    )
    yield _assert_error(
        client.post(
            "/qa",
            files=_files(
                document=_many_page_pdf(201),
                document_name="document.pdf",
                document_type="application/pdf",
            ),
        ),
        case="PDF page limit",
        status=413,
        code="too_many_pages",
    )
    yield _assert_error(
        client.post(
            "/qa",
            files=_files(
                document=b"%PDF-" + b"0" * (10 * 1024 * 1024),
                document_name="document.pdf",
                document_type="application/pdf",
            ),
        ),
        case="PDF byte limit",
        status=413,
        code="file_too_large",
    )
    yield _assert_error(
        client.post(
            "/qa",
            files=_files(
                document=b"%PDF-" + b"0" * (11 * 1024 * 1024),
                document_name="document.pdf",
                document_type="application/pdf",
            ),
        ),
        case="total request byte limit",
        status=413,
        code="request_too_large",
    )


async def _busy_case(url: str) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=40, trust_env=False) as client:
        first = asyncio.create_task(client.post("/qa", files=_files()))
        await asyncio.sleep(0.05)
        second = await client.post("/qa", files=_files())
        _assert_error(second, case="admission limit", status=503, code="service_busy")
        completed = await first
        assert completed.status_code == 200, completed.text
        print(json.dumps({"case": "admission recovery", "status": 200}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Base URL of the running bench")
    parser.add_argument(
        "--profile",
        required=True,
        choices=(
            "validation",
            "provider-timeout",
            "request-timeout",
            "provider-unavailable",
            "invalid-evidence",
            "busy",
        ),
    )
    args = parser.parse_args()
    url = args.url.rstrip("/")
    if args.profile == "busy":
        asyncio.run(_busy_case(url))
        return

    with httpx.Client(base_url=url, timeout=40, trust_env=False) as client:
        health = client.get("/healthz")
        assert health.status_code == 200 and health.json() == {"status": "ok"}
        if args.profile == "validation":
            results = list(_validation_cases(client))
            print(json.dumps({"profile": args.profile, "passed": len(results)}))
            return
        expected = {
            "provider-timeout": (504, "provider_timeout"),
            "request-timeout": (504, "request_timeout"),
            "provider-unavailable": (503, "provider_unavailable"),
            "invalid-evidence": (502, "provider_response_invalid"),
        }[args.profile]
        response = client.post("/qa", files=_files())
        _assert_error(response, case=args.profile, status=expected[0], code=expected[1])
        print(json.dumps({"profile": args.profile, "passed": 1}))


if __name__ == "__main__":
    main()
