import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError

from app.config import Settings
from app.main import create_app
from app.models import NOT_FOUND
from tests.fakes import DeterministicEmbeddings, GroundedGenerator
from tests.pdf_factory import make_pdf


@asynccontextmanager
async def api_client(*, settings=None, embeddings=None, generator=None):
    embeddings = embeddings or DeterministicEmbeddings()
    generator = generator or GroundedGenerator()
    app = create_app(
        settings or Settings(_env_file=None, openai_api_key=None),
        embeddings=embeddings,
        generator=generator,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, embeddings, generator


def uploads(questions=None, document=None, *, kind="json"):
    if questions is None:
        questions = ["Which cloud provider?"]
    if document is None:
        document = b'{"hosting": "AWS"}'
    return {
        "questions": ("questions.json", json.dumps(questions).encode(), "application/json"),
        "document": (f"document.{kind}", document, f"application/{kind}"),
    }


async def test_health_and_docs_work_without_provider_calls():
    async with api_client() as (client, embeddings, generator):
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["x-request-id"]
        assert (await client.get("/docs")).status_code == 200
        schema = (await client.get("/openapi.json")).json()
        body = schema["paths"]["/qa"]["post"]["requestBody"]["content"]["multipart/form-data"]
        assert set(body["schema"]["required"]) == {"questions", "document"}
        assert not embeddings.batches and not generator.calls


async def test_upload_ui_and_assets_work_without_provider_calls():
    async with api_client() as (client, embeddings, generator):
        page = await client.get("/")
        assert page.status_code == 200
        assert 'name="questions"' in page.text and 'name="document"' in page.text
        assert "default-src 'self'" in page.headers["content-security-policy"]
        for asset in ("app.js", "styles.css"):
            assert (await client.get(f"/static/{asset}")).status_code == 200
        assert not embeddings.batches and not generator.calls


async def test_json_rag_deduplicates_work_and_restores_original_order():
    questions = ["Which cloud provider?", "What encryption is used?", "Which cloud provider?"]
    generator = GroundedGenerator(delays_by_question={questions[0]: 0.03})
    async with api_client(generator=generator) as (client, embeddings, generator):
        response = await client.post(
            "/qa", files=uploads(questions, b'{"hosting": "AWS", "encryption": "AES-256"}')
        )
        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert [result["question"] for result in results] == questions
        assert results[0] == results[2]
        assert [result["answer"] for result in results[:2]] == [
            "The service is hosted on AWS.",
            "Data is encrypted using AES-256.",
        ]
        assert results[0]["citations"][0]["page"] is None
        assert '"hosting": "AWS"' in results[0]["citations"][0]["excerpt"]
        assert len(embeddings.batches) == 2  # One document batch, one deduplicated query batch.
        assert len(embeddings.batches[0]) == 1
        assert embeddings.batches[1] == questions[:2]
        assert len(generator.calls) == 2
        assert generator.completed_questions == [questions[1], questions[0]]


async def test_pdf_uses_real_page_extraction_chunking_and_vector_retrieval():
    settings = Settings(_env_file=None, retrieval_k=1, chunk_size=80, chunk_overlap=10)
    document = make_pdf("", "The\n \nservice is\t hosted  on AWS.", "Encryption uses AES-256.")
    async with api_client(settings=settings) as (client, _, generator):
        response = await client.post(
            "/qa",
            files=uploads(
                ["What encryption is used?", "Which cloud provider?"], document, kind="pdf"
            ),
        )
        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert results[0]["citations"][0]["page"] == 3
        assert "AES-256" in results[0]["citations"][0]["excerpt"]
        assert results[1]["citations"][0]["page"] == 2
        assert "AWS" in results[1]["citations"][0]["excerpt"]
        assert "\n" not in results[1]["citations"][0]["excerpt"]
        assert all(len(context) == 1 for _, context in generator.calls)


async def test_absent_evidence_returns_exact_not_found_and_empty_citations():
    async with api_client() as (client, _, _):
        response = await client.post("/qa", files=uploads(["What is the retention policy?"]))
        assert response.status_code == 200
        assert response.json()["results"] == [
            {"question": "What is the retention policy?", "answer": NOT_FOUND, "citations": []}
        ]


@pytest.mark.parametrize(
    ("files", "status"),
    [
        (uploads([]), 422),
        (uploads([" "]), 422),
        (uploads([1]), 422),
        (uploads(["\ud800"]), 422),
        (uploads(document=rb'{"hosting": "\udfff"}'), 422),
        (uploads(document=b"{"), 422),
        (uploads(document=b"{}"), 422),
        (uploads(document=b"", kind="pdf"), 422),
        (uploads(document=b"not a pdf", kind="pdf"), 422),
        (
            {
                "questions": ("questions.json", b"{}", "application/json"),
                "document": ("source.json", b"{}", "application/json"),
            },
            422,
        ),
        (
            {
                "questions": ("questions.txt", b'["Q?"]', "text/plain"),
                "document": ("source.json", b"{}", "application/json"),
            },
            415,
        ),
        (
            {
                "questions": ("questions.json", b'["Q?"]', "application/json"),
                "document": ("source.pdf", b"%PDF-", "application/json"),
            },
            415,
        ),
        ({"questions": ("questions.json", b'["Q?"]', "application/json")}, 422),
        ({**uploads(), "extra": ("extra.json", b"{}", "application/json")}, 422),
    ],
)
async def test_invalid_inputs_fail_before_any_paid_call(files, status):
    async with api_client() as (client, embeddings, generator):
        response = await client.post("/qa", files=files)
        assert response.status_code == status, response.text
        error = response.json()["error"]
        assert error["message"]
        assert error["request_id"] == response.headers["x-request-id"]
        assert not embeddings.batches and not generator.calls


@pytest.mark.parametrize(
    ("settings_overrides", "files"),
    [
        ({"max_document_bytes": 4}, uploads()),
        ({"max_questions_bytes": 4}, uploads()),
        ({"max_questions": 1}, uploads(["Q1?", "Q2?"])),
        ({"max_question_chars": 4}, uploads()),
        ({"max_pdf_pages": 1}, uploads(document=make_pdf("one", "two"), kind="pdf")),
        ({"max_extracted_chars": 4}, uploads()),
        ({"max_chunks": 1, "chunk_size": 5, "chunk_overlap": 0}, uploads()),
    ],
)
async def test_limits_are_http_413_before_provider_calls(settings_overrides, files):
    settings = Settings(_env_file=None, **settings_overrides)
    async with api_client(settings=settings) as (client, embeddings, generator):
        response = await client.post("/qa", files=files)
        assert response.status_code == 413, response.text
        assert not embeddings.batches and not generator.calls


async def test_plain_body_and_text_form_fields_are_rejected():
    async with api_client() as (client, embeddings, _):
        assert (await client.post("/qa", json={})).status_code == 415
        response = await client.post(
            "/qa",
            files={"document": ("source.json", b'{"hosting": "AWS"}', "application/json")},
            data={"questions": '["Q?"]'},
        )
        assert response.status_code == 422
        assert not embeddings.batches


async def test_duplicate_multipart_fields_are_rejected():
    questions = ("questions.json", b'["Which cloud provider?"]', "application/json")
    async with api_client() as (client, embeddings, _):
        response = await client.post(
            "/qa", files=[("questions", questions), ("questions", questions)]
        )
        assert response.status_code == 422
        assert not embeddings.batches


async def test_actual_body_limit_applies_without_content_length():
    settings = Settings(_env_file=None, max_request_bytes=100)

    async def body():
        yield b"--test-boundary\r\n" + b"x" * 60
        yield b"y" * 60

    async with api_client(settings=settings) as (client, embeddings, _):
        request = client.build_request(
            "POST",
            "/qa",
            content=body(),
            headers={"Content-Type": "multipart/form-data; boundary=test-boundary"},
        )
        assert "content-length" not in request.headers
        response = await client.send(request)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"
        assert not embeddings.batches


async def test_unknown_evidence_chunk_id_is_a_provider_error():
    async with api_client(generator=GroundedGenerator(invalid_chunk_id=True)) as (
        client,
        _,
        _,
    ):
        response = await client.post("/qa", files=uploads())
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "provider_response_invalid"
        assert "results" not in response.json()
        assert "invented" not in response.text


async def test_provider_failure_is_retried_once_then_http_503():
    failure = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/test"))
    generator = GroundedGenerator(error=failure)
    async with api_client(generator=generator) as (client, _, _):
        response = await client.post("/qa", files=uploads())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "provider_unavailable"
        assert len(generator.calls) == 2
        assert NOT_FOUND not in response.text


async def test_provider_auth_error_is_safe_and_not_retried():
    failure = AuthenticationError(
        "secret-provider-diagnostic",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.openai.com/v1/test")
        ),
        body=None,
    )
    generator = GroundedGenerator(error=failure)
    async with api_client(generator=generator) as (client, _, _):
        response = await client.post("/qa", files=uploads())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "provider_configuration"
        assert len(generator.calls) == 1
        assert "secret-provider-diagnostic" not in response.text


async def test_provider_timeout_cancels_work_and_does_not_become_not_found():
    settings = Settings(_env_file=None, provider_timeout_seconds=0.02, request_timeout_seconds=2)
    generator = GroundedGenerator(delay=0.2)
    async with api_client(settings=settings, generator=generator) as (client, _, _):
        response = await client.post("/qa", files=uploads())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "provider_timeout"
        assert len(generator.calls) == 1 and generator.cancelled == 1
        assert NOT_FOUND not in response.text


async def test_request_deadline_includes_receiving_uploads():
    settings = Settings(_env_file=None, request_timeout_seconds=0.02)

    async def slow_body():
        yield b"--boundary\r\n"
        await asyncio.sleep(0.2)
        yield b"--boundary--\r\n"

    async with api_client(settings=settings) as (client, embeddings, _):
        response = await client.post(
            "/qa",
            content=slow_body(),
            headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        )
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "request_timeout"
        assert not embeddings.batches
        assert (await client.get("/healthz")).status_code == 200


async def test_request_deadline_cancels_generation():
    settings = Settings(_env_file=None, request_timeout_seconds=0.1, provider_timeout_seconds=2)
    generator = GroundedGenerator(delay=1)
    async with api_client(settings=settings, generator=generator) as (client, _, _):
        response = await client.post("/qa", files=uploads())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "request_timeout"
        assert generator.cancelled == 1


async def test_admission_limit_is_released_after_request_finishes():
    gate = asyncio.Event()
    generator = GroundedGenerator(gate=gate)
    settings = Settings(_env_file=None, max_active_requests=1)
    async with api_client(settings=settings, generator=generator) as (client, _, _):
        first = asyncio.create_task(client.post("/qa", files=uploads()))
        try:
            await asyncio.wait_for(generator.started.wait(), timeout=2)
            second = await client.post("/qa", files=uploads())
            assert second.status_code == 503
            assert second.json()["error"]["code"] == "service_busy"
        finally:
            gate.set()
            assert (await first).status_code == 200
        assert (await client.post("/qa", files=uploads())).status_code == 200


async def test_provider_concurrency_limit_is_shared_across_requests():
    generator = GroundedGenerator(delay=0.03)
    settings = Settings(_env_file=None, max_provider_calls=2, max_active_requests=2)
    async with api_client(settings=settings, generator=generator) as (client, _, _):
        responses = await asyncio.gather(
            client.post("/qa", files=uploads(["Cloud one?", "Cloud two?", "Cloud three?"])),
            client.post("/qa", files=uploads(["Cloud four?", "Cloud five?", "Cloud six?"])),
        )
        assert all(response.status_code == 200 for response in responses)
        assert generator.peak_active == 2
        assert len(generator.calls) == 6


async def test_document_indexes_are_isolated_between_requests():
    async with api_client() as (client, embeddings, generator):
        responses = await asyncio.gather(
            client.post("/qa", files=uploads(document=b'{"hosting": "AWS"}')),
            client.post("/qa", files=uploads(document=b'{"hosting": "Azure"}')),
        )
        assert all(response.status_code == 200 for response in responses)
        assert responses[0].json()["results"][0]["answer"] == "The service is hosted on AWS."
        assert responses[1].json()["results"][0]["answer"] == "The service is hosted on Azure."
        assert len(embeddings.batches) == 4
        assert len(generator.calls) == 2
        assert all(
            not ("AWS" in chunk.page_content and "Azure" in chunk.page_content)
            for _, chunks in generator.calls
            for chunk in chunks
        )


async def test_missing_key_returns_configuration_error_after_input_validation():
    app = create_app(Settings(_env_file=None, openai_api_key=None))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            invalid = await client.post("/qa", files=uploads([]))
            assert invalid.status_code == 422
            response = await client.post("/qa", files=uploads())
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "missing_api_key"
