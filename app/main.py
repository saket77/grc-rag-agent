"""HTTP contract. Domain work lives in the QA service and its dependencies."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.embeddings import Embeddings
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.middleware import RequestGuard, error_response
from app.core.observability import configure_logging
from app.rag.generation import AnswerGenerator
from app.rag.ingestion import validate_file_type
from app.schemas.qa import QAResponse
from app.services.qa import QAService

STATIC_DIR = Path(__file__).resolve().parent / "static"

UPLOAD_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["questions", "document"],
                    "properties": {
                        "questions": {
                            "type": "string",
                            "format": "binary",
                            "description": "JSON question array",
                        },
                        "document": {
                            "type": "string",
                            "format": "binary",
                            "description": "PDF or JSON source",
                        },
                    },
                }
            }
        },
    }
}


async def read_upload(file: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(min(64 * 1024, limit + 1)):
        if len(data) + len(chunk) > limit:
            raise ServiceError(413, "file_too_large", "An uploaded file exceeds its size limit.")
        data.extend(chunk)
    if not data:
        raise ServiceError(422, "empty_file", "Uploaded files must not be empty.")
    return bytes(data)


def create_app(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    generator: AnswerGenerator | None = None,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging()
    service = QAService(settings, embeddings, generator)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await service.close()

    app = FastAPI(
        title="Zania Document QA",
        version="0.1.0",
        lifespan=lifespan,
        description="Upload questions and a source document for grounded answers with citations.",
    )
    app.state.service = service
    app.add_middleware(RequestGuard, settings=settings)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError):
        return error_response(exc, request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return error_response(
            ServiceError(422, "invalid_request", "The request does not match the required format."),
            request.state.request_id,
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException):
        code = 422 if exc.status_code == 400 else exc.status_code
        return error_response(
            ServiceError(
                code,
                "invalid_request",
                "The request is malformed or the requested route is unavailable.",
            ),
            request.state.request_id,
        )

    @app.get("/", include_in_schema=False)
    async def upload_page():
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; base-uri 'none'; object-src 'none'; "
                    "frame-ancestors 'none'; form-action 'self'"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.post("/qa", response_model=QAResponse, openapi_extra=UPLOAD_SCHEMA)
    async def qa(request: Request):
        # Explicit parsing provides bounded part counts and a context manager for cleanup.
        async with request.form(max_files=2, max_fields=0) as form:
            if sorted(form.keys()) != ["document", "questions"] or len(form.multi_items()) != 2:
                raise ServiceError(
                    422, "invalid_uploads", "Upload exactly questions and document files."
                )
            questions = form["questions"]
            document = form["document"]
            if not isinstance(questions, UploadFile) or not isinstance(document, UploadFile):
                raise ServiceError(422, "invalid_uploads", "Both inputs must be uploaded files.")
            validate_file_type(questions.filename, questions.content_type, {"json"})
            kind = validate_file_type(document.filename, document.content_type, {"pdf", "json"})
            question_bytes = await read_upload(questions, settings.max_questions_bytes)
            document_bytes = await read_upload(document, settings.max_document_bytes)
            return await service.answer(question_bytes, document_bytes, kind)

    return app


app = create_app()
