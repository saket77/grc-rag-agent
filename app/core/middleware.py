"""Admission, actual-byte limits, and a deadline including multipart parsing."""

import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.observability import request_id_context

logger = logging.getLogger("app")


def error_response(error: ServiceError, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


class RequestGuard:
    def __init__(self, app: ASGIApp, settings: Settings):
        self.app = app
        self.settings = settings
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        token = request_id_context.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()
        status = 500
        response_started = False
        admitted = False

        async def observed_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status = message["status"]
                response_started = True
                message["headers"] = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [(b"x-request-id", request_id.encode())]
            await send(message)

        try:
            if scope["path"] != "/qa" or scope["method"] != "POST":
                await self.app(scope, receive, observed_send)
                return
            # No await between checking/incrementing: atomic within this event loop.
            if self.active >= self.settings.max_active_requests:
                raise ServiceError(
                    503, "service_busy", "The service is busy. Please retry shortly."
                )
            self.active += 1
            admitted = True
            async with asyncio.timeout(self.settings.request_timeout_seconds):
                headers = dict(scope["headers"])
                if not headers.get(b"content-type", b"").lower().startswith(b"multipart/form-data"):
                    raise ServiceError(
                        415, "multipart_required", "Use multipart/form-data uploads."
                    )
                body = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        status = 499
                        return
                    fragment = message.get("body", b"")
                    if len(body) + len(fragment) > self.settings.max_request_bytes:
                        raise ServiceError(
                            413, "request_too_large", "The total upload exceeds the limit."
                        )
                    body.extend(fragment)
                    if not message.get("more_body", False):
                        break
                delivered = False

                async def bounded_receive():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, bounded_receive, observed_send)
        except TimeoutError:
            if not response_started:
                await error_response(
                    ServiceError(
                        504, "request_timeout", "The request exceeded its processing deadline."
                    ),
                    request_id,
                )(scope, receive, observed_send)
        except ServiceError as exc:
            if not response_started:
                await error_response(exc, request_id)(scope, receive, observed_send)
        except Exception:
            # Do not log exception text/tracebacks: parsers/providers may include report content.
            logger.error("request_failed", extra={"code": "internal_error"})
            if not response_started:
                await error_response(
                    ServiceError(500, "internal_error", "An unexpected error occurred."), request_id
                )(scope, receive, observed_send)
        finally:
            if admitted:
                self.active -= 1
            logger.info(
                "request_complete",
                extra={
                    "status_code": status,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )
            request_id_context.reset(token)
