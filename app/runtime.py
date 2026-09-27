"""Bounded provider I/O and CPU work shared by all requests in one process."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.config import Settings
from app.errors import ServiceError

logger = logging.getLogger("app")


class ProviderRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.slots = asyncio.Semaphore(settings.max_provider_calls)

    async def call[T](
        self,
        request: Callable[[], Coroutine[Any, Any, T]],
        *,
        operation: str = "provider",
        batch_number: int | None = None,
        item_count: int | None = None,
    ) -> T:
        """One retry for transient failures. The outer request deadline includes queueing."""
        for attempt in range(2):
            fields = {"operation": operation, "attempt": attempt + 1}
            if batch_number is not None:
                fields["batch_number"] = batch_number
            if item_count is not None:
                fields["item_count"] = item_count
            call_started = None
            try:
                async with self.slots:
                    call_started = perf_counter()
                    logger.info("provider_call_started", extra=fields)
                    async with asyncio.timeout(self.settings.provider_timeout_seconds):
                        result = await request()
                    logger.info(
                        "provider_call_complete",
                        extra={
                            **fields,
                            "duration_ms": round((perf_counter() - call_started) * 1000, 2),
                        },
                    )
                    return result
            except asyncio.CancelledError:
                logger.info(
                    "provider_call_cancelled",
                    extra={**fields, **self._duration(call_started)},
                )
                raise
            except ServiceError as exc:
                logger.warning(
                    "provider_call_failed",
                    extra={**fields, **self._duration(call_started), "code": exc.code},
                )
                raise
            except (TimeoutError, APITimeoutError):
                logger.warning(
                    "provider_call_failed",
                    extra={**fields, **self._duration(call_started), "code": "provider_timeout"},
                )
                # A slow request should not automatically double its cost.
                raise ServiceError(504, "provider_timeout", "The AI service timed out.") from None
            except (APIConnectionError, RateLimitError) as exc:
                reason = (
                    "provider_rate_limited"
                    if isinstance(exc, RateLimitError)
                    else "provider_connection"
                )
                if attempt == 1:
                    logger.warning(
                        "provider_call_failed",
                        extra={
                            **fields,
                            **self._duration(call_started),
                            "code": "provider_unavailable",
                        },
                    )
                    raise ServiceError(
                        503, "provider_unavailable", "The AI service is temporarily unavailable."
                    ) from None
            except APIStatusError as exc:
                if exc.status_code in (401, 403, 404):
                    logger.warning(
                        "provider_call_failed",
                        extra={
                            **fields,
                            **self._duration(call_started),
                            "code": "provider_configuration",
                        },
                    )
                    raise ServiceError(
                        503,
                        "provider_configuration",
                        "Check the OpenAI key and access to the configured models.",
                    ) from None
                if exc.status_code < 500:
                    logger.warning(
                        "provider_call_failed",
                        extra={
                            **fields,
                            **self._duration(call_started),
                            "code": "provider_request_failed",
                        },
                    )
                    raise ServiceError(
                        502,
                        "provider_request_failed",
                        "The AI service could not process the request.",
                    ) from None
                if attempt == 1:
                    logger.warning(
                        "provider_call_failed",
                        extra={
                            **fields,
                            **self._duration(call_started),
                            "code": "provider_unavailable",
                        },
                    )
                    raise ServiceError(
                        503, "provider_unavailable", "The AI service is temporarily unavailable."
                    ) from None
                reason = "provider_server_error"
            logger.warning(
                "provider_call_retry",
                extra={**fields, **self._duration(call_started), "code": reason},
            )
            await asyncio.sleep(0.25)
        raise AssertionError("unreachable")

    @staticmethod
    def _duration(started: float | None) -> dict[str, float]:
        if started is None:
            return {}
        return {"duration_ms": round((perf_counter() - started) * 1000, 2)}


class WorkerPool:
    """Limit running work even when its waiting HTTP request is cancelled."""

    def __init__(self, size: int):
        self.executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix="qa-worker")
        self.slots = asyncio.Semaphore(size)

    async def run[T](self, operation: Callable[..., T], *args: Any) -> T:
        await self.slots.acquire()
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self.executor, copy_context().run, operation, *args
            )
        except BaseException:
            self.slots.release()
            raise

        def completed(done: asyncio.Future) -> None:
            self.slots.release()
            if not done.cancelled():
                done.exception()  # Consume errors if the request no longer awaits this work.

        future.add_done_callback(completed)
        return await asyncio.shield(future)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
