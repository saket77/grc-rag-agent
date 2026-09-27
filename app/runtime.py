"""Bounded provider I/O and CPU work shared by all requests in one process."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.config import Settings
from app.errors import ServiceError

logger = logging.getLogger("app")


class ProviderRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.slots = asyncio.Semaphore(settings.max_provider_calls)

    async def call[T](self, operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """One retry for transient failures. The outer request deadline includes queueing."""
        for attempt in range(2):
            try:
                async with self.slots:
                    async with asyncio.timeout(self.settings.provider_timeout_seconds):
                        return await operation()
            except ServiceError:
                raise
            except (TimeoutError, APITimeoutError):
                # A slow request should not automatically double its cost.
                raise ServiceError(504, "provider_timeout", "The AI service timed out.") from None
            except (APIConnectionError, RateLimitError):
                if attempt == 1:
                    raise ServiceError(
                        503, "provider_unavailable", "The AI service is temporarily unavailable."
                    ) from None
            except APIStatusError as exc:
                if exc.status_code in (401, 403, 404):
                    raise ServiceError(
                        503,
                        "provider_configuration",
                        "Check the OpenAI key and access to the configured models.",
                    ) from None
                if exc.status_code < 500:
                    raise ServiceError(
                        502,
                        "provider_request_failed",
                        "The AI service could not process the request.",
                    ) from None
                if attempt == 1:
                    raise ServiceError(
                        503, "provider_unavailable", "The AI service is temporarily unavailable."
                    ) from None
            logger.info("provider_retry", extra={"attempt": attempt + 1})
            await asyncio.sleep(0.25)
        raise AssertionError("unreachable")


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
