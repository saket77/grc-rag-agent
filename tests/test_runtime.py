import asyncio
import logging
import threading

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.runtime import ProviderRunner, WorkerPool


async def test_provider_calls_share_concurrency_limit():
    runner = ProviderRunner(Settings(_env_file=None, max_provider_calls=2))
    active = peak = 0

    async def operation():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return 1

    assert await asyncio.gather(*(runner.call(operation) for _ in range(8))) == [1] * 8
    assert peak == 2


async def test_provider_call_lifecycle_logs_safe_metadata(caplog):
    runner = ProviderRunner(Settings(_env_file=None))

    with caplog.at_level(logging.INFO, logger="app"):
        result = await runner.call(
            lambda: asyncio.sleep(0, result="private result"),
            operation="document_embedding",
            batch_number=2,
            item_count=64,
        )

    assert result == "private result"
    started = next(record for record in caplog.records if record.msg == "provider_call_started")
    completed = next(record for record in caplog.records if record.msg == "provider_call_complete")
    assert started.operation == completed.operation == "document_embedding"
    assert started.batch_number == completed.batch_number == 2
    assert started.item_count == completed.item_count == 64
    assert completed.duration_ms >= 0
    assert "private result" not in caplog.text


async def test_transient_failure_retries_exactly_once():
    runner = ProviderRunner(Settings(_env_file=None))
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        return "ok"

    assert await runner.call(operation) == "ok"
    assert calls == 2


@pytest.mark.parametrize(
    "status, expected, attempts",
    [(401, 503, 1), (403, 503, 1), (400, 502, 1), (500, 503, 2), (429, 503, 2)],
)
async def test_provider_failures_are_sanitized_and_bounded(status, expected, attempts):
    runner = ProviderRunner(Settings(_env_file=None))
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
        error_type = RateLimitError if status == 429 else APIStatusError
        raise error_type("PRIVATE CONTENT", response=response, body={"key": "secret"})

    with pytest.raises(ServiceError) as caught:
        await runner.call(operation)
    assert caught.value.status_code == expected
    assert "PRIVATE" not in str(caught.value)
    assert calls == attempts


async def test_provider_timeout_does_not_retry():
    runner = ProviderRunner(Settings(_env_file=None, provider_timeout_seconds=0.01))
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    with pytest.raises(ServiceError) as caught:
        await runner.call(operation)
    assert caught.value.status_code == 504
    assert calls == 1
    # Its slot is released for subsequent work.
    assert await runner.call(lambda: asyncio.sleep(0, result="ok")) == "ok"


async def test_cancelled_worker_keeps_its_slot_until_work_finishes():
    pool = WorkerPool(1)
    started = threading.Event()
    release = threading.Event()
    second_started = threading.Event()

    def first():
        started.set()
        release.wait(timeout=2)

    first_task = asyncio.create_task(pool.run(first))
    assert await asyncio.to_thread(started.wait, 1)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    second_task = asyncio.create_task(pool.run(second_started.set))
    try:
        await asyncio.sleep(0.02)
        assert not second_started.is_set()
        release.set()
        await asyncio.wait_for(second_task, timeout=1)
        assert second_started.is_set()
    finally:
        release.set()
        pool.close()
