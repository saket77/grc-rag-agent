"""The default suite must not need credentials or reach an external service."""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def blocked(*args, **kwargs):
        raise AssertionError("External network access is forbidden in the offline test suite")

    # Leave socketpair alone: asyncio uses it internally to wake the event loop.
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
