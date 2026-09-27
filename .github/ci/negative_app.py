"""Credential-free failure-mode apps used by the real-HTTP negative bench."""

import httpx
from openai import APIConnectionError

from app.core.config import Settings
from app.main import create_app
from tests.fakes import DeterministicEmbeddings, GroundedGenerator


def _app(settings: Settings, generator: GroundedGenerator):
    return create_app(
        settings,
        embeddings=DeterministicEmbeddings(),
        generator=generator,
    )


provider_timeout_app = _app(
    Settings(
        _env_file=None,
        provider_timeout_seconds=0.02,
        request_timeout_seconds=2,
    ),
    GroundedGenerator(delay=0.2),
)

request_timeout_app = _app(
    Settings(
        _env_file=None,
        provider_timeout_seconds=2,
        request_timeout_seconds=0.05,
    ),
    GroundedGenerator(delay=0.2),
)

provider_unavailable_app = _app(
    Settings(_env_file=None),
    GroundedGenerator(
        error=APIConnectionError(request=httpx.Request("POST", "https://provider.invalid/test"))
    ),
)

invalid_evidence_app = _app(
    Settings(_env_file=None),
    GroundedGenerator(invalid_chunk_id=True),
)

busy_app = _app(
    Settings(_env_file=None, max_active_requests=1),
    GroundedGenerator(delay=0.3),
)
