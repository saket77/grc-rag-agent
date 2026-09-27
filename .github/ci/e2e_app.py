"""Credential-free provider wiring used only by the real-HTTP CI smoke test."""

from app.core.config import Settings
from app.main import create_app
from tests.fakes import DeterministicEmbeddings, GroundedGenerator

# Both provider dependencies are supplied explicitly, so QAService never constructs an
# OpenAI embeddings or chat client. The production FastAPI routes and pipeline remain unchanged.
app = create_app(
    Settings(_env_file=None),
    embeddings=DeterministicEmbeddings(),
    generator=GroundedGenerator(),
)
