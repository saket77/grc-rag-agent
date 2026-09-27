"""Local setup diagnostics. Never prints credentials or makes network calls."""

import importlib
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    valid = sys.version_info[:2] == (3, 12)
    if not valid:
        print("ERROR: this project requires Python 3.12.")
    for name in (
        "fastapi",
        "uvicorn",
        "pydantic_settings",
        "langchain_openai",
        "langchain_text_splitters",
        "langchain_community.vectorstores",
        "faiss",
        "pypdf",
        "pytest",
    ):
        try:
            importlib.import_module(name)
        except Exception:
            # Import errors can include environment details; only report the module name.
            print(f"FAIL: {name} — run make install")
            valid = False
        else:
            print(f"OK: {name}")
    if not valid:
        return 1

    from app.core.config import Settings

    try:
        settings = Settings()
    except Exception:
        print("FAIL: invalid configuration; check .env against .env.example.")
        return 1
    key_configured = bool(
        settings.openai_api_key and settings.openai_api_key.get_secret_value().strip()
    )
    print(f"OPENAI_API_KEY: {'configured (not verified)' if key_configured else 'not configured'}")
    print("Health, API docs, and tests work without a key. Live answers require a valid key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
