PYTHON ?= python3.12
PORT ?= 8000
VENV_PYTHON := .venv/bin/python

.PHONY: help install env doctor dev test check sample require-venv

help:
	@echo "make install  Install pinned dependencies into .venv (Python 3.12 required)"
	@echo "make env      Create local .env from the example without overwriting it"
	@echo "make doctor   Check the interpreter, imports, and key configuration"
	@echo "make dev      Run the API at http://127.0.0.1:$(PORT)/docs; Ctrl-C stops it"
	@echo "make test     Run the offline test suite"
	@echo "make check    Check code style and formatting"
	@echo "make sample   Send synthetic files to the running API (uses OpenAI credits)"

install:
	@$(PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
	@test -x $(VENV_PYTHON) || $(PYTHON) -m venv .venv
	@$(VENV_PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv must use Python 3.12"'
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt

env:
	@if test -e .env; then echo "Existing .env preserved."; else cp .env.example .env; echo "Created .env. Enter your key locally in your editor."; fi

require-venv:
	@test -x $(VENV_PYTHON) || { echo "Run make install first."; exit 1; }

doctor: require-venv
	$(VENV_PYTHON) -m scripts.doctor

dev: require-venv
	$(VENV_PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload --reload-dir app --no-access-log

test: require-venv
	$(VENV_PYTHON) -m pytest -q

check: require-venv
	$(VENV_PYTHON) -m ruff check app tests scripts .github/ci
	$(VENV_PYTHON) -m ruff format --check app tests scripts .github/ci

sample:
	curl --fail-with-body http://127.0.0.1:$(PORT)/qa \
	  -F 'questions=@examples/questions.json;type=application/json' \
	  -F 'document=@examples/document.json;type=application/json'
