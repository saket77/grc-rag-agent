#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

port="${CI_E2E_PORT:-8765}"
artifacts="$(mktemp -d)"
server_pid=""
python_command="python"
if [[ -x .venv/bin/python ]]; then
  python_command=".venv/bin/python"
fi

cleanup() {
  status=$?
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    printf '%s\n' "End-to-end server log:"
    sed -n '1,240p' "$artifacts/server.log" 2>/dev/null || true
  fi
  rm -rf "$artifacts"
}
trap cleanup EXIT

# Never inherit developer credentials into this test. e2e_app.py also injects both providers.
unset OPENAI_API_KEY OPENAI_BASE_URL LANGSMITH_API_KEY
export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false
export PYTHONUNBUFFERED=1

"$python_command" -m uvicorn e2e_app:app \
  --app-dir .github/ci \
  --host 127.0.0.1 \
  --port "$port" \
  --no-access-log \
  >"$artifacts/server.log" 2>&1 &
server_pid=$!

ready=""
for _ in {1..60}; do
  if curl --silent --show-error --fail \
    "http://127.0.0.1:${port}/healthz" >"$artifacts/health.json" 2>/dev/null; then
    ready="yes"
    break
  fi
  sleep 0.25
done
test "$ready" = "yes"

curl --silent --show-error --fail \
  "http://127.0.0.1:${port}/openapi.json" \
  >"$artifacts/openapi.json"

curl --silent --show-error --fail-with-body \
  --dump-header "$artifacts/headers.txt" \
  --output "$artifacts/response.json" \
  "http://127.0.0.1:${port}/qa" \
  -F 'questions=@examples/questions.json;type=application/json' \
  -F 'document=@examples/document.json;type=application/json'

kill -INT "$server_pid"
wait "$server_pid" || true
server_pid=""

"$python_command" - \
  "$artifacts/health.json" \
  "$artifacts/openapi.json" \
  "$artifacts/headers.txt" \
  "$artifacts/response.json" \
  "$artifacts/server.log" <<'PY'
import json
import sys
from pathlib import Path

health_path, openapi_path, headers_path, response_path, log_path = map(Path, sys.argv[1:])

assert json.loads(health_path.read_text()) == {"status": "ok"}
paths = json.loads(openapi_path.read_text())["paths"]
assert {"/healthz", "/qa"} <= set(paths)

headers = {}
for line in headers_path.read_text().splitlines():
    if ":" in line:
        name, value = line.split(":", 1)
        headers[name.lower()] = value.strip()
assert headers.get("x-request-id")

payload = json.loads(response_path.read_text())
results = payload["results"]
assert [result["question"] for result in results] == [
    "Which cloud providers do you rely on?",
    "Is personal information disclosed to third parties?",
    "How frequently do you conduct penetration tests?",
]
assert results[0]["answer"] == "The service is hosted on AWS."
assert results[0]["citations"][0]["page"] is None
assert "hosted on AWS" in results[0]["citations"][0]["excerpt"]
assert results[1]["answer"] == "Not found in document"
assert results[1]["citations"] == []
assert results[2]["answer"] == "Not found in document"
assert results[2]["citations"] == []

events = []
for line in log_path.read_text().splitlines():
    if line.startswith("{"):
        events.append(json.loads(line))

required = {"ingestion_complete", "index_complete", "retrieval_complete", "generation_complete"}
assert required <= {event.get("event") for event in events}
for event in events:
    if event.get("event") in required | {"request_complete"}:
        assert isinstance(event.get("duration_ms"), (int, float))
        assert event["duration_ms"] >= 0

question_embedding_calls = [
    event
    for event in events
    if event.get("event") == "provider_call_started"
    and event.get("operation") == "question_embedding"
]
assert len(question_embedding_calls) == 1
assert question_embedding_calls[0]["item_count"] == 3
PY

printf '%s\n' "Real-HTTP end-to-end test passed with deterministic providers."
